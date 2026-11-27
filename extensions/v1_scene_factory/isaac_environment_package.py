"""Freeze local USD/robot assets separately from the installed Isaac runtime.

OpenUSD is imported lazily, so the normal compiler and HTTP client remain usable
without Isaac. Asset paths are rewritten to relative package paths. Engine MDL
identifiers are the only allowed external USD assets and their installed files
and owning runtime versions are locked explicitly.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any

from .environment_package import PACKAGE_SCHEMA, LOCK_SCHEMA, _descriptor, _safe_path, canonical_bytes
from .session_contracts import finite_json


ISAAC_CLOSURE = "complete_local_usd_and_robot_assets_with_locked_runtime_builtins"
FRANKA_RELATIVE = "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_lock() -> dict[str, Any]:
    try:
        engine = importlib.metadata.distribution("isaacsim")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Isaac packages must be built with the installed Isaac Python") from exc
    root = Path(engine.locate_file("isaacsim")).resolve()
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if name.lower().startswith(("isaacsim", "isaac-sim", "omni-", "omniverse")):
            versions[name.lower()] = distribution.version
    source_root = Path(__file__).parent
    return {
        "engine": "isaac", "engine_version": engine.version,
        "python_version": platform.python_version(),
        "distributions": dict(sorted(versions.items())),
        "source_sha256": {path.relative_to(source_root).as_posix(): _sha(path)
                           for path in sorted(source_root.rglob("*.py")) if "__pycache__" not in path.parts},
        "extension_config_sha256": {
            path.relative_to(root).as_posix(): _sha(path)
            for folder in (root / "exts", root / "extsDeprecated", root / "kit/exts", root / "kit/extscore")
            if folder.is_dir() for path in sorted(folder.glob("*/config/extension.toml"))
        },
        "image_digest": os.environ.get("SCENE_FACTORY_RUNTIME_IMAGE_DIGEST"),
        "runtime_builtins": {"OmniPBR.mdl": {
            "relative_path": "kit/mdl/core/Base/OmniPBR.mdl",
            "sha256": _sha(root / "kit/mdl/core/Base/OmniPBR.mdl"),
        }},
        # OmniPBR imports ClearCoat and NVIDIA modules, which import further
        # modules/resources. Lock the entire small installed MDL tree rather
        # than claiming that the one top-level shader hash closes that graph.
        "mdl_runtime_files_sha256": {
            path.relative_to(root).as_posix(): _sha(path)
            for path in sorted((root / "kit/mdl").rglob("*")) if path.is_file()
        },
        "runtime_scope": "installed engine and extension distributions; container digest supplied by deployment",
    }


def _usd():
    try:
        from pxr import Ar, Sdf, UsdUtils
    except ImportError as exc:
        raise RuntimeError("OpenUSD is required; run using the Isaac Python") from exc
    return Ar, Sdf, UsdUtils


def _collect(roots: list[Path], builtins: set[str]):
    _Ar, Sdf, UsdUtils = _usd()
    layers, assets = {}, set()
    for root in roots:
        root_layers, root_assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(root.as_posix()))
        missing = set(unresolved) - builtins
        if missing:
            raise FileNotFoundError(f"unresolved USD dependencies: {sorted(missing)}")
        for layer in root_layers:
            path = Path(layer.realPath).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"USD layer is not a local file: {layer.identifier}")
            layers[path] = layer
        for asset in root_assets:
            path = Path(str(asset)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"USD asset is not a local file: {asset}")
            assets.add(path)
    return layers, assets


def build_isaac_environment(
    scene_path: str | Path, output_dir: str | Path, *, usd_path: str | Path,
    robot_asset_root: str | Path, registry_path: str | Path | None = None,
    backend_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a portable asset package; this does not perform dynamics acceptance."""
    from .paths import default_registry_path
    from .runtime_service import _number, _integer

    Ar, Sdf, UsdUtils = _usd()
    source = Path(scene_path).expanduser().resolve()
    usd = Path(usd_path).expanduser().resolve()
    asset_root = Path(robot_asset_root).expanduser().resolve()
    robot = asset_root / FRANKA_RELATIVE
    output = Path(output_dir).expanduser().resolve()
    registry = Path(registry_path or default_registry_path()).expanduser().resolve()
    for path in (source, usd, robot, registry):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_relative_to(output):
            raise ValueError("output cannot contain package inputs")
    if output.exists():
        raise FileExistsError(f"environment output already exists: {output}")
    scene = finite_json(json.loads(source.read_text(encoding="utf-8-sig")))
    if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list) or not scene["objects"]:
        raise ValueError("Isaac environment requires a nonempty layout")
    supplied = finite_json(backend_options or {})
    if not isinstance(supplied, dict) or supplied.keys() - {"max_steps", "physics_dt", "enable_rgbd", "enable_general_observation"}:
        raise ValueError("unknown locked Isaac backend options")
    enabled = supplied.get("enable_rgbd", True)
    general = supplied.get("enable_general_observation", True)
    if not isinstance(enabled, bool) or not isinstance(general, bool):
        raise ValueError("enable_rgbd must be boolean")
    options = {
        "max_steps": _integer(supplied.get("max_steps", 10000), "max_steps", 1, 10_000_000),
        "physics_dt": _number(supplied.get("physics_dt", 1 / 60), "physics_dt", 1e-6, 1),
        "enable_rgbd": enabled,
        "enable_general_observation": general,
    }
    runtime = _runtime_lock()
    layers, assets = _collect([usd, robot], set(runtime["runtime_builtins"]))
    mapping = {usd: "scene/scene.usd", robot.resolve(): "assets/robot/" + FRANKA_RELATIVE}
    for path in sorted(set(layers) | assets):
        mapping.setdefault(path, "assets/dependencies/" + _sha(path)[:16] + "_" + path.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    source_records = []
    try:
        for path, relative in mapping.items():
            destination = _safe_path(staging, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ValueError(f"asset changed while being frozen: {path.name}")
            destination.write_bytes(content)
            source_records.append({"package_path": relative, "source_sha256": hashlib.sha256(content).hexdigest(),
                                   "source_name": path.name})
        for path, layer in layers.items():
            destination = _safe_path(staging, mapping[path])
            copied = Sdf.Layer.FindOrOpen(destination.as_posix())
            if copied is None:
                raise RuntimeError(f"cannot open copied USD layer {destination}")

            def rewrite(value: str, *, source_layer=layer, target=destination) -> str:
                if not value or value in runtime["runtime_builtins"]:
                    return value
                computed = Sdf.ComputeAssetPathRelativeToLayer(source_layer, value)
                resolved = str(Ar.GetResolver().Resolve(computed))
                candidate = Path(resolved or computed).resolve()
                if candidate not in mapping:
                    raise ValueError(f"USD path cannot be frozen: {value}")
                return os.path.relpath(_safe_path(staging, mapping[candidate]), target.parent).replace("\\", "/")

            UsdUtils.ModifyAssetPaths(copied, rewrite)
            copied.Save()
        (staging / "scene/layout.json").write_bytes(canonical_bytes(scene))
        (staging / "assets/registry.jsonl").write_bytes(registry.read_bytes())
        (staging / "runtime").mkdir(exist_ok=True)
        (staging / "runtime/source_assets.json").write_bytes(canonical_bytes(source_records))
        _collect([staging / mapping[usd], staging / mapping[robot.resolve()]], set(runtime["runtime_builtins"]))
        dependencies = []
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                role = "usd_layer" if file.suffix.lower() in {".usd", ".usda", ".usdc"} else "asset_or_config"
                dependencies.append(_descriptor(file.relative_to(staging).as_posix(), file.read_bytes(), role))
        lock = {
            "schema_version": LOCK_SCHEMA, "backend": "isaac", "dependency_closure": ISAAC_CLOSURE,
            "dependencies": dependencies, "runtime": runtime,
            "scene": {"scene_id": scene["scene_id"], "layout_path": "scene/layout.json",
                      "registry_path": "assets/registry.jsonl", "compiled_path": mapping[usd],
                      "robot_asset_root": "assets/robot", "robot_usd_path": mapping[robot.resolve()]},
            "backend_options": options, "task": scene.get("task", {}),
            "validation": {"asset_closure": "verified", "dynamics": "not_verified", "task": "not_verified"},
            "distribution_policy": {
                "scope": "private_local_runtime_package",
                "redistribution_rights": "not_asserted_by_builder",
                "generic_source_delivery": "exclude_robot_assets_and_local_runtime_package",
                "asset_provenance": "operator_supplied_local_Isaac_asset_cache",
            },
        }
        digest = hashlib.sha256(canonical_bytes(lock)).hexdigest()
        lock["environment_sha256"] = digest
        (staging / "runtime/lock.json").write_bytes(canonical_bytes(lock))
        manifest = {"schema_version": PACKAGE_SCHEMA, "backend": "isaac", "environment_sha256": digest,
                    "lock_path": "runtime/lock.json", "evidence_status": "not_verified"}
        (staging / "manifest.json").write_bytes(canonical_bytes(manifest))
        verification = verify_isaac_environment(staging)
        if not verification["valid"]:
            raise ValueError(f"generated Isaac package invalid: {verification['errors']}")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return {**manifest, "package_dir": str(output), "verification": verification}


def verify_isaac_environment(
    package_dir: str | Path, *, check_runtime: bool = True, resolve_assets: bool = True,
) -> dict[str, Any]:
    """Verify frozen bytes and runtime; optionally re-run the USD resolver.

    A fresh Kit worker must use ``resolve_assets=False`` before SimulationApp:
    importing standalone pxr there can load incompatible USD DLLs. The manager
    already resolves the closure; the worker still checks every file and the
    entire installed runtime lock. This option does not disable those checks.
    """
    root, errors, lock = Path(package_dir).resolve(), [], {}
    try:
        manifest = finite_json(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
        if manifest.get("schema_version") != PACKAGE_SCHEMA:
            raise ValueError("unsupported environment schema")
        lock = finite_json(json.loads(_safe_path(root, manifest["lock_path"]).read_text(encoding="utf-8")))
        if lock.get("schema_version") != LOCK_SCHEMA or lock.get("backend") != "isaac" or lock.get("dependency_closure") != ISAAC_CLOSURE:
            raise ValueError("unsupported Isaac runtime lock")
        digest = hashlib.sha256(canonical_bytes({k: v for k, v in lock.items() if k != "environment_sha256"})).hexdigest()
        if lock["environment_sha256"] != digest or manifest.get("environment_sha256") != digest:
            errors.append("environment_digest_mismatch")
        seen = set()
        for item in lock["dependencies"]:
            if item["path"] in seen:
                raise ValueError("duplicate dependency path")
            seen.add(item["path"])
            path = _safe_path(root, item["path"])
            if path.stat().st_size != item["size"] or _sha(path) != item["sha256"]:
                errors.append("dependency_mismatch:" + item["path"])
        for key in ("layout_path", "registry_path", "compiled_path", "robot_usd_path"):
            if lock["scene"][key] not in seen:
                errors.append("required_dependency_missing:" + key)
        if check_runtime:
            actual = _runtime_lock()
            for key in ("engine_version", "python_version", "distributions", "source_sha256", "extension_config_sha256", "runtime_builtins", "mdl_runtime_files_sha256"):
                if actual[key] != lock["runtime"][key]:
                    errors.append("runtime_mismatch:" + key)
                    if key == "source_sha256":
                        previous, current = lock["runtime"][key], actual[key]
                        errors.extend("runtime_source_changed:" + name for name in sorted(previous.keys() | current.keys())
                                      if previous.get(name) != current.get(name))
            if resolve_assets:
                layers, assets = _collect([_safe_path(root, lock["scene"]["compiled_path"]),
                                          _safe_path(root, lock["scene"]["robot_usd_path"])], set(actual["runtime_builtins"]))
                for path in set(layers) | assets:
                    if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in seen:
                        errors.append("usd_dependency_escapes_lock")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_package:{type(exc).__name__}:{exc}")
    return {"schema_version": "scene_factory.environment_verification.v1", "valid": not errors,
            "errors": errors, "backend": "isaac", "environment_sha256": lock.get("environment_sha256"),
            "runtime_checked": check_runtime, "usd_dependency_resolution_checked": check_runtime and resolve_assets,
            "physics_acceptance": "not_evaluated"}
