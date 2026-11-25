"""Content-addressed, independently versioned runtime packages.

The closure is complete for the current MJCF proxy exporter: all geometry is
embedded in the generated XML. Isaac assets are frozen by the separate lazy
OpenUSD builder; engine/plugin binaries remain an explicit installed dependency.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from .exporters.mujoco_mjcf import MujocoMjcfExporter
from .paths import default_registry_path
from .physics_dependencies import build_dependency_graph
from .registry import AssetRegistry
from .session_contracts import finite_json


LOCK_SCHEMA = "scene_factory.runtime_lock.v1"
PACKAGE_SCHEMA = "scene_factory.environment_package.v1"
_RUNTIME_SOURCES = (
    "exporters/mujoco_mjcf.py", "asset_geometry.py", "models.py", "registry.py", "tasks.py", "backends/mujoco.py",
    "agent.py", "episode_reset.py", "episode_randomization.py", "constrained_randomization.py", "geometry.py",
    "session.py", "session_contracts.py", "sensor_artifacts.py", "joint_control.py", "pose_migration.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(finite_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _descriptor(path: str, data: bytes, role: str) -> dict[str, Any]:
    return {"path": path, "sha256": _sha(data), "size": len(data), "role": role}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("environment dependency path must be a safe relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {".", "..", ""} for part in relative.parts):
        raise ValueError("environment dependency path escapes package")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("environment dependency resolves outside package")
    return target


def _effective_instances(xml: str) -> list[dict[str, Any]]:
    """Record authored values; compiler-derived inertia remains explicitly null."""
    root = ET.fromstring(xml)
    rows = []
    for body in root.findall("./worldbody/body"):
        inertial = body.find("inertial")
        geoms = body.findall("geom")
        rows.append({
            "body_name": body.attrib["name"], "dynamic": body.find("freejoint") is not None,
            "position_m": [float(v) for v in body.attrib["pos"].split()],
            "orientation_wxyz": [float(v) for v in body.attrib["quat"].split()],
            "authored_inertial": dict(inertial.attrib) if inertial is not None else None,
            "geoms": [{key: geom.attrib[key] for key in ("name", "type", "mass", "friction", "size", "mesh") if key in geom.attrib} for geom in geoms],
            "inertia_source": "explicit_inertial" if inertial is not None else "mujoco_compiler_from_locked_geometry",
        })
    return rows


def build_environment(
    scene_path: str | Path, output_dir: str | Path, *, registry_path: str | Path | None = None,
    backend: str = "mujoco", cache_dir: str | Path | None = None,
    usd_path: str | Path | None = None, robot_asset_root: str | Path | None = None,
    backend_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if backend == "isaac":
        from .isaac_environment_package import build_isaac_environment

        if usd_path is None or robot_asset_root is None:
            raise ValueError("Isaac packages require usd_path and robot_asset_root")
        return build_isaac_environment(scene_path, output_dir, usd_path=usd_path,
                                       robot_asset_root=robot_asset_root, registry_path=registry_path,
                                       backend_options=backend_options)
    if backend != "mujoco":
        raise ValueError("complete environment packages support mujoco or isaac")
    if usd_path is not None or robot_asset_root is not None or backend_options:
        raise ValueError("Isaac-only options cannot be supplied for a MuJoCo package")
    scene_path = Path(scene_path).expanduser().resolve()
    registry_path = Path(registry_path or default_registry_path()).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"environment output must not already exist: {output}")
    if scene_path.is_relative_to(output) or registry_path.is_relative_to(output):
        raise ValueError("environment output cannot contain source inputs")
    scene = finite_json(json.loads(scene_path.read_text(encoding="utf-8-sig")))
    if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list) or not scene["objects"]:
        raise ValueError("environment scene requires nonempty objects")
    if scene.get("task", {}).get("physics_overrides") is not None:
        raise ValueError("task.physics_overrides is supported only by the Isaac backend")
    registry = AssetRegistry.load(registry_path)
    ids = [item["object_id"] for item in scene["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("environment object IDs must be unique")
    selected_ids = sorted({str(item["asset_id"]) for item in scene["objects"]})
    registry_rows = {}
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            row = json.loads(line)
            if row.get("asset_id") in selected_ids:
                registry_rows[row["asset_id"]] = row
    if set(registry_rows) != set(selected_ids):
        raise ValueError("selected registry dependency missing")
    files: dict[str, tuple[bytes, str]] = {"scene/layout.json": (canonical_bytes(scene), "scene")}
    for asset_id in selected_ids:
        asset = registry.get(asset_id)
        if not asset.license:
            raise ValueError(f"asset license is not declared: {asset_id}")
        if "kitchen_dining_pack" in asset.tags:
            for name in ("SOURCE.json", "geometry.json"):
                source = registry.base_dir / "source" / asset_id / name
                if not source.is_file():
                    raise FileNotFoundError(f"missing MuJoCo geometry dependency: {source}")
                files[f"assets/source/{asset_id}/{name}"] = (source.read_bytes(), "collision_geometry")
    files["assets/registry.jsonl"] = (b"\n".join(canonical_bytes(registry_rows[key]) for key in selected_ids) + b"\n", "registry")
    source_modules = {
        name: _sha((Path(__file__).parent / name).read_bytes())
        for name in _RUNTIME_SOURCES
    }
    # The cache key covers source data and generator implementation, never evidence.
    cache_key = _sha(canonical_bytes({
        "schema": LOCK_SCHEMA, "backend": backend, "sources": source_modules,
        "inputs": {name: _sha(data) for name, (data, _) in sorted(files.items())},
    }))
    cached = False
    cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else None
    xml: bytes | None = None
    if cache_path:
        entry = cache_path / f"{cache_key}.json"
        if entry.is_file():
            cached_record = json.loads(entry.read_text(encoding="utf-8"))
            candidate = cached_record.get("xml", "").encode("utf-8")
            if cached_record.get("key") != cache_key or _sha(candidate) != cached_record.get("xml_sha256"):
                raise ValueError("environment compilation cache entry is corrupt")
            xml, cached = candidate, True
    if xml is None:
        xml = MujocoMjcfExporter(registry).to_string(scene).encode("utf-8")
        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
            entry = cache_path / f"{cache_key}.json"
            temporary = entry.with_name(entry.name + f".{os.getpid()}.tmp")
            temporary.write_bytes(canonical_bytes({"key": cache_key, "xml_sha256": _sha(xml), "xml": xml.decode("utf-8")}))
            temporary.replace(entry)
    xml_root = ET.fromstring(xml)
    if any("file" in element.attrib for element in xml_root.iter()):
        raise ValueError("MJCF exporter introduced external file dependencies; closure must be extended")
    files["scene/scene.xml"] = (xml, "compiled_scene")
    graph = build_dependency_graph(scene, backend=backend).to_dict()
    files["runtime/dependencies.json"] = (canonical_bytes(graph), "dependency_graph")
    engine_version = _version("mujoco")
    if engine_version is None:
        raise RuntimeError("MuJoCo must be installed to freeze its exact runtime version")
    lock = {
        "schema_version": LOCK_SCHEMA, "backend": backend,
        "dependency_closure": "complete_for_current_mujoco_proxy_exporter",
        "dependencies": [_descriptor(name, data, role) for name, (data, role) in sorted(files.items())],
        "scene": {"scene_id": scene["scene_id"], "layout_path": "scene/layout.json", "registry_path": "assets/registry.jsonl", "compiled_path": "scene/scene.xml"},
        "instances": _effective_instances(xml.decode("utf-8")),
        "task": scene.get("task", {}),
        "randomization": scene.get("randomization", {"seed": scene.get("seed"), "sampled_values": {}}),
        "runtime": {"engine": "mujoco", "engine_version": engine_version, "python_version": platform.python_version(), "source_sha256": source_modules, "image_digest": None},
        "solver": {"authored_option": dict(xml_root.find("option").attrib), "resolved_defaults": f"locked by mujoco {engine_version}"},
        "controller": {"mode": "noop_or_object_forces", "action_repeat": 5},
        "validation": {"static_structure": "compiled", "dynamics": "not_verified", "task": "not_verified"},
    }
    environment_sha = _sha(canonical_bytes(lock))
    lock["environment_sha256"] = environment_sha
    files["runtime/lock.json"] = (canonical_bytes(lock), "runtime_lock")
    blob_hits, blob_writes = 0, 0
    if cache_path:
        blobs = cache_path / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        for _, (data, _) in files.items():
            blob = blobs / _sha(data)
            if blob.is_file():
                if blob.read_bytes() != data:
                    raise ValueError("environment blob cache is corrupt")
                blob_hits += 1
            else:
                temporary = blob.with_name(blob.name + f".{os.getpid()}.tmp")
                temporary.write_bytes(data)
                temporary.replace(blob)
                blob_writes += 1
    manifest = {"schema_version": PACKAGE_SCHEMA, "environment_sha256": environment_sha, "lock_path": "runtime/lock.json", "cache": {"key": cache_key, "hit": cached, "reused_blobs": blob_hits, "written_blobs": blob_writes}, "evidence_status": "not_verified"}
    files["manifest.json"] = (canonical_bytes(manifest), "manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, (data, _) in files.items():
            target = _safe_path(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        verification = verify_environment(staging)
        if not verification["valid"]:
            raise ValueError(f"generated environment failed verification: {verification['errors']}")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return {**manifest, "package_dir": str(output), "verification": verification}


def verify_environment(
    package_dir: str | Path, *, check_runtime: bool = True, resolve_assets: bool = True,
) -> dict[str, Any]:
    root = Path(package_dir).expanduser().resolve()
    errors: list[str] = []
    lock: dict[str, Any] = {}
    try:
        manifest = finite_json(json.loads((root / "manifest.json").read_text(encoding="utf-8")))
        if manifest.get("backend") == "isaac":
            from .isaac_environment_package import verify_isaac_environment

            return verify_isaac_environment(root, check_runtime=check_runtime, resolve_assets=resolve_assets)
        if manifest.get("schema_version") != PACKAGE_SCHEMA:
            raise ValueError("unsupported environment package schema")
        lock = finite_json(json.loads(_safe_path(root, manifest["lock_path"]).read_text(encoding="utf-8")))
        if lock.get("schema_version") != LOCK_SCHEMA or lock.get("backend") != "mujoco":
            raise ValueError("unsupported runtime lock schema/backend")
        if lock.get("dependency_closure") != "complete_for_current_mujoco_proxy_exporter":
            raise ValueError("package does not declare the supported dependency closure")
        if set(lock["runtime"]["source_sha256"]) != set(_RUNTIME_SOURCES):
            raise ValueError("runtime source lock is incomplete")
        expected = lock["environment_sha256"]
        content = {key: value for key, value in lock.items() if key != "environment_sha256"}
        if _sha(canonical_bytes(content)) != expected or manifest.get("environment_sha256") != expected:
            errors.append("environment_digest_mismatch")
        seen = set()
        for item in lock["dependencies"]:
            name = item["path"]
            if name in seen:
                raise ValueError("duplicate dependency path")
            seen.add(name)
            path = _safe_path(root, name)
            data = path.read_bytes()
            if _sha(data) != item["sha256"] or len(data) != item["size"]:
                errors.append(f"dependency_mismatch:{name}")
        if check_runtime:
            if _version("mujoco") != lock["runtime"]["engine_version"]:
                errors.append("engine_version_mismatch")
            if platform.python_version() != lock["runtime"]["python_version"]:
                errors.append("python_version_mismatch")
            for name, digest in lock["runtime"]["source_sha256"].items():
                source = _safe_path(Path(__file__).parent, name)
                if _sha(source.read_bytes()) != digest:
                    errors.append(f"runtime_source_mismatch:{name}")
        for key in ("layout_path", "registry_path", "compiled_path"):
            if lock["scene"][key] not in seen:
                errors.append(f"required_dependency_missing:{key}")
        if not errors and check_runtime:
            scene = json.loads(_safe_path(root, lock["scene"]["layout_path"]).read_text(encoding="utf-8"))
            registry = AssetRegistry.load(_safe_path(root, lock["scene"]["registry_path"]))
            actual_xml = _safe_path(root, lock["scene"]["compiled_path"]).read_bytes()
            if MujocoMjcfExporter(registry).to_string(scene).encode("utf-8") != actual_xml:
                errors.append("compiled_scene_does_not_match_frozen_inputs")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_package:{type(exc).__name__}:{exc}")
    return {"schema_version": "scene_factory.environment_verification.v1", "valid": not errors, "errors": errors, "environment_sha256": lock.get("environment_sha256"), "runtime_checked": check_runtime, "physics_acceptance": "not_evaluated"}


def archive_environment(package_dir: str | Path, output_zip: str | Path) -> Path:
    root = Path(package_dir).resolve()
    result = verify_environment(root)
    if not result["valid"]:
        raise ValueError(f"cannot archive invalid environment: {result['errors']}")
    output = Path(output_zip).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("archive output must be new and outside environment")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                if not path.resolve().is_relative_to(root):
                    raise ValueError("package symlink escapes root")
                archive.write(path, path.relative_to(root).as_posix())
    return output

