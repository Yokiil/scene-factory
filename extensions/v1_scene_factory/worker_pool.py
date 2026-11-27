"""Conservative compatibility signatures for optional MuJoCo worker reuse.

No physics runtime is initialized here. The complete generated MJCF, source
scene, registry, engine binaries, Python sources, settings and admission
resources must match. Paths inside equivalent frozen packages may differ.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import threading
from pathlib import Path
from typing import Any

_binary_cache: dict[tuple[str, int, int], str] = {}
_cache_lock = threading.Lock()
_RUNTIME_SOURCES = (
    "worker_pool.py", "sandbox_worker.py", "runtime_service.py", "agent.py", "session.py", "session_contracts.py",
    "episode_reset.py", "episode_randomization.py", "constrained_randomization.py", "backends/mujoco.py",
    "models.py", "registry.py", "tasks.py", "exporters/mujoco_mjcf.py", "asset_geometry.py", "geometry.py",
    "paths.py", "pose_migration.py", "sensor_artifacts.py",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _engine_lock() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("mujoco")
    binaries = {}
    for entry in distribution.files or []:
        name = entry.as_posix()
        if not name.startswith("mujoco/") or not (name.endswith((".dll", ".pyd", ".dylib")) or ".so" in name):
            continue
        path = Path(distribution.locate_file(entry)).resolve()
        info = path.stat()
        key = (str(path), info.st_size, info.st_mtime_ns)
        with _cache_lock:
            checksum = _binary_cache.get(key)
        if checksum is None:
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            with _cache_lock:
                _binary_cache[key] = checksum
        binaries[name] = checksum
    if not binaries:
        raise ValueError("MuJoCo worker pooling requires identifiable installed engine binaries")
    return {"version": distribution.version, "binaries_sha256": binaries,
            "distribution_record_sha256": hashlib.sha256((distribution.read_text("RECORD") or "").encode()).hexdigest()}


def mujoco_worker_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Compute a strict fingerprint; there is no cross-backend/topology reuse."""
    if config["backend"] != "mujoco":
        raise ValueError("only MuJoCo workers support pooling")
    from .exporters.mujoco_mjcf import MujocoMjcfExporter
    from .registry import AssetRegistry
    from .session_contracts import finite_json

    scene = finite_json(json.loads(Path(config["scene_path"]).read_text(encoding="utf-8")))
    registry_path = Path(config["registry_path"])
    registry = AssetRegistry.load(registry_path)
    registry.base_dir = Path(config["registry_base_dir"])
    options = config["backend_options"]
    xml = MujocoMjcfExporter(registry, offscreen_width=options.get("render_width", 960),
                            offscreen_height=options.get("render_height", 720)).to_string(scene)
    source_root = Path(__file__).parent
    sources = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in _RUNTIME_SOURCES}
    package_manifest = None
    if config.get("environment_path"):
        package_manifest = hashlib.sha256((Path(config["environment_path"]) / "manifest.json").read_bytes()).hexdigest()
    record = {"schema_version": "scene_factory.mujoco_worker_compatibility.v1", "backend": "mujoco",
              "scene_sha256": _digest(scene), "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
              "package_manifest_sha256": package_manifest, "backend_options": options, "resources": config["resources"],
              "engine": _engine_lock(), "python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
              "platform": platform.platform(), "source_sha256": sources}
    return {"sha256": _digest(record), "record": record}
