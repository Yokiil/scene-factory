"""Read-only Linux/container preflight; it does not certify a physics task."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=15, check=False)
        return {"ok": result.returncode == 0, "stdout": result.stdout.strip(),
                "error": result.stderr.strip() if result.returncode else None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe(*, stage="host", backend="mujoco", assets=None):
    checks = {"linux": platform.system() == "Linux", "python": sys.version_info >= (3, 12)}
    result = {"schema": "scene_factory.deployment_probe.v1", "stage": stage,
              "backend": backend, "platform": platform.platform(), "python": sys.version,
              "checks": checks, "physics_acceptance": "not_evaluated"}
    if assets is not None:
        root = Path(assets).expanduser().resolve()
        result["asset_root"] = str(root)
        checks["asset_registry"] = (root / "registry.jsonl").is_file()
        checks["asset_root_readable"] = root.is_dir()
    if stage == "host":
        result["docker"] = _command(["docker", "info", "--format", "{{.OSType}}"])
        checks["docker_linux_engine"] = bool(result["docker"]["ok"] and result["docker"].get("stdout") == "linux")
        if backend == "isaac":
            result["nvidia"] = _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
            checks["nvidia_driver"] = result["nvidia"]["ok"]
            result["nvidia_container_toolkit_present"] = bool(shutil.which("nvidia-container-cli"))
            checks["nvidia_container_toolkit"] = result["nvidia_container_toolkit_present"]
    elif backend == "mujoco":
        result["mujoco_version"] = _version("mujoco")
        checks["mujoco_pinned_version"] = result["mujoco_version"] == "3.3.7"
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom type='sphere' size='.1'/></body></worldbody></mujoco>")
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            checks["mujoco_step"] = bool(data.time > 0)
        except Exception as exc:
            checks["mujoco_step"] = False
            result["engine_error"] = str(exc)
    else:
        result["isaac_version"] = _version("isaacsim")
        checks["isaac_installed"] = result["isaac_version"] is not None
        result["nvidia"] = _command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        checks["nvidia_visible"] = result["nvidia"]["ok"]
        try:
            import torch

            result["cuda_runtime"] = torch.version.cuda
            checks["cuda_available"] = bool(torch.cuda.is_available())
        except ImportError:
            checks["cuda_available"] = False
        result["isaac_simulation_app"] = "not_started_by_read_only_preflight"
    result["status"] = "passed" if all(checks.values()) else "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("host", "container"), default="host")
    parser.add_argument("--backend", choices=("mujoco", "isaac"), default="mujoco")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(stage=args.stage, backend=args.backend, assets=args.assets)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
