"""Measured local runtime baselines, without implying physics acceptance.

Creation below starts a fresh backend in the current Python process. It is not
process-cold startup and it is not a resident-worker warm-start measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import DryRunBackend, SceneFactoryEnv
from .dataset import write_json_atomic
from .paths import default_registry_path
from .session_contracts import finite_json


BENCHMARK_SCHEMA = "scene_factory.runtime_benchmark.v1"
BACKENDS = ("dry-run", "mujoco", "isaac")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mujoco_dependencies(scene: dict[str, Any], registry_path: Path) -> dict[str, Any]:
    """Match the external file reads performed by load_asset_parts()."""
    from .registry import AssetRegistry
    files = []
    try:
        registry = AssetRegistry.load(registry_path)
        for asset_id in sorted({str(item["asset_id"]) for item in scene["objects"]}):
            asset = registry.get(asset_id)
            if "kitchen_dining_pack" in asset.tags:
                for name in ("SOURCE.json", "geometry.json"):
                    path = registry.base_dir / "source" / asset_id / name
                    files.append({"asset_id": asset_id, "path": str(path), "sha256": _sha256(path)})
        return {"status": "complete_for_current_mjcf_exporter", "files": files}
    except (KeyError, OSError, ValueError) as exc:
        return {"status": "failed", "files": files, "error": str(exc)}


def _gpu_inventory() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode:
            return {"status": "unavailable", "devices": [], "reason": "nvidia-smi failed"}
        devices = []
        for line in result.stdout.splitlines():
            name, driver, memory = line.rsplit(",", 2)
            devices.append({"name": name.strip(), "driver": driver.strip(), "memory_total_mib": int(memory)})
        return {"status": "reported", "devices": devices}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"status": "unavailable", "devices": [], "reason": "nvidia-smi unavailable"}


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Linear-interpolated percentiles, with failed attempts kept in raw data."""
    values = sorted(sample["elapsed_s"] for sample in samples if sample["status"] == "ok")
    def percentile(fraction: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {
        "attempt_count": len(samples), "sample_count": len(values),
        "failure_count": len(samples) - len(values), "p50_s": percentile(0.5),
        "p95_s": percentile(0.95), "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
        "mean_s": statistics.fmean(values) if values else None,
        "total_success_s": sum(values), "samples": samples,
    }


def _runtime_metadata(backend: Any, name: str, render: bool) -> dict[str, Any]:
    model = getattr(backend, "_model", None)
    option = getattr(model, "opt", None)
    frame_skip = getattr(backend, "frame_skip", 1 if name == "isaac" else None)
    dt = getattr(option, "timestep", getattr(backend, "physics_dt", None))
    return {
        "engine_version": _version("mujoco" if name == "mujoco" else "isaacsim") if name != "dry-run" else None,
        "fidelity": {"dry-run": "no_physics", "mujoco": "mjcf_collision_proxies", "isaac": "isaac_usd_scripted_mug"}[name],
        "physics_dt_s": float(dt) if dt is not None else None,
        "action_repeat": frame_skip,
        "control_dt_s": float(dt) * frame_skip if dt is not None and frame_skip is not None else None,
        "solver": {
            key: int(getattr(option, key)) if option is not None and hasattr(option, key) else None
            for key in ("solver", "iterations", "integrator")
        },
        "joint_count": int(model.njnt) if model is not None else None,
        "geom_count": int(model.ngeom) if model is not None else None,
        "mesh_vertex_count": int(model.nmeshvert) if model is not None else None,
        "render_enabled": render,
        "render_resolution": [backend.render_width, backend.render_height] if hasattr(backend, "render_width") else None,
        "camera_config": getattr(backend, "camera_config", None),
        "observation_kind": "state_only",
        "step_includes_sensor_render": bool(name == "isaac" and render),
        "concurrency": 1,
    }


def create_benchmark_backend(
    name: str, *, max_steps: int, usd_path: Path | None,
    registry_path: Path, render: bool,
) -> Any:
    if name == "dry-run":
        return DryRunBackend(max_steps=max_steps)
    if name == "mujoco":
        from .backends.mujoco import MujocoBackend
        from .registry import AssetRegistry
        return MujocoBackend(registry=AssetRegistry.load(registry_path), max_steps=max_steps)
    from .backends.isaac import IsaacSimBackend
    return IsaacSimBackend(usd_path, headless=True, max_steps=max_steps, enable_rgbd=render)


def run_benchmark(
    scene_path: str | Path, *, backend: str = "dry-run", output: str | Path | None = None,
    reset_trials: int = 100, step_trials: int = 100, creation_trials: int = 3,
    seed: int = 0, render: bool = False, usd_path: str | Path | None = None,
    registry_path: str | Path | None = None, options: dict[str, Any] | None = None,
    backend_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    for key, value in {"reset_trials": reset_trials, "step_trials": step_trials, "creation_trials": creation_trials}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if backend not in BACKENDS:
        raise ValueError(f"unknown benchmark backend: {backend}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if backend == "isaac" and usd_path is None and backend_factory is None:
        raise ValueError("Isaac benchmark requires --usd")
    scene_path = Path(scene_path).resolve()
    registry_path = Path(registry_path or default_registry_path()).resolve()
    usd_path = Path(usd_path).resolve() if usd_path else None
    output = Path(output).resolve() if output else None
    if output in {scene_path, registry_path, usd_path} and output is not None:
        raise ValueError("benchmark output must not overwrite an input")
    scene_bytes = scene_path.read_bytes()
    scene = json.loads(scene_bytes)
    if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
        raise ValueError("benchmark input must be a scene layout with an objects array")
    # Reject non-finite inputs before starting any simulator.
    json.dumps({"scene": scene, "options": options}, allow_nan=False)
    canonical_scene_sha256 = hashlib.sha256(json.dumps(
        scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    geometry = _mujoco_dependencies(scene, registry_path) if backend == "mujoco" else {
        "status": "not_applicable", "files": [],
    }
    objects = scene["objects"]
    report: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "physics_acceptance": "not_evaluated",
        "backend": backend, "synthetic_backend": backend_factory is not None,
        "configuration": {"creation_trials": creation_trials, "reset_trials": reset_trials,
                          "step_trials": step_trials, "seed": seed, "options": options or {}, "render": render,
                          "action": "scripted" if backend == "isaac" else "noop"},
        "inputs": {"scene_path": str(scene_path), "scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
                   "registry_path": str(registry_path), "registry_sha256": _sha256(registry_path),
                   "usd_path": str(usd_path) if usd_path else None,
                   "usd_sha256": _sha256(usd_path) if usd_path else None,
                   "usd_dependency_hashes": None,
                   "mujoco_geometry_dependencies": geometry},
        "host": {"platform": platform.platform(), "cpu": platform.processor(),
                 "logical_cpu_count": os.cpu_count(), "python": sys.version,
                 "scene_factory_version": _version("scene-factory"), "gpus": _gpu_inventory()},
        "scene": {"scene_id": scene.get("scene_id"), "object_count": len(objects),
                  "dynamic_object_count": sum(item.get("dynamic") is True for item in objects),
                  "static_object_count": sum(item.get("dynamic") is False for item in objects),
                  "unspecified_dynamic_object_count": sum(item.get("dynamic") not in (True, False) for item in objects)},
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)).replace("\\", "/"): _sha256(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "protocol": {
            "clock": "time.perf_counter_ns", "percentile": "linear interpolation of successful samples",
            "creation_scope": "fresh backend in current Python process to first state observation; includes reset",
            "reset_scope": "same environment object and scene topology; includes first state observation",
            "reset_seed_policy": "same explicit seed and options on every reset",
            "step_scope": "synchronous environment step; excludes optional separate render measurement",
            "state_validation": "reset and step timings include finite-JSON observation/info validation",
            "terminal_policy": "reset before next step after terminated or truncated",
            "close_scope": "environment close after each creation trial and final run",
            "output_write_scope": "initial serialization and atomic write; final timing-metadata rewrite excluded",
        },
        "unmeasured": {
            "process_cold_start": "not instrumented; fresh backend is not a fresh process",
            "warm_worker_start": "resident worker reuse is not implemented",
            "dependency_download": "not instrumented", "asset_processing": "included in backend reset where applicable",
            "compile": "included in backend reset where applicable", "engine_start": "included in backend reset where applicable",
            "scene_load": "included in backend reset where applicable", "network_latency": "local call only",
            "peak_ram_bytes": None, "peak_gpu_memory_bytes": None, "reset_state_drift": None,
            "restore_trajectory_error": None, "failure_recovery_rate": None,
        },
        "warnings": ["No physical validity or task acceptance is inferred from successful benchmark calls."],
        "runtime": None, "last_reset_info": None, "peak_observed_contact_count": None,
        "stages": {}, "failures": [], "outcomes": {"terminated": 0, "truncated": 0},
    }
    if creation_trials < 30:
        report["warnings"].append("Fewer than 30 creation trials: this is a smoke sample, not the proposed baseline protocol.")
    if reset_trials < 100:
        report["warnings"].append("Fewer than 100 reset trials; sample count is below the proposed reset baseline protocol.")
    report["warnings"].append("The 1,000-reset state-pollution and memory-leak test is separate and unmeasured.")
    if geometry["status"] == "failed":
        report["failures"].append({"stage": "dependency_discovery", "error": geometry["error"]})
    samples: dict[str, list[dict[str, Any]]] = {}

    def measured(stage: str, call: Callable[[], Any]) -> Any:
        rows = samples.setdefault(stage, [])
        row: dict[str, Any] = {"index": len(rows), "status": "ok"}
        started = time.perf_counter_ns()
        try:
            return call()
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            report["failures"].append({"stage": stage, **row})
            raise
        finally:
            row["elapsed_s"] = (time.perf_counter_ns() - started) / 1_000_000_000
            rows.append(row)

    def reset(env: SceneFactoryEnv) -> Any:
        observation, info = env.reset(seed=seed, options=options)
        if not isinstance(observation, dict) or not observation:
            raise ValueError("reset did not return a nonempty state observation")
        observation = finite_json(observation, "reset observation")
        info = finite_json(info, "reset info")
        reset_record = info.get("episode_reset", {})
        if reset_record.get("source_scene_sha256") != canonical_scene_sha256:
            raise ValueError("reset input changed since benchmark scene hashing")
        report["last_reset_info"] = info
        return observation

    def close(env: SceneFactoryEnv) -> None:
        try:
            measured("close", env.close)
        except Exception:
            pass  # Failure retained; do not mask the originating simulator error.

    factory = backend_factory or (lambda: create_benchmark_backend(
        backend, max_steps=max(step_trials + 1, 1000), usd_path=usd_path,
        registry_path=registry_path, render=render,
    ))
    env = None
    try:
        for trial in range(creation_trials):
            candidate = None
            def create() -> SceneFactoryEnv:
                nonlocal candidate
                instance = measured("backend_construct", factory)
                candidate = SceneFactoryEnv(scene_path, backend=instance)
                measured("first_reset", lambda: reset(candidate))
                return candidate
            try:
                current = measured("in_process_creation", create)
            except Exception:
                if candidate is not None:
                    close(candidate)
                continue
            if trial == creation_trials - 1:
                env = current
            else:
                close(current)
        if env is not None:
            report["runtime"] = measured("runtime_metadata", lambda: _runtime_metadata(env.backend, backend, render))
            ready = True
            for _ in range(reset_trials):
                try:
                    measured("reset", lambda: reset(env))
                except Exception:
                    ready = False
                    break
            done = False
            if ready:
                for _ in range(step_trials):
                    try:
                        if done:
                            measured("terminal_reset", lambda: reset(env))
                        action = "scripted" if backend == "isaac" else None
                        def step() -> Any:
                            result = env.step(action)
                            observation, reward, terminated, truncated, info = result
                            if not isinstance(observation, dict) or not observation or not math.isfinite(float(reward)):
                                raise ValueError("step returned an invalid observation or reward")
                            return (finite_json(observation, "step observation"),
                                    finite_json(reward, "step reward"), terminated, truncated,
                                    finite_json(info, "step info"))
                        observation, reward, terminated, truncated, info = measured("step", step)
                        report["outcomes"]["terminated"] += int(bool(terminated))
                        report["outcomes"]["truncated"] += int(bool(truncated))
                        done = bool(terminated or truncated)
                        data = getattr(env.backend, "_data", None)
                        if data is not None and hasattr(data, "ncon"):
                            report["peak_observed_contact_count"] = max(report["peak_observed_contact_count"] or 0, int(data.ncon))
                        if render:
                            def capture() -> Any:
                                image = env.backend.capture_rgbd_observation() if backend == "isaac" else env.render()
                                if image is None:
                                    raise RuntimeError("backend returned no render output; rendering is unsupported")
                                return image
                            measured("render", capture)
                    except Exception as exc:
                        if not report["failures"] or report["failures"][-1].get("error") != str(exc):
                            report["failures"].append({"stage": "step_validation", "error_type": type(exc).__name__, "error": str(exc)})
                        break
    except Exception as exc:
        if not report["failures"] or report["failures"][-1].get("error") != str(exc):
            report["failures"].append({"stage": "run", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if env is not None:
            close(env)
    for label, path in (("scene", scene_path), ("registry", registry_path), ("usd", usd_path)):
        if path is not None:
            try:
                if _sha256(path) != report["inputs"][f"{label}_sha256"]:
                    report["failures"].append({"stage": "input_verification", "error": f"{label} changed during benchmark"})
            except OSError as exc:
                report["failures"].append({"stage": "input_verification", "error": str(exc)})
    for dependency in geometry["files"]:
        try:
            if _sha256(Path(dependency["path"])) != dependency["sha256"]:
                report["failures"].append({"stage": "input_verification", "error": f"geometry dependency changed: {dependency['path']}"})
        except OSError as exc:
            report["failures"].append({"stage": "input_verification", "error": str(exc)})
    for stage in ("backend_construct", "first_reset", "in_process_creation", "reset", "step", "terminal_reset", "render", "close", "output_write", "runtime_metadata"):
        report["stages"][stage] = summarize_samples(samples.get(stage, []))
    steps = report["stages"]["step"]
    elapsed = steps["total_success_s"]
    rate = steps["sample_count"] / elapsed if elapsed > 0 else None
    repeat = (report["runtime"] or {}).get("action_repeat")
    report["throughput"] = {"environment_steps_per_s": rate,
                            "physics_steps_per_s": rate * repeat if rate is not None and repeat is not None else None,
                            "scope": "successful step call time only; excludes resets and separate rendering"}
    report["status"] = "failed" if report["failures"] else "completed"
    if output is not None:
        completed_status = report["status"]
        # An interruption before the final atomic replacement must not leave an
        # apparently completed report that lacks its write timing.
        report["status"] = "writing_report"
        try:
            measured("output_write", lambda: write_json_atomic(output, report))
        except Exception:
            report["status"] = "failed"
            report["stages"]["output_write"] = summarize_samples(samples["output_write"])
            return report
        report["stages"]["output_write"] = summarize_samples(samples["output_write"])
        report["status"] = completed_status
        try:
            write_json_atomic(output, report)
        except Exception as exc:
            report["status"] = "failed"
            report["failures"].append({"stage": "output_write_finalize", "error_type": type(exc).__name__, "error": str(exc)})
    return report


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scene", type=Path, help="Existing layout.json")
    parser.add_argument("--backend", choices=BACKENDS, default="dry-run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--creation-trials", type=int, default=3)
    parser.add_argument("--reset-trials", type=int, default=100)
    parser.add_argument("--step-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--usd", type=Path)


def run_benchmark_from_args(args: argparse.Namespace) -> int:
    report = run_benchmark(
        args.scene, backend=args.backend, output=args.output, creation_trials=args.creation_trials,
        reset_trials=args.reset_trials, step_trials=args.step_trials, seed=args.seed,
        render=args.render, usd_path=args.usd, registry_path=getattr(args, "registry", None),
    )
    print(json.dumps({"status": report["status"], "report": str(args.output.resolve()),
                      "backend": report["backend"], "physics_acceptance": report["physics_acceptance"],
                      "failures": report["failures"], "warnings": report["warnings"]}, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    add_benchmark_arguments(parser)
    args = parser.parse_args(argv)
    try:
        return run_benchmark_from_args(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"runtime benchmark: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
