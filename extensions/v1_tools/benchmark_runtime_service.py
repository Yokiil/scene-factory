"""Measure real loopback HTTP MuJoCo worker creation, reset and one-step latency.

The service process and OS asset caches stay resident. Each creation uses a new
worker; this does not measure uncached host startup or an idle-worker warm pool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_factory.dataset import write_json_atomic
from scene_factory.factory import SceneFactory
from scene_factory.paths import default_registry_path
from scene_factory.runtime_client import RuntimeClient
from scene_factory.runtime_service import RuntimeService, serve


def summarize(samples):
    values = sorted(row["elapsed_s"] for row in samples if row["status"] == "ok")
    def percentile(fraction):
        if not values:
            return None
        offset = (len(values) - 1) * fraction
        lower, upper = math.floor(offset), math.ceil(offset)
        return values[lower] + (values[upper] - values[lower]) * (offset - lower)
    return {
        "attempt_count": len(samples), "sample_count": len(values),
        "failure_count": len(samples) - len(values), "p50_s": percentile(.5), "p95_s": percentile(.95),
        "mean_s": statistics.fmean(values) if values else None,
        "min_s": min(values) if values else None, "max_s": max(values) if values else None,
        "samples": samples,
    }


def source_hashes():
    paths = [*sorted((ROOT / "scene_factory").rglob("*.py")), Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def host_metadata():
    cpu = platform.processor()
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    gpu = {"status": "unavailable", "devices": [], "used_for_simulation": False}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if not result.returncode:
            for line in result.stdout.splitlines():
                name, driver, memory = line.rsplit(",", 2)
                gpu["devices"].append({"name": name.strip(), "driver_version": driver.strip(),
                                       "memory_total_mib": int(memory.strip())})
            gpu["status"] = "reported"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpu_count": os.cpu_count(),
            "python": sys.version, "gpus": gpu, "engine": "mujoco",
            "engine_version": importlib.metadata.version("mujoco")}


def model_metadata(scene, registry):
    """Describe a matching export outside the timed HTTP requests."""
    import mujoco
    from scene_factory.exporters.mujoco_mjcf import MujocoMjcfExporter
    xml = MujocoMjcfExporter(registry).to_string(scene)
    model = mujoco.MjModel.from_xml_string(xml)
    return {
        "scope": "same_scene_export_preflight_outside_timed_region",
        "mjcf_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "physics_dt_s": float(model.opt.timestep), "action_repeat": 1,
        "solver": {"name": mujoco.mjtSolver(int(model.opt.solver)).name,
                   "iterations": int(model.opt.iterations), "tolerance": float(model.opt.tolerance),
                   "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name},
        "body_count": int(model.nbody), "joint_count": int(model.njnt), "geom_count": int(model.ngeom),
        "mesh_vertex_count": int(model.nmeshvert), "fidelity": "collision_proxy",
    }


def benchmark(output: Path, *, scene_path: Path | None = None, recipe: str = "living_room_returned_home",
              seed: int = 19, creation_trials: int = 30, reset_trials: int = 100, step_trials: int = 100):
    if output.exists():
        raise FileExistsError("benchmark output exists; choose a new evidence name")
    for value in (creation_trials, reset_trials, step_trials):
        if type(value) is not int or value < 1:
            raise ValueError("trial counts must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    factory = SceneFactory()
    if scene_path:
        scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    else:
        built = factory.build_from_recipe(recipe, seed)
        if not built.valid:
            raise ValueError("benchmark recipe failed static validation")
        scene = built.scene.to_dict()
    canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    scene_digest = hashlib.sha256(canonical).hexdigest()
    sources = source_hashes()
    report = {
        "schema": "scene_factory.runtime_service_benchmark.v1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "physics_acceptance": "not_evaluated",
        "host": host_metadata(), "model": model_metadata(scene, factory.registry),
        "scene": {"scene_sha256": scene_digest, "object_count": len(scene["objects"]),
                  "dynamic_object_count": sum(item.get("dynamic") is True for item in scene["objects"]),
                  "static_object_count": sum(item.get("dynamic") is False for item in scene["objects"]),
                  "layout": scene},
        "registry_sha256": hashlib.sha256(default_registry_path().read_bytes()).hexdigest(),
        "source_sha256": sources,
        "configuration": {"creation_trials": creation_trials, "reset_trials": reset_trials,
                          "step_trials": step_trials, "seed": seed, "concurrency": 1,
                          "action_repeat": 1, "requested_environment_steps_per_http_step": 1,
                          "rendering": False, "resources": {"cpu_threads": 1, "memory_mb": 1024}},
        "protocol": {
            "clock": "time.perf_counter_ns", "percentile": "linear interpolation of successful observations",
            "creation_scope": "loopback HTTP create plus explicit reset to first valid state observation; new worker each trial",
            "creation_cache_state": "resident management process; OS file caches not flushed; assets already present and preflight export loaded",
            "reset_scope": "same live worker, same topology and seed, to initial state observation",
            "step_scope": "one HTTP request, one environment step, one physics step; response includes state and contacts",
            "includes": ["HTTP serialization", "worker IPC", "service SQLite evidence commits", "state observation"],
            "excludes": ["server startup", "asset download", "rendering", "cleanup", "report writing"],
            "warm_idle_worker_creation": {"status": "not_implemented_not_measured"},
            "uncached_host_startup": {"status": "not_measured"},
            "remote_network_latency": {"status": "not_measured_loopback_only"},
            "background_load": "other SceneFactory test runs coordinated to stop; OS background processes not isolated",
        },
        "failures": [],
    }
    samples = {"new_worker_to_first_observation": [], "same_worker_reset": [], "single_step": []}
    sid = None
    initial = None
    with tempfile.TemporaryDirectory(prefix="scenefactory-http-benchmark-") as temporary:
        temp_root = Path(temporary)
        input_path = temp_root / "layout.json"
        input_path.write_bytes(canonical)
        token = secrets.token_urlsafe(32)
        manager = RuntimeService(temp_root / "runtime", allowed_scene_root=temp_root, max_sessions=1,
                                 cpu_budget=1, memory_budget_mb=2048, token=token)
        server = serve(manager, "127.0.0.1", 0)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = RuntimeClient(f"http://127.0.0.1:{server.server_port}", token=token)

        def validate_observation(value, expected_step):
            if value.get("scene_sha256") != scene_digest or value.get("step_index") != expected_step:
                raise ValueError("unexpected scene digest or step index in observation")
            if not isinstance(value.get("observation", {}).get("objects"), dict):
                raise ValueError("response lacks MuJoCo object state")

        try:
            for index in range(creation_trials):
                request = {"scene_path": str(input_path), "backend": "mujoco", "idempotency_key": uuid.uuid4().hex,
                           "backend_options": {"max_steps": max(1000, step_trials + 1), "frame_skip": 1},
                           "resources": {"cpu_threads": 1, "memory_mb": 1024}}
                reset_request = {"idempotency_key": uuid.uuid4().hex, "seed": seed, "options": {}}
                started = time.perf_counter_ns()
                row = {"trial": index, "status": "failed"}
                samples["new_worker_to_first_observation"].append(row)
                created = client.create(request)
                created_at = time.perf_counter_ns()
                sid = created["session"]["session_id"]
                initial = client.reset(sid, reset_request)
                finished = time.perf_counter_ns()
                validate_observation(initial, 0)
                row.update(status="ok", elapsed_s=(finished - started) / 1e9,
                           create_request_s=(created_at - started) / 1e9,
                           initial_reset_request_s=(finished - created_at) / 1e9,
                           worker_pid=created.get("worker_pid"), session_id=sid,
                           initial_sim_time=initial["sim_time"], scene_sha256=initial["scene_sha256"])
                if index != creation_trials - 1:
                    client.close(sid)
                    sid = None
            for index in range(reset_trials):
                request = {"idempotency_key": uuid.uuid4().hex, "seed": seed, "options": {}}
                row = {"trial": index, "status": "failed"}
                samples["same_worker_reset"].append(row)
                started = time.perf_counter_ns()
                initial = client.reset(sid, request)
                elapsed = (time.perf_counter_ns() - started) / 1e9
                validate_observation(initial, 0)
                row.update(status="ok", elapsed_s=elapsed, episode_id=initial["episode_id"],
                           initial_sim_time=initial["sim_time"], scene_sha256=initial["scene_sha256"])
            previous_time = initial["sim_time"]
            for index in range(step_trials):
                request = {"episode_id": initial["episode_id"], "sequence_id": index,
                           "idempotency_key": uuid.uuid4().hex, "steps": 1,
                           "action": {"mode": "noop", "payload": {}}}
                row = {"trial": index, "status": "failed"}
                samples["single_step"].append(row)
                started = time.perf_counter_ns()
                observed = client.step(sid, request)
                elapsed = (time.perf_counter_ns() - started) / 1e9
                validate_observation(observed, index + 1)
                delta = observed["sim_time"] - previous_time
                if not math.isclose(delta, report["model"]["physics_dt_s"], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("step did not advance exactly one physics timestep")
                previous_time = observed["sim_time"]
                row.update(status="ok", elapsed_s=elapsed, sim_time=observed["sim_time"],
                           sim_time_delta_s=delta, contact_count=len(observed["observation"].get("contacts", [])),
                           terminated=observed["terminated"], truncated=observed["truncated"])
            report["status"] = "completed"
        except Exception as exc:
            report["status"] = "failed"
            report["failures"].append({"type": type(exc).__name__, "message": str(exc).replace(token, "<redacted>")})
        finally:
            if sid:
                try:
                    client.close(sid)
                except Exception as exc:
                    report["status"] = "failed"
                    report["failures"].append({"type": "cleanup", "message": str(exc).replace(token, "<redacted>")})
            server.shutdown()
            server.server_close()
            manager.close()
            server_thread.join(timeout=5)
    report["measurements"] = {name: summarize(rows) for name, rows in samples.items()}
    report["source_unchanged_during_measurement"] = source_hashes() == sources
    if not report["source_unchanged_during_measurement"]:
        report["status"] = "failed"
        report["failures"].append({"type": "source_changed", "message": "core source changed during benchmark"})
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--recipe", default="living_room_returned_home")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--creation-trials", type=int, default=30)
    parser.add_argument("--reset-trials", type=int, default=100)
    parser.add_argument("--step-trials", type=int, default=100)
    args = parser.parse_args()
    report = benchmark(args.output, scene_path=args.scene, recipe=args.recipe, seed=args.seed,
                       creation_trials=args.creation_trials, reset_trials=args.reset_trials, step_trials=args.step_trials)
    print(json.dumps({"status": report["status"], "report": str(args.output), "failures": report["failures"],
                      "measurements": {name: {key: value for key, value in data.items() if key != "samples"}
                                       for name, data in report["measurements"].items()}}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
