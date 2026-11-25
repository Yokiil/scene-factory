"""Measure compatible MuJoCo cross-session reuse and dirty-reset isolation.

Loopback-only, state-only proxy physics. Baseline workers are new processes;
the management process and OS caches stay warm. No robot-task or GPU claim.
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
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_factory.dataset import write_json_atomic
from scene_factory.runtime_client import RuntimeClient
from scene_factory.runtime_service import RuntimeService, serve
from scene_factory.worker_pool import _RUNTIME_SOURCES


def summary(values):
    ordered = sorted(values)
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"count": len(values), "p50_s": statistics.median(values), "p95_s": percentile(.95), "samples_s": values}


def source_hashes():
    return {name: hashlib.sha256((ROOT / "scene_factory" / name).read_bytes()).hexdigest() for name in _RUNTIME_SOURCES}


def process_memory(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            query = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if not query(handle, ctypes.byref(counters), counters.cb):
                return None
            return {"working_set_bytes": counters.WorkingSetSize, "private_bytes": counters.PrivateUsage}
        finally:
            kernel.CloseHandle(handle)
    statm = Path(f"/proc/{pid}/statm")
    return {"working_set_bytes": int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"), "private_bytes": None} if statm.is_file() else None


def validate(output: Path, trials: int, resets: int):
    if output.exists():
        raise FileExistsError("choose a fresh output directory to preserve evidence")
    if type(trials) is not int or trials < 1 or type(resets) is not int or resets < 1:
        raise ValueError("trial counts must be positive integers")
    output.mkdir(parents=True)
    scene = {"scene_id": "worker-pool-measured", "room_dimensions_m": [2, 2, 3], "objects": [
        {"object_id": "part", "asset_id": "mug_blue", "dynamic": True, "bbox_m": [.04, .04, .06],
         "pose": {"position": [0, 0, .031], "yaw_deg": 0}}], "task": {}}
    write_json_atomic(output / "scene.json", scene)
    report = {"schema_version": "scene_factory.worker_pool_validation.v1", "status": "running", "host": {"platform": platform.platform(), "python": sys.version,
              "engine": "mujoco", "engine_version": importlib.metadata.version("mujoco")}, "source_sha256": source_hashes(),
              "conditions": {"loopback": True, "concurrency": 1, "rendering": False, "physics_dt_s": .002, "frame_skip": 1,
                             "scene": scene, "resources": {"cpu_threads": 1, "memory_mb": 256}, "background_load_isolated": False,
                             "baseline_cache_state": "new_worker_each_trial; resident_manager_and_unflushed_OS_caches",
                             "warm_cache_state": "same_exact_scene_settings_resources_engine_sources; initial_pool_fill_excluded"},
              "physics_task_acceptance": "not_evaluated", "runs": {}}
    expected_objects = None
    try:
        for enabled in (False, True):
            manager = RuntimeService(output / ("pool_service" if enabled else "baseline_service"), allowed_scene_root=output,
                                     max_sessions=1, cpu_budget=1, memory_budget_mb=1024, mujoco_worker_pool=enabled, pool_max_uses=trials + 2)
            server = serve(manager, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
            samples, pid_set, sid, initial = [], set(), None, None
            run = {"creation_samples": samples}
            report["runs"]["pooled" if enabled else "new_worker_baseline"] = run
            try:
                for index in range(trials + int(enabled)):
                    started = time.perf_counter_ns()
                    created = client.create({"scene_path": str(output / "scene.json"), "backend": "mujoco", "backend_options": {"max_steps": 10000, "frame_skip": 1},
                                             "resources": {"cpu_threads": 1, "memory_mb": 256}})
                    sid = created["session_id"]
                    initial = client.reset(sid, {"idempotency_key": "reset", "seed": 42})
                    elapsed = (time.perf_counter_ns() - started) / 1e9
                    if expected_objects is None:
                        expected_objects = initial["observation"]["objects"]
                    if initial["observation"]["objects"] != expected_objects:
                        raise ValueError("cross-session initial physics state drift")
                    if initial["info"]["model_compile_count"] != 1:
                        raise ValueError("compatible worker unexpectedly recompiled its model")
                    row = {"trial": index, "elapsed_s": elapsed, "worker_pid": created["worker_pid"], "session_id": sid,
                           "episode_id": initial["episode_id"], "worker_reused": created["worker_reused"], "model_compile_count": initial["info"]["model_compile_count"],
                           "reset_kind": initial["info"]["reset_kind"], "compatibility_sha256": created.get("worker_pool_compatibility_sha256")}
                    if enabled and index == 0:
                        run["initial_pool_fill"] = row
                    else:
                        samples.append(row)
                        pid_set.add(created["worker_pid"])
                        if enabled and not created["worker_reused"]:
                            raise ValueError("pooled creation used a new process")
                    if not enabled or index != trials:
                        closed = client.close(sid)
                        if enabled and closed["worker_disposition"] != "pooled":
                            raise ValueError("healthy initialized worker was not parked")
                        sid = None
                run["creation_to_first_observation"] = summary([row["elapsed_s"] for row in samples])
                run["distinct_measured_worker_pids"] = sorted(pid_set)
                if enabled:
                    if len(pid_set) != 1:
                        raise ValueError("compatible pooled trials did not retain a single worker PID")
                    reset_times, max_difference = [], 0.
                    memory = [{"reset": 0, "memory": process_memory(next(iter(pid_set)))}]
                    for index in range(resets):
                        before = client.step(sid, {"episode_id": initial["episode_id"], "sequence_id": 0, "idempotency_key": "dirty-step",
                                                   "action": {"mode": "object_forces", "payload": {"object_forces": {"part": [.1, 0, 6]}}}})
                        if before["observation"]["objects"] == expected_objects:
                            raise ValueError("stress action did not perturb the measured physical state")
                        started = time.perf_counter_ns()
                        initial = client.reset(sid, {"idempotency_key": f"stress-reset-{index}", "seed": 42})
                        reset_times.append((time.perf_counter_ns() - started) / 1e9)
                        for object_id, expected in expected_objects.items():
                            measured = initial["observation"]["objects"][object_id]
                            for key in ("position_m", "orientation_wxyz", "linear_velocity_mps", "angular_velocity_rps"):
                                max_difference = max(max_difference, max(abs(a - b) for a, b in zip(expected[key], measured[key], strict=True)))
                        if (index + 1) % 100 == 0 or index == resets - 1:
                            memory.append({"reset": index + 1, "memory": process_memory(next(iter(pid_set)))})
                    run["dirty_reset_stress"] = {"reset_count": resets, "max_state_component_difference": max_difference,
                                                  "model_compile_count": initial["info"]["model_compile_count"], "reset_timing": summary(reset_times), "memory_samples": memory,
                                                  "memory_interpretation": "sampled bounded-run worker memory; no general leak-free or hard-quota claim"}
                    if max_difference != 0 or initial["info"]["model_compile_count"] != 1:
                        raise ValueError("stress reset drift or unintended model recompilation")
                run["pool_status"] = manager.worker_pool_status()
            finally:
                if sid is not None:
                    client.close(sid)
                server.shutdown()
                server.server_close()
                manager.close()
                thread.join(3)
        report["source_unchanged"] = report["source_sha256"] == source_hashes()
        if not report["source_unchanged"]:
            raise ValueError("runtime sources changed during the measurement")
        baseline = report["runs"]["new_worker_baseline"]["creation_to_first_observation"]["p50_s"]
        pooled = report["runs"]["pooled"]["creation_to_first_observation"]["p50_s"]
        report["creation_median_reduction_fraction"] = 1 - pooled / baseline
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json_atomic(output / "report.json", report)
    print(json.dumps({"status": report["status"], "trials_per_group": trials, "dirty_resets": resets,
                      "baseline_p50_s": baseline, "pooled_p50_s": pooled, "reduction_fraction": report["creation_median_reduction_fraction"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--reset-trials", type=int, default=1000)
    args = parser.parse_args()
    validate(args.output.resolve(), args.trials, args.reset_trials)


if __name__ == "__main__":
    main()
