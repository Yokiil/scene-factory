"""Compare full reload against model reuse and measure reset/restore drift."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scene_factory.backends.mujoco import MujocoBackend
from scene_factory.dataset import write_json_atomic
from scene_factory.factory import SceneFactory


def _memory_bytes():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
                )
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        if not query(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            return None
        return {"working_set": counters.WorkingSetSize, "private_bytes": counters.PrivateUsage}
    path = Path("/proc/self/statm")
    if path.is_file():
        return {"working_set": int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
                "private_bytes": None}
    return None


def _summary(values):
    ordered = sorted(values)
    def quantile(fraction):
        index = (len(ordered) - 1) * fraction
        low = int(index)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)
    return {"samples_s": values, "count": len(values), "p50_s": statistics.median(values), "p95_s": quantile(.95)}


def validate(output: Path, *, reset_count: int = 100, stress_count: int = 1000):
    import mujoco
    import numpy as np

    factory = SceneFactory()
    scene = json.loads(json.dumps(factory.build_from_recipe("living_room_returned_home", 19).scene.to_dict()))
    backends = {"full_reload": MujocoBackend(registry=factory.registry, reuse_model=False),
                "model_reuse": MujocoBackend(registry=factory.registry, reuse_model=True)}
    times = {key: [] for key in backends}
    report = {
        "schema": "scene_factory.mujoco_reset_restore_validation.v1",
        "host": {"platform": platform.platform(), "cpu": platform.processor(), "python": sys.version,
                 "engine": mujoco.__version__},
        "scene": scene, "physics_acceptance": "not_evaluated",
        "conditions": {"background_load_isolated": False, "action_repeat": 5, "physics_dt_s": .002,
                       "rendering": False, "timing_includes_state_and_contact_observation": True},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT / "scene_factory/backends/mujoco.py", ROOT / "scene_factory/exporters/mujoco_mjcf.py",
                                       ROOT / "scene_factory/tasks.py", Path(__file__))},
    }
    try:
        for backend in backends.values():
            backend.reset(scene)
        initial = backends["model_reuse"].capture_state()
        max_reset_diff = 0.0
        for iteration in range(reset_count):
            order = list(backends) if iteration % 2 == 0 else list(reversed(backends))
            for key in order:
                backend = backends[key]
                backend.step(None)
                start = time.perf_counter_ns()
                backend.reset(scene)
                times[key].append((time.perf_counter_ns() - start) / 1e9)
                max_reset_diff = max(max_reset_diff, float(np.max(np.abs(
                    np.asarray(backend.capture_state()["state"]) - np.asarray(initial["state"]),
                ))))
        report["reset_timing"] = {key: _summary(values) for key, values in times.items()}
        report["reset_timing"]["median_reduction_fraction"] = 1 - statistics.median(times["model_reuse"]) / statistics.median(times["full_reload"])
        report["full_vs_reuse_max_integration_state_difference"] = max_reset_diff
        fast = backends["model_reuse"]
        dynamic_id = next(item["object_id"] for item in scene["objects"] if item["dynamic"])
        action = {"object_forces": {dynamic_id: [.01, .01, .01]}}
        memory = []
        gc.collect()
        memory.append({"iteration": 0, "memory_bytes": _memory_bytes()})
        max_stress_diff = 0.0
        for index in range(stress_count):
            fast.step(action)
            fast.reset(scene)
            max_stress_diff = max(max_stress_diff, float(np.max(np.abs(
                np.asarray(fast.capture_state()["state"]) - np.asarray(initial["state"]),
            ))))
            if (index + 1) % 100 == 0 or index == stress_count - 1:
                gc.collect()
                memory.append({"iteration": index + 1, "memory_bytes": _memory_bytes()})
        report["stress"] = {
            "reset_count": stress_count, "max_integration_state_difference": max_stress_diff,
            "memory_samples": memory, "compile_count": fast._model_compile_count,
            "cache_hits": fast._model_cache_hits,
            "memory_interpretation": "sampled process memory; bounded run does not prove absence of all leaks",
        }
        for _ in range(50):
            fast.step(None)
        snapshot = fast.capture_state()
        expected = []
        for _ in range(100):
            fast.step(action)
            expected.append((fast._data.qpos.copy(), fast._data.qvel.copy()))
        fast.restore_state(snapshot)
        max_position_error, max_velocity_error = 0.0, 0.0
        for qpos, qvel in expected:
            fast.step(action)
            max_position_error = max(max_position_error, float(np.max(np.abs(fast._data.qpos - qpos))))
            max_velocity_error = max(max_velocity_error, float(np.max(np.abs(fast._data.qvel - qvel))))
        report["restore"] = {"continuation_environment_steps": len(expected),
                             "max_qpos_absolute_difference": max_position_error,
                             "max_qvel_absolute_difference": max_velocity_error,
                             "scope": "same_process_model_engine_no_render"}
        report["status"] = "passed" if max(max_reset_diff, max_stress_diff, max_position_error, max_velocity_error) < 1e-12 else "failed"
        write_json_atomic(output, report)
        return report
    finally:
        for backend in backends.values():
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--stress-resets", type=int, default=1000)
    args = parser.parse_args()
    if args.resets < 1 or args.stress_resets < 1:
        parser.error("counts must be positive")
    report = validate(args.output, reset_count=args.resets, stress_count=args.stress_resets)
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "median_reduction_fraction": report["reset_timing"]["median_reduction_fraction"],
                      "reset_drift": report["stress"]["max_integration_state_difference"],
                      "restore": report["restore"]}, ensure_ascii=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
