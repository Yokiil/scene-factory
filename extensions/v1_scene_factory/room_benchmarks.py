"""Room-scale proxy fixtures and measured static-settling benchmarks.

These scenes reserve a workbench area, but contain no robot or assembly task.
The benchmark only establishes a background physics/performance slice.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any

from .dataset import write_json_atomic
from .factory import SceneFactory
from .paths import default_registry_path
from .registry import AssetRegistry


def generate_room_scene(object_count: int, *, seed: int = 19, registry: AssetRegistry | None = None) -> dict[str, Any]:
    if isinstance(object_count, bool) or object_count not in {20, 50, 100}:
        raise ValueError("room benchmark object_count must be 20, 50 or 100")
    factory = SceneFactory()
    registry = registry or factory.registry
    scene = copy.deepcopy(factory.build_from_recipe("living_room_returned_home", seed).scene.to_dict())
    # Convert dataclass tuples to the portable JSON contract before extending it.
    scene = json.loads(json.dumps(scene))
    scene["scene_id"] = f"living_room_scale_{object_count}_seed_{seed}"
    scene["room_dimensions_m"] = [14, 14, 3]
    scene["task"] = {}
    scene["description"] = "Living-room background scale fixture with reserved workbench; robot task unconfigured."
    workbench = registry.get("kitchen_counter_basic")
    scene["objects"].append({"object_id": "workbench", "asset_id": workbench.asset_id,
                            "category": workbench.category, "dynamic": False, "support": "floor",
                            "bbox_m": list(workbench.bbox_m), "pose": {"position": [0, -4, workbench.bbox_m[2]/2], "yaw_deg": 0}})
    asset = registry.get("remote_black")
    positions = []
    for y in range(-6, 7):
        for x in range(-6, 7):
            # Preserve central furniture and a 3x3m workbench movement envelope.
            if abs(x) < 3 and abs(y) < 3:
                continue
            if abs(x) <= 2 and -6 <= y <= -2:
                continue
            positions.append((float(x), float(y)))
    needed = object_count - len(scene["objects"])
    if needed > len(positions):
        raise ValueError("room fixture cannot place requested objects outside reserved work area")
    for index, (x, y) in enumerate(positions[:needed]):
        scene["objects"].append({"object_id": f"background_{index:03d}", "asset_id": asset.asset_id,
                                "category": asset.category, "dynamic": index % 5 != 0, "support": "floor",
                                "bbox_m": list(asset.bbox_m), "pose": {"position": [x, y, asset.bbox_m[2]/2], "yaw_deg": 0}})
    scene["benchmark_scope"] = {"background_collision": "primitive_proxy", "workbench_reserved_xy": [-1.5, 1.5, -5.5, -2.5],
                                "precision_partition": "fixed_at_build", "robot": "not_instantiated",
                                "local_task_acceptance": "not_verified"}
    return scene


def _quantile(values: list[float], probability: float) -> float:
    sorted_values = sorted(values)
    position = (len(values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    return sorted_values[left] + (sorted_values[right] - sorted_values[left]) * (position-left)


def run_room_benchmarks(output_dir: str | Path, *, physics_steps: int = 1000) -> dict[str, Any]:
    import importlib.metadata
    from .backends.mujoco import MujocoBackend

    if isinstance(physics_steps, bool) or not isinstance(physics_steps, int) or physics_steps < 300:
        raise ValueError("physics_steps must be >=300 to include a settled observation window")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = AssetRegistry.load(default_registry_path())
    rows = []
    for count in (20, 50, 100):
        scene = generate_room_scene(count, registry=registry)
        path = output / f"room_{count}.json"
        if path.exists():
            raise FileExistsError(f"room benchmark output exists: {path}")
        write_json_atomic(path, scene)
        backend = MujocoBackend(registry=registry, frame_skip=1, max_steps=physics_steps+1)
        started = time.perf_counter()
        try:
            _, info = backend.reset(scene)
            reset_s = time.perf_counter()-started
            initial_positions = backend._object_positions()
            durations, positions, speeds, contacts = [], [], [], []
            for index in range(physics_steps):
                step_start = time.perf_counter()
                backend.step(None)
                durations.append(time.perf_counter()-step_start)
                contacts.append(int(backend._data.ncon))
                if index >= physics_steps-100:
                    positions.append({object_id: list(position) for object_id, position in backend._object_positions().items()})
                    speeds.append(float(max(abs(value) for value in backend._data.qvel)) if len(backend._data.qvel) else 0)
            dynamic = [item["object_id"] for item in scene["objects"] if item["dynamic"]]
            drift = max((math.dist(positions[0][object_id], row[object_id]) for row in positions for object_id in dynamic), default=0)
            initial_displacement = max((math.dist(initial_positions[object_id], positions[-1][object_id]) for object_id in dynamic), default=0)
            finite = all(math.isfinite(value) for row in positions for point in row.values() for value in point)
            stable = finite and drift <= 0.005 and max(speeds) <= 0.1 and initial_displacement <= 0.01
            rows.append({"object_count": count, "dynamic_count": len(dynamic), "static_count": count-len(dynamic),
                         "joint_count": int(backend._model.njnt), "geom_count": int(backend._model.ngeom),
                         "mesh_vertices": int(backend._model.nmeshvert), "physics_dt_s": float(backend._model.opt.timestep),
                         "physics_steps": physics_steps, "peak_observed_contact_count": max(contacts),
                         "reset_s": reset_s, "step_p50_s": statistics.median(durations), "step_p95_s": _quantile(durations, .95),
                         "last_100_step_max_position_drift_m": drift, "last_100_step_max_generalized_speed": max(speeds),
                         "max_initial_to_final_displacement_m": initial_displacement,
                         "finite_state": finite, "background_settling": "passed" if stable else "failed",
                         "local_task_acceptance": "not_verified", "rendering": "disabled", "reset_info": info,
                         "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "step_samples_s": durations,
                         "resource_measurement": "RAM/GPU peak not instrumented"})
        finally:
            backend.close()
    report = {"schema_version": "scene_factory.room_scale_benchmark.v1", "platform": platform.platform(),
              "python": platform.python_version(), "mujoco": importlib.metadata.version("mujoco"),
              "scope": "background_proxy_settling_and_single_process_timing", "local_robot_task": "not_run", "rows": rows}
    write_json_atomic(output / "room_benchmarks.json", report)
    return report
