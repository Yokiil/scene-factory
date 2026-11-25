"""Produce package/cache, constrained-seed and room-scale evidence locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.constrained_randomization import sample_candidate_seeds, sample_instance_physics
from scene_factory.dataset import write_json_atomic
from scene_factory.environment_package import build_environment, verify_environment
from scene_factory.factory import SceneFactory
from scene_factory.physics_dependencies import plan_scene_update
from scene_factory.room_benchmarks import run_room_benchmarks


def run(output: Path, *, skip_rooms: bool = False) -> dict:
    if output.exists():
        raise FileExistsError("evidence output must be a new directory")
    output.mkdir(parents=True)
    factory = SceneFactory()
    scene = json.loads(json.dumps(factory.build_from_recipe("living_room_returned_home", 19).scene.to_dict()))
    path = output / "source_layout.json"
    write_json_atomic(path, scene)
    cache = output / "cache"
    builds = []
    for name in ("environment", "environment_cached"):
        start = time.perf_counter()
        result = build_environment(path, output / name, cache_dir=cache)
        builds.append({"name": name, "elapsed_s": time.perf_counter()-start, "cache": result["cache"],
                       "environment_sha256": result["environment_sha256"], "valid": verify_environment(output / name)["valid"]})
    config = {"schema_version": "scene_factory.constrained_randomization.v1", "instances": {
        item["object_id"]: {"scale": [0.95, 1.05], "density_scale": [0.8, 1.2], "friction": [0.2, 0.8]}
        for item in scene["objects"] if item["dynamic"]
    }}
    write_json_atomic(output / "randomization_config.json", config)
    collections = []
    for count in (100, 1000):
        report = sample_candidate_seeds(scene, config, list(range(count)), registry=factory.registry)
        write_json_atomic(output / f"randomization_{count}.json", report)
        collections.append({"candidate_count": count, "counts": report["counts"], "rejection_counts": report["rejection_counts"]})
    sampled, sampling = sample_instance_physics(scene, config, 42, registry=factory.registry)
    write_json_atomic(output / "sampled_layout.json", sampled)
    write_json_atomic(output / "sampled_parameters.json", sampling)
    changed = build_environment(output / "sampled_layout.json", output / "environment_sampled", cache_dir=cache)
    diff = plan_scene_update(scene, sampled)
    write_json_atomic(output / "invalidation.json", diff)
    rooms = {"rows": []} if skip_rooms else run_room_benchmarks(output / "rooms", physics_steps=1000)
    summary = {"schema_version": "scene_factory.environment_delivery_evidence.v1",
               "packages": builds, "sampled_package": changed, "collections": collections,
               "room_settling": [{key: row[key] for key in ("object_count", "dynamic_count", "static_count", "joint_count", "geom_count", "peak_observed_contact_count", "background_settling", "last_100_step_max_position_drift_m")} for row in rooms["rows"]],
               "room_benchmark_status": "not_run_requested_skip" if skip_rooms else "completed",
               "assembly_robot_acceptance": "not_run", "linux_gpu_acceptance": "not_run",
               "incremental_compiler_scope": "dependency invalidation and unchanged content reuse; changed MJCF fully recompiled",
               "physics_evidence_reuse": "forbidden_after_changes_without_revalidation"}
    write_json_atomic(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-rooms", action="store_true", help="Regenerate environment evidence without rerunning room physics or timings")
    args = parser.parse_args()
    result = run(args.output, skip_rooms=args.skip_rooms)
    print(json.dumps({"output": str(args.output.resolve()), "collections": result["collections"], "room_settling": result["room_settling"]}, indent=2))
