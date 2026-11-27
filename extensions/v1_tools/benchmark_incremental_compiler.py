"""Four-strategy, finite-horizon contact-proxy compilation experiment.

This does not benchmark Franka assembly. It measures native MuJoCo rebuilds,
whole-file caching, object dependencies plus safe parameter patches, and a fixed
background partition of geometrically equivalent boxes. Every variant in every
arm is fully replayed; no previous physics acceptance is cached.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.dataset import write_json_atomic
from scene_factory.incremental_compiler import IncrementalMujocoCompiler, compiler_source_paths
from scene_factory.incremental_evidence import CONTACT_SCHEMA, compare_contacts, freeze_inputs, validate_contacts, verify_inputs
from scene_factory.paths import default_registry_path
from scene_factory.registry import AssetRegistry


def fixture(registry: AssetRegistry, backgrounds: int) -> tuple[dict, set[str]]:
    primitive = registry.get("remote_black")
    mesh = registry.get("kd_box_01")
    part = {"object_id": "part", "asset_id": primitive.asset_id, "dynamic": True,
            "support": "floor", "bbox_m": [.08, .08, .08],
            "pose": {"position": [0, 0, .04], "yaw_deg": 0},
            "physics": {"mass_kg": 1., "inertia_kg_m2": [.08**2 / 6]*3,
                        "friction": .2, "source": "declared_synthetic_contact_proxy"}}
    stop = {"object_id": "stop", "asset_id": primitive.asset_id, "dynamic": False,
            "support": "floor", "bbox_m": [.1, .8, .2],
            "pose": {"position": [.16, 0, .1], "yaw_deg": 0}}
    objects = [part, stop]
    for index in range(backgrounds):
        objects.append({"object_id": f"background_{index:03d}", "asset_id": mesh.asset_id,
                        "dynamic": False, "support": "floor", "bbox_m": list(mesh.bbox_m),
                        "pose": {"position": [3 + index % 10 * .4, -2 + index // 10 * .4, mesh.bbox_m[2]/2], "yaw_deg": 0}})
    return {"scene_id": "incremental_contact_proxy", "room_dimensions_m": [16, 10, 3],
            "objects": objects, "task": {},
            "benchmark_task": {"kind": "push_free_body_against_fixed_stop", "robot": None}}, {
                item["object_id"] for item in objects if item["object_id"].startswith("background_")}


def variants(base: dict, count: int, *, pattern: str = "repeated") -> list[dict]:
    result = []
    for index in range(count):
        scene = copy.deepcopy(base)
        part = scene["objects"][0]
        # Five unique variants are repeated to expose the whole-file cache arm.
        tick = index % 10
        mass = .85 + (.005 * index if pattern == "unique" else .05 * (tick % 5))
        part["physics"]["mass_kg"] = mass
        part["physics"]["inertia_kg_m2"] = [mass*.08**2/6]*3
        part["physics"]["friction"] = .15 + (.001 * index if pattern == "unique" else .025 * (tick % 5))
        part["pose"]["position"][1] = -.01 + .005 * (tick % 5)
        if tick == 5:
            part["bbox_m"][0] = .084  # Native geometry fallback.
        if tick == 6:
            scene["objects"][1]["pose"]["position"][0] = .17  # Static pose fallback.
        if tick == 7:
            part["pose"]["position"][0] = .015  # New/changed initial contact candidates.
        if tick == 8:
            added = copy.deepcopy(scene["objects"][1])
            added["object_id"] = "new_obstacle"
            added["pose"]["position"] = [-.4, .5, .1]
            scene["objects"].append(added)  # Topology fallback.
        result.append(scene)
    return result


def quantile(values, probability):
    values = sorted(values)
    point = (len(values)-1) * probability
    lower, upper = math.floor(point), math.ceil(point)
    return values[lower] + (values[upper]-values[lower])*(point-lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=30)
    parser.add_argument("--backgrounds", type=int, default=60)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--pattern", choices=("repeated", "unique"), default="repeated")
    args = parser.parse_args()
    if args.variants < 10 or args.backgrounds < 1 or args.steps < 100:
        parser.error("need >=10 variants, >=1 background, >=100 physics steps")
    if args.output.exists():
        raise FileExistsError("use a new evidence directory")
    args.output.mkdir(parents=True)
    import mujoco
    import numpy as np

    source_registry = AssetRegistry.load(default_registry_path())
    base, backgrounds = fixture(source_registry, args.backgrounds)
    registry, input_manifest = freeze_inputs(default_registry_path(), {item["asset_id"] for item in base["objects"]}, args.output / "inputs")
    input_manifest_sha256 = hashlib.sha256((args.output / "inputs/manifest.json").read_bytes()).hexdigest()
    scenes = variants(base, args.variants, pattern=args.pattern)
    write_json_atomic(args.output / "base_scene.json", base)
    write_json_atomic(args.output / "variants.json", scenes)
    references = []
    rows, raw = [], {}
    for mode in ("full", "file_cache", "incremental", "partition"):
        compiler = IncrementalMujocoCompiler(registry, mode=mode, background_ids=backgrounds,
                                             horizon_s=args.steps*.002, max_linear_speed_mps=2.)
        warmup = compiler.compile(base)
        mode_rows = []
        for index, scene in enumerate(scenes):
            result = compiler.compile(scene)
            model, data = result.model, result.data
            part_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object__part")
            stop_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object__stop")
            contact_seen, max_speed, peak_contact_force = False, 0., 0.
            trajectory, contact_steps, step_durations = [], [], []
            rollout_start = time.perf_counter()
            for step_index in range(args.steps):
                data.xfrc_applied[part_id, :3] = [12., 0., 0.]
                step_start = time.perf_counter()
                mujoco.mj_step(model, data)
                step_durations.append(time.perf_counter()-step_start)
                trajectory.append(np.concatenate([data.qpos, data.qvel]).copy())
                contact_points = []
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
                    if bodies == {part_id, stop_id}:
                        contact_seen = True
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, contact_index, force)
                    peak_contact_force = max(peak_contact_force, abs(float(force[0])))
                    contact_points.append({
                        "geom_pair": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or f"geom_{int(geom)}" for geom in (contact.geom1, contact.geom2)],
                        "geom_indices": [int(contact.geom1), int(contact.geom2)],
                        "body_pair": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "world" for geom in (contact.geom1, contact.geom2)],
                        "normal_force_N": abs(float(force[0])), "wrench_contact_frame_N_Nm": force.tolist(),
                        "position_m": contact.pos.tolist(), "signed_distance_m": float(contact.dist)})
                contact_points.sort(key=lambda row: (row["geom_pair"], row["position_m"]))
                contact_steps.append({"step_index": step_index+1, "sim_time_s": float(data.time),
                                      "contact_count": int(data.ncon), "contacts": contact_points})
                max_speed = max(max_speed, float(np.linalg.norm(data.qvel[:3])))
            rollout_s = time.perf_counter() - rollout_start
            contacts = {"schema_version": CONTACT_SCHEMA, "scope": "all_MuJoCo_contact_points_after_each_mj_step",
                        "force_type": "instantaneous_contact_frame_force_from_mj_contactForce",
                        "normal_force_aggregation": "point_values_saved_individually_pair_sum_compared", "steps": contact_steps}
            validate_contacts(contacts, expected_steps=args.steps, dt_s=.002)
            trajectory = np.asarray(trajectory)
            finite = bool(np.isfinite(trajectory).all())
            speed_bound_ok = max_speed <= 2.
            final = data.xpos[part_id].copy()
            wall_front = float(scene["objects"][1]["pose"]["position"][0]) - .05
            expected_x = wall_front - scene["objects"][0]["bbox_m"][0]/2
            position_ok = abs(float(final[0])-expected_x) < .003
            success = finite and speed_bound_ok and contact_seen and position_ok
            if mode == "full":
                references.append((trajectory, contacts))
                delta = 0.
                contact_comparison = compare_contacts(contacts, contacts)
            else:
                delta = float(np.max(np.abs(trajectory-references[index][0])))
                contact_comparison = compare_contacts(references[index][1], contacts)
            trace_directory = args.output / "trajectories" / mode
            trace_directory.mkdir(parents=True, exist_ok=True)
            trace_path = trace_directory / f"{index:03d}.npy"
            np.save(trace_path, trajectory, allow_pickle=False)
            contacts_path = trace_directory / f"{index:03d}.contacts.json"
            write_json_atomic(contacts_path, contacts)
            row = {"variant": index, **result.report, "rollout_s": rollout_s,
                   "physics_steps": args.steps, "step_p50_s": statistics.median(step_durations),
                   "step_p95_s": quantile(step_durations, .95), "step_samples_s": step_durations,
                   "max_state_difference_vs_full": delta, "contact_pairs_identical_vs_full": contact_comparison["contact_pairs_identical"],
                   "stepwise_contact_comparison_vs_full": contact_comparison,
                   "final_position_m": final.tolist(), "finite": finite,
                   "measured_max_linear_speed_mps": max_speed, "envelope_speed_bound_satisfied": speed_bound_ok,
                   "contact_with_stop": contact_seen, "peak_normal_contact_N": peak_contact_force,
                   "target_position_ok": position_ok, "task_success": success,
                   "nbody": int(model.nbody), "ngeom": int(model.ngeom), "nmesh": int(model.nmesh),
                   "nmeshvert": int(model.nmeshvert), "dynamics_replay": "full_every_variant",
                   "local_evidence_reuse": False,
                   "trajectory": {"path": str(trace_path.relative_to(args.output)).replace('\\','/'),
                                  "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                                  "shape": list(trajectory.shape), "dtype": str(trajectory.dtype)},
                   "contact_trace": {"path": str(contacts_path.relative_to(args.output)).replace('\\','/'),
                                     "sha256": hashlib.sha256(contacts_path.read_bytes()).hexdigest()}}
            mode_rows.append(row)
        raw[mode] = mode_rows
        durations = [row["compile_total_s"] for row in mode_rows]
        total = [row["compile_total_s"]+row["rollout_s"] for row in mode_rows]
        rows.append({"mode": mode, "variants": len(mode_rows), "warmup": warmup.report,
                     "compile_p50_s": statistics.median(durations), "compile_p95_s": quantile(durations, .95),
                     "compile_and_full_replay_p50_s": statistics.median(total),
                     "task_success_count": sum(row["task_success"] for row in mode_rows),
                     "native_compile_count_measured": sum(row["kind"] == "native_compile" for row in mode_rows),
                     "parameter_patch_count_measured": sum(row["kind"] == "episode_boundary_parameter_patch" for row in mode_rows),
                     "file_cache_hits": sum(row["whole_file_cache_hit"] for row in mode_rows),
                     "max_state_difference_vs_full": max(row["max_state_difference_vs_full"] for row in mode_rows),
                     "all_contact_pairs_identical_vs_full": all(row["contact_pairs_identical_vs_full"] for row in mode_rows),
                     "max_normal_force_difference_N": max(row["stepwise_contact_comparison_vs_full"]["max_normal_force_difference_N"] for row in mode_rows),
                     "max_pair_normal_sum_difference_N": max(row["stepwise_contact_comparison_vs_full"]["max_pair_normal_sum_difference_N"] for row in mode_rows),
                     "max_wrench_component_difference": max(row["stepwise_contact_comparison_vs_full"]["max_wrench_component_difference"] for row in mode_rows),
                     "max_contact_position_difference_m": max(row["stepwise_contact_comparison_vs_full"]["max_contact_position_difference_m"] for row in mode_rows),
                     "all_speed_bounds_satisfied": all(row["envelope_speed_bound_satisfied"] for row in mode_rows)})
        print(json.dumps({key: value for key, value in rows[-1].items() if key != "warmup"}), flush=True)
    compiler_hashes = {row["warmup"]["compiler_source_sha256"] for row in rows}
    input_check = verify_inputs(args.output / "inputs", input_manifest_sha256)
    live_source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in compiler_source_paths())).hexdigest()
    sources_match = compiler_hashes == {input_manifest["compiler_source_sha256"]} == {live_source_hash}
    acceptance = sources_match and input_check["status"] == "passed" and all(row["task_success_count"] == args.variants and row["max_state_difference_vs_full"] < 1e-10 and row["all_contact_pairs_identical_vs_full"] and row["all_speed_bounds_satisfied"] and row["max_normal_force_difference_N"] < 1e-8 and row["max_pair_normal_sum_difference_N"] < 1e-8 and row["max_wrench_component_difference"] < 1e-8 and row["max_contact_position_difference_m"] < 1e-10 for row in rows)
    report = {"schema_version": "scene_factory.incremental_four_arm_benchmark.v2",
              "status": "passed" if acceptance else "failed", "platform": platform.platform(),
              "python": platform.python_version(), "mujoco": importlib.metadata.version("mujoco"),
              "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
              "scope": "CPU_free_body_contact_proxy_not_robot_assembly",
              "fixed_partition": sorted(backgrounds), "randomization": "deterministic_declared_variant_sequence",
              "variant_pattern": args.pattern,
              "compiler_sources_unchanged_between_arms": len(compiler_hashes) == 1,
              "input_lock": {"path": "inputs/manifest.json", "sha256": input_manifest_sha256,
                             "consumed_registry": "inputs/registry.jsonl", "raw_registry_sha256": input_manifest["original_registry_sha256"],
                             "verified_after_run": input_check, "compiler_matches_frozen_source_closure": sources_match},
              "scene_inputs": {name: {"path": name, "sha256": hashlib.sha256((args.output/name).read_bytes()).hexdigest()} for name in ("base_scene.json", "variants.json")},
              "physics": {"dt_s": .002, "integrator": "implicitfast", "rendering": False,
                          "steps_per_variant": args.steps, "concurrency": 1, "force_N": [12, 0, 0],
                          "trace_disk_write_in_timing": False},
              "validation": {"state_atol": 1e-10, "normal_force_atol_N": 1e-8, "wrench_component_atol": 1e-8,
                             "contact_position_atol_m": 1e-10, "target_x_error_m": .003, "max_linear_speed_mps": 2.,
                             "full_rollout_every_variant": True, "accepted_cached_task_evidence": False},
              "limitations": ["Background exact-box mesh-to-box replacement changes contact algorithms; no background dynamic contacts occur within monitored horizon.",
                              "No isolation from unrelated OS load; no peak RAM measurement.",
                              "The local envelope queries include new candidates but do not authorize reusing prior physics/task evidence.",
                              "Results do not establish patent novelty or high-fidelity robot insertion performance."],
              "sources": {str(path.relative_to(Path(__file__).resolve().parents[1])).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), Path(__file__).resolve().parents[1]/"scene_factory/incremental_compiler.py")},
              "rows": rows, "raw": raw}
    write_json_atomic(args.output / "benchmark.json", report)
    return 0 if acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
