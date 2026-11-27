"""Measure one Isaac room's background settling while holding default joint targets.

This is a background-only validation, never a robot assembly acceptance. Run it
with Isaac Python, once per scene/process. Importing it or using --help starts
no simulator.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCOPE = "background_settling_only_not_robot_task"
THRESHOLDS = {"drift_m": .001, "displacement_m": .005,
              "speed_mps": .005, "angular_speed_rps": .05}
FIXTURE_FILES = ("scene.usda", "layout.json", "assembly_spec.json", "SOURCE.json")


class IncompleteEvidence(RuntimeError):
    """A physical pass cannot be established with incomplete measurements."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IncompleteEvidence(f"{label} is not a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise IncompleteEvidence(f"{label} exceeds floating point range") from exc
    if not math.isfinite(result):
        raise IncompleteEvidence(f"{label} is not finite")
    return result


def _vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise IncompleteEvidence(f"{label} requires {size} values")
    return [_number(item, label) for item in value]


def _norm(value: list[float]) -> float:
    return math.sqrt(sum(item*item for item in value))


def source_manifest() -> dict[str, Any]:
    paths = [*sorted((ROOT / "scene_factory").rglob("*.py")), Path(__file__).resolve()]
    hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}
    return {"scope": "all_scene_factory_python_and_this_validator",
            "sha256": hashlib.sha256(canonical_bytes(hashes)).hexdigest(), "files": hashes}


def freeze_fixture(fixture: Path, output: Path) -> tuple[dict, dict, dict]:
    """Preserve exact original bytes; reject a manifest/geometry mismatch."""
    frozen = output / "frozen_inputs"
    frozen.mkdir()
    hashes = {}
    for name in FIXTURE_FILES:
        data = (fixture / name).read_bytes()
        (frozen / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    source = json.loads((frozen / "SOURCE.json").read_text(encoding="utf-8"))
    layout = json.loads((frozen / "layout.json").read_text(encoding="utf-8"))
    for manifest_key, filename in (("usd_sha256", "scene.usda"),
                                   ("layout_sha256", "layout.json"),
                                   ("frozen_spec_sha256", "assembly_spec.json")):
        recorded = source.get(manifest_key)
        if not isinstance(recorded, str) or recorded != hashes[filename]:
            raise IncompleteEvidence(f"fixture SOURCE.{manifest_key} does not bind {filename}")
    backgrounds = [row for row in layout.get("objects", []) if row.get("category") == "background"]
    dynamics = [row["object_id"] for row in backgrounds if row.get("dynamic")]
    if not backgrounds or not dynamics:
        raise ValueError("fixture must contain identified static/dynamic room backgrounds")
    declared = layout.get("room_fixture", {})
    if (len(backgrounds) != declared.get("background_count")
            or len(dynamics) != declared.get("background_dynamic_count")
            or len(layout["objects"]) != declared.get("object_count")):
        raise IncompleteEvidence("fixture background/object counts differ from declared room scale")
    if "assembly_spec" in layout.get("task", {}):
        raise ValueError("background validation requires the original non-inline-assembly room layout")
    return hashes, layout, source


def make_sample(response: dict, background_ids: list[str], *, kind: str,
                action: dict | None, wall_time_ns: int | None) -> dict:
    observation = response.get("observation", {})
    objects = observation.get("objects", {})
    robot = observation.get("robot", {})
    return {"sample_kind": kind, "time_s": response.get("sim_time"),
            "episode_id": response.get("episode_id"), "step_index": response.get("step_index"),
            "state_version": response.get("state_version"),
            "objects": {name: objects.get(name) for name in background_ids},
            "contacts": observation.get("contacts"),
            "contact_coverage": observation.get("contact_coverage"),
            "robot": {key: robot.get(key) for key in ("joint_names", "joint_positions", "joint_velocities")},
            "action": action, "step_call_wall_time_ns": wall_time_ns,
            "terminated": response.get("terminated"), "truncated": response.get("truncated"),
            "pose_writes_after_initialization": False}


def validate_sample(sample: dict, *, background_ids: list[str], dynamic_ids: set[str],
                    scene_ids: set[str], dt: float, previous: dict | None = None) -> None:
    """Reject missing motion or incomplete full-stage contact measurements."""
    timestamp = _number(sample.get("time_s"), "time_s")
    if previous is not None:
        elapsed = timestamp - _number(previous.get("time_s"), "previous.time_s")
        if not math.isclose(elapsed, dt, rel_tol=1e-6, abs_tol=1e-8):
            raise IncompleteEvidence("trace is not continuous at every physics step")
        if (sample.get("step_index") != previous.get("step_index", -1) + 1
                or sample.get("state_version") != previous.get("state_version", -1) + 1
                or sample.get("episode_id") != previous.get("episode_id")):
            raise IncompleteEvidence("episode/step/state sequence is not continuous")
    objects = sample.get("objects")
    if not isinstance(objects, dict) or set(objects) != set(background_ids):
        raise IncompleteEvidence("background motion coverage does not match the frozen layout")
    for name, state in objects.items():
        if not isinstance(state, dict):
            raise IncompleteEvidence(f"missing background state: {name}")
        _vector(state.get("position"), 3, f"{name}.position")
        quaternion = _vector(state.get("orientation_wxyz"), 4, f"{name}.orientation_wxyz")
        if not math.isclose(_norm(quaternion), 1., abs_tol=1e-4):
            raise IncompleteEvidence(f"{name} orientation is not a unit quaternion")
        _vector(state.get("linear_velocity_mps"), 3, f"{name}.linear_velocity_mps")
        _vector(state.get("angular_velocity_rps"), 3, f"{name}.angular_velocity_rps")
        if name in dynamic_ids and state.get("physics_handle_valid") is not True:
            raise IncompleteEvidence(f"dynamic background has no live PhysX handle: {name}")
    coverage = sample.get("contact_coverage")
    if not isinstance(coverage, dict) or any(coverage.get(key) is not True for key in ("complete", "continuous")):
        raise IncompleteEvidence("contact coverage is missing or incomplete")
    if (coverage.get("scope") != "all_scene_contacts"
            or coverage.get("force_aggregation") != "sum_per_point_normal_magnitudes"
            or coverage.get("buffer_saturated") is not False
            or not isinstance(coverage.get("source"), str) or not coverage["source"].strip()):
        raise IncompleteEvidence("contact coverage scope/force sum/buffer attestation is unsupported")
    covered = coverage.get("object_ids")
    if not isinstance(covered, list) or not (scene_ids | {"robot"}).issubset(covered):
        raise IncompleteEvidence("contact coverage omits a scene object or the robot")
    if not math.isclose(_number(coverage.get("sample_dt_s"), "coverage.sample_dt_s"), dt,
                        rel_tol=1e-6, abs_tol=1e-9):
        raise IncompleteEvidence("contact sample window differs from the physics timestep")
    paths, buffers = coverage.get("collider_paths"), coverage.get("contact_buffers")
    if (not isinstance(paths, list) or not paths or len(paths) != len(set(paths))
            or coverage.get("collision_shape_count") != len(paths)
            or not isinstance(buffers, list) or not buffers
            or coverage.get("sensor_root_count") != len(buffers)):
        raise IncompleteEvidence("contact collider/buffer inventory is incomplete")
    for buffer in buffers:
        used, capacity = buffer.get("used_contact_points"), buffer.get("capacity")
        if (not isinstance(used, int) or isinstance(used, bool) or not isinstance(capacity, int)
                or isinstance(capacity, bool) or not 0 <= used < capacity
                or buffer.get("saturated") is not False):
            raise IncompleteEvidence("contact buffer is saturated or malformed")
    contacts = sample.get("contacts")
    if not isinstance(contacts, list):
        raise IncompleteEvidence("contact pair measurements are missing")
    pairs = set()
    for contact in contacts:
        pair = contact.get("pair")
        if (not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2
                or not set(pair).issubset(covered) or tuple(sorted(pair)) in pairs):
            raise IncompleteEvidence("contact pair owner/uniqueness is invalid")
        pairs.add(tuple(sorted(pair)))
        force = _number(contact.get("normal_force_N"), "contact.normal_force_N")
        resultant = _number(contact.get("normal_force_resultant_N"), "contact.normal_force_resultant_N")
        count = contact.get("contact_point_count")
        if (force < 0 or resultant < 0 or resultant > force + 1e-4
                or not isinstance(count, int) or isinstance(count, bool) or count < 0
                or contact.get("measurement") != "sum_of_absolute_per_contact_point_normal_forces"
                or contact.get("units") != "N"
                or not math.isclose(_number(contact.get("sample_window_s"), "contact.sample_window_s"),
                                    dt, rel_tol=1e-6, abs_tol=1e-9)):
            raise IncompleteEvidence("normal contact point sum/count/units are invalid")


def evaluate_background(initial: dict, window: list[dict], dynamic_ids: set[str], dt: float) -> dict:
    """Apply the unchanged assembly-room background protocol to a final window."""
    result = {"status": "not_verified", "passed": False, "thresholds": dict(THRESHOLDS),
              "sampling": "every_physics_step_final_continuous_1s_rolling_window",
              "initial_reference": "first_committed_post_reset_observation_after_engine_warmup",
              "evaluated_dynamic_ids": sorted(dynamic_ids), "sample_count": len(window)}
    if not dynamic_ids or len(window) < 2:
        return {**result, "reason": "insufficient_background_window"}
    span = window[-1]["time_s"] - window[0]["time_s"]
    result["window_s"] = span
    if span < 1. - 1e-8 or any(not math.isclose(right["time_s"] - left["time_s"], dt,
                                               rel_tol=1e-6, abs_tol=1e-8)
                               for left, right in zip(window, window[1:])):
        return {**result, "reason": "final_continuous_1s_not_observed"}
    first, last = window[0]["objects"], window[-1]["objects"]
    drift = max(math.dist(first[name]["position"], row["objects"][name]["position"])
                for row in window for name in dynamic_ids)
    displacement = max(math.dist(initial["objects"][name]["position"], last[name]["position"])
                       for name in dynamic_ids)
    speed = max(_norm(row["objects"][name]["linear_velocity_mps"])
                for row in window for name in dynamic_ids)
    spin = max(_norm(row["objects"][name]["angular_velocity_rps"])
               for row in window for name in dynamic_ids)
    passed = (drift <= THRESHOLDS["drift_m"] and displacement <= THRESHOLDS["displacement_m"]
              and speed <= THRESHOLDS["speed_mps"] and spin <= THRESHOLDS["angular_speed_rps"])
    return {**result, "status": "passed" if passed else "failed", "passed": passed,
            "max_position_drift_m": drift, "max_total_displacement_m": displacement,
            "max_speed_mps": speed, "max_angular_speed_rps": spin}


def timing_summary(rows: list[dict], dt: float) -> dict:
    durations = [row["wall_time_ns"]/1e6 for row in rows if row["status"] == "completed"]
    result = {"scope": "in_process_RuntimeSession_step_with_general_observation_and_retry_history",
              "sample_count": len(durations), "failed_call_count": len(rows)-len(durations),
              "steps_per_call": 1, "physics_dt_s": dt,
              "excluded": ["engine_startup", "reset", "trace_serialization_and_write", "image_rendering"],
              "statistic_scope": "one_run_descriptive_only_not_repeated_startup_benchmark"}
    if durations:
        ordered = sorted(durations)
        total_s = sum(durations)/1000
        result.update(p50_ms=statistics.median(durations), p95_ms=ordered[math.ceil(.95*len(ordered))-1],
                      min_ms=min(durations), max_ms=max(durations), total_step_call_wall_s=total_s,
                      measured_physics_steps_per_wall_s=len(durations)/total_s if total_s > 0 else None,
                      measured_simulated_s_per_wall_s=len(durations)*dt/total_s if total_s > 0 else None)
    return result


def hold_default_action(backend: Any) -> tuple[dict, dict]:
    """Translate the measured default pose into the supported drive-target API."""
    from scene_factory.joint_control import ARM_JOINT_NAMES

    names = list(backend._robot.dof_names)
    positions = [float(value) for value in backend._robot.get_joint_positions()]
    arm = [positions[names.index(name)] for name in ARM_JOINT_NAMES]
    config = backend._finger_gripper_config
    fractions = []
    for index, low, high in zip(config["indices"], config["closed_positions"],
                                config["open_positions"], strict=True):
        if high <= low:
            raise IncompleteEvidence("default gripper limits are not increasing")
        fractions.append((positions[index]-low)/(high-low))
    fraction = statistics.mean(fractions)
    if (not math.isfinite(fraction) or not -.0001 <= fraction <= 1.0001
            or max(fractions)-min(fractions) > .001):
        raise IncompleteEvidence("default finger pose cannot be represented by one supported opening fraction")
    # Only roundoff outside runtime limits is clamped; raw measured values stay recorded.
    fraction = min(1., max(0., fraction))
    return {"mode": "joint_position", "payload": {"arm_positions_rad": arm,
                                                   "gripper_open_fraction": fraction}}, {
        "source": "measured_post_reset_joint_positions", "joint_names": names,
        "joint_positions_rad_or_m": positions, "raw_finger_fractions": fractions,
        "finger_mapping": copy.deepcopy(config), "writes": "joint_drive_targets_only"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=5.)
    parser.add_argument("--physics-dt-s", type=float, default=1/120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physics-overrides", type=Path,
                        help="stage-local explicit geometry estimate JSON; original fixture/assets stay frozen")
    args = parser.parse_args()
    args.fixture, args.output = args.fixture.resolve(), args.output.resolve()
    if args.output == args.fixture or args.fixture in args.output.parents:
        parser.error("output must be outside the original fixture directory")
    if not math.isfinite(args.duration_s) or not 1 <= args.duration_s <= 60:
        parser.error("duration-s must be in [1,60]")
    if not math.isfinite(args.physics_dt_s) or not 1/480 <= args.physics_dt_s <= 1/120:
        parser.error("physics-dt-s must be in [1/480,1/120]")
    args.output.mkdir(parents=True, exist_ok=True)
    reserved = ("report.json", "trace.jsonl", "timings.jsonl", "frozen_inputs", "layout.json", "source_manifest.json")
    if any((args.output / name).exists() for name in reserved):
        parser.error("output already contains validation evidence; use a new directory")

    from scene_factory.dataset import write_json_atomic

    report: dict[str, Any] = {
        "schema_version": "scene_factory.isaac_room_background_validation.v1", "scope": SCOPE,
        "status": "failed", "background_acceptance": "not_verified", "assembly_acceptance": "not_evaluated",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "host": {"platform": platform.platform(), "python": sys.version},
        "transport": "in_process_RuntimeSession", "rendering": "disabled",
        "pose_writes_after_initialization": False, "cleanup": "not_started",
        "coverage_scope": "all_enabled_scene_collision_shapes_and_pair_summed_normal_forces",
        "unavailable": ["per_contact_point_vectors_in_saved_trace", "friction_force_vectors", "soft_body_contacts"],
    }
    backend = session = trace_stream = timing_stream = None
    initial = None
    window: deque = deque()
    timings: list[dict] = []
    hashes: dict = {}
    sources: dict = {}
    background_ids: list[str] = []
    dynamic_ids: set[str] = set()
    completed = validated = 0
    dt, requested = args.physics_dt_s, math.ceil(args.duration_s/args.physics_dt_s)
    report.update(requested_physics_steps=requested, requested_simulated_s=requested*dt)
    try:
        hashes, layout, source = freeze_fixture(args.fixture, args.output)
        sources = source_manifest()
        write_json_atomic(args.output / "source_manifest.json", sources)
        report.update(original_fixture_sha256=hashes, source_sha256=sources["sha256"],
                      fixture_id=source.get("fixture_id"), scene_id=layout.get("scene_id"))
        backgrounds = [row for row in layout["objects"] if row.get("category") == "background"]
        background_ids = [row["object_id"] for row in backgrounds]
        dynamic_ids = {row["object_id"] for row in backgrounds if row.get("dynamic")}
        scene_ids = {row["object_id"] for row in layout["objects"]}
        report["scene_scale"] = {"objects": len(scene_ids), "dynamic_objects": sum(bool(row.get("dynamic")) for row in layout["objects"]),
                                 "background_objects": len(background_ids), "dynamic_background_ids": sorted(dynamic_ids),
                                 "static_background_ids": sorted(set(background_ids)-dynamic_ids),
                                 "background_motion_scope": "dynamic_PhysX_raw_motion_static_USD_poses_and_zero_velocities"}
        if args.physics_overrides is not None:
            from scene_factory.robot_physics_overrides import normalize_physics_overrides

            override_bytes = args.physics_overrides.read_bytes()
            normalized = normalize_physics_overrides(json.loads(override_bytes))
            if normalized is None:
                raise ValueError("physics overrides must be an explicit configuration object")
            layout["task"]["physics_overrides"] = normalized
            report["physics_overrides_input_sha256"] = hashlib.sha256(override_bytes).hexdigest()
            (args.output / "physics_overrides_input.json").write_bytes(override_bytes)
            write_json_atomic(args.output / "physics_overrides.json", normalized)
        write_json_atomic(args.output / "layout.json", layout)
        report["runtime_layout_sha256"] = sha256_file(args.output / "layout.json")
        report["runtime_layout_change_scope"] = "optional_task.physics_overrides_only_original_spec_unchanged"
        from scene_factory.agent import SceneFactoryEnv
        from scene_factory.backends.isaac import IsaacSimBackend
        from scene_factory.session import RuntimeSession

        backend = IsaacSimBackend(args.fixture / "scene.usda", max_steps=requested+10,
                                  robot_asset_root=args.asset_root, physics_dt=dt,
                                  enable_general_observation=True)
        session = RuntimeSession(SceneFactoryEnv(args.output / "layout.json", backend), history_limit=requested+10)
        started = time.perf_counter_ns()
        response = session.reset({"seed": args.seed, "options": {}, "idempotency_key": "background-reset"})
        report["reset_to_initial_observation_wall_s"] = (time.perf_counter_ns()-started)/1e9
        action, default_pose = hold_default_action(backend)
        report.update(default_joint_hold=default_pose, hold_action=action, engine_version=backend.isaac_sim_version,
                      reset= {key: response.get(key) for key in ("episode_id", "step_index", "state_version", "sim_time", "info")})
        trace_stream = (args.output / "trace.jsonl").open("w", encoding="utf-8", newline="\n")
        timing_stream = (args.output / "timings.jsonl").open("w", encoding="utf-8", newline="\n")
        initial = make_sample(response, background_ids, kind="reset_initial", action=None, wall_time_ns=None)
        trace_stream.write(json.dumps(initial, allow_nan=False) + "\n")
        trace_stream.flush()
        validate_sample(initial, background_ids=background_ids, dynamic_ids=dynamic_ids, scene_ids=scene_ids, dt=dt)
        validated += 1
        window.append(initial)
        previous = initial
        for index in range(requested):
            request = {"episode_id": session.episode_id, "sequence_id": index,
                       "idempotency_key": f"hold-{index}", "steps": 1, "action": action}
            started = time.perf_counter_ns()
            try:
                response = session.step(request)
            except Exception:
                timing = {"sequence_id": index, "status": "failed", "wall_time_ns": time.perf_counter_ns()-started}
                timings.append(timing)
                timing_stream.write(json.dumps(timing, allow_nan=False)+"\n")
                timing_stream.flush()
                raise
            elapsed = time.perf_counter_ns()-started
            timing = {"sequence_id": index, "step_index": response.get("step_index"), "status": "completed", "wall_time_ns": elapsed}
            timings.append(timing)
            completed += 1
            timing_stream.write(json.dumps(timing, allow_nan=False)+"\n")
            timing_stream.flush()
            sample = make_sample(response, background_ids, kind="physics_step", action=action, wall_time_ns=elapsed)
            trace_stream.write(json.dumps(sample, allow_nan=False)+"\n")
            trace_stream.flush()
            validate_sample(sample, background_ids=background_ids, dynamic_ids=dynamic_ids, scene_ids=scene_ids,
                            dt=dt, previous=previous)
            if response.get("info", {}).get("executed_steps") != 1:
                raise IncompleteEvidence("one requested action did not execute exactly one physics step")
            validated += 1
            window.append(sample)
            while len(window) > 2 and sample["time_s"]-window[1]["time_s"] >= 1.-1e-8:
                window.popleft()
            previous = sample
            if response.get("terminated") or response.get("truncated"):
                raise IncompleteEvidence("episode ended unexpectedly during background-only hold")
        report["status"] = "completed"
    except Exception as exc:
        report.update(error={"type": type(exc).__name__, "message": str(exc)}, traceback=traceback.format_exc())
    finally:
        for stream in (trace_stream, timing_stream):
            if stream is not None:
                stream.close()
        report.update(completed_physics_steps=completed, validated_sample_count=validated,
                      step_call_timing=timing_summary(timings, dt))
        if backend is not None:
            report.update(runtime_summary=backend.runtime_summary,
                          observation_capabilities=backend.observation_capabilities)
        if initial is not None and window:
            report["background_stability"] = evaluate_background(initial, list(window), dynamic_ids, dt)
            write_json_atomic(args.output / "background_trace.json", list(window))
        try:
            after = {name: sha256_file(args.fixture / name) for name in hashes}
            report["original_fixture_unchanged"] = bool(hashes) and after == hashes
            report["source_unchanged"] = bool(sources) and source_manifest() == sources
        except Exception as exc:
            report.update(original_fixture_unchanged=False, source_unchanged=False,
                          lock_check_error=str(exc))
        stability = report.get("background_stability", {})
        full = (completed == requested and validated == requested+1 and not report.get("error")
                and report["original_fixture_unchanged"] and report["source_unchanged"])
        if full and stability.get("status") in {"passed", "failed"}:
            report["background_acceptance"] = stability["status"]
            report["status"] = stability["status"]
        report["acceptance_gate"] = {"requested_steps_completed": completed == requested,
                                     "all_initial_and_step_samples_valid": validated == requested+1,
                                     "fixture_and_source_unchanged": report["original_fixture_unchanged"] and report["source_unchanged"],
                                     "final_continuous_window": stability.get("window_s", 0) >= 1.-1e-8,
                                     "process_exit_is_task_acceptance": False}
        report["cleanup"] = "pending_process_shutdown" if session is not None else "not_started"
        # Bind every generated artifact before Kit's fast shutdown can exit Python.
        report["output_sha256"] = {path.relative_to(args.output).as_posix(): sha256_file(path)
                                   for path in sorted(args.output.rglob("*")) if path.is_file()
                                   and path.name not in {"report.json", "runtime.log"}}
        write_json_atomic(args.output / "report.json", report)
        digest = sha256_file(args.output / "report.json")
        (args.output / "report.sha256").write_text(digest+"  report.json\n", encoding="ascii")
        print(json.dumps({"report": str(args.output / "report.json"), "scope": SCOPE,
                          "background_acceptance": report["background_acceptance"], "report_sha256": digest}), flush=True)
        if session is not None:
            session.close()
    return 0 if report["background_acceptance"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
