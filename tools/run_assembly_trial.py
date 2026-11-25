"""Experimental physical peg/socket controller; every run retains its measured outcome.

Run with Isaac Python and a fixture created by build_assembly_fixture.py. Only
joint drive targets are written after initialization. This is a development
trial, not a claim that the full assembly benchmark has passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.agent import SceneFactoryEnv
from scene_factory.assembly import evaluate_assembly_trace
from scene_factory.backends.isaac import IsaacSimBackend
from scene_factory.dataset import write_json_atomic
from scene_factory.joint_control import ARM_JOINT_NAMES
from scene_factory.runtime_worker import _live_rigid_state, _new_rigid_view
from scene_factory.session import RuntimeSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--arm-stiffness", type=float, default=1000.0)
    parser.add_argument("--arm-damping", type=float, default=200.0)
    parser.add_argument("--grip-fraction", type=float, default=0.22)
    parser.add_argument("--grasp-height-m", type=float, default=0.995)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise FileExistsError("use a new output directory to preserve trial evidence")
    spec = json.loads((args.fixture / "assembly_spec.json").read_text(encoding="utf-8"))
    source = json.loads((args.fixture / "SOURCE.json").read_text(encoding="utf-8"))
    report = {"schema_version": "scene_factory.assembly_trial.v1", "status": "failed",
              "scope": "experimental_reference_controller", "pose_writes_after_initialization": False,
              "fixture_sha256": hashlib.sha256((args.fixture / "scene.usda").read_bytes()).hexdigest(),
              "spec_sha256": hashlib.sha256((args.fixture / "assembly_spec.json").read_bytes()).hexdigest(),
              "lateral_offset_m": args.lateral_offset_m, "phases": [],
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "controller_parameters": {"arm_stiffness": args.arm_stiffness, "arm_damping": args.arm_damping,
                                        "grip_fraction": args.grip_fraction, "grasp_height_m": args.grasp_height_m},
              "force_measurement": "PhysX tensor normal contact force resultants, N, one physics dt; sums magnitudes per collider",
              "unmeasured_contacts": "contacts outside peg/socket, peg/table and peg/fingers are not instrumented"}
    samples = []
    session = None
    trace_file = None
    try:
        backend = IsaacSimBackend(args.fixture / "scene.usda", max_steps=3500,
                                  robot_asset_root=args.asset_root, physics_dt=1/120)
        layout = json.loads((args.fixture / "layout.json").read_text(encoding="utf-8"))
        layout["task"]["controller"] = {"arm_stiffness": args.arm_stiffness, "arm_damping": args.arm_damping}
        write_json_atomic(args.output / "layout.json", layout)
        write_json_atomic(args.output / "controller_source.json", {"python_source": Path(__file__).read_text(encoding="utf-8")})
        session = RuntimeSession(SceneFactoryEnv(args.output / "layout.json", backend), history_limit=4096)
        session.reset({"seed": 0, "options": {}, "idempotency_key": "trial-reset"})
        import numpy as np
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.api.sensors import RigidContactView

        peg_view = _new_rigid_view(RigidPrim, "/World/Objects/peg_1", "trial_peg")
        peg_view.initialize(backend._simulation.physics_sim_view)
        filter_paths = ["/World/Objects/socket_1/Base", "/World/Objects/socket_1/Xminus1",
                        "/World/Objects/socket_1/X1", "/World/Objects/socket_1/Yminus1",
                        "/World/Objects/socket_1/Y1", "/World/Objects/table_1"]
        contacts_view = RigidContactView(prim_paths_expr="/World/Objects/peg_1",
                                         filter_paths_expr=filter_paths, name="assembly_contacts",
                                         max_contact_count=128)
        contacts_view.initialize(backend._simulation.physics_sim_view)
        names = list(backend._robot.dof_names)
        indices = [names.index(name) for name in ARM_JOINT_NAMES]
        arm = np.asarray(backend._robot.get_joint_positions())[indices].astype(float)
        socket = next(item for item in backend.scene["objects"] if item["object_id"] == "socket_1")
        socket_pose = {"position_m": socket["pose"]["position"], "orientation_wxyz": [1, 0, 0, 0]}
        released = False
        phase = "initialized"
        trace_file = (args.output / "trace.jsonl").open("w", encoding="utf-8")

        def record(action=None):
            live = _live_rigid_state(peg_view, "peg_1")
            matrix = np.asarray(contacts_view.get_contact_force_matrix(dt=backend.physics_dt), dtype=float)
            if matrix.shape != (1, len(filter_paths), 3) or not np.isfinite(matrix).all():
                raise RuntimeError(f"invalid assembly contact matrix {matrix.shape}")
            magnitudes = np.linalg.norm(matrix[0], axis=1)
            contacts = []
            for other, magnitude in [("socket_1", float(magnitudes[:5].sum())), ("table_1", float(magnitudes[5]))]:
                if magnitude > 1e-6:
                    contacts.append({"pair": ["peg_1", other], "normal_force_N": magnitude})
            if not backend._contact_force_read_valid:
                raise RuntimeError("finger contact evidence unavailable")
            grip_force = sum(pair["force_magnitude"] for pair in backend._contact_force_pairs.values())
            if grip_force > 1e-6:
                contacts.append({"pair": ["peg_1", "robot"], "normal_force_N": grip_force})
            sample = {"time_s": backend.steps * backend.physics_dt, "phase": phase,
                      "geometry_clearance_m": source["geometry"]["clearance_m"],
                      "parts": {"peg_1": {"position_m": live["position"], "orientation_wxyz": live["orientation_wxyz"]}, "socket_1": socket_pose},
                      "contacts": contacts, "released": released,
                      "motion_source": "robot_joint_control" if action else "physics_passive",
                      "action": action, "teleported": False, "kinematic_pose_write": False,
                      "live_peg": live, "end_effector_position_m": list(backend._end_effector_pose()[0])}
            samples.append(sample)
            trace_file.write(json.dumps(sample, allow_nan=False) + "\n")
            trace_file.flush()
            return live

        def command(target, grip):
            nonlocal arm
            payload = {"arm_positions_rad": [float(v) for v in target], "gripper_open_fraction": float(grip)}
            action = {"mode": "joint_position", "payload": payload}
            session.step({"episode_id": session.episode_id, "sequence_id": backend.steps,
                          "idempotency_key": f"step-{backend.steps}", "steps": 1, "action": action})
            arm = np.asarray(target).copy()
            live = record(action)
            if any(contact["normal_force_N"] > spec["max_force_N"] for contact in samples[-1]["contacts"]):
                raise RuntimeError("measured contact force exceeded the frozen limit; controller stopped")
            return live

        def ik(goal):
            base_pos, base_quat = backend._robot.get_world_pose()
            backend._kinematics.get_kinematics_solver().set_robot_base_pose(base_pos, base_quat)
            target, ok = backend._kinematics.compute_inverse_kinematics(np.asarray(goal), np.asarray([0., 1., 0., 0.]), position_tolerance=0.0001, orientation_tolerance=0.002)
            if not ok:
                raise RuntimeError(f"IK failed in {phase} for {goal}")
            result = np.asarray(backend._robot.get_joint_positions(), dtype=float)
            target_indices = target.joint_indices if target.joint_indices is not None else range(len(target.joint_positions))
            for index, value in zip(target_indices, target.joint_positions, strict=True):
                if value is not None:
                    result[index] = value
            return result[indices]

        def begin(name):
            nonlocal phase
            phase = name
            live = _live_rigid_state(peg_view, "peg_1")
            report["phases"].append({"phase": phase, "step": backend.steps, "peg": live,
                                     "end_effector": list(backend._end_effector_pose()[0]),
                                     "robot": backend._last_observation.get("robot", {})})
            write_json_atomic(args.output / "progress.json", report)
            print(json.dumps({"phase": name, "step": backend.steps, "peg": live["position"]}), flush=True)

        def move(goal, count, grip):
            start = np.asarray(backend._end_effector_pose()[0], dtype=float)
            for tick in range(count):
                point = start + (np.asarray(goal) - start) * ((tick + 1)/count)
                command(ik(point), grip)

        record()
        begin("approach")
        target = ik([.55, -.13, 1.15])
        start_arm = arm.copy()
        for tick in range(360):
            command(start_arm + (target - start_arm) * ((tick + 1)/360), 1)
        begin("descend_to_grasp")
        move([.55, -.13, args.grasp_height_m], 300, 1)
        begin("close_gripper")
        for tick in range(180):
            command(arm, 1 - (1 - args.grip_fraction) * min(1., (tick+1)/120))
        begin("lift")
        move([.55, -.13, 1.14], 300, args.grip_fraction)
        live = _live_rigid_state(peg_view, "peg_1")
        if live["position"][2] < 1.06:
            raise RuntimeError("physical grasp/lift failed; insertion was not attempted")
        # Correct for measured grasp offset using targets only, never part pose writes.
        ee = np.asarray(backend._end_effector_pose()[0])
        offset = ee - np.asarray(live["position"])
        report["measured_grasp_offset_m"] = offset.tolist()
        begin("align_socket")
        peg_above = np.asarray([.60 + args.lateral_offset_m, .13, 1.09])
        move(peg_above + offset, 420, args.grip_fraction)
        begin("insert")
        move(np.asarray([.60 + args.lateral_offset_m, .13, .990]) + offset, 720, args.grip_fraction)
        begin("hold")
        for _ in range(90):
            command(arm, args.grip_fraction)
        begin("release")
        for tick in range(120):
            command(arm, args.grip_fraction + (1 - args.grip_fraction) * ((tick+1)/120))
        released = True
        begin("withdraw")
        move(np.asarray(backend._end_effector_pose()[0]) + [0, 0, .12], 240, 1)
        begin("stability")
        for _ in range(120):
            command(arm, 1)
        report["status"] = "trial_completed"
        report["engine_version"] = backend.isaac_sim_version
        trace_file.close()
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    finally:
        if trace_file is not None:
            trace_file.close()
        report["assembly_evaluation"] = evaluate_assembly_trace(spec, samples)
        # Incomplete general-contact coverage cannot certify assembly success.
        report["assembly_acceptance"] = "not_verified" if report["assembly_evaluation"]["status"] == "passed" else "failed"
        report["sample_count"] = len(samples)
        report["cleanup"] = "pending_process_shutdown"
        write_json_atomic(args.output / "report.json", report)
        if session is not None:
            session.close()
    return 0 if report["assembly_acceptance"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
