"""Measured Cartesian feedback controller for the original physical assembly fixture.

Run with Isaac Python. After initialization this tool writes joint drive targets
only. Each trial preserves its source, frozen acceptance specification and full
PhysX measurements; neither a process exit nor a final pose is task acceptance.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.agent import SceneFactoryEnv
from scene_factory.assembly import evaluate_assembly_trace
from scene_factory.backends.isaac import IsaacSimBackend
from scene_factory.dataset import write_json_atomic
from scene_factory.grasp_control import ContactGraspController, GraspControlConfig, anchored_joint_target
from scene_factory.grasp_diagnostics import summarize_grasp_phases
from scene_factory.joint_control import ARM_JOINT_NAMES
from scene_factory.runtime_worker import _live_rigid_state, _new_rigid_view
from scene_factory.session import RuntimeSession


class OnlineAssemblyTerminal(RuntimeError):
    """A terminal API response ends the controller before its next drive request."""

    def __init__(self, response):
        self.response = response
        self.success = bool(response.get("terminated") and not response.get("truncated")
                            and response.get("observation", {}).get("task_success"))
        super().__init__("online_task_success" if self.success else "online_task_ended_without_success")


class GraspPrefixCompleted(RuntimeError):
    """Stop a requested grasp diagnostic without claiming assembly success."""


def check_online_terminal(response):
    if response.get("terminated") or response.get("truncated"):
        raise OnlineAssemblyTerminal(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--arm-stiffness", type=float, default=50000)
    parser.add_argument("--arm-damping", type=float, default=1500)
    parser.add_argument("--grip-fraction", type=float, default=.15)
    parser.add_argument("--prelift-grip-fraction", type=float,
                        help="light initial grasp, then lift 1 mm before slowly applying final grip")
    parser.add_argument("--grasp-control", choices=("contact", "timed"), default="contact",
                        help="contact gates are the default; timed preserves the historical controller")
    parser.add_argument("--grasp-config", type=Path,
                        help="JSON overrides for measured grasp control parameters")
    parser.add_argument("--grasp-only", action="store_true",
                        help="stop after measured grasp/clearance/stability; not full assembly acceptance")
    parser.add_argument("--grasp-contact-details", action="store_true",
                        help="record raw contact points and friction patches during measured grasp")
    parser.add_argument("--lateral-offset-m", type=float, default=0)
    parser.add_argument("--grasp-height-m", type=float, default=.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seat-center-z-m", type=float, default=.9894)
    parser.add_argument("--seat-duration-s", type=float, default=3.)
    parser.add_argument("--release-pause-s", type=float, default=1.)
    parser.add_argument("--freeze-seated-pose", action="store_true")
    parser.add_argument("--lift-duration-s", type=float, default=8.)
    parser.add_argument("--close-duration-s", type=float, default=4.)
    parser.add_argument("--solver-position-iterations", type=int)
    parser.add_argument("--solver-velocity-iterations", type=int)
    parser.add_argument("--physics-dt-s", type=float, default=1/120)
    parser.add_argument("--physics-overrides", type=Path,
                        help="explicit stage-local geometry estimate configuration; never edits the asset cache")
    parser.add_argument("--finger-stiffness", type=float, default=1000.)
    parser.add_argument("--finger-damping", type=float, default=200.)
    parser.add_argument("--finger-max-force", type=float, default=2000.)
    parser.add_argument("--online", action="store_true",
                        help="require full-stage measurements and the API online assembly result")
    args = parser.parse_args()
    if args.grasp_control == "contact" and args.prelift_grip_fraction is not None:
        parser.error("--prelift-grip-fraction is only used by --grasp-control timed")
    if args.grasp_control == "timed" and (args.grasp_config or args.grasp_only):
        parser.error("--grasp-config and --grasp-only require contact control")
    if args.grasp_contact_details and args.grasp_control != "contact":
        parser.error("--grasp-contact-details requires contact control")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise FileExistsError("trial output already exists")
    spec_path = args.fixture / "assembly_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source = json.loads((args.fixture / "SOURCE.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "scene_factory.precision_assembly_trial.v1",
        "status": "failed", "assembly_acceptance": "not_verified", "phases": [],
        "pose_writes_after_initialization": False,
        "fixture_sha256": hashlib.sha256((args.fixture / "scene.usda").read_bytes()).hexdigest(),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256((Path(__file__).resolve().parents[1]/"scene_factory/assembly.py").read_bytes()).hexdigest(),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "force_measurement": "sum of absolute PhysX normal contact-point forces per body pair; N; one physics dt",
        "phases_completed": False,
        "online_mode": args.online,
        "transport": "in_process_RuntimeSession",
    }
    samples = []
    session = None
    backend = None
    trace_file = None
    background_views = {}
    background_initial = {}
    background_window = deque()
    grasp_controller = None
    grasp_trace_file = None
    control_decision = None
    control_decisions = []
    try:
        import numpy as np

        dt = args.physics_dt_s
        if not math.isfinite(dt) or not 1/480 <= dt <= 1/120:
            raise ValueError("physics-dt-s must be between 1/480 and 1/120 seconds")
        def ticks(duration):
            return math.ceil(duration/dt)
        maximum_steps = ticks(100.)
        if args.grasp_control == "contact":
            config_values = json.loads(args.grasp_config.read_text(encoding="utf-8")) if args.grasp_config else {}
            grasp_config = GraspControlConfig(**config_values)
            if grasp_config.stop_angular_speed_rps >= spec["max_angular_speed_rps"]:
                raise ValueError("grasp protection must be below the frozen angular-speed limit")
            if 2*grasp_config.max_finger_force_N >= spec["max_force_N"]:
                raise ValueError("grasp force protection must be below the frozen robot-pair force limit")
            report["grasp_control_config"] = asdict(grasp_config)
            control_path = Path(__file__).resolve().parents[1]/"scene_factory/grasp_control.py"
            report["grasp_controller_sha256"] = hashlib.sha256(control_path.read_bytes()).hexdigest()
            (args.output / "grasp_controller_source.py").write_bytes(control_path.read_bytes())
            write_json_atomic(args.output / "grasp_control_config.json", asdict(grasp_config))
        backend = IsaacSimBackend(args.fixture / "scene.usda", max_steps=maximum_steps,
                                  robot_asset_root=args.asset_root, physics_dt=dt,
                                  enable_general_observation=args.online)
        layout = json.loads((args.fixture / "layout.json").read_text(encoding="utf-8"))
        if args.physics_overrides is not None:
            from scene_factory.robot_physics_overrides import normalize_physics_overrides

            override_bytes = args.physics_overrides.read_bytes()
            layout["task"]["physics_overrides"] = normalize_physics_overrides(json.loads(override_bytes))
            report["physics_overrides_sha256"] = hashlib.sha256(override_bytes).hexdigest()
            write_json_atomic(args.output / "physics_overrides.json", layout["task"]["physics_overrides"])
        layout["task"]["controller"] = {"arm_stiffness": args.arm_stiffness,
                                         "arm_damping": args.arm_damping,
                                         "finger_stiffness": args.finger_stiffness,
                                         "finger_damping": args.finger_damping,
                                         "finger_max_force": args.finger_max_force}
        if args.solver_position_iterations is not None:
            layout["task"]["controller"]["solver_position_iterations"] = args.solver_position_iterations
        if args.solver_velocity_iterations is not None:
            layout["task"]["controller"]["solver_velocity_iterations"] = args.solver_velocity_iterations
        if args.online:
            layout["task"]["assembly_spec"] = spec
            layout["task"]["geometry_clearance_m"] = source["geometry"]["clearance_m"]
        write_json_atomic(args.output / "layout.json", layout)
        write_json_atomic(args.output / "spec.json", spec)
        write_json_atomic(args.output / "controller_source.json", {"python_source": Path(__file__).read_text(encoding="utf-8")})
        session = RuntimeSession(SceneFactoryEnv(args.output / "layout.json", backend), history_limit=maximum_steps)
        session.reset({"seed": args.seed, "options": {}, "idempotency_key": "precision-reset"})
        report["solver_runtime"] = {
            "robot_position_iterations": int(backend._robot.get_solver_position_iteration_count()),
            "robot_velocity_iterations": int(backend._robot.get_solver_velocity_iteration_count()),
            "configured_before_reset": True,
            "runtime_settings": backend.runtime_summary.get("runtime_solver_settings"),
        }
        report["runtime_drive_gains"] = backend.runtime_summary.get("runtime_drive_gains")
        report["controller_settings"] = backend.runtime_summary.get("controller_settings")
        report["physics_overrides"] = backend.runtime_summary.get("physics_overrides")
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.api.sensors import RigidContactView
        from pxr import Usd, UsdPhysics

        peg_view = _new_rigid_view(RigidPrim, "/World/Objects/peg_1", "precision_peg")
        peg_view.initialize(backend._simulation.physics_sim_view)
        actual_mass = np.asarray(peg_view.get_masses(), dtype=float).reshape(-1)
        actual_inertia = np.asarray(peg_view.get_inertias(), dtype=float).reshape(-1, 3, 3)
        expected_mass = float(source.get("physical_parameters", {}).get("mass_kg", .08))
        expected_diagonal = source.get("physical_parameters", {}).get("inertia_kg_m2")
        if expected_diagonal is None:
            radius = source["geometry"]["peg_radius_m"]
            length = source["geometry"]["peg_length_m"]
            expected_diagonal = [expected_mass*(3*radius**2+length**2)/12]*2+[expected_mass*radius**2/2]
        expected_inertia = np.diag(expected_diagonal)
        report["peg_runtime_physics"] = {
            "source": "live_PhysX_tensor_API", "mass_kg": actual_mass.tolist(),
            "inertia_kg_m2": actual_inertia.tolist(), "expected_mass_kg": expected_mass,
            "expected_inertia_kg_m2": expected_inertia.tolist(),
            "verified": bool(actual_mass.shape == (1,) and actual_inertia.shape == (1, 3, 3)
                             and np.allclose(actual_mass, expected_mass, rtol=1e-5, atol=1e-9)
                             and np.allclose(actual_inertia[0], expected_inertia, rtol=1e-5, atol=1e-9)),
        }
        if not report["peg_runtime_physics"]["verified"]:
            raise RuntimeError("runtime_peg_mass_or_inertia_mismatch")
        background_window = deque(maxlen=ticks(1.)+1)
        for item in layout["objects"]:
            if item.get("dynamic") and item["object_id"] != "peg_1":
                object_id = item["object_id"]
                view = _new_rigid_view(RigidPrim, f"/World/Objects/{object_id}", f"background_{object_id}")
                view.initialize(backend._simulation.physics_sim_view)
                background_views[object_id] = view
                background_initial[object_id] = _live_rigid_state(view, object_id)
        report["scene_scale"] = {"objects": len(layout["objects"]),
                                 "dynamic_objects": sum(bool(item.get("dynamic")) for item in layout["objects"]),
                                 "background_initial": background_initial}
        filter_paths = []
        for prim in Usd.PrimRange.Stage(backend._stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not path.startswith("/World/Objects/peg_1"):
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if enabled is not False:
                    filter_paths.append(path)
        filter_paths.sort()
        if not filter_paths:
            raise RuntimeError("no collision shapes available for contact coverage")
        contact_view = RigidContactView(prim_paths_expr="/World/Objects/peg_1",
                                        filter_paths_expr=filter_paths, name="precision_all_contacts",
                                        prepare_contact_sensors=False, disable_stablization=False,
                                        max_contact_count=2048)
        contact_view.initialize(backend._simulation.physics_sim_view)
        report["contact_coverage"] = {"scope": "all enabled authored collision shapes against peg",
                                      "filter_paths": filter_paths, "max_contact_count": 2048}
        if args.grasp_contact_details:
            report["grasp_contact_details"] = {
                "file": "trace.jsonl", "scope": "close_gripper through grasp_stable",
                "source": "RigidContactView raw PhysX contact and friction buffers",
                "units": "positions and separation in m; forces in N using one physics dt",
                "normal_force_sign": "signed as returned by PhysX; acceptance uses absolute sums",
            }
        grasp_coverage = all(any(token in path.split("/") for path in filter_paths)
                             for token in ("panda_leftfinger", "panda_rightfinger", "table_1"))
        if args.grasp_control == "contact" and not grasp_coverage:
            raise RuntimeError("grasp finger/table contact filters are incomplete")
        names = list(backend._robot.dof_names)
        indices = [names.index(name) for name in ARM_JOINT_NAMES]
        arm = np.asarray(backend._robot.get_joint_positions(), dtype=float)[indices]
        solver = backend._kinematics.get_kinematics_solver()
        frame = backend._kinematics.get_end_effector_frame()
        base_pos, base_quat = backend._robot.get_world_pose()
        solver.set_robot_base_pose(base_pos, base_quat)
        socket = next(item for item in layout["objects"] if item["object_id"] == "socket_1")
        socket_pose = {"position_m": socket["pose"]["position"], "orientation_wxyz": [1, 0, 0, 0]}
        phase, released = "initialized", False
        grasp_orientation = None
        orientation_bias = np.array([1., 0, 0, 0])
        trace_file = (args.output / "trace.jsonl").open("w", encoding="utf-8")

        def body_id(path):
            if path.startswith("/World/Robot/"):
                return "robot"
            if path.startswith("/World/Objects/"):
                return path.split("/")[3]
            return path

        def record(action=None):
            live = _live_rigid_state(peg_view, "peg_1")
            data = contact_view.get_contact_force_data(dt=dt)
            if data is None:
                raise RuntimeError("contact point measurements unavailable")
            forces, points, normals, separations, counts, starts = (np.asarray(value) for value in data)
            if counts.shape != (1, len(filter_paths)) or not np.isfinite(forces).all():
                raise RuntimeError("contact buffers invalid")
            if int(counts.sum()) >= 2048:
                raise RuntimeError("contact buffer saturated; cannot certify force coverage")
            details = None
            if args.grasp_contact_details and phase in {
                "close_gripper", "light_grasp_clear_table", "increase_grip_off_table", "grasp_stable",
            }:
                friction = contact_view.get_friction_data(dt=dt)
                if friction is None:
                    raise RuntimeError("friction patch measurements unavailable")
                patch_forces, patch_points, patch_counts, patch_starts = (np.asarray(value) for value in friction)
                if patch_counts.shape != counts.shape or int(patch_counts.sum()) >= 2048:
                    raise RuntimeError("friction patch buffer invalid or saturated")
                details = []
            pairs = {}
            shape_contacts = []
            for index, path in enumerate(filter_paths):
                count, start = int(counts[0, index]), int(starts[0, index])
                if details is not None:
                    patch_count = int(patch_counts[0, index])
                    patch_start = int(patch_starts[0, index])
                    if count or patch_count:
                        point_slice = slice(start, start + count) if count else slice(0, 0)
                        patch_slice = slice(patch_start, patch_start + patch_count) if patch_count else slice(0, 0)
                        details.append({"collider": path,
                                        "point_positions_m": points[point_slice].tolist(),
                                        "point_normals_world": normals[point_slice].tolist(),
                                        "point_separations_m": separations[point_slice].reshape(-1).tolist(),
                                        "point_normal_forces_N": forces[point_slice].reshape(-1).tolist(),
                                        "friction_points_m": patch_points[patch_slice].tolist(),
                                        "friction_forces_world_N": patch_forces[patch_slice].tolist()})
                if count:
                    force = float(np.abs(forces[start:start + count]).sum())
                    if force > 1e-7:
                        shape_contacts.append({"collider": path, "normal_force_N": force})
                        other = body_id(path)
                        pairs[other] = pairs.get(other, 0.) + force
            contacts = [{"pair": ["peg_1", other], "normal_force_N": force}
                        for other, force in sorted(pairs.items())]
            sample = {"time_s": backend.steps * dt, "phase": phase,
                      "geometry_clearance_m": source["geometry"]["clearance_m"],
                      "parts": {"peg_1": {"position_m": live["position"], "orientation_wxyz": live["orientation_wxyz"]},
                                "socket_1": socket_pose},
                      "contacts": contacts, "released": released,
                      "motion_source": "robot_joint_control" if action else "physics_passive",
                      "action": action, "teleported": False, "kinematic_pose_write": False,
                      "live_peg": live, "end_effector_position_m": list(backend._end_effector_pose()[0]),
                      "end_effector_orientation_wxyz": list(backend._end_effector_pose()[1]),
                      "shape_contacts": shape_contacts,
                      "grasp_contact_coverage_complete": grasp_coverage,
                      "grasp_control": control_decision,
                      "finger_positions_m": [float(backend._robot.get_joint_positions()[index])
                                               for index in backend._finger_gripper_config["indices"]]}
            if details is not None:
                sample["grasp_contact_details"] = details
            samples.append(sample)
            if background_views:
                background_window.append({"time_s": sample["time_s"], "objects": {
                    key: _live_rigid_state(view, key) for key, view in background_views.items()}})
            trace_file.write(json.dumps(sample, allow_nan=False) + "\n")
            trace_file.flush()
            return live

        def command(target, grip):
            nonlocal arm
            # A bounded setpoint change prevents IK branch changes from becoming
            # discontinuous drive commands. This does not write actual joints.
            target = arm + np.clip(np.asarray(target) - arm, -.48*dt, .48*dt)
            payload = {"arm_positions_rad": target.tolist(), "gripper_open_fraction": float(grip)}
            action = {"mode": "joint_position", "payload": payload}
            response = session.step({"episode_id": session.episode_id, "sequence_id": backend.steps,
                                     "idempotency_key": f"step-{backend.steps}", "steps": 1, "action": action})
            arm = target
            live = record(action)
            if args.online:
                check_online_terminal(response)
            if any(c["normal_force_N"] > spec["max_force_N"] for c in samples[-1]["contacts"]):
                raise RuntimeError("force_limit_exceeded; controller stopped")
            if samples[-1]["time_s"] > spec["timeout_s"]:
                raise RuntimeError("task timeout")
            return live

        def multiply(a, b):
            w, x, y, z = a
            v, i, j, k = b
            return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                             w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])

        def normalize(q):
            return q / np.linalg.norm(q)

        def ik(goal, orientation, *, warm_start=None, position_tolerance=.0002):
            target, ok = solver.compute_inverse_kinematics(frame, np.asarray(goal), np.asarray(orientation),
                                                            warm_start=arm if warm_start is None else warm_start,
                                                            position_tolerance=position_tolerance,
                                                            orientation_tolerance=.002)
            if not ok:
                raise RuntimeError(f"IK failed in {phase}: {list(goal)}")
            return np.asarray(target)

        def begin(name):
            nonlocal phase
            phase = name
            live = _live_rigid_state(peg_view, "peg_1")
            report["phases"].append({"phase": name, "step": backend.steps, "peg": live,
                                     "end_effector": list(backend._end_effector_pose()[0])})
            write_json_atomic(args.output / "progress.json", report)
            print(json.dumps({"phase": name, "step": backend.steps, "peg": live["position"]}), flush=True)

        def move_ee(goal, duration, grip, orientation=None, hold=False):
            start = np.asarray(backend._end_effector_pose()[0])
            integral = np.zeros(3)
            orientation = np.array([0., 1., 0., 0.]) if orientation is None else orientation
            count = math.ceil(duration/dt)
            for tick in range(count):
                u = 1. if hold else (tick+1)/count
                u = u*u*(3-2*u)
                desired = start + (np.asarray(goal) - start)*u
                current = np.asarray(backend._end_effector_pose()[0])
                integral = np.clip(integral + (desired-current)*dt*2., -.015, .015)
                command(ik(desired+integral, orientation), grip)

        def move_peg(goal, duration, grip, hold=False):
            nonlocal grasp_orientation, orientation_bias
            live = _live_rigid_state(peg_view, "peg_1")
            start = np.asarray(live["position"])
            start_ee = np.asarray(backend._end_effector_pose()[0])
            integral = np.zeros(3)
            count = math.ceil(duration/dt)
            for tick in range(count):
                live = _live_rigid_state(peg_view, "peg_1")
                current = np.asarray(live["position"])
                u = 1. if hold else (tick+1)/count
                u = u*u*(3-2*u)
                desired = start+(np.asarray(goal)-start)*u
                ee, ee_q = backend._end_effector_pose()
                peg_q = normalize(np.asarray(live["orientation_wxyz"]))
                if peg_q[0] < 0:
                    peg_q = -peg_q
                tip = multiply(multiply(peg_q, np.array([0., 0, 0, -.05])),
                               peg_q*np.array([1, -1, -1, -1]))[1:]+current
                error = desired-current
                # Center tracking alone can place the lower tip against a rim
                # when the grasp tilts. Keep the actual insertion tip on axis.
                error[:2] = desired[:2]-tip[:2]
                integral = np.clip(integral+error*dt*.8, -.02, .02)
                # Measure grasp offset every step instead of assuming a rigid
                # attachment. Correct orientation through robot targets as well.
                inv_peg = peg_q*np.array([1, -1, -1, -1])
                if grasp_orientation is None:
                    grasp_orientation = normalize(multiply(inv_peg, np.asarray(ee_q)))
                orientation_gain = 1-math.exp(-2.4*dt)
                correction = normalize((1-orientation_gain)*np.array([1., 0, 0, 0])+orientation_gain*inv_peg)
                orientation_bias = normalize(multiply(correction, orientation_bias))
                angle = 2*math.acos(min(1., abs(float(orientation_bias[0]))))
                if angle > .12:
                    orientation_bias = normalize(np.r_[math.cos(.06), orientation_bias[1:]/np.linalg.norm(orientation_bias[1:])*math.sin(.06)])
                target_q = normalize(multiply(orientation_bias, grasp_orientation))
                # The feed-forward reference is fixed at segment start. Adding
                # an integral repeatedly to the current measured TCP turns a
                # position controller into unintended accelerating motion.
                goal_ee = start_ee+(desired-start)+error*.35+integral
                command(ik(goal_ee, target_q), grip)

        record()
        begin("approach")
        target = ik([.55, -.13, 1.15], [0., 1., 0., 0.])
        start_arm = arm.copy()
        for tick in range(ticks(5.)):
            u = (tick+1)/ticks(5.)
            command(start_arm+(target-start_arm)*(u*u*(3-2*u)), 1.)
        move_ee([.55, -.13, 1.15], 1., 1., hold=True)
        begin("descend_to_grasp")
        move_ee([.55, -.13, args.grasp_height_m], 4., 1.)
        move_ee([.55, -.13, args.grasp_height_m], .75, 1., hold=True)
        begin("close_gripper")
        grip = args.grip_fraction
        if args.grasp_control == "contact":
            finger_config = backend._finger_gripper_config
            geometry = source["geometry"]
            grasp_width = geometry.get("peg_width_m")
            if grasp_width is None:
                grasp_width = 2*geometry["peg_radius_m"]
            report["grasp_object_width_m"] = grasp_width
            report["grasp_gripper_mapping"] = {
                key: list(finger_config[key]) for key in ("closed_positions", "open_positions")}
            grasp_controller = ContactGraspController(
                physics_dt_s=dt, minimum_fraction=args.grip_fraction, config=grasp_config,
                object_width_m=grasp_width,
                closed_positions=list(finger_config["closed_positions"]),
                open_positions=list(finger_config["open_positions"]))
            grasp_trace_file = (args.output / "grasp_control_trace.jsonl").open("w", encoding="utf-8")
            micro_start = micro_orientation = micro_ik_origin = None
            micro_arm = arm.copy()
            while True:
                control_decision = grasp_controller.update(samples[-1])
                control_decision["input_time_s"] = samples[-1]["time_s"]
                control_decisions.append(control_decision)
                grasp_trace_file.write(json.dumps(control_decision, allow_nan=False)+"\n")
                grasp_trace_file.flush()
                report["grasp_control_final"] = control_decision
                if control_decision["status"] == "failed":
                    raise RuntimeError(control_decision["reason"])
                grip = control_decision["gripper_open_fraction"]
                if phase != control_decision["phase"]:
                    begin(control_decision["phase"])
                if control_decision["status"] == "ready":
                    report["grasp_prefix_acceptance"] = "passed"
                    report["grasp_prefix_scope"] = "measured_bilateral_grasp_table_clearance_and_stability_only"
                    if args.grasp_only:
                        raise GraspPrefixCompleted()
                    break
                if control_decision["phase"] == "close_gripper":
                    target = arm
                else:
                    if micro_start is None:
                        micro_start, micro_orientation = backend._end_effector_pose()
                        micro_start, micro_orientation = np.asarray(micro_start), np.asarray(micro_orientation)
                        micro_arm = arm.copy()
                        micro_ik_origin = ik(micro_start, micro_orientation,
                                             warm_start=micro_arm, position_tolerance=1e-6)
                        report["micro_lift_reference"] = {
                            "tcp_position_m": micro_start.tolist(),
                            "tcp_orientation_wxyz": micro_orientation.tolist(),
                            "drive_origin_rad": micro_arm.tolist(),
                            "ik_origin_rad": micro_ik_origin.tolist(),
                            "method": "fixed_seed_IK_displacement_plus_existing_drive_target"}
                    # Absolute TCP displacement from the measured start; no
                    # integration of moving measurements and no body pose writes.
                    offset = control_decision["lift_offset_m"]
                    if offset:
                        ik_target = ik(micro_start+[0, 0, offset], micro_orientation,
                                       warm_start=micro_arm, position_tolerance=1e-6)
                        target = anchored_joint_target(micro_arm.tolist(), micro_ik_origin.tolist(), ik_target.tolist())
                    else:
                        target = micro_arm
                command(target, grip)
            control_decision = None
        else:
            initial_grip = args.grip_fraction if args.prelift_grip_fraction is None else args.prelift_grip_fraction
            if not 0 <= args.grip_fraction <= initial_grip <= 1:
                raise ValueError("grip fractions must satisfy 0 <= final <= initial <= 1")
            close_ticks = math.ceil(args.close_duration_s/dt)
            for tick in range(close_ticks):
                u = min(1., (tick+1)/(close_ticks*.8))
                u = u*u*(3-2*u)
                command(arm, 1-(1-initial_grip)*u)
            if args.prelift_grip_fraction is not None:
                begin("light_grasp_clear_table")
                micro_start, micro_orientation = backend._end_effector_pose()
                micro_target = ik(np.asarray(micro_start)+[0, 0, .001], np.asarray(micro_orientation))
                micro_start_arm = arm.copy()
                for tick in range(ticks(2.)):
                    u = (tick+1)/ticks(2.)
                    command(micro_start_arm+(micro_target-micro_start_arm)*(u*u*(3-2*u)), initial_grip)
                begin("increase_grip_off_table")
                for tick in range(ticks(2.)):
                    u = (tick+1)/ticks(2.)
                    command(arm, initial_grip+(args.grip_fraction-initial_grip)*(u*u*(3-2*u)))
        begin("lift")
        # Clear the table using an absolute TCP path. A relative peg feedback
        # target can follow arm sag and push down before its integral responds.
        lift_start, lift_orientation = backend._end_effector_pose()
        lift_target = ik(np.asarray(lift_start)+[0, 0, .03], np.asarray(lift_orientation))
        lift_start_arm = arm.copy()
        for tick in range(ticks(3.)):
            u = (tick+1)/ticks(3.)
            command(lift_start_arm+(lift_target-lift_start_arm)*(u*u*(3-2*u)), grip)
        move_peg([.55, -.13, 1.10], args.lift_duration_s-3., grip)
        if samples[-1]["live_peg"]["position"][2] < 1.06:
            raise RuntimeError("physical_grasp_failed")
        begin("align_socket")
        above = [.60+args.lateral_offset_m, .13, 1.09]
        move_peg(above, 7., grip)
        move_peg(above, 2., grip, hold=True)
        live = _live_rigid_state(peg_view, "peg_1")
        q = normalize(np.asarray(live["orientation_wxyz"]))
        if 2*math.acos(min(1., abs(float(q[0])))) > .005:
            raise RuntimeError("preinsertion_orientation_not_converged")
        begin("insert")
        goal = [.60+args.lateral_offset_m, .13, .9901]
        move_peg(goal, 12., grip)
        begin("hold")
        if args.freeze_seated_pose:
            for _ in range(ticks(1.)):
                command(arm, grip)
        else:
            move_peg(goal, 1., grip, hold=True)
        begin("seat_with_compliant_grip")
        # Relax the grasp while retaining measured pose feedback so gravity
        # seats the part on the socket base before the fingers withdraw.
        seat_goal = [.60+args.lateral_offset_m, .13, args.seat_center_z_m]
        if args.freeze_seated_pose:
            for tick in range(math.ceil(args.seat_duration_s/dt)):
                fraction = grip + (.30-grip)*min(1., (tick+1)*dt/args.seat_duration_s)
                command(arm, fraction)
        else:
            move_peg(seat_goal, args.seat_duration_s, .245, hold=True)
        begin("release")
        for tick in range(ticks(1.5)):
            fraction = .30 if args.freeze_seated_pose else .245
            command(arm, fraction+(1-fraction)*(tick+1)/ticks(1.5))
        released = True
        begin("released_settling")
        for _ in range(math.ceil(args.release_pause_s/dt)):
            command(arm, 1.)
        begin("withdraw")
        move_ee(np.asarray(backend._end_effector_pose()[0])+[0, 0, .12], 4., 1.)
        begin("stability")
        for _ in range(ticks(1.)):
            command(arm, 1.)
        report["phases_completed"] = True
        report["status"] = "trial_completed"
        report["engine_version"] = backend.isaac_sim_version
    except GraspPrefixCompleted:
        report["status"] = "grasp_prefix_completed"
    except OnlineAssemblyTerminal as exc:
        report["online_final"] = exc.response
        report["online_acceptance"] = "passed" if exc.success else "failed"
        report["status"] = "online_task_completed" if exc.success else "failed"
        report["control_ended_on_terminal"] = True
        if not exc.success:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    finally:
        if trace_file is not None:
            trace_file.close()
        if grasp_trace_file is not None:
            grasp_trace_file.close()
        if backend is not None:
            report["physics_overrides"] = backend.runtime_summary.get("physics_overrides")
        if background_views and background_window:
            first, last = background_window[0], background_window[-1]
            window_s = last["time_s"]-first["time_s"]
            drift = max(math.dist(first["objects"][key]["position"], row["objects"][key]["position"])
                        for row in background_window for key in background_views)
            displacement = max(math.dist(background_initial[key]["position"], last["objects"][key]["position"])
                               for key in background_views)
            max_speed = max(math.sqrt(sum(v*v for v in row["objects"][key]["linear_velocity_mps"]))
                            for row in background_window for key in background_views)
            max_spin = max(math.sqrt(sum(v*v for v in row["objects"][key]["angular_velocity_rps"]))
                           for row in background_window for key in background_views)
            report["background_stability"] = {"window_s": window_s,
                                               "sampling": "every_physics_step_final_1s_rolling_window",
                                               "max_position_drift_m": drift,
                                               "max_total_displacement_m": displacement,
                                               "max_speed_mps": max_speed, "max_angular_speed_rps": max_spin,
                                               "thresholds": {"drift_m": .001, "displacement_m": .005,
                                                              "speed_mps": .005, "angular_speed_rps": .05},
                                               "passed": window_s >= 1.-1e-8 and drift <= .001 and displacement <= .005 and max_speed <= .005 and max_spin <= .05}
            write_json_atomic(args.output / "background_trace.json", list(background_window))
        report["assembly_evaluation"] = evaluate_assembly_trace(spec, samples)
        raw_linear = max((math.sqrt(sum(v*v for v in item["live_peg"]["linear_velocity_mps"])) for item in samples), default=0.)
        raw_angular = max((math.sqrt(sum(v*v for v in item["live_peg"]["angular_velocity_rps"])) for item in samples), default=0.)
        report["live_velocity_audit"] = {"max_linear_speed_mps": raw_linear,
                                          "max_angular_speed_rps": raw_angular,
                                          "passed": raw_linear <= spec["max_linear_speed_mps"] and raw_angular <= spec["max_angular_speed_rps"]}
        report["assembly_acceptance"] = "passed" if (
            report["phases_completed"] and report["assembly_evaluation"]["status"] == "passed"
            and "contact_coverage" in report
            and report["live_velocity_audit"]["passed"]
            and report.get("background_stability", {"passed": True})["passed"]
        ) else "failed"
        if args.online:
            report["assembly_acceptance"] = report.get("online_acceptance", "failed")
            if background_views and not report.get("background_stability", {}).get("passed"):
                report["assembly_acceptance"] = "failed"
            if backend is not None:
                online_trace = backend.assembly_trace()
                with (args.output / "online_trace.jsonl").open("w", encoding="utf-8") as evidence:
                    for row in online_trace:
                        evidence.write(json.dumps(row, allow_nan=False) + "\n")
                report["online_sample_count"] = len(online_trace)
                report["online_state"] = backend.runtime_summary.get("task_oracle")
                report["runtime_drive_gains"] = backend.runtime_summary.get("runtime_drive_gains")
        report["sample_count"] = len(samples)
        if args.grasp_only and report.get("grasp_prefix_acceptance") == "passed":
            report["assembly_acceptance"] = "not_verified"
            if not report["live_velocity_audit"]["passed"]:
                report["grasp_prefix_acceptance"] = "failed"
                report["status"] = "failed"
                report["error"] = {"type": "GraspPrefixAudit", "message": "prefix_raw_velocity_limit_exceeded"}
        if grasp_controller is not None:
            report["grasp_control_final"] = grasp_controller.decision()
        report["grasp_phase_summary"] = summarize_grasp_phases(
            samples, contact_force_N=grasp_controller.config.contact_force_N if grasp_controller else .05,
            control_decisions=control_decisions)
        report["cleanup"] = "pending_process_shutdown"
        write_json_atomic(args.output / "report.json", report)
        print(json.dumps({"report": str(args.output / "report.json"),
                          "assembly_acceptance": report["assembly_acceptance"]}), flush=True)
        if session is not None:
            session.close()
    return 0 if (report["assembly_acceptance"] == "passed"
                 or args.grasp_only and report.get("grasp_prefix_acceptance") == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
