"""Run a real Isaac external-position-control and optional aligned RGB-D smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.agent import SceneFactoryEnv
from scene_factory.backends.isaac import IsaacSimBackend
from scene_factory.dataset import write_json_atomic
from scene_factory.joint_control import ARM_JOINT_NAMES
from scene_factory.session import RuntimeSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("usd", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "scene_factory.isaac_session_smoke.v1", "status": "failed",
              "scope": "joint_drive_and_aligned_sensor_protocol_not_assembly_acceptance",
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    session = None
    try:
        if args.steps < 1 or args.steps > 10000:
            raise ValueError("steps must be in [1,10000]")
        backend = IsaacSimBackend(args.usd, max_steps=args.steps + 10, enable_rgbd=args.rgbd,
                                  robot_asset_root=args.asset_root)
        session = RuntimeSession(SceneFactoryEnv(args.scene, backend))
        session.artifact_root = args.output / "artifacts"
        initial = session.reset({"seed": 0, "options": {}, "idempotency_key": "smoke-reset"})
        joint_positions = backend._robot.get_joint_positions()
        names = list(backend._robot.dof_names)
        arm = [float(joint_positions[names.index(name)]) for name in ARM_JOINT_NAMES]
        # A small bounded change proves that a client target reaches the position drives.
        arm[0] += 0.01
        request = {"episode_id": session.episode_id, "sequence_id": 0,
                   "idempotency_key": "smoke-joints", "steps": args.steps,
                   "action": {"mode": "joint_position", "payload": {
                       "arm_positions_rad": arm, "gripper_open_fraction": 1.0}}}
        final = session.step(request)
        retried = session.step(request)
        if retried != final or backend.steps != args.steps:
            raise RuntimeError("duplicate request advanced simulation")
        images = session.observe({"modalities": ["state", "rgb", "depth"]}) if args.rgbd else final
        if args.rgbd:
            repeated = session.observe({"modalities": ["rgb", "depth"]})
            if repeated["state_version"] != images["state_version"] or backend.steps != args.steps:
                raise RuntimeError("repeated observation changed committed state")
            if {frame["sim_time"] for frame in repeated["images"]} != {final["sim_time"]}:
                raise RuntimeError("repeated image capture was not aligned")
            report["repeated_observation_same_state"] = True
        report.update(status="passed", initial=initial, final=final, observation=images,
                      engine_version=backend.isaac_sim_version,
                      external_action=request, duplicate_did_not_step=True,
                      robot_asset_source=backend.runtime_summary.get("robot_asset_source"))
    except Exception as exc:
        report.update(error={"type": type(exc).__name__, "message": str(exc)}, traceback=traceback.format_exc())
    finally:
        # Isaac fast shutdown can terminate the Python process inside close().
        # Persist evidence first; the supervising process binds its exit status.
        report["cleanup"] = "pending_process_shutdown"
        write_json_atomic(args.output / "report.json", report)
        if session is not None:
            try:
                session.close()
                report["cleanup"] = "completed"
            except Exception as exc:
                report.update(status="failed", cleanup=str(exc))
        write_json_atomic(args.output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json")}, ensure_ascii=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
