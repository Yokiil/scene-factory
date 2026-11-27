"""Run one fixed Franka mug episode, preserving actions before and observations after every step."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_factory.runtime_worker import TraceWriter, _atomic_json, _sha256

RECIPE = "kitchen_franka_mug_pick_place"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_journal(path):
    """Keep the original journal untouched when a process dies mid-write."""
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
    except (OSError, ValueError) as exc:
        return rows, f"{path.name} after {len(rows)} rows: {exc}"
    return rows, None


def stage_measurements(scene, frames):
    """Find ordered witnesses using body pose, current force pairs and the task's settle window."""
    from scene_factory.runtime_evidence import _force_contact

    task = scene["task"]
    goal = task["success"]
    target = task["target_object"]
    witnesses = dict.fromkeys(("grasp", "transport", "release", "placement"))
    if not frames:
        return witnesses
    initial = frames[0]["objects"][target]["position"]
    region = goal["target_region_xy"]
    stable_steps = 0
    last_pos = initial
    for obs in frames:
        obj = obs["objects"][target]
        pos = obj["position"]
        diag = obs["robot"]["grasp_diagnostics"]
        held = _force_contact(diag, target) and diag.get("gripper_open") is False
        inside_xy = region[0] <= pos[0] <= region[1] and region[2] <= pos[1] <= region[3]
        dz = pos[2] - initial[2]
        valid = all(
            diag.get(k) is True
            for k in (
                "contact_report_available",
                "contact_force_read_valid",
                "contact_report_subscribed",
            )
        )
        released = valid and diag.get("gripper_open") is True and not _force_contact(diag, target)
        supported = any(
            set(pair["pair"]) == {target, goal["target_support"]} and pair["normal_force_N"] > 0
            for pair in obs.get("contacts", [])
        )
        stable = (
            inside_xy
            and released
            and supported
            and abs(pos[2] - goal["target_position_m"][2]) <= goal["target_tolerance_m"][2]
            and math.dist(last_pos, pos) <= goal["max_settle_step_distance_m"]
        )
        stable_steps = stable_steps + 1 if stable else 0
        sample = {
            "step": obs["simulation_step"],
            "position_m": pos,
            "lift_m": dz,
            "gripper_open": diag.get("gripper_open"),
            "finger_force_contact": held,
            "support_contact": supported,
        }
        if witnesses["grasp"] is None and held and dz >= goal["min_lift_delta_m"]:
            witnesses["grasp"] = sample
        elif (
            witnesses["grasp"]
            and witnesses["transport"] is None
            and held
            and inside_xy
            and dz >= goal["target_tolerance_m"][2]
        ):
            witnesses["transport"] = sample
        elif witnesses["transport"] and witnesses["release"] is None and released and inside_xy:
            witnesses["release"] = sample
        elif witnesses["release"] and stable_steps >= goal["settle_steps"]:
            # Require support, release, and stability at the terminal observation as well.
            witnesses["placement"] = (
                {**sample, "consecutive_samples": stable_steps} if obs["terminated"] else None
            )
        last_pos = pos
    return witnesses


def run_worker(args):
    from scene_factory.backends import IsaacSimBackend

    out = args.output
    scene = load(out / "layout.json")
    header = load(out / "episode.json")
    options = (
        {} if args.mode == "scripted" else {"mug_initial_candidate": header["initial_candidate"]}
    )
    backend = IsaacSimBackend(
        out / "scene.usd",
        robot_asset_root=args.asset_root,
        minimal_kit=True,
        enable_general_observation=True,
        max_steps=args.max_steps,
        **options,
    )
    agent = None
    if args.mode != "scripted":
        from scene_factory.embodied_agent import EmbodiedAgentRuntime
        from scene_factory.world_state import build_world_state

        agent = EmbodiedAgentRuntime()
    trace = TraceWriter(out / "trajectory.jsonl")
    decisions = TraceWriter(out / "actions.jsonl")
    error = None
    try:
        obs, info = backend.reset(scene)
        reward, terminated, truncated = 0.0, False, False
        while True:
            step = obs["simulation_step"]
            trace.write({**obs, "reward": reward, "terminated": terminated, "truncated": truncated})
            if agent is not None:
                world = build_world_state(
                    scene,
                    obs,
                    backend="isaac",
                    episode_id=header["episode_id"],
                    step_index=step,
                    scene_sha256=header["layout_sha256"],
                )
                if step == 0:
                    _atomic_json(out / "initial_world_state.json", world)
                skill = agent.next_skill(world)
                if skill is None or terminated or truncated:
                    _atomic_json(out / "final_world_state.json", world)
                    break
                action = {"mode": "mug_skill", "skill": skill}
            else:
                if terminated or truncated:
                    break
                action = "scripted"
            decisions.write(
                {"episode_id": header["episode_id"], "from_step": step, "action": action}
            )
            obs, reward, terminated, truncated, info = backend.step(action)
    except Exception:
        error = traceback.format_exc()
    finally:
        trace.close()
        decisions.close()
        _atomic_json(
            out / "worker.json",
            {
                "error": error,
                "steps": backend.steps,
                "isaac_version": backend.isaac_sim_version,
                "summary": backend.runtime_summary,
                "agent_result": agent.result if agent else None,
                "agent_reason": agent.reason if agent else None,
                "events": agent.events if agent else [],
            },
        )
        backend.close()  # Kit fast shutdown may exit Python here.
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="new directory, ASCII on Windows"
    )
    parser.add_argument(
        "--asset-root", type=Path, required=True, help="official local Isaac asset root"
    )
    parser.add_argument("--mode", choices=["agent", "recovery", "scripted"], default="agent")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--random-seed",
        type=int,
        help="sample the existing mug pick/place data recipe before physics startup",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.asset_root = Path(os.path.abspath(args.asset_root))  # keep ASCII junction spelling
    if args.worker:
        return run_worker(args)
    if args.max_steps < 1 or (os.name == "nt" and not str(args.output).isascii()):
        parser.error("use a positive budget and an ASCII output path on Windows")
    if not (args.asset_root / "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd").is_file():
        parser.error("asset-root must contain the official FrankaPanda/franka.usd")
    from scene_factory.factory import SceneFactory
    from scene_factory.runtime_evidence import assess_task_trace

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    factory = SceneFactory()
    recipe, seed = (
        (RECIPE, 77) if args.random_seed is None else (RECIPE + "_data", args.random_seed)
    )
    case = factory.build_from_recipe(recipe, seed)
    mug = next(item for item in case.scene.objects if item.object_id == "mug_1")
    if (
        not case.valid
        or mug.asset_id != "mug_001"
        or factory.registry.get(mug.asset_id).status != "ready"
    ):
        raise ValueError("requires valid scene with ready mug_001")
    factory.write_result(case, out, export_usd=True)
    _atomic_json(
        out / "episode.json",
        {
            "episode_id": str(uuid4()),
            "recipe": recipe,
            "seed": seed,
            "mode": args.mode,
            "initial_candidate": "lateral" if args.mode == "recovery" else "nominal",
            "max_steps": args.max_steps,
            "physics_dt": 1 / 60,
            "layout_sha256": _sha256(out / "layout.json"),
            "scene_usd_sha256": _sha256(out / "scene.usd"),
            "robot_usd_sha256": _sha256(
                args.asset_root / "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            ),
            "code_sha256": {
                p.relative_to(ROOT).as_posix(): _sha256(p)
                for p in (
                    Path(__file__).resolve(),
                    ROOT / "scene_factory/embodied_agent.py",
                    ROOT / "scene_factory/world_state.py",
                    ROOT / "scene_factory/backends/isaac.py",
                    ROOT / "scene_factory/backends/isaac_mug_skills.py",
                    ROOT / "scene_factory/backends/mug_physics.kit",
                    ROOT / "scene_factory/robotics.py",
                    ROOT / "scene_factory/tasks.py",
                    ROOT / "scene_factory/runtime_evidence.py",
                )
            },
        },
    )
    with (out / "isaac.log").open("xb") as log:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(out),
                "--asset-root",
                str(args.asset_root),
                "--mode",
                args.mode,
                "--max-steps",
                str(args.max_steps),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    worker = (
        load(out / "worker.json")
        if (out / "worker.json").exists()
        else {"error": "worker exited before reporting"}
    )
    trace_path = out / "trajectory.jsonl"
    frames, frame_error = read_journal(trace_path)
    action_path = out / "actions.jsonl"
    actions, action_error = read_journal(action_path)
    worker["error"] = worker.get("error") or frame_error or action_error
    report = assess_task_trace(
        load(out / "layout.json"),
        frames,
        {"runtime_summary": worker.get("summary")},
        recovery_actions=actions if args.mode == "recovery" else None,
    )
    episode_id = load(out / "episode.json")["episode_id"]
    action_alignment = len(actions) + 1 == len(frames) and all(
        row.get("episode_id") == episode_id and row.get("from_step") == i
        for i, row in enumerate(actions)
    )
    stages = stage_measurements(load(out / "layout.json"), frames)
    passed = (
        all(stages.values())
        and action_alignment
        and (args.mode != "recovery" or "recovery_transition_step" in report["metrics"])
        and process.returncode == 0
        and not worker["error"]
        and report["status"] == "passed"
        and (args.mode == "scripted" or worker["agent_result"] == "success")
    )
    _atomic_json(
        out / "acceptance.json",
        {
            "status": "passed" if passed else "failed",
            "mode": args.mode,
            "returncode": process.returncode,
            "error": worker["error"],
            "agent_result": worker.get("agent_result"),
            "agent_reason": worker.get("agent_reason"),
            "trace_replay": report,
            "stages": stages,
            "actions_aligned": action_alignment,
            "actions_sha256": _sha256(action_path) if action_path.exists() else None,
            "trajectory_sha256": _sha256(trace_path) if trace_path.exists() else None,
        },
    )
    print(f"{args.mode}: {'passed' if passed else 'failed'}, {len(frames)} frames | {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
