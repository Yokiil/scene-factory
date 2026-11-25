"""Measure four-layer resets and bounded decision-policy HTTP rollouts.

Default decision responses are an explicitly named test double. --live-jev
uses TypeSafe's authenticated endpoint; no network inference claim is made
when that flag is absent. Real MuJoCo/HTTP physics runs in both cases.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.agent_adapter import (
    JevDecisionPolicy, JevSkillSelector, RegisteredSkill, RuntimeAgentAdapter,
)
from scene_factory.dataset import write_json_atomic
from scene_factory.episode_randomization import assign_family_splits
from scene_factory.runtime_client import RuntimeClient
from scene_factory.runtime_service import RuntimeService, serve


def reference_decision(payload):
    """Contract test double; selects only a fixed, registered upward force."""
    options = payload["questions"]["skill"]["criteria"]
    selected = "bounded_lift" if "bounded_lift" in options else next(iter(options))
    return {"model": "contract_test_double_not_jev_inference", "answers": {
        "skill": {"type": "choice", "choice": selected, "confidence": 1.,
                  "probabilities": {k: float(k == selected) for k in options}},
        "risk": {"type": "score", "score": 0.}, "recoverable": {"type": "noul", "noul": .9}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--live-jev", action="store_true")
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("use a new directory to preserve prior evidence")
    source = {"scene_id": "agent-proxy-lift", "room_dimensions_m": [2, 2, 3],
              "objects": [{"object_id": "module", "asset_id": "mug_blue", "dynamic": True,
                           "bbox_m": [.04, .04, .06], "pose": {"position": [0, 0, .031], "yaw_deg": 0}}],
              "task": {"instruction": "Lift the proxy with an external force actuator; this is not robot assembly",
                       "target_object": "module", "success": {"predicate": "lifted", "min_height_delta_m": .03}}}
    write_json_atomic(output / "scene.json", source)
    manager = RuntimeService(output / "service", allowed_scene_root=output, max_sessions=1)
    server = serve(manager, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = {"schema_version": "scene_factory.agent_randomization_validation.v1", "status": "running",
              "physics_backend": "mujoco_collision_proxy", "live_jev": args.live_jev,
              "decision_source": "typesafe_endpoint" if args.live_jev else "contract_test_double",
              "claims": "four_layer_episode_parameters_and_external_policy_transport; not fine_robot_assembly",
              "episodes": [], "counts": {}, "fault_model": "initial_physical_xy_yaw_misalignment",
              "observation_model": "additive_gaussian_positions_and_optional_rgb"}
    counts = Counter()
    client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
    try:
        for seed in range(args.candidates):
            difficulty = ("easy", "medium", "hard")[seed % 3]
            radius = {"easy": .001, "medium": .004, "hard": .01}[difficulty]
            options = {"randomization": {
                "difficulty": difficulty,
                "layout": {"pose_jitter": {"module": {"x_m": [-.02, .02], "y_m": [-.02, .02]}}},
                "physics": {"instances": {"module": {"density_scale": [.8, 1.2], "friction": [.2, .6]}}},
                "initial_state": {"pose_jitter": {"module": {"x_m": [-radius, radius], "yaw_deg": [-2, 2]}}},
                "observation": {"position_noise_std_m": [.0001, radius], "rgb_noise_std_uint8": [.5, 2]}}}
            selector = JevSkillSelector(model=args.model, transport=None if args.live_jev else reference_decision)
            skills = [RegisteredSkill("bounded_lift", "Apply the registered bounded upward force to the proxy",
                       {"mode": "object_forces", "payload": {"object_forces": {"module": [0, 0, 6]}}}, ("state_valid",)),
                      RegisteredSkill("wait", "Observe passive physical evolution", {"mode": "noop"})]
            policy = JevDecisionPolicy(selector, skills, safe_action={"mode": "noop"})
            rollout = RuntimeAgentAdapter(client, policy).run_episode({
                "scene_path": str(output / "scene.json"), "backend": "mujoco",
                "backend_options": {"frame_skip": 1, "max_steps": 100}},
                seed=seed, options=options, max_actions=80)
            rollout["decision_records"] = policy.decisions
            write_json_atomic(output / f"episode-{seed:04}.json", rollout)
            counts[rollout["outcome"]] += 1
            counts["candidate_count"] += 1
            report["episodes"].append({"seed": seed, "difficulty": difficulty, "status": rollout["status"],
                                       "outcome": rollout["outcome"], "path": f"episode-{seed:04}.json"})
        splits = assign_family_splits([
            {"episode_id": str(i), "asset_families": ["mug_proxy"], "task_structure": "external_force_lift", "seed": i}
            for i in range(args.candidates)])
        write_json_atomic(output / "family_splits.json", splits)
        report["family_split_limitation"] = "one shared asset/task component; no independent validation/test claimed"
        report["counts"] = dict(counts)
        report["status"] = "passed" if all(row["status"] == "completed" for row in report["episodes"]) else "failed"
        report["live_model_acceptance"] = "measured_in_this_report" if args.live_jev else "not_run_no_credential_inference"
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(3)
        write_json_atomic(output / "report.json", report)
    print(json.dumps({"status": report["status"], "counts": dict(counts), "live_jev": args.live_jev}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
