from __future__ import annotations

import copy
import json

import pytest

from scene_factory.agent import SceneFactoryEnv
from scene_factory.agent_adapter import JevDecisionPolicy, JevSkillSelector, RegisteredSkill
from scene_factory.backends.mujoco import MujocoBackend
from scene_factory.decision_dataset import export_decision_dataset, validate_decision_sample, verify_decision_dataset
from scene_factory.session import RuntimeSession


def _decision(payload):
    return {"model": "contract_test_double_not_jev_inference", "answers": {
        "skill": {"type": "choice", "choice": "wait", "confidence": 1, "probabilities": {"wait": 1}},
        "risk": {"type": "score", "score": 0}, "recoverable": {"type": "noul", "noul": 1}}}


@pytest.fixture
def recorded(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"scene_id": "training_export", "room_dimensions_m": [2, 2, 3], "objects": [
        {"object_id": "module", "asset_id": "mug_blue", "dynamic": True, "bbox_m": [.04, .04, .06],
         "pose": {"position": [0, 0, .031], "yaw_deg": 0}}],
        "task": {"target_object": "module", "success": {"predicate": "lifted", "min_height_delta_m": .03}}}), encoding="utf-8")
    skills = [RegisteredSkill("wait", "Observe passive physics", {"mode": "noop"})]
    policy = JevDecisionPolicy(JevSkillSelector(transport=_decision), skills, safe_action={"mode": "noop"})
    session = RuntimeSession(SceneFactoryEnv(path, MujocoBackend(frame_skip=1, max_steps=3)))
    try:
        initial = current = session.reset({"idempotency_key": "reset", "seed": 1})
        transitions = []
        for index in range(3):
            action = policy.act(current, session.capabilities)
            request = {"episode_id": current["episode_id"], "sequence_id": index, "idempotency_key": f"step-{index}", "action": action}
            previous = current
            current = session.step(request)
            transitions.append({"sequence_id": index, "request": request, "previous_state_version": previous["state_version"], "result": current})
        assert current["truncated"] and not current["info"]["task"]["task_success"]
        rollout = {"schema_version": "scene_factory.agent_rollout.v1", "status": "completed", "outcome": "engine_time_limit", "seed": 1,
                   "initial": initial, "final": current, "transitions": transitions, "decision_records": policy.decisions}
        recorded_path = tmp_path / "rollout.json"
        recorded_path.write_text(json.dumps(rollout), encoding="utf-8")
        return recorded_path, skills, rollout
    finally:
        session.close()


def _sources(path):
    return [{"path": path, "asset_families": ["mug_proxy"], "task_structure": "external_force_lift", "policy_source": "contract_test_double"}]


def test_export_actual_failed_physics_rollout_retains_failure_and_behavior_provenance(tmp_path, recorded):
    path, skills, _ = recorded
    manifest = export_decision_dataset(_sources(path), tmp_path / "dataset", skills=skills)
    assert manifest["sample_count"] == 3 and manifest["episode_count"] == 1
    assert manifest["training_performed"] is False and manifest["empty_splits"]
    assert verify_decision_dataset(tmp_path / "dataset")["valid"]
    rows = [json.loads(line) for line in (tmp_path / "dataset" / "samples.jsonl").read_text().splitlines()]
    assert all(row["label"]["kind"] == "observed_behavior" and row["label"]["optimality"] == "not_established" for row in rows)
    assert all(row["outcome"]["failure_flags"] == ["engine_time_limit"] and row["outcome"]["task_success"] is False for row in rows)
    assert rows[0]["state"]["step_index"] == 0 and rows[0]["source"]["model_inference_evidence"] == "test_double_no_live_inference"
    with pytest.raises(FileExistsError):
        export_decision_dataset(_sources(path), tmp_path / "dataset", skills=skills)


@pytest.mark.parametrize("mutation", ["action", "time", "backend", "episode", "model", "nonfinite", "sequence"])
def test_export_rejects_contradictory_or_unmeasured_source(tmp_path, recorded, mutation):
    path, skills, original = recorded
    rollout = copy.deepcopy(original)
    if mutation == "action":
        rollout["transitions"][0]["request"]["action"] = {"mode": "object_forces"}
    elif mutation == "time":
        rollout["transitions"][0]["result"]["sim_time"] = 0
    elif mutation == "backend":
        rollout["initial"]["backend"] = "dry_run"
    elif mutation == "episode":
        rollout["transitions"][0]["result"]["episode_id"] = "another-episode"
    elif mutation == "model":
        sources = _sources(path)
        sources[0]["policy_source"] = "typesafe_endpoint"
        with pytest.raises(ValueError, match="test-double"):
            export_decision_dataset(sources, tmp_path / "bad", skills=skills)
        return
    elif mutation == "nonfinite":
        rollout["initial"]["observation"]["value"] = float("nan")
    else:
        rollout["transitions"][0]["request"]["sequence_id"] = 8
    path.write_text(json.dumps(rollout), encoding="utf-8")
    with pytest.raises(ValueError):
        export_decision_dataset(_sources(path), tmp_path / "bad", skills=skills)
    assert not (tmp_path / "bad").exists()


def test_strict_schema_rejects_oracle_and_checksum_tampering(tmp_path, recorded):
    path, skills, _ = recorded
    export_decision_dataset(_sources(path), tmp_path / "dataset", skills=skills)
    sample_path = tmp_path / "dataset" / "samples.jsonl"
    row = json.loads(sample_path.read_text().splitlines()[0])
    row["label"]["kind"] = "optimal_action"
    with pytest.raises(ValueError, match="oracle"):
        validate_decision_sample(row)
    sample_path.write_text(sample_path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_decision_dataset(tmp_path / "dataset")


def test_duplicate_episodes_rejected_before_export(tmp_path, recorded):
    path, skills, _ = recorded
    with pytest.raises(ValueError, match="duplicate"):
        export_decision_dataset(_sources(path) * 2, tmp_path / "dataset", skills=skills)
