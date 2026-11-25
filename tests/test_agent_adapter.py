from __future__ import annotations

import copy

import pytest

from scene_factory.agent_adapter import JevDecisionPolicy, JevHttpTransport, JevSkillSelector, RegisteredSkill


SKILLS = [RegisteredSkill("inspect", "Inspect measured state", {"mode": "noop"}),
          RegisteredSkill("recover", "Bounded retreat", {"mode": "noop"}, ("safe_retreat",))]


def reply(payload):
    keys = list(payload["questions"]["skill"]["criteria"])
    probabilities = {key: (1.0 if index == 0 else 0.0) for index, key in enumerate(keys)}
    return {"model": "jev-2026-test", "answers": {
        "skill": {"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": 1.0},
        "risk": {"type": "score", "score": 0.0}, "recoverable": {"type": "noul", "noul": .9}}}


def test_jev_bounded_selection_does_not_accept_arbitrary_actuation():
    sent = []

    def transport(payload):
        sent.append(copy.deepcopy(payload))
        return reply(payload)

    result = JevSkillSelector(transport=transport).select({"verified_preconditions": {}}, SKILLS)
    assert result["status"] == "selected"
    assert result["action"] == {"mode": "noop"}
    assert result["candidate_skills"] == ["inspect"]
    assert sent[0]["model"] == "jev-latest"
    assert sent[0]["questions"]["skill"]["type"] == "choice"
    assert result["model_resolved"] == "jev-2026-test"
    assert result["latency_ms"] >= 0


@pytest.mark.parametrize("mutation", [
    lambda r: r["answers"]["skill"].update(choice="write_pose"),
    lambda r: r["answers"]["skill"].update(probabilities={"inspect": .8}),
    lambda r: r["answers"]["skill"].update(probabilities={"inspect": float("nan")}),
    lambda r: r["answers"]["skill"].update(type="score"),
    lambda r: r["answers"]["risk"].update(score=99),
    lambda r: r["answers"]["recoverable"].update(noul=True),
    lambda r: r["answers"].update(arbitrary={"code": "import os"}),
])
def test_jev_rejects_malformed_output(mutation):
    def transport(payload):
        result = reply(payload)
        mutation(result)
        return result

    result = JevSkillSelector(transport=transport).select({}, SKILLS)
    assert result["status"] == "abstained"
    assert result["action"] is None
    assert result["reason"] == "model_or_transport_rejected"


def test_jev_physics_hard_gate_skips_model_and_uncertainty_abstains():
    def forbidden(_):
        raise AssertionError("measured hard violations must not call a model")

    result = JevSkillSelector(transport=forbidden).select({"hard_violations": ["force_limit"]}, SKILLS)
    assert result["reason"] == "measured_hard_constraint_violation"

    def uncertain(payload):
        value = reply(payload)
        value["answers"]["skill"].update(probabilities={"inspect": .55, "recover": .45}, confidence=.1)
        return value

    result = JevSkillSelector(transport=uncertain).select({"verified_preconditions": {"safe_retreat": True}}, SKILLS)
    assert result["reason"] == "uncertain_choice"
    assert result["action"] is None


def test_jev_risk_and_failed_recovery_gate():
    def risky(payload):
        value = reply(payload)
        value["answers"]["risk"]["score"] = 2
        return value

    assert JevSkillSelector(transport=risky).select({}, SKILLS)["reason"] == "risk_gate"

    def unrecoverable(payload):
        value = reply(payload)
        value["answers"]["recoverable"]["noul"] = .1
        return value

    result = JevSkillSelector(transport=unrecoverable).select({"failure_history": ["slipped"]}, SKILLS)
    assert result["reason"] == "recovery_not_supported"


def test_jev_credentials_and_endpoint_restrictions(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = JevSkillSelector().select({}, SKILLS)
    assert result["reason"] == "model_or_transport_rejected"
    assert "TYPESAFE_API_KEY" not in str(result)
    with pytest.raises(ValueError):
        JevHttpTransport("http://example.com/v1/systemone")
    with pytest.raises(ValueError):
        JevHttpTransport("https://key:secret@example.com/v1/systemone")
    with pytest.raises(ValueError):
        JevHttpTransport("https://example.com/v1/systemone", api_key_env=None)
    assert JevHttpTransport("http://127.0.0.1:8741/v1/systemone", api_key_env=None).api_key_env is None


def test_physical_force_gate_reads_actual_mujoco_field():
    from scene_factory.session_contracts import backend_capabilities

    policy = JevDecisionPolicy(JevSkillSelector(transport=reply), SKILLS, safe_action={"mode": "noop"})
    with pytest.raises(RuntimeError, match="force limit"):
        policy.act({"episode_id": "physical", "step_index": 1, "sim_time": .01,
                    "observation": {"contacts": [{"force_world_n": [0, 0, 31], "force_unit": "N"}]}},
                    backend_capabilities("mujoco"))
    assert policy.decisions[-1]["reason"] == "measured_hard_constraint_violation"
