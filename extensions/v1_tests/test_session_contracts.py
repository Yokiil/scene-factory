from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scene_factory.session_contracts import (
    MAX_REQUEST_STEPS,
    OBSERVATION_SCHEMA,
    RESET_SCHEMA,
    STEP_SCHEMA,
    backend_capabilities,
    finite_json,
    make_observation_envelope,
    runtime_session_schema,
    translate_action,
    validate_reset_request,
    validate_step_request,
)


def step_request(mode="noop", payload=None):
    return {
        "episode_id": "episode-1", "sequence_id": 0, "idempotency_key": "request-1",
        "action": {"mode": mode, "payload": payload if payload is not None else {}},
    }


def envelope(**overrides):
    fields = {
        "session_id": "session-1", "episode_id": "episode-1", "scene_sha256": "a" * 64,
        "state_version": 1, "step_index": 0, "sim_time": None, "backend": "dry_run",
        "observation": {"scene_graph": []},
    }
    fields.update(overrides)
    return make_observation_envelope(**fields)


def test_capabilities_distinguish_physics_and_do_not_advertise_unimplemented_controls():
    dry, mujoco, isaac = (backend_capabilities(name) for name in ("dry_run", "mujoco", "isaac"))
    assert not dry["physical"]
    assert mujoco["fidelity"] == "collision_proxy"
    assert isaac["action_modes"] == ["scripted", "joint_position"]
    assert isaac["reset_options"] == []
    assert mujoco["reset_options"] == ["pose_overrides", "pose_jitter", "randomization"]
    for name in ("dry_run", "mujoco", "isaac"):
        capabilities = backend_capabilities(name)
        assert capabilities["external_joint_control"] == (name == "isaac")
        assert ("rebuild" in capabilities["snapshot_levels"]) == (name != "isaac")
        with pytest.raises(ValueError, match="not supported"):
            validate_step_request(step_request("torque", {"targets": [0.1]}), capabilities)


def test_defaults_are_canonical_and_returned_trees_are_detached():
    request = step_request("object_forces", {"object_forces": {"cup": [1, 2, 3]}})
    original = copy.deepcopy(request)
    result = validate_step_request(request, "mujoco")
    assert result["schema_version"] == STEP_SCHEMA
    assert result["steps"] == 1
    result["action"]["payload"]["object_forces"]["cup"][0] = 99
    assert request == original
    caps = backend_capabilities("isaac")
    caps["action_modes"].append("joint_position")
    assert backend_capabilities("isaac")["action_modes"] == ["scripted", "joint_position"]
    with pytest.raises(ValueError, match="capabilities"):
        validate_step_request(step_request("scripted"), caps)


@pytest.mark.parametrize("field,value", [
    ("sequence_id", True), ("sequence_id", -1), ("sequence_id", 0.0),
    ("steps", 0), ("steps", -1), ("steps", True), ("steps", MAX_REQUEST_STEPS + 1),
    ("episode_id", " "), ("idempotency_key", ""), ("idempotency_key", "a" * 257),
    ("schema_version", "v999"), ("typo", 1),
])
def test_invalid_step_requests_are_rejected(field, value):
    request = step_request()
    request[field] = value
    with pytest.raises(ValueError):
        validate_step_request(request, "mujoco")


@pytest.mark.parametrize("force", [
    [True, 1, 2], [1, "2", 3], [1, 2], [1, 2, 3, 4],
    [1, 2, float("nan")], [float("inf"), 1, 2], [10**400, 1, 2],
    (1, 2, 3), {"x": 1},
])
def test_force_actions_reject_malformed_or_nonfinite_vectors(force):
    with pytest.raises(ValueError):
        validate_step_request(step_request("object_forces", {"object_forces": {"cup": force}}), "mujoco")


@pytest.mark.parametrize("mode,payload,backend", [
    ("noop", {"ignored": True}, "mujoco"),
    ("scripted", {"target": "cup"}, "isaac"),
    ("object_forces", {}, "mujoco"),
    ("object_forces", {"object_forces": {}}, "mujoco"),
    ("object_forces", {"object_forces": {"cup": [1, 2, 3]}, "ctrl": [1]}, "mujoco"),
    ("object_forces", {"object_forces": {"": [1, 2, 3]}}, "mujoco"),
    ("scripted", {}, "mujoco"),
    ("noop", {}, "isaac"),
])
def test_payloads_cannot_silently_smuggle_unsupported_commands(mode, payload, backend):
    with pytest.raises(ValueError):
        validate_step_request(step_request(mode, payload), backend)


def test_translate_actions_preserves_force_and_torque_order():
    vector = [1, 2, 3, 4, 5, 6]
    action = {"mode": "object_forces", "payload": {"object_forces": {"cup": vector}}}
    assert translate_action(action, "mujoco") == {"object_forces": {"cup": vector}}
    assert translate_action({"mode": "noop"}, "mujoco") is None
    assert translate_action({"mode": "scripted"}, "isaac") == "scripted"
    assert translate_action({"mode": "dry_run", "payload": {"label": "integration"}}, "dry_run") == {
        "label": "integration",
    }


def test_reset_contract_defers_options_and_requires_idempotency():
    assert validate_reset_request({"idempotency_key": "reset-1"}) == {
        "schema_version": RESET_SCHEMA, "idempotency_key": "reset-1", "seed": None,
        "options": {},
    }
    options = {"pose_jitter": {"cup": {"x_m": [-0.01, 0.01]}}}
    result = validate_reset_request({"idempotency_key": "r", "seed": 2**64 - 1, "options": options})
    assert result["options"] == options
    # Unknown option *semantics* are owned by SceneFactoryEnv, not transport validation.
    assert validate_reset_request({"idempotency_key": "r", "options": {"custom": {}}})["options"] == {
        "custom": {},
    }
    with pytest.raises(ValueError, match="idempotency_key"):
        validate_reset_request({})


@pytest.mark.parametrize("fields", [
    {"seed": True}, {"seed": -1}, {"seed": 2**64}, {"seed": 1.0},
    {"options": []}, {"options": None}, {"options": {"bad": float("inf")}},
    {"unknown": True}, {"schema_version": "old"},
])
def test_reset_request_rejects_invalid_transport(fields):
    with pytest.raises(ValueError):
        validate_reset_request({"idempotency_key": "reset-1", **fields})


def test_observation_does_not_invent_physical_fields_or_image_alignment():
    raw = {"scene_graph": [{"object_id": "cup", "pose": {"yaw_deg": 30}}]}
    result = envelope(observation=raw)
    assert result["schema_version"] == OBSERVATION_SCHEMA
    assert result["sim_time"] is None
    assert "contacts" in result["unavailable_fields"]
    assert "image_references" in result["unavailable_fields"]
    assert result["images"] == []
    result["observation"]["scene_graph"][0]["object_id"] = "changed"
    assert raw["scene_graph"][0]["object_id"] == "cup"


@pytest.mark.parametrize("overrides", [
    {"scene_sha256": "a" * 63}, {"scene_sha256": "A" * 64},
    {"state_version": True}, {"step_index": -1}, {"sim_time": -1},
    {"sim_time": float("nan")}, {"reward": float("inf")}, {"reward": True},
    {"terminated": 1}, {"truncated": None}, {"observation": []},
    {"observation": {"position": [float("nan"), 0, 0]}},
    {"info": {"diagnostics": {3: "bad key"}}},
])
def test_observation_validation_rejects_invalid_state(overrides):
    with pytest.raises(ValueError):
        envelope(**overrides)


@pytest.mark.parametrize("value", [set(), object(), {1: 2}, {"tuple": (1, 2)}, {"nan": float("nan")}])
def test_finite_json_rejects_non_json_values(value):
    with pytest.raises(ValueError):
        finite_json(value)


def test_finite_json_rejects_cycles_but_allows_shared_values():
    shared = {"ok": [1, 2, 3]}
    assert finite_json([shared, shared]) == [shared, shared]
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="circular"):
        finite_json(cycle)
    nested = []
    for _ in range(130):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        finite_json(nested)


def test_static_schema_matches_public_schema_and_canonical_versions():
    path = Path(__file__).resolve().parents[1] / "schemas" / "runtime_session.schema.json"
    schema = runtime_session_schema()
    assert json.loads(path.read_text(encoding="utf-8")) == schema
    assert schema["$defs"]["step"]["properties"]["schema_version"]["const"] == STEP_SCHEMA
    assert schema["$defs"]["reset"]["properties"]["schema_version"]["const"] == RESET_SCHEMA
    assert schema["$defs"]["observation"]["properties"]["schema_version"]["const"] == OBSERVATION_SCHEMA
    schema["$defs"]["step"]["required"].clear()
    assert "episode_id" in runtime_session_schema()["$defs"]["step"]["required"]
