from __future__ import annotations

import contextlib
import io
import json

import pytest

from scene_factory.cli import main


def invoke(args):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(args)
    return code, json.loads(stdout.getvalue())


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({
        "scene_id": "cli-test", "seed": 42, "objects": [{
            "object_id": "part", "dynamic": True, "support": "floor",
            "pose": {"position": [0, 0, 0.05], "yaw_deg": 0},
        }], "task": {},
    }), encoding="utf-8")
    return path


def test_schema_and_capabilities_distinguish_joint_control_support():
    code, schema = invoke(["sandbox", "schema"])
    assert code == 0
    assert "$defs" in schema
    code, capabilities = invoke(["sandbox", "capabilities"])
    assert code == 0
    assert capabilities["isaac"]["action_modes"] == ["scripted", "joint_position"]
    assert capabilities["isaac"]["external_joint_control"]
    assert not capabilities["mujoco"]["external_joint_control"]


def test_smoke_uses_session_contract_and_persists_finite_report(scene, tmp_path):
    options = tmp_path / "reset.json"
    options.write_text(json.dumps({"pose_jitter": {"part": {"x_m": [-0.01, 0.01]}}}), encoding="utf-8")
    output = tmp_path / "smoke.json"
    code, report = invoke([
        "sandbox", "smoke", str(scene), "--seed", "123", "--steps", "2",
        "--reset-options", str(options), "--output", str(output),
    ])
    assert code == 0 and report["status"] == "passed"
    assert report["final"]["step_index"] == 2
    assert report["retry_response_identical"] is True
    assert report["final"]["sim_time"] is None
    assert report["initial"]["info"]["episode_reset"]["seed"] == 123
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("mode", ["joint_position", "scripted"])
def test_unsupported_action_produces_failure_evidence(scene, tmp_path, mode):
    action = tmp_path / "action.json"
    action.write_text(json.dumps({"mode": mode, "payload": {}}), encoding="utf-8")
    output = tmp_path / "failure.json"
    code, report = invoke([
        "sandbox", "smoke", str(scene), "--action", str(action), "--output", str(output),
    ])
    assert code == 1 and report["status"] == "failed"
    assert "not supported" in report["error"]["message"]
    assert output.is_file()


def test_malformed_options_also_leave_failure_evidence(scene, tmp_path):
    options = tmp_path / "reset.json"
    options.write_text("{", encoding="utf-8")
    output = tmp_path / "failure.json"
    code, report = invoke([
        "sandbox", "smoke", str(scene), "--reset-options", str(options), "--output", str(output),
    ])
    assert code == 1 and report["error"]["type"] == "JSONDecodeError"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_smoke_cannot_overwrite_scene(scene):
    before = scene.read_bytes()
    with contextlib.redirect_stderr(io.StringIO()):
        assert main(["sandbox", "smoke", str(scene), "--output", str(scene)]) == 1
    assert scene.read_bytes() == before


def test_benchmark_command_is_wired_to_baseline_harness(scene, tmp_path):
    output = tmp_path / "baseline.json"
    code, report = invoke([
        "sandbox", "benchmark", str(scene), "--backend", "dry-run",
        "--creation-trials", "1", "--reset-trials", "2", "--step-trials", "3",
        "--output", str(output),
    ])
    assert code == 0
    assert report["physics_acceptance"] == "not_evaluated"
    assert output.is_file()
