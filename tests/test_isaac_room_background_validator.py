from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.validate_isaac_room_background import (
    IncompleteEvidence,
    evaluate_background,
    freeze_fixture,
    hold_default_action,
    make_sample,
    timing_summary,
    validate_sample,
)


def sample(index: int) -> dict:
    state = {"position": [0., 0., .1], "orientation_wxyz": [1., 0., 0., 0.],
             "linear_velocity_mps": [0., 0., 0.], "angular_velocity_rps": [0., 0., 0.],
             "physics_handle_valid": True}
    return {"time_s": index*.1, "episode_id": "test", "step_index": index, "state_version": index+1,
            "objects": {"background": state}, "contacts": [],
            "contact_coverage": {"complete": True, "continuous": True, "scope": "all_scene_contacts",
                                 "force_aggregation": "sum_per_point_normal_magnitudes", "buffer_saturated": False,
                                 "source": "synthetic test measurements", "object_ids": ["background", "robot"],
                                 "sample_dt_s": .1, "collider_paths": ["/background", "/robot/finger"],
                                 "collision_shape_count": 2, "sensor_root_count": 1,
                                 "contact_buffers": [{"used_contact_points": 0, "capacity": 32, "saturated": False}]}}


def validate(row: dict, previous: dict | None = None) -> None:
    validate_sample(row, background_ids=["background"], dynamic_ids={"background"},
                    scene_ids={"background"}, dt=.1, previous=previous)


def test_full_motion_and_contact_coverage_is_required_each_step():
    first, second = sample(0), sample(1)
    validate(first)
    validate(second, first)
    bad = copy.deepcopy(second)
    bad["objects"]["background"]["physics_handle_valid"] = False
    with pytest.raises(IncompleteEvidence, match="live PhysX handle"):
        validate(bad, first)
    bad = copy.deepcopy(second)
    bad["contact_coverage"]["object_ids"] = ["background"]
    with pytest.raises(IncompleteEvidence, match="omits"):
        validate(bad, first)
    bad = copy.deepcopy(second)
    bad["contact_coverage"]["contact_buffers"][0]["used_contact_points"] = 32
    with pytest.raises(IncompleteEvidence, match="saturated"):
        validate(bad, first)
    with pytest.raises(IncompleteEvidence, match="every physics step"):
        validate(sample(2), first)


def test_pair_force_sum_cannot_be_replaced_by_a_resultant():
    row = sample(1)
    row["contacts"] = [{"pair": ["background", "robot"], "normal_force_N": 2.,
                        "normal_force_resultant_N": 3., "contact_point_count": 2,
                        "measurement": "sum_of_absolute_per_contact_point_normal_forces",
                        "sample_window_s": .1, "units": "N"}]
    with pytest.raises(IncompleteEvidence, match="point sum"):
        validate(row)


def test_raw_spin_spike_fails_even_when_all_positions_are_stationary():
    rows = [sample(index) for index in range(11)]
    passed = evaluate_background(rows[0], rows, {"background"}, .1)
    assert passed["status"] == "passed"
    rows[5]["objects"]["background"]["angular_velocity_rps"] = [0., 0., .051]
    failed = evaluate_background(rows[0], rows, {"background"}, .1)
    assert failed["status"] == "failed"
    assert failed["max_position_drift_m"] == 0.
    assert failed["max_angular_speed_rps"] == .051


def test_terminal_pose_does_not_hide_final_window_drift_or_incomplete_duration():
    rows = [sample(index) for index in range(11)]
    rows[5]["objects"]["background"]["position"][0] = .00101
    result = evaluate_background(rows[0], rows, {"background"}, .1)
    assert result["status"] == "failed"
    assert result["max_total_displacement_m"] == 0.
    assert result["max_position_drift_m"] == .00101
    assert evaluate_background(rows[0], rows[1:], {"background"}, .1)["status"] == "not_verified"


def test_timing_n_is_only_actual_completed_calls():
    result = timing_summary([{"status": "completed", "wall_time_ns": 1_000_000},
                             {"status": "failed", "wall_time_ns": 20_000_000},
                             {"status": "completed", "wall_time_ns": 3_000_000}], .1)
    assert result["sample_count"] == 2
    assert result["failed_call_count"] == 1
    assert result["p50_ms"] == 2.
    assert result["p95_ms"] == 3.
    assert result["total_step_call_wall_s"] == .004


def test_default_hold_uses_measured_arm_and_runtime_finger_mapping():
    positions = [.1, -.2, .3, -.4, .5, -.6, .7, .025, .02]
    robot = SimpleNamespace(dof_names=[*(f"panda_joint{i}" for i in range(1, 8)), "finger1", "finger2"],
                            get_joint_positions=lambda: positions)
    backend = SimpleNamespace(_robot=robot, _finger_gripper_config={"indices": [7, 8],
                              "closed_positions": [.01, 0.], "open_positions": [.04, .04]})
    action, provenance = hold_default_action(backend)
    assert action["payload"]["arm_positions_rad"] == positions[:7]
    assert action["payload"]["gripper_open_fraction"] == pytest.approx(.5)
    assert provenance["joint_positions_rad_or_m"] == positions
    assert provenance["writes"] == "joint_drive_targets_only"


def test_frozen_public_square_fixture_rejects_geometry_tampering(tmp_path):
    source = Path(__file__).resolve().parents[1]/"examples/assembly/square_room_assembly/room20"
    fixture = tmp_path/"fixture"
    fixture.mkdir()
    for name in ("scene.usda", "layout.json", "assembly_spec.json", "SOURCE.json"):
        (fixture/name).write_bytes((source/name).read_bytes())
    output = tmp_path/"valid"
    output.mkdir()
    hashes, layout, metadata = freeze_fixture(fixture, output)
    assert hashes["scene.usda"] == metadata["usd_sha256"]
    assert layout["room_fixture"]["background_dynamic_count"] == 6
    assert (output/"frozen_inputs/assembly_spec.json").read_bytes() == (source/"assembly_spec.json").read_bytes()
    (fixture/"scene.usda").write_bytes((fixture/"scene.usda").read_bytes()+b"\n# tampered\n")
    rejected = tmp_path/"invalid"
    rejected.mkdir()
    with pytest.raises(IncompleteEvidence, match="does not bind"):
        freeze_fixture(fixture, rejected)


def test_sample_keeps_full_raw_background_motion_and_contacts():
    raw = sample(0)
    response = {**{key: raw[key] for key in ("time_s", "episode_id", "step_index", "state_version")},
                "sim_time": raw["time_s"],
                "observation": {"objects": raw["objects"], "contacts": [], "contact_coverage": raw["contact_coverage"]}}
    result = make_sample(response, ["background"], kind="physics_step", action={"hold": True}, wall_time_ns=100)
    assert result["objects"] == raw["objects"]
    assert result["contact_coverage"] == raw["contact_coverage"]
    assert json.loads(json.dumps(result, allow_nan=False))["step_call_wall_time_ns"] == 100
