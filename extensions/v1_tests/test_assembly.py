from __future__ import annotations

import copy
import math

import pytest

from scene_factory.assembly import ASSEMBLY_SCHEMA, AssemblySpec, evaluate_assembly_trace


@pytest.fixture
def fixture():
    spec = {"schema_version": ASSEMBLY_SCHEMA, "spec_id": "peg-socket-v1", "peg_id": "peg", "socket_id": "socket",
            "peg_interface_position_m": [0, 0, 0], "socket_interface_position_m": [0, 0, 0],
            "insertion_axis_socket": [0, 0, -1], "target_depth_m": 0.02, "clearance_m": [0.0001, 0.001],
            "position_tolerance_m": 0.001, "angular_tolerance_rad": 0.05, "max_force_N": 20,
            "min_hold_s": 0.5, "release_stable_s": 0.5, "timeout_s": 5, "jam_window_s": 0.5,
            "min_progress_m": 0.001, "max_linear_speed_mps": 0.5, "max_angular_speed_rps": 4,
            "drop_below_z_m": -0.1, "allowed_contact_pairs": [["peg", "socket"], ["robot", "peg"]]}
    rows = [{"time_s": i / 10, "geometry_clearance_m": 0.0005,
             "parts": {"peg": {"position_m": [0, 0, -min(i * 0.002, 0.02)], "orientation_wxyz": [1, 0, 0, 0]},
                       "socket": {"position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0]}},
             "contacts": [{"pair": ["peg", "socket"], "normal_force_N": 3}], "released": i >= 12,
             "motion_source": "robot_joint_control" if i < 12 else "physics_passive",
             "action": {"mode": "joint_position", "targets": [0.1]}} for i in range(21)]
    return spec, rows


def test_full_pose_contract_and_contact_trace_pass_fixture(fixture):
    spec, trace = fixture
    parsed = AssemblySpec.from_dict(spec)
    assert AssemblySpec.from_dict(parsed.to_dict()) == parsed
    result = evaluate_assembly_trace(parsed, trace)
    assert result["status"] == "passed" and result["classification"] == "success"
    assert "trace_classifier_only" in result["metrics"]["evidence"]


@pytest.mark.parametrize("kind", ["missing_contact", "missing_action", "teleport", "overforce", "wrong_hole", "tilt", "drop"])
def test_success_cannot_be_faked_by_only_final_position(fixture, kind):
    spec, trace = fixture
    if kind == "missing_contact":
        for row in trace:
            row["contacts"] = []
    elif kind == "missing_action":
        del trace[1]["action"]
    elif kind == "teleport":
        trace[1]["kinematic_pose_write"] = True
    elif kind == "overforce":
        trace[8]["contacts"][0]["normal_force_N"] = 25
    elif kind == "wrong_hole":
        trace[0]["geometry_clearance_m"] = -0.001
    elif kind == "tilt":
        for row in trace:
            row["parts"]["peg"]["orientation_wxyz"] = [math.cos(0.1), math.sin(0.1), 0, 0]
    else:
        trace[15]["dropped"] = True
    assert evaluate_assembly_trace(spec, trace)["status"] != "passed"


def test_discontinuous_hold_and_release_do_not_add_up_to_success(fixture):
    spec, trace = fixture
    for index in range(11, 19):
        trace[index]["parts"]["peg"]["position_m"][0] = 0.002
    assert evaluate_assembly_trace(spec, trace)["status"] == "failed"
    spec, trace = copy.deepcopy(fixture)
    for index in range(13, 19):
        trace[index]["released"] = False
    assert evaluate_assembly_trace(spec, trace)["status"] == "failed"


def test_invalid_time_and_force_evidence_is_not_verified(fixture):
    spec, trace = fixture
    trace[1]["time_s"] = 0
    assert evaluate_assembly_trace(spec, trace)["status"] == "not_verified"
    spec, trace = copy.deepcopy(fixture)
    del trace[1]["contacts"][0]["normal_force_N"]
    assert evaluate_assembly_trace(spec, trace)["status"] == "not_verified"


def test_jammed_trace_has_distinct_classification(fixture):
    spec, trace = fixture
    for row in trace:
        row["parts"]["peg"]["position_m"] = [0, 0, -0.005]
    assert evaluate_assembly_trace(spec, trace)["classification"] == "jammed"


def test_initial_pose_inside_socket_cannot_count_as_assembly(fixture):
    spec, trace = fixture
    for row in trace:
        row["parts"]["peg"]["position_m"] = [0, 0, -0.02]
    assert evaluate_assembly_trace(spec, trace)["reasons"] == ["initially_assembled_is_not_insertion"]


def test_depth_uses_frozen_tolerance_on_both_sides_of_target(fixture):
    spec, trace = fixture
    for row in trace:
        row["parts"]["peg"]["position_m"][2] *= .99
    assert evaluate_assembly_trace(spec, trace)["status"] == "passed"
    for row in trace:
        row["parts"]["peg"]["position_m"][2] *= .9
    assert evaluate_assembly_trace(spec, trace)["status"] == "failed"


def test_release_settling_restarts_continuous_terminal_stability_window(fixture):
    spec, trace = fixture
    trace[13]["parts"]["peg"]["position_m"][0] = .0007
    result = evaluate_assembly_trace(spec, trace)
    assert result["status"] == "passed"
    assert result["metrics"]["stable_release_duration_s"] == pytest.approx(.5)
    trace[-1]["parts"]["peg"]["position_m"][0] = .0007
    assert evaluate_assembly_trace(spec, trace)["reasons"] == ["released_assembly_unstable"]
