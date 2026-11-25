from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from scene_factory.assembly import ASSEMBLY_SCHEMA
from scene_factory.assembly_runtime import AssemblyRuntimeEvaluator, CONTACT_SCOPE, FORCE_AGGREGATION


@pytest.fixture
def evidence():
    spec = {"schema_version": ASSEMBLY_SCHEMA, "spec_id": "online-peg-v1", "peg_id": "peg", "socket_id": "socket",
            "peg_interface_position_m": [0, 0, 0], "socket_interface_position_m": [0, 0, 0],
            "insertion_axis_socket": [0, 0, -1], "target_depth_m": .02, "clearance_m": [.0001, .001],
            "position_tolerance_m": .001, "angular_tolerance_rad": .05, "max_force_N": 20,
            "min_hold_s": .5, "release_stable_s": .5, "timeout_s": 5, "jam_window_s": .5,
            "min_progress_m": .001, "max_linear_speed_mps": .5, "max_angular_speed_rps": 4,
            "drop_below_z_m": -.1, "allowed_contact_pairs": [["peg", "socket"], ["robot", "peg"]]}
    rows = []
    for index in range(21):
        objects = {"peg": {"position_m": [0, 0, -min(index*.002, .02)], "orientation_wxyz": [1, 0, 0, 0],
                           "linear_velocity_mps": [0, 0, -.02] if index <= 10 else [0, 0, 0], "angular_velocity_rps": [0, 0, 0]},
                   "socket": {"position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0],
                              "linear_velocity_mps": [0, 0, 0], "angular_velocity_rps": [0, 0, 0]}}
        rows.append({"sim_time": index/10, "objects": objects,
                     "contacts": [{"pair": ["peg", "socket"], "normal_force_N": 3, "normal_force_points_N": [1, 2]}],
                     "coverage": {"complete": True, "continuous": True, "scope": CONTACT_SCOPE,
                                  "object_ids": ["peg", "socket"], "force_aggregation": FORCE_AGGREGATION,
                                  "sample_dt_s": .1, "source": "synthetic_measured_input_for_unit_test"},
                     "action": {"mode": "joint_position", "payload": {"arm_positions_rad": [0]*7}} if index < 12 else None,
                     "released": index >= 12, "geometry_clearance_m": .0005})
    return spec, rows


def run(spec, rows, **kwargs):
    evaluator = AssemblyRuntimeEvaluator(spec, physics_dt_s=.1, **kwargs)
    evaluator.reset("episode-1")
    return evaluator, [evaluator.record(**row) for row in rows]


def test_reward_requires_continuous_hold_and_released_stable_window(evidence):
    spec, rows = evidence
    evaluator, states = run(spec, rows)
    assert all(not state["task_success"] and state["reward"] == 0 for state in states[:17])
    assert states[-1]["evaluation_status"] == "passed"
    assert sum(state["reward"] for state in states) == 1
    assert evaluator.status()["reward"] == 0
    assert evaluator.finalize()["reward"] == 0
    with pytest.raises(RuntimeError, match="finalized"):
        evaluator.record(**rows[-1])


@pytest.mark.parametrize("fault", ["missing_velocity", "partial_contacts", "wrong_aggregation", "missing_part", "gap", "conflicting_time"])
def test_missing_evidence_cannot_be_repaired_by_later_goal_pose_and_reset_clears_it(evidence, fault):
    spec, rows = evidence
    corrupted = copy.deepcopy(rows)
    if fault == "missing_velocity":
        del corrupted[1]["objects"]["peg"]["linear_velocity_mps"]
    elif fault == "partial_contacts":
        corrupted[1]["coverage"]["complete"] = False
    elif fault == "wrong_aggregation":
        corrupted[1]["coverage"]["force_aggregation"] = "norm_of_net_force_vector"
    elif fault == "missing_part":
        corrupted[1]["coverage"]["object_ids"] = ["peg"]
    elif fault == "gap":
        del corrupted[1]
    else:
        corrupted[1]["sim_time"] = 0
    evaluator, states = run(spec, corrupted)
    assert states[-1]["evaluation_status"] == "not_verified"
    assert not any(state["task_success"] or state["reward"] for state in states)
    evaluator.reset("episode-2")
    assert evaluator.trace() == []
    recovered = [evaluator.record(**row, episode_id="episode-2") for row in rows]
    assert recovered[-1]["task_success"]


def test_duplicate_snapshot_is_idempotent_without_replaying_or_reward(evidence):
    evaluator, states = run(*evidence)
    duplicate = evaluator.record(**evidence[1][-1])
    assert duplicate["duplicate_sample"] and duplicate["reward"] == 0
    assert duplicate["sample_count"] == len(evidence[1])
    assert states[-1]["task_success"]


def test_point_normal_magnitudes_sum_before_force_gate_even_for_opposing_contacts(evidence):
    spec, rows = evidence
    rows[1]["contacts"] = [{"pair": ["socket", "peg"], "normal_force_N": 12},
                            {"pair": ["peg", "socket"], "normal_force_N": 12}]
    _, states = run(spec, rows, evaluation_interval_s=10)
    assert states[1]["evaluation_status"] == "failed"
    assert states[1]["assembly_result"]["classification"] == "overforce"
    assert not states[-1]["task_success"]


def test_declared_point_sum_cannot_conceal_peak_force(evidence):
    spec, rows = evidence
    rows[1]["contacts"][0]["normal_force_points_N"] = [12, 12]
    _, states = run(spec, rows)
    assert states[-1]["evaluation_status"] == "not_verified"
    assert "normal_force_is_not_sum_of_points" in states[-1]["failure_reason"]


@pytest.mark.parametrize("fault", ["initial_goal", "pose_write", "raw_velocity", "unstable_release", "unlisted_contact", "no_action", "no_fit_contact"])
def test_live_dynamics_and_actions_are_required_independently_of_final_pose(evidence, fault):
    spec, rows = evidence
    if fault == "initial_goal":
        for row in rows:
            row["objects"]["peg"]["position_m"] = [0, 0, -.02]
    elif fault == "pose_write":
        rows[5]["kinematic_pose_write"] = True
    elif fault == "raw_velocity":
        rows[5]["objects"]["peg"]["linear_velocity_mps"] = [1, 0, 0]
    elif fault == "unstable_release":
        for row in rows[12:]:
            row["objects"]["peg"]["linear_velocity_mps"] = [.02, 0, 0]
    elif fault == "unlisted_contact":
        rows[5]["contacts"] = [{"pair": ["socket", "intruder"], "normal_force_N": .01}]
    elif fault == "no_action":
        for row in rows:
            row["action"] = None
    else:
        for row in rows:
            row["contacts"] = []
    evaluator, states = run(spec, rows)
    assert not any(state["reward"] for state in states)
    assert not evaluator.finalize()["task_success"]


def test_scope_reset_deadline_and_trace_are_explicit(evidence):
    spec, rows = evidence
    evaluator = AssemblyRuntimeEvaluator(spec, physics_dt_s=.1)
    assert evaluator.record(**rows[0])["evaluation_status"] == "not_verified"
    evaluator.reset("episode-1")
    assert evaluator.record(**rows[0], episode_id="other")["failure_reason"] == "episode_id_mismatch"
    evaluator.reset("episode-2")
    for index in range(51):
        row = copy.deepcopy(rows[0])
        row["sim_time"] = index*.1
        state = evaluator.record(**row)
    assert state["truncated"] and not state["task_success"]
    trace = evaluator.trace()
    trace[0]["parts"]["peg"]["position_m"][0] = 100
    assert evaluator.trace()[0]["parts"]["peg"]["position_m"][0] == 0
    json.dumps(state, allow_nan=False)


def test_runtime_module_does_not_import_a_simulator():
    result = subprocess.run([sys.executable, "-c", "import sys; import scene_factory.assembly_runtime; assert not any(n in sys.modules for n in ('numpy','mujoco','isaacsim','pxr'))"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
