from __future__ import annotations

import copy
import json

import pytest

from scene_factory.grasp_diagnostics import summarize_grasp_phases


def row(time_s, *, phase="close_gripper", height=1., angular=0., left=.2, right=.3):
    return {"time_s": time_s, "phase": phase,
            "live_peg": {"position": [0, 0, height], "linear_velocity_mps": [0., 0., .01],
                         "angular_velocity_rps": [0., 0., angular]},
            "end_effector_position_m": [0., 0., height+.04],
            "contacts": [{"pair": ["peg_1", "robot"], "normal_force_N": left+right}],
            "shape_contacts": [
                {"collider": "/World/Robot/panda_leftfinger/geometry", "normal_force_N": left},
                {"collider": "/World/Robot/panda_rightfinger/geometry", "normal_force_N": right}],
            "grasp_contact_coverage_complete": True}


def test_raw_spike_and_partial_bilateral_contact_are_preserved_without_mutation():
    rows = [row(0.), row(.1, height=1.001, angular=2.4), row(.2, height=1.002, left=0.)]
    rows[-1]["grasp_control"] = {"status": "failed", "reason": "grasp_angular_speed_protection"}
    before = copy.deepcopy(rows)
    summary = summarize_grasp_phases(rows)
    phase = summary["phases"][0]
    assert phase["max_angular_speed_rps"] == 2.4
    assert phase["max_pair_normal_force_N"] == .5
    assert phase["min_bilateral_finger_force_N"] == 0.
    assert phase["bilateral_contact_fraction_of_measured_samples"] == pytest.approx(2/3)
    assert phase["peg_rise_m"] == pytest.approx(.002)
    assert phase["tcp_rise_m"] == pytest.approx(.002)
    assert phase["first_recorded_event"]["sample_index"] == 2
    assert rows == before
    assert "acceptance" not in summary and "success" not in summary
    json.dumps(summary, allow_nan=False)


def test_missing_measurements_are_not_reported_as_zero_or_full_coverage():
    rows = [row(0.), row(.1)]
    for item in rows:
        item.pop("shape_contacts")
        item.pop("contacts")
        item.pop("grasp_contact_coverage_complete")
        item["live_peg"]["angular_velocity_rps"] = [0., float("nan"), 0.]
    rows[-1]["live_peg"].pop("position")
    phase = summarize_grasp_phases(rows)["phases"][0]
    assert phase["max_angular_speed_rps"] is None
    assert phase["angular_velocity_sample_count"] == 0
    assert phase["max_pair_normal_force_N"] is None
    assert phase["min_bilateral_finger_force_N"] is None
    assert phase["bilateral_contact_fraction_of_measured_samples"] is None
    assert phase["missing_shape_contact_sample_count"] == 2
    assert phase["contact_coverage_attested_count"] == 0
    assert phase["peg_rise_m"] is None
    json.dumps(phase, allow_nan=False)


def test_revisited_phases_are_separate_and_output_is_bounded():
    rows = [row(0.), row(.1, phase="lift"), row(.2), row(.3)]
    summary = summarize_grasp_phases(rows, max_phases=2)
    assert summary["phase_segment_count"] == 3
    assert summary["omitted_phase_segment_count"] == 1
    assert [p["sample_count"] for p in summary["phases"]] == [1, 1]
    assert summarize_grasp_phases([])["phases"] == []


def test_invalid_time_does_not_produce_a_plausible_duration():
    phase = summarize_grasp_phases([row(.2), row(.1)])["phases"][0]
    assert not phase["strictly_increasing_time"]
    assert phase["duration_s"] is None


@pytest.mark.parametrize("value", [0., -1., float("nan"), True])
def test_invalid_contact_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="contact_force_N"):
        summarize_grasp_phases([], contact_force_N=value)


def test_terminal_decision_is_bound_to_its_input_measurement_without_next_action():
    rows = [row(0.), row(.1), row(.2, phase="light_grasp_clear_table", angular=.8)]
    decisions = [{"input_time_s": .1, "status": "running", "reason": None},
                 {"input_time_s": .2, "status": "failed", "reason": "grasp_angular_speed_protection"}]
    original_rows, original_decisions = copy.deepcopy(rows), copy.deepcopy(decisions)
    summary = summarize_grasp_phases(rows, control_decisions=decisions)
    assert summary["phases"][0]["first_recorded_event"] is None
    assert summary["phases"][1]["first_recorded_event"] == {
        "sample_index": 2, "time_s": .2, "input_time_s": .2,
        "source_field": "control_decisions", "decision_index": 1,
        "status": "failed", "reason": "grasp_angular_speed_protection"}
    assert summary["control_decision_events"]["bound_count"] == 1
    assert rows == original_rows and decisions == original_decisions
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("input_time", [.099999999999, .100000000001, .3])
def test_unmatched_event_timestamp_is_never_rounded_or_attributed(input_time):
    summary = summarize_grasp_phases([row(0.), row(.1)], control_decisions=[
        {"input_time_s": input_time, "status": "failed", "reason": "protection"}])
    assert summary["phases"][0]["first_recorded_event"] is None
    assert summary["control_decision_events"]["bound_count"] == 0
    assert summary["control_decision_events"]["unmatched_count"] == 1


@pytest.mark.parametrize("input_time", [None, True, -.1, float("nan"), float("inf")])
def test_invalid_event_timestamp_cannot_attach_to_a_measurement(input_time):
    summary = summarize_grasp_phases([row(0.), row(.1)], control_decisions=[
        {"input_time_s": input_time, "status": "failed", "reason": "protection"}])
    assert summary["phases"][0]["first_recorded_event"] is None
    assert summary["control_decision_events"]["invalid_time_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_duplicate_measurement_times_are_ambiguous_for_external_decisions():
    summary = summarize_grasp_phases([row(.1), row(.1, phase="lift")], control_decisions=[
        {"input_time_s": .1, "status": "failed", "reason": "protection"}])
    assert all(phase["first_recorded_event"] is None for phase in summary["phases"])
    assert summary["control_decision_events"]["ambiguous_count"] == 1


def test_previous_action_decision_is_not_misattributed_to_next_measurement():
    rows = [row(.1), row(.2, phase="lift")]
    decision = {"input_time_s": .1, "status": "failed", "reason": "protection"}
    rows[1]["grasp_control"] = decision
    summary = summarize_grasp_phases(rows, control_decisions=[decision])
    assert summary["phases"][0]["first_recorded_event"]["sample_index"] == 0
    assert summary["phases"][1]["first_recorded_event"] is None


def test_first_event_uses_measurement_order_even_if_decisions_are_reordered():
    summary = summarize_grasp_phases([row(.1), row(.2)], control_decisions=[
        {"input_time_s": .2, "reason": "later"}, {"input_time_s": .1, "reason": "earlier"}])
    assert summary["phases"][0]["first_recorded_event"]["reason"] == "earlier"
    assert summary["control_decision_events"]["bound_count"] == 2


def test_threshold_override_changes_contact_statistics():
    summary = summarize_grasp_phases([row(0., left=.1, right=.1)], contact_force_N=.2)
    assert summary["bilateral_contact_threshold_N"] == .2
    assert summary["phases"][0]["bilateral_contact_sample_count"] == 0


@pytest.mark.parametrize("decisions", ["bad", [None], {"input_time_s": .1}])
def test_invalid_decision_collection_rejected(decisions):
    with pytest.raises(ValueError, match="control_decisions"):
        summarize_grasp_phases([], control_decisions=decisions)
