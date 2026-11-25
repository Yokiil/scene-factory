"""Synthetic sensor contracts for grasp control; these are not physics acceptance."""
from __future__ import annotations

import copy
import math

import pytest

from scene_factory.grasp_control import ContactGraspController, GraspControlConfig, anchored_joint_target


def measurement(time_s, *, left=.6, right=.6, table=.78, peg=(0., 0., 1.),
                tcp=(0., 0., 1.05), orientation=(1., 0., 0., 0.),
                linear=(0., 0., 0.), angular=(0., 0., 0.), fingers=(.01, .01)):
    contacts = []
    for part, force in (("panda_leftfinger", left), ("panda_rightfinger", right)):
        if force:
            contacts.append({"collider": f"/World/Robot/{part}/geometry/{part}",
                             "normal_force_N": force})
    if table:
        contacts.append({"collider": "/World/Objects/table_1", "normal_force_N": table})
    return {"time_s": time_s, "grasp_contact_coverage_complete": True,
            "live_peg": {"position": list(peg), "linear_velocity_mps": list(linear),
                         "angular_velocity_rps": list(angular)},
            "end_effector_position_m": list(tcp),
            "end_effector_orientation_wxyz": list(orientation),
            "finger_positions_m": list(fingers), "shape_contacts": contacts}


def acquire(controller, *, peg=(0., 0., 1.), tcp=(0., 0., 1.05)):
    for tick in range(100):
        result = controller.update(measurement(tick * controller.dt, peg=peg, tcp=tcp))
        if result["phase"] == "light_grasp_clear_table":
            return tick
    pytest.fail("valid sustained bilateral contact never permitted a micro-lift")


def test_complete_measured_grasp_prefix_requires_contact_clearance_and_force_hold():
    # A deterministic ideal sensor fixture, including spring-like compression.
    # This tests controller reactions; it does not establish real grasp success.
    dt = .01
    controller = ContactGraspController(physics_dt_s=dt)
    decision = controller.decision()
    phases = [decision["phase"]]
    for tick in range(1600):
        commanded_finger = decision["gripper_open_fraction"] * .04
        actual_finger = max(.01, commanded_finger)
        force = max(0., (.01 - commanded_finger) * 1000.)
        rise = decision["lift_offset_m"]
        decision = controller.update(measurement(
            tick * dt, left=force, right=force, table=.78 if rise < .0005 else 0.,
            peg=(0., 0., 1. + rise), tcp=(0., 0., 1.05 + rise),
            fingers=(actual_finger, actual_finger)))
        if phases[-1] != decision["phase"]:
            phases.append(decision["phase"])
        if decision["status"] != "running":
            break
    assert decision["status"] == "ready", decision
    assert phases == ["close_gripper", "light_grasp_clear_table",
                      "increase_grip_off_table", "grasp_stable"]
    assert decision["metrics"]["measured_rise_m"] >= controller.config.clear_height_m
    assert decision["metrics"]["table_force_N"] == 0.
    assert "assembly_acceptance" not in decision and "task_success" not in decision


@pytest.mark.parametrize("forces", [(0., 0.), (.6, 0.), (0., .6)])
def test_missing_or_unilateral_contact_times_out_without_lifting(forces):
    controller = ContactGraspController(physics_dt_s=.01, config=GraspControlConfig(stage_timeout_s=.3))
    for tick in range(32):
        result = controller.update(measurement(tick * .01, left=forces[0], right=forces[1]))
    assert result["status"] == "failed"
    assert result["reason"] == "grasp_stage_timeout:close_gripper"
    assert result["lift_offset_m"] == 0.


def test_interrupted_bilateral_contact_cannot_accumulate_stable_time():
    controller = ContactGraspController(physics_dt_s=.01)
    for tick in range(41):
        result = controller.update(measurement(tick * .01, right=0. if tick == 19 else .6))
        if tick < 40:
            assert result["phase"] == "close_gripper"
    assert result["phase"] == "light_grasp_clear_table"


@pytest.mark.parametrize("still_touching_table", [False, True])
def test_micro_lift_needs_actual_peg_rise_and_no_table_contact(still_touching_table):
    controller = ContactGraspController(physics_dt_s=.01)
    tick = acquire(controller)
    for offset in range(1, 80):
        rise = .0006 if still_touching_table else 0.
        result = controller.update(measurement((tick + offset) * .01,
            table=.78 if still_touching_table else 0.,
            peg=(0., 0., 1. + rise), tcp=(0., 0., 1.05 + rise)))
        assert result["phase"] == "light_grasp_clear_table"
        assert result["lift_offset_m"] <= controller.config.max_lift_m


def test_tcp_motion_alone_cannot_prove_peg_clearance():
    controller = ContactGraspController(physics_dt_s=.01)
    tick = acquire(controller)
    result = controller.update(measurement((tick + 1) * .01, table=0., tcp=(0., 0., 1.051)))
    assert result["status"] == "failed"
    assert result["reason"] == "grasp_relative_slip"


def test_contact_dropout_pauses_lift_then_latches_failure():
    controller = ContactGraspController(physics_dt_s=.01)
    tick = acquire(controller)
    result = controller.update(measurement((tick + 1) * .01))
    before = result["lift_offset_m"]
    for offset in range(2, 9):
        result = controller.update(measurement((tick + offset) * .01, right=0.))
        assert result["lift_offset_m"] == before
    assert result["status"] == "failed" and result["reason"] == "grasp_contact_lost"
    assert controller.update(measurement((tick + 9) * .01)) == result


def test_relative_grasp_offset_is_in_tcp_frame():
    controller = ContactGraspController(physics_dt_s=.01)
    tick = acquire(controller, peg=(.02, .01, 1.))
    # Rigidly translating and rotating both parts leaves the relative grasp unchanged.
    quaternion = (math.sqrt(.5), 0., 0., math.sqrt(.5))
    result = controller.update(measurement((tick + 1) * .01, tcp=(.001, .002, 1.05),
        peg=(-.009, .022, 1.), orientation=quaternion))
    assert result["status"] == "running"
    assert result["metrics"]["relative_slip_m"] < 1e-12
    result = controller.update(measurement((tick + 2) * .01, tcp=(.001, .002, 1.05),
        peg=(-.009, .022, .9994), orientation=quaternion))
    assert result["status"] == "failed" and result["reason"] == "grasp_relative_slip"


def test_raw_angular_spike_is_not_hidden_by_static_pose_and_failure_is_latched():
    controller = ContactGraspController(physics_dt_s=.01)
    sample = measurement(0., angular=(0., 0., .76))
    original = copy.deepcopy(sample)
    result = controller.update(sample)
    assert result["status"] == "failed" and result["reason"] == "grasp_angular_speed_protection"
    assert result["metrics"]["angular_speed_rps"] == .76
    assert sample == original
    assert controller.update(measurement(.01)) == result


@pytest.mark.parametrize("time_s", [0., -.01, .005, .02])
def test_duplicate_backward_or_gapped_samples_cannot_extend_stability(time_s):
    controller = ContactGraspController(physics_dt_s=.01)
    controller.update(measurement(0.))
    result = controller.update(measurement(time_s))
    assert result["status"] == "failed"
    assert "consecutive physics steps" in result["reason"]


@pytest.mark.parametrize("dt", [0., -.01, True, float("nan"), float("inf")])
def test_invalid_step_sizes_rejected(dt):
    with pytest.raises(ValueError):
        ContactGraspController(physics_dt_s=dt)


@pytest.mark.parametrize("kind", ["coverage_missing", "coverage_bool", "contact_missing",
    "negative_force", "boolean_force", "nan_velocity", "boolean_position", "bad_quaternion",
    "negative_time", "finger_missing"])
def test_invalid_measurements_fail_closed(kind):
    controller = ContactGraspController(physics_dt_s=.01)
    sample = measurement(0.)
    if kind == "coverage_missing":
        del sample["grasp_contact_coverage_complete"]
    elif kind == "coverage_bool":
        sample["grasp_contact_coverage_complete"] = 1
    elif kind == "contact_missing":
        del sample["shape_contacts"]
    elif kind == "negative_force":
        sample["shape_contacts"][0]["normal_force_N"] = -.1
    elif kind == "boolean_force":
        sample["shape_contacts"][0]["normal_force_N"] = True
    elif kind == "nan_velocity":
        sample["live_peg"]["linear_velocity_mps"][0] = float("nan")
    elif kind == "boolean_position":
        sample["live_peg"]["position"][0] = True
    elif kind == "bad_quaternion":
        sample["end_effector_orientation_wxyz"] = [2., 0., 0., 0.]
    elif kind == "negative_time":
        sample["time_s"] = -.01
    elif kind == "finger_missing":
        del sample["finger_positions_m"]
    result = controller.update(sample)
    assert result["status"] == "failed"
    assert result["reason"].startswith("invalid_grasp_measurement:")


def test_similarly_named_colliders_do_not_count_as_fingers_or_table():
    controller = ContactGraspController(physics_dt_s=.01)
    for tick in range(30):
        sample = measurement(tick * .01, left=0., right=0., table=0.)
        sample["shape_contacts"] = [
            {"collider": f"/World/Robot/{name}", "normal_force_N": .6}
            for name in ("panda_leftfinger_extra", "panda_rightfinger_extra", "table_10")]
        result = controller.update(sample)
    assert result["phase"] == "close_gripper"
    assert not result["metrics"]["bilateral_contact"]
    assert result["metrics"]["table_force_N"] == 0.


def test_command_speed_bounds_use_each_runtime_finger_stroke():
    closed, opened, dt = (.002, .004), (.022, .054), .01
    controller = ContactGraspController(physics_dt_s=dt, closed_positions=closed, open_positions=opened)
    previous = controller.decision()["gripper_open_fraction"]
    for tick in range(50):
        # Transition from free travel to one-side contact, then excess pressure.
        force = 0. if tick < 10 else .1 if tick < 30 else 2.
        sample = measurement(tick * dt, left=force, right=0., fingers=(.007, .0165))
        result = controller.update(sample)
        speed = controller.config.free_close_speed_mps if tick < 10 else controller.config.contact_close_speed_mps
        for a, b in zip(closed, opened, strict=True):
            assert abs(result["gripper_open_fraction"] - previous) * (b - a) <= speed * dt + 1e-14
        assert controller.minimum_fraction <= result["gripper_open_fraction"] <= 1.
        previous = result["gripper_open_fraction"]


@pytest.mark.parametrize("positions", [(-.001, .01), (.041, .01)])
def test_out_of_range_live_finger_positions_cannot_count_as_a_valid_grasp(positions):
    controller = ContactGraspController(physics_dt_s=.01)
    result = controller.update(measurement(0., fingers=positions))
    assert result["status"] == "failed"
    assert result["reason"].startswith("invalid_grasp_measurement:")


def test_high_imbalanced_force_must_settle_before_micro_lift():
    controller = ContactGraspController(physics_dt_s=.01)
    for tick in range(30):
        result = controller.update(measurement(tick * .01, left=4., right=.6))
    assert result["phase"] == "close_gripper"
    assert result["lift_offset_m"] == 0.


def test_airborne_force_target_does_not_bypass_lost_table_clearance():
    controller = ContactGraspController(physics_dt_s=.01)
    tick = acquire(controller)
    while controller.phase == "light_grasp_clear_table":
        tick += 1
        controller.update(measurement(tick * .01, table=0.,
            peg=(0., 0., 1.0006), tcp=(0., 0., 1.0506)))
    result = controller.update(measurement((tick + 1) * .01, left=1.2, right=1.2,
        table=.1, peg=(0., 0., 1.0006), tcp=(0., 0., 1.0506)))
    assert result["status"] == "failed" and result["reason"] == "grasp_table_clearance_lost"


def test_precontact_slowdown_reduces_target_lead_with_lagged_finger_measurements():
    # A first-order actuator lag illustrates the demand already stored when
    # contact arrives. This is a control regression, not a contact-physics model
    # or evidence that the corresponding Isaac task will pass.
    dt, lag_s, half_width = .01, .1, .01
    response = 1. - math.exp(-dt / lag_s)

    def first_contact(config):
        controller = ContactGraspController(physics_dt_s=dt, object_width_m=2 * half_width,
                                            config=config)
        actual = target = .04
        for tick in range(800):
            actual = max(half_width, actual + response * (target - actual))
            touching = actual == half_width
            lead = actual - target
            decision = controller.update(measurement(tick * dt, fingers=(actual, actual),
                left=.06 if touching else 0., right=.06 if touching else 0.))
            assert decision["status"] == "running", decision
            if touching:
                return lead, tick * dt
            target = .04 * decision["gripper_open_fraction"]
        pytest.fail("bounded closing never reached the measured object surface")

    slow_lead, slow_time = first_contact(GraspControlConfig())
    # Setting both pre-contact rates equal reproduces the historical free-rate
    # behavior until measured contact, with all other settings held fixed.
    fast_lead, fast_time = first_contact(GraspControlConfig(approach_close_speed_mps=.01))
    assert 0. < slow_lead < .0002
    assert fast_lead > .0008
    assert slow_lead < .25 * fast_lead
    assert slow_time > fast_time


def test_zero_ik_displacement_preserves_drive_compensation_for_static_sag():
    command = [0.2, -.3, .1, -1.9, .05, 2.1, -.7]
    # The measured stationary pose has a different IK solution from the
    # drive target maintaining it under load. Switching to that solution
    # would discard the compensation even when no motion was requested.
    stationary_ik = [.199, -.302, .1005, -1.897, .049, 2.098, -.7004]
    assert anchored_joint_target(command, stationary_ik, stationary_ik) == tuple(command)


def test_repeated_fixed_reference_displacements_do_not_integrate_joint_drift():
    command = [0.2, -.3, .1, -1.9, .05, 2.1, -.7]
    origin = [.199, -.302, .1005, -1.897, .049, 2.098, -.7004]
    target = [.1991, -.3022, .1003, -1.8967, .0492, 2.0976, -.7003]
    original = copy.deepcopy((command, origin, target))
    first = anchored_joint_target(command, origin, target)
    # Holding the same Cartesian offset repeatedly must produce one fixed
    # drive target, regardless of how many control cycles have elapsed.
    assert all(anchored_joint_target(command, origin, target) == first for _ in range(1000))
    # A closed excursion returns exactly to the original compensated target.
    assert anchored_joint_target(command, origin, origin) == tuple(command)
    assert (command, origin, target) == original


def test_common_ik_bias_cancels_without_changing_the_requested_displacement():
    command = [.2, -.3, .1, -1.9, .05, 2.1, -.7]
    origin = [.199, -.302, .1005, -1.897, .049, 2.098, -.7004]
    target = [.1991, -.3022, .1003, -1.8967, .0492, 2.0976, -.7003]
    reference = anchored_joint_target(command, origin, target)
    bias = [.001, -.002, .003, -.001, .002, -.003, .0005]
    shifted_origin = [value + correction for value, correction in zip(origin, bias, strict=True)]
    shifted_target = [value + correction for value, correction in zip(target, bias, strict=True)]
    assert anchored_joint_target(command, shifted_origin, shifted_target) == pytest.approx(reference, abs=1e-14)


@pytest.mark.parametrize("argument", range(3))
@pytest.mark.parametrize("invalid", [[0.] * 6, [0.] * 8, [True] + [0.] * 6,
    [float("nan")] + [0.] * 6, [float("inf")] + [0.] * 6, "invalid", None])
def test_anchor_rejects_invalid_joint_vectors(argument, invalid):
    values = [[0.] * 7 for _ in range(3)]
    values[argument] = invalid
    with pytest.raises(ValueError):
        anchored_joint_target(*values)


def test_anchor_rejects_finite_inputs_whose_displacement_overflows():
    with pytest.raises(ValueError, match="anchored target"):
        anchored_joint_target([1e308] * 7, [-1e308] * 7, [1e308] * 7)


def test_velocity_recovery_opens_at_bounded_rate_and_cannot_advance_lift():
    controller = ContactGraspController(physics_dt_s=.01,
        config=GraspControlConfig(recovery_open_speed_mps=.0004))
    decision = controller.decision()
    for tick in range(1200):
        finger = decision["gripper_open_fraction"]*.04
        force = max(0., (.01-finger)*1000.)
        decision = controller.update(measurement(tick*.01, left=force, right=force,
            fingers=(max(.01, finger), max(.01, finger))))
        if decision["phase"] == "light_grasp_clear_table":
            break
    assert decision["phase"] == "light_grasp_clear_table"
    original_lift = decision["lift_offset_m"]
    for _ in range(10):
        tick += 1
        before = decision["gripper_open_fraction"]
        decision = controller.update(measurement(tick*.01, angular=(0., 0., .6)))
        assert decision["status"] == "running" and decision["paused"]
        opening = (decision["gripper_open_fraction"]-before)*.04
        assert 0 < opening <= controller.config.recovery_open_speed_mps*.01+1e-12
        assert decision["lift_offset_m"] == original_lift
    for _ in range(11):
        tick += 1
        decision = controller.update(measurement(tick*.01))
        assert decision["lift_offset_m"] == original_lift
    assert not decision["paused"]
    tick += 1
    decision = controller.update(measurement(tick*.01))
    assert decision["lift_offset_m"] > original_lift


def test_velocity_recovery_timeout_latches_failure_without_masking_raw_speed():
    controller = ContactGraspController(physics_dt_s=.01)
    for tick in range(77):
        result = controller.update(measurement(tick*.01, angular=(0., 0., .6)))
    assert result["status"] == "failed"
    assert result["reason"] == "grasp_velocity_recovery_timeout"
    assert result["metrics"]["angular_speed_rps"] == .6
    assert controller.update(measurement(.77)) == result


def test_recovery_opening_cannot_exceed_contact_rate():
    with pytest.raises(ValueError, match="recovery opening"):
        GraspControlConfig(recovery_open_speed_mps=.0005)


def test_default_recovery_holds_grip_and_lift_targets():
    controller = ContactGraspController(physics_dt_s=.01)
    decision = controller.update(measurement(0., left=0., right=0., fingers=(.04, .04)))
    decision = controller.update(measurement(.01, left=0., right=0., fingers=(.04, .04)))
    before = decision["gripper_open_fraction"]
    for tick in range(2, 20):
        decision = controller.update(measurement(tick*.01, angular=(0., 0., .6)))
        assert decision["paused"]
        assert decision["gripper_open_fraction"] == before
        assert decision["lift_offset_m"] == 0.
