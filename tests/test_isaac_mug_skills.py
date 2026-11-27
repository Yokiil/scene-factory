"""Pure tests of the boundary between policy decisions and physical drive commands."""

import pytest

from scene_factory.backends.isaac_mug_skills import MugSkillBridge
from scene_factory.robotics import MugLiftController, MugLiftPhase


ORIGIN = (0.56, 0.0, 0.965)
TARGET = (0.562, -0.004, 0.964)
VALID = {
    flag: True
    for flag in (
        "dof_limits_available",
        "dof_limits_valid",
        "all_finger_positions_within_limits",
        "contact_report_available",
        "contact_report_subscribed",
        "contact_force_read_valid",
        "finger_material_resolved",
        "target_material_resolution",
    )
}
DETACHED = {**VALID, "finger_target_contact": False, "gripper_open": False}
SUPPORTED = {
    "contact_coverage": {"complete": True},
    "contacts": [{"pair": ["island_1", "mug_1"], "normal_force_N": 0.88}],
}


def failed_probe():
    controller = MugLiftController(ORIGIN, max_steps=1000)
    bridge = MugSkillBridge(controller, "lateral", "island_1")
    assert controller.grasp_anchor_position == pytest.approx(
        (ORIGIN[0] - 0.15, -0.16, ORIGIN[2] + 0.005)
    )
    for step in range(1, 12):
        bridge.observe(controller, TARGET, DETACHED, step, SUPPORTED, "mug_1")
    controller.fail("grasp_failure")
    bridge.observe(controller, TARGET, DETACHED, 12, SUPPORTED, "mug_1")
    return controller, bridge


def test_only_phase_matching_bounded_commands_are_accepted():
    controller = MugLiftController(ORIGIN, max_steps=20)
    bridge = MugSkillBridge(controller, "nominal", "island_1")
    bridge.validate({"mode": "mug_skill", "skill": "reach"}, controller, 0)
    for action in (
        {"mode": "mug_skill", "skill": "lift"},
        {"mode": "mug_skill", "skill": "reach", "goal": [0, 0, 0]},
    ):
        with pytest.raises(ValueError):
            bridge.validate(action, controller, 0)
    with pytest.raises(RuntimeError):
        bridge.validate({"mode": "mug_skill", "skill": "reach"}, controller, 20)


@pytest.mark.parametrize(
    "observation",
    [
        {},
        {**SUPPORTED, "contacts": []},
        {**SUPPORTED, "contact_coverage": {"complete": False}},
        {**SUPPORTED, "contacts": [{"pair": ["mug_1", "other"], "normal_force_N": 1.0}]},
        {
            **SUPPORTED,
            "contacts": [{"pair": ["mug_1", "island_1"], "normal_force_N": float("nan")}],
        },
    ],
)
def test_failed_grasp_requires_present_measured_support(observation):
    controller, bridge = failed_probe()
    assert bridge.snapshot()["can_recover"]
    bridge.observe(controller, TARGET, DETACHED, 13, observation, "mug_1")
    with pytest.raises(RuntimeError):
        bridge.validate({"mode": "mug_skill", "skill": "recover"}, controller, 13)


def test_retry_requires_open_then_retreat_and_preserves_episode_baseline():
    controller, bridge = failed_probe()
    bridge.validate({"mode": "mug_skill", "skill": "recover"}, controller, 12)
    ee = (0.56, -0.11, 1.00)
    command = bridge.recovery_command(ee, TARGET)
    assert command.gripper == "recover_open" and command.goal_position is None
    bridge.advance_recovery(controller, TARGET, ee, 0, DETACHED, 13, None)
    assert bridge.recovery_stage == "open"
    opened = {**DETACHED, "gripper_open": True}
    bridge.advance_recovery(controller, TARGET, ee, 0, opened, 14, None)
    retreat = bridge.recovery_command(ee, TARGET)
    assert retreat.goal_position[:2] == ee[:2] and retreat.goal_position[2] > ee[2]
    assert retreat.goal_position[2] >= bridge.retreat_goal[2]
    controller = bridge.advance_recovery(
        controller, TARGET, bridge.retreat_goal, 0, opened, 15, True
    )
    assert bridge.recovery_stage == "home" and bridge.attempt == 1
    controller = bridge.advance_recovery(
        controller, TARGET, bridge.retreat_goal, 0, opened, 16, None, home_ready=True
    )
    assert controller.phase == MugLiftPhase.PRE_GRASP
    assert controller.initial_target_position == ORIGIN
    assert controller.grasp_anchor_position == TARGET
    assert controller.total_steps == 16 and controller.max_steps == 1000
    assert (bridge.candidate, bridge.attempt) == ("nominal", 2)
    controller.fail("grasp_failure")
    bridge.observe(controller, TARGET, opened, 16, SUPPORTED, "mug_1")
    assert bridge.can_recover is False


def test_grip_feedback_slews_into_contact_and_unloads():
    controller = MugLiftController(ORIGIN)
    bridge = MugSkillBridge(controller, "nominal", "island_1")
    obs = {
        "contact_coverage": {"complete": True},
        "contacts": [],
        "robot": {
            "grasp_diagnostics": {
                "contact_force_read_valid": True,
                "finger_dofs": [{"position": 0.026, "lower": 0, "upper": 0.04}] * 2,
            }
        },
    }
    targets = bridge.regulated_fingers(obs, "mug_1", 1000, 1 / 60)
    assert targets == pytest.approx([0.0194, 0.0209])
    obs["contacts"].append({"pair": ["mug_1", "robot"], "normal_force_N": 26})
    relieved = bridge.regulated_fingers(obs, "mug_1", 1000, 1 / 60)
    assert relieved == pytest.approx([0.0264, 0.0264])
    obs["contact_coverage"]["complete"] = False
    with pytest.raises(ValueError):
        bridge.regulated_fingers(obs, "mug_1", 1000, 1 / 60)
