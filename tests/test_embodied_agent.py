"""Policy contract tests with synthetic observations, independent of the Isaac gate."""

import pytest

from scene_factory.embodied_agent import EmbodiedAgentRuntime, SKILLS


def frame(
    step=0,
    phase="PRE_GRASP",
    *,
    height=1.0,
    x=0.0,
    held=False,
    released=False,
    supported=True,
    candidate="nominal",
    attempt=1,
):
    mug = [x, 0.0, height]
    diagnostics = {
        key: True
        for key in (
            "dof_limits_valid",
            "all_finger_positions_within_limits",
            "contact_report_available",
            "contact_report_subscribed",
            "contact_force_read_valid",
            "finger_material_resolved",
            "target_material_resolution",
        )
    }
    diagnostics.update(finger_target_contact=held, gripper_open=not held)
    oracle = dict(
        pick_success=True,
        holding=held,
        released=released,
        placement_stable=released,
        in_target_region=x == 0.3,
        task_success=released,
    )
    spec = dict(
        predicate="pick_and_place",
        subject="mug_1",
        source_support="island_1",
        target_support="island_1",
        target_position_m=[0.3, 0, 1],
        target_tolerance_m=[0.02, 0.02, 0.02],
        min_lift_delta_m=0.10,
        max_settle_step_distance_m=0.005,
        settle_steps=3,
    )
    obj = dict(
        position_m=mug,
        orientation_wxyz=[1, 0, 0, 0],
        pose_source="PhysX tensor API",
        dynamic=True,
        physics_handle_valid=True,
        asset_id="mug_001",
        instance={"bbox_m": [0.1, 0.08, 0.1]},
    )
    robot = dict(
        name="franka",
        phase=phase,
        failure_reason=None,
        pick_status="passed",
        place_status="passed",
        released=released,
        end_effector_pose={"position": mug},
        finger_positions={"left": mug, "right": mug},
    )
    cup_support = {"pair": ["mug_1", "island_1"], "normal_force_N": 2.0, "units": "N"}
    return {
        "schema_version": "scene_factory.world_state.v1",
        "backend": "isaac",
        "source": "runtime_observation",
        "step_index": step,
        "sim_time": step / 60.0,
        "state_version": step + 45,
        "episode_id": "a",
        "robot": robot,
        "objects": {
            "mug_1": obj,
            "island_1": dict(
                position_m=[0.0, 0.0, 0.0],
                orientation_wxyz=[1, 0, 0, 0],
                pose_source="USD static transform",
            ),
        },
        "task": {"target_object": "mug_1", "success": spec},
        "mug_skill": {
            "attempt": attempt,
            "candidate": candidate,
            "last_failure": None,
            "can_recover": False,
            "recovery_stage": None,
        },
        "grasp_diagnostics": diagnostics,
        "task_oracle": oracle,
        "contacts": [cup_support] if supported else [],
        "contact_coverage": dict(
            scope="all_scene_contacts",
            complete=True,
            continuous=True,
            force_aggregation="sum_per_point_normal_magnitudes",
            object_ids=["mug_1", "robot", "island_1"],
        ),
    }


def test_named_skills_and_abort():
    runtime = EmbodiedAgentRuntime()
    for name in (
        "observe",
        "detect_object",
        "estimate_graspability",
        "grasp",
        "verify_grasp",
        "place",
        "verify_place",
        "recover",
        "abort",
    ):
        assert name in SKILLS and callable(getattr(runtime, name))
    assert runtime.next_skill(frame()) == "reach"
    assert runtime.abort("operator_stop") is None
    assert runtime.next_skill(frame(1)) is None
    assert runtime.reason == "operator_stop"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: w.update(backend="mujoco"),
        lambda w: w["objects"]["mug_1"].update(physics_handle_valid=False),
        lambda w: w["objects"]["mug_1"].update(orientation_wxyz=None),
        lambda w: w.update(contacts=None),
        lambda w: w["contact_coverage"].update(complete=False),
        lambda w: w["grasp_diagnostics"].pop("contact_force_read_valid"),
    ],
)
def test_missing_evidence_latches_abort(mutate):
    runtime = EmbodiedAgentRuntime()
    world = frame()
    mutate(world)
    assert runtime.next_skill(world) is None and runtime.result == "not_verified"
    assert runtime.next_skill(frame(1)) is None


@pytest.mark.parametrize(
    "changed", [{"episode_id": "b"}, {"state_version": 45}, {"step_index": 2}, {"sim_time": 0.0}]
)
def test_episode_clock_and_version(changed):
    runtime = EmbodiedAgentRuntime()
    runtime.next_skill(frame())
    world = frame(1)
    world.update(changed)
    assert runtime.next_skill(world) is None and runtime.result == "not_verified"


def test_overforce_combines_all_points_of_same_pair():
    runtime = EmbodiedAgentRuntime()
    world = frame()
    world["contacts"] = [
        {"pair": ["mug_1", "robot"], "normal_force_N": 20.0, "units": "N"},
        {"pair": ["robot", "mug_1"], "normal_force_N": 11.0, "units": "N"},
    ]
    assert runtime.next_skill(world) is None and runtime.reason == "measured_target_overforce"
    with pytest.raises(ValueError):
        EmbodiedAgentRuntime(max_contact_force_N=31)


def test_recover_only_supported_missed_candidate_then_new_attempt():
    runtime = EmbodiedAgentRuntime()
    runtime.next_skill(frame(candidate="lateral"))
    miss = frame(1, "FAILED", candidate="lateral")
    miss["mug_skill"].update(can_recover=True, last_failure="approach_timeout")
    miss["robot"]["failure_reason"] = "approach_timeout"
    assert runtime.next_skill(miss) == "recover"
    assert runtime.next_skill(frame(2, attempt=2)) == "reach"
    miss = frame(3, "FAILED", attempt=2)
    miss["mug_skill"].update(can_recover=True, last_failure="grasp_failure")
    miss["robot"]["failure_reason"] = "grasp_failure"
    assert runtime.next_skill(miss) is None and runtime.result == "failed"


@pytest.mark.parametrize("supported,expected", [(True, "success"), (False, "not_verified")])
def test_verified_grasp_transport_release_and_continuous_support(supported, expected):
    runtime = EmbodiedAgentRuntime()
    sequence = [
        ("PRE_GRASP", {}),
        ("APPROACH", {}),
        ("GRASP", {}),
        *([("VERIFY_GRASP", dict(held=True))] * 10),
        ("LIFT", dict(height=1.01, held=True)),
        ("TRANSFER", dict(height=1.12, held=True)),
        ("LOWER", dict(x=0.3, height=1.12, held=True)),
        ("RELEASE", dict(x=0.3, held=True)),
        *([("RELEASE", dict(x=0.3, released=True, supported=supported))] * 3),
        ("VERIFY_PLACE", dict(x=0.3, released=True, supported=supported)),
        ("DONE", dict(x=0.3, released=True, supported=supported)),
    ]
    for n, (phase, kwargs) in enumerate(sequence):
        skill = runtime.next_skill(frame(n, phase, **kwargs))
        if phase != "DONE":
            assert skill in SKILLS
    assert skill is None and runtime.result == expected
