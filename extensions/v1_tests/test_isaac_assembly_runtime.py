"""Adapter contract tests use measured-state fixtures; these are not GPU trials."""
from __future__ import annotations

import copy
import json

import pytest

from scene_factory.backends.isaac import IsaacSimBackend


def spec():
    return {"schema_version": "scene_factory.assembly_spec.v1", "spec_id": "adapter-test",
            "peg_id": "peg", "socket_id": "socket", "peg_interface_position_m": [0, 0, 0],
            "socket_interface_position_m": [0, 0, 0], "insertion_axis_socket": [0, 0, -1],
            "target_depth_m": .02, "clearance_m": [.0001, .001], "position_tolerance_m": .001,
            "angular_tolerance_rad": .05, "max_force_N": 20, "min_hold_s": .5,
            "release_stable_s": .5, "timeout_s": 5, "jam_window_s": .5, "min_progress_m": .001,
            "max_linear_speed_mps": .5, "max_angular_speed_rps": 4, "drop_below_z_m": -.1,
            "allowed_contact_pairs": [["peg", "socket"], ["peg", "robot"]]}


def scene():
    return {"objects": [{"object_id": "peg"}, {"object_id": "socket"}],
            "task": {"target_object": "peg", "success": {"predicate": "assembly"},
                     "assembly_spec": spec(), "geometry_clearance_m": .0005}}


def observation(index):
    def state(pos):
        return {"position": pos, "orientation_wxyz": [1, 0, 0, 0],
                "linear_velocity_mps": [0, 0, 0], "angular_velocity_rps": [0, 0, 0]}
    return {"objects": {"peg": state([0, 0, -min(index*.002, .02)]), "socket": state([0, 0, 0])},
            "robot": {}, "contacts": [{"pair": ["peg", "socket"], "normal_force_N": 3}],
            "contact_coverage": {"complete": True, "continuous": True, "scope": "all_scene_contacts",
                                 "force_aggregation": "sum_per_point_normal_magnitudes",
                                 "sample_dt_s": .1, "object_ids": ["peg", "socket", "robot"],
                                 "source": "fixture_contract_only_not_a_physical_trial"}}


def test_inline_assembly_is_validated_before_runtime(tmp_path):
    backend = IsaacSimBackend(tmp_path / "unused.usd", physics_dt=.1)
    invalid = scene()
    invalid["task"]["geometry_clearance_m"] = True
    with pytest.raises(ValueError, match="geometry_clearance_m"):
        backend._configure_assembly_runtime(invalid)
    invalid = scene()
    invalid["task"]["target_object"] = "socket"
    with pytest.raises(ValueError, match="target_object"):
        backend._configure_assembly_runtime(invalid)
    invalid = scene()
    invalid["task"]["assembly_spec"]["unknown"] = "ignored constraints must fail"
    with pytest.raises(ValueError):
        backend._configure_assembly_runtime(invalid)


def test_backend_assembly_requires_measured_release_and_rewards_once(tmp_path):
    backend = IsaacSimBackend(tmp_path / "unused.usd", physics_dt=.1)
    backend._configure_assembly_runtime(scene())
    rewards = []
    for index in range(23):
        backend.steps = index
        opened = index >= 12
        backend._grasp_diagnostics = {"contact_force_read_valid": True,
                                      "finger_target_contact": 0 < index < 12,
                                      "gripper_open": opened}
        action = None if index == 0 else {"mode": "joint_position", "payload": {
            "arm_positions_rad": [0]*7, "gripper_open_fraction": 1. if opened else 0.}}
        item = observation(index)
        result = backend._record_assembly_step(item, action=action)
        rewards.append(result["reward"])
    assert item["task_success"] is True
    assert item["robot"]["released"] is True
    assert sum(rewards) == 1
    assert len(backend.assembly_trace()) == 23
    assert backend._assembly_runtime.status()["reward"] == 0
    # Reading observations/status is not allowed to create extra hold time.
    before = copy.deepcopy(backend.assembly_trace())
    backend._apply_assembly_status(item)
    assert backend.assembly_trace() == before


@pytest.mark.parametrize("missing", ["coverage", "prior_contact", "actual_open"])
def test_requested_open_or_final_pose_alone_cannot_pass(tmp_path, missing):
    backend = IsaacSimBackend(tmp_path / "unused.usd", physics_dt=.1)
    backend._configure_assembly_runtime(scene())
    for index in range(23):
        backend.steps = index
        opened = index >= 12
        backend._grasp_diagnostics = {"contact_force_read_valid": True,
                                      "finger_target_contact": missing != "prior_contact" and 0 < index < 12,
                                      "gripper_open": opened and missing != "actual_open"}
        action = None if index == 0 else {"mode": "joint_position", "payload": {
            "arm_positions_rad": [0]*7, "gripper_open_fraction": 1. if opened else 0.}}
        item = observation(index)
        if missing == "coverage":
            item.pop("contact_coverage")
        backend._record_assembly_step(item, action=action)
    assert item["task_success"] is False
    if missing == "coverage":
        assert item["assembly_status"]["evaluation_status"] == "not_verified"
        assert "contact_coverage_missing" in item["assembly_status"]["failure_reason"]


def test_missing_per_step_evidence_cannot_be_recovered_by_later_perfect_pose(tmp_path):
    backend = IsaacSimBackend(tmp_path / "unused.usd", physics_dt=.1)
    backend._configure_assembly_runtime(scene())
    backend._record_assembly_step(observation(0), action=None)
    backend.steps = 2
    result = backend._record_assembly_step(observation(2), action=None)
    assert result["evaluation_status"] == "not_verified"
    assert "physics_step_evidence_gap" in result["failure_reason"]


def test_last_allowed_step_preserves_success_found_during_finalization(tmp_path):
    backend = IsaacSimBackend(tmp_path / "unused.usd", max_steps=1)

    class Evaluator:
        def finalize(self):
            return {"task_success": True, "reward": 1., "terminated": True,
                    "truncated": False, "failure_reason": None}

    backend._assembly_runtime = Evaluator()
    backend.steps = 1
    pending = {"task_success": False, "reward": 0., "terminated": False, "truncated": False}
    backend._task_status = pending
    item = {"robot": {}}
    assert backend._external_step_outcome(item, pending) == (1., True, False)
    assert item["task_success"] is True
    assert item["assembly_status"]["task_success"] is True
    assert backend._external_done


def test_session_binds_initial_evidence_to_public_episode_id(tmp_path):
    from scene_factory.agent import SceneFactoryEnv
    from scene_factory.session import RuntimeSession

    class ContractBackend(IsaacSimBackend):
        def reset(self, scene):
            self.scene = scene
            self._configure_assembly_runtime(scene)
            self._last_observation = observation(0)
            self._record_assembly_step(self._last_observation, action=None)
            return self._last_observation, {}

        def close(self):
            pass

    payload = scene()
    payload["scene_id"] = "episode-binding"
    source = tmp_path / "layout.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    backend = ContractBackend(tmp_path / "unused.usd", physics_dt=.1)
    session = RuntimeSession(SceneFactoryEnv(source, backend))
    request = {"seed": 0, "options": {}, "idempotency_key": "initial"}
    response = session.reset(request)
    assert response["observation"]["assembly_status"]["episode_id"] == response["episode_id"]
    assert backend._assembly_runtime.episode_id == response["episode_id"]
    assert len(backend.assembly_trace()) == 1
    assert session.reset(request) == response
    backend.steps = 1
    with pytest.raises(RuntimeError, match="before its first action"):
        backend.bind_episode_id("another", response["observation"])


def test_requested_finger_drive_parameters_are_applied_and_read_back():
    import numpy as np
    from scene_factory.backends.isaac import _configure_franka_runtime_drives

    class View:
        kps, kds, forces = np.zeros((1, 9)), np.zeros((1, 9)), np.zeros((1, 9))

        def set_gains(self, *, kps, kds, joint_indices):
            self.kps[:, joint_indices] = kps
            self.kds[:, joint_indices] = kds

        def set_max_efforts(self, *, values, joint_indices):
            self.forces[:, joint_indices] = values

        def get_gains(self, *, joint_indices):
            return self.kps[:, joint_indices], self.kds[:, joint_indices]

        def get_max_efforts(self, *, joint_indices):
            return self.forces[:, joint_indices]

    class Robot:
        _articulation_view = View()

    observed = _configure_franka_runtime_drives(
        Robot(), [7, 8], {"finger_stiffness": 500., "finger_damping": 20., "finger_max_force": 20.})
    assert observed["verified"]
    assert observed["fingers"]["stiffness"] == [500., 500.]
    assert observed["fingers"]["damping"] == observed["fingers"]["max_force"] == [20., 20.]
    assert observed["arm"]["max_force"] == [2000.]*7


def test_online_controller_stops_on_success_or_failed_terminal_response():
    from tools.run_precision_assembly import OnlineAssemblyTerminal, check_online_terminal

    check_online_terminal({"terminated": False, "truncated": False})
    success = {"terminated": True, "truncated": False, "observation": {"task_success": True}, "reward": 1}
    with pytest.raises(OnlineAssemblyTerminal) as ended:
        check_online_terminal(success)
    assert ended.value.success and ended.value.response is success
    with pytest.raises(OnlineAssemblyTerminal) as failed:
        check_online_terminal({"terminated": False, "truncated": True, "observation": {"task_success": False}})
    assert not failed.value.success


def test_optional_solver_iterations_preserve_defaults_and_verify_runtime_readback():
    from scene_factory.backends.isaac import _configure_franka_solver_iterations

    class Robot:
        position, velocity = 32, 4
        calls = []

        def set_solver_position_iteration_count(self, value):
            self.calls.append(("position", value))
            self.position = value

        def set_solver_velocity_iteration_count(self, value):
            self.calls.append(("velocity", value))
            self.velocity = value

        def get_solver_position_iteration_count(self):
            return self.position

        def get_solver_velocity_iteration_count(self):
            return self.velocity

    robot = Robot()
    defaults = _configure_franka_solver_iterations(robot)
    assert not robot.calls
    assert defaults["observed"] == {"position": 32, "velocity": 4}
    assert defaults["requested"] == {"position": None, "velocity": None}
    requested = {"solver_position_iterations": 64, "solver_velocity_iterations": 16}
    applied = _configure_franka_solver_iterations(robot, requested)
    assert applied["verified"] and applied["observed"] == {"position": 64, "velocity": 16}
    robot.get_solver_velocity_iteration_count = lambda: 4
    with pytest.raises(RuntimeError, match="differ from requested"):
        _configure_franka_solver_iterations(robot, requested)
    assert not _configure_franka_solver_iterations(object())["verified"]
    with pytest.raises(RuntimeError, match="cannot configure/verify"):
        _configure_franka_solver_iterations(object(), requested)


@pytest.mark.parametrize("general", [False, True])
def test_general_finger_contact_sensor_preserves_authored_solver_settings(tmp_path, monkeypatch, general):
    import sys
    from types import ModuleType, SimpleNamespace

    captured = []

    class View:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def initialize(self, physics_view):
            pass

    class Prim:
        def __init__(self, path):
            self.path = path

        def IsValid(self):
            return True

        def GetPath(self):
            return self.path

    modules = {name: ModuleType(name) for name in (
        "omni", "omni.physics", "omni.physics.core", "pxr", "isaacsim", "isaacsim.core",
        "isaacsim.core.api", "isaacsim.core.api.sensors")}
    modules["omni.physics.core"].get_physics_simulation_interface = lambda: SimpleNamespace(
        subscribe_physics_contact_report_events=lambda callback: object())
    modules["pxr"].PhysxSchema = SimpleNamespace(PhysxContactReportAPI=SimpleNamespace(
        Apply=lambda prim: SimpleNamespace(CreateThresholdAttr=lambda: SimpleNamespace(Set=lambda value: None))))
    modules["isaacsim.core.api.sensors"].RigidContactView = View
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    backend = IsaacSimBackend(tmp_path / "unused.usd", enable_general_observation=general)
    backend.scene = {"task": {"target_object": "peg"}}
    backend._object_prims = {"peg": Prim("/peg")}
    backend._target_root_path = "/peg"
    backend._simulation = SimpleNamespace(physics_sim_view=object())
    backend._initialize_contact_reporting(SimpleNamespace(GetPrimAtPath=Prim))
    assert backend._grasp_diagnostics["contact_report_subscribed"]
    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["prepare_contact_sensors"] is (not general)
        assert kwargs["disable_stablization"] is (not general)
    assert backend.runtime_summary["finger_contact_sensor_settings"]["mode"] == (
        "preserve_authored_settings" if general else "legacy_sensor_defaults")
