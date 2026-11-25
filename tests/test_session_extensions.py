from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scene_factory.agent import DryRunBackend, SceneFactoryEnv
from scene_factory.joint_control import ARM_JOINT_NAMES, resolve_joint_targets
from scene_factory.session import RuntimeSession


def write_scene(tmp_path):
    scene = {"scene_id": "extensions", "room_dimensions_m": [2, 2, 2], "objects": [{
        "object_id": "cube", "asset_id": "cube", "dynamic": True,
        "bbox_m": [0.1, 0.1, 0.1], "pose": {"position": [0, 0, 0.2], "yaw_deg": 0},
    }], "task": {"success": {"predicate": "lifted", "subject": "cube", "min_height_delta_m": 10}}}
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path


def step(session, key, steps=1):
    return session.step({"episode_id": session.episode_id, "sequence_id": session.describe()["next_sequence_id"],
                         "idempotency_key": key, "steps": steps, "action": {"mode": "noop"}})


def test_rebuild_snapshot_starts_new_episode_and_retry_does_not_reset(tmp_path):
    with RuntimeSession(SceneFactoryEnv(write_scene(tmp_path), DryRunBackend())) as session:
        initial = session.reset({"idempotency_key": "reset", "seed": 42})
        state = session.snapshot({"idempotency_key": "snapshot", "level": "rebuild"})
        step(session, "s", 3)
        restored = session.restore({"idempotency_key": "restore", "snapshot": state})
        assert restored["step_index"] == 0
        assert restored["episode_id"] != initial["episode_id"]
        assert restored["observation"] == initial["observation"]
        step(session, "new-s", 2)
        assert session.restore({"idempotency_key": "restore", "snapshot": state}) == restored
        assert session.observe()["step_index"] == 2


def test_bad_snapshot_checksum_leaves_valid_session_intact(tmp_path):
    with RuntimeSession(SceneFactoryEnv(write_scene(tmp_path), DryRunBackend())) as session:
        session.reset({"idempotency_key": "reset", "seed": 42})
        state = session.snapshot({"idempotency_key": "snapshot", "level": "rebuild"})
        before = session.observe()
        state["step_index"] = 100
        with pytest.raises(ValueError, match="digest"):
            session.restore({"idempotency_key": "restore", "snapshot": state})
        assert session.observe() == before


@pytest.mark.parametrize("changed", [{"step_index": 99}, {"terminated": True}, {"runtime_state": {}}])
def test_rebuild_rejects_semantically_invalid_metadata_even_with_valid_checksum(tmp_path, changed):
    from scene_factory.session import _digest
    with RuntimeSession(SceneFactoryEnv(write_scene(tmp_path), DryRunBackend())) as session:
        session.reset({"idempotency_key": "reset", "seed": 42})
        state = session.snapshot({"idempotency_key": "snapshot", "level": "rebuild"})
        before = session.observe()
        state.update(changed)
        state["sha256"] = _digest({k: v for k, v in state.items() if k != "sha256"})
        with pytest.raises(ValueError, match="initial episode"):
            session.restore({"idempotency_key": "restore", "snapshot": state})
        assert session.observe() == before


def test_mujoco_session_snapshot_continuation_and_image_state_alignment(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("mujoco")
    from scene_factory.backends.mujoco import MujocoBackend
    from scene_factory.models import AssetRecord
    from scene_factory.registry import AssetRegistry

    registry = AssetRegistry([AssetRecord("cube", "cube", (0.1, 0.1, 0.1))])
    backend = MujocoBackend(registry=registry, max_steps=100)
    with RuntimeSession(SceneFactoryEnv(write_scene(tmp_path), backend)) as session:
        session.reset({"idempotency_key": "reset", "seed": 3})
        step(session, "s0", 5)
        state = session.snapshot({"idempotency_key": "snap", "level": "runtime_state"})
        from scene_factory.session import _digest
        altered = copy.deepcopy(state)
        altered["step_index"] += 1
        altered["sha256"] = _digest({k: v for k, v in altered.items() if k != "sha256"})
        with pytest.raises(ValueError, match="metadata"):
            session.restore({"idempotency_key": "bad", "snapshot": altered})
        continuation = step(session, "s1", 5)
        restored = session.restore({"idempotency_key": "restore", "snapshot": state})
        assert restored["step_index"] == 5
        replay = step(session, "s2", 5)
        assert replay["observation"] == continuation["observation"]
        session.artifact_root = tmp_path / "images"
        backend.render = lambda: np.zeros((12, 16, 3), dtype=np.uint8)
        observed = session.observe({"modalities": ["state", "rgb"]})
        frame = observed["images"][0]
        data = (session.artifact_root / frame["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == frame["sha256"]
        assert frame["step_index"] == replay["step_index"]
        assert frame["state_version"] == replay["state_version"]
        assert frame["sim_time"] == replay["sim_time"]
        assert session.observe() == replay

        def advancing_failed_render():
            backend._data.time += 0.01
            raise ValueError("sensor failure")
        backend.render = advancing_failed_render
        with pytest.raises(RuntimeError, match="advanced simulation"):
            session.observe({"modalities": ["rgb"]})
        with pytest.raises(RuntimeError, match="reset is required"):
            session.observe()


def test_isaac_repeated_image_capture_allows_same_committed_state():
    np = pytest.importorskip("numpy")
    from types import SimpleNamespace
    from scene_factory.backends.isaac import IsaacSimBackend
    backend = IsaacSimBackend("unused.usd")
    backend.steps = 2
    backend._app = object()
    backend._simulation = SimpleNamespace(current_time=2/60)
    backend._camera = SimpleNamespace(
        get_rgb=lambda **kw: np.zeros((4, 6, 3), dtype=np.uint8),
        get_depth=lambda **kw: np.ones((4, 6), dtype=float),
        get_resolution=lambda: (6, 4),
        get_intrinsics_matrix=lambda **kw: np.eye(3),
        get_world_pose=lambda **kw: (np.zeros(3), np.asarray([1, 0, 0, 0])),
    )
    backend._run_rgbd_render_step = lambda: None
    backend._camera_reference_time = lambda: {"referenceTimeNumerator": 2, "referenceTimeDenominator": 60}
    first = backend.capture_rgbd_observation()
    second = backend.capture_rgbd_observation()
    assert first["rendering_frame"] == second["rendering_frame"]
    assert first["simulation_step"] == second["simulation_step"] == 2


def test_joint_targets_follow_names_and_asymmetric_runtime_finger_limits():
    names = list(ARM_JOINT_NAMES) + ["finger1", "finger2"]
    limits = [[-1, 1]] * 7 + [[0.001, 0.041], [0.002, 0.038]]
    config = {"indices": [7, 8], "closed_positions": [0.001, 0.002], "open_positions": [0.041, 0.038]}
    payload = {"arm_positions_rad": [0.2] * 7, "gripper_open_fraction": 0.5}
    indices, targets = resolve_joint_targets(payload, names, limits, config)
    assert indices == list(range(9))
    assert targets[7:] == pytest.approx([0.021, 0.020])
    bad = copy.deepcopy(payload)
    bad["arm_positions_rad"][3] = 2
    with pytest.raises(ValueError, match="outside runtime limits"):
        resolve_joint_targets(bad, names, limits, config)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "1", -0.1, 1.1])
def test_invalid_gripper_targets_rejected(bad):
    from scene_factory.joint_control import validate_joint_position_payload
    with pytest.raises(ValueError):
        validate_joint_position_payload({"arm_positions_rad": [0] * 7, "gripper_open_fraction": bad})
