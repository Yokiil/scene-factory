from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scene_factory.backends.mujoco import MujocoBackend
from scene_factory.exporters.mujoco_mjcf import MujocoMjcfExporter
from scene_factory.models import AssetRecord
from scene_factory.registry import AssetRegistry


pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")


@pytest.fixture
def fixture():
    registry = AssetRegistry([AssetRecord(
        asset_id="cube", category="cube", bbox_m=(0.1, 0.1, 0.1), mass_kg=1.0,
    )])
    scene = {
        "scene_id": "runtime-cube", "room_dimensions_m": [2, 2, 2], "seed": 3,
        "objects": [{"object_id": "part", "asset_id": "cube", "dynamic": True,
                     "bbox_m": [0.1, 0.1, 0.1], "support": "floor",
                     "pose": {"position": [0, 0, 0.08], "yaw_deg": 0}}],
        "task": {"target_object": "part", "success": {"predicate": "lifted", "min_height_delta_m": 2}},
    }
    return registry, scene


def backend_for(fixture, **kwargs):
    registry, scene = fixture
    backend = MujocoBackend(registry=registry, **kwargs)
    backend.reset(copy.deepcopy(scene))
    return backend


def resigned(snapshot):
    result = copy.deepcopy(snapshot)
    result.pop("sha256")
    result["sha256"] = hashlib.sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()
    return result


def test_identical_and_dynamic_pose_reset_reuse_model_without_state_pollution(fixture):
    backend = backend_for(fixture)
    try:
        model = backend._model
        before = backend.capture_state()
        backend.step({"object_forces": {"part": [20, 0, 10]}})
        _, info = backend.reset(copy.deepcopy(fixture[1]))
        assert backend._model is model
        assert info["reset_kind"] == "model_reused"
        assert backend.capture_state() == before
        varied = copy.deepcopy(fixture[1])
        varied["objects"][0]["pose"] = {"position": [0.3, -0.2, 0.1], "yaw_deg": 40}
        obs, info = backend.reset(varied)
        assert backend._model is model
        assert info["model_cache_hits"] == 2
        assert obs["objects"]["part"]["position_m"] == pytest.approx([0.3, -0.2, 0.1])
        reference = MujocoBackend(registry=fixture[0], reuse_model=False)
        try:
            reference.reset(varied)
            for _ in range(30):
                backend.step(None)
                reference.step(None)
                np.testing.assert_allclose(backend._data.qpos, reference._data.qpos, atol=1e-13, rtol=0)
                np.testing.assert_allclose(backend._data.qvel, reference._data.qvel, atol=1e-13, rtol=0)
        finally:
            reference.close()
    finally:
        backend.close()


def test_geometry_and_physics_changes_invalidate_model(fixture):
    backend = backend_for(fixture)
    try:
        previous = backend._model
        scene = copy.deepcopy(fixture[1])
        scene["objects"][0]["bbox_m"][0] = 0.2
        _, info = backend.reset(scene)
        assert backend._model is not previous
        assert info["reset_kind"] == "model_compiled"
        previous = backend._model
        scene["objects"][0]["physics"] = {
            "mass_kg": 2, "inertia_kg_m2": [0.01, 0.01, 0.01],
            "friction": 0.0, "source": "test",
        }
        backend.reset(scene)
        assert backend._model is not previous
        np.testing.assert_allclose(backend._model.body_mass[1], 2)
        np.testing.assert_allclose(backend._model.body_inertia[1], [0.01, 0.01, 0.01])
        assert backend._model.geom_friction[-1][0] == 0
    finally:
        backend.close()


def test_snapshot_restores_contact_continuation_exactly_on_same_runtime(fixture):
    backend = backend_for(fixture)
    try:
        for _ in range(25):
            backend.step(None)
        assert backend._data.ncon > 0
        snapshot = backend.capture_state()
        json.dumps(snapshot, allow_nan=False)
        actions = [{"object_forces": {"part": [0.2, 0.1, 0]}}] * 30
        expected = []
        for action in actions:
            backend.step(action)
            expected.append((backend._data.qpos.copy(), backend._data.qvel.copy()))
        observation, info = backend.restore_state(snapshot)
        assert info["step"] == snapshot["steps"]
        assert observation["sim_time"] == snapshot["state"][0]
        assert backend.capture_state() == snapshot
        for action, (qpos, qvel) in zip(actions, expected, strict=True):
            backend.step(action)
            np.testing.assert_array_equal(backend._data.qpos, qpos)
            np.testing.assert_array_equal(backend._data.qvel, qvel)
    finally:
        backend.close()


@pytest.mark.parametrize("corruption", ["checksum", "version", "model", "task", "quaternion", "shape"])
def test_bad_snapshot_is_rejected_without_partial_mutation(fixture, corruption):
    backend = backend_for(fixture)
    try:
        backend.step({"object_forces": {"part": [1, 2, 3]}})
        original = backend.capture_state()
        bad = copy.deepcopy(original)
        if corruption == "checksum":
            bad["sha256"] = "0" * 64
        elif corruption == "version":
            bad["mujoco_version"] = "0.0"
        elif corruption == "model":
            bad["model_sha256"] = "0" * 64
        elif corruption == "task":
            bad["task"]["picked"] = "false"
        elif corruption == "quaternion":
            bad["state"][4:8] = [0, 0, 0, 0]
        elif corruption == "shape":
            bad["state"].pop()
        if corruption != "checksum":
            bad = resigned(bad)
        with pytest.raises(ValueError):
            backend.restore_state(bad)
        assert backend.capture_state() == original
    finally:
        backend.close()


def test_observation_has_aligned_pose_velocity_and_contact_forces(fixture):
    backend = backend_for(fixture)
    try:
        for _ in range(100):
            observation, *_ = backend.step(None)
        obj = observation["objects"]["part"]
        assert obj["orientation_wxyz"] == pytest.approx([1, 0, 0, 0], abs=1e-10)
        assert obj["linear_velocity_mps"] == pytest.approx([0, 0, 0], abs=1e-5)
        assert obj["position_m"] == list(backend._data.qpos[:3])
        contacts = observation["contacts"]
        assert contacts
        assert all(pair["sim_time"] == observation["sim_time"] for pair in contacts)
        assert sum(pair["force_world_n"][2] for pair in contacts) == pytest.approx(9.81, rel=1e-3)
        assert all(pair["sample_window_s"] == 0 and pair["force_on"] == "geom2" for pair in contacts)
    finally:
        backend.close()


def test_rebuild_reference_option_disables_cache(fixture):
    backend = backend_for(fixture, reuse_model=False)
    try:
        previous = backend._model
        _, info = backend.reset(fixture[1])
        assert backend._model is not previous
        assert info["model_cache_hits"] == 0
    finally:
        backend.close()


def test_snapshot_preserves_task_counters_without_reevaluation(fixture):
    backend = backend_for(fixture)
    try:
        evaluator = backend._task_evaluator
        evaluator._picked = True
        evaluator._released = True
        evaluator._placement_stable_steps = 7
        evaluator._max_lift_delta_m = .12
        evaluator._last_target_position = (0.1, 0.2, 0.3)
        evaluator._last_target_step_distance_m = .002
        snapshot = backend.capture_state()
        backend._task_evaluator._placement_stable_steps = 1
        backend.restore_state(snapshot)
        assert backend.capture_state() == snapshot
    finally:
        backend.close()


def test_direct_backend_bad_action_does_not_clear_or_partially_apply_forces(fixture):
    backend = backend_for(fixture)
    try:
        backend.step({"object_forces": {"part": [1, 2, 3]}})
        before = backend.capture_state()
        with pytest.raises(ValueError):
            backend.step({"object_forces": {"part": [4, 5, 6], "missing": [1, 2, 3]}})
        assert backend.capture_state() == before
        with pytest.raises(ValueError):
            backend.step({"object_forces": {"part": [True, 0, 0]}})
        assert backend.capture_state() == before
    finally:
        backend.close()


@pytest.mark.parametrize("field,value", [
    ("mass_kg", 0), ("mass_kg", float("nan")), ("inertia_kg_m2", [1, 1, 3]),
    ("inertia_kg_m2", [True, 1, 1]), ("friction", -1), ("source", ""), ("unknown", 1),
])
def test_instance_physics_rejects_invalid_values(fixture, field, value):
    scene = copy.deepcopy(fixture[1])
    physics = {"mass_kg": 1, "inertia_kg_m2": [1, 1, 1], "friction": 0.5, "source": "test"}
    physics[field] = value
    scene["objects"][0]["physics"] = physics
    with pytest.raises(ValueError):
        MujocoMjcfExporter(fixture[0]).to_string(scene)
