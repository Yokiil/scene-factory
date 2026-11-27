from __future__ import annotations

import copy
import json
import math

import pytest

from scene_factory.agent import SceneFactoryEnv
from scene_factory.backends.mujoco import MujocoBackend
from scene_factory.episode_randomization import assign_family_splits, noisy_rgb
from scene_factory.episode_reset import prepare_episode_reset
from scene_factory.session import RuntimeSession


def scene():
    return {"scene_id": "four_layers", "room_dimensions_m": [2, 2, 3], "objects": [
        {"object_id": "ball", "asset_id": "mug_blue", "dynamic": True,
         "bbox_m": [.1, .1, .1], "pose": {"position": [0, 0, .051], "yaw_deg": 0}},
    ], "task": {"instruction": "Observe motion"}}


def config():
    return {"difficulty": "easy",
        "layout": {"pose_jitter": {"ball": {"x_m": [.01, .01]}}},
        "physics": {"instances": {"ball": {"density_scale": [1.2, 1.2], "friction": [.4, .4]}}},
        "initial_state": {"pose_jitter": {"ball": {"y_m": [.002, .002], "yaw_deg": [1, 1]}}},
        "observation": {"position_noise_std_m": [.001, .001], "rgb_noise_std_uint8": [2, 2]}}


def test_four_layers_reproduce_and_independent_streams():
    first, record = prepare_episode_reset(scene(), seed=123, options={"randomization": config()})
    second, same = prepare_episode_reset(scene(), seed=123, options={"randomization": config()})
    assert first == second and record == same
    assert first["objects"][0]["pose"]["position"][:2] == [.01, .002]
    more = config()
    more["observation"]["position_noise_std_m"] = [.003, .004]
    changed, changed_record = prepare_episode_reset(scene(), seed=123, options={"randomization": more})
    for layer in ("layout", "physics", "initial_state"):
        assert record["domain_randomization"]["layers"][layer] == changed_record["domain_randomization"]["layers"][layer]
    assert first["objects"] == changed["objects"]
    assert len({v["seed"] for v in record["domain_randomization"]["layers"].values()}) == 4


def test_real_mujoco_applies_physics_and_noise_preserves_truth_and_restore(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene()), encoding="utf-8")
    env = SceneFactoryEnv(path, MujocoBackend(frame_skip=1))
    session = RuntimeSession(env)
    try:
        initial = session.reset({"idempotency_key": "a", "seed": 123, "options": {"randomization": config()}})
        body = env.backend._body_ids["ball"]
        assert env.backend._model.body_mass[body] == pytest.approx(env._scene["objects"][0]["physics"]["mass_kg"])
        assert initial["observation"]["object_poses"]["ball"][:2] == [.01, .002]
        assert initial["observation"]["sensor_observations"]["object_positions_m"]["ball"] != initial["observation"]["object_poses"]["ball"]
        step = session.step({"episode_id": session.episode_id, "sequence_id": 0, "idempotency_key": "step", "action": {"mode": "noop"}})
        snapshot = session.snapshot({"idempotency_key": "snap"})
        restored = session.restore({"idempotency_key": "restore", "snapshot": snapshot})
        assert step["observation"]["sensor_observations"] == restored["observation"]["sensor_observations"]
    finally:
        session.close()


def test_rgb_noise_is_repeated_same_state_and_does_not_mutate_truth():
    import numpy as np

    sampled, _ = prepare_episode_reset(scene(), seed=123, options={"randomization": config()})
    truth = np.full((10, 10, 3), 100, np.uint8)
    first, second = noisy_rgb(sampled, truth, 1), noisy_rgb(sampled, truth, 1)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, noisy_rgb(sampled, truth, 2))
    assert np.all(truth == 100)


def test_fault_and_sensor_unknown_capability_rejected():
    for raw in [{"initial_state": {"broken": True}}, {"observation": {"depth_noise": [0, 1]}},
                {"layout": {"asset_family": ["new"]}}, {"observation": {"position_noise_std_m": [-1, 0]}}]:
        with pytest.raises(ValueError):
            prepare_episode_reset(scene(), seed=0, options={"randomization": raw})


def test_full_orientation_jitter_composes_world_z_and_scaling_tilt_is_rejected():
    from scene_factory.models import Pose

    source = scene()
    source["objects"][0]["pose"]["orientation_wxyz"] = [math.cos(.1), math.sin(.1), 0, 0]
    changed, _ = prepare_episode_reset(source, seed=0, options={"pose_jitter": {"ball": {"yaw_deg": [90, 90]}}})
    attitude = Pose.from_dict(changed["objects"][0]["pose"])
    assert attitude.yaw_deg == pytest.approx(90)
    assert not attitude.is_upright
    assert attitude.orientation_wxyz != tuple(source["objects"][0]["pose"]["orientation_wxyz"])
    raw = {"physics": {"instances": {"ball": {"scale": [.9, 1.1]}}}}
    with pytest.raises(ValueError, match="tilted"):
        prepare_episode_reset(source, seed=0, options={"randomization": raw})


def test_split_connected_families_never_leak_and_reports_impossible_splits():
    rows = [{"id": "a", "asset_families": ["peg"], "task_structure": "insert"},
            {"id": "b", "asset_families": ["peg"], "task_structure": "remove"},
            {"id": "c", "asset_families": ["bolt"], "task_structure": "remove"}]
    report = assign_family_splits(copy.deepcopy(rows))
    assert len({r["split"] for r in report["rows"]}) == 1
    assert report["component_count"] == 1 and len(report["empty_splits"]) == 2
    assert not report["asset_family_leakage"] and not report["task_structure_leakage"]

