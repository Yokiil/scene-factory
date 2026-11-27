import json
import threading

import pytest

from scene_factory.agent import SceneFactoryEnv
from scene_factory.runtime_client import RuntimeClient, RuntimeClientError
from scene_factory.runtime_service import RuntimeService, serve
from scene_factory.session import RuntimeSession
from scene_factory.world_state import build_world_state


def scene():
    return {
        "scene_id": "sample",
        "room_dimensions_m": [3, 3, 2],
        "task": {"target_object": "cup"},
        "objects": [
            {
                "object_id": key,
                "asset_id": "cube",
                "category": "cube",
                "dynamic": True,
                "support": "floor",
                "bbox_m": [0.1, 0.1, 0.1],
                "pose": {"position": [x, 0, 0.3]},
                "interactions": {
                    "joints": [{"joint_id": "door", "position": 0}],
                    "regions": [{"region_id": "handle", "allowed_actions": ["pull"]}],
                },
            }
            for key, x in (("cup", 0), ("shelf", 1))
        ],
    }


def test_world_uses_one_observation_and_preserves_sources():
    layout = scene()
    layout["task"]["success"] = {"predicate": "pick_and_place"}
    robot = {
        "phase": "GRASP",
        "grasp_diagnostics": {"gripper_open": False},
        "task_oracle": {"holding": True},
        "end_effector_pose": {"position": [0, 1, 1], "orientation_wxyz": [1, 0, 0, 0]},
    }
    obs = {
        "simulation_step": 9,
        "timestamp": 0.15,
        "objects": {
            "cup": {
                "position": [0, 0, 1],
                "orientation_wxyz": [1, 0, 0, 0],
                "pose_source": "PhysX tensor API",
            },
            "shelf": {"position": [0, 0, 3]},
        },
        "robot": robot,
        "mug_skill": {"attempt": 1},
        "contacts": [{"pair": ["cup", "robot"], "normal_force_N": 3}],
        "contact_coverage": {"complete": True},
        "sensor_observations": {"object_positions_m": {"cup": [100, 100, 100]}},
    }
    world = build_world_state(layout, obs, backend="isaac", episode_id="episode", step_index=9)
    cup = world["objects"]["cup"]
    assert cup["position_m"] == [0, 0, 1] and cup["instance"]["initial_pose"]["position"] == [
        0,
        0,
        0.3,
    ]
    assert cup["pose_source"] == "PhysX tensor API" and cup["linear_velocity_mps"] is None
    assert cup["joint_states"] is None and cup["instance"]["initial_joints"][0]["position"] == 0
    assert cup["affordances"][0]["available"] is None
    assert [(row["actions"], row["available"]) for row in cup["affordances"][1:]] == [
        (["grasp"], True),
        (["place"], False),
    ]
    assert world["timestamp"] == world["sim_time"] == 0.15 and world["state_version"] == 9
    assert world["spatial_relations"][0]["distance_m"] == 2
    assert world["spatial_relations"][1]["kind"] == "contact"
    assert world["task_oracle"]["holding"] and world["mug_skill"] == {"attempt": 1}
    world["robot"]["grasp_diagnostics"].clear()
    assert obs["robot"]["grasp_diagnostics"] == {"gripper_open": False}
    with pytest.raises(ValueError, match="step"):
        build_world_state(layout, obs, backend="isaac", episode_id="episode", step_index=10)


def test_mujoco_query_tracks_measured_bodies_and_restored_state(tmp_path):
    pytest.importorskip("mujoco")
    from scene_factory.backends.mujoco import MujocoBackend
    from scene_factory.registry import AssetRegistry
    from scene_factory.models import AssetRecord

    layout = scene()
    for item, name in zip(layout["objects"], ("a/b", "a_b"), strict=True):
        item["object_id"] = name
    layout["task"] = {}
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    registry = AssetRegistry([AssetRecord("cube", "cube", (0.1, 0.1, 0.1))])
    with RuntimeSession(SceneFactoryEnv(path, MujocoBackend(registry=registry))) as session:
        session.reset({"idempotency_key": "r"})
        initial = session.world_state()
        assert initial["objects"]["a/b"]["position_m"][0] == 0
        assert initial["objects"]["a_b"]["position_m"][0] == 1
        state = session.snapshot({"idempotency_key": "checkpoint"})
        session.step(
            {
                "episode_id": session.episode_id,
                "idempotency_key": "s",
                "sequence_id": 0,
                "action": {"mode": "noop"},
                "steps": 3,
            }
        )
        moved = session.world_state()
        assert moved["objects"]["a/b"]["position_m"][2] < initial["objects"]["a/b"]["position_m"][2]
        assert moved["timestamp"] == session.observe()["sim_time"] and moved["robot"] is None
        assert session.world_state() == moved
        session.restore({"idempotency_key": "restore", "snapshot": state})
        restored = session.world_state()
        assert restored["objects"] == initial["objects"] and restored["step_index"] == 0
        assert (
            restored["episode_id"] != initial["episode_id"]
            and restored["state_version"] > moved["state_version"]
        )


def test_http_world_state_read_committed_and_dry_run(tmp_path):
    with RuntimeService(tmp_path / "service", allowed_scene_root=tmp_path) as service:
        server = serve(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
            sid = client.create({"scene": scene(), "backend": "dry-run"})["session_id"]
            with pytest.raises(RuntimeClientError):
                client.world_state(sid)
            frame = client.reset(sid, {"idempotency_key": "reset"})
            world = client.world_state(sid)
            assert (
                world["episode_id"] == frame["episode_id"]
                and world["state_version"] == frame["state_version"]
            )
            assert world["source"] == "dry_run" and world["objects"]["cup"]["position_m"] is None
            assert world["contacts"] is None and world["timestamp"] is None
            assert client.observe(sid) == frame and client.world_state(sid) == world
            client.close(sid)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
