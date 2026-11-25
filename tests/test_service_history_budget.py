from __future__ import annotations

import json
import threading
from contextlib import contextmanager

import pytest

from scene_factory.runtime_client import RuntimeClient, RuntimeClientError
from scene_factory.runtime_service import RuntimeService, serve


@contextmanager
def running(tmp_path, *, pool=False):
    manager = RuntimeService(tmp_path / "service", allowed_scene_root=tmp_path, max_sessions=1, mujoco_worker_pool=pool)
    server = serve(manager, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield manager, RuntimeClient(f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(3)


def request(**extra):
    return {"scene": {"scene_id": "history", "objects": [], "task": {}}, "backend": "dry_run", "backend_options": {"max_steps": 10000}, **extra}


@pytest.mark.parametrize("invalid", [True, 0, -1, 100001, 1.5, "6000", None])
def test_history_budget_rejected_before_worker_creation(tmp_path, invalid):
    with running(tmp_path) as (manager, client):
        with pytest.raises(RuntimeClientError) as exc:
            client.create(request(history_limit=invalid))
        assert exc.value.status == 400 and not manager.list()["sessions"]


def test_default_is_4096_and_explicit_budget_remains_bounded_with_retry_support(tmp_path):
    with running(tmp_path) as (_, client):
        created = client.create(request())
        assert created["history_limit"] == created["session"]["history_limit"] == 4096
        client.close(created["session_id"])
        created = client.create(request(history_limit=2))
        sid = created["session_id"]
        initial = client.reset(sid, {"idempotency_key": "reset"})
        steps = [{"episode_id": initial["episode_id"], "sequence_id": index, "idempotency_key": f"step-{index}", "action": {"mode": "noop"}} for index in range(3)]
        first = client.step(sid, steps[0])
        client.step(sid, steps[1])
        with pytest.raises(RuntimeClientError, match="history limit"):
            client.step(sid, steps[2])
        assert client.step(sid, steps[0]) == first
        assert client.inspect(sid)["history_limit"] == 2


def test_real_worker_accepts_more_than_4096_distinct_requests_when_explicitly_budgeted(tmp_path):
    with running(tmp_path) as (manager, client):
        created = client.create(request(history_limit=5000))
        sid = created["session_id"]
        initial = client.reset(sid, {"idempotency_key": "reset"})
        # Exercise the real worker/session history through private local IPC;
        # HTTP create/reset/observe remain covered without 4,097 disk commits.
        proc = manager._sessions[sid]
        for index in range(4097):
            result = proc.call("step", {"episode_id": initial["episode_id"], "sequence_id": index,
                                       "idempotency_key": f"step-{index}", "action": {"mode": "noop"}})
        assert result["step_index"] == 4097 and client.observe(sid) == result
        assert client.inspect(sid)["session"]["history_limit"] == 5000
        replay = proc.call("step", {"episode_id": initial["episode_id"], "sequence_id": 4096,
                                    "idempotency_key": "step-4096", "action": {"mode": "noop"}})
        assert replay == result


def test_pooled_worker_rebinds_history_budget_without_inheriting_entries(tmp_path):
    scene = {"scene_id": "history-pool", "room_dimensions_m": [2, 2, 3], "objects": [{"object_id": "part", "asset_id": "mug_blue", "dynamic": True,
             "bbox_m": [.04, .04, .06], "pose": {"position": [0, 0, .031], "yaw_deg": 0}}], "task": {}}
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    with running(tmp_path, pool=True) as (_, client):
        base = {"scene_path": str(path), "backend": "mujoco", "backend_options": {"max_steps": 100, "frame_skip": 1}}
        first = client.create({**base, "history_limit": 1})
        initial = client.reset(first["session_id"], {"idempotency_key": "reset"})
        client.step(first["session_id"], {"episode_id": initial["episode_id"], "sequence_id": 0, "idempotency_key": "step", "action": {"mode": "noop"}})
        assert client.close(first["session_id"])["worker_disposition"] == "pooled"
        second = client.create({**base, "history_limit": 6000})
        assert second["worker_reused"] and second["worker_pid"] == first["worker_pid"]
        assert second["session"]["history_limit"] == second["history_limit"] == 6000
        new = client.reset(second["session_id"], {"idempotency_key": "reset"})
        result = client.step(second["session_id"], {"episode_id": new["episode_id"], "sequence_id": 0, "idempotency_key": "step", "action": {"mode": "noop"}})
        assert result["step_index"] == 1 and new["episode_id"] != initial["episode_id"]
