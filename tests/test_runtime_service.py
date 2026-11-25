from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scene_factory.runtime_client import RuntimeClient, RuntimeClientError
from scene_factory.runtime_service import RuntimeService, ServiceError, serve


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps(
            {
                "scene_id": "service",
                "task": {},
                "objects": [
                    {
                        "object_id": "part",
                        "dynamic": True,
                        "pose": {"position": [0, 0, 0.1], "yaw_deg": 0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def running(tmp_path: Path):
    manager = RuntimeService(
        tmp_path / "service", allowed_scene_root=tmp_path, max_sessions=2, lease_seconds=10
    )
    server = serve(manager, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
    yield manager, server, client
    server.shutdown()
    server.server_close()
    manager.close()
    thread.join(timeout=3)


def test_http_session_lifecycle_and_idempotency(running, scene):
    _, _, client = running
    created = client.create({"scene_path": str(scene), "backend": "dry-run"})
    sid = created["session"]["session_id"]
    reset = client.reset(sid, {"idempotency_key": "r", "seed": 1, "options": {}})
    request = {
        "episode_id": reset["episode_id"],
        "sequence_id": 0,
        "idempotency_key": "s",
        "steps": 2,
        "action": {"mode": "noop", "payload": {}},
    }
    first = client.step(sid, request)
    second = client.step(sid, request)
    assert first == second and first["step_index"] == 2
    assert client.observe(sid) == first
    assert client.close(sid)["closed"] is True


def test_capacity_and_path_boundary(running, scene):
    manager, _, client = running
    client.create({"scene_path": str(scene), "backend": "dry-run"})
    client.create(
        {"scene": {"scene_id": "inline", "task": {}, "objects": []}, "backend": "dry-run"}
    )
    with pytest.raises(RuntimeClientError) as exc:
        client.create({"scene_path": str(scene), "backend": "dry-run"})
    assert exc.value.status == 409
    assert len(manager.list()["sessions"]) == 2


def test_restart_marks_sessions_lost(tmp_path, scene):
    root = tmp_path / "service"
    first = RuntimeService(root, allowed_scene_root=tmp_path)
    server = serve(first, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
    sid = client.create({"scene_path": str(scene)})["session"]["session_id"]
    server.shutdown()
    server.server_close()
    first.close()
    thread.join(timeout=3)
    # Simulate metadata left active by an abrupt parent crash. A normal shutdown
    # closes workers and retains closed status, which must not become lost.
    with sqlite3.connect(root / "service.sqlite3") as database:
        database.execute("UPDATE sessions SET status='ready' WHERE id=?", (sid,))
    with RuntimeService(root, allowed_scene_root=tmp_path) as second:
        assert any(
            row["session_id"] == sid and row["status"] == "lost"
            for row in second.list()["sessions"]
        )


def test_unsupported_snapshot_is_explicit(running, scene):
    _, _, client = running
    sid = client.create({"scene_path": str(scene)})["session"]["session_id"]
    with pytest.raises(RuntimeClientError) as exc:
        client.snapshot(sid, {"idempotency_key": "snap"})
    assert exc.value.status in {400, 409}


def test_create_retries_and_conflicts_are_persistent(running, scene):
    manager, _, client = running
    request = {"idempotency_key": "create-1", "scene_path": str(scene)}
    first = client.create(request)
    second = client.create(request)
    assert first["session_id"] == second["session_id"]
    assert len(manager.list()["sessions"]) == 1
    with pytest.raises(RuntimeClientError) as exc:
        client.create({**request, "backend_options": {"max_steps": 7}})
    assert exc.value.status == 409
    client.close(first["session_id"])
    assert client.create(request)["status"] == "closed"


def test_concurrent_http_duplicate_step_executes_once(running, scene):
    manager, _, client = running
    sid = client.create({"scene_path": str(scene)})["session_id"]
    initial = client.reset(sid, {"idempotency_key": "r", "seed": 0})
    request = {
        "episode_id": initial["episode_id"],
        "sequence_id": 0,
        "idempotency_key": "s",
        "steps": 3,
        "action": {"mode": "noop"},
    }
    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(lambda _: client.step(sid, request), range(5)))
    assert all(item == responses[0] for item in responses)
    assert client.observe(sid)["step_index"] == 3
    assert (
        manager._db.execute(
            "SELECT COUNT(*) FROM events WHERE session_id=? AND operation='step'", (sid,)
        ).fetchone()[0]
        == 5
    )


def test_scene_is_frozen_and_invalid_options_do_not_poison_session(running, scene):
    _, _, client = running
    sid = client.create({"scene_path": str(scene)})["session_id"]
    scene.write_text("{}", encoding="utf-8")
    initial = client.reset(sid, {"idempotency_key": "r", "seed": 1})
    assert initial["observation"]["scene_graph"][0]["object_id"] == "part"
    with pytest.raises(RuntimeClientError) as exc:
        client.reset(sid, {"idempotency_key": "invalid", "options": {"unknown": True}})
    assert exc.value.status == 400
    assert client.observe(sid) == initial
    assert client.inspect(sid)["status"] == "ready"


def test_input_path_and_unknown_fields_are_rejected(running, tmp_path):
    _, _, client = running
    for request in (
        {"scene_path": "../outside.json"},
        {"scene": {"objects": []}, "unknown": True},
        {"scene": {"objects": []}, "backend_options": {"frame_skip": 0}, "backend": "mujoco"},
    ):
        with pytest.raises(RuntimeClientError) as exc:
            client.create(request)
        assert exc.value.status == 400


def stalled_worker(connection, config):
    connection.send({"ok": True, "result": {"session_id": config["session_id"]}})
    connection.recv()
    time.sleep(10)


def test_timeout_kills_worker_preserves_failure_and_frees_capacity(tmp_path, scene):
    with RuntimeService(
        tmp_path / "timeout",
        allowed_scene_root=tmp_path,
        max_sessions=1,
        request_timeout=0.05,
        worker_target=stalled_worker,
    ) as manager:
        result = manager.create({"idempotency_key": "c", "scene_path": str(scene)})
        sid = result["session_id"]
        process = manager._sessions[sid].process
        with pytest.raises(ServiceError) as exc:
            manager.call(sid, "reset", {"idempotency_key": "r"})
        assert exc.value.status == 504
        assert not process.is_alive()
        assert manager.describe(sid)["status"] == "failed"
        assert not manager._sessions


def test_lease_reclaims_idle_session(tmp_path, scene):
    with RuntimeService(
        tmp_path / "lease", allowed_scene_root=tmp_path, lease_seconds=0.15
    ) as manager:
        result = manager.create({"idempotency_key": "c", "scene_path": str(scene)})
        sid = result["session_id"]
        deadline = time.monotonic() + 3
        while manager.describe(sid)["status"] == "ready" and time.monotonic() < deadline:
            time.sleep(0.03)
        assert manager.describe(sid)["status"] == "expired"
        assert not manager._sessions


def test_service_root_is_exclusive_and_release_allows_reopen(tmp_path):
    with RuntimeService(tmp_path / "locked"):
        with pytest.raises(ValueError, match="already owned"):
            RuntimeService(tmp_path / "locked")
    with RuntimeService(tmp_path / "locked") as second:
        assert second.list()["sessions"] == []


def test_remote_bind_requires_token_and_artifact_paths_are_constrained(tmp_path):
    with RuntimeService(tmp_path / "local") as manager:
        with pytest.raises(ValueError, match="bearer token"):
            serve(manager, "0.0.0.0", 0)


def exit_on_close_worker(connection, config):
    connection.send({"ok": True, "result": {"session_id": config["session_id"]}})
    message = connection.recv()
    if message["operation"] == "close":
        os._exit(0)


def test_isaac_clean_process_exit_is_successful_close(tmp_path, scene):
    with RuntimeService(
        tmp_path / "isaac-exit", allowed_scene_root=tmp_path, worker_target=exit_on_close_worker
    ) as manager:
        created = manager.create(
            {
                "idempotency_key": "c",
                "scene_path": str(scene),
                "backend": "isaac",
                "usd_path": str(scene),
            }
        )
        result = manager.close_session(created["session_id"])
        assert result["closed"] is True and result["status"] == "closed"


def test_failed_worker_is_not_restarted_by_retried_create(tmp_path, scene):
    with RuntimeService(tmp_path / "crash", allowed_scene_root=tmp_path) as manager:
        request = {"idempotency_key": "c", "scene_path": str(scene)}
        created = manager.create(request)
        sid = created["session_id"]
        process = manager._sessions[sid].process
        process.terminate()
        process.join(2)
        with pytest.raises(ServiceError):
            manager.call(sid, "observation")
        replay = manager.create(request)
        assert replay["session_id"] == sid and replay["status"] == "failed"
        assert not manager._sessions


def test_locked_environment_rejects_override(running, tmp_path):
    _, _, client = running
    with pytest.raises(RuntimeClientError) as exc:
        client.create({"environment_path": str(tmp_path), "backend_options": {"frame_skip": 100}})
    assert exc.value.status == 400


def test_async_build_and_locked_environment_session(running, tmp_path):
    from scene_factory.factory import SceneFactory

    factory = SceneFactory()
    scene = factory.build_from_recipe("living_room_returned_home", 19)
    files = factory.write_result(scene, tmp_path / "compiled", export_mjcf=False)
    _, _, client = running
    build = client.build({"idempotency_key": "build", "scene_path": str(files["layout"])})
    deadline = time.monotonic() + 20
    while build["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.05)
        build = client.build_status(build["build_id"])
    assert build["status"] == "completed", build
    env = client.create({"environment_path": build["result"]["package_dir"]})
    initial = client.reset(env["session_id"], {"idempotency_key": "r", "seed": 19})
    assert initial["backend"] == "mujoco"
    assert env["inputs"]["asset_closure"] == "verified_package"
