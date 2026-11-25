from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager

import pytest

from scene_factory.agent import SceneFactoryEnv
from scene_factory.backends.mujoco import MujocoBackend
from scene_factory.runtime_client import RuntimeClient, RuntimeClientError
from scene_factory.runtime_service import RuntimeService, serve
from scene_factory.session import RuntimeSession


@contextmanager
def running(tmp_path, **options):
    manager = RuntimeService(tmp_path / "service", allowed_scene_root=tmp_path, max_sessions=options.pop("max_sessions", 1),
                             mujoco_worker_pool=options.pop("mujoco_worker_pool", True), **options)
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


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"scene_id": "pool", "room_dimensions_m": [2, 2, 3], "objects": [
        {"object_id": "part", "asset_id": "mug_blue", "dynamic": True, "bbox_m": [.04, .04, .06],
         "pose": {"position": [0, 0, .031], "yaw_deg": 0}}], "task": {}}), encoding="utf-8")
    return path


def create(client, scene, **extra):
    return client.create({"scene_path": str(scene), "backend": "mujoco", "backend_options": {"max_steps": 10000, "frame_skip": 1}, **extra})


def reset(client, created, seed=1, **extra):
    return client.reset(created["session_id"], {"idempotency_key": "reset", "seed": seed, **extra})


def step(client, sid, episode, **extra):
    return client.step(sid, {"idempotency_key": "step", "episode_id": episode, "sequence_id": 0, "action": {"mode": "noop"}, **extra})


def test_real_http_reuses_pid_and_compiled_model_but_isolates_episode_history_and_state(tmp_path, scene):
    with running(tmp_path) as (manager, client):
        first = create(client, scene)
        initial = reset(client, first)
        moved = step(client, first["session_id"], initial["episode_id"], action={"mode": "object_forces", "payload": {"object_forces": {"part": [0, 0, 6]}}}, steps=20)
        assert moved["observation"]["objects"]["part"]["linear_velocity_mps"][2] > 0
        assert client.close(first["session_id"])["worker_disposition"] == "pooled"
        assert manager.worker_pool_status()["idle_pids"] == [first["worker_pid"]]
        second = create(client, scene)
        assert second["worker_pid"] == first["worker_pid"] and second["worker_reused"]
        assert second["session_id"] != first["session_id"] and second["session"]["episode_id"] is None
        assert second["session"]["next_sequence_id"] == 0
        restored = reset(client, second, seed=2)
        assert restored["episode_id"] != initial["episode_id"] and restored["state_version"] == 1
        assert restored["observation"]["objects"] == initial["observation"]["objects"]
        assert initial["info"]["model_compile_count"] == restored["info"]["model_compile_count"] == 1
        assert restored["info"]["reset_kind"] == "model_reused"
        result = step(client, second["session_id"], restored["episode_id"])
        assert step(client, second["session_id"], restored["episode_id"]) == result
        with pytest.raises(RuntimeClientError) as exc:
            step(client, first["session_id"], initial["episode_id"])
        assert exc.value.status == 410 and client.observe(second["session_id"]) == result
        assert client.close(second["session_id"])["worker_disposition"] == "pooled"
        assert manager.worker_pool_status()["reuse_hits"] == 1


@pytest.mark.parametrize("change", ["settings", "scene", "resources", "backend"])
def test_incompatible_input_evicts_idle_worker_before_new_admission(tmp_path, scene, change):
    with running(tmp_path, cpu_budget=2, memory_budget_mb=2048) as (manager, client):
        first = create(client, scene)
        reset(client, first)
        old = manager._sessions[first["session_id"]]
        client.close(first["session_id"])
        extra = {}
        if change == "settings":
            extra["backend_options"] = {"max_steps": 10000, "frame_skip": 2}
        elif change == "scene":
            payload = json.loads(scene.read_text())
            payload["objects"][0]["pose"]["position"][0] = .02
            scene.write_text(json.dumps(payload), encoding="utf-8")
        elif change == "resources":
            extra["resources"] = {"cpu_threads": 2, "memory_mb": 1024}
        else:
            extra = {"backend": "dry_run", "backend_options": {"max_steps": 10000}}
        second = create(client, scene, **extra)
        assert second["worker_pid"] != first["worker_pid"] and not second["worker_reused"]
        assert not old.process.is_alive() and manager.worker_pool_status()["idle_count"] == 0
        reset(client, second)


def test_default_remains_no_pool_and_uninitialized_worker_is_not_parked(tmp_path, scene):
    with running(tmp_path, mujoco_worker_pool=False) as (manager, client):
        first = create(client, scene)
        reset(client, first)
        old = manager._sessions[first["session_id"]]
        assert client.close(first["session_id"])["worker_disposition"] == "destroyed"
        assert not old.process.is_alive() and not manager.worker_pool_status()["enabled"]
    with running(tmp_path) as (manager, client):
        uninitialized = create(client, scene)
        assert client.close(uninitialized["session_id"])["worker_disposition"] == "destroyed"
        assert manager.worker_pool_status()["idle_count"] == 0


def test_request_exception_permanently_taints_worker_even_after_successful_reset(tmp_path, scene):
    with running(tmp_path) as (manager, client):
        first = create(client, scene)
        initial = reset(client, first)
        old = manager._sessions[first["session_id"]]
        with pytest.raises(RuntimeClientError):
            step(client, first["session_id"], initial["episode_id"], action={"mode": "object_forces", "payload": {"object_forces": {"unknown": [0, 0, 1]}}})
        fresh = client.reset(first["session_id"], {"idempotency_key": "new-reset", "seed": 1})
        assert fresh["step_index"] == 0
        assert client.inspect(first["session_id"])["session"]["worker_runtime"]["pool_tainted"]
        assert client.close(first["session_id"])["worker_disposition"] == "destroyed"
        assert not old.process.is_alive() and manager.worker_pool_status()["idle_count"] == 0
        assert create(client, scene)["worker_pid"] != first["worker_pid"]


def test_idle_timeout_and_worker_use_limit_destroy_process(tmp_path, scene):
    with running(tmp_path, pool_idle_seconds=.1) as (manager, client):
        first = create(client, scene)
        reset(client, first)
        old = manager._sessions[first["session_id"]]
        client.close(first["session_id"])
        deadline = time.monotonic() + 3
        while manager.worker_pool_status()["idle_count"] and time.monotonic() < deadline:
            time.sleep(.05)
        assert not old.process.is_alive() and manager.worker_pool_status()["idle_count"] == 0
    with running(tmp_path, pool_max_uses=2) as (manager, client):
        first = create(client, scene)
        reset(client, first)
        client.close(first["session_id"])
        second = create(client, scene)
        reset(client, second)
        old = manager._sessions[second["session_id"]]
        assert second["worker_reused"] and client.close(second["session_id"])["worker_disposition"] == "destroyed"
        assert not old.process.is_alive()


def test_startup_time_does_not_consume_ready_session_idle_lease(tmp_path, scene, monkeypatch):
    from scene_factory import runtime_service

    original = runtime_service.SessionProcess

    class StartupMeasuredProcess(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ready_handshake_completed = time.monotonic()

    monkeypatch.setattr(runtime_service, "SessionProcess", StartupMeasuredProcess)
    with running(tmp_path, lease_seconds=2) as (manager, client):
        created = create(client, scene)
        proc = manager._sessions[created["session_id"]]
        # The ready lease must begin after the completed worker handshake,
        # independently of slow imports, spawn scheduling, and model startup.
        assert proc.last_used >= proc.ready_handshake_completed
        initial = reset(client, created)
        assert initial["step_index"] == 0 and client.inspect(created["session_id"])["status"] == "ready"


def test_dead_idle_worker_lease_expiry_and_request_timeout_are_destroyed(tmp_path, scene):
    with running(tmp_path) as (manager, client):
        first = create(client, scene)
        reset(client, first)
        old = manager._sessions[first["session_id"]]
        client.close(first["session_id"])
        old.process.terminate()
        old.process.join(2)
        second = create(client, scene)
        assert not second["worker_reused"] and second["worker_pid"] != first["worker_pid"]
        initial = reset(client, second)
        manager.request_timeout = .01
        with pytest.raises(RuntimeClientError) as exc:
            step(client, second["session_id"], initial["episode_id"], steps=1000)
        assert exc.value.status == 504 and client.inspect(second["session_id"])["status"] == "failed"
        assert manager.worker_pool_status()["idle_count"] == 0
    with running(tmp_path, lease_seconds=10) as (manager, client):
        created = create(client, scene)
        reset(client, created)
        old = manager._sessions[created["session_id"]]
        # Isolate idle expiry from process startup and the initial physics load.
        # The separate readiness regression above covers the initial lease.
        with old.lock:
            old.config["lease_seconds"] = .2
            old.last_used = time.monotonic()
        deadline = time.monotonic() + 3
        while client.inspect(created["session_id"])["status"] == "ready" and time.monotonic() < deadline:
            time.sleep(.05)
        assert client.inspect(created["session_id"])["status"] == "expired"
        assert not old.process.is_alive() and manager.worker_pool_status()["idle_count"] == 0


def test_idle_workers_count_toward_cpu_memory_and_process_admission(tmp_path, scene):
    with running(tmp_path, max_sessions=2, cpu_budget=2, memory_budget_mb=128, pool_max_idle=2) as (manager, client):
        resources = {"cpu_threads": 1, "memory_mb": 64}
        first = create(client, scene, resources=resources)
        reset(client, first)
        client.close(first["session_id"])
        changed = create(client, scene, resources=resources, backend_options={"max_steps": 10000, "frame_skip": 2})
        reset(client, changed)
        assert manager.worker_pool_status()["idle_count"] == 1
        current = create(client, scene, resources=resources, backend_options={"max_steps": 10000, "frame_skip": 3})
        reset(client, current)
        assert manager.worker_pool_status()["idle_count"] == 0
        assert len(manager._sessions) == 2
        with pytest.raises(RuntimeClientError) as exc:
            create(client, scene, resources=resources)
        assert exc.value.status == 409
        assert manager.worker_pool_status()["resource_enforcement"] == "admission_only_active_plus_idle"


def test_handoff_rejects_nonfinite_physics_and_clears_session_state(scene):
    session = RuntimeSession(SceneFactoryEnv(scene, MujocoBackend(frame_skip=1)))
    try:
        session.reset({"idempotency_key": "reset", "seed": 1})
        session.env.backend._data.qpos[0] = float("nan")
        with pytest.raises(RuntimeError, match="nonfinite"):
            session.detach_mujoco_backend_for_pool()
        assert not session.describe()["closed"]
    finally:
        session.close()
