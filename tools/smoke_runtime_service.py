"""Run a real local HTTP/MuJoCo lifecycle smoke and save reviewable evidence."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.dataset import write_json_atomic
from scene_factory.factory import SceneFactory
from scene_factory.runtime_client import RuntimeClient
from scene_factory.runtime_service import RuntimeService, serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    factory = SceneFactory()
    files = factory.write_result(
        factory.build_from_recipe("living_room_returned_home", 19),
        output / "scene",
        export_mjcf=False,
    )
    report = {
        "schema_version": "scene_factory.http_service_smoke.v1",
        "scope": "local_mujoco_proxy_protocol_and_restore; not task acceptance",
        "status": "running",
        "started_at_unix": time.time(),
        "checks": {},
    }
    manager = RuntimeService(output / "service", allowed_scene_root=output)
    server = serve(manager, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RuntimeClient(f"http://127.0.0.1:{server.server_port}")
        request = {
            "idempotency_key": uuid.uuid4().hex,
            "scene_path": str(files["layout"]),
            "backend": "mujoco",
        }
        created = client.create(request)
        sid = created["session_id"]
        report["session_id"] = sid
        report["worker_pid"] = created["worker_pid"]
        report["checks"]["create_retry_same_session"] = client.create(request)["session_id"] == sid
        initial = client.reset(sid, {"idempotency_key": "reset-1", "seed": 0})
        action = {
            "episode_id": initial["episode_id"],
            "sequence_id": 0,
            "idempotency_key": "step-1",
            "action": {"mode": "noop"},
            "steps": 5,
        }
        stepped = client.step(sid, action)
        report["checks"]["step_retry_identical"] = client.step(sid, action) == stepped
        report["checks"]["observe_same_state"] = client.observe(sid) == stepped
        snapshot = client.snapshot(sid, {"idempotency_key": "snapshot-1", "level": "runtime_state"})
        client.step(sid, {**action, "sequence_id": 1, "idempotency_key": "step-2", "steps": 3})
        restored = client.restore(sid, {"idempotency_key": "restore-1", "snapshot": snapshot})
        report["checks"]["restore_time"] = restored["sim_time"] == stepped["sim_time"]
        report["checks"]["restore_object_positions"] = (
            restored["observation"]["object_poses"] == stepped["observation"]["object_poses"]
        )
        report["checks"]["persistent_worker"] = (
            client.inspect(sid)["worker_pid"] == created["worker_pid"]
        )
        report["checks"]["closed"] = client.close(sid)["status"] == "closed"
        report["checks"]["delete_idempotent"] = client.close(sid)["closed"] is True
        report["initial"] = initial
        report["stepped"] = stepped
        report["restored"] = restored
        report["status"] = "passed" if all(report["checks"].values()) else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(3)
        report["finished_at_unix"] = time.time()
        write_json_atomic(output / "service_smoke.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["checks"],
                "report": str(output / "service_smoke.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
