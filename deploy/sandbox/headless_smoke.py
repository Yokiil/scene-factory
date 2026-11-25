"""Exercise a running service via HTTP and retain a compact deployment report."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene_factory.dataset import write_json_atomic
from scene_factory.runtime_client import RuntimeClient


def smoke(*, url, scene, backend, steps, usd=None, token=None):
    client = RuntimeClient(url, token=token, timeout=240)
    session_id = None
    report = {"schema": "scene_factory.deployment_smoke.v1", "status": "running",
              "backend": backend, "physics_acceptance": "not_evaluated", "url": url}
    try:
        request = {"scene_path": scene, "backend": backend,
                   "backend_options": {"max_steps": max(steps + 1, 720)},
                   "resources": {"cpu_threads": 2, "memory_mb": 4096 if backend == "isaac" else 1024}}
        if usd:
            request["usd_path"] = usd
        created = client.create(request)
        session_id = created["session"]["session_id"]
        initial = client.reset(session_id, {"idempotency_key": uuid.uuid4().hex, "seed": 42, "options": {}})
        action = {"episode_id": initial["episode_id"], "sequence_id": 0,
                  "idempotency_key": uuid.uuid4().hex, "steps": steps,
                  "action": {"mode": "scripted" if backend == "isaac" else "noop", "payload": {}}}
        final = client.step(session_id, action)
        retry = client.step(session_id, action)
        if retry != final:
            raise RuntimeError("identical step retry returned a different response")
        report.update(status="passed", session_id=session_id, initial=initial, final=final,
                      duplicate_action_replayed=False)
    except Exception as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if session_id:
            try:
                client.close(session_id)
                report["cleanup"] = "completed"
            except Exception as exc:
                report.update(status="failed", cleanup="failed", cleanup_error=str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--scene", required=True, help="Layout path visible inside the service container")
    parser.add_argument("--backend", choices=("dry_run", "mujoco", "isaac"), default="mujoco")
    parser.add_argument("--usd", help="USD path visible inside the service container")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("steps must be in [1, 10000]")
    if args.backend == "isaac" and not args.usd:
        parser.error("Isaac requires --usd")
    report = smoke(url=args.url, scene=args.scene, backend=args.backend, steps=args.steps, usd=args.usd,
                   token=os.environ.get("SCENE_FACTORY_SERVICE_TOKEN"))
    write_json_atomic(args.output, report)
    print(json.dumps({"status": report["status"], "report": str(args.output)}, ensure_ascii=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
