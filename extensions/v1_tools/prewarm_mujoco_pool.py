"""Compile a MuJoCo scene through the public API, then park its clean worker."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.runtime_client import RuntimeClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    parser.add_argument("--token-env", default="SCENE_FACTORY_SERVICE_TOKEN")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("backend", "mujoco" if "environment_path" in request else None) != "mujoco":
        raise ValueError("prewarming is supported only for MuJoCo")
    client = RuntimeClient(args.server_url, token=os.environ.get(args.token_env))
    created = client.create(request)
    sid = created["session_id"]
    try:
        initial = client.reset(sid, {"idempotency_key": "prewarm-reset", "seed": args.seed})
    finally:
        closed = client.close(sid)
    retained = closed.get("worker_disposition") == "pooled"
    print(json.dumps({"status": "parked" if retained else "not_retained", "session_id": sid,
                      "worker_pid": created["worker_pid"], "model_compile_count": initial["info"].get("model_compile_count"),
                      "scene_sha256": initial["scene_sha256"], "close": closed}))
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())
