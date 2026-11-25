"""Command-line helpers for the local runtime service."""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from typing import Any

from .runtime_client import RuntimeClient
from .runtime_service import RuntimeService, serve


def add_service_arguments(subparsers: Any) -> None:
    service = subparsers.add_parser("service", help="Run or call the local runtime HTTP service")
    commands = service.add_subparsers(dest="service_command", required=True)
    add_service_commands(commands)


def add_service_commands(commands: Any) -> None:
    """Register direct commands, also used beneath ``sandbox`` by the main CLI."""
    server = commands.add_parser("serve", help="Start the local HTTP service")
    server.add_argument("--root", type=Path, required=True)
    server.add_argument("--scene-root", type=Path)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--max-sessions", type=int, default=2)
    server.add_argument("--lease-seconds", type=float, default=300)
    server.add_argument("--request-timeout", type=float, default=30)
    server.add_argument("--startup-timeout", type=float, default=60)
    server.add_argument("--memory-budget-mb", type=int, default=16384)
    server.add_argument("--cpu-budget", type=int)
    server.add_argument("--mujoco-worker-pool", action="store_true", help="Reuse clean compatible MuJoCo models across sessions")
    server.add_argument("--pool-max-idle", type=int, default=1)
    server.add_argument("--pool-idle-seconds", type=float, default=60)
    server.add_argument("--pool-max-uses", type=int, default=100)
    server.add_argument("--token")
    server.add_argument("--token-env", default="SCENE_FACTORY_SERVICE_TOKEN")
    for name in (
        "list",
        "inspect",
        "create",
        "reset",
        "step",
        "observe",
        "close",
        "snapshot",
        "restore",
        "submit-build",
        "build-status",
    ):
        cmd = commands.add_parser(name)
        cmd.add_argument("--server-url", default="http://127.0.0.1:8765")
        cmd.add_argument("--token")
        cmd.add_argument("--token-env", default="SCENE_FACTORY_SERVICE_TOKEN")
        if name in {"inspect", "reset", "step", "observe", "close", "snapshot", "restore"}:
            cmd.add_argument("session_id")
        if name in {"create", "submit-build"}:
            cmd.add_argument("request", type=Path)
        if name in {"reset", "step", "snapshot", "restore"}:
            cmd.add_argument("request", type=Path)
        if name == "observe":
            cmd.add_argument("--modalities", default="state")
            cmd.add_argument("--camera", default="default")
        if name == "build-status":
            cmd.add_argument("build_id")


def run_service_command(args: Any) -> int:
    token = args.token or os.environ.get(args.token_env)
    if args.service_command == "serve":
        manager = RuntimeService(
            args.root,
            allowed_scene_root=args.scene_root,
            max_sessions=args.max_sessions,
            lease_seconds=args.lease_seconds,
            request_timeout=args.request_timeout,
            startup_timeout=args.startup_timeout,
            memory_budget_mb=args.memory_budget_mb,
            cpu_budget=args.cpu_budget,
            mujoco_worker_pool=args.mujoco_worker_pool,
            pool_max_idle=args.pool_max_idle,
            pool_idle_seconds=args.pool_idle_seconds,
            pool_max_uses=args.pool_max_uses,
            token=token,
        )
        server = serve(manager, args.host, args.port)

        def stop(*_: Any) -> None:
            # HTTPServer.shutdown must run outside serve_forever's thread.
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        try:
            print(
                json.dumps(
                    {"status": "serving", "url": f"http://{args.host}:{server.server_port}"}
                ),
                flush=True,
            )
            server.serve_forever()
        finally:
            server.server_close()
            manager.close()
        return 0
    client = RuntimeClient(args.server_url, token=token)
    if args.service_command == "list":
        result = client.list()
    elif args.service_command == "inspect":
        result = client.inspect(args.session_id)
    elif args.service_command == "create":
        result = client.create(json.loads(args.request.read_text(encoding="utf-8")))
    elif args.service_command == "reset":
        result = client.reset(args.session_id, json.loads(args.request.read_text(encoding="utf-8")))
    elif args.service_command == "step":
        result = client.step(args.session_id, json.loads(args.request.read_text(encoding="utf-8")))
    elif args.service_command == "observe":
        result = client.observe(
            args.session_id, modalities=args.modalities.split(","), camera=args.camera
        )
    elif args.service_command in {"snapshot", "restore"}:
        result = getattr(client, args.service_command)(
            args.session_id, json.loads(args.request.read_text(encoding="utf-8"))
        )
    elif args.service_command == "submit-build":
        result = client.build(json.loads(args.request.read_text(encoding="utf-8")))
    elif args.service_command == "build-status":
        result = client.build_status(args.build_id)
    else:
        result = client.close(args.session_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
