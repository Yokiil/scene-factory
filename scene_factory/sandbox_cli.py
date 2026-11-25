"""Small CLI surface shared with the process-local Python session API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import DryRunBackend, SceneFactoryEnv
from .dataset import write_json_atomic
from .paths import default_registry_path
from .registry import AssetRegistry
from .session import RuntimeSession
from .session_contracts import backend_capabilities, runtime_session_schema


def add_sandbox_arguments(subparsers: Any) -> None:
    parser = subparsers.add_parser("sandbox", help="Runtime contracts, session smoke and baseline timing")
    commands = parser.add_subparsers(dest="sandbox_command", required=True)
    from .runtime_service_cli import add_service_commands
    from .environment_cli import add_environment_commands

    add_service_commands(commands)
    add_environment_commands(commands)
    commands.add_parser("schema", help="Print the versioned session JSON Schema")
    capabilities = commands.add_parser("capabilities", help="Inspect implemented backend capabilities")
    capabilities.add_argument("--backend", choices=("dry-run", "mujoco", "isaac"))
    smoke = commands.add_parser("smoke", help="Run reset and action steps through the session API")
    smoke.add_argument("scene", type=Path)
    smoke.add_argument("--backend", choices=("dry-run", "mujoco", "isaac"), default="dry-run")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.add_argument("--steps", type=int, default=1)
    smoke.add_argument("--reset-options", type=Path, help="JSON file containing pose initialization options")
    smoke.add_argument("--action", type=Path, help="JSON file containing {mode, payload}")
    smoke.add_argument("--usd", type=Path, help="Existing USD required by the Isaac scripted backend")
    smoke.add_argument("--output", type=Path)
    from .runtime_benchmark import add_benchmark_arguments

    add_benchmark_arguments(commands.add_parser("benchmark", help="Measure current runtime baseline"))


def _load(path: Path | None) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path is not None else {}


def run_sandbox_from_args(args: Any) -> int:
    from .environment_cli import ENVIRONMENT_COMMANDS, run_environment_command

    if args.sandbox_command in ENVIRONMENT_COMMANDS:
        return run_environment_command(args)
    if args.sandbox_command in {"serve", "create", "reset", "step", "observe", "close", "inspect",
                                "list", "snapshot", "restore", "submit-build", "build-status"}:
        from .runtime_service_cli import run_service_command

        args.service_command = args.sandbox_command
        return run_service_command(args)
    if args.sandbox_command == "benchmark":
        from .runtime_benchmark import run_benchmark_from_args

        return run_benchmark_from_args(args)
    if args.sandbox_command == "schema":
        report = runtime_session_schema()
    elif args.sandbox_command == "capabilities":
        names = [args.backend.replace("-", "_")] if args.backend else ["dry_run", "mujoco", "isaac"]
        report = {name: backend_capabilities(name) for name in names}
    else:
        if not 1 <= args.steps <= 10000:
            raise ValueError("--steps must be in [1, 10000]")
        # Prevent a report from overwriting an input used to construct this session.
        if args.output is not None:
            sources = [args.scene, args.reset_options, args.action, args.usd,
                       args.registry or default_registry_path()]
            if any(path is not None and path.resolve() == args.output.resolve() for path in sources):
                raise ValueError("smoke output must differ from input paths")
        backend = None
        session = None
        try:
            options = _load(args.reset_options)
            action = _load(args.action) if args.action else {
                "mode": "scripted" if args.backend == "isaac" else "noop", "payload": {},
            }
            if args.backend == "dry-run":
                backend = DryRunBackend(max_steps=args.steps + 1)
            elif args.backend == "mujoco":
                from .backends.mujoco import MujocoBackend

                backend = MujocoBackend(
                    registry=AssetRegistry.load(args.registry or default_registry_path()),
                    max_steps=args.steps + 1,
                )
            else:
                from .backends.isaac import IsaacSimBackend

                if args.usd is None:
                    raise ValueError("--usd is required for Isaac")
                backend = IsaacSimBackend(args.usd, max_steps=args.steps + 1)
            session = RuntimeSession(SceneFactoryEnv(args.scene, backend=backend))
            initial = session.reset({"idempotency_key": "smoke-reset", "seed": args.seed, "options": options})
            request = {
                "episode_id": session.episode_id, "sequence_id": 0,
                "idempotency_key": "smoke-step", "action": action, "steps": args.steps,
            }
            final = session.step(request)
            retry = session.step(request)
            if retry != final:
                raise RuntimeError("idempotent step response mismatch")
            report = {
                "schema_version": "scene_factory.session_smoke.v1", "status": "passed",
                "scope": "session_protocol_smoke_not_physics_acceptance",
                "session": session.describe(), "initial": initial, "final": final,
                "retry_response_identical": True,
            }
        except Exception as exc:
            report = {
                "schema_version": "scene_factory.session_smoke.v1", "status": "failed",
                "scope": "session_protocol_smoke_not_physics_acceptance",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        finally:
            try:
                if session is not None:
                    session.close()
                elif backend is not None:
                    backend.close()
                report["cleanup"] = "completed" if backend is not None else "not_needed"
            except Exception as exc:
                report["status"] = "failed"
                report["cleanup"] = "failed"
                report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        if args.output:
            write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if report.get("status") == "failed" else 0
