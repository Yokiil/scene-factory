"""Dedicated, persistent simulator process used by the runtime HTTP service."""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any


def worker_main(connection: Any, config: dict[str, Any]) -> None:
    """Own the simulator for one session until close, parent exit or failure.

    The connection is private to the local parent process. Requests arriving
    through HTTP have already been decoded as finite JSON and checked there.
    """
    session = None
    backend = None
    try:
        threads = str(config["resources"]["cpu_threads"])
        os.environ["OMP_NUM_THREADS"] = threads
        os.environ["OPENBLAS_NUM_THREADS"] = threads
        os.environ["MKL_NUM_THREADS"] = threads
        from .agent import DryRunBackend, SceneFactoryEnv
        from .registry import AssetRegistry
        from .session import RuntimeSession
        from .session_contracts import finite_json

        backend_name = config["backend"]
        if config.get("environment_path"):
            from .environment_package import verify_environment

            # Kit must own USD library bootstrap. The manager fully resolved
            # the closure; this fresh worker checks all bytes and runtime hashes
            # without loading standalone pxr before SimulationApp.
            verified = verify_environment(config["environment_path"], resolve_assets=backend_name != "isaac")
            if not verified["valid"]:
                raise ValueError(f"frozen environment is incompatible: {verified['errors']}")
        options = config["backend_options"]
        if backend_name == "dry_run":
            backend = DryRunBackend(**options)
        elif backend_name == "mujoco":
            from .backends.mujoco import MujocoBackend

            registry = AssetRegistry.load(config["registry_path"])
            # The registry file is frozen; referenced assets keep their original
            # base directory. Full asset closure is supplied by environment locks.
            registry.base_dir = Path(config["registry_base_dir"])
            backend = MujocoBackend(registry=registry, **options)
        else:
            from .backends.isaac import IsaacSimBackend

            backend = IsaacSimBackend(config["usd_path"], headless=True, **options)
        pool_signature = config.get("mujoco_pool_signature")
        pool_enabled = backend_name == "mujoco" and bool(pool_signature)
        pool_tainted, worker_reuse_count = False, 0
        parked_evidence = None
        if pool_enabled:
            from .worker_pool import mujoco_worker_signature

            if mujoco_worker_signature(config)["sha256"] != pool_signature:
                raise ValueError("MuJoCo pool inputs changed before worker startup")

        def new_session(current_config: dict[str, Any]):
            history_limit = current_config.get("history_limit", 4096)
            if type(history_limit) is not int or not 1 <= history_limit <= 100000:
                raise ValueError("worker history_limit must be an integer in [1, 100000]")
            created = RuntimeSession(SceneFactoryEnv(current_config["scene_path"], backend), history_limit=history_limit)
            created.session_id = current_config["session_id"]
            created.artifact_root = Path(current_config["session_dir"]) / "artifacts"
            created.artifact_root.mkdir(exist_ok=True)
            return created

        session = new_session(config)

        def describe() -> dict[str, Any]:
            return {
                **(session.describe() if session is not None else {
                    "session_id": config["session_id"], "backend": backend_name, "closed": True,
                    "faulted": pool_tainted, "episode_id": None,
                }),
                "observation_capabilities": getattr(backend, "observation_capabilities", None),
                "worker_runtime": {
                    "pid": os.getpid(), "mujoco_pool_enabled": pool_enabled,
                    "reuse_count": worker_reuse_count, "pool_tainted": pool_tainted,
                    "model_compile_count": getattr(backend, "_model_compile_count", None),
                    "model_reset_count": getattr(backend, "_model_reset_count", None),
                    "parked_evidence": parked_evidence,
                },
                "service_constraints": {
                    "episodes_per_worker": 1 if backend_name == "isaac" else None,
                    "restore_in_worker": backend_name != "isaac",
                    "history_limit_max": 100000,
                },
            }

        connection.send({"ok": True, "result": finite_json(describe())})
        isaac_reset_key = None
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            operation = message["operation"]
            request = message.get("request", {})
            try:
                if operation == "pool_health":
                    if not pool_enabled or session is not None or pool_tainted or backend._model is None or backend._data is not None:
                        raise RuntimeError("worker is not a clean, parked MuJoCo runtime")
                    if mujoco_worker_signature(config)["sha256"] != pool_signature:
                        raise ValueError("MuJoCo pool compatibility changed while idle")
                    result = {"healthy": True, "compatibility_sha256": pool_signature, "worker": describe()["worker_runtime"]}
                elif operation == "pool_bind":
                    if not pool_enabled or session is not None or pool_tainted:
                        raise RuntimeError("worker cannot bind another session")
                    candidate = request["config"]
                    if candidate.get("backend") != "mujoco" or candidate.get("mujoco_pool_signature") != pool_signature:
                        raise ValueError("pooled worker backend or compatibility key mismatch")
                    if candidate.get("environment_path"):
                        from .environment_package import verify_environment

                        if not verify_environment(candidate["environment_path"])["valid"]:
                            raise ValueError("new pooled session environment failed verification")
                    if mujoco_worker_signature(candidate)["sha256"] != pool_signature:
                        raise ValueError("new pooled session inputs changed during handoff")
                    registry = AssetRegistry.load(candidate["registry_path"])
                    registry.base_dir = Path(candidate["registry_base_dir"])
                    backend.registry = registry
                    config = candidate
                    session = new_session(config)
                    worker_reuse_count += 1
                    result = describe()
                elif operation == "pool_park":
                    if not pool_enabled or session is None or pool_tainted:
                        raise RuntimeError("only a clean MuJoCo session can enter the idle pool")
                    if mujoco_worker_signature(config)["sha256"] != pool_signature:
                        raise ValueError("pool inputs changed before parking")
                    retained = session.detach_mujoco_backend_for_pool()
                    if retained is not backend:
                        raise RuntimeError("worker backend ownership changed unexpectedly")
                    session = None
                    parked_evidence = {"retained_model_sha256": backend._model_sha256,
                                       "model_compile_count": backend._model_compile_count,
                                       "episode_data_discarded": backend._data is None, "renderer_discarded": backend._renderer is None}
                    result = {"session_id": config["session_id"], "closed": True, "reusable": True, "evidence": parked_evidence}
                elif operation == "describe":
                    result = describe()
                elif operation == "reset":
                    if (
                        backend_name == "isaac"
                        and isaac_reset_key is not None
                        and request.get("idempotency_key") != isaac_reset_key
                    ):
                        raise NotImplementedError(
                            "Isaac currently requires a new session for another episode; retrying the same reset key is supported"
                        )
                    result = session.reset(request)
                    if backend_name == "isaac":
                        isaac_reset_key = request.get("idempotency_key")
                elif operation == "step":
                    result = session.step(request)
                elif operation == "observation":
                    result = session.observe(request) if request else session.observe()
                elif operation in {"snapshot", "restore"}:
                    if operation == "restore" and backend_name == "isaac":
                        raise NotImplementedError(
                            "Isaac restore in a live worker is unavailable; create a new session and reset from the saved initialization recipe"
                        )
                    method = getattr(session, operation, None)
                    if method is None:
                        raise NotImplementedError(
                            f"{operation} is unsupported by this session adapter"
                        )
                    result = method(request)
                elif operation == "close":
                    if session is not None:
                        session.close()
                    else:
                        backend.close()
                    result = {"session_id": config["session_id"], "closed": True}
                else:
                    raise ValueError(f"unknown worker operation: {operation}")
                connection.send(
                    {
                        "request_id": message["request_id"],
                        "ok": True,
                        "result": finite_json(result),
                        "session": finite_json(describe()),
                    }
                )
            except Exception as exc:
                # Conservative: any request exception retires this process from
                # future pooling, even if a later in-session reset succeeds.
                pool_tainted = True
                connection.send(
                    {
                        "request_id": message["request_id"],
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "session": finite_json(describe()),
                    }
                )
            if operation == "close":
                break
    except Exception as exc:
        try:
            connection.send(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        # Diagnostics are local evidence, not part of the HTTP protocol.
        try:
            (Path(config["session_dir"]) / "worker_error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
        except OSError:
            pass
    finally:
        if session is not None or backend is not None:
            try:
                session.close() if session is not None else backend.close()
            except Exception:
                pass
        connection.close()


def build_worker_main(connection: Any, config: dict[str, Any]) -> None:
    """Compile a trusted local scene in an isolated job process."""
    try:
        from .environment_package import build_environment

        result = build_environment(
            config["scene_path"],
            config["output_dir"],
            registry_path=config["registry_path"],
            backend=config["backend"],
            cache_dir=config["cache_dir"],
            **{key: config[key] for key in ("usd_path", "robot_asset_root", "backend_options") if key in config},
        )
        connection.send({"ok": True, "result": result})
    except Exception as exc:
        connection.send({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})
    finally:
        connection.close()
