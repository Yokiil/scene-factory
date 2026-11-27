"""Single-user HTTP service with one persistent simulator process per session.

SQLite stores lifecycle and request evidence, not simulator checkpoints. Resource
budgets are admission limits; container or platform limits provide enforcement.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import math
import multiprocessing as mp
import os
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from .dataset import write_json_atomic
from .paths import default_registry_path
from .sandbox_worker import build_worker_main, worker_main
from .session_contracts import finite_json

SERVICE_SCHEMA = "scene_factory.runtime_service.v1"
DEFAULT_MAX_BODY = 4 * 1024 * 1024
MAX_SESSION_HISTORY = 100000
_OPERATIONS = {"describe", "reset", "step", "observation", "world_state", "snapshot", "restore"}


class ServiceError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status, self.code = status, code
        super().__init__(message)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _safe_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 80
        and all(c.isascii() and (c.isalnum() or c in "-_") for c in value)
    )


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return float(value)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _request(value: Any, fields: set[str]) -> dict[str, Any]:
    value = finite_json(value, "request")
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    if value.keys() - fields:
        raise ValueError(f"unknown request fields: {sorted(value.keys() - fields)}")
    return value


def _key(request: dict[str, Any]) -> str:
    value = request.get("idempotency_key")
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("idempotency_key must be a nonempty string of at most 256 characters")
    return value


class SessionProcess:
    def __init__(self, manager: "RuntimeService", config: dict[str, Any]) -> None:
        self.manager, self.config = manager, config
        self.session_id = config["session_id"]
        self.parent, child = manager.context.Pipe()
        self.process = manager.context.Process(
            target=manager.worker_target, args=(child, config), daemon=True
        )
        self.lock = threading.RLock()
        self.last_used = time.monotonic()
        self.failed: dict[str, str] | None = None
        self.descriptor: dict[str, Any] = {}
        self.pool_uses = 1
        self.idle_since: float | None = None
        self.reused = False
        self.process.start()
        child.close()
        try:
            reply = self._receive(manager.startup_timeout)
            if not reply.get("ok"):
                raise RuntimeError(
                    reply.get("error", {}).get("message", "worker failed to initialize")
                )
            self.descriptor = finite_json(reply["result"])
        except Exception:
            self.kill()
            raise

    def _receive(self, timeout: float, *, ignore_stop: bool = False) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while not self.parent.poll(min(0.1, max(0, deadline - time.monotonic()))):
            if not ignore_stop and self.manager._stop.is_set():
                raise RuntimeError("worker request interrupted by service shutdown")
            if time.monotonic() >= deadline:
                raise TimeoutError("worker did not respond before timeout")
        try:
            reply = self.parent.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError("worker exited unexpectedly") from exc
        if not isinstance(reply, dict):
            raise RuntimeError("worker returned invalid protocol data")
        return reply

    def call(
        self, operation: str, request: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        budget = self.manager.request_timeout if timeout is None else timeout
        if not self.lock.acquire(timeout=budget):
            raise ServiceError(503, "session_busy", "another request still owns this session")
        try:
            if self.failed or not self.process.is_alive():
                self._fail("worker_failed", "worker exited unexpectedly")
                raise ServiceError(410, "worker_failed", "session worker is no longer available")
            message = {
                "request_id": uuid.uuid4().hex,
                "operation": operation,
                "request": request or {},
            }
            try:
                self.parent.send(message)
                reply = self._receive(budget, ignore_stop=operation == "close")
                if reply.get("request_id") != message["request_id"]:
                    raise RuntimeError("worker response request ID mismatch")
            except (TimeoutError, RuntimeError, OSError) as exc:
                if operation == "close" and self.config["backend"] == "isaac":
                    self.process.join(1)
                    if self.process.exitcode == 0:
                        return {
                            "session_id": self.session_id,
                            "closed": True,
                            "shutdown": "isaac_clean_process_exit",
                        }
                code = "worker_timeout" if isinstance(exc, TimeoutError) else "worker_failed"
                self._fail(code, str(exc))
                raise ServiceError(
                    504 if code == "worker_timeout" else 410, code, str(exc)
                ) from exc
            self.last_used = time.monotonic()
            self.descriptor = finite_json(reply.get("session", self.descriptor))
            if not reply.get("ok"):
                error = reply.get("error", {})
                kind = error.get("type", "RuntimeError")
                status = (
                    400
                    if kind in {"ValueError", "TypeError", "KeyError", "FileNotFoundError"}
                    else 409
                )
                raise ServiceError(
                    status,
                    "unsupported" if kind == "NotImplementedError" else "session_error",
                    error.get("message", "worker request failed"),
                )
            return finite_json(reply["result"])
        finally:
            self.lock.release()

    def _fail(self, code: str, message: str) -> None:
        self.failed = {"code": code, "message": message}
        self.kill()
        if self.idle_since is None:
            self.manager._terminal(self.session_id, "failed", error=self.failed)

    def kill(self) -> None:
        try:
            self.parent.close()
        except OSError:
            pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
        self.process.join(2)


class RuntimeService:
    def __init__(
        self,
        root: str | Path,
        *,
        allowed_scene_root: str | Path | None = None,
        max_sessions: int = 2,
        request_timeout: float = 30.0,
        startup_timeout: float = 60.0,
        lease_seconds: float = 300.0,
        token: str | None = None,
        max_body: int = DEFAULT_MAX_BODY,
        memory_budget_mb: int = 16384,
        cpu_budget: int | None = None,
        worker_target: Callable = worker_main,
        mujoco_worker_pool: bool = False,
        pool_max_idle: int = 1,
        pool_idle_seconds: float = 60.0,
        pool_max_uses: int = 100,
    ) -> None:
        self.max_sessions = _integer(max_sessions, "max_sessions", 1, 128)
        self.request_timeout = _number(request_timeout, "request_timeout", 0.01, 3600)
        self.startup_timeout = _number(startup_timeout, "startup_timeout", 0.01, 3600)
        self.lease_seconds = _number(lease_seconds, "lease_seconds", 0.05, 86400)
        self.max_body = _integer(max_body, "max_body", 1024, 64 * 1024 * 1024)
        self.memory_budget_mb = _integer(memory_budget_mb, "memory_budget_mb", 64, 2**30)
        self.cpu_budget = _integer(
            max(os.cpu_count() or 1, 2) if cpu_budget is None else cpu_budget, "cpu_budget", 1, 1024
        )
        if type(mujoco_worker_pool) is not bool:
            raise ValueError("mujoco_worker_pool must be boolean")
        if mujoco_worker_pool and worker_target is not worker_main:
            raise ValueError("MuJoCo pooling requires the standard trusted worker target")
        self.mujoco_worker_pool = mujoco_worker_pool
        self.pool_max_idle = _integer(pool_max_idle, "pool_max_idle", 1, 128)
        self.pool_idle_seconds = _number(pool_idle_seconds, "pool_idle_seconds", 0.05, 86400)
        self.pool_max_uses = _integer(pool_max_uses, "pool_max_uses", 1, 100000)
        if token is not None and (not isinstance(token, str) or not token):
            raise ValueError("token must be a nonempty string when configured")
        self.token = token
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_scene_root = Path(allowed_scene_root or self.root).expanduser().resolve()
        self.sessions_root, self.builds_root = self.root / "sessions", self.root / "builds"
        self.sessions_root.mkdir(exist_ok=True)
        self.builds_root.mkdir(exist_ok=True)
        self.context, self.worker_target = mp.get_context("spawn"), worker_target
        self._sessions: dict[str, SessionProcess] = {}
        self._idle_workers: list[SessionProcess] = []
        self._pool_counts = {"reuse_hits": 0, "cold_starts": 0, "destroyed_idle": 0, "parked": 0}
        self._lock, self._db_lock, self._create_lock = (
            threading.RLock(),
            threading.RLock(),
            threading.Lock(),
        )
        self._stop = threading.Event()
        self._closed = False
        self._build_threads: list[threading.Thread] = []
        self._build_slot = threading.Semaphore(1)
        self._root_lock = self._acquire_root_lock()
        try:
            self._db = sqlite3.connect(self.root / "service.sqlite3", check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, status TEXT, created REAL, updated REAL, metadata TEXT)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS requests (kind TEXT, key TEXT, fingerprint TEXT, target TEXT, PRIMARY KEY(kind,key))"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, session_id TEXT, operation TEXT, created REAL, request_sha256 TEXT, status TEXT, result TEXT)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS builds (id TEXT PRIMARY KEY, status TEXT, metadata TEXT)"
            )
            self._db.execute(
                "UPDATE sessions SET status='lost', updated=? WHERE status IN ('starting','ready','faulted')",
                (time.time(),),
            )
            self._db.execute("UPDATE builds SET status='lost' WHERE status IN ('queued','running')")
            self._db.commit()
        except Exception:
            self._release_root_lock()
            raise
        self._reaper = threading.Thread(
            target=self._reap, daemon=True, name="scene-factory-session-reaper"
        )
        self._reaper.start()

    def _acquire_root_lock(self):
        path = self.root / ".service.lock"
        if path.is_symlink():
            raise ValueError("service lock cannot be a symlink")
        handle = path.open("a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ValueError("service root is already owned by another process") from exc
        return handle

    def _release_root_lock(self) -> None:
        self._root_lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._root_lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._root_lock.fileno(), fcntl.LOCK_UN)
        self._root_lock.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ServiceError(503, "service_closed", "service is shutting down")

    def _safe_path(
        self, value: Any, *, directory: bool = False, preserve_alias: bool = False
    ) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError("input path must be a nonempty string")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.allowed_scene_root / path
        spelling = Path(os.path.abspath(path))
        path = path.resolve()
        if not path.is_relative_to(self.allowed_scene_root) or not (
            path.is_dir() if directory else path.is_file()
        ):
            raise ValueError("input path must exist under allowed_scene_root")
        return spelling if preserve_alias else path

    def _metadata(self, session_id: str) -> dict[str, Any]:
        if not _safe_id(session_id):
            raise ValueError("invalid session_id")
        with self._db_lock:
            row = self._db.execute(
                "SELECT status,created,updated,metadata FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise ServiceError(404, "not_found", "unknown session")
        return {
            **json.loads(row[3]),
            "session_id": session_id,
            "status": row[0],
            "created_at": row[1],
            "updated_at": row[2],
        }

    def _environment_path(self, value: Any) -> Path:
        """Permit trusted scene-root packages and this service's completed builds."""
        try:
            return self._safe_path(value, directory=True)
        except ValueError:
            if not isinstance(value, str) or not value:
                raise
            path = Path(value).expanduser().resolve()
            if not path.is_relative_to(self.builds_root) or not path.is_dir():
                raise
            relative = path.relative_to(self.builds_root)
            if len(relative.parts) != 2 or relative.parts[1] != "environment" or not _safe_id(relative.parts[0]):
                raise
            build = self.build_status(relative.parts[0])
            if build["status"] != "completed" or Path(build["result"]["package_dir"]).resolve() != path:
                raise ValueError("environment is not a completed build owned by this service")
            return path

    def _update(self, session_id: str, status: str, **fields: Any) -> dict[str, Any]:
        with self._db_lock:
            meta = self._metadata(session_id)
            meta.update(fields)
            now = time.time()
            meta.update(status=status, updated_at=now)
            self._db.execute(
                "UPDATE sessions SET status=?,updated=?,metadata=? WHERE id=?",
                (status, now, json.dumps(meta, allow_nan=False), session_id),
            )
            self._db.commit()
            return meta

    def _terminal(self, session_id: str, status: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._sessions.pop(session_id, None)
        return self._update(session_id, status, **fields)

    def _record(
        self, session_id: str, operation: str, request: Any, status: str, result: Any
    ) -> None:
        with self._db_lock:
            self._db.execute(
                "INSERT INTO events(session_id,operation,created,request_sha256,status,result) VALUES(?,?,?,?,?,?)",
                (
                    session_id,
                    operation,
                    time.time(),
                    _digest(request),
                    status,
                    json.dumps(finite_json(result), allow_nan=False),
                ),
            )
            self._db.commit()

    def _replay_create(self, kind: str, key: str, fingerprint: str) -> str | None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT fingerprint,target FROM requests WHERE kind=? AND key=?", (kind, key)
            ).fetchone()
        if row:
            if row[0] != fingerprint:
                raise ServiceError(
                    409, "idempotency_conflict", "idempotency key was used with different content"
                )
            return row[1]
        return None

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        request = _request(
            request,
            {
                "idempotency_key",
                "scene",
                "scene_path",
                "environment_path",
                "backend",
                "backend_options",
                "registry_path",
                "usd_path",
                "resources",
                "lease_seconds",
                "history_limit",
            },
        )
        key, fingerprint = _key(request), _digest(request)
        backend = request.get("backend", "mujoco" if "environment_path" in request else "dry_run")
        if not isinstance(backend, str) or backend.replace("-", "_") not in {
            "dry_run",
            "mujoco",
            "isaac",
        }:
            raise ValueError("backend must be dry-run, mujoco, or isaac")
        backend = backend.replace("-", "_")
        if sum(name in request for name in ("scene", "scene_path", "environment_path")) != 1:
            raise ValueError("create requires exactly one of scene, scene_path or environment_path")
        resources = _request(request.get("resources", {}), {"memory_mb", "cpu_threads"})
        resources = {
            "memory_mb": _integer(
                resources.get("memory_mb", 1024), "memory_mb", 64, self.memory_budget_mb
            ),
            "cpu_threads": _integer(
                resources.get("cpu_threads", 1), "cpu_threads", 1, self.cpu_budget
            ),
        }
        lease = _number(
            request.get("lease_seconds", self.lease_seconds),
            "lease_seconds",
            0.05,
            self.lease_seconds,
        )
        history_limit = _integer(request.get("history_limit", 4096), "history_limit", 1, MAX_SESSION_HISTORY)
        options = self._backend_options(backend, request.get("backend_options", {}))
        if "environment_path" in request and options:
            raise ValueError("backend_options cannot override a locked environment")
        with self._create_lock:
            self._check_open()
            previous = self._replay_create("session", key, fingerprint)
            if previous:
                return self.describe(previous)
            with self._lock:
                live = list(self._sessions.values())
                if len(live) >= self.max_sessions:
                    raise ServiceError(409, "capacity_exceeded", "session capacity reached")
                if backend == "isaac" and any(p.config["backend"] == "isaac" for p in live):
                    raise ServiceError(
                        409, "capacity_exceeded", "only one Isaac worker is admitted"
                    )
                for resource, budget in (
                    ("memory_mb", self.memory_budget_mb),
                    ("cpu_threads", self.cpu_budget),
                ):
                    if (
                        sum(p.config["resources"][resource] for p in live) + resources[resource]
                        > budget
                    ):
                        raise ServiceError(
                            409, "resource_budget_exceeded", f"{resource} admission budget exceeded"
                        )
            sid = uuid.uuid4().hex
            directory = self.sessions_root / sid
            directory.mkdir()
            config, inputs = self._freeze_inputs(sid, directory, backend, request)
            config.update(backend_options=config.pop("locked_backend_options", options), resources=resources,
                          lease_seconds=lease, history_limit=history_limit)
            if self.mujoco_worker_pool and backend == "mujoco":
                from .worker_pool import mujoco_worker_signature

                config["mujoco_pool_signature"] = mujoco_worker_signature(config)["sha256"]
            now = time.time()
            metadata = {
                "session_id": sid,
                "backend": backend,
                "inputs": inputs,
                "resources": resources,
                "resource_enforcement": "admission_only",
                "lease_seconds": lease,
                "history_limit": history_limit,
                "retry_scope": "live_worker; lost after service restart",
                "session": None,
            }
            with self._db_lock:
                self._db.execute(
                    "INSERT INTO sessions(id,status,created,updated,metadata) VALUES(?,?,?,?,?)",
                    (sid, "starting", now, now, json.dumps(metadata, allow_nan=False)),
                )
                self._db.execute(
                    "INSERT INTO requests(kind,key,fingerprint,target) VALUES(?,?,?,?)",
                    ("session", key, fingerprint, sid),
                )
                self._db.commit()
            proc = None
            try:
                proc = self._new_session_process(config)
                with self._lock:
                    # Engine/import startup is bounded by startup_timeout, not
                    # the idle lease. Start the lease when the worker becomes
                    # an admitted ready session, including a pooled rebind.
                    proc.last_used = time.monotonic()
                    self._sessions[sid] = proc
                self._update(sid, "ready", session=proc.descriptor, worker_pid=proc.process.pid,
                             worker_reused=proc.reused, worker_pool_compatibility_sha256=config.get("mujoco_pool_signature"))
            except Exception as exc:
                if proc is not None:
                    proc.kill()
                self._terminal(
                    sid, "failed", error={"type": type(exc).__name__, "message": str(exc)}
                )
                raise ServiceError(500, "worker_start_failed", str(exc)) from exc
            return self.describe(sid)

    def _resident_fits(self, incoming: dict[str, int], *, exclude: SessionProcess | None = None) -> bool:
        """Called under _lock; idle workers consume the same admission budgets."""
        resident = [proc for proc in [*self._sessions.values(), *self._idle_workers] if proc is not exclude]
        if len(resident) + 1 > self.max_sessions:
            return False
        return all(sum(proc.config["resources"][name] for proc in resident) + incoming[name] <= budget
                   for name, budget in (("memory_mb", self.memory_budget_mb), ("cpu_threads", self.cpu_budget)))

    def _destroy_idle_locked(self, proc: SessionProcess) -> None:
        self._idle_workers.remove(proc)
        proc.kill()
        self._pool_counts["destroyed_idle"] += 1

    def _prune_idle_locked(self) -> None:
        now = time.monotonic()
        for proc in list(self._idle_workers):
            if (not proc.process.is_alive() or proc.failed or proc.pool_uses >= self.pool_max_uses
                    or now - (proc.idle_since or now) > self.pool_idle_seconds):
                self._destroy_idle_locked(proc)

    def _new_session_process(self, config: dict[str, Any]) -> SessionProcess:
        candidate = None
        with self._lock:
            self._prune_idle_locked()
            signature = config.get("mujoco_pool_signature")
            if signature:
                for proc in reversed(self._idle_workers):
                    if proc.config.get("mujoco_pool_signature") == signature:
                        candidate = proc
                        self._idle_workers.remove(proc)
                        break
            # A parked process is accounted until it is killed. Reclaim idle
            # capacity before spawning, including when the new request is Isaac.
            while self._idle_workers and not self._resident_fits(config["resources"]):
                self._destroy_idle_locked(self._idle_workers[0])
            if not self._resident_fits(config["resources"]):
                if candidate is not None:
                    candidate.kill()
                raise ServiceError(409, "resource_budget_exceeded", "resident worker admission budget exceeded")
        if candidate is None:
            result = SessionProcess(self, config)
            if signature:
                self._pool_counts["cold_starts"] += 1
            return result
        with candidate.lock:
            try:
                candidate.call("pool_health", timeout=min(self.request_timeout, 5))
                candidate.config, candidate.session_id = config, config["session_id"]
                candidate.call("pool_bind", {"config": config}, timeout=self.startup_timeout)
                if candidate.descriptor.get("session_id") != config["session_id"] or candidate.descriptor.get("episode_id") is not None:
                    raise RuntimeError("pooled worker failed to isolate the new session")
                candidate.pool_uses += 1
                candidate.idle_since, candidate.reused = None, True
                self._pool_counts["reuse_hits"] += 1
                return candidate
            except Exception:
                candidate.kill()
                raise

    def worker_pool_status(self) -> dict[str, Any]:
        with self._lock:
            return {"enabled": self.mujoco_worker_pool, "supported_backends": ["mujoco"] if self.mujoco_worker_pool else [],
                    "compatibility": "exact_scene_mjcf_registry_runtime_sources_engine_settings_resources",
                    "idle_count": len(self._idle_workers), "idle_pids": [proc.process.pid for proc in self._idle_workers],
                    "max_idle": self.pool_max_idle, "idle_seconds": self.pool_idle_seconds, "max_uses": self.pool_max_uses,
                    "resource_enforcement": "admission_only_active_plus_idle", **self._pool_counts}

    def _backend_options(self, backend: str, value: Any) -> dict[str, Any]:
        allowed = {
            "dry_run": {"max_steps"},
            "mujoco": {"max_steps", "frame_skip", "render_width", "render_height"},
            "isaac": {"max_steps", "physics_dt", "enable_rgbd", "robot_asset_root", "enable_general_observation"},
        }[backend]
        value = _request(value, allowed)
        result = {}
        for name, item in value.items():
            if name == "robot_asset_root":
                result[name] = str(self._safe_path(item, directory=True, preserve_alias=True))
            elif name in {"enable_rgbd", "enable_general_observation"}:
                if not isinstance(item, bool):
                    raise ValueError(f"{name} must be boolean")
                result[name] = item
            elif name == "physics_dt":
                result[name] = _number(item, name, 1e-6, 1)
            else:
                limit = (
                    4096
                    if name.startswith("render_")
                    else (1000 if name == "frame_skip" else 10_000_000)
                )
                result[name] = _integer(item, name, 1, limit)
        return result

    def _freeze_inputs(
        self, sid: str, directory: Path, backend: str, request: dict
    ) -> tuple[dict, dict]:
        environment = None
        lock = None
        source_registry = (
            self._safe_path(request["registry_path"])
            if "registry_path" in request
            else default_registry_path().resolve()
        )
        if "environment_path" in request:
            from .environment_package import verify_environment

            environment = self._environment_path(request["environment_path"])
            verification = verify_environment(environment)
            if not verification.get("valid"):
                raise ValueError("environment package failed verification")
            if "registry_path" in request or "usd_path" in request:
                raise ValueError(
                    "environment_path cannot be combined with registry_path or usd_path"
                )
            import shutil

            frozen_environment = directory / "environment"
            frozen_environment.mkdir()
            manifest = json.loads((environment / "manifest.json").read_text(encoding="utf-8"))
            lock = json.loads((environment / manifest["lock_path"]).read_text(encoding="utf-8"))
            if lock["backend"] != backend:
                raise ValueError("requested backend does not match the environment lock")
            for name in [
                "manifest.json",
                manifest["lock_path"],
                *[item["path"] for item in lock["dependencies"]],
            ]:
                source, target = (
                    (environment / name).resolve(),
                    (frozen_environment / name).resolve(),
                )
                if not source.is_relative_to(environment) or not target.is_relative_to(
                    frozen_environment
                ):
                    raise ValueError("environment path escapes package")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if not verify_environment(frozen_environment).get("valid"):
                raise ValueError("environment changed while being frozen")
            environment = frozen_environment
            scene_path = environment / lock["scene"]["layout_path"]
            source_registry = environment / lock["scene"]["registry_path"]
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
        elif "scene_path" in request:
            scene = json.loads(
                self._safe_path(request["scene_path"]).read_text(encoding="utf-8-sig")
            )
        else:
            scene = request["scene"]
        scene = finite_json(scene, "scene")
        if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
            raise ValueError("scene must be a layout object with objects array")
        target = directory / "layout.json"
        write_json_atomic(target, scene)
        frozen_registry = directory / "registry.jsonl"
        frozen_registry.write_bytes(source_registry.read_bytes())
        usd = (
            self._safe_path(request["usd_path"], preserve_alias=True)
            if "usd_path" in request
            else None
        )
        if environment is not None and backend == "isaac":
            usd = environment / lock["scene"]["compiled_path"]
        if backend == "isaac" and usd is None:
            raise ValueError("Isaac sessions require usd_path and an Isaac Python service process")
        inputs = {
            "scene_sha256": _digest(scene),
            "registry_sha256": hashlib.sha256(frozen_registry.read_bytes()).hexdigest(),
            "environment_path": str(environment) if environment else None,
            "asset_closure": "verified_package" if environment else "not_locked",
            "usd_sha256": hashlib.sha256(usd.read_bytes()).hexdigest() if usd else None,
        }
        config = {
            "session_id": sid,
            "session_dir": str(directory),
            "scene_path": str(target),
            "backend": backend,
            "registry_path": str(frozen_registry),
            "registry_base_dir": str(source_registry.parent),
            "usd_path": str(usd) if usd else None,
        }
        if environment is not None:
            config["environment_path"] = str(environment)
            if backend == "isaac":
                config["locked_backend_options"] = {
                    **lock["backend_options"],
                    "robot_asset_root": str(environment / lock["scene"]["robot_asset_root"]),
                }
        write_json_atomic(directory / "inputs.json", inputs)
        return config, inputs

    def _live(self, sid: str) -> SessionProcess:
        self._check_open()
        meta = self._metadata(sid)
        with self._lock:
            proc = self._sessions.get(sid)
        if proc is None:
            raise ServiceError(
                410, "session_unavailable", f"session is {meta['status']}; create a new session"
            )
        return proc

    def call(
        self, sid: str, operation: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if operation not in _OPERATIONS:
            raise ServiceError(404, "not_found", "unknown session operation")
        request = finite_json(request or {})
        if not isinstance(request, dict):
            raise ValueError("operation request must be an object")
        proc = self._live(sid)
        if not proc.lock.acquire(timeout=self.request_timeout):
            raise ServiceError(503, "session_busy", "another request still owns this session")
        try:
            self._check_open()
            with self._lock:
                if self._sessions.get(sid) is not proc:
                    raise ServiceError(410, "session_unavailable", "session was closed")
            try:
                result = proc.call(operation, request)
            except ServiceError as exc:
                self._record(
                    sid, operation, request, "failed", {"code": exc.code, "message": str(exc)}
                )
                if not proc.failed:
                    self._update(
                        sid,
                        "faulted" if proc.descriptor.get("faulted") else "ready",
                        session=proc.descriptor,
                    )
                raise
            self._update(
                sid,
                "faulted" if proc.descriptor.get("faulted") else "ready",
                session=proc.descriptor,
            )
            self._record(sid, operation, request, "completed", result)
            return result
        finally:
            proc.lock.release()

    def describe(self, sid: str) -> dict[str, Any]:
        self._check_open()
        return {"schema_version": SERVICE_SCHEMA, **self._metadata(sid)}

    def list(self) -> dict[str, Any]:
        self._check_open()
        with self._db_lock:
            ids = [
                row[0] for row in self._db.execute("SELECT id FROM sessions ORDER BY created DESC")
            ]
        return {"schema_version": SERVICE_SCHEMA, "sessions": [self._metadata(sid) for sid in ids],
                "worker_pool": self.worker_pool_status()}

    def close_session(self, sid: str, *, reason: str = "closed") -> dict[str, Any]:
        meta = self._metadata(sid)
        with self._lock:
            proc = self._sessions.get(sid)
        if proc is None:
            return {"session_id": sid, "closed": True, "status": meta["status"]}
        with proc.lock:
            with self._lock:
                if self._sessions.get(sid) is not proc:
                    return {
                        "session_id": sid,
                        "closed": True,
                        "status": self._metadata(sid)["status"],
                    }
            error = None
            reuse_rejection = None
            pooled = False
            try:
                eligible = (self.mujoco_worker_pool and reason == "closed" and not self._stop.is_set()
                            and not self._create_lock.locked() and proc.config.get("mujoco_pool_signature")
                            and proc.pool_uses < self.pool_max_uses and not proc.failed
                            and not proc.descriptor.get("faulted")
                            and not proc.descriptor.get("worker_runtime", {}).get("pool_tainted"))
                if eligible:
                    try:
                        proc.call("pool_park", timeout=min(self.request_timeout, 5))
                        pooled = True
                    except ServiceError as exc:
                        # Clean session closure still succeeds when reuse is
                        # unavailable (e.g. no model has been initialized yet).
                        proc.kill()
                        if proc.failed:
                            raise
                        reuse_rejection = {"code": exc.code, "message": str(exc)}
                else:
                    proc.call("close", timeout=min(self.request_timeout, 5))
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                if not pooled:
                    proc.kill()
            terminal = "failed" if error else reason
            with self._lock:
                self._terminal(sid, terminal, close_error=error, worker_reuse_rejection=reuse_rejection,
                               session={**proc.descriptor, "closed": True},
                               worker_disposition="pooled" if pooled else "destroyed")
                if pooled:
                    # Do not grow an idle pool during an in-flight creation;
                    # that startup already reserved capacity from live workers.
                    if (self._stop.is_set() or self._create_lock.locked()
                            or not self._resident_fits(proc.config["resources"])):
                        pooled = False
                        proc.kill()
                    else:
                        while len(self._idle_workers) >= self.pool_max_idle:
                            self._destroy_idle_locked(self._idle_workers[0])
                        proc.idle_since = time.monotonic()
                        self._idle_workers.append(proc)
                        self._pool_counts["parked"] += 1
                if not pooled:
                    self._update(sid, terminal, worker_disposition="destroyed")
            return {"session_id": sid, "closed": True, "status": terminal, "error": error,
                    "worker_disposition": "pooled" if pooled else "destroyed"}

    def artifact(self, sid: str, name: str) -> Path:
        self._metadata(sid)
        if not name or "\\" in name or ":" in name or "/" in name:
            raise ValueError("invalid artifact path")
        root = (self.sessions_root / sid / "artifacts").resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ServiceError(404, "not_found", "unknown artifact")
        return path

    def build(self, request: dict[str, Any]) -> dict[str, Any]:
        request = _request(
            request, {"idempotency_key", "scene", "scene_path", "registry_path", "backend",
                      "usd_path", "robot_asset_root", "backend_options"}
        )
        key, fingerprint = _key(request), _digest(request)
        if sum(field in request for field in ("scene", "scene_path")) != 1:
            raise ValueError("build requires exactly one of scene or scene_path")
        backend = request.get("backend", "mujoco")
        if backend not in {"mujoco", "isaac"}:
            raise ValueError("environment builds support MuJoCo or Isaac")
        if backend == "isaac" and not all(key in request for key in ("usd_path", "robot_asset_root")):
            raise ValueError("Isaac builds require usd_path and robot_asset_root")
        if backend == "mujoco" and any(key in request for key in ("usd_path", "robot_asset_root", "backend_options")):
            raise ValueError("Isaac build inputs cannot be supplied to MuJoCo")
        build_options = self._backend_options("isaac", request.get("backend_options", {})) if backend == "isaac" else {}
        if "robot_asset_root" in build_options:
            raise ValueError("use the top-level robot_asset_root in builds")
        with self._create_lock:
            self._check_open()
            previous = self._replay_create("build", key, fingerprint)
            if previous:
                return self.build_status(previous)
            with self._db_lock:
                pending = self._db.execute(
                    "SELECT COUNT(*) FROM builds WHERE status IN ('queued','running')"
                ).fetchone()[0]
            if pending >= 16:
                raise ServiceError(
                    409, "build_capacity_exceeded", "at most 16 queued/running builds are admitted"
                )
            bid = uuid.uuid4().hex
            directory = self.builds_root / bid
            directory.mkdir()
            scene = (
                json.loads(self._safe_path(request["scene_path"]).read_text(encoding="utf-8-sig"))
                if "scene_path" in request
                else request["scene"]
            )
            scene = finite_json(scene)
            if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
                raise ValueError("build scene must contain an objects array")
            scene_path = directory / "layout.json"
            write_json_atomic(scene_path, scene)
            registry = (
                self._safe_path(request["registry_path"])
                if "registry_path" in request
                else default_registry_path().resolve()
            )
            config = {
                "scene_path": str(scene_path),
                "registry_path": str(registry),
                "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                "output_dir": str(directory / "environment"),
                "cache_dir": str(self.root / "build_cache"),
                "backend": backend,
            }
            if backend == "isaac":
                config.update(
                    usd_path=str(self._safe_path(request["usd_path"], preserve_alias=True)),
                    robot_asset_root=str(self._safe_path(request["robot_asset_root"], directory=True, preserve_alias=True)),
                    backend_options=build_options,
                )
            meta = {
                "build_id": bid,
                "created_at": time.time(),
                "inputs": {
                    "scene_sha256": _digest(scene),
                    "registry_sha256": config["registry_sha256"],
                },
            }
            with self._db_lock:
                self._db.execute(
                    "INSERT INTO builds(id,status,metadata) VALUES(?,?,?)",
                    (bid, "queued", json.dumps(meta)),
                )
                self._db.execute(
                    "INSERT INTO requests(kind,key,fingerprint,target) VALUES(?,?,?,?)",
                    ("build", key, fingerprint, bid),
                )
                self._db.commit()
            thread = threading.Thread(target=self._run_build, args=(bid, config), daemon=True)
            self._build_threads.append(thread)
            thread.start()
            return self.build_status(bid)

    def build_status(self, build_id: str) -> dict[str, Any]:
        if not _safe_id(build_id):
            raise ValueError("invalid build ID")
        with self._db_lock:
            row = self._db.execute(
                "SELECT status,metadata FROM builds WHERE id=?", (build_id,)
            ).fetchone()
        if row is None:
            raise ServiceError(404, "not_found", "unknown build")
        return {"schema_version": SERVICE_SCHEMA, **json.loads(row[1]), "status": row[0]}

    def _update_build(self, build_id: str, status: str, **fields: Any) -> None:
        with self._db_lock:
            meta = self.build_status(build_id)
            meta.update(fields, updated_at=time.time())
            self._db.execute(
                "UPDATE builds SET status=?,metadata=? WHERE id=?",
                (status, json.dumps(meta, allow_nan=False), build_id),
            )
            self._db.commit()

    def _run_build(self, build_id: str, config: dict[str, Any]) -> None:
        acquired = False
        process = parent = None
        try:
            while not self._stop.is_set():
                if self._build_slot.acquire(timeout=0.1):
                    acquired = True
                    break
            if not acquired or self._stop.is_set():
                raise RuntimeError("build interrupted by service shutdown")
            self._update_build(build_id, "running")
            if (
                hashlib.sha256(Path(config["registry_path"]).read_bytes()).hexdigest()
                != config["registry_sha256"]
            ):
                raise ValueError("registry changed after build was queued")
            parent, child = self.context.Pipe()
            process = self.context.Process(
                target=build_worker_main, args=(child, config), daemon=True
            )
            process.start()
            child.close()
            started = time.monotonic()
            while not parent.poll(0.1):
                if self._stop.is_set() or time.monotonic() - started > 300:
                    raise TimeoutError("build timed out or service shut down")
                if not process.is_alive():
                    raise RuntimeError("build worker exited without a result")
            result = parent.recv()
            if not result.get("ok"):
                raise RuntimeError(result.get("error", {}).get("message", "build worker failed"))
            if (
                hashlib.sha256(Path(config["registry_path"]).read_bytes()).hexdigest()
                != config["registry_sha256"]
            ):
                raise ValueError("registry changed during build")
            self._update_build(build_id, "completed", result=finite_json(result["result"]))
        except Exception as exc:
            self._update_build(
                build_id, "failed", error={"type": type(exc).__name__, "message": str(exc)}
            )
        finally:
            if parent is not None:
                parent.close()
            if process is not None:
                process.join(0.5)
                if process.is_alive():
                    process.terminate()
                    process.join(2)
                if process.is_alive():
                    process.kill()
                    process.join(2)
            if acquired:
                self._build_slot.release()

    def _reap(self) -> None:
        while not self._stop.wait(min(0.5, self.lease_seconds / 4)):
            with self._lock:
                processes = list(self._sessions.items())
            for sid, proc in processes:
                if not proc.lock.acquire(blocking=False):
                    continue
                try:
                    if not proc.process.is_alive():
                        proc._fail("worker_failed", "worker exited without a response")
                    elif time.monotonic() - proc.last_used > proc.config["lease_seconds"]:
                        self.close_session(sid, reason="expired")
                except (ServiceError, sqlite3.Error):
                    pass
                finally:
                    proc.lock.release()
            with self._lock:
                self._prune_idle_locked()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._create_lock:
            pass  # Let interrupted startup/build submissions finish recording.
        self._reaper.join(timeout=15)
        with self._lock:
            ids = list(self._sessions)
        for sid in ids:
            self.close_session(sid)
        with self._lock:
            for proc in list(self._idle_workers):
                self._destroy_idle_locked(proc)
        for thread in self._build_threads:
            thread.join(timeout=5)
        with self._db_lock:
            self._db.close()
        self._release_root_lock()

    def __enter__(self) -> "RuntimeService":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


SessionManager = RuntimeService


class RuntimeHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], manager: RuntimeService):
        if address[0] not in {"127.0.0.1", "localhost", "::1"} and not manager.token:
            raise ValueError("non-loopback bind requires a configured bearer token")
        self.manager = manager
        super().__init__(address, RuntimeRequestHandler)


class RuntimeRequestHandler(http.server.BaseHTTPRequestHandler):
    server: RuntimeHTTPServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(finite_json(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise ServiceError(400, "invalid_request", "chunked bodies are unsupported")
        if self.headers.get_content_type() != "application/json":
            raise ServiceError(415, "invalid_content_type", "Content-Type must be application/json")
        if "Content-Length" not in self.headers:
            raise ServiceError(411, "length_required", "Content-Length is required")
        length = int(self.headers["Content-Length"])
        if not 0 < length <= self.server.manager.max_body:
            raise ServiceError(413, "body_too_large", "request body exceeds configured limit")

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON object key")
                result[key] = value
            return result

        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("incomplete request body")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return finite_json(value, "request")

    def _handle(self, method: str) -> None:
        manager = self.server.manager
        try:
            if self.headers.get("Origin"):
                raise ServiceError(
                    403, "cross_origin_rejected", "browser cross-origin requests are unsupported"
                )
            if manager.token and not secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + manager.token
            ):
                raise ServiceError(401, "unauthorized", "Bearer token required")
            parsed = urlsplit(self.path)
            parts = [unquote(p) for p in parsed.path.split("/") if p]
            query = parse_qs(parsed.query, keep_blank_values=True)
            body = self._body() if method == "POST" else {}
            if query and not (method == "GET" and len(parts) == 4 and parts[3] == "observation"):
                raise ValueError("unexpected query parameters")
            status = 200
            if parsed.path == "/healthz" and method == "GET":
                result = {"status": "ok", "schema_version": SERVICE_SCHEMA,
                          "worker_pool": manager.worker_pool_status()}
            elif parts == ["v1", "sessions"] and method == "GET":
                result = manager.list()
            elif parts == ["v1", "sessions"] and method == "POST":
                result, status = manager.create(body), 201
            elif parts == ["v1", "builds"] and method == "POST":
                result, status = manager.build(body), 202
            elif len(parts) == 3 and parts[:2] == ["v1", "builds"] and method == "GET":
                result = manager.build_status(parts[2])
            elif len(parts) == 3 and parts[:2] == ["v1", "sessions"] and method == "GET":
                result = manager.describe(parts[2])
            elif (len(parts) == 4 and parts[:2] == ["v1", "sessions"]
                  and method == "GET" and parts[3] == "world-state"):
                result = manager.call(parts[2], "world_state")
            elif len(parts) == 3 and parts[:2] == ["v1", "sessions"] and method == "DELETE":
                result = manager.close_session(parts[2])
            elif (
                len(parts) == 4
                and parts[:2] == ["v1", "sessions"]
                and method == "POST"
                and parts[3] in {"reset", "step", "snapshots", "restore"}
            ):
                result = manager.call(
                    parts[2], "snapshot" if parts[3] == "snapshots" else parts[3], body
                )
            elif (
                len(parts) == 4
                and parts[:2] == ["v1", "sessions"]
                and method == "GET"
                and parts[3] == "observation"
            ):
                if query.keys() - {"modalities", "camera"}:
                    raise ValueError("unknown observation query parameters")
                modalities = [
                    word for item in query.get("modalities", []) for word in item.split(",")
                ]
                observation_request = {"modalities": modalities} if modalities else {}
                if "camera" in query:
                    if len(query["camera"]) != 1:
                        raise ValueError("camera must be specified once")
                    observation_request["camera"] = query["camera"][0]
                result = manager.call(parts[2], "observation", observation_request)
            elif (
                len(parts) == 5
                and parts[:2] == ["v1", "sessions"]
                and method == "GET"
                and parts[3] == "artifacts"
            ):
                path = manager.artifact(parts[2], "/".join(parts[4:]))
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(data)
                return
            else:
                raise ServiceError(404, "not_found", "route not found")
            self._reply(status, result)
        except ServiceError as exc:
            self._reply(
                exc.status,
                {
                    "schema_version": SERVICE_SCHEMA,
                    "error": {"code": exc.code, "message": str(exc)},
                },
            )
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            self._reply(
                400,
                {
                    "schema_version": SERVICE_SCHEMA,
                    "error": {"code": "invalid_request", "message": str(exc)},
                },
            )
        except Exception as exc:
            self._reply(
                500,
                {
                    "schema_version": SERVICE_SCHEMA,
                    "error": {"code": "internal_error", "message": str(exc)},
                },
            )

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_DELETE(self) -> None:
        self._handle("DELETE")


def serve(manager: RuntimeService, host: str = "127.0.0.1", port: int = 8765) -> RuntimeHTTPServer:
    return RuntimeHTTPServer((host, port), manager)
