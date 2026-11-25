"""Serialized, process-local sessions for the first runtime contract.

Retry protection lasts for this Python object only. This is not a persistent
service, a worker pool, or a simulator checkpoint implementation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from typing import Any
from pathlib import Path

from .agent import DryRunBackend, SceneFactoryEnv
from .backends.mujoco import MujocoBackend
from .session_contracts import (
    backend_capabilities,
    make_observation_envelope,
    translate_action,
    validate_reset_request,
    validate_step_request,
    finite_json,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


class RuntimeSession:
    """Own one environment; serialize reset/step/observe/close operations.

    Sequence IDs start at zero in each episode. Failed simulator operations
    poison the session until a successful reset with a new idempotency key.
    Request histories are bounded by rejecting new requests, never by silently
    evicting a key that could cause an action to run twice.
    """

    def __init__(self, env: SceneFactoryEnv, *, history_limit: int = 4096) -> None:
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit < 1:
            raise ValueError("history_limit must be a positive integer")
        if isinstance(env.backend, DryRunBackend):
            backend = "dry_run"
        elif isinstance(env.backend, MujocoBackend):
            backend = "mujoco"
        else:
            from .backends.isaac import IsaacSimBackend

            if not isinstance(env.backend, IsaacSimBackend):
                raise ValueError("session backend has no registered capabilities")
            backend = "isaac"
        self.env = env
        self.backend = backend
        self.session_id = uuid.uuid4().hex
        self.episode_id: str | None = None
        self._capabilities = backend_capabilities(backend)
        self._lock = threading.RLock()
        self._history_limit = history_limit
        self._resets: dict[str, tuple[str, str | None]] = {}
        self._steps: dict[str, tuple[str, dict[str, Any] | None]] = {}
        self._reset_response: dict[str, Any] | None = None
        self._observation: dict[str, Any] | None = None
        self._next_sequence = 0
        self._step_index = 0
        self._state_version = 0
        self._timestep: float | None = None
        self._closed = False
        self._faulted = False
        self._done = False
        self._snapshots: dict[str, tuple[str, dict[str, Any]]] = {}
        self._restores: dict[str, tuple[str, str | None, dict[str, Any] | None]] = {}
        self._reset_record: dict[str, Any] | None = None
        self.artifact_root: Path | None = None

    @property
    def capabilities(self) -> dict[str, Any]:
        return copy.deepcopy(self._capabilities)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id, "episode_id": self.episode_id,
                "backend": self.backend, "capabilities": self.capabilities,
                "closed": self._closed, "faulted": self._faulted,
                "next_sequence_id": self._next_sequence,
                "history_limit": self._history_limit,
                "retry_scope": "current_process_session",
            }

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_open()
            request = validate_reset_request(request)
            key, fingerprint = request["idempotency_key"], _digest(request)
            previous = self._resets.get(key)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise ValueError("idempotency_key was already used with a different reset")
                if previous[1] is None:
                    raise RuntimeError("previous reset failed; use a new idempotency_key")
                if previous[1] != self.episode_id:
                    raise ValueError("reset belongs to a stale episode")
                if self._faulted:
                    raise RuntimeError("session operation failed; reset with a new idempotency_key")
                return copy.deepcopy(self._reset_response)
            if len(self._resets) >= self._history_limit:
                raise RuntimeError("reset history limit reached; create a new session")
            if self.backend == "isaac" and self.episode_id is not None:
                raise ValueError("Isaac requires a new session for a new episode; Kit cannot restart safely in this process")
            # Reserve before calling the backend, including failures with partial effects.
            self._resets[key] = (fingerprint, None)
            previous_attempts = self.env.reset_attempts
            previous_faulted = self._faulted
            reset_returned = False
            self._faulted = True
            try:
                observation, info = self.env.reset(seed=request["seed"], options=request["options"])
                reset_returned = True
                self.episode_id = uuid.uuid4().hex
                bind_episode = getattr(self.env.backend, "bind_episode_id", None)
                if callable(bind_episode):
                    observation = bind_episode(self.episode_id, observation)
                self._next_sequence = 0
                self._step_index = 0
                self._state_version += 1
                self._timestep = info.get("timestep") if self.backend == "mujoco" else None
                self._done = False
                response = self._wrap(observation, info=info)
            except Exception:
                if not reset_returned and self.env.reset_attempts == previous_attempts:
                    # Env validation happens before a backend reset and preserves
                    # its active scene; keep the matching session state as well.
                    self._faulted = previous_faulted
                    del self._resets[key]
                else:
                    self._observation = None
                raise
            self._steps.clear()
            self._snapshots.clear()
            self._reset_record = copy.deepcopy(info["episode_reset"])
            self._resets[key] = (fingerprint, self.episode_id)
            self._reset_response = copy.deepcopy(response)
            self._observation = copy.deepcopy(response)
            self._faulted = False
            return response

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_ready()
            request = validate_step_request(request, self._capabilities)
            if request["episode_id"] != self.episode_id:
                raise ValueError("step belongs to a stale or unknown episode")
            key, fingerprint = request["idempotency_key"], _digest(request)
            previous = self._steps.get(key)
            if previous is not None:
                if previous[0] != fingerprint:
                    raise ValueError("idempotency_key was already used with a different step")
                if previous[1] is None:
                    raise RuntimeError("previous step failed; reset is required")
                return copy.deepcopy(previous[1])
            if self._done:
                raise RuntimeError("episode ended; reset is required")
            if request["sequence_id"] != self._next_sequence:
                raise ValueError(f"expected sequence_id {self._next_sequence}")
            if len(self._steps) >= self._history_limit:
                raise RuntimeError("step history limit reached; reset is required")
            action = translate_action(request["action"], self.backend)
            if request["action"]["mode"] == "object_forces":
                objects = {item["object_id"]: item for item in self.env.backend.scene["objects"]}
                for object_id in request["action"]["payload"]["object_forces"]:
                    if object_id not in objects or not objects[object_id].get("dynamic"):
                        raise ValueError(f"force target must be a dynamic scene object: {object_id}")
            self._steps[key] = (fingerprint, None)
            total_reward = 0.0
            try:
                for index in range(request["steps"]):
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    self._step_index += 1
                    self._state_version += 1
                    if terminated or truncated:
                        break
                self._next_sequence += 1
                self._done = bool(terminated or truncated)
                response = self._wrap(
                    observation, reward=total_reward, terminated=terminated, truncated=truncated,
                    info={**info, "executed_steps": index + 1, "requested_steps": request["steps"]},
                )
            except Exception:
                self._faulted = True
                self._observation = None
                raise
            self._steps[key] = (fingerprint, copy.deepcopy(response))
            self._observation = copy.deepcopy(response)
            return response

    def observe(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read committed state; optionally capture images without advancing physics."""
        with self._lock:
            self._check_ready()
            response = copy.deepcopy(self._observation)
            if request is not None:
                from .sensor_artifacts import capture_observation_images

                response = capture_observation_images(self, request, response)
            return response

    def snapshot(self, request: dict[str, Any]) -> dict[str, Any]:
        """Capture a rebuild recipe or backend-scoped serialized running state."""
        with self._lock:
            self._check_ready()
            request = self._state_request(request, {"idempotency_key", "level"})
            level = request.setdefault("level", "runtime_state")
            if level not in self._capabilities["snapshot_levels"]:
                raise ValueError(f"snapshot level {level!r} is unsupported by {self.backend}")
            key, fingerprint = request["idempotency_key"], _digest(request)
            if key in self._snapshots:
                prior = self._snapshots[key]
                if prior[0] != fingerprint:
                    raise ValueError("idempotency_key reused with a different snapshot request")
                return copy.deepcopy(prior[1])
            if len(self._snapshots) >= self._history_limit:
                raise RuntimeError("snapshot history limit reached; reset is required")
            result = {
                "schema_version": "scene_factory.session_snapshot.v1", "level": level,
                "backend": self.backend, "scene_sha256": self.env.scene_sha256,
                "source_scene_sha256": self._reset_record["source_scene_sha256"],
                "seed": self._reset_record["seed"], "options": self._reset_record["options"],
                "step_index": self._step_index if level == "runtime_state" else 0,
                "terminated": self._observation["terminated"] if level == "runtime_state" else False,
                "truncated": self._observation["truncated"] if level == "runtime_state" else False,
                "runtime_state": self.env.backend.capture_state() if level == "runtime_state" else None,
                "scope": "same_backend_compatible_model" if level == "runtime_state" else "episode_initial_conditions",
            }
            result = finite_json(result, "snapshot")
            result["sha256"] = _digest(result)
            self._snapshots[key] = (fingerprint, copy.deepcopy(result))
            return result

    def restore(self, request: dict[str, Any]) -> dict[str, Any]:
        """Restore into a new episode so old action retries cannot cross a fork."""
        with self._lock:
            self._check_ready()
            request = self._state_request(request, {"idempotency_key", "snapshot"})
            state = request.get("snapshot")
            if not isinstance(state, dict):
                raise ValueError("restore requires a snapshot object")
            expected = {"schema_version", "level", "backend", "scene_sha256", "source_scene_sha256",
                        "seed", "options", "step_index", "terminated", "truncated", "runtime_state", "scope", "sha256"}
            if set(state) != expected or state.get("schema_version") != "scene_factory.session_snapshot.v1":
                raise ValueError("unsupported or malformed snapshot")
            if state["sha256"] != _digest({k: v for k, v in state.items() if k != "sha256"}):
                raise ValueError("snapshot digest mismatch")
            if state["backend"] != self.backend or state["scene_sha256"] != self.env.scene_sha256:
                raise ValueError("snapshot backend or initialized scene mismatch")
            level = state["level"]
            if level not in self._capabilities["snapshot_levels"]:
                raise ValueError("unsupported restore level")
            if isinstance(state["step_index"], bool) or not isinstance(state["step_index"], int) or state["step_index"] < 0:
                raise ValueError("snapshot step_index must be nonnegative")
            if any(not isinstance(state[k], bool) for k in ("terminated", "truncated")):
                raise ValueError("snapshot termination flags must be booleans")
            if level == "rebuild":
                if state["step_index"] != 0 or state["terminated"] or state["truncated"] or state["runtime_state"] is not None or state["scope"] != "episode_initial_conditions":
                    raise ValueError("rebuild snapshot must describe initial episode conditions")
                from .episode_reset import prepare_episode_reset

                _, rebuilt = prepare_episode_reset(
                    json.loads(self.env.scene_path.read_text(encoding="utf-8")),
                    seed=state["seed"], options=state["options"],
                )
                if rebuilt["episode_scene_sha256"] != state["scene_sha256"]:
                    raise ValueError("snapshot source inputs changed")
            else:
                runtime = state["runtime_state"]
                if not isinstance(runtime, dict) or runtime.get("steps") != state["step_index"] or state["scope"] != "same_backend_compatible_model":
                    raise ValueError("snapshot metadata does not match runtime state")
                task_status = runtime.get("task_status")
                settings = runtime.get("settings")
                if not isinstance(task_status, dict) or not isinstance(settings, dict):
                    raise ValueError("snapshot is missing runtime task/settings")
                terminated = task_status.get("task_success", False)
                if not isinstance(terminated, bool):
                    raise ValueError("snapshot task_success must be a boolean")
                truncated = state["step_index"] >= self.env.backend.max_steps and not terminated
                if state["terminated"] != terminated or state["truncated"] != truncated:
                    raise ValueError("snapshot termination flags do not match runtime state")
            key, fingerprint = request["idempotency_key"], _digest(request)
            if key in self._restores:
                old_fingerprint, episode, response = self._restores[key]
                if old_fingerprint != fingerprint:
                    raise ValueError("idempotency_key reused with a different restore request")
                if episode != self.episode_id or response is None:
                    raise RuntimeError("restore is stale or previously failed; use a new key")
                return copy.deepcopy(response)
            if len(self._restores) >= self._history_limit:
                raise RuntimeError("restore history limit reached; create a new session")
            self._restores[key] = (fingerprint, None, None)
            try:
                if level == "rebuild":
                    observation, info = self.env.reset(seed=state["seed"], options=state["options"])
                    if self.env.scene_sha256 != state["scene_sha256"]:
                        raise ValueError("rebuild restored a different scene")
                else:
                    observation, info = self.env.backend.restore_state(state["runtime_state"])
                    observation = self.env.transform_observation(observation)
                self.episode_id = uuid.uuid4().hex
                self._step_index = state["step_index"]
                self._next_sequence = 0
                self._state_version += 1
                self._done = state["terminated"] or state["truncated"]
                response = self._wrap(observation, info={**info, "restored_level": level},
                                      terminated=state["terminated"], truncated=state["truncated"])
            except Exception:
                # Backend-level validation is transactional, but any unexpected
                # restore failure must not expose a possibly mismatched old state.
                self._faulted = True
                self._observation = None
                raise
            self._steps.clear()
            self._snapshots.clear()
            self._observation = copy.deepcopy(response)
            self._restores[key] = (fingerprint, self.episode_id, copy.deepcopy(response))
            return response

    @staticmethod
    def _state_request(request: Any, allowed: set[str]) -> dict[str, Any]:
        request = finite_json(request)
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("invalid state operation request")
        key = request.get("idempotency_key")
        if not isinstance(key, str) or not key.strip() or len(key) > 256:
            raise ValueError("idempotency_key must be a nonempty string up to 256 characters")
        return request

    def detach_mujoco_backend_for_pool(self) -> MujocoBackend:
        """Private service handoff: close this session, retain only a clean model."""
        with self._lock:
            self._check_open()
            if self.backend != "mujoco" or self._faulted:
                raise RuntimeError("only a healthy MuJoCo session can release its backend")
            backend = self.env.backend
            backend.prepare_for_worker_reuse()
            self._closed = True
            self._observation = self._reset_response = None
            self._resets.clear()
            self._steps.clear()
            self._snapshots.clear()
            self._restores.clear()
            self._reset_record = None
            self.env._scene = None
            self.env._scene_sha256 = None
            return backend

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._observation = None
                self._reset_response = None
                self._steps.clear()
                self._resets.clear()
                self._snapshots.clear()
                self._restores.clear()
                self.env.close()

    def _wrap(self, observation: dict, *, info: dict, reward: float = 0.0,
              terminated: bool = False, truncated: bool = False) -> dict[str, Any]:
        sim_time = None
        if self.backend == "isaac":
            sim_time = observation.get("timestamp")
        elif self.backend == "mujoco" and self._timestep is not None:
            sim_time = observation.get("sim_time", self._step_index * self._timestep)
        return make_observation_envelope(
            session_id=self.session_id, episode_id=self.episode_id,
            scene_sha256=self.env.scene_sha256,
            state_version=self._state_version, step_index=self._step_index,
            sim_time=sim_time, backend=self.backend, observation=observation,
            reward=reward, terminated=terminated, truncated=truncated,
            info={**info, "next_sequence_id": self._next_sequence},
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")

    def _check_ready(self) -> None:
        self._check_open()
        if self._faulted:
            raise RuntimeError("session operation failed; reset is required")
        if self._observation is None:
            raise RuntimeError("reset must be called first")

    def __enter__(self) -> RuntimeSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
