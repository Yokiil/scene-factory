from __future__ import annotations

import copy
import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .episode_reset import prepare_episode_reset
from .episode_randomization import noisy_rgb, sensor_measurements


@runtime_checkable
class SimulatorBackend(Protocol):
    def reset(self, scene: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class DryRunBackend:
    """Gym-like backend for integration tests before a robot-specific simulator is attached."""

    supports_scene_reset_options = True

    def __init__(self, max_steps: int = 100) -> None:
        self.max_steps = max_steps
        self.scene: dict[str, Any] | None = None
        self.steps = 0

    def reset(self, scene: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.scene = scene
        self.steps = 0
        return self._observation(), {"scene_id": scene["scene_id"], "dry_run": True}

    def step(self, action: Any):
        if self.scene is None:
            raise RuntimeError("reset must be called before step")
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._observation(), 0.0, False, truncated, {"dry_run": True, "action": action}

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.scene = None

    def _observation(self) -> dict[str, Any]:
        if self.scene is None:
            return {}
        return {
            "language_instruction": self.scene.get("task", {}).get("instruction", ""),
            "scene_graph": self.scene["objects"],
            "proprioception": [],
        }


class SceneFactoryEnv:
    """Small Gymnasium-compatible facade with a pluggable simulator backend."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, scene_path: str | Path, backend: SimulatorBackend | None = None) -> None:
        self.scene_path = Path(scene_path)
        if backend is None:
            from .backends.mujoco import MujocoBackend

            backend = MujocoBackend()
        self.backend = backend
        self._scene: dict[str, Any] | None = None
        self._scene_sha256: str | None = None
        self._reset_attempts = 0

    @property
    def reset_attempts(self) -> int:
        """Number of backend reset calls started, including failed attempts."""
        return self._reset_attempts

    @property
    def scene_sha256(self) -> str | None:
        """Canonical digest of the last successfully initialized episode scene."""
        return self._scene_sha256

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        with self.scene_path.open("r", encoding="utf-8") as handle:
            source_scene = json.load(handle)
        scene, reset_record = prepare_episode_reset(source_scene, seed=seed, options=options)
        if reset_record["scene_modified"] and not getattr(
            self.backend, "supports_scene_reset_options", False,
        ):
            raise ValueError(
                f"{type(self.backend).__name__} does not support reset pose changes; "
                "pre-exported simulation assets must be rebuilt before changing initial poses"
            )
        # Validation failures above preserve the active episode. Once reset starts,
        # backend failure invalidates it; never expose stale state as a fresh episode.
        self._reset_attempts += 1
        self._scene = None
        self._scene_sha256 = None
        try:
            observation, info = self.backend.reset(copy.deepcopy(scene))
        except Exception:
            if getattr(self.backend, "close_on_reset_error", True):
                with suppress(Exception):
                    self.backend.close()
            raise
        self._scene = scene
        self._scene_sha256 = reset_record["episode_scene_sha256"]
        return self.transform_observation(observation), {**info, "episode_reset": reset_record}

    def step(self, action: Any):
        if self._scene is None:
            raise RuntimeError("reset must succeed before step")
        observation, reward, terminated, truncated, info = self.backend.step(action)
        return self.transform_observation(observation), reward, terminated, truncated, info

    def render(self):
        if self._scene is None:
            raise RuntimeError("reset must succeed before render")
        return noisy_rgb(self._scene, self.backend.render(), getattr(self.backend, "steps", 0))

    def transform_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Attach configured sensor measurements without altering truth or rewards."""
        return sensor_measurements(self._scene, observation, getattr(self.backend, "steps", 0))

    def close(self) -> None:
        self._scene = None
        self._scene_sha256 = None
        self.backend.close()
