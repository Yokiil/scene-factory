from __future__ import annotations

import importlib
import hashlib
import json
import math
import copy
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

from ..exporters.mujoco_mjcf import MujocoMjcfExporter
from ..paths import default_registry_path
from ..registry import AssetRegistry
from ..tasks import TaskEvaluator


class MujocoBackend:
    """Gym-like MuJoCo backend using deterministic MJCF collision proxies."""

    supports_scene_reset_options = True

    def __init__(
        self,
        *,
        registry: AssetRegistry | None = None,
        max_steps: int = 1000,
        frame_skip: int = 5,
        render_width: int = 960,
        render_height: int = 720,
        reuse_model: bool = True,
    ) -> None:
        if max_steps < 1 or frame_skip < 1:
            raise ValueError("max_steps and frame_skip must be positive")
        self.registry = registry or AssetRegistry.load(default_registry_path())
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.render_width = render_width
        self.render_height = render_height
        self.reuse_model = bool(reuse_model)
        self.scene: dict[str, Any] | None = None
        self.steps = 0
        self._mujoco: Any = None
        self._model: Any = None
        self._data: Any = None
        self._renderer: Any = None
        self._body_ids: dict[str, int] = {}
        self._task_evaluator: TaskEvaluator | None = None
        self._model_sha256: str | None = None
        self._xml: str | None = None
        self._structure_xml: str | None = None
        self._model_cache_hits = 0
        self._model_compile_count = 0
        self._model_reset_count = 0
        self._reset_kind = "none"
        self._scene_sha256: str | None = None
        self._last_task_status: dict[str, Any] = {}
        self._implementation_sha256 = hashlib.sha256(
            Path(__file__).read_bytes() + (Path(__file__).parents[1] / "tasks.py").read_bytes()
            + (Path(__file__).parents[1] / "exporters" / "mujoco_mjcf.py").read_bytes()
        ).hexdigest()

    def reset(self, scene: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self._mujoco = self._load_mujoco()
        xml = MujocoMjcfExporter(
            self.registry,
            offscreen_width=self.render_width,
            offscreen_height=self.render_height,
        ).to_string(scene)
        model_sha256 = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        structure_xml, dynamic_poses = self._structure_and_poses(xml)
        can_reuse = bool(
            self.reuse_model and self._model is not None and self._structure_xml == structure_xml
        )
        if can_reuse:
            if self._data is None:
                self._data = self._mujoco.MjData(self._model)
            for name, (position, quaternion) in dynamic_poses.items():
                body_id = int(self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, name))
                joint = int(self._model.body_jntadr[body_id])
                start = int(self._model.jnt_qposadr[joint])
                self._model.qpos0[start:start + 7] = [*position, *quaternion]
                self._model.body_pos[body_id] = position
                self._model.body_quat[body_id] = quaternion
            if self._xml != xml:
                self._mujoco.mj_setConst(self._model, self._data)
            self._mujoco.mj_resetData(self._model, self._data)
            self._model_cache_hits += 1
            self._reset_kind = "model_reused"
        else:
            self._release_runtime(keep_model=False)
            self._model = self._mujoco.MjModel.from_xml_string(xml)
            self._model_compile_count += 1
            self._reset_kind = "model_compiled"
        if self._data is None:
            self._data = self._mujoco.MjData(self._model)
        self._mujoco.mj_forward(self._model, self._data)
        self._xml = xml
        self._structure_xml = structure_xml
        self._model_sha256 = model_sha256
        self._model_reset_count += 1
        self.scene = scene
        self._scene_sha256 = self._json_digest(scene)
        self.steps = 0
        self._body_ids = {
            object_id: self._body_id(body_name)
            for object_id, body_name in MujocoMjcfExporter.object_body_names(scene).items()
        }
        initial_state = self._object_positions()
        self._task_evaluator = TaskEvaluator(dict(scene.get("task", {})), initial_state)
        self._last_task_status = {"task_success": False}
        return self._observation(), {
            "scene_id": scene["scene_id"],
            "backend": "mujoco",
            "timestep": float(self._model.opt.timestep) * self.frame_skip,
            "body_count": len(self._body_ids),
            "reset_kind": self._reset_kind,
            "model_sha256": self._model_sha256,
            "model_cache_hits": self._model_cache_hits,
            "model_compile_count": self._model_compile_count,
            "model_reset_count": self._model_reset_count,
            "reuse_model": self.reuse_model,
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._model is None or self._data is None or self.scene is None:
            raise RuntimeError("reset must be called before step")
        self._apply_action(action)
        for _ in range(self.frame_skip):
            self._mujoco.mj_step(self._model, self._data)
        self._mujoco.mj_forward(self._model, self._data)
        self.steps += 1
        positions = self._object_positions()
        task_status = (
            self._task_evaluator.status(positions)
            if self._task_evaluator is not None
            else {"task_success": False}
        )
        terminated = bool(task_status.get("task_success"))
        truncated = self.steps >= self.max_steps and not terminated
        self._last_task_status = task_status
        return self._observation(), float(terminated), terminated, truncated, {
            "backend": "mujoco",
            "step": self.steps,
            "task": task_status,
        }

    def render(self) -> Any:
        if self._model is None or self._data is None:
            raise RuntimeError("reset must be called before render")
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self._model,
                height=self.render_height,
                width=self.render_width,
            )
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def close(self) -> None:
        self._release_runtime(keep_model=False)

    def prepare_for_worker_reuse(self) -> dict[str, Any]:
        """Retain a healthy compiled model, discard episode data and renderer.

        Called only by the private worker handoff. The next RuntimeSession
        must perform a normal reset before exposing an observation.
        """
        self._require_runtime()
        import numpy as np

        signature = int(self._mujoco.mjtState.mjSTATE_INTEGRATION)
        values = np.zeros(self._mujoco.mj_stateSize(self._model, signature))
        self._mujoco.mj_getState(self._model, self._data, values, signature)
        if not np.isfinite(values).all() or any(int(warning.number) for warning in self._data.warning):
            raise RuntimeError("MuJoCo worker has nonfinite state or solver warnings; it cannot be pooled")
        result = {"retained_model_sha256": self._model_sha256, "model_compile_count": self._model_compile_count,
                  "model_reset_count": self._model_reset_count, "renderer_discarded": True,
                  "episode_data_discarded": True, "health": "finite_integration_state_no_solver_warnings"}
        self._release_runtime(keep_model=True)
        return result

    def _release_runtime(self, *, keep_model: bool) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        if not keep_model:
            self._model = None
            self._xml = None
            self._structure_xml = None
            self._model_sha256 = None
        self._data = None
        self._body_ids = {}
        self._task_evaluator = None
        self.scene = None
        self._scene_sha256 = None
        self._last_task_status = {}
        self.steps = 0

    @staticmethod
    def _structure_and_poses(xml: str) -> tuple[str, dict[str, tuple[list[float], list[float]]]]:
        """Only direct world free bodies may change pose without recompilation.

        All geometry, inertial, solver, static pose and nested-body input remains
        in this comparison; an arbitrary topology/physics change reloads fully.
        """
        root = ET.fromstring(xml)
        poses = {}
        for body in root.findall("./worldbody/body"):
            if body.find("freejoint") is None or body.find("body") is not None:
                continue
            position = [float(v) for v in body.attrib.get("pos", "0 0 0").split()]
            quaternion = [float(v) for v in body.attrib.get("quat", "1 0 0 0").split()]
            norm = math.sqrt(sum(v * v for v in quaternion))
            quaternion = [v / norm for v in quaternion]
            poses[body.attrib["name"]] = (position, quaternion)
            body.set("pos", "0 0 0")
            body.set("quat", "1 0 0 0")
        return ET.tostring(root, encoding="unicode"), poses

    @staticmethod
    def _load_mujoco() -> Any:
        try:
            return importlib.import_module("mujoco")
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is the default backend but the 'mujoco' package is missing. "
                "Install the project again or run: python -m pip install mujoco"
            ) from exc

    def _body_id(self, body_name: str) -> int:
        body_id = int(
            self._mujoco.mj_name2id(
                self._model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
        )
        if body_id < 0:
            raise RuntimeError(f"MuJoCo body was not created: {body_name!r}")
        return body_id

    def _object_positions(self) -> dict[str, tuple[float, float, float]]:
        return {
            object_id: tuple(float(value) for value in self._data.xpos[body_id])
            for object_id, body_id in self._body_ids.items()
        }

    def _observation(self) -> dict[str, Any]:
        if self.scene is None or self._data is None:
            return {}
        object_states = self._object_states()
        return {
            "language_instruction": self.scene.get("task", {}).get("instruction", ""),
            "scene_graph": self.scene.get("objects", []),
            "object_poses": {
                object_id: list(position)
                for object_id, position in self._object_positions().items()
            },
            "objects": object_states,
            "contacts": self._contact_pairs(),
            "sim_time": float(self._data.time),
            "proprioception": {
                "qpos": [float(value) for value in self._data.qpos],
                "qvel": [float(value) for value in self._data.qvel],
            },
        }

    def _object_states(self) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for object_id, body_id in self._body_ids.items():
            state: dict[str, Any] = {
                "position_m": [float(value) for value in self._data.xpos[body_id]],
                "orientation_wxyz": [float(value) for value in self._data.xquat[body_id]],
                "velocity_frame": "world",
                "velocity_reference": "body_center_of_mass",
                "center_of_mass_position_m": [float(value) for value in self._data.xipos[body_id]],
            }
            import numpy as np

            velocity = np.zeros(6, dtype=float)
            self._mujoco.mj_objectVelocity(
                self._model, self._data, self._mujoco.mjtObj.mjOBJ_BODY,
                body_id, velocity, 0,
            )
            state["angular_velocity_rps"] = [float(value) for value in velocity[:3]]
            state["linear_velocity_mps"] = [float(value) for value in velocity[3:]]
            states[object_id] = state
        return states

    def _contact_pairs(self) -> list[dict[str, Any]]:
        if self._model is None or self._data is None:
            return []
        import numpy as np

        body_ids = {body: object_id for object_id, body in self._body_ids.items()}
        pairs: list[dict[str, Any]] = []
        for index in range(int(self._data.ncon)):
            contact = self._data.contact[index]
            first_geom, second_geom = int(contact.geom1), int(contact.geom2)
            if first_geom < 0 or second_geom < 0:
                raise RuntimeError("flex contacts are unsupported by this rigid-body adapter")
            wrench = np.zeros(6, dtype=float)
            self._mujoco.mj_contactForce(self._model, self._data, index, wrench)
            frame = np.asarray(contact.frame).reshape(3, 3)
            pairs.append({
                "object1": body_ids.get(int(self._model.geom_bodyid[first_geom]), "world"),
                "object2": body_ids.get(int(self._model.geom_bodyid[second_geom]), "world"),
                "geom1": self._mujoco.mj_id2name(self._model, self._mujoco.mjtObj.mjOBJ_GEOM, first_geom) or str(first_geom),
                "geom2": self._mujoco.mj_id2name(self._model, self._mujoco.mjtObj.mjOBJ_GEOM, second_geom) or str(second_geom),
                "position_m": [float(v) for v in contact.pos],
                "distance_m": float(contact.dist),
                "wrench_contact_frame": wrench.tolist(),
                "force_world_n": (frame.T @ wrench[:3]).tolist(),
                "torque_world_nm": (frame.T @ wrench[3:]).tolist(),
                "force_on": "geom2", "type": "instantaneous_solver_force",
                "contact_axes_world": frame.tolist(),
                "force_unit": "N", "torque_unit": "N*m",
                "sim_time": float(self._data.time), "sample_window_s": 0.0,
            })
        return pairs

    def capture_state(self) -> dict[str, Any]:
        """Serialize integration state for this exact scene/model/backend configuration."""
        self._require_runtime()
        import numpy as np
        from ..session_contracts import finite_json

        signature = int(self._mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.zeros(self._mujoco.mj_stateSize(self._model, signature))
        self._mujoco.mj_getState(self._model, self._data, state, signature)
        snapshot = {
            "schema": "scene_factory.mujoco_state.v1",
            "model_sha256": self._model_sha256,
            "scene_sha256": self._scene_sha256,
            "mujoco_version": str(self._mujoco.__version__),
            "backend_version": "2",
            "implementation_sha256": self._implementation_sha256,
            "state_signature": signature,
            "state": state.tolist(), "steps": self.steps,
            "settings": {"max_steps": self.max_steps, "frame_skip": self.frame_skip},
            "task": self._task_state(),
            "task_status": copy.deepcopy(self._last_task_status),
            "limitations": ["same_model_engine_and_backend_only", "no_renderer_or_external_rng_state"],
        }
        snapshot = finite_json(snapshot, "snapshot")
        snapshot["sha256"] = self._json_digest(snapshot)
        return snapshot

    def restore_state(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Validate and stage a restore before atomically replacing active data."""
        self._require_runtime()
        import numpy as np
        from ..session_contracts import finite_json

        value = finite_json(snapshot, "snapshot")
        expected = set(self.capture_state())
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("snapshot fields do not match mujoco_state.v1")
        checksum = value.pop("sha256")
        if checksum != self._json_digest(value):
            raise ValueError("snapshot checksum mismatch")
        for key, expected_value in {
            "schema": "scene_factory.mujoco_state.v1",
            "model_sha256": self._model_sha256,
            "scene_sha256": self._scene_sha256,
            "mujoco_version": str(self._mujoco.__version__), "backend_version": "2",
            "implementation_sha256": self._implementation_sha256,
            "state_signature": int(self._mujoco.mjtState.mjSTATE_INTEGRATION),
            "settings": {"max_steps": self.max_steps, "frame_skip": self.frame_skip},
        }.items():
            if value[key] != expected_value:
                raise ValueError(f"snapshot {key} does not match active runtime")
        steps = value["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= self.max_steps:
            raise ValueError("snapshot steps are outside the episode bounds")
        signature = value["state_signature"]
        state_values = self._finite_vector(value["state"], (
            int(self._mujoco.mj_stateSize(self._model, signature)),
        ), "integration state")
        state = np.asarray(state_values, dtype=float)
        candidate_task = self._validated_task_state(value["task"])
        if not isinstance(value["task_status"], dict):
            raise ValueError("snapshot task_status must be an object")
        candidate = self._mujoco.MjData(self._model)
        self._mujoco.mj_setState(self._model, candidate, state, signature)
        if candidate.time < 0 or not math.isclose(
            float(candidate.time), steps * self.frame_skip * float(self._model.opt.timestep),
            abs_tol=1e-8, rel_tol=1e-8,
        ):
            raise ValueError("snapshot simulation time does not match environment steps")
        for joint in range(self._model.njnt):
            kind = int(self._model.jnt_type[joint])
            if kind in (int(self._mujoco.mjtJoint.mjJNT_FREE), int(self._mujoco.mjtJoint.mjJNT_BALL)):
                start = int(self._model.jnt_qposadr[joint]) + (3 if kind == 0 else 0)
                if not math.isclose(float(np.linalg.norm(candidate.qpos[start:start + 4])), 1.0, abs_tol=1e-6):
                    raise ValueError("snapshot contains a nonunit joint quaternion")
        self._mujoco.mj_forward(self._model, candidate)
        # Forward recomputes derived contacts, but must not overwrite the saved
        # warmstart/control integration fields before continuation.
        self._mujoco.mj_setState(self._model, candidate, state, signature)
        old = self._data, self._task_evaluator, self.steps, self._last_task_status
        try:
            self._data = candidate
            self._task_evaluator = candidate_task
            self.steps = steps
            self._last_task_status = copy.deepcopy(value["task_status"])
            observation = finite_json(self._observation(), "restored observation")
        except Exception:
            self._data, self._task_evaluator, self.steps, self._last_task_status = old
            raise
        return observation, {
            "backend": "mujoco", "restore_kind": "integration_state",
            "model_sha256": self._model_sha256, "step": self.steps,
            "sim_time": float(self._data.time), "task": copy.deepcopy(self._last_task_status),
            "timestep": float(self._model.opt.timestep) * self.frame_skip,
            "restoration_precision": "trajectory_error_must_be_measured_on_target_runtime",
        }

    @staticmethod
    def _json_digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()

    def _require_runtime(self) -> None:
        if self._model is None or self._data is None or self.scene is None:
            raise RuntimeError("reset must succeed before state operations")

    def _task_state(self) -> dict[str, Any]:
        evaluator = self._task_evaluator
        return {
            "max_lift_delta_m": evaluator._max_lift_delta_m,
            "picked": evaluator._picked, "released": evaluator._released,
            "placement_stable_steps": evaluator._placement_stable_steps,
            "last_target_position": list(evaluator._last_target_position)
            if evaluator._last_target_position is not None else None,
            "last_target_step_distance_m": evaluator._last_target_step_distance_m,
            "initial_state": {key: list(position) for key, position in evaluator.initial_state.items()},
        }

    def _validated_task_state(self, task: dict[str, Any]) -> TaskEvaluator:
        if not isinstance(task, dict) or set(task) != set(self._task_state()):
            raise ValueError("invalid snapshot task fields")
        initial = task["initial_state"]
        if not isinstance(initial, dict) or set(initial) != set(self._body_ids):
            raise ValueError("snapshot initial_state does not match scene objects")
        initial = {key: tuple(self._finite_vector(v, (3,), "initial position")) for key, v in initial.items()}
        if initial != self._task_evaluator.initial_state:
            raise ValueError("snapshot task initial_state does not match active episode")
        evaluator = TaskEvaluator(copy.deepcopy(self.scene.get("task", {})), initial)
        for key in ("max_lift_delta_m", "last_target_step_distance_m"):
            value = task[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid task state: {key}")
            setattr(evaluator, "_" + key, value)
        for key in ("picked", "released"):
            if not isinstance(task[key], bool):
                raise ValueError(f"invalid task state: {key}")
            setattr(evaluator, "_" + key, task[key])
        stable = task["placement_stable_steps"]
        if isinstance(stable, bool) or not isinstance(stable, int) or stable < 0:
            raise ValueError("invalid task placement_stable_steps")
        evaluator._placement_stable_steps = stable
        position = task["last_target_position"]
        evaluator._last_target_position = None if position is None else tuple(
            self._finite_vector(position, (3,), "last target position"),
        )
        return evaluator

    def _apply_action(self, action: Any) -> None:
        controls = None
        prepared_forces = {}
        if isinstance(action, Mapping):
            if set(action) - {"ctrl", "object_forces"}:
                raise ValueError("unsupported MuJoCo action fields")
            controls = action.get("ctrl")
            if controls is not None:
                controls = self._finite_vector(controls, (int(self._model.nu),), "control")
            forces = action.get("object_forces", {})
            if not isinstance(forces, Mapping):
                raise ValueError("object_forces must map object IDs to 3D or 6D forces")
            for object_id, values in forces.items():
                if object_id not in self._body_ids:
                    raise ValueError(f"unknown object force target: {object_id}")
                if int(self._model.body_dofnum[self._body_ids[object_id]]) == 0:
                    raise ValueError(f"object force target must be dynamic: {object_id}")
                vector = self._finite_vector(values, (3, 6), "object force")
                prepared_forces[object_id] = vector
        elif action is not None:
            controls = self._finite_vector(action, (int(self._model.nu),), "control")
        self._data.xfrc_applied[:] = 0.0
        if controls is not None and self._model.nu:
            self._data.ctrl[:] = controls
        for object_id, vector in prepared_forces.items():
            self._data.xfrc_applied[self._body_ids[object_id], : len(vector)] = vector

    def _set_controls(self, values: Any) -> None:
        vector = self._finite_vector(values, (int(self._model.nu),), "control")
        if self._model.nu:
            self._data.ctrl[:] = vector

    @staticmethod
    def _finite_vector(values: Any, lengths: tuple[int, ...], label: str) -> list[float]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"{label} must be a numeric sequence")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError(f"{label} must contain finite numeric values")
        try:
            vector = [float(value) for value in values]
        except OverflowError as exc:
            raise ValueError(f"{label} exceeds floating-point range") from exc
        if len(vector) not in lengths or any(not math.isfinite(value) for value in vector):
            expected = " or ".join(str(value) for value in lengths)
            raise ValueError(f"{label} must contain {expected} finite values")
        return vector
