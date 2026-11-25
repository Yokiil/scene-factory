"""Versioned contracts for local and HTTP physics runtime sessions.

These contracts describe current backend abilities, not proposed robot controls.
They do not start a simulator or equate proxy/dry-run output with physical evidence.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any


CAPABILITIES_SCHEMA = "scene_factory.session_capabilities.v1"
STEP_SCHEMA = "scene_factory.session_step.v1"
RESET_SCHEMA = "scene_factory.session_reset.v1"
OBSERVATION_SCHEMA = "scene_factory.session_observation.v1"
MAX_REQUEST_STEPS = 10_000

_BACKENDS = {
    "dry_run": {
        "physical": False,
        "fidelity": "dry_run",
        "action_modes": ["noop", "dry_run"],
        "reset_options": ["pose_overrides", "pose_jitter", "randomization"],
        "missing_fields": ["object_poses", "object_velocities", "robot_state", "contacts"],
    },
    "mujoco": {
        "physical": True,
        "fidelity": "collision_proxy",
        "action_modes": ["noop", "object_forces"],
        "reset_options": ["pose_overrides", "pose_jitter", "randomization"],
        "missing_fields": [
            "robot_state", "robot_end_effector",
        ],
    },
    "isaac": {
        "physical": True,
        "fidelity": "franka_mug_scripted",
        "action_modes": ["scripted", "joint_position"],
        "reset_options": [],
        "missing_fields": ["object_orientations", "object_velocities", "general_contacts"],
    },
}


def finite_json(value: Any, label: str = "value") -> Any:
    """Return a detached JSON tree, rejecting non-JSON types and nonfinite numbers."""
    ancestors: set[int] = set()

    def visit(item: Any, path: str, depth: int) -> Any:
        if depth > 128:
            raise ValueError(f"{label} JSON nesting exceeds 128 levels")
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path} must contain only finite numbers")
            return item
        if not isinstance(item, (dict, list)):
            raise ValueError(f"{path} must contain only JSON values")
        if id(item) in ancestors:
            raise ValueError(f"{path} must not contain a circular reference")
        ancestors.add(id(item))
        try:
            if isinstance(item, list):
                return [visit(child, f"{path}[{index}]", depth + 1)
                        for index, child in enumerate(item)]
            if not all(isinstance(key, str) for key in item):
                raise ValueError(f"{path} object keys must be strings")
            return {key: visit(child, f"{path}.{key}", depth + 1) for key, child in item.items()}
        finally:
            ancestors.remove(id(item))

    return visit(value, label, 0)


def _object(value: Any, label: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    result = finite_json(value, label)
    unknown = result.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} must be a nonempty string of at most 256 characters")
    return value


def _integer(value: Any, label: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")
    return value


def _schema(value: dict[str, Any], expected: str) -> None:
    if value.get("schema_version", expected) != expected:
        raise ValueError(f"unsupported schema_version; expected {expected}")
    value["schema_version"] = expected


def backend_capabilities(backend_name: str) -> dict[str, Any]:
    """Return detached capabilities; optional images require configured sensors."""
    if not isinstance(backend_name, str) or backend_name not in _BACKENDS:
        raise ValueError(f"unsupported session backend: {backend_name!r}")
    return {
        "schema_version": CAPABILITIES_SCHEMA,
        "backend": backend_name,
        **copy.deepcopy(_BACKENDS[backend_name]),
        "observation_modalities": ["state"],
        "image_modalities": {"dry_run": [], "mujoco": ["rgb"], "isaac": ["rgb", "depth"]}[backend_name],
        "snapshot_levels": {"dry_run": ["rebuild"], "mujoco": ["rebuild", "runtime_state"], "isaac": []}[backend_name],
        "reset_level": "new_session" if backend_name == "isaac" else "rebuild",
        "max_request_steps": MAX_REQUEST_STEPS,
        "external_joint_control": backend_name == "isaac",
        "contact_units": "N_and_Nm" if backend_name == "mujoco" else None,
    }


def validate_step_request(
    request: Any, capabilities: dict[str, Any] | str,
) -> dict[str, Any]:
    """Validate a whole request before a caller performs any simulator mutation."""
    if isinstance(capabilities, str):
        capabilities = backend_capabilities(capabilities)
    if not isinstance(capabilities, dict):
        raise ValueError("capabilities must be an object or backend name")
    known = backend_capabilities(capabilities.get("backend"))
    if capabilities != known:
        raise ValueError("capabilities must match a supported backend capability contract")
    result = _object(request, "step request", {
        "schema_version", "episode_id", "sequence_id", "idempotency_key", "action", "steps",
    })
    _schema(result, STEP_SCHEMA)
    for name in ("episode_id", "idempotency_key"):
        _text(result.get(name), name)
    _integer(result.get("sequence_id"), "sequence_id")
    result["steps"] = _integer(result.get("steps", 1), "steps", maximum=MAX_REQUEST_STEPS)
    if result["steps"] == 0:
        raise ValueError("steps must be positive")
    action = _object(result.get("action"), "action", {"mode", "payload"})
    mode = _text(action.get("mode"), "action.mode")
    if mode not in known["action_modes"]:
        raise ValueError(f"action mode {mode!r} is not supported by {known['backend']}")
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("action.payload must be an object")
    if mode in ("noop", "scripted") and payload:
        raise ValueError(f"{mode} action.payload must be empty")
    if mode == "joint_position":
        from .joint_control import validate_joint_position_payload

        payload = validate_joint_position_payload(payload)
    if mode == "object_forces":
        payload = _object(payload, "object_forces payload", {"object_forces"})
        forces = payload.get("object_forces")
        if not isinstance(forces, dict) or not forces:
            raise ValueError("object_forces must be a nonempty object ID to force mapping")
        for object_id, vector in forces.items():
            _text(object_id, "force target")
            if not isinstance(vector, list) or len(vector) not in (3, 6):
                raise ValueError("object force must contain 3 or 6 finite numbers")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector):
                raise ValueError("object force must contain 3 or 6 finite numbers")
            try:
                if not all(math.isfinite(float(v)) for v in vector):
                    raise ValueError("object force must contain 3 or 6 finite numbers")
            except OverflowError as exc:
                raise ValueError("object force magnitude exceeds floating-point range") from exc
    result["action"] = {"mode": mode, "payload": payload}
    return result


def translate_action(action: Any, backend: str) -> Any:
    """Translate a validated session action into the legacy backend API."""
    # Validate here too, so using this helper outside RuntimeSession is safe.
    validated = validate_step_request({
        "episode_id": "adapter", "sequence_id": 0, "idempotency_key": "adapter",
        "action": action,
    }, backend)["action"]
    if validated["mode"] == "noop":
        return None
    if validated["mode"] == "scripted":
        return "scripted"
    if validated["mode"] == "joint_position":
        return validated
    return validated["payload"]


def validate_reset_request(request: Any) -> dict[str, Any]:
    """Check reset transport structure; SceneFactoryEnv owns option semantics."""
    result = _object(request, "reset request", {
        "schema_version", "idempotency_key", "seed", "options",
    })
    _schema(result, RESET_SCHEMA)
    _text(result.get("idempotency_key"), "idempotency_key")
    result.setdefault("seed", None)
    if result["seed"] is not None:
        _integer(result["seed"], "seed", maximum=2**64 - 1)
    result.setdefault("options", {})
    if not isinstance(result["options"], dict):
        raise ValueError("reset options must be an object")
    return result


def make_observation_envelope(
    *, session_id: str, episode_id: str, scene_sha256: str, state_version: int,
    step_index: int, sim_time: float | None, backend: str, observation: dict[str, Any],
    reward: float = 0.0, terminated: bool = False, truncated: bool = False,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap one backend state without fabricating unavailable physical fields.

    ``state_version`` identifies a session state. The sensor_artifacts adapter
    attaches optional image references carrying that same state version.
    """
    _text(session_id, "session_id")
    _text(episode_id, "episode_id")
    if not isinstance(scene_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", scene_sha256) is None:
        raise ValueError("scene_sha256 must be a lowercase SHA-256 digest")
    _integer(state_version, "state_version")
    _integer(step_index, "step_index")
    for name, value in (("sim_time", sim_time), ("reward", reward)):
        if value is None and name == "sim_time":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite or (name == "sim_time" and value < 0):
            raise ValueError(f"{name} must be finite and sim_time must be nonnegative")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise ValueError("terminated and truncated must be booleans")
    if not isinstance(observation, dict) or (info is not None and not isinstance(info, dict)):
        raise ValueError("observation and info must be objects")
    capabilities = backend_capabilities(backend)
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "session_id": session_id,
        "episode_id": episode_id,
        "scene_sha256": scene_sha256,
        "state_version": state_version,
        "step_index": step_index,
        "sim_time": sim_time,
        "coordinates": {
            "frame": "world", "up_axis": "Z", "length": "m", "time": "s",
            "angle": "rad", "legacy_yaw_deg": "deg", "quaternion_order": "wxyz",
        },
        "backend": backend,
        "fidelity": capabilities["fidelity"],
        "observation": finite_json(observation, "observation"),
        "unavailable_fields": [
            *[field for field in capabilities["missing_fields"] if field not in observation.get("available_fields", [])], "image_references",
            *(["sim_time"] if sim_time is None else []),
        ],
        "images": [],
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": finite_json(info if info is not None else {}, "info"),
    }


def runtime_session_schema() -> dict[str, Any]:
    """Return JSON Schema for canonical messages (including inserted defaults).

    This is a message contract schema, not an OpenAPI endpoint document.
    Backend-specific support is additionally checked by ``validate_step_request``.
    """
    text = {"type": "string", "minLength": 1, "maxLength": 256, "pattern": r"\S"}
    integer = {"type": "integer", "minimum": 0}
    json_object = {"type": "object"}

    def record(properties: dict, required: list[str] | None = None) -> dict:
        return {
            "type": "object", "additionalProperties": False,
            "required": required if required is not None else list(properties),
            "properties": properties,
        }

    action = {"oneOf": [
        record({"mode": {"enum": ["noop", "scripted"]}, "payload": {
            "type": "object", "maxProperties": 0,
        }}),
        record({"mode": {"const": "dry_run"}, "payload": json_object}),
        record({"mode": {"const": "joint_position"}, "payload": record({
            "arm_positions_rad": {"type": "array", "minItems": 7, "maxItems": 7, "items": {"type": "number"}},
            "gripper_open_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        })}),
        record({"mode": {"const": "object_forces"}, "payload": record({
            "object_forces": {
                "type": "object", "minProperties": 1, "propertyNames": text,
                "additionalProperties": {
                    "type": "array", "items": {"type": "number"},
                    "oneOf": [{"minItems": 3, "maxItems": 3}, {"minItems": 6, "maxItems": 6}],
                },
            },
        })}),
    ]}
    step = record({
        "schema_version": {"const": STEP_SCHEMA}, "episode_id": text,
        "sequence_id": integer, "idempotency_key": text, "action": action,
        "steps": {"type": "integer", "minimum": 1, "maximum": MAX_REQUEST_STEPS},
    })
    reset = record({
        "schema_version": {"const": RESET_SCHEMA}, "idempotency_key": text,
        "seed": {"type": ["integer", "null"], "minimum": 0, "maximum": 2**64 - 1},
        "options": json_object,
    })
    observation = record({
        "schema_version": {"const": OBSERVATION_SCHEMA}, "session_id": text,
        "episode_id": text,
        "scene_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "state_version": integer, "step_index": integer,
        "sim_time": {"type": ["number", "null"], "minimum": 0},
        "coordinates": record({key: {"const": value} for key, value in {
            "frame": "world", "up_axis": "Z", "length": "m", "time": "s",
            "angle": "rad", "legacy_yaw_deg": "deg", "quaternion_order": "wxyz",
        }.items()}),
        "backend": {"enum": list(_BACKENDS)},
        "fidelity": {"enum": [item["fidelity"] for item in _BACKENDS.values()]},
        "observation": json_object,
        "unavailable_fields": {"type": "array", "items": text, "uniqueItems": True},
        "images": {"type": "array", "items": {"$ref": "#/$defs/image"}},
        "reward": {"type": "number"}, "terminated": {"type": "boolean"},
        "truncated": {"type": "boolean"}, "info": json_object,
    })
    capabilities = {"oneOf": [
        {"const": backend_capabilities(name)} for name in _BACKENDS
    ]}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    image = record({
        "modality": {"enum": ["rgb", "depth"]}, "camera": text, "path": text,
        "url": {"type": "string"}, "sha256": digest, "size_bytes": integer,
        "format": {"const": "npy"}, "dtype": text,
        "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 3},
        "units": {"enum": ["m", "uint8_srgb"]},
        "nonfinite_depth_means": {"enum": [None, "no_return"]},
        "episode_id": text, "state_version": integer, "step_index": integer,
        "sim_time": {"type": ["number", "null"], "minimum": 0},
        "calibration": {"type": ["object", "null"]},
    })
    snapshot = record({
        "schema_version": {"const": "scene_factory.session_snapshot.v1"},
        "level": {"enum": ["rebuild", "runtime_state"]}, "backend": {"enum": list(_BACKENDS)},
        "scene_sha256": digest, "source_scene_sha256": digest, "sha256": digest,
        "seed": reset["properties"]["seed"], "options": json_object,
        "step_index": integer, "terminated": {"type": "boolean"}, "truncated": {"type": "boolean"},
        "runtime_state": {"type": ["object", "null"]},
        "scope": {"enum": ["same_backend_compatible_model", "episode_initial_conditions"]},
    })
    snapshot["allOf"] = [{"if": {"properties": {"level": {"const": "rebuild"}}}, "then": {
        "properties": {"step_index": {"const": 0}, "terminated": {"const": False}, "truncated": {"const": False},
                       "runtime_state": {"type": "null"}, "scope": {"const": "episode_initial_conditions"}},
    }, "else": {"properties": {"runtime_state": {"type": "object"}, "scope": {"const": "same_backend_compatible_model"}}}}]
    return copy.deepcopy({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "scene_factory.runtime_session.v1",
        "title": "SceneFactory runtime session canonical messages",
        "description": (
            "Physics session message contracts shared by local and HTTP adapters. "
            "All messages must be finite JSON; backend support and reset option semantics "
            "also require runtime validation. Force vectors use world-frame N and N*m "
            "in [Fx,Fy,Fz,Tx,Ty,Tz] order; a three-vector supplies force only."
        ),
        "oneOf": [{"$ref": f"#/$defs/{name}"} for name in (
            "capabilities", "reset", "step", "observation", "snapshot",
        )],
        "$defs": {
            "capabilities": capabilities, "reset": reset, "step": step,
            "observation": observation, "image": image, "snapshot": snapshot,
        },
    })
