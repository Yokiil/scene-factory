"""Versioned, deterministic initial-state sampling for the environment facade.

This is a pose-initialization contract, not a collision or task-validity check.
It does not alter the layout-generation seed or shared asset records.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import secrets
from typing import Any


RESET_SCHEMA = "scene_factory.episode_reset.v1"
RESET_RNG = "sha256-field-stream/python-random-uniform-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _pair(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two finite numbers")
    return [_number(item, label) for item in value]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def prepare_episode_reset(
    source_scene: dict[str, Any], *, seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a fresh scene and a complete reset record without mutating inputs.

    ``pose_overrides[id]`` accepts absolute ``position_xy_m`` and ``yaw_deg``.
    ``pose_jitter[id]`` accepts additive uniform ranges for ``x_m``, ``y_m`` and
    ``yaw_deg``. Jitter is applied after overrides; z is always preserved.
    Only dynamic leaf objects can be changed. Independent object/field streams
    mean that adding an unrelated random variable cannot change existing draws.
    """
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64), or None")
    raw_options = _mapping(options if options is not None else {}, "reset options")
    unknown = set(raw_options) - {"pose_overrides", "pose_jitter", "randomization"}
    if unknown:
        raise ValueError(f"unsupported reset options: {sorted(unknown)}")
    scene = copy.deepcopy(_mapping(source_scene, "scene"))
    raw_objects = scene.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("scene.objects must be an array")
    objects: dict[str, dict[str, Any]] = {}
    for item in raw_objects:
        if not isinstance(item, dict) or not isinstance(item.get("object_id"), str):
            raise ValueError("scene objects must have string object_id values")
        if item["object_id"] in objects:
            raise ValueError(f"duplicate object_id: {item['object_id']}")
        objects[item["object_id"]] = item
    support_ids = {
        item["support"].partition(":")[0]
        for item in raw_objects if isinstance(item.get("support"), str)
    }
    normalized: dict[str, dict[str, Any]] = {"pose_overrides": {}, "pose_jitter": {}}
    for group in normalized:
        changes = _mapping(raw_options.get(group, {}), group)
        allowed = {"position_xy_m", "yaw_deg"} if group == "pose_overrides" else {
            "x_m", "y_m", "yaw_deg",
        }
        for object_id, raw_change in changes.items():
            if object_id not in objects:
                raise ValueError(f"unknown reset object: {object_id}")
            item = objects[object_id]
            if item.get("dynamic") is not True or object_id in support_ids:
                raise ValueError(f"reset pose changes require a dynamic leaf object: {object_id}")
            change = _mapping(raw_change, f"{group}.{object_id}")
            if not change or set(change) - allowed:
                raise ValueError(f"invalid fields in {group}.{object_id}; allowed: {sorted(allowed)}")
            parsed: dict[str, Any] = {}
            for field, value in change.items():
                label = f"{group}.{object_id}.{field}"
                if group == "pose_jitter":
                    parsed[field] = _pair(value, label)
                    if parsed[field][0] > parsed[field][1]:
                        raise ValueError(f"{label} lower bound must not exceed upper bound")
                elif field == "position_xy_m":
                    parsed[field] = _pair(value, label)
                else:
                    parsed[field] = _number(value, label)
            normalized[group][object_id] = parsed
    # Validate all affected base poses before drawing or touching a backend.
    targets = sorted(set(normalized["pose_overrides"]) | set(normalized["pose_jitter"]))
    base_poses: dict[str, dict[str, Any]] = {}
    for object_id in targets:
        pose = _mapping(objects[object_id].get("pose"), f"{object_id}.pose")
        position = pose.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ValueError(f"{object_id}.pose.position must contain three finite numbers")
        base_poses[object_id] = {
            "position": [_number(v, f"{object_id}.pose.position") for v in position],
            "yaw_deg": _number(pose.get("yaw_deg", 0), f"{object_id}.pose.yaw_deg"),
        }
        if pose.get("orientation_wxyz") is not None:
            from .models import Pose

            parsed_pose = Pose.from_dict(pose)
            base_poses[object_id]["yaw_deg"] = parsed_pose.yaw_deg
            base_poses[object_id]["orientation_wxyz"] = list(parsed_pose.effective_orientation_wxyz)
    effective_seed = seed if seed is not None else secrets.randbits(64)
    sampled: dict[str, dict[str, Any]] = {}
    for object_id in targets:
        base = base_poses[object_id]
        pose = copy.deepcopy(base)
        override = normalized["pose_overrides"].get(object_id, {})
        if "position_xy_m" in override:
            pose["position"][:2] = override["position_xy_m"]
        if "yaw_deg" in override:
            pose["yaw_deg"] = override["yaw_deg"]
        offsets: dict[str, float] = {}
        for field, (low, high) in sorted(normalized["pose_jitter"].get(object_id, {}).items()):
            stream = _digest([RESET_RNG, effective_seed, object_id, field])
            offsets[field] = random.Random(int(stream, 16)).uniform(low, high)
            if field == "yaw_deg":
                pose["yaw_deg"] += offsets[field]
            else:
                pose["position"][0 if field == "x_m" else 1] += offsets[field]
        # Adding individually finite coordinates can overflow; reject the result.
        for value in [*pose["position"], pose["yaw_deg"]]:
            _number(value, f"{object_id}.sampled_pose")
        if "orientation_wxyz" in base:
            # Rotate the full attitude about world Z, retaining roll/pitch.
            # Updating yaw alone would leave explicit quaternion poses unchanged.
            angle = math.radians(pose["yaw_deg"] - base["yaw_deg"]) / 2
            c, s = math.cos(angle), math.sin(angle)
            w, x, y, z = base["orientation_wxyz"]
            pose["orientation_wxyz"] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
        objects[object_id]["pose"].update(copy.deepcopy(pose))
        sampled[object_id] = {
            "base_pose": base, "overrides": copy.deepcopy(override),
            "sampled_offsets": offsets, "final_pose": pose,
        }
    domain_report = None
    if "randomization" in raw_options:
        from .episode_randomization import prepare_episode_randomization

        normalized["randomization"] = copy.deepcopy(raw_options["randomization"])
        scene, domain_report = prepare_episode_randomization(scene, raw_options["randomization"], effective_seed)
    record = {
        "schema": RESET_SCHEMA,
        "seed": effective_seed,
        "seed_source": "explicit" if seed is not None else "system_entropy",
        "rng": RESET_RNG,
        "options": normalized,
        "options_sha256": _digest(normalized),
        "source_scene_sha256": _digest(source_scene),
        "episode_scene_sha256": _digest(scene),
        "randomization_applied": bool(normalized["pose_jitter"]) or domain_report is not None,
        "scene_modified": bool(targets) or domain_report is not None,
        "sampled_values": sampled,
        "units": {"position": "m", "yaw": "deg"},
        "validation": "initial_pose_contract_only; physical_and_task_validity_not_verified",
    }
    if domain_report is not None:
        record["domain_randomization"] = domain_report
    return scene, record
