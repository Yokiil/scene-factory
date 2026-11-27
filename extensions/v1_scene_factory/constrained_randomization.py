"""Reproducible, correlated physical sampling for dynamic primitive instances."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from typing import Any

from .paths import default_registry_path
from .registry import AssetRegistry
from .session_contracts import finite_json


RANDOMIZATION_SCHEMA = "scene_factory.constrained_randomization.v1"
RNG_ALGORITHM = "sha256-field-stream/python-random-uniform-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(finite_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds floating-point range") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _range(value: Any, label: str, *, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two finite numbers")
    values = tuple(_number(item, label, positive=positive) for item in value)
    if values[0] > values[1]:
        raise ValueError(f"{label} must be an ordered range")
    return values


def _draw(seed: int, stream: str, variable: str, bounds: tuple[float, float]) -> float:
    key = _digest([RNG_ALGORITHM, seed, stream, variable])
    return random.Random(int(key, 16)).uniform(*bounds)


def _inertia(mass: float, bbox: list[float], primitive: str) -> list[float]:
    if primitive == "sphere":
        radius = max(bbox) / 2
        return [2 * mass * radius * radius / 5] * 3
    if primitive == "cylinder":
        radius, length = max(bbox[:2]) / 2, bbox[2]
        return [mass * (3 * radius * radius + length * length) / 12] * 2 + [mass * radius * radius / 2]
    x, y, z = bbox
    return [mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12]


def sample_instance_physics(
    scene: dict[str, Any], config: dict[str, Any], seed: int, *, registry: AssetRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one candidate; ``valid`` denotes parameter constraints only.

    ``scale_group`` shares exactly the same uniform scale draw. Density scaling
    preserves m~s^3 and I~s^5. Only dynamic leaf primitive proxies are supported.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    scene, config = finite_json(scene), finite_json(config)
    if not isinstance(scene, dict) or not isinstance(scene.get("objects"), list):
        raise ValueError("scene must contain an objects array")
    if not isinstance(config, dict) or set(config) - {"schema_version", "instances", "interfaces"}:
        raise ValueError("randomization config accepts schema_version, instances, interfaces only")
    if config.get("schema_version", RANDOMIZATION_SCHEMA) != RANDOMIZATION_SCHEMA:
        raise ValueError("unsupported randomization schema")
    instances = config.get("instances")
    if not isinstance(instances, dict) or not instances:
        raise ValueError("config.instances must be a nonempty object")
    registry = registry or AssetRegistry.load(default_registry_path())
    result = copy.deepcopy(scene)
    objects = {item["object_id"]: item for item in result["objects"]}
    if len(objects) != len(result["objects"]) or set(instances) - objects.keys():
        raise ValueError("unknown or duplicate scene object IDs")
    parents = {item["support"].partition(":")[0] for item in result["objects"] if isinstance(item.get("support"), str)}
    prepared, groups = {}, {}
    for object_id, raw in instances.items():
        if not isinstance(raw, dict) or set(raw) - {"scale", "density_scale", "friction", "scale_group"}:
            raise ValueError(f"invalid randomization fields for {object_id}")
        item = objects[object_id]
        asset = registry.get(item["asset_id"])
        if item.get("dynamic") is not True or object_id in parents:
            raise ValueError(f"randomization requires dynamic leaf object: {object_id}")
        if "kitchen_dining_pack" in asset.tags or asset.primitive not in {"cube", "box", "sphere", "cylinder"}:
            raise ValueError(f"randomization supports box/sphere/cylinder proxies only: {object_id}")
        scale = _range(raw.get("scale", [1, 1]), f"{object_id}.scale", positive=True)
        if item.get("pose", {}).get("orientation_wxyz") is not None and scale != (1.0, 1.0):
            from .models import Pose

            if asset.primitive != "sphere" and not Pose.from_dict(item["pose"]).is_upright:
                raise ValueError(f"scaling tilted proxies has no support-height contract: {object_id}")
        density = _range(raw.get("density_scale", [1, 1]), f"{object_id}.density_scale", positive=True)
        friction = _range(raw.get("friction", [asset.friction, asset.friction]), f"{object_id}.friction")
        if friction[0] < 0:
            raise ValueError("friction cannot be negative")
        group = raw.get("scale_group", object_id)
        if not isinstance(group, str) or not group:
            raise ValueError("scale_group must be nonempty")
        if group in groups and groups[group] != scale:
            raise ValueError("members of a scale_group must use identical scale ranges")
        groups[group] = scale
        prepared[object_id] = (asset, scale, density, friction, group)
    scales = {group: _draw(seed, group, "scale", bounds) for group, bounds in groups.items()}
    sampled = {}
    for object_id, (asset, _, density_bounds, friction_bounds, group) in prepared.items():
        item = objects[object_id]
        bbox = item.get("bbox_m")
        if not isinstance(bbox, list) or len(bbox) != 3:
            raise ValueError(f"invalid bbox for {object_id}")
        bbox = [_number(value, "bbox_m", positive=True) for value in bbox]
        scale = scales[group]
        density = _draw(seed, object_id, "density_scale", density_bounds)
        friction = _draw(seed, object_id, "friction", friction_bounds)
        previous = item.get("physics", {})
        mass = _number(previous.get("mass_kg", asset.mass_kg), "mass_kg", positive=True)
        base_inertia = previous.get("inertia_kg_m2", _inertia(mass, bbox, asset.primitive))
        if not isinstance(base_inertia, list) or len(base_inertia) != 3:
            raise ValueError("inertia_kg_m2 must contain three positive values")
        base_inertia = [_number(value, "inertia_kg_m2", positive=True) for value in base_inertia]
        if max(base_inertia) > sum(base_inertia) - max(base_inertia) + 1e-12:
            raise ValueError("inertia violates triangle inequalities")
        effective = {"mass_kg": mass * scale**3 * density,
                     "inertia_kg_m2": [value * scale**5 * density for value in base_inertia],
                     "friction": friction, "source": "constrained_sample"}
        finite_json(effective)
        item["physics"] = effective
        item["bbox_m"] = [value * scale for value in bbox]
        position = item.get("pose", {}).get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("sampled objects require position pose")
        height = max(bbox) if asset.primitive == "sphere" else bbox[2]
        position[2] = _number(position[2], "pose.z") + height * (scale - 1) / 2
        sampled[object_id] = {"scale": scale, "scale_group": group, "density_scale": density,
                              "base_mass_kg": mass, "base_inertia_kg_m2": base_inertia,
                              "effective": effective, "physics_parameters_source": asset.physics_parameters_source,
                              "bbox_m": item["bbox_m"], "position_m": list(position)}
    rejected = []
    interfaces = config.get("interfaces", [])
    if not isinstance(interfaces, list):
        raise ValueError("interfaces must be an array")
    measured_interfaces = []
    for interface in interfaces:
        if not isinstance(interface, dict) or set(interface) != {"peg_id", "socket_id", "peg_radius_m", "socket_radius_m", "clearance_m"}:
            raise ValueError("interface requires peg_id/socket_id/radii/clearance_m")
        peg, socket = interface["peg_id"], interface["socket_id"]
        if peg not in objects or socket not in objects or peg == socket:
            raise ValueError("interface references unknown or identical parts")
        peg_radius = _number(interface["peg_radius_m"], "peg_radius_m", positive=True) * sampled.get(peg, {}).get("scale", 1)
        socket_radius = _number(interface["socket_radius_m"], "socket_radius_m", positive=True) * sampled.get(socket, {}).get("scale", 1)
        bounds = _range(interface["clearance_m"], "clearance_m")
        if bounds[0] < 0:
            raise ValueError("allowed clearance must be nonnegative")
        clearance = socket_radius - peg_radius
        measured_interfaces.append({"peg_id": peg, "socket_id": socket, "clearance_m": clearance})
        if not bounds[0] <= clearance <= bounds[1]:
            rejected.append({"pair": [peg, socket], "reason": "interface_clearance_out_of_range", "actual_m": clearance, "allowed_m": list(bounds)})
    report = {"schema_version": RANDOMIZATION_SCHEMA, "seed": seed, "rng": RNG_ALGORITHM,
              "config_sha256": _digest(config), "source_scene_sha256": _digest(scene),
              "sampled_values": sampled, "interfaces": measured_interfaces, "rejections": rejected,
              "valid": not rejected, "classification": "parameter_constraints_only",
              "physics_validity": "not_verified", "controller_outcome": "not_run"}
    result["randomization"] = copy.deepcopy(report)
    return finite_json(result), report


def sample_candidate_seeds(
    scene: dict[str, Any], config: dict[str, Any], seeds: list[int], *, registry: AssetRegistry | None = None,
) -> dict[str, Any]:
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique list")
    registry = registry or AssetRegistry.load(default_registry_path())
    rows, reasons = [], Counter()
    for seed in seeds:
        try:
            _, report = sample_instance_physics(scene, config, seed, registry=registry)
            rows.append(report)
            reasons.update(item["reason"] for item in report["rejections"])
        except (TypeError, ValueError, OverflowError) as exc:
            reasons[type(exc).__name__] += 1
            rows.append({"seed": seed, "valid": False, "rejections": [{"reason": type(exc).__name__, "message": str(exc)}]})
    valid = sum(row["valid"] for row in rows)
    return {"schema_version": "scene_factory.randomization_collection.v1", "candidate_count": len(rows),
            "counts": {"parameter_valid": valid, "parameter_invalid": len(rows) - valid,
                       "physics_valid": None, "controller_failed": None, "task_success": None},
            "rejection_counts": dict(reasons), "rows": rows, "physics_validity": "not_verified",
            "controller_outcome": "not_run", "split_policy": "no_train_test_split_inferred_from_seed"}
