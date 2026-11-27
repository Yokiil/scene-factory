"""Four independent episode sampling layers for explicitly supported proxies.

Initial faults currently mean physical XY/yaw misalignment. They are not a
claim to model electrical failures, flexible parts, stiction or broken joints.
Sensor measurements are separate from simulation truth and task judgement.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from typing import Any


SCHEMA = "scene_factory.episode_randomization.v1"
RNG = "sha256-independent-layer/python-random-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def layer_seed(seed: int, layer: str) -> int:
    return int(digest([RNG, seed, layer])[:16], 16)


def _bounds(value: Any, label: str, *, nonnegative: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-number range")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{label} must be a finite range")
    try:
        result = [float(v) for v in value]
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds floating-point range") from exc
    if not all(math.isfinite(v) for v in result) or result[0] > result[1]:
        raise ValueError(f"{label} must be a finite ordered range")
    if nonnegative and result[0] < 0:
        raise ValueError(f"{label} cannot be negative")
    return result


def prepare_episode_randomization(scene: dict[str, Any], config: Any, seed: int):
    """Return a detached sampled scene and all four layer records.

    Layout and initial_state use the existing dynamic-leaf pose contract.
    Physics uses the constrained primitive sampler, including m~s^3/I~s^5.
    Observation parameters are sampled now and measurements at a state index
    use independent stateless streams, making retries and restores repeatable.
    """
    from .constrained_randomization import sample_instance_physics
    from .episode_reset import prepare_episode_reset
    from .session_contracts import finite_json

    config = finite_json(config, "episode randomization")
    if not isinstance(config, dict) or not config or set(config) - {
        "schema_version", "difficulty", "layout", "physics", "initial_state", "observation",
    }:
        raise ValueError("episode randomization accepts layout, physics, initial_state, observation and difficulty")
    if config.get("schema_version", SCHEMA) != SCHEMA:
        raise ValueError("unsupported episode randomization schema")
    difficulty = config.get("difficulty", "unspecified")
    if difficulty not in {"easy", "medium", "hard", "unspecified"}:
        raise ValueError("difficulty must be easy, medium, hard or unspecified")
    result = copy.deepcopy(scene)
    streams = {name: layer_seed(seed, name) for name in ("layout", "physics", "initial_state", "observation")}
    layers: dict[str, Any] = {}
    for name in ("layout", "physics", "initial_state", "observation"):
        raw = config.get(name)
        if raw is None:
            layers[name] = {"enabled": False, "seed": streams[name], "sampled_values": {}}
            continue
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"{name} must be a nonempty object when configured")
        if name == "layout":
            if set(raw) - {"pose_overrides", "pose_jitter"}:
                raise ValueError("layout supports pose_overrides and pose_jitter only")
            result, record = prepare_episode_reset(result, seed=streams[name], options=raw)
            layers[name] = {"enabled": True, "seed": streams[name], "sampled_values": record["sampled_values"]}
        elif name == "physics":
            result, record = sample_instance_physics(result, raw, streams[name])
            if not record["valid"]:
                raise ValueError("episode physics constraints rejected: " + json.dumps(record["rejections"]))
            layers[name] = {"enabled": True, "seed": streams[name], **record}
        elif name == "initial_state":
            if set(raw) != {"pose_jitter"}:
                raise ValueError("initial_state currently supports physical misalignment via pose_jitter only")
            result, record = prepare_episode_reset(result, seed=streams[name], options=raw)
            layers[name] = {"enabled": True, "seed": streams[name],
                            "fault_model": "physical_initial_xy_yaw_misalignment",
                            "sampled_values": record["sampled_values"]}
        else:
            if set(raw) - {"position_noise_std_m", "rgb_noise_std_uint8"}:
                raise ValueError("observation supports position_noise_std_m and rgb_noise_std_uint8 only")
            values = {}
            for key, value in raw.items():
                bounds = _bounds(value, key, nonnegative=True)
                values[key] = random.Random(layer_seed(streams[name], key)).uniform(*bounds)
            layers[name] = {"enabled": True, "seed": streams[name], "sampled_values": values,
                            "truth_modified": False, "measurement_distribution": "independent_additive_gaussian"}
    report = {"schema_version": SCHEMA, "seed": seed, "rng": RNG, "difficulty": difficulty,
              "config_sha256": digest(config), "layers": layers,
              "validity": "parameter_and_initialization_contracts_only",
              "physics_validity": "not_evaluated", "controller_outcome": "not_run"}
    result["episode_randomization"] = copy.deepcopy(report)
    return finite_json(result), report


def sensor_measurements(scene: dict[str, Any] | None, observation: dict[str, Any], step: int) -> dict[str, Any]:
    """Add noisy position measurements; preserve all ground-truth fields."""
    layer = (scene or {}).get("episode_randomization", {}).get("layers", {}).get("observation", {})
    if not layer.get("enabled") or "position_noise_std_m" not in layer.get("sampled_values", {}):
        return observation
    result = copy.deepcopy(observation)
    sigma = layer["sampled_values"]["position_noise_std_m"]
    positions = observation.get("object_poses", {})
    if not positions:
        positions = {v["object_id"]: v["pose"]["position"] for v in observation.get("scene_graph", [])}
    measured = {}
    for object_id, position in positions.items():
        measured[object_id] = [float(value) + random.Random(layer_seed(layer["seed"],
                                  f"state:{step}:position:{object_id}:{axis}")).gauss(0, sigma)
                               for axis, value in enumerate(position)]
    result["sensor_observations"] = {
        "object_positions_m": measured, "step_index": step,
        "position_noise_std_m": sigma, "ground_truth_fields_preserved": True,
        "seed": layer["seed"], "rng": RNG,
    }
    return result


def noisy_rgb(scene: dict[str, Any] | None, frame: Any, step: int) -> Any:
    layer = (scene or {}).get("episode_randomization", {}).get("layers", {}).get("observation", {})
    sigma = layer.get("sampled_values", {}).get("rgb_noise_std_uint8", 0)
    if not layer.get("enabled") or sigma == 0:
        return frame
    import numpy as np

    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB sensor noise requires an HxWx3 uint8 frame")
    rng = random.Random(layer_seed(layer["seed"], f"state:{step}:rgb"))
    noise = np.fromiter((rng.gauss(0, sigma) for _ in range(array.size)),
                        dtype=np.float64, count=array.size).reshape(array.shape)
    return np.clip(np.rint(array.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def assign_family_splits(rows: list[dict[str, Any]], *, seed: int = 0,
                         fractions: tuple[float, float, float] = (.7, .15, .15)) -> dict[str, Any]:
    """Group connected asset/task families before assigning a deterministic split.

    A shared asset family OR task structure joins records. Large connected
    components can make requested proportions impossible; report imbalance
    rather than leak families across splits to meet percentages.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("split rows must be nonempty")
    if len(fractions) != 3 or any(not math.isfinite(v) or v < 0 for v in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("split fractions must be three nonnegative values summing to one")
    parents = list(range(len(rows)))
    seen: dict[tuple[str, str], int] = {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("asset_families"), list) or not row["asset_families"]:
            raise ValueError("each split record requires asset_families and task_structure")
        families = row["asset_families"]
        if any(not isinstance(v, str) or not v for v in families) or not isinstance(row.get("task_structure"), str) or not row["task_structure"]:
            raise ValueError("family names and task_structure must be nonempty strings")
        keys = [("asset", value) for value in families] + [("task", row["task_structure"])]
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    names = ("train", "validation", "test")
    counts = Counter({name: 0 for name in names})
    assignments = {}
    ordered = sorted(groups.values(), key=lambda indices: digest([seed, sorted(
        digest(rows[i]) for i in indices)]))
    for indices in ordered:
        eligible = [i for i, fraction in enumerate(fractions) if fraction > 0]
        selected = min(eligible, key=lambda i: (counts[names[i]] / (fractions[i] * len(rows)), i))
        name = names[selected]
        counts[name] += len(indices)
        for index in indices:
            assignments[index] = name
    return {"schema_version": "scene_factory.family_split.v1", "seed": seed,
            "policy": "connected_components_shared_asset_family_or_task_structure",
            "requested_fractions": dict(zip(names, fractions, strict=True)),
            "counts": dict(counts), "component_count": len(groups),
            "empty_splits": [name for name in names if not counts[name]],
            "asset_family_leakage": False, "task_structure_leakage": False,
            "rows": [{**copy.deepcopy(row), "split": assignments[index]} for index, row in enumerate(rows)]}
