"""Project one committed simulator observation into a portable world-state view."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from .session_contracts import backend_capabilities, finite_json

WORLD_STATE_SCHEMA = "scene_factory.world_state.v1"


def _vector(value, size, label):
    if value is None:
        return None
    if (
        not isinstance(value, (tuple, list))
        or len(value) != size
        or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in value
        )
    ):
        raise ValueError(f"{label} requires {size} finite numbers")
    result = list(value)
    if label == "orientation_wxyz" and abs(math.hypot(*result) - 1) > 0.001:
        raise ValueError("orientation_wxyz must be normalized")
    return result


def _entity(item, live, backend):
    interactions = item.get("interactions") or {}
    pos = _vector(live.get("position_m", live.get("position")), 3, "position_m")
    pose_source = live.get("pose_source") or {
        "mujoco": "MuJoCo MjData",
        "isaac": "Isaac scene transform",
    }.get(backend)
    velocities = {
        key: _vector(live.get(key), 3, key)
        for key in ("linear_velocity_mps", "angular_velocity_rps")
    }
    return {
        "object_id": item["object_id"],
        "asset_id": item.get("asset_id"),
        "category": item.get("category"),
        "dynamic": item.get("dynamic"),
        "asset_metadata": {
            "source": "layout_asset_declaration",
            "asset_id": item.get("asset_id"),
            "category": item.get("category"),
            "fallback_reason": item.get("fallback_reason"),
            "regions": interactions.get("regions", []),
            "interior_regions": interactions.get("interior_regions", []),
            "semantic_states": interactions.get("semantic_states", []),
        },
        "instance": {
            "source": "episode_layout",
            "initial_pose": item.get("pose"),
            "bbox_m": item.get("bbox_m"),
            "authored_support": item.get("support"),
            "authored_relations": item.get("relations", []),
            "physics_override": item.get("physics"),
            "initial_joints": interactions.get("joints", []),
            "initial_states": interactions.get("states", []),
        },
        "position_m": pos,
        "orientation_wxyz": _vector(live.get("orientation_wxyz"), 4, "orientation_wxyz"),
        **velocities,
        "pose_source": pose_source if pos is not None else None,
        "velocity_source": (
            live.get("velocity_source", pose_source)
            if any(x is not None for x in velocities.values())
            else None
        ),
        "velocity_reference": live.get("velocity_reference"),
        "velocity_frame": live.get("velocity_frame"),
        "center_of_mass_position_m": _vector(
            live.get("center_of_mass_position_m"), 3, "center_of_mass_position_m"
        ),
        "physics_handle_valid": live.get("physics_handle_valid"),
        "joint_states": live.get("joint_states"),
        "affordances": [
            {
                "region_id": region["region_id"],
                "actions": region.get("allowed_actions", []),
                "source": "asset_interaction_region",
                "available": None,
            }
            for region in interactions.get("regions", [])
        ],
    }


def _task_affordances(objects, task, observation, backend):
    success = task.get("success") or {}
    if success.get("predicate") != "pick_and_place":
        return
    target = task.get("target_object") or success.get("subject")
    if target not in objects:
        return
    # The phase describes admissible commands. It does not certify a feasible grasp.
    robot = observation.get("robot") or {}
    phase = robot.get("phase")
    known = (
        backend == "isaac"
        and isinstance(observation.get("mug_skill"), dict)
        and objects[target]["position_m"] is not None
        and isinstance(phase, str)
    )
    for action, phases in (
        ("grasp", {"GRASP", "VERIFY_GRASP"}),
        ("place", {"LOWER", "RELEASE", "VERIFY_PLACE"}),
    ):
        objects[target]["affordances"].append(
            {
                "region_id": None,
                "actions": [action],
                "source": "task_contract.pick_and_place",
                "available": phase in phases if known else None,
                "availability_source": "current_mug_controller_phase" if known else None,
                "availability_scope": "controller_stage",
                "phase": phase if known else None,
            }
        )


def _relations(objects, contacts):
    # Distances are between observed body origins. Directed contact quantities remain in contacts.
    located = [key for key, item in objects.items() if item["position_m"] is not None]
    result = [
        {
            "subject": a,
            "target": b,
            "kind": "origin_distance",
            "distance_m": math.dist(objects[a]["position_m"], objects[b]["position_m"]),
            "source": "observed_positions",
        }
        for a, b in combinations(sorted(located), 2)
    ]
    pairs = set()
    for contact in contacts or []:
        pair = contact.get("pair", [contact.get("object1"), contact.get("object2")])
        if len(pair) == 2 and all(isinstance(key, str) for key in pair):
            pairs.add(tuple(sorted(pair)))
    result.extend(
        {"subject": a, "target": b, "kind": "contact", "source": "backend_contacts"}
        for a, b in sorted(pairs)
    )
    return result


def build_world_state(
    scene: dict[str, Any],
    observation: dict[str, Any],
    *,
    backend: str,
    episode_id: str,
    step_index: int,
    state_version: int | None = None,
    session_id: str | None = None,
    scene_sha256: str | None = None,
    sim_time: float | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a detached JSON view. Runtime quantities have null values when absent.

    Callers supply the effective episode layout and one observation returned by reset,
    step or restore. Timestamps use the episode simulation clock in seconds.
    """
    capabilities = backend_capabilities(backend)
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be a nonempty string")
    if state_version is None:
        state_version = step_index
    for name, value in (("step_index", step_index), ("state_version", state_version)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if observation.get("simulation_step", step_index) != step_index:
        raise ValueError("observation and requested step_index differ")
    clock = observation.get("sim_time", observation.get("timestamp"))
    if (
        sim_time is not None
        and clock is not None
        and not math.isclose(sim_time, clock, rel_tol=1e-10, abs_tol=1e-12)
    ):
        raise ValueError("observation and requested sim_time differ")
    timestamp = sim_time if sim_time is not None else clock
    if timestamp is not None and (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
        or timestamp < 0
    ):
        raise ValueError("simulation timestamp must be finite and nonnegative")
    physical = capabilities["physical"]
    measured = observation.get("objects", {}) if physical else {}
    objects = {}
    for item in scene.get("objects", []):
        key = item["object_id"]
        if key in objects:
            raise ValueError(f"duplicate scene object_id: {key}")
        objects[key] = _entity(item, measured.get(key, {}), backend)
    if measured.keys() - objects.keys():
        raise ValueError("observation contains objects outside this episode layout")
    robot = observation.get("robot") if physical else None
    contacts = observation.get("contacts") if physical else None
    _task_affordances(objects, scene.get("task", {}), observation, backend)
    oracle = (robot or {}).get("task_oracle", (info or {}).get("task"))
    result = {
        "schema_version": WORLD_STATE_SCHEMA,
        "source": "runtime_observation" if physical else "dry_run",
        "scene_id": scene.get("scene_id"),
        "scene_sha256": scene_sha256,
        "backend": backend,
        "fidelity": capabilities["fidelity"],
        "session_id": session_id,
        "episode_id": episode_id,
        "step_index": step_index,
        "state_version": state_version,
        "timestamp": timestamp,
        "sim_time": timestamp,
        "coordinates": {
            "frame": "world",
            "up_axis": "Z",
            "length": "m",
            "time": "s",
            "angle": "rad",
            "quaternion_order": "wxyz",
            "clock": "episode_simulation",
        },
        "objects": objects,
        "spatial_relations": _relations(objects, contacts),
        "contacts": contacts,
        "robot": robot,
        "contact_coverage": observation.get("contact_coverage"),
        "generalized_coordinates": observation.get("proprioception")
        if backend == "mujoco"
        else None,
        "mug_skill": observation.get("mug_skill"),
        "grasp_diagnostics": (robot or {}).get("grasp_diagnostics"),
        "task": scene.get("task", {}),
        "task_oracle": oracle,
        "task_progress": {
            "success": observation.get("task_success", (oracle or {}).get("task_success")),
            "phase": (robot or {}).get("phase"),
            "failure_reason": (robot or {}).get("failure_reason"),
        },
    }
    result["unavailable_fields"] = [
        key
        for key in ("timestamp", "robot", "contacts", "contact_coverage", "task_oracle")
        if result[key] is None
    ]
    return finite_json(result, "world_state")
