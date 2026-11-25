"""Versioned peg/socket assembly contract and conservative trace evaluation.

The evaluator checks supplied simulation measurements. It does not create an
assembly task or imply that any robot/backend can execute it.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass
from typing import Any


ASSEMBLY_SCHEMA = "scene_factory.assembly_spec.v1"


def _vec(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{label} must contain {size} finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def _unit(value: tuple[float, ...], label: str) -> tuple[float, ...]:
    norm = math.sqrt(sum(item * item for item in value))
    if norm < 1e-12:
        raise ValueError(f"{label} must be nonzero")
    return tuple(item / norm for item in value)


def _quat(value: Any, label: str) -> tuple[float, float, float, float]:
    result = _vec(value, 4, label)
    norm = math.sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{label} must be normalized in wxyz order")
    return tuple(item / norm for item in result)  # type: ignore[return-value]


def _rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx))


def _orientation_error(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    dot = min(1.0, abs(sum(a * b for a, b in zip(first, second, strict=True))))
    return 2.0 * math.acos(dot)


def _multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j, w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


@dataclass(frozen=True)
class AssemblySpec:
    spec_id: str
    peg_id: str
    socket_id: str
    peg_interface_position_m: tuple[float, float, float]
    peg_interface_orientation_wxyz: tuple[float, float, float, float]
    socket_interface_position_m: tuple[float, float, float]
    socket_interface_orientation_wxyz: tuple[float, float, float, float]
    insertion_axis_socket: tuple[float, float, float]
    target_depth_m: float
    clearance_m: tuple[float, float]
    position_tolerance_m: float
    angular_tolerance_rad: float
    max_force_N: float
    min_hold_s: float
    release_stable_s: float
    timeout_s: float
    jam_window_s: float
    min_progress_m: float
    max_linear_speed_mps: float
    max_angular_speed_rps: float
    drop_below_z_m: float
    allowed_contact_pairs: tuple[tuple[str, str], ...]
    release_max_linear_speed_mps: float = 0.005
    release_max_angular_speed_rps: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AssemblySpec":
        if not isinstance(raw, dict) or raw.get("schema_version") != ASSEMBLY_SCHEMA:
            raise ValueError(f"assembly spec schema_version must be {ASSEMBLY_SCHEMA}")
        clearance = _vec(raw.get("clearance_m"), 2, "clearance_m")
        scalar_names = ("target_depth_m", "position_tolerance_m", "angular_tolerance_rad", "max_force_N", "min_hold_s", "release_stable_s", "timeout_s", "jam_window_s", "min_progress_m", "max_linear_speed_mps", "max_angular_speed_rps", "drop_below_z_m")
        allowed = {"schema_version", "spec_id", "peg_id", "socket_id", "peg_interface_position_m", "peg_interface_orientation_wxyz", "socket_interface_position_m", "socket_interface_orientation_wxyz", "insertion_axis_socket", "clearance_m", "allowed_contact_pairs", "release_max_linear_speed_mps", "release_max_angular_speed_rps", *scalar_names}
        if set(raw) - allowed:
            raise ValueError(f"unsupported assembly fields: {sorted(set(raw) - allowed)}")
        if any(isinstance(raw.get(name), bool) or not isinstance(raw.get(name), (int, float)) for name in scalar_names):
            raise ValueError("assembly thresholds must be numbers")
        scalars = {name: float(raw[name]) for name in scalar_names}
        if not all(math.isfinite(value) for value in scalars.values()):
            raise ValueError("assembly thresholds must be finite")
        for name, value in scalars.items():
            if name == "drop_below_z_m":
                continue
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if clearance[0] < 0 or clearance[1] < clearance[0]:
            raise ValueError("clearance_m must be a nonnegative ordered interval")
        if scalars["release_stable_s"] > scalars["timeout_s"] or scalars["jam_window_s"] > scalars["timeout_s"]:
            raise ValueError("hold windows cannot exceed timeout_s")
        pairs = raw.get("allowed_contact_pairs", [])
        if not isinstance(pairs, list):
            raise ValueError("allowed_contact_pairs must be an array")
        normalized_pairs = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(item, str) and item for item in pair):
                raise ValueError("each allowed contact pair must contain two object IDs")
            normalized_pairs.append(tuple(sorted(pair)))
        ids = {"spec_id": raw.get("spec_id"), "peg_id": raw.get("peg_id"), "socket_id": raw.get("socket_id")}
        if not all(isinstance(value, str) and value for value in ids.values()) or ids["peg_id"] == ids["socket_id"]:
            raise ValueError("spec_id, distinct peg_id, and socket_id are required")
        release_linear = float(raw.get("release_max_linear_speed_mps", 0.005))
        release_angular = float(raw.get("release_max_angular_speed_rps", 0.05))
        if not all(math.isfinite(v) and v > 0 for v in (release_linear, release_angular)):
            raise ValueError("release stability speed limits must be finite and positive")
        return cls(
            **ids,
            peg_interface_position_m=_vec(raw.get("peg_interface_position_m"), 3, "peg_interface_position_m"),
            peg_interface_orientation_wxyz=_quat(raw.get("peg_interface_orientation_wxyz", [1, 0, 0, 0]), "peg_interface_orientation_wxyz"),
            socket_interface_position_m=_vec(raw.get("socket_interface_position_m"), 3, "socket_interface_position_m"),
            socket_interface_orientation_wxyz=_quat(raw.get("socket_interface_orientation_wxyz", [1, 0, 0, 0]), "socket_interface_orientation_wxyz"),
            insertion_axis_socket=_unit(_vec(raw.get("insertion_axis_socket"), 3, "insertion_axis_socket"), "insertion_axis_socket"),
            target_depth_m=scalars["target_depth_m"], clearance_m=clearance,
            position_tolerance_m=scalars["position_tolerance_m"], angular_tolerance_rad=scalars["angular_tolerance_rad"],
            max_force_N=scalars["max_force_N"], min_hold_s=scalars["min_hold_s"], release_stable_s=scalars["release_stable_s"],
            timeout_s=scalars["timeout_s"], jam_window_s=scalars["jam_window_s"], min_progress_m=scalars["min_progress_m"],
            max_linear_speed_mps=scalars["max_linear_speed_mps"], max_angular_speed_rps=scalars["max_angular_speed_rps"],
            drop_below_z_m=scalars["drop_below_z_m"], allowed_contact_pairs=tuple(normalized_pairs),
            release_max_linear_speed_mps=release_linear, release_max_angular_speed_rps=release_angular,
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return json.loads(json.dumps({"schema_version": ASSEMBLY_SCHEMA, **asdict(self)}, allow_nan=False))


def evaluate_assembly_trace(spec: AssemblySpec | dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify an SE(3)/contact trace; absent evidence never passes."""
    spec = AssemblySpec.from_dict(spec) if isinstance(spec, dict) else spec
    if not isinstance(spec, AssemblySpec):
        raise TypeError("spec must be AssemblySpec or its v1 mapping")
    if not isinstance(samples, list) or len(samples) < 2:
        return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": ["trace_insufficient"], "metrics": {}}
    try:
        clearance = float(samples[0]["geometry_clearance_m"])
        if not math.isfinite(clearance) or not spec.clearance_m[0] <= clearance <= spec.clearance_m[1]:
            return {"schema_version": "scene_factory.assembly_result.v1", "status": "failed", "classification": "geometry_mismatch", "reasons": ["clearance_out_of_range"], "metrics": {"clearance_m": clearance if math.isfinite(clearance) else None}}
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": ["geometry_clearance_evidence_missing"], "metrics": {}}
    previous: dict[str, Any] | None = None
    rows = []
    for index, sample in enumerate(samples):
        try:
            time_s = float(sample["time_s"])
            parts = sample["parts"]
            peg, socket = parts[spec.peg_id], parts[spec.socket_id]
            peg_position = _vec(peg["position_m"], 3, "peg.position_m")
            socket_position = _vec(socket["position_m"], 3, "socket.position_m")
            peg_q = _quat(peg["orientation_wxyz"], "peg.orientation_wxyz")
            socket_q = _quat(socket["orientation_wxyz"], "socket.orientation_wxyz")
            contacts = sample["contacts"]
            released = sample["released"]
            if not math.isfinite(time_s) or time_s < 0 or not isinstance(contacts, list) or not isinstance(released, bool):
                raise ValueError("invalid time/contact/released sample")
            if previous is not None and time_s <= previous["time_s"]:
                raise ValueError("timestamps must be strictly increasing")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": [f"trace_invalid:{type(exc).__name__}"], "metrics": {"sample_index": index, "detail": str(exc)}}
        if index > 0 and sample.get("motion_source") not in {"robot_joint_control", "physics_passive"}:
            return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": ["physical_motion_source_missing"], "metrics": {"sample_index": index}}
        if sample.get("motion_source") == "robot_joint_control" and (not isinstance(sample.get("action"), dict) or not sample["action"]):
            return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": ["robot_action_missing"], "metrics": {"sample_index": index}}
        if sample.get("teleported") or sample.get("kinematic_pose_write"):
            return {"schema_version": "scene_factory.assembly_result.v1", "status": "failed", "classification": "invalid_trace", "reasons": ["direct_pose_write_not_assembly"], "metrics": {"sample_index": index}}
        peg_interface_world = tuple(a + b for a, b in zip(peg_position, _rotate(peg_q, spec.peg_interface_position_m), strict=True))
        socket_interface_world = tuple(a + b for a, b in zip(socket_position, _rotate(socket_q, spec.socket_interface_position_m), strict=True))
        socket_axis_local = _rotate(spec.socket_interface_orientation_wxyz, spec.insertion_axis_socket)
        axis_world = _unit(_rotate(socket_q, socket_axis_local), "world insertion axis")
        relative = tuple(a - b for a, b in zip(peg_interface_world, socket_interface_world, strict=True))
        axial = sum(a * b for a, b in zip(relative, axis_world, strict=True))
        lateral = math.sqrt(sum((relative[i] - axial * axis_world[i]) ** 2 for i in range(3)))
        angular = _orientation_error(_multiply(peg_q, spec.peg_interface_orientation_wxyz), _multiply(socket_q, spec.socket_interface_orientation_wxyz))
        force_samples = []
        unknown_contacts = []
        fitting_contact = False
        for contact in contacts:
            if not isinstance(contact, dict) or not isinstance(contact.get("pair"), list) or len(contact["pair"]) != 2 or not all(isinstance(v, str) and v for v in contact["pair"]) or "normal_force_N" not in contact:
                return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": ["contact_sample_invalid"], "metrics": {"sample_index": index}}
            pair = tuple(sorted(contact["pair"]))
            try:
                force = float(contact["normal_force_N"])
            except (TypeError, ValueError, OverflowError):
                force = float("nan")
            if not math.isfinite(force) or force < 0:
                return {"schema_version": "scene_factory.assembly_result.v1", "status": "not_verified", "reasons": ["contact_force_invalid"], "metrics": {"sample_index": index}}
            force_samples.append(force)
            fitting_contact = fitting_contact or (pair == tuple(sorted((spec.peg_id, spec.socket_id))) and force > 0)
            if pair not in spec.allowed_contact_pairs:
                unknown_contacts.append(list(pair))
        row = {"time_s": time_s, "depth_m": max(0.0, axial), "lateral_error_m": lateral, "angular_error_rad": angular, "force_N": max(force_samples, default=0.0), "contact": bool(contacts), "fitting_contact": fitting_contact, "unknown_contacts": unknown_contacts, "released": released, "dropped": bool(sample.get("dropped", False)), "linear_speed_mps": 0.0, "angular_speed_rps": 0.0, "position_m": peg_position}
        if previous is not None:
            dt = time_s - previous["time_s"]
            linear_speed = math.dist(peg_position, previous["position_m"]) / dt
            quaternion_delta = _orientation_error(peg_q, previous["peg_q"])
            angular_speed = quaternion_delta / dt
            if linear_speed > spec.max_linear_speed_mps or angular_speed > spec.max_angular_speed_rps:
                return {"schema_version": "scene_factory.assembly_result.v1", "status": "failed", "classification": "invalid_trace", "reasons": ["teleport_or_unresolved_motion"], "metrics": {"sample_index": index, "linear_speed_mps": linear_speed, "angular_speed_rps": angular_speed}}
            row["linear_speed_mps"] = linear_speed
            row["angular_speed_rps"] = angular_speed
        row["peg_q"] = peg_q
        rows.append(row)
        previous = row
    peak_force = max(row["force_N"] for row in rows)
    end = rows[-1]
    reasons: list[str] = []
    classification = "success"
    stable_release_duration = 0.0
    first = rows[0]
    if first["depth_m"] >= spec.target_depth_m - spec.position_tolerance_m and first["lateral_error_m"] <= spec.position_tolerance_m:
        classification, reasons = "invalid_trace", ["initially_assembled_is_not_insertion"]
    elif peak_force > spec.max_force_N:
        classification, reasons = "overforce", ["maximum_force_exceeded"]
    elif any(row["unknown_contacts"] for row in rows):
        classification, reasons = "unexpected_contact", ["unlisted_contact_pair"]
    elif any(row["dropped"] or row["position_m"][2] < spec.drop_below_z_m for row in rows):
        classification, reasons = "dropped", ["part_dropped"]
    elif end["angular_error_rad"] > spec.angular_tolerance_rad:
        classification, reasons = "tilted", ["angular_tolerance_exceeded"]
    elif abs(end["depth_m"]-spec.target_depth_m) > spec.position_tolerance_m or end["lateral_error_m"] > spec.position_tolerance_m:
        recent = [row for row in rows if row["time_s"] >= end["time_s"] - spec.jam_window_s]
        jammed = recent[-1]["time_s"] - recent[0]["time_s"] >= spec.jam_window_s - 1e-9 and end["contact"] and max(row["depth_m"] for row in recent) - min(row["depth_m"] for row in recent) < spec.min_progress_m
        classification = "jammed" if jammed else "timeout"
        reasons = [f"{classification}_before_target"]
    else:
        target_rows = []
        for row in reversed(rows):
            if not (spec.target_depth_m - spec.position_tolerance_m <= row["depth_m"] <= spec.target_depth_m + spec.position_tolerance_m and row["lateral_error_m"] <= spec.position_tolerance_m and row["angular_error_rad"] <= spec.angular_tolerance_rad):
                break
            target_rows.append(row)
        held = target_rows[0]["time_s"] - target_rows[-1]["time_s"] if target_rows else 0.0
        if held < spec.min_hold_s:
            classification, reasons = "timeout", ["hold_time_not_met"]
        elif not any(row["fitting_contact"] for row in rows) or not any(sample.get("motion_source") == "robot_joint_control" for sample in samples[1:]):
            classification, reasons = "invalid_trace", ["robot_action_and_contact_evidence_required"]
        elif not end["released"]:
            classification, reasons = "timeout", ["part_not_released"]
        else:
            released_rows = []
            for row in target_rows:
                if not row["released"]:
                    break
                released_rows.append(row)
            if not released_rows or end["time_s"] - released_rows[-1]["time_s"] < spec.release_stable_s:
                classification, reasons = "timeout", ["release_stability_window_not_met"]
            else:
                # A settling transient restarts the continuous stability timer;
                # it must not permanently invalidate every later stable window.
                # Target pose, release, and speed must all hold at the end.
                stable_rows = []
                for row in released_rows:
                    if row["linear_speed_mps"] > spec.release_max_linear_speed_mps or row["angular_speed_rps"] > spec.release_max_angular_speed_rps:
                        break
                    stable_rows.append(row)
                if stable_rows:
                    stable_release_duration = end["time_s"]-stable_rows[-1]["time_s"]
                if stable_release_duration+1e-9 < spec.release_stable_s:
                    classification, reasons = "timeout", ["released_assembly_unstable"]
    total_time = rows[-1]["time_s"] - rows[0]["time_s"]
    if classification == "success" and total_time > spec.timeout_s:
        classification, reasons = "timeout", ["task_timeout"]
    return {"schema_version": "scene_factory.assembly_result.v1", "status": "passed" if classification == "success" else "failed", "classification": classification, "reasons": reasons, "metrics": {"sample_count": len(rows), "duration_s": total_time, "peak_force_N": peak_force, "final_depth_m": end["depth_m"], "final_lateral_error_m": end["lateral_error_m"], "final_angular_error_rad": end["angular_error_rad"], "stable_release_duration_s": stable_release_duration, "evidence": "trace_classifier_only; physical_validity_depends_on_source_measurements"}}

