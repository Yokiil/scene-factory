"""Compact, read-only summaries of recorded grasp measurements.

No interpolation, filtering, acceptance classification or simulation is performed.
Repeated phase names remain separate segments so waits and retries are not merged.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _vector(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    values = tuple(_number(item) for item in value)
    return values if all(item is not None for item in values) else None


def _contact_forces(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    contacts = row.get("shape_contacts")
    if not isinstance(contacts, list):
        return None
    left = right = peak = 0.
    for contact in contacts:
        if not isinstance(contact, Mapping):
            return None
        path, force = contact.get("collider"), _number(contact.get("normal_force_N"))
        if not isinstance(path, str) or force is None or force < 0:
            return None
        components = path.split("/")
        if "panda_leftfinger" in components:
            left += force
        if "panda_rightfinger" in components:
            right += force
        peak = max(peak, force)
    return left, right, peak


def _event(row: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    # These are recorded controller events, not conclusions inferred from speed.
    for field in ("grasp_control", "grasp_decision", "control"):
        decision = row.get(field)
        if isinstance(decision, Mapping):
            # Runner rows may carry the action decision from the preceding
            # measurement. Do not attribute it to the resulting measurement.
            if "input_time_s" in decision:
                input_time = _number(decision["input_time_s"])
                if input_time is None or input_time < 0 or input_time != _number(row.get("time_s")):
                    continue
            reason = decision.get("reason")
            if isinstance(reason, str) and reason:
                return {"sample_index": index, "time_s": _number(row.get("time_s")),
                        "source_field": field, "status": str(decision.get("status", ""))[:80],
                        "reason": reason[:240]}
    for field in ("failure_reason", "protection_reason"):
        reason = row.get(field)
        if isinstance(reason, str) and reason:
            return {"sample_index": index, "time_s": _number(row.get("time_s")),
                    "source_field": field, "reason": reason[:240]}
    return None


def _decision_events(samples, decisions):
    """Bind events only to a unique measurement with the exact input timestamp."""
    times = {}
    for index, sample in enumerate(samples):
        time_s = _number(sample.get("time_s"))
        if time_s is not None and time_s >= 0:
            times.setdefault(time_s, []).append(index)
    events = {}
    counts = {"provided_count": len(decisions), "event_count": 0, "bound_count": 0,
              "unmatched_count": 0, "ambiguous_count": 0, "invalid_time_count": 0}
    for index, decision in enumerate(decisions):
        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason:
            continue
        counts["event_count"] += 1
        time_s = _number(decision.get("input_time_s"))
        if time_s is None or time_s < 0:
            counts["invalid_time_count"] += 1
            continue
        matches = times.get(time_s, [])
        if not matches:
            counts["unmatched_count"] += 1
            continue
        if len(matches) != 1:
            counts["ambiguous_count"] += 1
            continue
        sample_index = matches[0]
        counts["bound_count"] += 1
        events.setdefault(sample_index, {
            "sample_index": sample_index, "time_s": time_s, "input_time_s": time_s,
            "source_field": "control_decisions", "decision_index": index,
            "status": str(decision.get("status", ""))[:80], "reason": reason[:240]})
    return events, counts


def _segment(rows, first_index, contact_force_N, decision_events):
    start, end = rows[0], rows[-1]
    times = [_number(row.get("time_s")) for row in rows]
    valid_time = all(time is not None and time >= 0 for time in times)
    valid_time = valid_time and all(b > a for a, b in zip(times, times[1:]))
    result = {"phase": str(start.get("phase", "unknown"))[:120],
              "first_sample_index": first_index, "sample_count": len(rows),
              "start_time_s": times[0], "end_time_s": times[-1],
              "duration_s": times[-1]-times[0] if valid_time else None,
              "strictly_increasing_time": valid_time,
              "max_linear_speed_mps": None, "max_angular_speed_rps": None,
              "max_pair_normal_force_N": None, "max_shape_normal_force_N": None,
              "min_bilateral_finger_force_N": None, "bilateral_contact_sample_count": 0,
              "shape_contact_sample_count": 0, "contact_coverage_attested_count": 0,
              "linear_velocity_sample_count": 0, "angular_velocity_sample_count": 0,
              "peg_rise_m": None, "tcp_rise_m": None,
              "first_recorded_event": None}

    def maximum(name, value):
        previous = result[name]
        result[name] = value if previous is None else max(previous, value)

    for offset, row in enumerate(rows):
        live = row.get("live_peg", {})
        live = live if isinstance(live, Mapping) else {}
        for field, output, count in (
            ("linear_velocity_mps", "max_linear_speed_mps", "linear_velocity_sample_count"),
            ("angular_velocity_rps", "max_angular_speed_rps", "angular_velocity_sample_count"),
        ):
            vector = _vector(live.get(field))
            if vector is not None:
                maximum(output, math.hypot(*vector))
                result[count] += 1
        contacts = row.get("contacts")
        if isinstance(contacts, list):
            values = [_number(c.get("normal_force_N")) if isinstance(c, Mapping) else None
                      for c in contacts]
            if all(value is not None and value >= 0 for value in values):
                maximum("max_pair_normal_force_N", max(values, default=0.))
        forces = _contact_forces(row)
        if forces is not None:
            left, right, peak = forces
            result["shape_contact_sample_count"] += 1
            result["bilateral_contact_sample_count"] += int(min(left, right) >= contact_force_N)
            previous = result["min_bilateral_finger_force_N"]
            result["min_bilateral_finger_force_N"] = min(left, right) if previous is None else min(previous, left, right)
            maximum("max_shape_normal_force_N", peak)
        result["contact_coverage_attested_count"] += int(row.get("grasp_contact_coverage_complete") is True)
        if result["first_recorded_event"] is None:
            sample_index = first_index+offset
            result["first_recorded_event"] = decision_events.get(sample_index) or _event(row, sample_index)

    count = result["shape_contact_sample_count"]
    result["bilateral_contact_fraction_of_measured_samples"] = (
        result["bilateral_contact_sample_count"]/count if count else None)
    result["missing_shape_contact_sample_count"] = len(rows)-count
    # Use exact segment endpoints: a missing endpoint must not become a shorter rise.
    for key, output in (("position", "peg_rise_m"), ("end_effector_position_m", "tcp_rise_m")):
        a = start.get("live_peg", {}) if key == "position" else start
        b = end.get("live_peg", {}) if key == "position" else end
        a = _vector(a.get(key)) if isinstance(a, Mapping) else None
        b = _vector(b.get(key)) if isinstance(b, Mapping) else None
        if a is not None and b is not None:
            result[output] = b[2]-a[2]
    return result


def summarize_grasp_phases(samples: Sequence[Mapping[str, Any]], *,
                           contact_force_N: float = .05, max_phases: int = 64,
                           control_decisions: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize raw trial rows without altering them or declaring task success.

    Missing or nonfinite measurements are omitted from maxima and counted, never
    interpreted as zero. Empty measured contact lists are zero contact; absent
    lists are missing evidence. ``max_phases`` bounds the returned segments.
    Optional controller events bind by exact ``input_time_s`` to one unique
    measurement, including the terminal decision that causes no next action.
    Unmatched, invalid or ambiguous event timestamps are counted but never
    attributed to a phase. Neither input sequence is modified.
    """
    threshold = _number(contact_force_N)
    if threshold is None or threshold <= 0:
        raise ValueError("contact_force_N must be finite and positive")
    if isinstance(max_phases, bool) or not isinstance(max_phases, int) or not 1 <= max_phases <= 256:
        raise ValueError("max_phases must be an integer between 1 and 256")
    if not isinstance(samples, Sequence) or any(not isinstance(row, Mapping) for row in samples):
        raise ValueError("samples must be a sequence of measurement mappings")
    if control_decisions is None:
        control_decisions = ()
    if (not isinstance(control_decisions, Sequence)
            or any(not isinstance(decision, Mapping) for decision in control_decisions)):
        raise ValueError("control_decisions must be a sequence of decision mappings")
    decision_events, event_counts = _decision_events(samples, control_decisions)
    phases = []
    index = segment_count = 0
    while index < len(samples):
        end = index+1
        while end < len(samples) and samples[end].get("phase") == samples[index].get("phase"):
            end += 1
        segment_count += 1
        if len(phases) < max_phases:
            phases.append(_segment(samples[index:end], index, threshold, decision_events))
        index = end
    return {"schema_version": "scene_factory.grasp_phase_summary.v1",
            "scope": "saved raw measurements only; no simulation, filtering or task acceptance",
            "sample_count": len(samples), "phase_segment_count": segment_count,
            "omitted_phase_segment_count": segment_count-len(phases),
            "bilateral_contact_threshold_N": threshold,
            "control_decision_events": event_counts, "phases": phases}
