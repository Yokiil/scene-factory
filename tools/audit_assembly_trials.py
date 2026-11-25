"""Read-only audit of recorded assembly evidence; never starts a simulator.

Historical report fields are preserved as claims. Current acceptance is
recomputed from the saved measurements and current evaluator source. A process
exit code is transport evidence only, including when it is zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.assembly import AssemblySpec, evaluate_assembly_trace
from scene_factory.dataset import write_json_atomic


SCHEMA = "scene_factory.assembly_trials_audit.v1"
DEFAULT_PATTERNS = ("assembly_precision_*", "room*_physics_*", "assembly_online_*",
                    "square_assembly_*", "square_room*_physics_*", "square_room*_online_*")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _velocity(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        return None
    return [float(item) for item in value]


def _state(row: dict[str, Any], spec: AssemblySpec) -> dict[str, Any]:
    part = row.get("parts", {}).get(spec.peg_id, {})
    return part if "linear_velocity_mps" in part or "angular_velocity_rps" in part else row.get("live_peg", {})


def audit_live_velocity(rows: list[dict[str, Any]], spec: AssemblySpec) -> dict[str, Any]:
    missing, violations, release_violations = [], [], []
    maximum_linear = maximum_angular = 0.
    end_time = rows[-1].get("time_s") if rows else None
    for index, row in enumerate(rows):
        state = _state(row, spec)
        linear, angular = _velocity(state.get("linear_velocity_mps")), _velocity(state.get("angular_velocity_rps"))
        if linear is None or angular is None:
            missing.append(index)
            continue
        linear_speed = math.sqrt(sum(value*value for value in linear))
        angular_speed = math.sqrt(sum(value*value for value in angular))
        maximum_linear, maximum_angular = max(maximum_linear, linear_speed), max(maximum_angular, angular_speed)
        detail = {"sample_index": index, "time_s": row.get("time_s"), "phase": row.get("phase"),
                  "linear_speed_mps": linear_speed, "angular_speed_rps": angular_speed}
        if linear_speed > spec.max_linear_speed_mps or angular_speed > spec.max_angular_speed_rps:
            violations.append(detail)
        time_s = row.get("time_s")
        if (row.get("released") is True and isinstance(time_s, (int, float)) and isinstance(end_time, (int, float))
                and time_s >= end_time-spec.release_stable_s-1e-9
                and (linear_speed > spec.release_max_linear_speed_mps or angular_speed > spec.release_max_angular_speed_rps)):
            release_violations.append(detail)
    status = "failed" if violations or release_violations else ("not_verified" if missing or not rows else "passed")
    return {"status": status, "scope": "measured_peg_linear_and_angular_velocity_each_saved_physics_step",
            "max_linear_speed_mps": maximum_linear if rows else None, "max_angular_speed_rps": maximum_angular if rows else None,
            "limits": {"linear_speed_mps": spec.max_linear_speed_mps, "angular_speed_rps": spec.max_angular_speed_rps,
                       "release_linear_speed_mps": spec.release_max_linear_speed_mps, "release_angular_speed_rps": spec.release_max_angular_speed_rps},
            "missing_sample_count": len(missing), "missing_sample_indices_first20": missing[:20],
            "violation_count": len(violations), "violations_first20": violations[:20],
            "terminal_release_velocity_violation_count": len(release_violations),
            "terminal_release_velocity_violations_first20": release_violations[:20],
            "finite_difference_velocity_is_not_substituted": True}


def audit_coverage(rows: list[dict[str, Any]], spec: AssemblySpec, report: dict[str, Any]) -> dict[str, Any]:
    errors = []
    previous_time, previous_dt = None, None
    scopes = set()
    for index, row in enumerate(rows):
        coverage = row.get("contact_coverage")
        reason = None
        if not isinstance(coverage, dict):
            reason = "per_step_coverage_absent"
        elif coverage.get("complete") is not True or coverage.get("continuous") is not True:
            reason = "coverage_incomplete_or_not_continuous"
        elif coverage.get("scope") not in {"all_scene_contacts", "all_contacts_involving_task_parts"}:
            reason = "coverage_scope_insufficient"
        elif coverage.get("force_aggregation") != "sum_per_point_normal_magnitudes":
            reason = "per_point_normal_magnitude_sum_not_attested"
        elif not {spec.peg_id, spec.socket_id}.issubset(set(coverage.get("object_ids", []))):
            reason = "both_task_parts_not_covered"
        elif coverage.get("buffer_saturated") is True or any(item.get("saturated") is True for item in coverage.get("contact_buffers", [])):
            reason = "contact_buffer_saturated"
        if reason:
            errors.append({"sample_index": index, "reason": reason})
            continue
        scopes.add(coverage["scope"])
        time_s, dt = row.get("time_s"), coverage.get("sample_dt_s")
        if (isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s)
                or isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0):
            errors.append({"sample_index": index, "reason": "coverage_time_window_invalid"})
        elif previous_time is not None and (not math.isclose(time_s-previous_time, dt, rel_tol=1e-6, abs_tol=1e-8)
                                           or not math.isclose(dt, previous_dt, rel_tol=1e-6, abs_tol=1e-9)):
            errors.append({"sample_index": index, "reason": "physics_step_gap_or_changed_dt"})
        previous_time, previous_dt = time_s, dt
        for object_id in (spec.peg_id, spec.socket_id):
            state = row.get("parts", {}).get(object_id, {})
            if _velocity(state.get("linear_velocity_mps")) is None or _velocity(state.get("angular_velocity_rps")) is None:
                errors.append({"sample_index": index, "reason": f"full_live_task_part_motion_absent:{object_id}"})
        if not isinstance(coverage.get("source"), str) or not coverage["source"]:
            errors.append({"sample_index": index, "reason": "measurement_source_missing"})
    report_coverage = report.get("contact_coverage", {})
    return {"status": "passed" if rows and not errors else "not_verified",
            "complete_task_part_coverage_each_physics_step": bool(rows) and not errors,
            "scopes": sorted(scopes), "error_count": len(errors), "errors_first20": errors[:20],
            "historical_report_scope": report_coverage.get("scope"),
            "historical_peg_only_filtered_contact_sensor": "against peg" in str(report_coverage.get("scope", "")),
            "coverage_source": "saved_backend_coverage_attestation; no independent sensor rerun",
            "per_contact_point_normal_forces_persisted": False,
            "point_force_note": "Current Isaac adapter saves body-pair sums computed from point magnitudes, plus point counts; it does not persist every point force vector."}


def audit_release(rows: list[dict[str, Any]], spec: AssemblySpec, report: dict[str, Any]) -> dict[str, Any]:
    released_rows = [row for row in rows if row.get("released") is True]
    current_contacts = []
    if rows:
        for row in reversed(rows):
            if row.get("released") is not True:
                break
            if any(set(item.get("pair", [])) == {"robot", spec.peg_id} and item.get("normal_force_N", 0.) > 1e-6 for item in row.get("contacts", [])):
                current_contacts.append(row.get("time_s"))
    state = report.get("online_state") or report.get("online_final", {}).get("observation", {}).get("assembly_status", {})
    declaration = state.get("release_evidence", {}) if isinstance(state, dict) else {}
    measured = (declaration.get("finger_contact_before_open") is True and declaration.get("actual_gripper_open") is True
                and declaration.get("current_robot_peg_contact") is False and declaration.get("released") is True
                and isinstance(declaration.get("source"), str) and bool(declaration["source"]))
    finger_rows = [row for row in rows if "finger_positions_m" in row]
    return {"status": "passed" if released_rows and measured and not current_contacts else "not_verified",
            "released_flag_appears": bool(released_rows), "actual_open_and_grasp_history_attested": measured,
            "terminal_released_window_robot_contacts": current_contacts[:20],
            "terminal_released_window_robot_contact_count": len(current_contacts),
            "recorded_terminal_release_evidence": declaration,
            "offline_finger_DOF_samples": len(finger_rows),
            "last_offline_finger_positions_m": finger_rows[-1]["finger_positions_m"] if finger_rows else None,
            "scope": "online release proof binds grasp history, measured open DOFs and no robot/peg contacts; offline commanded release alone is insufficient"}


def _fixture_path(report: dict[str, Any], fixture_mappings: dict[str, Path] | None = None) -> tuple[Path | None, dict[str, Any]]:
    fixture_value = report.get("parameters", {}).get("fixture")
    normalized = fixture_value.replace("\\", "/").rstrip("/") if isinstance(fixture_value, str) else None
    for original, replacement in sorted((fixture_mappings or {}).items(), key=lambda item: -len(item[0])):
        prefix = original.replace("\\", "/").rstrip("/")
        if normalized is not None and (normalized.casefold() == prefix.casefold() or normalized.casefold().startswith(prefix.casefold()+"/")):
            suffix = normalized[len(prefix):].lstrip("/")
            if ".." in suffix.split("/"):
                raise ValueError("fixture mapping suffix escapes its declared root")
            target = Path(replacement).expanduser().resolve()/suffix
            return target, {"original": fixture_value, "mapped_from": original, "mapped_to": str(replacement), "resolved": str(target)}
    fixture = Path(fixture_value).expanduser() if isinstance(fixture_value, str) else None
    if fixture is not None and not fixture.is_absolute():
        fixture = Path(__file__).resolve().parents[1]/fixture
    return fixture, {"original": fixture_value, "resolved": str(fixture) if fixture else None, "mapped_from": None}


def _geometry(directory: Path, report: dict[str, Any], rows: list[dict[str, Any]], spec: AssemblySpec,
              fixture_mappings: dict[str, Path] | None = None) -> dict[str, Any]:
    fixture, mapping = _fixture_path(report, fixture_mappings)
    actual_fixture = fixture/"scene.usda" if fixture is not None else None
    expected = report.get("fixture_sha256")
    actual = sha256(actual_fixture) if actual_fixture and actual_fixture.is_file() else None
    source_path = fixture/"SOURCE.json" if fixture is not None else None
    clearance, source = None, {}
    if source_path and source_path.is_file():
        source = _json(source_path)
        clearance = source.get("geometry", {}).get("clearance_m")
    values = [row.get("geometry_clearance_m") for row in rows]
    clearance_matches = bool(values) and isinstance(clearance, (int, float)) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        and math.isclose(value, clearance, rel_tol=0, abs_tol=1e-12) and spec.clearance_m[0] <= value <= spec.clearance_m[1] for value in values)
    source_usd_matches = bool(actual) and source.get("usd_sha256") == actual
    source_spec_matches = "frozen_spec_sha256" not in source or source["frozen_spec_sha256"] == report.get("spec_sha256")
    matched = bool(expected) and actual == expected and clearance_matches and source_usd_matches and source_spec_matches
    return {"status": "passed" if matched else "not_verified", "fixture_path": str(actual_fixture) if actual_fixture else None,
            "fixture_mapping": mapping, "fixture_id": source.get("fixture_id"), "peg_shape": source.get("geometry", {}).get("peg_shape", "legacy_fixture"),
            "recorded_fixture_sha256": expected, "current_fixture_sha256": actual,
            "fixture_hash_matches_recorded": bool(expected) and actual == expected,
            "source_manifest_sha256": sha256(source_path) if source_path and source_path.is_file() else None,
            "source_manifest_usd_hash_matches": source_usd_matches, "source_manifest_spec_hash_matches_when_present": source_spec_matches,
            "source_clearance_m": clearance, "trace_clearance_matches_source_and_frozen_range": clearance_matches,
            "scope": "hash-bound authored fixture and declared clearance; no independent geometry/solver/robot-contact acceptance is inferred"}


def _spec_binding(spec_path: Path, report: dict[str, Any], spec: AssemblySpec,
                  fixture_mappings: dict[str, Path] | None = None) -> dict[str, Any]:
    """Allow a reformatted trial copy only through a matching frozen original."""
    expected = report.get("spec_sha256")
    copy_hash = sha256(spec_path)
    direct_match = bool(expected) and copy_hash == expected
    fixture, mapping = _fixture_path(report, fixture_mappings)
    original = fixture/"assembly_spec.json" if fixture else None
    original_hash = sha256(original) if original and original.is_file() else None
    equivalent = False
    if original and original.is_file() and original_hash == expected:
        equivalent = AssemblySpec.from_dict(_json(original)).to_dict() == spec.to_dict()
    return {"status": "passed" if direct_match or equivalent else "not_verified",
            "recorded_source_spec_sha256": expected, "trial_copy_sha256": copy_hash,
            "trial_copy_byte_hash_matches_recorded": direct_match,
            "original_source_spec_path": str(original) if original else None,
            "fixture_mapping": mapping,
            "original_source_spec_sha256": original_hash,
            "trial_copy_semantically_matches_hash_bound_original": equivalent,
            "scope": "Exact source hash plus parsed frozen thresholds; JSON key order/newlines in a trial copy may differ."}


def audit_supervisor_receipt(directory: Path) -> dict[str, Any]:
    """Bind an external lifecycle attestation without changing task acceptance."""
    path = directory/"supervisor_receipt.json"
    if not path.is_file():
        return {"status": "not_recorded", "exit_status_used_as_acceptance": False}
    receipt = _json(path)
    errors = []
    if receipt.get("schema_version") == "scene_factory.trial_supervisor_receipt.v1":
        if type(receipt.get("supervisor_exit_code")) is not int:
            errors.append("receipt_exit_code_invalid")
        if receipt.get("cleanup") != "process_exited":
            errors.append("receipt_process_exit_not_attested")
        if receipt.get("report_sha256") != sha256(directory/"report.json"):
            errors.append("receipt_file_hash_mismatch:report.json")
        return {"status": "not_bound" if errors else "bound_legacy_report_attestation", "errors": errors,
                "sha256": sha256(path), "receipt": receipt,
                "scope": "Legacy external supervisor record binds the report only; observer provenance and trace hashes were not recorded in this receipt.",
                "exit_status_used_as_acceptance": False}
    if receipt.get("schema_version") != "scene_factory.assembly_trial_supervisor.v1":
        errors.append("unsupported_receipt_schema")
    if receipt.get("trial") != directory.name:
        errors.append("receipt_trial_mismatch")
    if type(receipt.get("process_exit_code")) is not int:
        errors.append("receipt_exit_code_invalid")
    if receipt.get("cleanup") != "process_exited":
        errors.append("receipt_process_exit_not_attested")
    if not isinstance(receipt.get("source"), str) or not receipt["source"]:
        errors.append("receipt_source_missing")
    bindings = receipt.get("file_sha256")
    if not isinstance(bindings, dict) or "report.json" not in bindings:
        errors.append("receipt_report_binding_missing")
        bindings = {} if not isinstance(bindings, dict) else bindings
    for name, digest in bindings.items():
        if name not in {"report.json", "trace.jsonl", "online_trace.jsonl", "runtime.log"}:
            errors.append("receipt_file_not_allowed")
            continue
        target = directory/name
        if not target.is_file() or digest != sha256(target):
            errors.append(f"receipt_file_hash_mismatch:{name}")
    return {"status": "not_bound" if errors else "bound_external_attestation", "errors": errors,
            "sha256": sha256(path), "receipt": receipt,
            "scope": "External supervisor attestation bound to saved bytes; auditor did not observe or rerun the process.",
            "exit_status_used_as_acceptance": False}


def audit_room_background(directory: Path, report: dict[str, Any], *, end_time_s: float | None) -> dict[str, Any]:
    """Recompute final-window motion checks; never trust a recorded passed flag."""
    thresholds = {"drift_m": .001, "displacement_m": .005, "speed_mps": .005, "angular_speed_rps": .05}
    layout_path = directory/"layout.json"
    layout = _json(layout_path) if layout_path.is_file() else {}
    initial = report.get("scene_scale", {}).get("background_initial", {})
    declared = {item["object_id"] for item in layout.get("objects", [])
                if item.get("dynamic") is True and item.get("object_id") != "peg_1"}
    expected = declared | set(initial)
    base = {"required": bool(expected), "thresholds": thresholds,
            "scope": "Independent saved final-1s background position/live-velocity audit; no stability claim for the entire preceding episode.",
            "recorded_result": report.get("background_stability"), "object_ids": sorted(expected)}
    if not expected:
        return {**base, "status": "not_applicable", "reason": "no dynamic background declared in saved layout or initial measurements"}
    path = directory/"background_trace.json"
    if not path.is_file():
        return {**base, "status": "not_verified", "errors": ["background_trace_missing"]}
    rows = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if not isinstance(rows, list) or len(rows) < 2:
        return {**base, "status": "not_verified", "errors": ["background_window_missing"], "sha256": sha256(path)}
    if declared != set(initial):
        errors.append("declared_and_initial_background_sets_differ")
    dt = report.get("parameters", {}).get("physics_dt_s", 1/120)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        errors.append("physics_dt_missing_or_invalid")
    previous = None
    drift = displacement = speed = spin = 0.
    valid_times = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"background_row_invalid:{index}")
            continue
        time_s = row.get("time_s")
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s):
            errors.append(f"background_time_invalid:{index}")
            continue
        valid_times.append(time_s)
        if previous is not None and isinstance(dt, (int, float)) and not math.isclose(time_s-previous, dt, rel_tol=1e-6, abs_tol=1e-8):
            errors.append(f"background_physics_step_gap:{index}")
        previous = time_s
        objects = row.get("objects", {})
        if not isinstance(objects, dict):
            errors.append(f"background_objects_invalid:{index}")
            continue
        if set(objects) != expected:
            errors.append(f"background_object_coverage_incomplete:{index}")
        for object_id in expected:
            state = objects.get(object_id, {})
            position, linear, angular = (_velocity(state.get(name)) for name in ("position", "linear_velocity_mps", "angular_velocity_rps"))
            first = _velocity(rows[0].get("objects", {}).get(object_id, {}).get("position"))
            start = _velocity(initial.get(object_id, {}).get("position"))
            if any(value is None for value in (position, linear, angular, first, start)) or state.get("physics_handle_valid") is not True:
                errors.append(f"background_live_state_incomplete:{index}:{object_id}")
                continue
            drift = max(drift, math.dist(first, position))
            displacement = max(displacement, math.dist(start, position))
            speed = max(speed, math.sqrt(sum(value*value for value in linear)))
            spin = max(spin, math.sqrt(sum(value*value for value in angular)))
    window = valid_times[-1]-valid_times[0] if len(valid_times) >= 2 else 0.
    if window < 1.-1e-8:
        errors.append("background_stability_window_shorter_than_1s")
    if end_time_s is None or not valid_times or not math.isclose(valid_times[-1], end_time_s, rel_tol=0., abs_tol=1e-8):
        errors.append("background_window_not_aligned_with_terminal_assembly_measurement")
    recorded_thresholds = report.get("background_stability", {}).get("thresholds")
    if recorded_thresholds != thresholds:
        errors.append("background_thresholds_missing_or_differ_from_frozen_protocol")
    metrics = {"window_s": window, "max_position_drift_m": drift, "max_total_displacement_m": displacement,
               "max_speed_mps": speed, "max_angular_speed_rps": spin}
    failed = drift > thresholds["drift_m"] or displacement > thresholds["displacement_m"] or speed > thresholds["speed_mps"] or spin > thresholds["angular_speed_rps"]
    return {**base, "status": "failed" if failed else ("not_verified" if errors else "passed"),
            "metrics": metrics, "sample_count": len(rows), "error_count": len(errors), "errors_first20": errors[:20],
            "sha256": sha256(path), "reported_passed_flag_used_as_acceptance": False}


def audit_trial(directory: Path, *, fixture_mappings: dict[str, Path] | None = None) -> dict[str, Any]:
    directory = directory.resolve()
    report_path, spec_path = directory/"report.json", directory/"spec.json"
    report = _json(report_path) if report_path.is_file() else {}
    result: dict[str, Any] = {"trial": directory.name, "directory": str(directory),
                              "historical": {"recorded_acceptance": report.get("assembly_acceptance"),
                                             "recorded_online_acceptance": report.get("online_acceptance"),
                                             "recorded_pose_result": report.get("assembly_evaluation"),
                                             "recorded_process_status": report.get("status"),
                                             "supervisor": report.get("supervisor"), "transport": report.get("transport"),
                                             "cleanup": report.get("cleanup"), "exit_status_used_as_acceptance": False},
                              "hashes": {}, "traces": [], "current_audited_acceptance": "not_verified", "audit_reasons": []}
    result["supervisor_receipt_audit"] = audit_supervisor_receipt(directory)
    for filename in ("report.json", "spec.json", "layout.json", "controller_source.json", "supervisor_receipt.json", "background_trace.json"):
        path = directory/filename
        if path.is_file():
            result["hashes"][filename] = sha256(path)
    if not spec_path.is_file():
        result["audit_reasons"].append("frozen_trial_spec_missing")
        return result
    spec = AssemblySpec.from_dict(_json(spec_path))
    binding = _spec_binding(spec_path, report, spec, fixture_mappings)
    spec_match = binding["status"] == "passed"
    result["spec_binding"] = binding
    result["recorded_evaluator_sha256"] = report.get("evaluator_sha256")
    result["recorded_controller_sha256"] = report.get("controller_sha256")
    controller_path = directory/"controller_source.json"
    if controller_path.is_file():
        source = _json(controller_path).get("python_source")
        if isinstance(source, str):
            result["controller_source_text_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            result["controller_source_text_matches_recorded_byte_hash"] = result["controller_source_text_sha256"] == report.get("controller_sha256")
            result["controller_hash_scope"] = "recorded UTF-8 source text; newline normalization may differ from original file bytes"
    for filename in ("trace.jsonl", "online_trace.jsonl"):
        path = directory/filename
        if not path.is_file():
            continue
        size_before = path.stat().st_size
        content = path.read_bytes()
        capture_stable = size_before == len(content) == path.stat().st_size
        result["hashes"][filename] = hashlib.sha256(content).hexdigest()
        try:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            pose_result = evaluate_assembly_trace(spec, rows)
            live = audit_live_velocity(rows, spec)
            coverage = audit_coverage(rows, spec, report)
            release = audit_release(rows, spec, report)
            geometry = _geometry(directory, report, rows, spec, fixture_mappings)
            known_failure = pose_result.get("status") == "failed" or live["status"] == "failed"
            complete = (capture_stable and spec_match and pose_result.get("status") == "passed" and live["status"] == "passed"
                        and coverage["status"] == "passed" and release["status"] == "passed" and geometry["status"] == "passed")
            acceptance = "failed" if known_failure else ("passed" if complete else "not_verified")
            result["traces"].append({"file": filename, "sample_count": len(rows), "capture_stable": capture_stable,
                                      "last_time_s": rows[-1].get("time_s") if rows else None,
                                      "pose_trace_recomputed": pose_result, "live_velocity_audit": live,
                                      "contact_coverage_audit": coverage, "release_audit": release,
                                      "geometry_audit": geometry, "current_audited_acceptance": acceptance})
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            result["traces"].append({"file": filename, "current_audited_acceptance": "not_verified",
                                      "error": f"invalid_saved_evidence:{type(exc).__name__}:{exc}"})
    preferred = next((row for row in result["traces"] if row["file"] == "online_trace.jsonl"), result["traces"][0] if result["traces"] else None)
    if preferred:
        result["preferred_trace"] = preferred["file"]
        result["current_audited_acceptance"] = preferred["current_audited_acceptance"]
        result["local_assembly_audited_acceptance"] = preferred["current_audited_acceptance"]
        background = audit_room_background(directory, report, end_time_s=preferred.get("last_time_s"))
        result["background_audit"] = background
        if background["required"]:
            if background["status"] == "failed":
                result["current_audited_acceptance"] = "failed"
            elif background["status"] != "passed" and result["current_audited_acceptance"] == "passed":
                result["current_audited_acceptance"] = "not_verified"
    else:
        result["audit_reasons"].append("saved_trace_missing")
    if not spec_match:
        result["audit_reasons"].append("spec_hash_not_bound_to_recorded_trial")
    result["historical_acceptance_changed"] = result["historical"]["recorded_acceptance"] != result["current_audited_acceptance"]
    return result


def audit_directories(directories: list[Path], *, fixture_mappings: dict[str, Path] | None = None) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[1]
    trials = []
    for directory in sorted({path.resolve() for path in directories}):
        try:
            trials.append(audit_trial(directory, fixture_mappings=fixture_mappings))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            trials.append({"trial": directory.name, "directory": str(directory), "current_audited_acceptance": "not_verified",
                           "audit_reasons": [f"cannot_audit:{type(exc).__name__}:{exc}"]})
    return {"schema_version": SCHEMA, "status": "audit_completed", "trial_count": len(trials),
            "current_passed_count": sum(row["current_audited_acceptance"] == "passed" for row in trials),
            "current_failed_count": sum(row["current_audited_acceptance"] == "failed" for row in trials),
            "current_not_verified_count": sum(row["current_audited_acceptance"] == "not_verified" for row in trials),
            "audit_sources": {path.relative_to(project).as_posix(): sha256(path) for path in (Path(__file__).resolve(), project/"scene_factory/assembly.py", project/"scene_factory/assembly_runtime.py")},
            "scope": "saved-measurement audit using each trial frozen spec and current evaluator; no physical rerun or independent sensor certification",
            "exit_code_is_not_task_acceptance": True, "original_reports_modified": False, "trials": trials}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path("outputs/completion_20261104"))
    parser.add_argument("--pattern", action="append")
    parser.add_argument("--fixture-map", action="append", default=[], metavar="ORIGINAL=EXTRACTED",
                        help="Map a recorded fixture directory/prefix to hash-identical extracted public files; does not bypass hashes.")
    parser.add_argument("--fixture-map-file", type=Path, help="JSON scene_factory.assembly_fixture_paths.v1 mapping; relative targets resolve against the source root.")
    parser.add_argument("--output", type=Path, default=Path("outputs/completion_20261104/assembly_audit.json"))
    args = parser.parse_args()
    directories = args.directories or [path for pattern in (args.pattern or DEFAULT_PATTERNS) for path in args.root.glob(pattern) if path.is_dir()]
    mappings = {}
    if args.fixture_map_file:
        mapping_file = _json(args.fixture_map_file)
        if mapping_file.get("schema_version") != "scene_factory.assembly_fixture_paths.v1" or not isinstance(mapping_file.get("mappings"), dict):
            parser.error("unsupported --fixture-map-file")
        for original, target in mapping_file["mappings"].items():
            if not isinstance(original, str) or not isinstance(target, str) or not original or not target:
                parser.error("fixture mapping keys/values must be nonempty strings")
            value = Path(target)
            mappings[original] = value if value.is_absolute() else Path(__file__).resolve().parents[1]/value
    for value in args.fixture_map:
        original, delimiter, target = value.partition("=")
        if not delimiter or not original or not target:
            parser.error("--fixture-map requires ORIGINAL=EXTRACTED")
        mappings[original] = Path(target)
    result = audit_directories(directories, fixture_mappings=mappings)
    write_json_atomic(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"trials", "audit_sources"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
