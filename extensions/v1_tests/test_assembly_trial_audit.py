from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scene_factory.assembly import ASSEMBLY_SCHEMA
from scene_factory.dataset import write_json_atomic
from tools.audit_assembly_trials import audit_directories, audit_trial


def make_trial(directory: Path, *, raw_angular_spike: bool = False) -> None:
    directory.mkdir()
    spec = {"schema_version": ASSEMBLY_SCHEMA, "spec_id": "audit-v1", "peg_id": "peg", "socket_id": "socket",
            "peg_interface_position_m": [0, 0, 0], "socket_interface_position_m": [0, 0, 0],
            "insertion_axis_socket": [0, 0, -1], "target_depth_m": .02, "clearance_m": [.0001, .001],
            "position_tolerance_m": .001, "angular_tolerance_rad": .05, "max_force_N": 20,
            "min_hold_s": .5, "release_stable_s": .5, "timeout_s": 5, "jam_window_s": .5,
            "min_progress_m": .001, "max_linear_speed_mps": .5, "max_angular_speed_rps": 4,
            "drop_below_z_m": -.1, "allowed_contact_pairs": [["peg", "socket"], ["robot", "peg"]]}
    write_json_atomic(directory/"spec.json", spec)
    fixture = directory/"fixture"
    fixture.mkdir()
    (fixture/"scene.usda").write_text("synthetic hash-bound geometry placeholder for auditor test", encoding="utf-8")
    write_json_atomic(fixture/"SOURCE.json", {"geometry": {"clearance_m": .0005},
                                            "usd_sha256": hashlib.sha256((fixture/"scene.usda").read_bytes()).hexdigest()})
    rows = []
    for index in range(21):
        rows.append({"time_s": index/10, "geometry_clearance_m": .0005,
                     "parts": {"peg": {"position_m": [0, 0, -min(index*.002, .02)], "orientation_wxyz": [1, 0, 0, 0],
                                       "linear_velocity_mps": [0, 0, -.02] if index <= 10 else [0, 0, 0],
                                       "angular_velocity_rps": [0, 0, 5 if raw_angular_spike and index == 5 else 0]},
                               "socket": {"position_m": [0, 0, 0], "orientation_wxyz": [1, 0, 0, 0],
                                          "linear_velocity_mps": [0, 0, 0], "angular_velocity_rps": [0, 0, 0]}},
                     "contacts": [{"pair": ["peg", "socket"], "normal_force_N": 3}],
                     "released": index >= 12, "action": {"mode": "joint_position", "payload": {"targets": [0]}},
                     "motion_source": "robot_joint_control", "teleported": False, "kinematic_pose_write": False,
                     "contact_coverage": {"complete": True, "continuous": True, "scope": "all_scene_contacts",
                                          "object_ids": ["peg", "socket"], "force_aggregation": "sum_per_point_normal_magnitudes",
                                          "sample_dt_s": .1, "source": "synthetic test measurement", "buffer_saturated": False}})
    (directory/"online_trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    report = {"assembly_acceptance": "passed", "status": "trial_completed", "supervisor": {"exit_code": 0},
              "spec_sha256": hashlib.sha256((directory/"spec.json").read_bytes()).hexdigest(),
              "parameters": {"fixture": str(fixture.resolve())}, "fixture_sha256": hashlib.sha256((fixture/"scene.usda").read_bytes()).hexdigest(),
              "online_state": {"release_evidence": {"finger_contact_before_open": True, "actual_gripper_open": True,
                                                    "current_robot_peg_contact": False, "released": True, "source": "synthetic test measured DOFs"}}}
    write_json_atomic(directory/"report.json", report)


def test_pose_pass_and_zero_exit_do_not_hide_live_velocity_failure(tmp_path):
    directory = tmp_path/"raw_failure"
    make_trial(directory, raw_angular_spike=True)
    original = (directory/"report.json").read_bytes()
    result = audit_trial(directory)
    assert result["historical"]["recorded_acceptance"] == "passed"
    assert result["historical"]["supervisor"]["exit_code"] == 0
    trace = result["traces"][0]
    assert trace["pose_trace_recomputed"]["status"] == "passed"
    assert trace["live_velocity_audit"]["status"] == "failed"
    assert trace["live_velocity_audit"]["violation_count"] == 1
    assert result["current_audited_acceptance"] == "failed"
    assert (directory/"report.json").read_bytes() == original


def test_process_exit_without_evidence_never_counts_as_acceptance(tmp_path):
    directory = tmp_path/"only_exit"
    directory.mkdir()
    write_json_atomic(directory/"report.json", {"status": "passed", "assembly_acceptance": "passed", "supervisor": {"exit_code": 0}})
    result = audit_directories([directory])
    assert result["status"] == "audit_completed"
    assert result["current_passed_count"] == 0
    assert result["trials"][0]["current_audited_acceptance"] == "not_verified"
    assert result["exit_code_is_not_task_acceptance"]


def test_complete_saved_online_measurements_pass_with_explicit_scope(tmp_path):
    directory = tmp_path/"complete"
    make_trial(directory)
    result = audit_trial(directory)
    assert result["current_audited_acceptance"] == "passed"
    trace = result["traces"][0]
    assert trace["contact_coverage_audit"]["complete_task_part_coverage_each_physics_step"]
    assert trace["release_audit"]["actual_open_and_grasp_history_attested"]
    assert "no independent geometry" in trace["geometry_audit"]["scope"]


def test_offline_partial_contacts_and_commanded_release_do_not_become_full_acceptance(tmp_path):
    directory = tmp_path/"offline"
    make_trial(directory)
    path = directory/"online_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        del row["contact_coverage"]
    (directory/"trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    path.unlink()
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    report.pop("online_state")
    report["contact_coverage"] = {"scope": "all enabled authored collision shapes against peg"}
    write_json_atomic(directory/"report.json", report)
    result = audit_trial(directory)
    assert result["traces"][0]["pose_trace_recomputed"]["status"] == "passed"
    assert result["traces"][0]["contact_coverage_audit"]["historical_peg_only_filtered_contact_sensor"]
    assert result["current_audited_acceptance"] == "not_verified"


def test_spec_mutation_is_not_silently_accepted(tmp_path):
    directory = tmp_path/"changed_spec"
    make_trial(directory)
    path = directory/"spec.json"
    path.write_bytes(path.read_bytes()+b" ")
    result = audit_trial(directory)
    assert result["spec_binding"]["status"] == "not_verified"
    assert result["current_audited_acceptance"] == "not_verified"


def test_reformatted_copy_binds_through_original_source_hash_and_equal_thresholds(tmp_path):
    directory = tmp_path/"reformatted_spec"
    make_trial(directory)
    spec_path = directory/"spec.json"
    original = directory/"fixture/assembly_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    original.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    report["spec_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    write_json_atomic(directory/"report.json", report)
    result = audit_trial(directory)
    assert not result["spec_binding"]["trial_copy_byte_hash_matches_recorded"]
    assert result["spec_binding"]["trial_copy_semantically_matches_hash_bound_original"]
    assert result["current_audited_acceptance"] == "passed"


def test_supervisor_receipt_binds_saved_bytes_without_overriding_raw_failure(tmp_path):
    directory = tmp_path/"supervised_failure"
    make_trial(directory, raw_angular_spike=True)
    original = (directory/"report.json").read_bytes()
    receipt = {"schema_version": "scene_factory.assembly_trial_supervisor.v1", "trial": directory.name,
               "process_exit_code": 0, "cleanup": "process_exited", "source": "test external process poll",
               "file_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                               for name in ("report.json", "online_trace.jsonl")}}
    write_json_atomic(directory/"supervisor_receipt.json", receipt)
    result = audit_trial(directory)
    assert result["supervisor_receipt_audit"]["status"] == "bound_external_attestation"
    assert result["supervisor_receipt_audit"]["receipt"]["cleanup"] == "process_exited"
    assert not result["supervisor_receipt_audit"]["exit_status_used_as_acceptance"]
    assert result["current_audited_acceptance"] == "failed"
    assert (directory/"report.json").read_bytes() == original


def test_supervisor_receipt_for_other_bytes_is_not_bound(tmp_path):
    directory = tmp_path/"receipt_mismatch"
    make_trial(directory)
    write_json_atomic(directory/"supervisor_receipt.json", {
        "schema_version": "scene_factory.assembly_trial_supervisor.v1", "trial": directory.name,
        "process_exit_code": 0, "cleanup": "process_exited", "source": "external poll",
        "file_sha256": {"report.json": "0"*64}})
    result = audit_trial(directory)
    assert result["supervisor_receipt_audit"]["status"] == "not_bound"
    assert result["supervisor_receipt_audit"]["errors"] == ["receipt_file_hash_mismatch:report.json"]


def test_legacy_root_report_receipt_is_bound_with_explicit_provenance_limit(tmp_path):
    directory = tmp_path/"root_receipt"
    make_trial(directory, raw_angular_spike=True)
    write_json_atomic(directory/"supervisor_receipt.json", {
        "schema_version": "scene_factory.trial_supervisor_receipt.v1",
        "report_sha256": hashlib.sha256((directory/"report.json").read_bytes()).hexdigest(),
        "supervisor_exit_code": 0, "cleanup": "process_exited"})
    result = audit_trial(directory)
    assert result["supervisor_receipt_audit"]["status"] == "bound_legacy_report_attestation"
    assert "provenance" in result["supervisor_receipt_audit"]["scope"]
    assert result["current_audited_acceptance"] == "failed"


def test_extracted_fixture_mapping_binds_variant_without_a_hardcoded_geometry_hash(tmp_path):
    directory = tmp_path/"mapped_variant"
    make_trial(directory)
    report_path = directory/"report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["parameters"]["fixture"] = "D:/old-machine/fixtures/square"
    write_json_atomic(report_path, report)
    source_path = directory/"fixture/SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(fixture_id="independent-square-variant", frozen_spec_sha256=report["spec_sha256"])
    source["geometry"]["peg_shape"] = "square_prism"
    write_json_atomic(source_path, source)
    assert audit_trial(directory)["current_audited_acceptance"] == "not_verified"
    result = audit_trial(directory, fixture_mappings={"D:/old-machine/fixtures": directory/"fixture/.."})
    assert result["current_audited_acceptance"] == "not_verified"  # wrong replacement still fails
    result = audit_trial(directory, fixture_mappings={"D:/old-machine/fixtures/square": directory/"fixture"})
    assert result["current_audited_acceptance"] == "passed"
    geometry = result["traces"][0]["geometry_audit"]
    assert geometry["peg_shape"] == "square_prism"
    assert geometry["source_manifest_usd_hash_matches"]
    (directory/"fixture/scene.usda").write_bytes(b"modified geometry")
    result = audit_trial(directory, fixture_mappings={"D:/old-machine/fixtures/square": directory/"fixture"})
    assert result["current_audited_acceptance"] == "not_verified"
    assert not result["traces"][0]["geometry_audit"]["fixture_hash_matches_recorded"]


def test_variant_manifest_cannot_claim_a_different_frozen_spec_or_usd(tmp_path):
    directory = tmp_path/"bad_manifest"
    make_trial(directory)
    source_path = directory/"fixture/SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(frozen_spec_sha256="0"*64)
    write_json_atomic(source_path, source)
    result = audit_trial(directory)
    assert result["current_audited_acceptance"] == "not_verified"
    assert not result["traces"][0]["geometry_audit"]["source_manifest_spec_hash_matches_when_present"]


def add_background(directory: Path, *, speed_spike: bool = False, incomplete_window: bool = False):
    report_path = directory/"report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = {"position": [1., 1., .1], "orientation_wxyz": [1., 0., 0., 0.],
             "linear_velocity_mps": [0., 0., 0.], "angular_velocity_rps": [0., 0., 0.], "physics_handle_valid": True}
    report["scene_scale"] = {"background_initial": {"background": state}}
    report["parameters"]["physics_dt_s"] = .1
    report["background_stability"] = {"passed": True, "thresholds": {"drift_m": .001, "displacement_m": .005,
                                                                    "speed_mps": .005, "angular_speed_rps": .05}}
    write_json_atomic(report_path, report)
    write_json_atomic(directory/"layout.json", {"objects": [{"object_id": "background", "dynamic": True}]})
    rows = [{"time_s": 1.+index/10, "objects": {"background": json.loads(json.dumps(state))}} for index in range(11)]
    if speed_spike:
        rows[4]["objects"]["background"]["linear_velocity_mps"] = [.006, 0., 0.]
    if incomplete_window:
        rows = rows[::2]
    write_json_atomic(directory/"background_trace.json", rows)


def test_room_needs_independent_background_pass_even_when_recorded_flag_and_assembly_pass(tmp_path):
    directory = tmp_path/"room_speed_failure"
    make_trial(directory)
    add_background(directory, speed_spike=True)
    result = audit_trial(directory)
    assert result["local_assembly_audited_acceptance"] == "passed"
    assert result["background_audit"]["recorded_result"]["passed"]
    assert result["background_audit"]["status"] == "failed"
    assert result["background_audit"]["metrics"]["max_speed_mps"] == .006
    assert result["current_audited_acceptance"] == "failed"


def test_complete_aligned_background_window_passes_and_sparse_or_missing_trace_cannot(tmp_path):
    directory = tmp_path/"room_complete"
    make_trial(directory)
    add_background(directory)
    result = audit_trial(directory)
    assert result["current_audited_acceptance"] == "passed"
    assert result["background_audit"]["status"] == "passed"
    add_background(directory, incomplete_window=True)
    result = audit_trial(directory)
    assert result["current_audited_acceptance"] == "not_verified"
    assert any("physics_step_gap" in reason for reason in result["background_audit"]["errors_first20"])
    (directory/"background_trace.json").unlink()
    result = audit_trial(directory)
    assert result["current_audited_acceptance"] == "not_verified"
    assert result["background_audit"]["errors"] == ["background_trace_missing"]


def test_cli_fixture_mapping_file_rechecks_saved_measurements(tmp_path):
    import subprocess
    import sys

    directory = tmp_path/"mapped_cli"
    make_trial(directory, raw_angular_spike=True)
    report_path = directory/"report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["parameters"]["fixture"] = "D:/original-host/square-fixture"
    write_json_atomic(report_path, report)
    mapping = tmp_path/"fixture_paths.json"
    write_json_atomic(mapping, {"schema_version": "scene_factory.assembly_fixture_paths.v1",
                                "mappings": {"D:/original-host/square-fixture": str(directory/"fixture")}})
    output = tmp_path/"audit.json"
    script = Path(__file__).resolve().parents[1]/"tools/audit_assembly_trials.py"
    subprocess.run([sys.executable, str(script), str(directory), "--fixture-map-file", str(mapping), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    audit = json.loads(output.read_text(encoding="utf-8"))
    trial = audit["trials"][0]
    assert trial["traces"][0]["geometry_audit"]["status"] == "passed"
    assert trial["current_audited_acceptance"] == "failed"
    assert audit["current_passed_count"] == 0
