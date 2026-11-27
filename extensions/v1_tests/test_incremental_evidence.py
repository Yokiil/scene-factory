from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scene_factory.incremental_evidence import CONTACT_SCHEMA, compare_contacts, freeze_inputs, validate_contacts, verify_inputs
from scene_factory.paths import default_registry_path
from scene_factory.dataset import write_json_atomic


def contact_trace():
    return {"schema_version": CONTACT_SCHEMA, "steps": [
        {"step_index": 1, "sim_time_s": .002, "contact_count": 2, "contacts": [
            {"geom_pair": ["floor", "part"], "normal_force_N": 12., "position_m": [0, 0, 0], "wrench_contact_frame_N_Nm": [12., 0, 0, 0, 0, 0]},
            {"geom_pair": ["floor", "part"], "normal_force_N": 8., "position_m": [0, 1, 0], "wrench_contact_frame_N_Nm": [8., 0, 0, 0, 0, 0]},
        ]},
        {"step_index": 2, "sim_time_s": .004, "contact_count": 0, "contacts": []},
    ]}


def test_frozen_registry_consumed_with_raw_rows_and_geometry_closure(tmp_path):
    registry, manifest = freeze_inputs(default_registry_path(), {"remote_black", "kd_box_01"}, tmp_path/"inputs")
    assert registry.registry_path == (tmp_path/"inputs/registry.jsonl").resolve()
    assert (registry.base_dir/"source/kd_box_01/geometry.json").is_file()
    assert manifest["original_registry_sha256"] == hashlib.sha256(default_registry_path().read_bytes()).hexdigest()
    original_rows = {json.loads(row)["asset_id"]: row for row in default_registry_path().read_text(encoding="utf-8").splitlines() if row.strip() and not row.startswith("#")}
    selected_rows = (registry.base_dir/"registry.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(original_rows[json.loads(row)["asset_id"]] == row for row in selected_rows)
    lock_sha = hashlib.sha256((registry.base_dir/"manifest.json").read_bytes()).hexdigest()
    assert verify_inputs(registry.base_dir, lock_sha)["status"] == "passed"
    path = registry.base_dir/"source/kd_box_01/geometry.json"
    path.write_bytes(path.read_bytes()+b" ")
    assert "input_hash:source/kd_box_01/geometry.json" in verify_inputs(registry.base_dir, lock_sha)["errors"]


def test_complete_step_contact_trace_compares_per_point_and_sum_not_only_peak():
    reference = contact_trace()
    candidate = copy.deepcopy(reference)
    candidate["steps"][0]["contacts"][1]["normal_force_N"] = 7
    candidate["steps"][0]["contacts"][1]["wrench_contact_frame_N_Nm"][0] = 7
    comparison = compare_contacts(reference, candidate)
    assert comparison["contact_pairs_identical"]
    assert comparison["max_normal_force_difference_N"] == 1
    assert comparison["max_pair_normal_sum_difference_N"] == 1
    assert comparison["max_wrench_component_difference"] == 1
    assert comparison["compared_steps"] == 2 and comparison["compared_contact_points"] == 2
    assert max(point["normal_force_N"] for point in reference["steps"][0]["contacts"]) == max(point["normal_force_N"] for point in candidate["steps"][0]["contacts"])


@pytest.mark.parametrize("fault", ["missing_step", "missing_point", "negative_force", "wrong_window", "wrong_sum"])
def test_incomplete_or_invalid_contact_measurements_rejected(fault):
    trace = contact_trace()
    if fault == "missing_step":
        del trace["steps"][1]
    elif fault == "missing_point":
        trace["steps"][0]["contacts"].pop()
    elif fault == "negative_force":
        trace["steps"][0]["contacts"][0]["normal_force_N"] = -1
    elif fault == "wrong_window":
        trace["steps"][1]["sim_time_s"] = .1
    else:
        trace["steps"][0]["contacts"][0]["wrench_contact_frame_N_Nm"][0] = 1
    with pytest.raises(ValueError):
        validate_contacts(trace, expected_steps=2, dt_s=.002)


def test_absent_geom_contact_cannot_compare_as_zero_error():
    trace = contact_trace()
    candidate = copy.deepcopy(trace)
    candidate["steps"][0]["contacts"].pop()
    comparison = compare_contacts(trace, candidate)
    assert not comparison["contact_pairs_identical"] and comparison["mismatched_steps"] == [0]


def test_independent_verifier_catches_nonpeak_force_change_even_with_updated_file_checksum(tmp_path):
    np = pytest.importorskip("numpy")
    from tools.verify_incremental_benchmark import verify

    _, manifest = freeze_inputs(default_registry_path(), {"remote_black", "kd_box_01"}, tmp_path/"inputs")
    scene = {"objects": [{"bbox_m": [.08, .08, .08]}, {"pose": {"position": [.16, 0, .1]}}]}
    write_json_atomic(tmp_path/"variants.json", [scene])
    write_json_atomic(tmp_path/"base_scene.json", scene)
    report = {"schema_version": "scene_factory.incremental_four_arm_benchmark.v2", "raw": {},
              "input_lock": {"sha256": hashlib.sha256((tmp_path/"inputs/manifest.json").read_bytes()).hexdigest()},
              "scene_inputs": {name: {"path": name, "sha256": hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()} for name in ("variants.json", "base_scene.json")},
              "physics": {"dt_s": .002},
              "validation": {"state_atol": 1e-10, "normal_force_atol_N": 1e-8, "wrench_component_atol": 1e-8,
                             "contact_position_atol_m": 1e-10, "target_x_error_m": .003, "max_linear_speed_mps": 2}}
    for mode in ("full", "file_cache", "incremental", "partition"):
        folder = tmp_path/mode
        folder.mkdir()
        state = np.zeros((2, 13))
        state[:, 0] = .07
        np.save(folder/"trajectory.npy", state, allow_pickle=False)
        trace = contact_trace()
        for step in trace["steps"]:
            for point in step["contacts"]:
                point["body_pair"] = ["object__part", "object__stop"]
        if mode == "incremental":
            trace["steps"][0]["contacts"][1]["normal_force_N"] = 7
            trace["steps"][0]["contacts"][1]["wrench_contact_frame_N_Nm"][0] = 7
        write_json_atomic(folder/"contacts.json", trace)
        report["raw"][mode] = [{"physics_steps": 2, "compiler_source_sha256": manifest["compiler_source_sha256"],
                                "trajectory": {"path": f"{mode}/trajectory.npy", "sha256": hashlib.sha256((folder/"trajectory.npy").read_bytes()).hexdigest(), "shape": [2, 13]},
                                "contact_trace": {"path": f"{mode}/contacts.json", "sha256": hashlib.sha256((folder/"contacts.json").read_bytes()).hexdigest()}}]
    write_json_atomic(tmp_path/"benchmark.json", report)
    result = verify(tmp_path)
    assert result["status"] == "failed"
    assert result["errors"] == ["baseline_difference:incremental:0"]
    assert result["max_normal_force_difference_N"] == 1
    assert result["max_pair_normal_sum_difference_N"] == 1
