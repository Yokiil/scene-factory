"""Verify hashes and compare saved four-arm trajectories independently of compilation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.dataset import write_json_atomic
from scene_factory.incremental_evidence import compare_contacts, validate_contacts, verify_inputs


def verify(directory: Path) -> dict:
    import numpy as np

    directory = directory.resolve()
    report = json.loads((directory / "benchmark.json").read_text(encoding="utf-8"))
    variants = json.loads((directory / "variants.json").read_text(encoding="utf-8"))
    errors, checked, differences = [], 0, []
    contact_differences = []
    version2 = report.get("schema_version") == "scene_factory.incremental_four_arm_benchmark.v2"
    input_check = None
    if version2:
        input_check = verify_inputs(directory / "inputs", report["input_lock"]["sha256"])
        errors.extend(input_check["errors"])
        for descriptor in report["scene_inputs"].values():
            path = (directory/descriptor["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("scene input escapes benchmark")
            if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
                errors.append(f"scene_input_hash:{descriptor['path']}")
    loaded = {}
    if set(report["raw"]) != {"full", "file_cache", "incremental", "partition"}:
        errors.append("four_arm_set_incomplete")
    for mode, rows in report["raw"].items():
        if len(rows) != len(variants):
            errors.append(f"variant_count:{mode}")
        loaded[mode] = []
        for index, row in enumerate(rows):
            values = []
            for field in ("trajectory", "contact_trace"):
                descriptor = row[field]
                path = (directory / descriptor["path"]).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("evidence path escapes benchmark")
                if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
                    errors.append(f"hash:{mode}:{index}:{field}")
                values.append(path)
            trajectory = np.load(values[0], allow_pickle=False)
            contacts = json.loads(values[1].read_text(encoding="utf-8"))
            if list(trajectory.shape) != row["trajectory"]["shape"] or not np.isfinite(trajectory).all():
                errors.append(f"shape_or_finite:{mode}:{index}")
            scene = variants[index]
            expected_x = scene["objects"][1]["pose"]["position"][0] - .05 - scene["objects"][0]["bbox_m"][0]/2
            if abs(float(trajectory[-1, 0])-expected_x) >= report["validation"]["target_x_error_m"]:
                errors.append(f"final_position:{mode}:{index}")
            if version2:
                try:
                    validate_contacts(contacts, expected_steps=row["physics_steps"], dt_s=report["physics"]["dt_s"])
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(f"contact_trace_invalid:{mode}:{index}:{exc}")
                contact_seen = any(set(point["body_pair"]) == {"object__part", "object__stop"} for step in contacts["steps"] for point in step["contacts"])
                if row["compiler_source_sha256"] != input_check["compiler_source_sha256"]:
                    errors.append(f"compiler_input_lock:{mode}:{index}")
            else:
                contact_seen = any([1, 2] in timestep for timestep in contacts)
            if not contact_seen:
                errors.append(f"stop_contact:{mode}:{index}")
            max_speed = float(np.max(np.linalg.norm(trajectory[:, 7:10], axis=1)))
            if max_speed > report["validation"]["max_linear_speed_mps"]:
                errors.append(f"speed_bound:{mode}:{index}")
            loaded[mode].append((trajectory, contacts))
            checked += 1
    for mode, rows in loaded.items():
        for index, (trajectory, contacts) in enumerate(rows):
            difference = float(np.max(np.abs(trajectory-loaded["full"][index][0])))
            differences.append(difference)
            if version2:
                comparison = compare_contacts(loaded["full"][index][1], contacts)
                contact_differences.append(comparison)
                contact_error = (not comparison["contact_pairs_identical"]
                                 or comparison["max_normal_force_difference_N"] > report["validation"]["normal_force_atol_N"]
                                 or comparison["max_pair_normal_sum_difference_N"] > report["validation"]["normal_force_atol_N"]
                                 or comparison["max_wrench_component_difference"] > report["validation"]["wrench_component_atol"]
                                 or comparison["max_contact_position_difference_m"] > report["validation"]["contact_position_atol_m"])
            else:
                contact_error = contacts != loaded["full"][index][1]
            if difference > report["validation"]["state_atol"] or contact_error:
                errors.append(f"baseline_difference:{mode}:{index}")
    return {"schema_version": "scene_factory.incremental_benchmark_verification.v2" if version2 else "scene_factory.incremental_benchmark_verification.v1",
            "status": "passed" if not errors else "failed", "errors": errors,
            "checked_trajectories": checked, "max_state_difference_vs_full": max(differences),
            "frozen_inputs": input_check,
            "stepwise_contact_forces_verified": version2,
            "compared_physics_steps": sum(row["compared_steps"] for row in contact_differences),
            "compared_contact_points": sum(row["compared_contact_points"] for row in contact_differences),
            "max_normal_force_difference_N": max((row["max_normal_force_difference_N"] for row in contact_differences), default=None),
            "max_pair_normal_sum_difference_N": max((row["max_pair_normal_sum_difference_N"] for row in contact_differences), default=None),
            "max_wrench_component_difference": max((row["max_wrench_component_difference"] for row in contact_differences), default=None),
            "max_contact_position_difference_m": max((row["max_contact_position_difference_m"] for row in contact_differences), default=None),
            "scope": "saved_contact_proxy_trajectories_not_robot_assembly",
            "benchmark_sha256": hashlib.sha256((directory / "benchmark.json").read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
