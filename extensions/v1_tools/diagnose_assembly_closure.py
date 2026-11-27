"""Summarize saved closure dynamics; no simulator, velocity filter, or acceptance change."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_factory.dataset import write_json_atomic


def magnitude(vector):
    return math.sqrt(sum(value*value for value in vector))


def summarize(directory: Path) -> dict:
    trace_path = directory/"trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    closing = [(index, row) for index, row in enumerate(rows) if row.get("phase") == "close_gripper"]
    if not closing:
        raise ValueError("recorded close_gripper phase missing")
    peak_index, peak = max(closing, key=lambda item: magnitude(item[1]["live_peg"]["angular_velocity_rps"]))
    dt = closing[-1][1]["time_s"]-closing[-2][1]["time_s"]
    tail = [(index, row) for index, row in closing if peak["time_s"]-.5-1e-9 <= row["time_s"] <= peak["time_s"]+1e-9]
    bilateral = [(index, row) for index, row in closing
                 if sum("panda_leftfinger" in contact["collider"] and contact["normal_force_N"] > .05
                        for contact in row.get("shape_contacts", []))
                 and sum("panda_rightfinger" in contact["collider"] and contact["normal_force_N"] > .05
                         for contact in row.get("shape_contacts", []))]
    contact_window = [(index, row) for index, row in closing
                      if bilateral and bilateral[0][1]["time_s"] <= row["time_s"] <= peak["time_s"]]
    first_arm = closing[0][1]["action"]["payload"]["arm_positions_rad"]

    def measured(row):
        return {"sample_index": row[0], "time_s": row[1]["time_s"],
                "angular_velocity_rps": row[1]["live_peg"]["angular_velocity_rps"],
                "angular_speed_rps": magnitude(row[1]["live_peg"]["angular_velocity_rps"]),
                "linear_speed_mps": magnitude(row[1]["live_peg"]["linear_velocity_mps"]),
                "gripper_open_fraction": row[1]["action"]["payload"]["gripper_open_fraction"],
                "finger_positions_m": row[1]["finger_positions_m"],
                "shape_contacts": row[1].get("shape_contacts", [])}

    thresholds = {}
    for threshold in (.5, 1., 1.5, 2.):
        over = [item for item in closing if magnitude(item[1]["live_peg"]["angular_velocity_rps"]) > threshold]
        thresholds[str(threshold)] = {"count": len(over), "first": measured(over[0]) if over else None}
    forces = {}
    for _, row in tail:
        for contact in row.get("shape_contacts", []):
            forces.setdefault(contact["collider"], []).append(contact["normal_force_N"])
    for collider, values in forces.items():
        forces[collider] = {"present_sample_count": len(values), "total_tail_samples": len(tail),
                           "minimum_when_present_N": min(values), "maximum_N": max(values)}
    fingers = [[row["finger_positions_m"][finger] for _, row in tail] for finger in range(2)]
    end_fraction = tail[-1][1]["action"]["payload"]["gripper_open_fraction"]
    start_fraction = tail[0][1]["action"]["payload"]["gripper_open_fraction"]
    # Both trial reports bind the standard 0..0.04 m runtime finger limits.
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    params = report.get("parameters", {})
    omega = peak["live_peg"]["angular_velocity_rps"]
    return {"schema_version": "scene_factory.assembly_closure_diagnosis.v1", "trial": directory.name,
            "scope": "Read-only statistics over saved unfiltered PhysX velocities and commanded drive targets; no physical rerun or proof of a unique root cause.",
            "files_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                             for name in ("report.json", "trace.jsonl", "controller_source.json")},
            "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "physics_dt_s_from_trace": dt, "parameters": params,
            "closure": {"sample_count": len(closing), "start_time_s": closing[0][1]["time_s"],
                        "end_time_s": closing[-1][1]["time_s"],
                        "max_arm_setpoint_change_rad": max(abs(a-b) for _, row in closing
                                                            for a, b in zip(row["action"]["payload"]["arm_positions_rad"], first_arm, strict=True)),
                        "peak": measured((peak_index, peak)), "peak_axis_z_squared_fraction": omega[2]**2/sum(v*v for v in omega),
                        "thresholds_rps": thresholds},
            "last_half_second": {"sample_count": len(tail), "duration_s": tail[-1][1]["time_s"]-tail[0][1]["time_s"],
                                 "actual_finger_range_m": [max(values)-min(values) for values in fingers],
                                 "command_fraction_start": start_fraction, "command_fraction_end": end_fraction,
                                 "commanded_finger_inward_travel_m_using_0_04_range": .04*(start_fraction-end_fraction),
                                 "contact_force_ranges": forces},
            "first_bilateral_contact_to_peak": {
                "first": measured(bilateral[0]) if bilateral else None,
                "sample_count": len(contact_window),
                "actual_finger_range_m": [max(row["finger_positions_m"][finger] for _, row in contact_window)
                                          - min(row["finger_positions_m"][finger] for _, row in contact_window)
                                          for finger in range(2)] if contact_window else [],
                "commanded_finger_inward_travel_m_using_0_04_range": .04*(
                    contact_window[0][1]["action"]["payload"]["gripper_open_fraction"]
                    - contact_window[-1][1]["action"]["payload"]["gripper_open_fraction"]) if contact_window else None},
            "notes": ["Axis-z velocity is part of the frozen full angular-speed norm; cylinder symmetry does not exempt it.",
                      "Zero arm setpoint change does not prove zero physical arm motion.",
                      "A large live-velocity spike with a small pose delta remains a failed measurement under the frozen spec.",
                      "Force variation and continued pressure at almost fixed finger aperture suggest a constrained-contact closure hypothesis, not a proven sole cause."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"trials": [summarize(directory) for directory in args.directories]}
    write_json_atomic(args.output, result)
    print(json.dumps({"output": str(args.output), "trials": [row["trial"] for row in result["trials"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
