"""Online, episode-scoped assembly evidence without simulator dependencies.

Adapters must ingest every physics step and attest complete contact coverage
for both task parts. Missing evidence taints the episode until reset; a final
pose cannot recover lost contact or action evidence. This module never writes
simulator state and does not infer force from image/pose observations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any
import uuid

from .assembly import AssemblySpec, evaluate_assembly_trace


RUNTIME_SCHEMA = "scene_factory.assembly_runtime_status.v1"
CONTACT_SCOPE = "all_contacts_involving_task_parts"
FORCE_AGGREGATION = "sum_per_point_normal_magnitudes"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [_number(item, name) for item in value]


class AssemblyRuntimeEvaluator:
    """Accumulate measured SE(3), velocity, drive and contact evidence.

    Public adapter API: ``reset(episode_id)`` once, then ``record(...)`` once
    per physics step. Read-only observations use ``status()``. ``finalize()``
    marks an incomplete prefix as an actual failed/unknown episode outcome.
    Success reward is emitted once, after the frozen hold/release criteria pass.
    """

    def __init__(self, spec: AssemblySpec | dict[str, Any], *, physics_dt_s: float | None = None,
                 max_samples: int = 50000, evaluation_interval_s: float = 0.1):
        self.spec = AssemblySpec.from_dict(spec) if isinstance(spec, dict) else spec
        if not isinstance(self.spec, AssemblySpec):
            raise TypeError("spec must be AssemblySpec or its versioned mapping")
        self.spec_sha256 = hashlib.sha256(_canonical(self.spec.to_dict())).hexdigest()
        if physics_dt_s is not None and _number(physics_dt_s, "physics_dt_s") <= 0:
            raise ValueError("physics_dt_s must be positive")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError("max_samples must be an integer >=2")
        if _number(evaluation_interval_s, "evaluation_interval_s") < 0:
            raise ValueError("evaluation_interval_s must be nonnegative")
        self.physics_dt_s, self.max_samples = physics_dt_s, max_samples
        self.evaluation_interval_s = evaluation_interval_s
        self.episode_id: str | None = None
        self.samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._last_payload_sha256: str | None = None
        self._last_sim_time: float | None = None
        self._last_status: dict[str, Any] | None = None
        self._reward_emitted = False
        self._finalized = False
        self._last_full_evaluation_time: float | None = None
        self._immediate_failure: dict[str, Any] | None = None

    def reset(self, episode_id: str | None = None) -> dict[str, Any]:
        if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
            raise ValueError("episode_id must be a nonempty string")
        self.episode_id = episode_id or uuid.uuid4().hex
        self.samples = []
        self._errors = []
        self._last_payload_sha256 = None
        self._last_sim_time = None
        self._last_status = None
        self._reward_emitted = False
        self._finalized = False
        self._last_full_evaluation_time = None
        self._immediate_failure = None
        return self.status()

    def _taint(self, reason: str) -> None:
        if reason not in self._errors:
            self._errors.append(reason)

    def _coverage(self, coverage: Any) -> float:
        if not isinstance(coverage, dict):
            raise ValueError("contact_coverage_missing")
        if coverage.get("complete") is not True or coverage.get("continuous") is not True:
            raise ValueError("contact_coverage_incomplete")
        if coverage.get("scope") not in {CONTACT_SCOPE, "all_scene_contacts"}:
            raise ValueError("contact_coverage_scope_unsupported")
        if coverage.get("force_aggregation") != FORCE_AGGREGATION:
            raise ValueError("contact_normal_point_sum_not_attested")
        covered = coverage.get("object_ids")
        if not isinstance(covered, (list, tuple)) or not {self.spec.peg_id, self.spec.socket_id}.issubset(covered):
            raise ValueError("task_part_contact_coverage_missing")
        if not isinstance(coverage.get("source"), str) or not coverage["source"].strip():
            raise ValueError("contact_measurement_source_missing")
        dt = _number(coverage.get("sample_dt_s"), "coverage.sample_dt_s")
        if dt <= 0:
            raise ValueError("contact sample window must be positive")
        if self.physics_dt_s is not None and not math.isclose(dt, self.physics_dt_s, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("contact_sample_window_does_not_match_physics_dt")
        return dt

    def record(self, *, sim_time: Any, objects: Any, contacts: Any, coverage: Any,
               action: Any, released: Any, geometry_clearance_m: Any,
               episode_id: str | None = None, teleported: bool = False,
               kinematic_pose_write: bool = False) -> dict[str, Any]:
        if self.episode_id is None:
            self._taint("episode_reset_required")
            return self._evaluate()
        if self._finalized:
            raise RuntimeError("assembly episode was finalized; reset before adding evidence")
        if episode_id is not None and episode_id != self.episode_id:
            self._taint("episode_id_mismatch")
            return self._evaluate()
        try:
            payload = {"sim_time": sim_time, "objects": objects, "contacts": contacts, "coverage": coverage,
                       "action": action, "released": released, "geometry_clearance_m": geometry_clearance_m,
                       "teleported": teleported, "kinematic_pose_write": kinematic_pose_write}
            payload_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
            time_s = _number(sim_time, "sim_time")
            if time_s < 0:
                raise ValueError("sim_time must be nonnegative")
            if self._last_sim_time is not None and time_s == self._last_sim_time:
                if payload_sha256 == self._last_payload_sha256:
                    # Capturing an observation twice is not another physics
                    # step and cannot issue another reward.
                    result = self.status()
                    result["reward"] = 0.
                    result["duplicate_sample"] = True
                    return result
                raise ValueError("same_sim_time_has_conflicting_measurements")
            if self._last_sim_time is not None and time_s < self._last_sim_time:
                raise ValueError("sim_time_moved_backwards_without_reset")
            dt = self._coverage(coverage)
            if self._last_sim_time is not None and not math.isclose(time_s-self._last_sim_time, dt, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("physics_step_evidence_gap")
            if not isinstance(released, bool) or not isinstance(teleported, bool) or not isinstance(kinematic_pose_write, bool):
                raise ValueError("released_and_pose_write_flags_must_be_boolean")
            clearance = _number(geometry_clearance_m, "geometry_clearance_m")
            if self.samples and not math.isclose(clearance, self.samples[0]["geometry_clearance_m"], rel_tol=0, abs_tol=1e-12):
                raise ValueError("geometry_changed_within_episode")
            if not isinstance(objects, dict):
                raise ValueError("full_object_state_missing")
            parts = {}
            for object_id in (self.spec.peg_id, self.spec.socket_id):
                item = objects.get(object_id)
                if not isinstance(item, dict):
                    raise ValueError(f"task_object_state_missing:{object_id}")
                position = _vector(item.get("position_m", item.get("position")), 3, f"{object_id}.position_m")
                orientation = _vector(item.get("orientation_wxyz"), 4, f"{object_id}.orientation_wxyz")
                if abs(math.sqrt(sum(value*value for value in orientation))-1) > 1e-3:
                    raise ValueError(f"{object_id}.orientation_wxyz_not_normalized")
                parts[object_id] = {"position_m": position, "orientation_wxyz": orientation,
                                    "linear_velocity_mps": _vector(item.get("linear_velocity_mps"), 3, f"{object_id}.linear_velocity_mps"),
                                    "angular_velocity_rps": _vector(item.get("angular_velocity_rps"), 3, f"{object_id}.angular_velocity_rps")}
            if not isinstance(contacts, list):
                raise ValueError("contact_samples_missing")
            aggregate: dict[tuple[str, str], float] = {}
            raw_contacts = []
            for contact in contacts:
                if not isinstance(contact, dict):
                    raise ValueError("contact_sample_must_be_object")
                pair = contact.get("pair")
                if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(value, str) and value for value in pair) or pair[0] == pair[1]:
                    raise ValueError("contact_object_pair_invalid")
                force = _number(contact.get("normal_force_N"), "contact.normal_force_N")
                if force < 0:
                    raise ValueError("contact_normal_force_must_be_nonnegative")
                if "normal_force_points_N" in contact:
                    points = contact["normal_force_points_N"]
                    if not isinstance(points, list) or any(_number(value, "normal_force_points_N") < 0 for value in points):
                        raise ValueError("normal_force_points_invalid")
                    if not math.isclose(sum(points), force, rel_tol=1e-6, abs_tol=1e-8):
                        raise ValueError("normal_force_is_not_sum_of_points")
                if self.spec.peg_id not in pair and self.spec.socket_id not in pair:
                    continue
                normalized = tuple(sorted(pair))
                aggregate[normalized] = aggregate.get(normalized, 0.)+force
                raw_contacts.append(copy.deepcopy(contact))
            if action is not None and (not isinstance(action, dict) or not action):
                raise ValueError("robot_action_must_be_nonempty_object_or_null")
            if action is not None and (action.get("mode") != "joint_position" or not isinstance(action.get("payload"), dict) or not action["payload"]):
                raise ValueError("physical_joint_position_action_evidence_required")
            if len(self.samples) >= self.max_samples:
                raise ValueError("assembly_evidence_capacity_exceeded")
            sample = {"time_s": time_s, "geometry_clearance_m": clearance, "parts": parts,
                      "contacts": [{"pair": list(pair), "normal_force_N": force} for pair, force in sorted(aggregate.items())],
                      "raw_contact_points": raw_contacts, "contact_coverage": copy.deepcopy(coverage),
                      "released": released, "action": copy.deepcopy(action),
                      "motion_source": "robot_joint_control" if action is not None else "physics_passive",
                      "teleported": teleported, "kinematic_pose_write": kinematic_pose_write}
            self.samples.append(sample)
            self._last_sim_time, self._last_payload_sha256 = time_s, payload_sha256
            self._check_immediate_failure(sample)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._taint(f"evidence_invalid:{exc}")
        return self._evaluate()

    def _check_immediate_failure(self, sample: dict[str, Any]) -> None:
        """Cheap safety checks run for every measured physics step."""
        if self._immediate_failure is not None:
            return
        classification, reason = None, None
        peg = sample["parts"][self.spec.peg_id]
        linear = math.sqrt(sum(value*value for value in peg["linear_velocity_mps"]))
        angular = math.sqrt(sum(value*value for value in peg["angular_velocity_rps"]))
        if sample["teleported"] or sample["kinematic_pose_write"]:
            classification, reason = "invalid_trace", "direct_pose_write_not_assembly"
        elif not self.spec.clearance_m[0] <= sample["geometry_clearance_m"] <= self.spec.clearance_m[1]:
            classification, reason = "geometry_mismatch", "clearance_out_of_range"
        elif linear > self.spec.max_linear_speed_mps or angular > self.spec.max_angular_speed_rps:
            classification, reason = "invalid_trace", "measured_velocity_limit_exceeded"
        elif any(contact["normal_force_N"] > self.spec.max_force_N for contact in sample["contacts"]):
            classification, reason = "overforce", "maximum_force_exceeded"
        elif any(tuple(contact["pair"]) not in self.spec.allowed_contact_pairs for contact in sample["contacts"]):
            classification, reason = "unexpected_contact", "unlisted_contact_pair"
        elif peg["position_m"][2] < self.spec.drop_below_z_m:
            classification, reason = "dropped", "part_dropped"
        if classification:
            self._immediate_failure = {"schema_version": "scene_factory.assembly_result.v1", "status": "failed",
                                       "classification": classification, "reasons": [reason],
                                       "metrics": {"sample_index": len(self.samples)-1}}

    def _evaluate(self, *, finalize: bool = False) -> dict[str, Any]:
        errors = list(self._errors)
        elapsed = self.samples[-1]["time_s"]-self.samples[0]["time_s"] if self.samples else 0.
        deadline = elapsed >= self.spec.timeout_s
        current_time = self.samples[-1]["time_s"] if self.samples else None
        full_due = (finalize or deadline or self._last_full_evaluation_time is None
                    or (self._last_status is not None and self._last_status.get("task_success"))
                    or (current_time is not None and current_time-self._last_full_evaluation_time >= self.evaluation_interval_s-1e-9))
        if self._immediate_failure is not None:
            result = copy.deepcopy(self._immediate_failure)
        elif full_due:
            result = evaluate_assembly_trace(self.spec, self.samples)
            self._last_full_evaluation_time = current_time
        elif self._last_status is not None:
            result = copy.deepcopy(self._last_status["assembly_result"])
        else:
            result = evaluate_assembly_trace(self.spec, self.samples)
        irreversible = result.get("classification") in {"overforce", "unexpected_contact", "dropped", "invalid_trace", "geometry_mismatch"}
        passed = result.get("status") == "passed" and not errors
        if passed and self.samples:
            end_time = self.samples[-1]["time_s"]
            released_window = [sample for sample in self.samples if sample["released"] and sample["time_s"] >= end_time-self.spec.release_stable_s-1e-9]
            if any(math.sqrt(sum(v*v for v in sample["parts"][self.spec.peg_id]["linear_velocity_mps"])) > self.spec.release_max_linear_speed_mps
                   or math.sqrt(sum(v*v for v in sample["parts"][self.spec.peg_id]["angular_velocity_rps"])) > self.spec.release_max_angular_speed_rps
                   for sample in released_window):
                passed = False
                result = {**result, "status": "failed", "classification": "timeout", "reasons": ["measured_release_velocity_unstable"]}
        if errors or result.get("status") == "not_verified":
            evaluation_status = "not_verified"
        elif passed:
            evaluation_status = "passed"
        elif irreversible or deadline or finalize:
            evaluation_status = "failed"
        else:
            evaluation_status = "in_progress"
        reward = 1. if passed and not self._reward_emitted else 0.
        if reward:
            self._reward_emitted = True
        status = {"schema_version": RUNTIME_SCHEMA, "episode_id": self.episode_id,
                  "spec_id": self.spec.spec_id, "spec_sha256": self.spec_sha256,
                  "task_success": passed, "evaluation_status": evaluation_status,
                  "reward": reward, "terminated": passed or (evaluation_status == "failed" and not deadline),
                  "truncated": bool((deadline or finalize) and not passed and (errors or deadline or result.get("status") == "not_verified")),
                  "failure_reason": errors[0] if errors else ((result.get("reasons") or [None])[0] if evaluation_status in {"failed", "not_verified"} else None),
                  "evidence_errors": errors, "sample_count": len(self.samples), "elapsed_s": elapsed,
                  "assembly_result": result,
                  "state_scope": {"pose": "full_SE3_wxyz_world_m", "velocity": "live_physics_mps_rps",
                                  "contacts": CONTACT_SCOPE, "normal_force": FORCE_AGGREGATION,
                                  "sampling": "each_physics_step", "evidence_reuse_across_episodes": False,
                                  "full_trace_evaluation_interval_s": self.evaluation_interval_s,
                                  "source_validity": "depends_on_backend_measurement_and_coverage_attestation"}}
        self._last_status = status
        return copy.deepcopy(status)

    def status(self) -> dict[str, Any]:
        if self._last_status is None:
            result = self._evaluate()
        else:
            result = copy.deepcopy(self._last_status)
        result["reward"] = 0.
        return result

    def finalize(self) -> dict[str, Any]:
        result = self._evaluate(finalize=True)
        self._finalized = True
        return result

    def trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.samples)
