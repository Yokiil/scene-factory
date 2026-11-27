"""Select one registered Franka skill from each committed Isaac world state."""

from __future__ import annotations

import math
from types import MappingProxyType

from .agent_adapter import RegisteredSkill
from .backends.isaac_mug_skills import PHASE_SKILLS
from .robotics import MUG_LIFT_PHASE_TRANSITIONS

_LOCAL = {
    "observe",
    "detect_object",
    "estimate_graspability",
    "verify_grasp",
    "place",
    "verify_place",
    "abort",
}
SKILLS = MappingProxyType(
    {
        name: RegisteredSkill(
            name,
            f"Franka {name}",
            {"mode": "local_skill" if name in _LOCAL else "mug_skill", "skill": name},
        )
        for name in sorted(_LOCAL | set(PHASE_SKILLS.values()))
    }
)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "missing_or_invalid_number")
    return float(value)


def _vector(value, size):
    _require(isinstance(value, (list, tuple)) and len(value) == size, "missing_or_invalid_vector")
    return tuple(_number(item) for item in value)


def _bool(value):
    _require(type(value) is bool, "missing_physical_boolean")
    return value


def _forces(world, target, source, support):
    coverage = world["contact_coverage"]
    _require(
        coverage["complete"] is True
        and coverage["continuous"] is True
        and coverage["scope"] == "all_scene_contacts"
        and coverage["force_aggregation"] == "sum_per_point_normal_magnitudes"
        and {target, source, support, "robot"}.issubset(coverage["object_ids"]),
        "incomplete_contact_coverage",
    )
    contacts = world["contacts"]
    _require(isinstance(contacts, list), "contact_list_missing")
    forces = {}
    for row in contacts:
        pair = row["pair"]
        _require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(v, str) for v in pair)
            and pair[0] != pair[1],
            "invalid_contact_pair",
        )
        force = _number(row["normal_force_N"])
        _require(force >= 0 and row["units"] == "N", "invalid_normal_force")
        if target in pair:
            other = pair[1] if pair[0] == target else pair[0]
            forces[other] = forces.get(other, 0.0) + force
    return forces


class EmbodiedAgentRuntime:
    """Each instance follows one episode, including at most one measured recovery."""

    def __init__(self, *, max_contact_force_N=30.0):
        self.max_contact_force_N = _number(max_contact_force_N)
        _require(0 < self.max_contact_force_N <= 30, "invalid_force_limit")
        self.result, self.reason, self.events = "running", None, []
        self._episode, self._phase, self._candidate = None, None, None
        self._step, self._time, self._attempt = -1, -1.0, 0
        self._version = None
        self._initial = self._previous = self._verify_start = None
        self._contact_steps = self._settled = 0
        self._grasped = self._carried = self._transported = self._lowered = False

    @staticmethod
    def observe(world):
        _require(
            world["schema_version"] == "scene_factory.world_state.v1"
            and world["backend"] == "isaac"
            and world["source"] == "runtime_observation",
            "requires_live_isaac_world",
        )
        return world

    @staticmethod
    def detect_object(world, object_id):
        obj = world["objects"][object_id]
        position = _vector(obj["position_m"], 3)
        orientation = _vector(obj["orientation_wxyz"], 4)
        _require(
            abs(math.hypot(*orientation) - 1.0) <= 0.001 and obj["pose_source"],
            "object_pose_unverified",
        )
        return {"object_id": object_id, "position_m": position, "orientation_wxyz": orientation}

    def estimate_graspability(self, world, target):
        obj = world["objects"][target]
        _require(
            obj["dynamic"] is True
            and obj["asset_id"] == "mug_001"
            and obj["physics_handle_valid"] is True
            and obj["pose_source"] == "PhysX tensor API",
            "requires_live_mug_body",
        )
        position = self.detect_object(world, target)["position_m"]
        tcp = _vector(world["robot"]["end_effector_pose"]["position"], 3)
        extents = _vector(obj["instance"]["bbox_m"], 3)
        _require(min(extents) > 0, "invalid_target_dimensions")
        finger_positions = world["robot"]["finger_positions"]
        distances = [
            math.dist(_vector(finger_positions[k], 3), position) for k in ("left", "right")
        ]
        # Conservative geometric candidate test. Subsequent measured contact and follow
        # are required to verify that closing the gripper physically captured the cup.
        close_to_fingers = max(distances) <= 2 * math.hypot(*extents)
        return {
            "within_grasp_neighborhood": close_to_fingers,
            "tcp_distance_m": math.dist(position, tcp),
            "finger_distances_m": distances,
            "scope": "geometric_candidate",
        }

    @staticmethod
    def grasp(phase, graspability):
        _require(phase in ("GRASP", "VERIFY_GRASP"), "invalid_grasp_phase")
        _require(
            graspability["within_grasp_neighborhood"] is True, "mug_outside_grasp_neighborhood"
        )
        return SKILLS["grasp"].skill_id

    def recover(self, bridge, robot, finger_contact, source_force):
        failure = robot["failure_reason"]
        allowed = (
            robot["phase"] == "FAILED"
            and bridge["attempt"] == 1
            and bridge["candidate"] == "lateral"
            and not self._grasped
            and not finger_contact
            and source_force > 1e-6
            and bridge["can_recover"] is True
            and failure == bridge["last_failure"]
            and failure in ("grasp_failure", "approach_timeout", "pre_grasp_timeout")
        )
        return (
            SKILLS["recover"].skill_id
            if allowed
            else self.abort(failure or "unrecoverable_skill_failure")
        )

    def verify_grasp(self, pos, held, diagnostics):
        return (
            self._verify_start is not None
            and self._contact_steps >= 10
            and held
            and diagnostics["all_finger_positions_within_limits"] is True
            and pos[2] - self._verify_start >= 0.005
        )

    def place(self, phase):
        _require(
            phase in ("TRANSFER", "LOWER", "RELEASE", "VERIFY_PLACE"), "invalid_placement_phase"
        )
        return SKILLS[PHASE_SKILLS[phase]].skill_id

    def verify_place(self, robot, oracle, settle_steps):
        _require(type(settle_steps) is int and settle_steps > 0, "invalid_settle_steps")
        return (
            self._lowered
            and self._settled >= settle_steps
            and robot["pick_status"] == "passed"
            and robot["place_status"] == "passed"
            and robot["released"] is True
            and all(
                oracle[k] is True
                for k in (
                    "task_success",
                    "pick_success",
                    "placement_stable",
                    "in_target_region",
                    "released",
                )
            )
        )

    def abort(self, reason, status="failed"):
        _require(status in ("failed", "not_verified"), "invalid_abort_status")
        if self.result == "running":
            self._finish(status, reason)
        return None

    def _finish(self, status, reason):
        self.result, self.reason = status, reason
        self.events.append(
            {"step_index": self._step, "phase": self._phase, "result": status, "reason": reason}
        )
        return None

    def next_skill(self, world):
        if self.result != "running":
            return None
        try:
            return self._decide(self.observe(world))
        except (KeyError, TypeError, IndexError, ValueError, AttributeError, OverflowError) as exc:
            return self.abort(str(exc), "not_verified")

    def _decide(self, world):
        episode, step, now = world["episode_id"], world["step_index"], _number(world["sim_time"])
        _require(
            isinstance(episode, str)
            and episode
            and (self._episode is None or episode == self._episode),
            "episode_identity_changed",
        )
        _require(
            type(step) is int and step == self._step + 1 and now >= 0 and now > self._time,
            "nonconsecutive_physics_step",
        )
        version = world["state_version"]
        _require(
            type(version) is int
            and version >= 0
            and (self._version is None or version == self._version + 1),
            "nonconsecutive_state_version",
        )
        self._step, self._time, self._episode, self._version = step, now, episode, version
        robot, bridge, task = world["robot"], world["mug_skill"], world["task"]
        spec = task["success"]
        target = task["target_object"]
        _require(
            robot["name"] == "franka"
            and spec["predicate"] == "pick_and_place"
            and target == "mug_1"
            and spec["subject"] == target,
            "unsupported_task_or_robot",
        )
        source, support = spec["source_support"], spec["target_support"]
        pos = self.detect_object(world, target)["position_m"]
        self.detect_object(world, support)
        graspability = self.estimate_graspability(world, target)
        _vector(robot["end_effector_pose"]["position"], 3)
        for finger in ("left", "right"):
            _vector(robot["finger_positions"][finger], 3)
        diagnostics, oracle = world["grasp_diagnostics"], world["task_oracle"]
        for key in (
            "dof_limits_valid",
            "contact_report_available",
            "contact_report_subscribed",
            "contact_force_read_valid",
            "finger_material_resolved",
            "target_material_resolution",
        ):
            _require(diagnostics[key] is True, "invalid_grasp_diagnostic:" + key)
        finger_contact, opened = (
            _bool(diagnostics["finger_target_contact"]),
            _bool(diagnostics["gripper_open"]),
        )
        _bool(diagnostics["all_finger_positions_within_limits"])
        if (
            robot["phase"] == "VERIFY_GRASP"
            and diagnostics["all_finger_positions_within_limits"] is not True
        ):
            return self.abort("finger_position_outside_runtime_limits")
        forces = _forces(world, target, source, support)
        if max(forces.values(), default=0.0) > self.max_contact_force_N:
            self.events.append(
                {
                    "step_index": step,
                    "phase": robot["phase"],
                    "state_version": version,
                    "force_by_contact_owner_N": forces,
                    "limit_N": self.max_contact_force_N,
                }
            )
            return self.abort("measured_target_overforce")
        if any(f > 1e-6 and owner not in {"robot", source, support} for owner, f in forces.items()):
            return self.abort("unexpected_target_contact")
        phase, attempt, candidate = robot["phase"], bridge["attempt"], bridge["candidate"]
        _require(
            type(attempt) is int and attempt in (1, 2) and candidate in ("nominal", "lateral"),
            "unknown_grasp_candidate",
        )
        if self._phase is None:
            _require(phase == "PRE_GRASP" and attempt == 1, "initial_phase_invalid")
            self._initial = self._previous = pos
        elif attempt != self._attempt:
            _require(
                self._phase == "FAILED"
                and phase == "PRE_GRASP"
                and self._attempt == 1
                and attempt == 2
                and self._candidate == "lateral"
                and candidate == "nominal",
                "invalid_recovery_transition",
            )
            self._contact_steps, self._verify_start = 0, None
        else:
            _require(candidate == self._candidate, "grasp_candidate_changed_without_recovery")
            if phase != self._phase and phase != "FAILED":
                _require(
                    phase in MUG_LIFT_PHASE_TRANSITIONS.get(self._phase, ()),
                    "invalid_skill_phase_transition",
                )
        previous_phase = self._phase
        self._attempt, self._candidate, self._phase = attempt, candidate, phase
        held = finger_contact and not opened
        goal = _vector(spec["target_position_m"], 3)
        tol = _vector(spec["target_tolerance_m"], 3)
        if spec.get("target_region_xy") is not None:
            xmin, xmax, ymin, ymax = _vector(spec["target_region_xy"], 4)
            xy_ok = xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax
        else:
            xy_ok = all(abs(pos[i] - goal[i]) <= tol[i] for i in (0, 1))
        in_region = xy_ok and abs(pos[2] - goal[2]) <= tol[2]
        supported = forces.get(support, 0.0) > 1e-6
        released = opened and not finger_contact and oracle["released"] is True
        stable = (
            released
            and supported
            and in_region
            and math.dist(self._previous, pos) <= _number(spec["max_settle_step_distance_m"])
        )
        self._previous = pos
        self._settled = self._settled + 1 if self._lowered and stable else 0
        if phase == "FAILED":
            if self.recover(bridge, robot, finger_contact, forces.get(source, 0.0)) is None:
                return None
        elif phase in ("VERIFY_GRASP", "LIFT") and not self._grasped:
            if self._verify_start is None:
                self._verify_start = pos[2]
            self._contact_steps = self._contact_steps + 1 if held else 0
            if phase == "LIFT":
                if not self.verify_grasp(pos, held, diagnostics):
                    return self._finish("not_verified", "grasp_follow_or_contact_unverified")
                self._grasped = True
        elif phase in ("TRANSFER", "LOWER"):
            if (
                not self._grasped
                or not held
                or oracle["holding"] is not True
                or oracle["pick_success"] is not True
            ):
                return self._finish("failed", "physical_grasp_lost")
            if phase == "TRANSFER" and previous_phase != "TRANSFER":
                if pos[2] - self._initial[2] < _number(spec["min_lift_delta_m"]):
                    return self._finish("not_verified", "measured_lift_below_minimum")
                self._carried = True
            if phase == "LOWER":
                if not self._carried or not xy_ok:
                    return self._finish("not_verified", "cup_not_over_target_region")
                self._transported = True
        elif phase == "RELEASE" and previous_phase != "RELEASE":
            if (
                not self._transported
                or not held
                or not in_region
                or oracle["in_target_region"] is not True
            ):
                return self._finish("not_verified", "placement_before_release_unverified")
            self._lowered = True
        elif phase == "DONE":
            if not self.verify_place(robot, oracle, spec["settle_steps"]):
                return self._finish("not_verified", "stable_supported_placement_unverified")
            return self._finish("success", "physical_pick_place_verified")
        if phase in ("TRANSFER", "LOWER", "RELEASE", "VERIFY_PLACE"):
            skill = self.place(phase)
        elif phase in ("GRASP", "VERIFY_GRASP"):
            skill = self.grasp(phase, graspability)
        else:
            skill = SKILLS[PHASE_SKILLS[phase]].skill_id
        self.events.append(
            {
                "step_index": step,
                "sim_time": now,
                "phase": phase,
                "skill": skill,
                "candidate": candidate,
                "attempt": attempt,
                "state_version": version,
                "cup_position_m": list(pos),
                "graspability": graspability,
                "held": held,
                "released": released,
                "target_support_contact": supported,
                "in_target_region": in_region,
                "stable_steps": self._settled,
                "failure": bridge["last_failure"],
            }
        )
        return skill
