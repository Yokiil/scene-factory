"""Restricted Franka mug commands and one same-episode physical retry."""

from __future__ import annotations

import copy
import math
from collections import deque

from ..robotics import (
    MugLiftCommand,
    MugLiftController,
    MugLiftPhase,
    _grasp_diagnostics_unavailable,
)


PHASE_SKILLS = {
    MugLiftPhase.PRE_GRASP: "reach",
    MugLiftPhase.APPROACH: "approach",
    MugLiftPhase.GRASP: "grasp",
    MugLiftPhase.VERIFY_GRASP: "grasp",
    MugLiftPhase.LIFT: "lift",
    MugLiftPhase.TRANSFER: "transfer",
    MugLiftPhase.LOWER: "lower",
    MugLiftPhase.RELEASE: "release",
    MugLiftPhase.VERIFY_PLACE: "release",
    MugLiftPhase.FAILED: "recover",
}
CANDIDATES = ("nominal", "lateral")


class MugSkillBridge:
    """Dispatch only the skill allowed by the current measured controller phase."""

    def __init__(self, controller: MugLiftController, candidate: str, source_support: str):
        if candidate not in CANDIDATES:
            raise ValueError("unknown mug grasp candidate")
        self.source_support = source_support
        # Rectify the right_gripper frame against the measured YCB handle.
        controller.grasp_offset = (0.0, 0.0, 0.001)
        self.template = copy.copy(controller)
        self.candidate = candidate
        self.attempt = 1
        self.recovery_stage = None
        self.last_failure = None
        self.can_recover = False
        self.recovery_steps = 0
        self.retreat_goal = None
        self.positions = deque(maxlen=12)
        self.loaded_grip = False
        self.finger_targets = None
        self.grasp_feedback = None
        if candidate == "lateral":
            x, y, z = controller.grasp_anchor_position
            controller.grasp_anchor_position = (x - 0.15, y - 0.16, z + 0.005)

    def snapshot(self):
        return {
            "attempt": self.attempt,
            "candidate": self.candidate,
            "can_recover": self.can_recover,
            "last_failure": self.last_failure,
            "recovery_stage": self.recovery_stage,
        }

    def validate(self, action, controller, steps):
        if set(action) != {"mode", "skill"} or action["mode"] != "mug_skill":
            raise ValueError("mug_skill accepts only mode and a skill name")
        if steps >= controller.max_steps or controller.phase == MugLiftPhase.DONE:
            raise RuntimeError("mug skill episode has ended")
        expected = PHASE_SKILLS[controller.phase]
        if action["skill"] != expected:
            raise ValueError(
                f"skill {action['skill']!r} is unavailable in phase {controller.phase.value}"
            )
        if expected == "recover" and not self.can_recover:
            raise RuntimeError("no physical retry available for this failure")

    def observe(self, controller, position, diagnostics, steps, observation, target_id):
        position = tuple(float(v) for v in position)
        self.positions.append(position)
        self.can_recover = False
        if controller.phase != MugLiftPhase.FAILED:
            return
        if self.recovery_stage is None:
            self.last_failure = controller.failure_reason
        if (
            self.candidate != "lateral"
            or self.attempt != 1
            or steps >= controller.max_steps
            or controller.failure_reason
            not in {"grasp_failure", "pre_grasp_timeout", "approach_timeout"}
            or controller.pick_status == "passed"
            or _grasp_diagnostics_unavailable(diagnostics)
        ):
            return
        # Require twelve successive measurements at the original support height.
        origin = controller.initial_target_position
        self.can_recover = (
            observation.get("contact_coverage", {}).get("complete") is True
            and any(
                set(pair["pair"]) == {target_id, self.source_support}
                and math.isfinite(pair["normal_force_N"])
                and pair["normal_force_N"] > 1e-6
                for pair in observation.get("contacts", [])
            )
            and diagnostics.get("finger_target_contact") is False
            and len(self.positions) == self.positions.maxlen
            and all(
                math.dist(sample[:2], origin[:2]) <= 0.065
                and abs(sample[2] - origin[2]) <= 0.012
                and math.dist(sample, position) <= 0.002
                for sample in self.positions
            )
        )

    def regulated_fingers(self, observation, target_id, stiffness, dt):
        """Slew-limit drive compression against measured position and normal force."""
        if observation.get("contact_coverage", {}).get("complete") is not True:
            raise ValueError("mug grip requires complete current contact coverage")
        measured = sum(
            float(row["normal_force_N"])
            for row in observation["contacts"]
            if set(row["pair"]) == {"robot", target_id}
        )
        diag = observation["robot"]["grasp_diagnostics"]
        if (
            not math.isfinite(measured)
            or measured < 0
            or diag.get("contact_force_read_valid") is not True
        ):
            raise ValueError("mug grip requires finite measured target contact force")
        fingers = diag["finger_dofs"]
        if len(fingers) != 2 or stiffness <= 0 or not math.isfinite(stiffness):
            raise ValueError("mug grip requires measured two-finger positions and drive stiffness")
        # The reference mug's verified hold position was 19.4 / 20.9 mm.
        # Start both fingers during side entry so the moving palm cannot push
        # the cup away. Start the feedback slew when grip load is measured.
        first_command = self.finger_targets is None
        if first_command:
            self.finger_targets = [0.0194, 0.0209]
        if measured >= 3.0 and sum(float(f["position"]) for f in fingers) < 0.066:
            self.loaded_grip = True
        target_force = 20.0  # N, total normal force on the mug from both fingers
        compression = min(
            0.0095, max(0.0, (target_force + 0.6 * (target_force - measured)) / (2 * stiffness))
        )
        targets = []
        for finger, previous in zip(fingers, self.finger_targets, strict=True):
            actual, low, high = (float(finger[k]) for k in ("position", "lower", "upper"))
            if not all(map(math.isfinite, (actual, low, high))) or high <= low:
                raise ValueError("invalid measured finger position or limits")
            # Unload early, before the independent 30 N hard stop. Never store
            # large spring compression even if a finger is blocked in contact.
            if measured > 25.0:
                goal = actual + 0.0004
            elif first_command or not self.loaded_grip:
                goal = previous
            else:
                goal = actual - compression
                goal = max(previous - 0.045 * dt, min(previous + 0.045 * dt, goal))
                goal = max(goal, actual - compression)
            targets.append(max(low + 0.00075, min(high - 0.00025, goal)))
        self.finger_targets = targets
        self.grasp_feedback = {
            "target_normal_force_N": target_force,
            "measured_normal_force_N": measured,
            "drive_targets_m": list(targets),
        }
        return targets

    def recovery_command(self, ee_position, target_position):
        if self.recovery_stage is None:
            self.recovery_stage = "open"
            self.retreat_goal = (
                float(ee_position[0]),
                float(ee_position[1]),
                max(float(ee_position[2]) + 0.065, float(target_position[2]) + 0.14),
            )
        goal = None
        if self.recovery_stage == "retreat":
            x, y, z = self.retreat_goal
            correction = min(0.035, max(0.0, z - float(ee_position[2])))
            goal = (x, y, z + correction)
        if self.recovery_stage == "home":
            return MugLiftCommand(MugLiftPhase.RELEASE, None, "recover_home", False)
        return MugLiftCommand(MugLiftPhase.RELEASE, goal, "recover_open", goal is not None)

    def advance_recovery(
        self,
        controller,
        target,
        ee_position,
        orientation_error,
        diagnostics,
        steps,
        ik_ok,
        home_ready=False,
    ):
        self.recovery_steps += 1
        controller.total_steps = steps
        if steps >= controller.max_steps:
            controller.fail("timeout")
            return controller
        if self.recovery_steps >= (300 if self.recovery_stage == "home" else 180):
            controller.fail("recovery_timeout")
            return controller
        detached = diagnostics.get("finger_target_contact") is False
        opened = diagnostics.get("gripper_open") is True
        if _grasp_diagnostics_unavailable(diagnostics) or not detached:
            controller.fail("recovery_contact_unavailable_or_present")
            return controller
        if self.recovery_stage == "open" and opened:
            self.recovery_stage = "retreat"
        elif (
            self.recovery_stage == "retreat"
            and ik_ok
            and opened
            and orientation_error <= controller._ORIENTATION_TOLERANCE_RAD
            and math.dist(tuple(ee_position), self.retreat_goal) <= 0.025
        ):
            self.recovery_stage = "home"
        elif self.recovery_stage == "home" and home_ready and opened:
            # Refresh the drive controller from its original configuration. Keep
            # the global budget and the original target height for lift scoring.
            controller = copy.copy(self.template)
            controller.grasp_anchor_position = tuple(float(v) for v in target)
            controller.total_steps = steps
            self.loaded_grip = False
            self.finger_targets = self.grasp_feedback = None
            self.attempt = 2
            self.candidate = "nominal"
            self.recovery_stage = None
        return controller
