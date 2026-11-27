"""Measured contact gates for a bounded Franka grasp, independent of Isaac.

This controller proposes drive targets only. ``ready`` means the grasp prefix
completed; it is never an assembly-success label. Raw simulator measurements
and the separately frozen task evaluator remain authoritative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    return tuple(_number(v, name) for v in value)


def _local_offset(position, origin, quaternion):
    """Rotate world displacement into the measured TCP frame (wxyz)."""
    w, x, y, z = quaternion
    if abs(math.hypot(w, x, y, z)-1.) > 1e-3:
        raise ValueError("TCP orientation must be normalized")
    dx, dy, dz = (p-o for p, o in zip(position, origin, strict=True))
    return ((1-2*(y*y+z*z))*dx + 2*(x*y+w*z)*dy + 2*(x*z-w*y)*dz,
            2*(x*y-w*z)*dx + (1-2*(x*x+z*z))*dy + 2*(y*z+w*x)*dz,
            2*(x*z+w*y)*dx + 2*(y*z-w*x)*dy + (1-2*(x*x+y*y))*dz)


def anchored_joint_target(command_origin, ik_origin, ik_target) -> tuple[float, ...]:
    """Apply an IK displacement without dropping the existing drive bias.

    All IK samples must share a fixed warm start and pose reference. Re-solving
    the measured stationary pose alone loses drive compensation for static sag.
    The caller still applies runtime joint limits and per-step rate limits.
    """
    command = _vector(command_origin, 7, "command_origin")
    origin = _vector(ik_origin, 7, "ik_origin")
    target = _vector(ik_target, 7, "ik_target")
    return tuple(_number(c+(t-o), "anchored target")
                 for c, o, t in zip(command, origin, target, strict=True))


@dataclass(frozen=True)
class GraspControlConfig:
    contact_force_N: float = .05
    table_grip_force_N: float = .6
    airborne_grip_force_N: float = 1.2
    force_deadband_N: float = .15
    max_finger_force_N: float = 5.
    table_clear_force_N: float = .02
    free_close_speed_mps: float = .01
    approach_close_speed_mps: float = .001
    approach_clearance_m: float = .003
    contact_close_speed_mps: float = .0004
    max_compression_m: float = .0012
    stable_linear_speed_mps: float = .005
    stable_angular_speed_rps: float = .25
    pause_angular_speed_rps: float = .5
    stop_angular_speed_rps: float = .75
    recovery_duration_s: float = .1
    recovery_timeout_s: float = .75
    recovery_open_speed_mps: float = 0.
    stable_duration_s: float = .2
    clear_duration_s: float = .1
    contact_loss_timeout_s: float = .05
    max_slip_m: float = .0005
    clear_height_m: float = .0005
    lift_speed_mps: float = .0005
    max_lift_m: float = .003
    stage_timeout_s: float = 12.

    def __post_init__(self):
        for name, value in asdict(self).items():
            number = _number(value, name)
            if number < 0 or number == 0 and name != "recovery_open_speed_mps":
                raise ValueError(f"{name} must be positive (recovery opening may be zero)")
        if not self.contact_force_N < self.table_grip_force_N <= self.airborne_grip_force_N < self.max_finger_force_N:
            raise ValueError("grip force thresholds must be ordered")
        if self.force_deadband_N >= self.table_grip_force_N-self.contact_force_N:
            raise ValueError("force deadband must preserve bilateral contact")
        if not self.stable_angular_speed_rps <= self.pause_angular_speed_rps < self.stop_angular_speed_rps:
            raise ValueError("stable, pause and stop angular speeds must be ordered")
        if self.recovery_duration_s >= self.recovery_timeout_s:
            raise ValueError("recovery window must be shorter than its timeout")
        if self.recovery_open_speed_mps > self.contact_close_speed_mps:
            raise ValueError("recovery opening must be no faster than contact closing")
        if self.clear_height_m >= self.max_lift_m:
            raise ValueError("clear height must be below bounded lift")
        if not self.contact_close_speed_mps <= self.approach_close_speed_mps <= self.free_close_speed_mps:
            raise ValueError("closing speeds must decrease toward contact")
        if max(self.stable_duration_s, self.clear_duration_s) >= self.stage_timeout_s:
            raise ValueError("stable windows must fit within the stage timeout")


class ContactGraspController:
    """One update per measured physics step; failure is latched until recreated.

    A contact coverage attestation is required even for an empty contact list.
    Durations accumulate only from consecutive, valid measured samples. The
    two fingers must have increasing, finite runtime position limits.
    """

    def __init__(self, *, physics_dt_s: float, minimum_fraction: float = .15,
                 closed_positions=(0., 0.), open_positions=(.04, .04),
                 object_width_m: float = .02,
                 config: GraspControlConfig | None = None):
        self.config = config or GraspControlConfig()
        self.dt = _number(physics_dt_s, "physics_dt_s")
        self.minimum_fraction = _number(minimum_fraction, "minimum_fraction")
        if self.dt <= 0 or not 0 <= self.minimum_fraction < 1:
            raise ValueError("invalid dt or minimum gripper fraction")
        self.closed = _vector(closed_positions, 2, "closed_positions")
        self.opened = _vector(open_positions, 2, "open_positions")
        self.strokes = tuple(b-a for a, b in zip(self.closed, self.opened, strict=True))
        if min(self.strokes) <= 0:
            raise ValueError("finger opening ranges must be positive")
        self.object_width_m = _number(object_width_m, "object_width_m")
        if not 0 < self.object_width_m < sum(self.opened):
            raise ValueError("object width must fit within the open fingers")
        self.phase = "close_gripper"
        self.status = "running"
        self.reason = None
        self.fraction = 1.
        self.lift_offset_m = 0.
        self.started = self.last_time = None
        self.stable_since = self.missing_since = None
        self.pause_started = self.recovery_since = None
        self.contact_seen = False
        self.approach_seen = False
        self.lift_origin = self.grasp_offset = None
        self.metrics: dict[str, Any] = {}

    def _fail(self, reason):
        self.status, self.reason = "failed", reason

    def _transition(self, phase, time_s):
        self.phase, self.started = phase, time_s
        self.stable_since = self.missing_since = None

    def _stable(self, condition, time_s, duration):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = time_s
        return time_s-self.stable_since >= duration-1e-9

    def _target(self, forces, positions, dt, force_target):
        cfg = self.config
        # Decelerate BEFORE first impact. At fast free closure, a damped drive
        # can be almost a millimetre behind its target when contact is reported.
        # Slowing only afterwards retains that stored compression demand.
        commanded_aperture = sum(self.closed)+self.fraction*sum(self.strokes)
        self.approach_seen |= min(sum(positions), commanded_aperture) <= self.object_width_m+2*cfg.approach_clearance_m
        speed = (cfg.contact_close_speed_mps if self.contact_seen else
                 cfg.approach_close_speed_mps if self.approach_seen else cfg.free_close_speed_mps)
        delta = speed*dt/max(self.strokes)
        if max(forces) > force_target+cfg.force_deadband_N:
            self.fraction = min(1., self.fraction+delta)
        elif min(forces) < force_target-cfg.force_deadband_N:
            lower = self.minimum_fraction
            if self.contact_seen:
                # Bound drive compression against the actual finger positions.
                lower = max(lower, *( (p-c-cfg.max_compression_m)/s
                                      for p, c, s in zip(positions, self.closed, self.strokes, strict=True)))
            # A newly observed bound cannot cause an abrupt opening command.
            desired = max(lower, self.fraction-delta)
            self.fraction = min(1., max(self.minimum_fraction, min(desired, self.fraction+delta)))

    def update(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.status != "running":
            return self.decision()
        try:
            self._update(sample)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self._fail(f"invalid_grasp_measurement:{exc}")
        return self.decision()

    def _update(self, sample):
        cfg = self.config
        now = _number(sample["time_s"], "time_s")
        dt = 0. if self.last_time is None else now-self.last_time
        if now < 0 or (self.last_time is not None and not math.isclose(dt, self.dt, rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError("grasp measurements must be consecutive physics steps")
        if sample.get("grasp_contact_coverage_complete") is not True:
            raise ValueError("complete finger/table contact coverage is required")
        peg = sample["live_peg"]
        position = _vector(peg["position"], 3, "peg position")
        tcp = _vector(sample["end_effector_position_m"], 3, "TCP position")
        orientation = _vector(sample["end_effector_orientation_wxyz"], 4, "TCP orientation")
        offset = _local_offset(position, tcp, orientation)
        linear = math.hypot(*_vector(peg["linear_velocity_mps"], 3, "linear velocity"))
        angular = math.hypot(*_vector(peg["angular_velocity_rps"], 3, "angular velocity"))
        fingers = _vector(sample["finger_positions_m"], 2, "finger positions")
        if any(p < low-1e-5 or p > high+1e-5
               for p, low, high in zip(fingers, self.closed, self.opened, strict=True)):
            raise ValueError("measured finger position is outside runtime limits")
        forces = [0., 0.]
        table_force = 0.
        contacts = sample["shape_contacts"]
        if not isinstance(contacts, list):
            raise ValueError("shape contacts must be a list")
        for contact in contacts:
            force = _number(contact["normal_force_N"], "contact force")
            path = contact["collider"]
            if force < 0 or not isinstance(path, str):
                raise ValueError("invalid contact force or collider")
            for index, token in enumerate(("panda_leftfinger", "panda_rightfinger")):
                if token in path.split("/"):
                    forces[index] += force
            if "table_1" in path.split("/"):
                table_force += force
        self.last_time = now
        if self.started is None:
            self.started = now
        bilateral = min(forces) >= cfg.contact_force_N
        self.contact_seen |= max(forces) >= cfg.contact_force_N
        stable_motion = linear <= cfg.stable_linear_speed_mps and angular <= cfg.stable_angular_speed_rps
        slip = math.dist(offset, self.grasp_offset) if self.grasp_offset is not None else 0.
        rise = position[2]-self.lift_origin if self.lift_origin is not None else 0.
        self.metrics = {"left_force_N": forces[0], "right_force_N": forces[1],
                        "table_force_N": table_force, "bilateral_contact": bilateral,
                        "linear_speed_mps": linear, "angular_speed_rps": angular,
                        "relative_slip_m": slip, "measured_rise_m": rise,
                        "finger_positions_m": list(fingers)}
        if angular > cfg.stop_angular_speed_rps:
            self._fail("grasp_angular_speed_protection")
            return
        if max(forces) > cfg.max_finger_force_N:
            self._fail("grasp_force_protection")
            return
        if now-self.started >= cfg.stage_timeout_s:
            self._fail(f"grasp_stage_timeout:{self.phase}")
            return
        if self.grasp_offset is not None:
            if slip > cfg.max_slip_m:
                self._fail("grasp_relative_slip")
                return
            if bilateral:
                self.missing_since = None
            elif self.missing_since is None:
                self.missing_since = now
            elif now-self.missing_since >= cfg.contact_loss_timeout_s-1e-9:
                self._fail("grasp_contact_lost")
                return

        if angular > cfg.pause_angular_speed_rps or self.pause_started is not None:
            # Hold both current drive references before adding more closure or
            # lift. Never filter the velocity or turn a protection into success.
            self.stable_since = None
            if self.pause_started is None:
                self.pause_started = now
            # A small, bounded opening relieves the contact constraint. This
            # is a recovery attempt; it never erases the measured spike or
            # changes the frozen task acceptance thresholds.
            self.fraction = min(1., self.fraction
                                + cfg.recovery_open_speed_mps*self.dt/max(self.strokes))
            if now-self.pause_started >= cfg.recovery_timeout_s:
                self._fail("grasp_velocity_recovery_timeout")
                return
            if stable_motion and bilateral:
                if self.recovery_since is None:
                    self.recovery_since = now
                if now-self.recovery_since >= cfg.recovery_duration_s-1e-9:
                    self.pause_started = self.recovery_since = None
            else:
                self.recovery_since = None
            return

        if self.phase == "close_gripper":
            self._target(forces, fingers, dt, cfg.table_grip_force_N)
            ready = (bilateral and stable_motion
                     and min(forces) >= cfg.table_grip_force_N-cfg.force_deadband_N
                     and max(forces) <= cfg.table_grip_force_N+cfg.force_deadband_N)
            if self._stable(ready, now, cfg.stable_duration_s):
                self.grasp_offset, self.lift_origin = offset, position[2]
                self._transition("light_grasp_clear_table", now)
        elif self.phase == "light_grasp_clear_table":
            clear = bilateral and table_force <= cfg.table_clear_force_N and rise >= cfg.clear_height_m
            if self._stable(clear and stable_motion, now, cfg.clear_duration_s):
                self._transition("increase_grip_off_table", now)
            elif bilateral and not clear:
                self.lift_offset_m = min(cfg.max_lift_m, self.lift_offset_m+cfg.lift_speed_mps*dt)
            # Hold the established grip; do not add pressure while on the table.
        elif self.phase in {"increase_grip_off_table", "grasp_stable"}:
            clear = table_force <= cfg.table_clear_force_N and rise >= cfg.clear_height_m
            if not clear:
                self._fail("grasp_table_clearance_lost")
                return
            if bilateral:
                self._target(forces, fingers, dt, cfg.airborne_grip_force_N)
            ready = (bilateral and stable_motion
                     and min(forces) >= cfg.airborne_grip_force_N-cfg.force_deadband_N
                     and max(forces) <= cfg.airborne_grip_force_N+cfg.force_deadband_N)
            if self._stable(ready, now, cfg.stable_duration_s):
                if self.phase == "increase_grip_off_table":
                    self._transition("grasp_stable", now)
                else:
                    self.status = "ready"

    def decision(self) -> dict[str, Any]:
        paused = self.status == "running" and self.pause_started is not None
        return {"phase": self.phase, "status": self.status,
                "reason": "grasp_waiting_for_velocity_settle" if paused else self.reason,
                "paused": paused,
                "gripper_open_fraction": self.fraction, "lift_offset_m": self.lift_offset_m,
                "metrics": dict(self.metrics)}
