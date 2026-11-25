"""Finite, bounded Franka position targets shared by API and Isaac adapter."""
from __future__ import annotations

import math
from typing import Any

ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


def validate_joint_position_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"arm_positions_rad", "gripper_open_fraction"}:
        raise ValueError("joint_position requires arm_positions_rad and gripper_open_fraction")
    arm = payload["arm_positions_rad"]
    if not isinstance(arm, list) or len(arm) != 7:
        raise ValueError("arm_positions_rad must contain seven finite joint positions")
    values = [*arm, payload["gripper_open_fraction"]]
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("joint position targets must be finite numbers")
        try:
            if not math.isfinite(float(value)):
                raise ValueError("joint position targets must be finite numbers")
        except OverflowError as exc:
            raise ValueError("joint position target exceeds floating point range") from exc
    fraction = float(payload["gripper_open_fraction"])
    if not 0 <= fraction <= 1:
        raise ValueError("gripper_open_fraction must be in [0, 1]")
    return {"arm_positions_rad": [float(v) for v in arm], "gripper_open_fraction": fraction}


def resolve_joint_targets(payload: Any, dof_names: list[str], limits: Any,
                          gripper_config: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Validate every target before any action is submitted to a simulator."""
    payload = validate_joint_position_payload(payload)
    if any(name not in dof_names for name in ARM_JOINT_NAMES):
        raise ValueError("runtime does not expose all seven Franka arm joints")
    indices = [dof_names.index(name) for name in ARM_JOINT_NAMES]
    positions = payload["arm_positions_rad"][:]
    fraction = payload["gripper_open_fraction"]
    for index, low, high in zip(gripper_config["indices"], gripper_config["closed_positions"],
                                gripper_config["open_positions"], strict=True):
        indices.append(int(index))
        positions.append(float(low + fraction * (high - low)))
    if len(set(indices)) != len(indices) or len(indices) != 9:
        raise ValueError("runtime gripper joint mapping is invalid")
    for index, value in zip(indices, positions, strict=True):
        low, high = (float(v) for v in limits[index])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("runtime joint limits are invalid")
        if not low <= value <= high:
            raise ValueError(f"joint target outside runtime limits: {dof_names[index]}")
    return indices, positions
