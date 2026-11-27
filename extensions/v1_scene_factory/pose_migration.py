"""Explicit upgrade of planar compiled layouts to a full SE(3) pose contract."""
from __future__ import annotations

import copy
from typing import Any

from .models import Pose
from .session_contracts import finite_json


def upgrade_scene_poses(scene: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(finite_json(scene))
    if not isinstance(result, dict) or not isinstance(result.get("objects"), list):
        raise ValueError("scene must contain objects")
    if result.get("schema_version") not in {None, "scene_factory.compiled_scene.v1", "scene_factory.compiled_scene.v2"}:
        raise ValueError("unsupported compiled scene schema_version")
    ids = []
    for item in result["objects"]:
        if not isinstance(item, dict) or not isinstance(item.get("object_id"), str):
            raise ValueError("each scene instance requires object_id")
        ids.append(item["object_id"])
        pose = Pose.from_dict(item["pose"])
        item["pose"] = {"position": list(pose.position),
                        "orientation_wxyz": list(pose.effective_orientation_wxyz),
                        "yaw_deg": pose.yaw_deg}
    if len(ids) != len(set(ids)):
        raise ValueError("scene object IDs must be unique")
    result["schema_version"] = "scene_factory.compiled_scene.v2"
    return result
