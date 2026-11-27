import math
from dataclasses import replace

import pytest

from scene_factory.geometry import objects_overlap, relative_box, rotated_half_extents
from scene_factory.models import PlacedObject, Pose
from scene_factory.pose_migration import upgrade_scene_poses


def box(position=(0, 0, 0), size=(1, 1, 1), q=None):
    return PlacedObject("box", "box", "box", size, Pose(position, orientation_wxyz=q), True, None)


def test_legacy_migration_preserves_yaw_transform_and_is_idempotent():
    scene = {"scene_id": "test", "objects": [{"object_id": "a", "pose": {"position": [1, 2, 3], "yaw_deg": 90}}]}
    result = upgrade_scene_poses(scene)
    assert "orientation_wxyz" not in scene["objects"][0]["pose"]
    pose = Pose.from_dict(result["objects"][0]["pose"])
    assert pose.effective_orientation_wxyz == pytest.approx((math.sqrt(.5), 0, 0, math.sqrt(.5)))
    assert upgrade_scene_poses(result) == result


@pytest.mark.parametrize("q", [[1, 1, 0, 0], [0, 0, 0, 0], [True, 0, 0, 0], [float("nan"), 0, 0, 0]])
def test_invalid_quaternion_rejected(q):
    with pytest.raises(ValueError):
        Pose.from_dict({"position": [0, 0, 0], "orientation_wxyz": q})


def test_conflicting_legacy_yaw_rejected():
    with pytest.raises(ValueError, match="conflicts"):
        Pose.from_dict({"position": [0, 0, 0], "yaw_deg": 90, "orientation_wxyz": [1, 0, 0, 0]})


def test_direct_quaternion_constructor_updates_upright_geometry():
    q = (math.sqrt(.5), 0, 0, math.sqrt(.5))
    rotated = box(size=(2, .2, .2), q=q)
    assert rotated.pose.yaw_deg == pytest.approx(90)
    assert objects_overlap(rotated, box((0, .8, 0), (.2, .2, .2)))
    assert not objects_overlap(rotated, box((.8, 0, 0), (.2, .2, .2)))
    with pytest.raises(ValueError, match="conflicts"):
        Pose((0, 0, 0), yaw_deg=45, orientation_wxyz=q)


def test_tilted_extents_and_overlap_use_all_three_axes():
    q = (math.sqrt(.5), 0, math.sqrt(.5), 0)
    tilted = box(size=(.2, .2, 2), q=q)
    assert rotated_half_extents(tilted.bbox_m, tilted.pose) == pytest.approx((1, .1, .1))
    assert objects_overlap(tilted, box((.8, 0, 0), (.2, .2, .2)))
    assert not objects_overlap(tilted, box((0, 0, .8), (.2, .2, .2)))


def test_relative_frame_restores_identically_tilted_boxes():
    q = (math.sqrt(.5), math.sqrt(.5), 0, 0)
    parent = box(q=q)
    child = replace(box((0, -1, 0), (.2, .4, .6), q=q), object_id="child")
    center, half = relative_box(child, parent)
    assert center == pytest.approx((0, 0, 1))
    assert half == pytest.approx((.1, .2, .3))
