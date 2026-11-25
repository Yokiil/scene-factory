from __future__ import annotations

import pytest

from tools.build_assembly_room_fixture import PROTECTED, intersects, plan_backgrounds


@pytest.mark.parametrize("count, dynamic", [(20, 6), (50, 16), (100, 32)])
def test_seeded_room_plan_preserves_protected_robot_and_assembly_volume(count, dynamic):
    rows = plan_backgrounds(count, 19)
    assert len(rows) == count-4
    assert sum(row["dynamic"] for row in rows) == dynamic
    assert len({row["object_id"] for row in rows}) == len(rows)
    for row in rows:
        assert not intersects(row["containment_cell"]["bounds_m"], PROTECTED)
        assert row["containment_cell"]["authored"] == row["dynamic"]
        assert row["physical_parameters"]["initial_velocity_mps"] == [0, 0, 0]
        assert all(dimension > 0 for dimension in row["bbox_m"])
        assert row["pose"]["position"][2] >= row["bbox_m"][2]/2


def test_room_sizes_share_the_same_prefix_and_seed_changes_actual_parameters():
    small = plan_backgrounds(20, 19)
    assert small == plan_backgrounds(100, 19)[:len(small)]
    assert small == plan_backgrounds(20, 19)
    assert small != plan_backgrounds(20, 20)


@pytest.mark.parametrize("count, seed", [(4, 0), (True, 0), (101, 0), (20, -1), (20, True), (20, 2**64)])
def test_invalid_room_dimensions_and_seed_fail_explicitly(count, seed):
    with pytest.raises(ValueError):
        plan_backgrounds(count, seed)
