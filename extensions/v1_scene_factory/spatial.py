"""Conservative spatial contracts shared by generation and independent validation."""

from __future__ import annotations

from .geometry import relative_box, rotated_half_extents


def blocks_clearance(item, zone, tolerance=0.003):
    # Any geometry intersecting walking height obstructs a reserved corridor.
    hx, hy, hz = rotated_half_extents(item.bbox_m, item.pose)
    bottom = item.pose.position[2] - hz
    if bottom >= 2.0:
        return False
    return (
        abs(item.pose.position[0] - zone.center[0]) < hx + zone.size[0] / 2 - tolerance
        and abs(item.pose.position[1] - zone.center[1]) < hy + zone.size[1] / 2 - tolerance
    )


def inside_region(item, parent, region, tolerance=0.003):
    local, half_extents = relative_box(item, parent)
    offsets = tuple(local[i]-region.center[i] for i in range(3))
    return all(
        abs(offset) + half <= size / 2 + tolerance
        for offset, half, size in zip(offsets, half_extents, region.size)
    )


def support_interior(registry, item, objects):
    if not item.support or item.support == "floor":
        return None
    target, _, name = item.support.partition(":")
    parent = objects.get(target)
    if parent is None:
        return None
    region = next(
        (r for r in registry.get(parent.asset_id).interior_regions if r.region_id == name), None
    )
    return (parent, region) if region is not None else None
