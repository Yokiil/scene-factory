from __future__ import annotations

import math

from .models import PlacedObject, Pose, Vec2, Vec3


def rotate_xy(point: Vec2, yaw_deg: float) -> Vec2:
    angle = math.radians(yaw_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (cosine * point[0] - sine * point[1], sine * point[0] + cosine * point[1])


def inverse_rotate_xy(point: Vec2, yaw_deg: float) -> Vec2:
    return rotate_xy(point, -yaw_deg)


def rotated_half_extents_xy(bbox: Vec3, yaw_deg: float) -> Vec2:
    half_x, half_y = bbox[0] / 2.0, bbox[1] / 2.0
    angle = math.radians(yaw_deg)
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    return (cosine * half_x + sine * half_y, sine * half_x + cosine * half_y)


def z_interval(position_z: float, bbox_z: float) -> tuple[float, float]:
    return (position_z - bbox_z / 2.0, position_z + bbox_z / 2.0)


def rotation_matrix(pose: Pose) -> tuple[Vec3, Vec3, Vec3]:
    w, x, y, z = pose.effective_orientation_wxyz
    return ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def rotated_half_extents(bbox: Vec3, pose: Pose) -> Vec3:
    return tuple(sum(abs(row[j])*bbox[j]/2 for j in range(3)) for row in rotation_matrix(pose))


def relative_box(item: PlacedObject, parent: PlacedObject) -> tuple[Vec3, Vec3]:
    """Center and conservative extents expressed in the parent's local frame."""
    a, b = rotation_matrix(parent.pose), rotation_matrix(item.pose)
    delta = tuple(item.pose.position[i]-parent.pose.position[i] for i in range(3))
    center = tuple(sum(a[k][i]*delta[k] for k in range(3)) for i in range(3))
    relative = tuple(tuple(sum(a[k][i]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))
    half = tuple(sum(abs(relative[i][j])*item.bbox_m[j]/2 for j in range(3)) for i in range(3))
    return center, half


def _oriented_overlap(a: PlacedObject, b: PlacedObject, tolerance: float) -> bool:
    ra, rb = rotation_matrix(a.pose), rotation_matrix(b.pose)
    aa = [tuple(ra[i][j] for i in range(3)) for j in range(3)]
    bb = [tuple(rb[i][j] for i in range(3)) for j in range(3)]
    axes = aa+bb
    for u in aa:
        for v in bb:
            axes.append((u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]))
    delta = tuple(b.pose.position[i]-a.pose.position[i] for i in range(3))
    for axis in axes:
        length = math.sqrt(sum(v*v for v in axis))
        if length < 1e-10:
            continue
        axis = tuple(v/length for v in axis)
        separation = abs(sum(delta[i]*axis[i] for i in range(3)))
        first = sum(a.bbox_m[j]/2*abs(sum(aa[j][i]*axis[i] for i in range(3))) for j in range(3))
        second = sum(b.bbox_m[j]/2*abs(sum(bb[j][i]*axis[i] for i in range(3))) for j in range(3))
        if first+second-separation <= tolerance:
            return False
    return True


def objects_overlap(a: PlacedObject, b: PlacedObject, tolerance: float = 1e-5) -> bool:
    if not a.pose.is_upright or not b.pose.is_upright:
        return _oriented_overlap(a, b, tolerance)
    az0, az1 = z_interval(a.pose.position[2], a.bbox_m[2])
    bz0, bz1 = z_interval(b.pose.position[2], b.bbox_m[2])
    if min(az1, bz1) - max(az0, bz0) <= tolerance:
        return False
    ahx, ahy = rotated_half_extents_xy(a.bbox_m, a.pose.yaw_deg)
    bhx, bhy = rotated_half_extents_xy(b.bbox_m, b.pose.yaw_deg)
    dx = abs(a.pose.position[0] - b.pose.position[0])
    dy = abs(a.pose.position[1] - b.pose.position[1])
    return dx < ahx + bhx - tolerance and dy < ahy + bhy - tolerance


def distance_xy(a: Pose, b: Pose) -> float:
    return math.hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])

