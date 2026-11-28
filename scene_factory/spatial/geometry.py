"""Pure-XML SE(3) kinematic solver, 3D OBB/SAT collision checker, and S-Space frame projection.

Resolves nested parent-child `<body>` frames, rotations (`quat` / `euler`), and
multi-geom assemblies directly from MuJoCo XML without rendering or launching a
physics engine, with optional MuJoCo cross-check for ground-truth validation.
"""

from __future__ import annotations

import importlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

_EYE: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _vec3(text: str | None, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if not text or not text.strip():
        return default
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 floats, got {text!r}")
    return (parts[0], parts[1], parts[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (_dot(m[0], v), _dot(m[1], v), _dot(m[2], v))


def _mat_mul(a: Mat3, b: Mat3) -> Mat3:
    bt = (
        (b[0][0], b[1][0], b[2][0]),
        (b[0][1], b[1][1], b[2][1]),
        (b[0][2], b[1][2], b[2][2]),
    )
    return (
        (_dot(a[0], bt[0]), _dot(a[0], bt[1]), _dot(a[0], bt[2])),
        (_dot(a[1], bt[0]), _dot(a[1], bt[1]), _dot(a[1], bt[2])),
        (_dot(a[2], bt[0]), _dot(a[2], bt[1]), _dot(a[2], bt[2])),
    )


def _mat_cols(m: Mat3) -> tuple[Vec3, Vec3, Vec3]:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def quat_wxyz_to_mat3(quat: tuple[float, float, float, float]) -> Mat3:
    w, x, y, z = quat
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0:
        return _EYE
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def euler_xyz_to_mat3(euler: Vec3, *, degrees: bool = True) -> Mat3:
    rx, ry, rz = (math.radians(v) for v in euler) if degrees else euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx: Mat3 = ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx))
    my: Mat3 = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    mz: Mat3 = ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0))
    # MuJoCo default eulerseq="xyz" composes intrinsic X -> Y -> Z (R = Rx @ Ry @ Rz).
    return _mat_mul(_mat_mul(mx, my), mz)


def _element_rotation(elem: ET.Element, *, angle_degrees: bool) -> Mat3:
    if "quat" in elem.attrib:
        vals = [float(v) for v in elem.attrib["quat"].split()]
        if len(vals) != 4:
            raise ValueError(f"invalid quat: {elem.attrib['quat']!r}")
        return quat_wxyz_to_mat3((vals[0], vals[1], vals[2], vals[3]))
    if "euler" in elem.attrib:
        return euler_xyz_to_mat3(_vec3(elem.attrib["euler"]), degrees=angle_degrees)
    if "axisangle" in elem.attrib:
        vals = [float(v) for v in elem.attrib["axisangle"].split()]
        if len(vals) != 4:
            raise ValueError(f"invalid axisangle: {elem.attrib['axisangle']!r}")
        ax = (vals[0], vals[1], vals[2])
        an = _norm(ax)
        if an <= 0:
            return _EYE
        theta = math.radians(vals[3]) if angle_degrees else vals[3]
        half = theta / 2.0
        s = math.sin(half) / an
        return quat_wxyz_to_mat3((math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s))
    return _EYE


@dataclass(frozen=True)
class GeomOBB:
    """World-frame oriented bounding box for one MuJoCo `<geom>`."""

    name: str
    body_name: str
    geom_type: str
    center: Vec3
    rotation: Mat3
    half_extents: Vec3

    def aabb(self) -> tuple[Vec3, Vec3]:
        hx, hy, hz = self.half_extents
        ax, ay, az = _mat_cols(self.rotation)
        ex = abs(ax[0]) * hx + abs(ay[0]) * hy + abs(az[0]) * hz
        ey = abs(ax[1]) * hx + abs(ay[1]) * hy + abs(az[1]) * hz
        ez = abs(ax[2]) * hx + abs(ay[2]) * hy + abs(az[2]) * hz
        cx, cy, cz = self.center
        return ((cx - ex, cy - ey, cz - ez), (cx + ex, cy + ey, cz + ez))


@dataclass(frozen=True)
class BodyGeometry:
    """Resolved world-frame geometry and nesting metadata for one `<body>`."""

    name: str
    parent: str | None
    depth: int
    local_pos: Vec3
    world_pos: Vec3
    world_center: Vec3
    world_rotation: Mat3
    aabb_min: Vec3
    aabb_max: Vec3
    aabb_half_extents: Vec3
    dynamic: bool
    geoms: tuple[GeomOBB, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "depth": self.depth,
            "local_pos_m": [round(v, 6) for v in self.local_pos],
            "world_pos_m": [round(v, 6) for v in self.world_pos],
            "world_center_m": [round(v, 6) for v in self.world_center],
            "aabb_min_m": [round(v, 6) for v in self.aabb_min],
            "aabb_max_m": [round(v, 6) for v in self.aabb_max],
            "aabb_half_extents_m": [round(v, 6) for v in self.aabb_half_extents],
            "dynamic": self.dynamic,
            "geom_count": len(self.geoms),
        }


@dataclass(frozen=True)
class PairCollision:
    """Analytical & optional MuJoCo collision verdict between two bodies."""

    body_a: str
    body_b: str
    signed_distance_m: float  # negative = penetration, 0 = touching, positive = separated
    penetration_m: float
    status: str  # "colliding" | "touching" | "separated"
    separation_axis: Vec3
    mujoco_signed_distance_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_a": self.body_a,
            "body_b": self.body_b,
            "signed_distance_m": round(self.signed_distance_m, 6),
            "penetration_m": round(self.penetration_m, 6),
            "status": self.status,
            "separation_axis": [round(v, 5) for v in self.separation_axis],
            "mujoco_signed_distance_m": (
                None if self.mujoco_signed_distance_m is None else round(self.mujoco_signed_distance_m, 6)
            ),
        }


@dataclass
class SpatialAnalysis:
    """Complete zero-render spatial geometry and assembly validity report."""

    bodies: dict[str, BodyGeometry]
    pairs: list[PairCollision]
    colliding_pairs: list[PairCollision]
    floating_bodies: list[dict[str, Any]]
    supported_bodies: list[dict[str, Any]]
    camera_sspace_coords: dict[str, dict[str, float]] = field(default_factory=dict)
    assembly_valid: bool = True
    mujoco_crosschecked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_count": len(self.bodies),
            "bodies": {k: v.to_dict() for k, v in self.bodies.items()},
            "colliding_pairs": [p.to_dict() for p in self.colliding_pairs],
            "pair_count": len(self.pairs),
            "floating_bodies": self.floating_bodies,
            "supported_bodies": self.supported_bodies,
            "camera_sspace_coords": self.camera_sspace_coords,
            "assembly_valid": self.assembly_valid,
            "mujoco_crosschecked": self.mujoco_crosschecked,
        }


def obb_signed_distance(a: GeomOBB, b: GeomOBB) -> tuple[float, Vec3]:
    """3D Separating Axis Theorem (SAT) signed distance between two OBBs.

    Returns `(signed_distance_m, unit_axis)` where negative distance means
    minimum penetration depth along the MTV axis, and positive distance is a
    lower bound on separating gap (exact for face-aligned separations).
    """
    ax = _mat_cols(a.rotation)
    bx = _mat_cols(b.rotation)
    ha = a.half_extents
    hb = b.half_extents
    delta = _sub(b.center, a.center)

    candidates: list[Vec3] = [*ax, *bx]
    for u in ax:
        for v in bx:
            c = _cross(u, v)
            if _norm(c) > 1e-6:
                candidates.append(c)

    max_sep = -1e18
    best_axis: Vec3 = (0.0, 0.0, 1.0)
    for axis in candidates:
        ln = _norm(axis)
        if ln <= 1e-9:
            continue
        n = (axis[0] / ln, axis[1] / ln, axis[2] / ln)
        proj_a = abs(_dot(ax[0], n)) * ha[0] + abs(_dot(ax[1], n)) * ha[1] + abs(_dot(ax[2], n)) * ha[2]
        proj_b = abs(_dot(bx[0], n)) * hb[0] + abs(_dot(bx[1], n)) * hb[1] + abs(_dot(bx[2], n)) * hb[2]
        dist_center = abs(_dot(delta, n))
        sep = dist_center - (proj_a + proj_b)
        if sep > max_sep:
            max_sep = sep
            sign = 1.0 if _dot(delta, n) >= 0.0 else -1.0
            best_axis = (n[0] * sign, n[1] * sign, n[2] * sign)

    return max_sep, best_axis


def _mesh_bounds_map(root: ET.Element) -> dict[str, tuple[Vec3, Vec3]]:
    """Compute local-frame (center, half_extents) for inline `<mesh vertex="..."/>`."""
    bounds: dict[str, tuple[Vec3, Vec3]] = {}
    asset = root.find("asset")
    if asset is None:
        return bounds
    for mesh in asset.findall("mesh"):
        name = mesh.attrib.get("name", "")
        raw_v = mesh.attrib.get("vertex", "")
        if not name or not raw_v:
            continue
        nums = [float(v) for v in raw_v.split()]
        if len(nums) < 3 or len(nums) % 3 != 0:
            continue
        xs, ys, zs = nums[0::3], nums[1::3], nums[2::3]
        lo = (min(xs), min(ys), min(zs))
        hi = (max(xs), max(ys), max(zs))
        center = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0)
        half = (
            max((hi[0] - lo[0]) / 2.0, 1e-5),
            max((hi[1] - lo[1]) / 2.0, 1e-5),
            max((hi[2] - lo[2]) / 2.0, 1e-5),
        )
        bounds[name] = (center, half)
    return bounds


def _geom_local_box(
    geom: ET.Element, mesh_bounds: dict[str, tuple[Vec3, Vec3]]
) -> tuple[Vec3, Vec3] | None:
    gtype = geom.attrib.get("type", "sphere" if "size" in geom.attrib and len(geom.attrib["size"].split()) == 1 else "box").lower()
    if gtype == "plane":
        return None
    if gtype == "mesh":
        mname = geom.attrib.get("mesh", "")
        return mesh_bounds.get(mname)
    sizes = [float(v) for v in geom.attrib.get("size", "").split()]
    if gtype == "box" and len(sizes) >= 3:
        return (0.0, 0.0, 0.0), (sizes[0], sizes[1], sizes[2])
    if gtype == "sphere" and len(sizes) >= 1:
        r = sizes[0]
        return (0.0, 0.0, 0.0), (r, r, r)
    if gtype == "cylinder" and len(sizes) >= 2:
        r, hh = sizes[0], sizes[1]
        return (0.0, 0.0, 0.0), (r, r, hh)
    if gtype == "capsule" and len(sizes) >= 2:
        r, hh = sizes[0], sizes[1]
        return (0.0, 0.0, 0.0), (r, r, hh + r)
    if len(sizes) >= 3:
        return (0.0, 0.0, 0.0), (sizes[0], sizes[1], sizes[2])
    return None


def _camera_frame(root: ET.Element) -> tuple[Vec3, Vec3, Vec3, Vec3]:
    """Return `(cam_pos, right, down, forward)` aligned with S-Space positive endpoints."""
    world = root.find("worldbody")
    cam_elem = None
    if world is not None:
        for cam in world.findall("camera"):
            if cam.attrib.get("name") == "cam_front":
                cam_elem = cam
                break
        if cam_elem is None:
            cam_elem = world.find("camera")
    if cam_elem is not None and "xyaxes" in cam_elem.attrib:
        pos = _vec3(cam_elem.attrib.get("pos"), (0.0, -2.6, 1.4))
        vals = [float(v) for v in cam_elem.attrib["xyaxes"].split()]
        if len(vals) == 6:
            right = (vals[0], vals[1], vals[2])
            rn = _norm(right) or 1.0
            right = (right[0] / rn, right[1] / rn, right[2] / rn)
            up = (vals[3], vals[4], vals[5])
            un = _norm(up) or 1.0
            up = (up[0] / un, up[1] / un, up[2] / un)
            # In MuJoCo cameras, -Z is the forward viewing axis = up x right.
            forward = _cross(up, right)
            fn = _norm(forward) or 1.0
            forward = (forward[0] / fn, forward[1] / fn, forward[2] / fn)
            down = (-up[0], -up[1], -up[2])
            return pos, right, down, forward
    # Default front camera at (0, -2.6, 1.4) looking toward +Y with +Z up.
    return (0.0, -2.6, 1.4), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)


def _mujoco_pair_distances(xml: str) -> dict[tuple[str, str], float] | None:
    try:
        mujoco = importlib.import_module("mujoco")
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception:
        return None
    dists: dict[tuple[str, str], float] = {}
    for i in range(int(data.ncon)):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[int(con.geom1)])
        b2 = int(model.geom_bodyid[int(con.geom2)])
        if b1 == b2 or b1 == 0 or b2 == 0:
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        if not n1 or not n2:
            continue
        key = (n1, n2) if n1 < n2 else (n2, n1)
        d = float(con.dist)
        if key not in dists or d < dists[key]:
            dists[key] = d
    return dists


def analyze_xml_spatial(
    xml: str,
    *,
    collision_tol_m: float = 0.003,
    touch_tol_m: float = 0.008,
    floating_tol_m: float = 0.015,
    crosscheck_mujoco: bool = False,
) -> SpatialAnalysis:
    """Resolve full 3D coordinates, nested transforms, and collisions directly from XML."""
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    angle_degrees = True
    if compiler is not None and compiler.attrib.get("angle", "degree").lower() == "radian":
        angle_degrees = False

    mesh_bounds = _mesh_bounds_map(root)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MuJoCo XML has no <worldbody>")

    floor_z: float | None = None
    for geom in world.findall("geom"):
        if geom.attrib.get("type", "").lower() == "plane":
            floor_z = _vec3(geom.attrib.get("pos"))[2]
            break

    bodies: dict[str, BodyGeometry] = {}

    def _walk(
        node: ET.Element,
        parent_name: str | None,
        parent_pos: Vec3,
        parent_rot: Mat3,
        depth: int,
        counter: list[int],
    ) -> None:
        for child in node.findall("body"):
            counter[0] += 1
            bname = child.attrib.get("name") or f"body_{counter[0]}"
            lpos = _vec3(child.attrib.get("pos"))
            lrot = _element_rotation(child, angle_degrees=angle_degrees)
            wpos = _add(parent_pos, _mat_vec(parent_rot, lpos))
            wrot = _mat_mul(parent_rot, lrot)
            dynamic = child.find("freejoint") is not None or child.find("joint") is not None

            geoms: list[GeomOBB] = []
            for gi, geom in enumerate(child.findall("geom")):
                if geom.attrib.get("contype") == "0" and geom.attrib.get("conaffinity") == "0":
                    continue
                local_box = _geom_local_box(geom, mesh_bounds)
                if local_box is None:
                    continue
                offset, half = local_box
                gpos_local = _vec3(geom.attrib.get("pos"))
                grot_local = _element_rotation(geom, angle_degrees=angle_degrees)
                grot_world = _mat_mul(wrot, grot_local)
                gcenter = _add(wpos, _mat_vec(wrot, _add(gpos_local, _mat_vec(grot_local, offset))))
                geoms.append(
                    GeomOBB(
                        name=geom.attrib.get("name") or f"{bname}_geom_{gi}",
                        body_name=bname,
                        geom_type=geom.attrib.get("type", "box").lower(),
                        center=gcenter,
                        rotation=grot_world,
                        half_extents=half,
                    )
                )

            if geoms:
                aabbs = [g.aabb() for g in geoms]
                amin = (
                    min(box[0][0] for box in aabbs),
                    min(box[0][1] for box in aabbs),
                    min(box[0][2] for box in aabbs),
                )
                amax = (
                    max(box[1][0] for box in aabbs),
                    max(box[1][1] for box in aabbs),
                    max(box[1][2] for box in aabbs),
                )
                wcenter = (
                    (amin[0] + amax[0]) / 2.0,
                    (amin[1] + amax[1]) / 2.0,
                    (amin[2] + amax[2]) / 2.0,
                )
                ahalf = (
                    (amax[0] - amin[0]) / 2.0,
                    (amax[1] - amin[1]) / 2.0,
                    (amax[2] - amin[2]) / 2.0,
                )
            else:
                amin = amax = wcenter = wpos
                ahalf = (0.0, 0.0, 0.0)

            bodies[bname] = BodyGeometry(
                name=bname,
                parent=parent_name,
                depth=depth,
                local_pos=lpos,
                world_pos=wpos,
                world_center=wcenter,
                world_rotation=wrot,
                aabb_min=amin,
                aabb_max=amax,
                aabb_half_extents=ahalf,
                dynamic=dynamic,
                geoms=tuple(geoms),
            )
            _walk(child, bname, wpos, wrot, depth + 1, counter)

    _walk(world, None, (0.0, 0.0, 0.0), _EYE, 0, [0])

    mj_dists = _mujoco_pair_distances(xml) if crosscheck_mujoco else None

    pairs: list[PairCollision] = []
    colliding: list[PairCollision] = []
    for name_a, name_b in combinations(sorted(bodies), 2):
        ba, bb = bodies[name_a], bodies[name_b]
        if not ba.geoms or not bb.geoms:
            continue
        # Skip direct parent-child frame links only if they share a rigid weld (no freejoint).
        if (ba.parent == name_b and not ba.dynamic) or (bb.parent == name_a and not bb.dynamic):
            pass  # still measure! Nested assembly checks need parent-child overlap detection too.
        best_sd = 1e18
        best_axis: Vec3 = (0.0, 0.0, 1.0)
        for ga in ba.geoms:
            for gb in bb.geoms:
                sd, axis = obb_signed_distance(ga, gb)
                if sd < best_sd:
                    best_sd = sd
                    best_axis = axis
        pen = max(0.0, -best_sd)
        if best_sd < -collision_tol_m:
            status = "colliding"
        elif best_sd <= touch_tol_m:
            status = "touching"
        else:
            status = "separated"
        mj_sd = mj_dists.get((name_a, name_b)) if mj_dists is not None else None
        pair = PairCollision(
            body_a=name_a,
            body_b=name_b,
            signed_distance_m=best_sd,
            penetration_m=pen,
            status=status,
            separation_axis=best_axis,
            mujoco_signed_distance_m=mj_sd,
        )
        pairs.append(pair)
        if status == "colliding":
            colliding.append(pair)

    # Support / floating analysis for dynamic (freejoint) bodies.
    floating: list[dict[str, Any]] = []
    supported: list[dict[str, Any]] = []
    for bname, body in sorted(bodies.items()):
        if not body.dynamic or not body.geoms:
            continue
        cx, cy, _ = body.world_center
        bottom_z = body.aabb_min[2]
        best_gap = 1e18
        best_supporter = "none"
        if floor_z is not None:
            gap_floor = bottom_z - floor_z
            if gap_floor >= -collision_tol_m and gap_floor < best_gap:
                best_gap = gap_floor
                best_supporter = "floor"
        for other_name, other in bodies.items():
            if other_name == bname or not other.geoms:
                continue
            for og in other.geoms:
                o_min, o_max = og.aabb()
                if o_min[0] <= cx <= o_max[0] and o_min[1] <= cy <= o_max[1]:
                    gap = bottom_z - o_max[2]
                    if gap >= -collision_tol_m and gap < best_gap:
                        best_gap = gap
                        best_supporter = other_name
        entry = {
            "body": bname,
            "supporter": best_supporter,
            "clearance_m": round(best_gap if best_gap < 1e17 else -1.0, 6),
        }
        if best_gap > floating_tol_m:
            floating.append(entry)
        else:
            supported.append(entry)

    # Project every body center into S-Space camera coordinates (right, below, close).
    cam_pos, cam_right, cam_down, cam_forward = _camera_frame(root)
    sspace_coords: dict[str, dict[str, float]] = {}
    for bname, body in sorted(bodies.items()):
        rel = _sub(body.world_center, cam_pos)
        depth = _dot(rel, cam_forward)
        sspace_coords[bname] = {
            "horizontal": round(_dot(rel, cam_right), 6),  # + right, - left
            "vertical": round(_dot(rel, cam_down), 6),     # + below, - above
            "distance": round(-depth, 6),                  # + close, - far (matches S-Space)
        }

    return SpatialAnalysis(
        bodies=bodies,
        pairs=pairs,
        colliding_pairs=colliding,
        floating_bodies=floating,
        supported_bodies=supported,
        camera_sspace_coords=sspace_coords,
        assembly_valid=(not colliding and not floating),
        mujoco_crosschecked=(mj_dists is not None),
    )
