"""Build an original three-tier dessert display as reusable SceneFactory assets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from build_kitchen_dining_assets import box, cylinder, glb_bytes, usd_text

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data" / "assets"

WOOD = (0.095, 0.050, 0.035)
WOOD_EDGE = (0.045, 0.025, 0.020)
PINK = (0.78, 0.24, 0.40)
PALE_PINK = (0.91, 0.54, 0.64)
CREAM = (0.94, 0.81, 0.66)
WHITE = (0.92, 0.89, 0.82)
RED = (0.72, 0.025, 0.035)
GREEN = (0.075, 0.30, 0.09)
GOLD = (0.58, 0.34, 0.16)
SEED = (0.96, 0.78, 0.41)


def source_provenance():
    """Record local authored source without inventing a published upstream URL."""
    sources = (Path(__file__), Path(__file__).with_name("build_kitchen_dining_assets.py"))
    hashes = {"tools/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    return {
        "source_url": "../../../../tools/build_dessert_display.py",
        "source_revision": "sha256:" + hashes["tools/build_dessert_display.py"],
        "source_revision_kind": "generator_content_sha256",
        "generator_sources_sha256": hashes,
    }


def moved(part, offset):
    name, vertices, faces, color = part
    return name, [tuple(v[i] + offset[i] for i in range(3)) for v in vertices], faces, color


def rotated_z(part, angle):
    name, vertices, faces, color = part
    c, s = math.cos(angle), math.sin(angle)
    return name, [(c * x - s * y, s * x + c * y, z) for x, y, z in vertices], faces, color


def fit_parts(parts, bbox):
    mins = [min(vertex[i] for _, vertices, _, _ in parts for vertex in vertices)
            for i in range(3)]
    maxs = [max(vertex[i] for _, vertices, _, _ in parts for vertex in vertices)
            for i in range(3)]
    scale = [bbox[i] / (maxs[i] - mins[i]) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2 for i in range(3)]
    return [(name, [tuple((vertex[i] - center[i]) * scale[i] for i in range(3))
                    for vertex in vertices], faces, color)
            for name, vertices, faces, color in parts]


def ellipsoid(name, radii, center, color, segments=12, rings=8):
    vertices = []
    for row in range(rings + 1):
        latitude = -math.pi / 2 + 0.0001 + (math.pi - 0.0002) * row / rings
        for col in range(segments):
            angle = 2 * math.pi * col / segments
            vertices.append((
                center[0] + radii[0] * math.cos(latitude) * math.cos(angle),
                center[1] + radii[1] * math.cos(latitude) * math.sin(angle),
                center[2] + radii[2] * math.sin(latitude),
            ))
    faces = []
    for row in range(rings):
        for col in range(segments):
            a = row * segments + col
            b = row * segments + (col + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    return name, vertices, faces, color


def frustum(name, bottom, top, z0, z1, color, segments=20):
    vertices = [
        (r * math.cos(2 * math.pi * i / segments),
         r * math.sin(2 * math.pi * i / segments), z)
        for z, r in ((z0, bottom), (z1, top))
        for i in range(segments)
    ]
    faces = [tuple(reversed(range(segments))), tuple(range(segments, 2 * segments))]
    faces += [
        (i, (i + 1) % segments, (i + 1) % segments + segments, i + segments)
        for i in range(segments)
    ]
    return name, vertices, faces, color


def fluted_cup(name, bottom, top, z0, z1, color, segments=48, flutes=16):
    vertices = [
        (r * (1 + 0.055 * math.cos(2 * math.pi * flutes * i / segments))
         * math.cos(2 * math.pi * i / segments),
         r * (1 + 0.055 * math.cos(2 * math.pi * flutes * i / segments))
         * math.sin(2 * math.pi * i / segments), z)
        for z, r in ((z0, bottom), (z0 + (z1 - z0) * 0.45, (bottom + top) / 2),
                     (z1, top))
        for i in range(segments)
    ]
    faces = [tuple(reversed(range(segments))),
             tuple(range(segments * 2, segments * 3))]
    faces += [
        (row * segments + i, row * segments + (i + 1) % segments,
         (row + 1) * segments + (i + 1) % segments, (row + 1) * segments + i)
        for row in range(2) for i in range(segments)
    ]
    return name, vertices, faces, color


def piped_cream(name, profile, color, segments=48, flutes=12):
    vertices = [
        (r * (1 + 0.10 * math.cos(flutes * angle + z * 15)) * math.cos(angle),
         r * (1 + 0.10 * math.cos(flutes * angle + z * 15)) * math.sin(angle), z)
        for z, r in profile
        for angle in (2 * math.pi * i / segments for i in range(segments))
    ]
    faces = [tuple(reversed(range(segments))),
             tuple(range((len(profile) - 1) * segments, len(profile) * segments))]
    faces += [
        (row * segments + i, row * segments + (i + 1) % segments,
         (row + 1) * segments + (i + 1) % segments, (row + 1) * segments + i)
        for row in range(len(profile) - 1) for i in range(segments)
    ]
    return name, vertices, faces, color


def rod(name, start, end, radius, color, segments=10):
    direction = tuple(end[i] - start[i] for i in range(3))
    length = math.sqrt(sum(v * v for v in direction))
    axis = tuple(v / length for v in direction)
    reference = (0, 1, 0) if abs(axis[1]) < 0.9 else (1, 0, 0)
    tangent = (axis[1] * reference[2] - axis[2] * reference[1],
               axis[2] * reference[0] - axis[0] * reference[2],
               axis[0] * reference[1] - axis[1] * reference[0])
    magnitude = math.sqrt(sum(v * v for v in tangent))
    tangent = tuple(v / magnitude for v in tangent)
    bitangent = (axis[1] * tangent[2] - axis[2] * tangent[1],
                 axis[2] * tangent[0] - axis[0] * tangent[2],
                 axis[0] * tangent[1] - axis[1] * tangent[0])
    vertices = [
        tuple(origin[k] + radius * (
            tangent[k] * math.cos(2 * math.pi * i / segments)
            + bitangent[k] * math.sin(2 * math.pi * i / segments)) for k in range(3))
        for origin in (start, end) for i in range(segments)
    ]
    faces = [tuple(reversed(range(segments))), tuple(range(segments, 2 * segments))]
    faces += [(i, (i + 1) % segments, (i + 1) % segments + segments, i + segments)
              for i in range(segments)]
    return name, vertices, faces, color


def handle_arch(name, half_width, top, bend_radius, tube_radius, color):
    """Sweep a round tube around the two quarter-circle shoulders of an inverted U."""
    shoulder = top - bend_radius
    path = []
    for side in (-1, 1):
        center = side * (half_width - bend_radius)
        for i in range(17):
            angle = (math.pi if side == -1 else math.pi / 2) - math.pi / 2 * i / 16
            path.append((center + bend_radius * math.cos(angle),
                         shoulder + bend_radius * math.sin(angle), angle))
    segments = 20
    vertices = [
        (x + tube_radius * math.cos(angle) * math.cos(theta),
         tube_radius * math.sin(theta),
         z + tube_radius * math.sin(angle) * math.cos(theta))
        for x, z, angle in path
        for theta in (2 * math.pi * i / segments for i in range(segments))
    ]
    # The profile's outward radial vector and Y vector face against the sweep,
    # so reverse the usual cylinder winding.
    faces = [tuple(range(segments)),
             tuple(reversed(range((len(path) - 1) * segments, len(path) * segments)))]
    faces += [
        (row * segments + i, (row + 1) * segments + i,
         (row + 1) * segments + (i + 1) % segments, row * segments + (i + 1) % segments)
        for row in range(len(path) - 1) for i in range(segments)
    ]
    collision = [
        rod(f"{name}_{i}", (a[0], 0, a[1]), (b[0], 0, b[1]), tube_radius, color, 12)
        for i, (a, b) in enumerate(zip(path, path[1:]))
    ]
    return (name, vertices, faces, color), collision


def stand():
    parts = []
    collision = []
    for name, size, top in (
        ("LowerTray", 0.66, 0.08),
        ("MiddleTray", 0.52, 0.31),
        ("UpperTray", 0.38, 0.54),
    ):
        z = top - 0.015 - 0.38
        parts.append(box(name, (size, size, 0.03), (0, 0, z), WOOD))
        parts.append(box(name + "Edge", (size, size, 0.006), (0, 0, z - 0.012), WOOD_EDGE))
        collision.append(box(name, (size, size, 0.03), (0, 0, z), WOOD))
    for i, x in enumerate((-0.27, 0.27)):
        for j, y in enumerate((-0.27, 0.27)):
            foot = box(f"Foot_{i}_{j}", (0.035, 0.035, 0.05), (x, y, -0.355), WOOD_EDGE)
            parts.append(foot)
            collision.append(foot)
    for i, x in enumerate((-0.095, 0.095)):
        upright = rod(f"FrameUpright_{i}", (x, 0, 0.08 - 0.38),
                      (x, 0, 0.717 - 0.38), 0.008, WOOD_EDGE, 20)
        parts.append(upright)
        collision.append(upright)
    arch, arch_collision = handle_arch("HandleArch", 0.095, 0.752 - 0.38,
                                       0.035, 0.008, WOOD_EDGE)
    parts.append(arch)
    collision.extend(arch_collision)
    return {
        "asset_id": "dessert_tiered_stand_01", "category": "dessert_stand",
        "bbox_m": [0.66, 0.66, 0.76], "mass": 3.0, "rigid_body": False,
        "parts": parts, "collision": collision,
        "support_surface": [
            {"name": name, "center": [0, 0, height - 0.38], "size": [size, size]}
            for name, height, size in (
                ("lower", 0.08, 0.64), ("middle", 0.31, 0.50), ("upper", 0.54, 0.36)
            )
        ],
    }


def strawberry(parts, prefix, x, y, z, scale=1):
    parts.append(ellipsoid(prefix + "Berry", (0.012 * scale, 0.011 * scale, 0.014 * scale),
                           (x, y, z), RED, 20, 12))
    for i in range(5):
        angle = 2 * math.pi * i / 5
        leaf = ellipsoid(prefix + f"Leaf_{i}",
                         (0.009 * scale, 0.0035 * scale, 0.0015 * scale),
                         (0.006 * scale, 0, 0), GREEN, 12, 6)
        parts.append(moved(rotated_z(leaf, angle),
                           (x, y, z + 0.011 * scale)))
    for row, level in enumerate((-0.004, 0.004)):
        for i in range(6):
            angle = 2 * math.pi * (i + row * 0.5) / 6
            radius = (0.011 if row == 0 else 0.010) * scale
            parts.append(ellipsoid(prefix + f"Seed_{row}_{i}",
                                   (0.0011 * scale, 0.0011 * scale, 0.0018 * scale),
                                   (x + radius * math.cos(angle),
                                    y + radius * math.sin(angle), z + level * scale),
                                   SEED, 8, 4))


def lower_cake():
    h = 0.09
    parts = [
        box("PinkBoard", (0.086, 0.086, 0.005), (0, 0, -0.0425), PALE_PINK),
        box("LowerSponge", (0.073, 0.073, 0.019), (0, 0, -0.030), PINK),
        box("VanillaFilling", (0.075, 0.075, 0.006), (0, 0, -0.017), WHITE),
        box("UpperSponge", (0.073, 0.073, 0.017), (0, 0, -0.006), PINK),
        box("PinkGlaze", (0.078, 0.078, 0.005), (0, 0, 0.005), PALE_PINK),
    ]
    for i, (x, y) in enumerate(((-0.021, -0.021), (-0.021, 0.021),
                                (0.021, -0.021), (0.021, 0.021))):
        parts.append(moved(piped_cream(f"CreamRosette_{i}",
                                       ((0.006, 0.012), (0.011, 0.010),
                                        (0.015, 0.007), (0.019, 0.004),
                                        (0.021, 0.001)), WHITE), (x, y, 0)))
    strawberry(parts, "Berry_", 0, 0, 0.032, 0.91)
    for i, x in enumerate((-0.026, -0.013, 0, 0.013, 0.026)):
        parts.append(box(f"FrontDrip_{i}", (0.004, 0.004, 0.008 + 0.003 * (i % 2)),
                         (x, -0.038, 0.001), PALE_PINK))
    parts = fit_parts(parts, (0.086, 0.086, h))
    return {"asset_id": "dessert_strawberry_square_01", "category": "petit_four",
            "bbox_m": [0.086, 0.086, h], "mass": 0.07, "rigid_body": True,
            "parts": parts, "collision": [box("Body", (0.086, 0.086, h), (0, 0, 0), PINK)],
            "support_surface": []}


def middle_cake():
    h = 0.11
    parts = [
        fluted_cup("PleatedGoldWrapper", 0.038, 0.044, -0.055, -0.014, GOLD),
        cylinder("SpongeTop", 0.039, 0.006, -0.013, color=PALE_PINK, segments=32),
        piped_cream("VanillaSwirl", ((-0.011, 0.037), (-0.004, 0.043),
                                     (0.004, 0.039), (0.012, 0.034),
                                     (0.019, 0.036), (0.026, 0.027),
                                     (0.033, 0.025), (0.039, 0.017),
                                     (0.043, 0.007)), WHITE),
    ]
    strawberry(parts, "Berry_", 0, 0, 0.045, 0.75)
    parts = fit_parts(parts, (0.092, 0.092, h))
    return {"asset_id": "dessert_cream_cupcake_01", "category": "cupcake",
            "bbox_m": [0.092, 0.092, h], "mass": 0.09, "rigid_body": True,
            "parts": parts, "collision": [box("Body", (0.092, 0.092, h), (0, 0, 0), GOLD)],
            "support_surface": []}


def upper_cake():
    h = 0.09
    parts = [
        fluted_cup("WhitePaperCup", 0.036, 0.039, -0.045, -0.029, WHITE),
        frustum("PinkCake", 0.035, 0.038, -0.037, -0.003, PALE_PINK, 32),
        cylinder("CreamRim", 0.038, 0.006, 0, color=WHITE),
        piped_cream("RoseBase", ((0.002, 0.036), (0.010, 0.037),
                                 (0.016, 0.030), (0.021, 0.022),
                                 (0.027, 0.011)), PINK),
    ]
    for ring, (count, radius, z, petal_size) in enumerate((
        (8, 0.023, 0.018, (0.013, 0.008, 0.006)),
        (6, 0.013, 0.029, (0.010, 0.006, 0.006)),
    )):
        for i in range(count):
            angle = 2 * math.pi * (i + 0.5 * ring) / count
            petal = ellipsoid(f"RosePetal_{ring}_{i}", petal_size,
                              (radius, 0, z), PALE_PINK if ring else PINK, 12, 8)
            parts.append(rotated_z(petal, angle))
    parts.append(ellipsoid("RoseHeart", (0.008, 0.008, 0.008),
                           (0, 0, 0.037), PINK, 16, 8))
    parts = fit_parts(parts, (0.08, 0.08, h))
    return {"asset_id": "dessert_pink_mini_cake_01", "category": "mini_cake",
            "bbox_m": [0.08, 0.08, h], "mass": 0.08, "rigid_body": True,
            "parts": parts, "collision": [box("Body", (0.08, 0.08, h), (0, 0, 0), PALE_PINK)],
            "support_surface": []}


def recipe():
    objects = [{
        "object_id": "stand_1", "category": "dessert_stand",
        "asset_id": "dessert_tiered_stand_01", "dynamic": False, "support": "floor",
        "fixed_pose": {"position": [0, 0, 0.38], "yaw_deg": 0},
    }]
    tiers = (
        ("lower", "petit_four", "dessert_strawberry_square_01", 0.09,
         [i * 0.112 for i in (-2, -1, 0, 1, 2)],
         [-0.25, -0.155, -0.06, 0.08, 0.19], 0.08),
        ("middle", "cupcake", "dessert_cream_cupcake_01", 0.11,
         [-0.16, 0, 0.16], [-0.18, -0.06, 0.06, 0.18], 0.31),
        ("upper", "mini_cake", "dessert_pink_mini_cake_01", 0.09,
         [i * 0.11 for i in (-1, 0, 1)], [-0.056, 0.056], 0.54),
    )
    for tier, category, asset_id, height, xs, ys, top in tiers:
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                if tier == "lower" and row == 2 and col == 2:
                    continue
                objects.append({
                    "object_id": f"{tier}_cake_{row + 1}_{col + 1}",
                    "category": category, "asset_id": asset_id,
                    "dynamic": True, "support": f"stand_1:{tier}",
                    "fixed_pose": {"position": [round(x, 6), round(y, 6),
                                               round(top + height / 2, 6)], "yaw_deg": 0},
                })
    return {
        "name": "three_tier_dessert_display", "room_type": "dining_room",
        "event": "dessert_display", "room_dimensions_m": [1.4, 1.4, 1.2],
        "description": "三层木质蛋糕展示架上整齐陈列粉色小蛋糕、奶油杯糕和草莓方糕。",
        "keywords": ["三层蛋糕架", "三层木质蛋糕展示架", "甜点展示", "蛋糕陈列",
                     "three tier dessert display"],
        "objects": objects, "task": {},
    }


def build(root=ROOT):
    asset_root = Path(root) / "data" / "assets"
    definitions = [stand(), lower_cake(), middle_cake(), upper_cake()]
    registry_path = asset_root / "registry.jsonl"
    old_lines = registry_path.read_text(encoding="utf-8").splitlines()
    previous = {entry["asset_id"]: entry for line in old_lines if line.strip()
                if (entry := json.loads(line))}
    records = []
    for item in definitions:
        asset_id = item["asset_id"]
        folder = asset_root / "source" / asset_id
        folder.mkdir(parents=True, exist_ok=True)
        visual = glb_bytes(item["parts"])
        (folder / "visual.glb").write_bytes(visual)
        geometry = json.dumps({"parts": [
            {"name": n, "vertices": v, "faces": f, "color": c}
            for n, v, f, c in item["parts"]
        ]}, separators=(",", ":")).encode()
        (folder / "geometry.json").write_bytes(geometry)
        for subdir, parts, collision in (
            ("usd", item["parts"], False), ("collision", item["collision"], True)
        ):
            path = asset_root / subdir / f"{asset_id}.usd"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(usd_text({"parts": parts}, collision), encoding="utf-8")
        usd_hash = hashlib.sha256((asset_root / "usd" / f"{asset_id}.usd").read_bytes()).hexdigest()
        collision_hash = hashlib.sha256(
            (asset_root / "collision" / f"{asset_id}.usd").read_bytes()).hexdigest()
        glb_hash = hashlib.sha256(visual).hexdigest()
        manifest = {
            "asset_id": asset_id, "source": "SceneFactory original dessert display geometry",
            **source_provenance(),
            "license": "MIT", "source_geometry": "visual.glb", "sha256": glb_hash,
            "geometry_sha256": hashlib.sha256(geometry).hexdigest(),
            "usd_sha256": usd_hash, "collision_sha256": collision_hash,
            "visual_transform": {"up_axis": "Y", "rotation_euler_deg": [90, 0, 0],
                                 "anchor": "bottom_center"},
            "bbox_m": item["bbox_m"], "units": "m", "origin": "bbox_center",
            "generator": "tools/build_dessert_display.py",
            "collision_strategy": "separate_simple_parts",
        }
        (folder / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        record = {
            "asset_id": asset_id, "name": asset_id.replace("_", " "),
            "category": item["category"], "bbox_m": item["bbox_m"],
            "primitive": "cube", "color": list(item["parts"][0][3]),
            "mass": item["mass"], "friction": 0.4,
            "static_friction": 0.55, "dynamic_friction": 0.4,
            "rigid_body": item["rigid_body"], "collision_enabled": True,
            "source_type": "local_usd", "usd_path": f"usd/{asset_id}.usd",
            "source_path": f"usd/{asset_id}.usd",
            "collision_path": f"collision/{asset_id}.usd",
            "collision_status": "authored", "collision_mode": "authored",
            "source": "SceneFactory original dessert display geometry",
            "license": "MIT", "hash": usd_hash, "status": "normalized",
            "support_surface": item["support_surface"],
            "tags": ["dessert_display_pack", "original_geometry"],
            "physics_parameters_source": "project_default",
        }
        prior = previous.get(asset_id, {})
        report_path = asset_root / prior.get("qa_report", "__missing__")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        if (report.get("usd_sha256") == usd_hash
                and report.get("collision_sha256") == collision_hash
                and report.get("glb_sha256") == glb_hash):
            for key in ("status", "collision_status", "qa_report"):
                if key in prior:
                    record[key] = prior[key]
        records.append(record)
    ids = {item["asset_id"] for item in records}
    retained = [line for line in old_lines if not line.strip() or line.startswith("#")
                or json.loads(line)["asset_id"] not in ids]
    registry_path.write_text("\n".join(retained + [json.dumps(r, separators=(",", ":"))
                                                  for r in records]) + "\n", encoding="utf-8")
    recipe_path = Path(root) / "recipes" / "three_tier_dessert_display.json"
    recipe_path.write_text(json.dumps(recipe(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    print(json.dumps({"assets": [item["asset_id"] for item in build()],
                      "objects": len(recipe()["objects"])}))
