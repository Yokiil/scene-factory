"""Three-level XML-Spatial-Collision benchmark generator compatible with mirros-lab/s-space.

Generates controlled Minimal Pairs across:
- Level 1 (`flat_primitives`): single-level bodies, axis-aligned boxes/cylinders
- Level 2 (`nested_frames`): parent->child SE(3) coordinate chains (anti-shortcut test!)
- Level 3 (`rotated_multipart`): rotated OBBs & multi-part high-fidelity library assets

Every sample exports exact MuJoCo-verified 3D world coordinates, 6-face AABB
bounds, signed separation/penetration distance, S-Space 3-axis labels, and exact
character spans for unambiguous object-token alignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.assets import AssetCatalog, yaw_deg_to_quat_wxyz
from ..core.mjcf import FragmentCache, create_base_xml, format_xml, merge_fragment
from ..core.verifier import verify_mujoco_xml
from .geometry import analyze_xml_spatial


BENCHMARK_SCHEMA_VERSION = "scene_factory.xml_spatial_benchmark.v1"

# Aligned with mirros-lab/s-space (horizontal/vertical/distance) + assembly collision/support.
XML_PROMPT_TEMPLATES: dict[str, str] = {
    "horizontal": "Given the MuJoCo XML above, is {query} to the left or right of {reference}? Answer with left or right.",
    "vertical": "Given the MuJoCo XML above, is {query} above or below {reference}? Answer with above or below.",
    "distance": "Given the MuJoCo XML above, is {query} farther from or closer to the front camera than {reference}? Answer with far or close.",
    "collision": "Given the MuJoCo XML above, is {query} colliding with or separated from {reference}? Answer with colliding or separated.",
    "support": "Given the MuJoCo XML above, is {query} supported or floating? Answer with supported or floating.",
}


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    pair_group_id: str
    level: str  # "level1_flat" | "level2_nested" | "level3_rotated_multipart"
    variant_kind: str  # "valid" | "colliding" | "floating"
    seed: int
    query_object: str
    reference_object: str
    xml: str
    xml_sha256: str
    prompts: dict[str, dict[str, Any]]
    xml_name_spans: dict[str, list[int]]
    ground_truth: dict[str, Any]
    renders: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render_prompt_with_spans(
    template: str, query: str, reference: str | None = None
) -> dict[str, Any]:
    """Render prompt and record exact half-open char spans for s-space token alignment."""
    q_marker = "__Q_OBJ__"
    r_marker = "__R_OBJ__"
    filled = template.replace("{query}", q_marker)
    if reference is not None:
        filled = filled.replace("{reference}", r_marker)
    q_idx = filled.index(q_marker)
    if reference is not None:
        r_idx = filled.index(r_marker)
        if q_idx < r_idx:
            text = filled.replace(q_marker, query, 1).replace(r_marker, reference, 1)
            q_span = [q_idx, q_idx + len(query)]
            shift = len(query) - len(q_marker)
            r_span = [r_idx + shift, r_idx + shift + len(reference)]
        else:
            text = filled.replace(r_marker, reference, 1).replace(q_marker, query, 1)
            r_span = [r_idx, r_idx + len(reference)]
            shift = len(reference) - len(r_marker)
            q_span = [q_idx + shift, q_idx + shift + len(query)]
        return {"text": text, "query_span": q_span, "reference_span": r_span}
    text = filled.replace(q_marker, query, 1)
    return {"text": text, "query_span": [q_idx, q_idx + len(query)]}


def _find_xml_body_spans(xml: str, names: tuple[str, ...]) -> dict[str, list[int]]:
    spans: dict[str, list[int]] = {}
    for name in names:
        needle = f'name="{name}"'
        idx = xml.find(needle)
        if idx >= 0:
            start = idx + len('name="')
            spans[name] = [start, start + len(name)]
    return spans


def _axis_labels(
    coords: dict[str, dict[str, float]], query: str, reference: str, eps: float = 1e-4
) -> dict[str, str]:
    qa, rb = coords[query], coords[reference]
    dh = qa["horizontal"] - rb["horizontal"]
    dv = qa["vertical"] - rb["vertical"]
    dd = qa["distance"] - rb["distance"]
    return {
        "horizontal": "tie" if abs(dh) <= eps else ("right" if dh > 0 else "left"),
        "vertical": "tie" if abs(dv) <= eps else ("below" if dv > 0 else "above"),
        "distance": "tie" if abs(dd) <= eps else ("close" if dd > 0 else "far"),
    }


def _add_box_body(
    parent: ET.Element,
    name: str,
    pos: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    *,
    rgba: tuple[float, float, float, float],
    dynamic: bool = True,
    yaw_deg: float = 0.0,
    euler_deg: tuple[float, float, float] | None = None,
) -> ET.Element:
    attrs = {"name": name, "pos": f"{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}"}
    if euler_deg is not None:
        attrs["euler"] = f"{euler_deg[0]:.3f} {euler_deg[1]:.3f} {euler_deg[2]:.3f}"
    elif abs(yaw_deg) > 1e-6:
        q = yaw_deg_to_quat_wxyz(yaw_deg)
        attrs["quat"] = f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
    body = ET.SubElement(parent, "body", attrs)
    if dynamic:
        if parent.tag == "worldbody":
            ET.SubElement(body, "freejoint", {"name": f"joint__{name}"})
        else:
            ET.SubElement(body, "joint", {"name": f"joint__{name}_z", "type": "slide", "axis": "0 0 1"})
    gattrs = {
        "name": f"geom__{name}",
        "type": "box",
        "size": f"{half_extents[0]:.5f} {half_extents[1]:.5f} {half_extents[2]:.5f}",
        "rgba": f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}",
    }
    if dynamic:
        gattrs["mass"] = "0.4"
    ET.SubElement(body, "geom", gattrs)
    return body


def _build_level1_pair(rng: random.Random, variant_kind: str) -> tuple[str, str, str]:
    """Level 1: flat single-level boxes on a table with minimal collision/floating shifts."""
    root = ET.fromstring(create_base_xml("level1_flat"))
    world = root.find("worldbody")
    assert world is not None

    table_top = 0.74
    _add_box_body(
        world,
        "support_table",
        (0.0, 0.0, table_top / 2.0),
        (0.65, 0.45, table_top / 2.0),
        rgba=(0.76, 0.65, 0.52, 1.0),
        dynamic=False,
    )
    ha = (rng.uniform(0.05, 0.11), rng.uniform(0.05, 0.11), rng.uniform(0.05, 0.10))
    hb = (rng.uniform(0.05, 0.11), rng.uniform(0.05, 0.11), rng.uniform(0.05, 0.10))

    bx = rng.uniform(-0.22, 0.22)
    by = rng.uniform(-0.18, 0.18)
    bz = table_top + hb[2] + 0.001

    sign_x = rng.choice((-1.0, 1.0))
    if variant_kind == "colliding":
        overlap = rng.uniform(0.02, min(0.05, ha[0] * 0.8))
        ax = bx + sign_x * (ha[0] + hb[0] - overlap)
        ay = by + rng.uniform(-0.02, 0.02)
        az = table_top + ha[2] + 0.001
    elif variant_kind == "floating":
        gap = rng.uniform(0.05, 0.14)
        ax = bx + sign_x * (ha[0] + hb[0] + rng.uniform(0.05, 0.14))
        ay = by + rng.uniform(-0.10, 0.10)
        az = table_top + ha[2] + gap
    else:
        ax = bx + sign_x * (ha[0] + hb[0] + rng.uniform(0.04, 0.16))
        ay = by + rng.uniform(-0.12, 0.12)
        az = table_top + ha[2] + 0.001

    _add_box_body(world, "red_block", (ax, ay, az), ha, rgba=(0.82, 0.22, 0.20, 1.0), dynamic=True)
    _add_box_body(world, "blue_block", (bx, by, bz), hb, rgba=(0.18, 0.38, 0.78, 1.0), dynamic=True)
    return format_xml(root), "red_block", "blue_block"


def _build_level2_nested_pair(rng: random.Random, variant_kind: str) -> tuple[str, str, str]:
    """Level 2: nested parent->child coordinate frames (tests true world-frame composition)."""
    root = ET.fromstring(create_base_xml("level2_nested"))
    world = root.find("worldbody")
    assert world is not None

    # Parent assembly shifted arbitrarily in world coordinates so local coords are misleading!
    frame_x = rng.uniform(-0.45, 0.45)
    frame_y = rng.uniform(-0.35, 0.35)
    table_h = rng.uniform(0.60, 0.80)

    station = ET.SubElement(
        world,
        "body",
        {"name": "workstation_frame", "pos": f"{frame_x:.5f} {frame_y:.5f} 0.0"},
    )
    table_body = _add_box_body(
        station,
        "workbench_top",
        (0.0, 0.0, table_h / 2.0),
        (0.60, 0.45, table_h / 2.0),
        rgba=(0.72, 0.64, 0.54, 1.0),
        dynamic=False,
    )
    tray_lx = rng.uniform(-0.10, 0.10)
    tray_ly = rng.uniform(-0.10, 0.10)
    tray_hx, tray_hy, tray_h = 0.20, 0.16, 0.02
    tray_body = _add_box_body(
        table_body,
        "tray_frame",
        (tray_lx, tray_ly, table_h / 2.0 + tray_h / 2.0 + 0.001),
        (tray_hx, tray_hy, tray_h / 2.0),
        rgba=(0.55, 0.58, 0.62, 1.0),
        dynamic=False,
    )

    ha = (0.06, 0.06, 0.06)
    hb = (0.07, 0.07, 0.07)
    # Child A is nested 3 levels deep inside tray_frame!
    child_lx = rng.uniform(-0.05, 0.05)
    child_ly = rng.uniform(-0.05, 0.05)
    tray_top_local_z = tray_h / 2.0 + ha[2] + (rng.uniform(0.06, 0.12) if variant_kind == "floating" else 0.001)
    _add_box_body(
        tray_body,
        "nested_peg",
        (child_lx, child_ly, tray_top_local_z),
        ha,
        rgba=(0.86, 0.30, 0.18, 1.0),
        dynamic=True,
    )

    # Compute exact world position of nested_peg so sibling world_block tests cross-frame reasoning!
    peg_wx = frame_x + tray_lx + child_lx
    peg_wy = frame_y + tray_ly + child_ly
    tray_top_wz = table_h + tray_h + 0.001

    sign_x = rng.choice((-1.0, 1.0))
    if variant_kind == "colliding":
        overlap = rng.uniform(0.02, 0.05)
        bx = peg_wx + sign_x * (ha[0] + hb[0] - overlap)
        by = peg_wy + rng.uniform(-0.02, 0.02)
        bz = tray_top_wz + hb[2] + 0.001
    else:
        # Place world_block outside the tray on the workbench top so valid assemblies have zero collisions.
        bx = frame_x + tray_lx + sign_x * (tray_hx + hb[0] + rng.uniform(0.03, 0.08))
        by = frame_y + tray_ly + rng.uniform(-0.10, 0.10)
        bz = table_h + hb[2] + 0.001

    _add_box_body(world, "world_block", (bx, by, bz), hb, rgba=(0.20, 0.55, 0.82, 1.0), dynamic=True)
    return format_xml(root), "nested_peg", "world_block"


def _build_level3_rotated_multipart(
    rng: random.Random, variant_kind: str, cache: FragmentCache
) -> tuple[str, str, str]:
    """Level 3: rotated 3D OBBs + high-fidelity multi-part library assets."""
    xml = create_base_xml("level3_rotated_multipart")
    catalog = cache.catalog

    table_entry = catalog.get("kd_table_01")
    table_h = table_entry.bbox_m[2]
    table_yaw = rng.uniform(-35.0, 35.0)
    a_t, b_t, d_t, _ = cache.get_or_build(
        "kd_table_01", name="dining_table", pos=(0.0, 0.0, table_h / 2.0), yaw_deg=table_yaw, dynamic=False
    )
    xml = merge_fragment(xml, a_t, b_t, enable_multiccd=d_t)

    mug_entry = catalog.get("kd_mug_01")
    bowl_entry = catalog.get("kd_bowl_01")
    mug_h, bowl_h = mug_entry.bbox_m[2], bowl_entry.bbox_m[2]

    # Place bowl on the rotated tabletop.
    bx = rng.uniform(-0.20, 0.20)
    by = rng.uniform(-0.14, 0.14)
    bowl_yaw = rng.uniform(-180.0, 180.0)
    a_b, b_b, d_b, _ = cache.get_or_build(
        "kd_bowl_01",
        name="ceramic_bowl",
        pos=(bx, by, table_h + bowl_h / 2.0 + 0.002),
        yaw_deg=bowl_yaw,
        dynamic=True,
    )
    xml = merge_fragment(xml, a_b, b_b, enable_multiccd=d_b)

    # Place mug rotated relative to bowl (minimal colliding vs floating vs valid shift).
    angle = rng.uniform(0.0, 2.0 * math.pi)
    r_sum = (max(mug_entry.bbox_m[:2]) + max(bowl_entry.bbox_m[:2])) / 2.0
    if variant_kind == "colliding":
        dist = max(0.02, r_sum - rng.uniform(0.035, 0.065))
        mz = table_h + mug_h / 2.0 + 0.002
    elif variant_kind == "floating":
        dist = r_sum + rng.uniform(0.06, 0.14)
        mz = table_h + mug_h / 2.0 + rng.uniform(0.06, 0.15)
    else:
        dist = r_sum + rng.uniform(0.06, 0.16)
        mz = table_h + mug_h / 2.0 + 0.002

    mx = bx + dist * math.cos(angle)
    my = by + dist * math.sin(angle)
    mug_yaw = rng.uniform(-180.0, 180.0)
    a_m, b_m, d_m, _ = cache.get_or_build(
        "kd_mug_01",
        name="blue_mug",
        pos=(mx, my, mz),
        yaw_deg=mug_yaw,
        dynamic=True,
    )
    xml = merge_fragment(xml, a_m, b_m, enable_multiccd=d_m)
    return xml, "blue_mug", "ceramic_bowl"


def generate_xml_spatial_benchmark(
    output_dir: str | Path,
    *,
    per_level: int = 30,
    seed: int = 42,
    levels: tuple[str, ...] = ("level1_flat", "level2_nested", "level3_rotated_multipart"),
    render_images: bool = False,
    crosscheck_mujoco: bool = True,
    catalog: AssetCatalog | None = None,
) -> dict[str, Any]:
    """Generate a 3-level minimal-pair XML spatial & collision benchmark dataset."""
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cat = catalog or AssetCatalog()
    cache = FragmentCache(cat)

    variants = ("valid", "colliding", "floating")
    samples: list[BenchmarkSample] = []

    for level_idx, level in enumerate(levels):
        for group_idx in range(per_level):
            group_seed = seed + level_idx * 100_000 + group_idx * 10
            group_id = f"{level}_g{group_idx:04d}"
            for var_idx, variant_kind in enumerate(variants):
                rng = random.Random(group_seed + var_idx)
                if level == "level1_flat":
                    xml, query, ref = _build_level1_pair(rng, variant_kind)
                elif level == "level2_nested":
                    xml, query, ref = _build_level2_nested_pair(rng, variant_kind)
                elif level == "level3_rotated_multipart":
                    xml, query, ref = _build_level3_rotated_multipart(rng, variant_kind, cache)
                else:
                    raise ValueError(f"unknown benchmark level: {level!r}")

                analysis = analyze_xml_spatial(xml, crosscheck_mujoco=crosscheck_mujoco)
                sample_id = f"{group_id}_{variant_kind}"
                pair_record = next(
                    (
                        p
                        for p in analysis.pairs
                        if {p.body_a, p.body_b} == {query, ref}
                    ),
                    None,
                )
                assert pair_record is not None
                query_floating = any(f["body"] == query for f in analysis.floating_bodies)

                renders: list[dict[str, Any]] = []
                if render_images:
                    rdir = out_root / "renders" / sample_id
                    vrep = verify_mujoco_xml(xml, settle_steps=0, render=True, render_dir=rdir)
                    renders = vrep.to_dict(include_base64=False)["renders"]

                prompts = {
                    axis: _render_prompt_with_spans(XML_PROMPT_TEMPLATES[axis], query, ref)
                    for axis in ("horizontal", "vertical", "distance", "collision")
                }
                prompts["support"] = _render_prompt_with_spans(XML_PROMPT_TEMPLATES["support"], query)

                gt = {
                    "axis_labels": _axis_labels(analysis.camera_sspace_coords, query, ref),
                    "collision_label": "colliding" if pair_record.status == "colliding" else "separated",
                    "support_label": "floating" if query_floating else "supported",
                    "assembly_valid": analysis.assembly_valid,
                    "pair": pair_record.to_dict(),
                    "query_body": analysis.bodies[query].to_dict(),
                    "reference_body": analysis.bodies[ref].to_dict(),
                    "camera_sspace_coords": {
                        query: analysis.camera_sspace_coords[query],
                        ref: analysis.camera_sspace_coords[ref],
                    },
                    "mujoco_crosschecked": analysis.mujoco_crosschecked,
                }
                samples.append(
                    BenchmarkSample(
                        sample_id=sample_id,
                        pair_group_id=group_id,
                        level=level,
                        variant_kind=variant_kind,
                        seed=group_seed + var_idx,
                        query_object=query,
                        reference_object=ref,
                        xml=xml,
                        xml_sha256=hashlib.sha256(xml.encode("utf-8")).hexdigest(),
                        prompts=prompts,
                        xml_name_spans=_find_xml_body_spans(xml, (query, ref)),
                        ground_truth=gt,
                        renders=renders,
                    )
                )

    jsonl_path = out_root / "samples.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(s.to_dict(), ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8",
    )

    by_level: dict[str, dict[str, int]] = {}
    for s in samples:
        bucket = by_level.setdefault(s.level, {"total": 0, "colliding": 0, "floating": 0, "valid": 0})
        bucket["total"] += 1
        if s.ground_truth["collision_label"] == "colliding":
            bucket["colliding"] += 1
        elif s.ground_truth["support_label"] == "floating":
            bucket["floating"] += 1
        elif s.ground_truth["assembly_valid"]:
            bucket["valid"] += 1

    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "seed": seed,
        "per_level_groups": per_level,
        "total_samples": len(samples),
        "levels": list(levels),
        "summary_by_level": by_level,
        "samples_path": str(jsonl_path),
        "render_images": render_images,
        "crosscheck_mujoco": crosscheck_mujoco,
        "fragment_cache": cache.stats(),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
