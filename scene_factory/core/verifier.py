"""Objective MuJoCo physics verifier and multi-view offscreen PNG renderer.

Acts strictly as a factual referee: compiles the XML, simulates settling,
measures penetration/drift/falls/warnings, and renders RGB images so VLM/LLM
agents can visually self-critique and self-correct. Never mutates or silently
"fixes" the caller's XML.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import math
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def encode_png(rgb: Any) -> bytes:
    """Encode an HxWx3 uint8 array into PNG bytes using Python stdlib only."""
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"expected HxWx3 RGB array, got shape {rgb.shape}")
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + _chunk(b"IEND", b"")
    )


@dataclass
class VerificationReport:
    """Factual compilation, settling physics, and rendering diagnostics."""

    compiled: bool
    xml_sha256: str
    error: str | None = None
    body_count: int = 0
    free_body_count: int = 0
    settle_steps: int = 0
    sim_time_s: float = 0.0
    finite_state: bool = False
    solver_warnings: list[str] = field(default_factory=list)
    initial_contact_count: int = 0
    initial_max_penetration_m: float = 0.0
    final_max_penetration_m: float = 0.0
    max_displacement_m: float = 0.0
    max_linear_speed_mps: float = 0.0
    fallen_bodies: list[str] = field(default_factory=list)
    moved_bodies: list[dict[str, Any]] = field(default_factory=list)
    renders: list[dict[str, Any]] = field(default_factory=list)
    render_error: str | None = None
    settled_stable: bool = False
    score: float = 0.0

    def to_dict(self, *, include_base64: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_base64:
            payload["renders"] = [
                {k: v for k, v in item.items() if k != "png_base64"}
                for item in self.renders
            ]
        return payload


def _inject_render_cameras(xml: str, width: int, height: int) -> tuple[str, list[str]]:
    """Return a temporary render copy guaranteed to have offscreen buffer and cameras."""
    root = ET.fromstring(xml)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(max(int(glob.attrib.get("offwidth", 0)), width)))
    glob.set("offheight", str(max(int(glob.attrib.get("offheight", 0)), height)))

    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    existing = [c.attrib.get("name", "") for c in world.findall("camera") if c.attrib.get("name")]
    if existing:
        return ET.tostring(root, encoding="unicode"), existing[:3]

    ET.SubElement(
        world,
        "camera",
        {
            "name": "_verify_overview",
            "pos": "2.4 -2.4 2.0",
            "xyaxes": "0.707 0.707 0 -0.408 0.408 0.816",
        },
    )
    ET.SubElement(
        world,
        "camera",
        {
            "name": "_verify_front",
            "pos": "0 -2.6 1.4",
            "xyaxes": "1 0 0 0 0.36 0.933",
        },
    )
    return ET.tostring(root, encoding="unicode"), ["_verify_overview", "_verify_front"]


def _max_penetration(data: Any) -> float:
    worst = 0.0
    for i in range(int(data.ncon)):
        dist = float(data.contact[i].dist)
        if dist < 0.0 and -dist > worst:
            worst = -dist
    return worst


def verify_mujoco_xml(
    xml: str,
    *,
    settle_steps: int = 150,
    render: bool = True,
    render_dir: str | Path | None = None,
    width: int = 512,
    height: int = 384,
    penetration_tol_m: float = 0.015,
    displacement_tol_m: float = 0.08,
    speed_tol_mps: float = 0.05,
) -> VerificationReport:
    """Compile and evaluate a MuJoCo XML factually without modifying it."""
    xml_sha = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    try:
        ET.fromstring(xml)
    except ET.ParseError as exc:
        return VerificationReport(compiled=False, xml_sha256=xml_sha, error=f"XML parse error: {exc}")

    try:
        mujoco = importlib.import_module("mujoco")
        np = importlib.import_module("numpy")
    except Exception as exc:
        return VerificationReport(
            compiled=False,
            xml_sha256=xml_sha,
            error=f"MuJoCo runtime unavailable: {type(exc).__name__}: {exc}",
        )

    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        return VerificationReport(
            compiled=False,
            xml_sha256=xml_sha,
            error=f"MuJoCo compile error: {exc}",
        )

    free_bodies: list[tuple[int, str]] = []
    for body_id in range(1, int(model.nbody)):
        jnt_num = int(model.body_jntnum[body_id])
        if jnt_num <= 0:
            continue
        jnt_adr = int(model.body_jntadr[body_id])
        if int(model.jnt_type[jnt_adr]) == int(mujoco.mjtJoint.mjJNT_FREE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            free_bodies.append((body_id, name))

    init_pos = {body_id: data.xpos[body_id].copy() for body_id, _ in free_bodies}
    init_contacts = int(data.ncon)
    init_penetration = _max_penetration(data)

    for _ in range(max(0, int(settle_steps))):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    warnings = [
        f"warning_{idx}(count={int(w.number)})"
        for idx, w in enumerate(data.warning)
        if int(w.number) > 0
    ]
    final_penetration = _max_penetration(data)

    max_disp = 0.0
    max_speed = 0.0
    fallen: list[str] = []
    moved: list[dict[str, Any]] = []
    for body_id, name in free_bodies:
        pos0 = init_pos[body_id]
        pos1 = data.xpos[body_id]
        disp = float(math.dist(pos0.tolist(), pos1.tolist()))
        jnt_adr = int(model.body_jntadr[body_id])
        dof_adr = int(model.jnt_dofadr[jnt_adr])
        vel = data.qvel[dof_adr : dof_adr + 3]
        speed = float(math.sqrt(sum(float(v) ** 2 for v in vel)))
        max_disp = max(max_disp, disp)
        max_speed = max(max_speed, speed)
        if float(pos1[2]) < -0.05:
            fallen.append(name)
        if disp > 0.01 or speed > 0.02 or float(pos1[2]) < -0.05:
            moved.append(
                {
                    "body": name,
                    "initial_pos": [round(float(v), 4) for v in pos0],
                    "final_pos": [round(float(v), 4) for v in pos1],
                    "displacement_m": round(disp, 4),
                    "final_speed_mps": round(speed, 4),
                }
            )

    renders: list[dict[str, Any]] = []
    render_error: str | None = None
    if render and finite:
        try:
            render_xml, cam_names = _inject_render_cameras(xml, width, height)
            rmodel = mujoco.MjModel.from_xml_string(render_xml)
            rdata = mujoco.MjData(rmodel)
            rdata.qpos[:] = data.qpos
            rdata.qvel[:] = data.qvel
            mujoco.mj_forward(rmodel, rdata)
            renderer = mujoco.Renderer(rmodel, height=height, width=width)
            try:
                out_dir = Path(render_dir) if render_dir else None
                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                for cam_name in cam_names:
                    renderer.update_scene(rdata, camera=cam_name)
                    rgb = renderer.render()
                    png_bytes = encode_png(rgb)
                    item: dict[str, Any] = {
                        "camera": cam_name,
                        "width": width,
                        "height": height,
                        "sha256": hashlib.sha256(png_bytes).hexdigest(),
                        "png_base64": base64.b64encode(png_bytes).decode("ascii"),
                    }
                    if out_dir:
                        img_path = out_dir / f"{cam_name}.png"
                        img_path.write_bytes(png_bytes)
                        item["path"] = str(img_path.resolve())
                    renders.append(item)
            finally:
                renderer.close()
        except Exception as exc:
            render_error = f"{type(exc).__name__}: {exc}"

    pen_ok = init_penetration <= penetration_tol_m and final_penetration <= penetration_tol_m
    stable = (
        finite
        and not warnings
        and not fallen
        and pen_ok
        and max_disp <= displacement_tol_m
        and max_speed <= speed_tol_mps
    )

    # Objective, transparent verifiable score in [0.0, 1.0] for RL / SFT trace filtering.
    score = 0.25  # compiled
    if finite and not warnings:
        score += 0.20
    if not fallen:
        score += 0.15
    if pen_ok:
        score += 0.20
    if max_disp <= displacement_tol_m and max_speed <= speed_tol_mps:
        score += 0.20

    return VerificationReport(
        compiled=True,
        xml_sha256=xml_sha,
        body_count=max(0, int(model.nbody) - 1),
        free_body_count=len(free_bodies),
        settle_steps=int(settle_steps),
        sim_time_s=round(float(data.time), 6),
        finite_state=finite,
        solver_warnings=warnings,
        initial_contact_count=init_contacts,
        initial_max_penetration_m=round(init_penetration, 6),
        final_max_penetration_m=round(final_penetration, 6),
        max_displacement_m=round(max_disp, 6),
        max_linear_speed_mps=round(max_speed, 6),
        fallen_bodies=fallen,
        moved_bodies=moved,
        renders=renders,
        render_error=render_error,
        settled_stable=stable,
        score=round(score, 4),
    )
