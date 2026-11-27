"""Fast deterministic layout and physics randomization directly over MuJoCo XML.

Compiles the base MjModel once and validates randomized candidates via in-place
pose/mass/friction patching (mj_setConst + mj_resetData + mj_forward) before
emitting standalone MuJoCo XML variants.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Sequence

from .mjcf import format_xml


@dataclass(frozen=True)
class RandomizedVariant:
    index: int
    seed: int
    xml: str
    xml_sha256: str
    perturbations: dict[str, dict[str, Any]]
    max_penetration_m: float
    attempts_used: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_floats(text: str, expected: int, default: tuple[float, ...]) -> list[float]:
    if not text:
        return list(default)
    values = [float(v) for v in text.split()]
    if len(values) != expected:
        raise ValueError(f"expected {expected} floats, got {len(values)}")
    return values


def _quat_mul_wxyz(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def randomize_xml(
    xml: str,
    *,
    count: int = 5,
    seed: int = 42,
    xy_jitter_m: float = 0.08,
    yaw_jitter_deg: float = 25.0,
    mass_scale_range: tuple[float, float] = (1.0, 1.0),
    friction_scale_range: tuple[float, float] = (1.0, 1.0),
    include_bodies: Sequence[str] | None = None,
    exclude_bodies: Sequence[str] | None = None,
    penetration_tol_m: float = 0.01,
    max_attempts: int = 32,
) -> tuple[list[RandomizedVariant], dict[str, Any]]:
    """Produce deterministic, collision-checked MuJoCo XML layout/physics variants."""
    if count < 1 or count > 500:
        raise ValueError("count must be in [1, 500]")
    if xy_jitter_m < 0 or yaw_jitter_deg < 0:
        raise ValueError("jitter bounds must be non-negative")
    if mass_scale_range[0] <= 0 or mass_scale_range[0] > mass_scale_range[1]:
        raise ValueError("invalid mass_scale_range")
    if friction_scale_range[0] <= 0 or friction_scale_range[0] > friction_scale_range[1]:
        raise ValueError("invalid friction_scale_range")

    base_root = ET.fromstring(xml)
    world = base_root.find("worldbody")
    if world is None:
        raise ValueError("MuJoCo XML has no <worldbody>")

    include_set = set(include_bodies) if include_bodies else None
    exclude_set = set(exclude_bodies) if exclude_bodies else set()

    targets: list[tuple[str, list[float], tuple[float, float, float, float]]] = []
    for body in world.findall("body"):
        name = body.attrib.get("name", "")
        if not name or body.find("freejoint") is None:
            continue
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        pos = _parse_floats(body.attrib.get("pos", ""), 3, (0.0, 0.0, 0.0))
        q_raw = _parse_floats(body.attrib.get("quat", ""), 4, (1.0, 0.0, 0.0, 0.0))
        norm = math.sqrt(sum(v * v for v in q_raw)) or 1.0
        quat = (q_raw[0] / norm, q_raw[1] / norm, q_raw[2] / norm, q_raw[3] / norm)
        targets.append((name, pos, quat))

    mujoco = None
    model = None
    data = None
    base_arrays: dict[str, Any] = {}
    body_map: dict[str, tuple[int, int]] = {}
    try:
        mujoco = importlib.import_module("mujoco")
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for arr in ("body_pos", "body_quat", "qpos0", "body_mass", "body_inertia", "geom_friction"):
            base_arrays[arr] = getattr(model, arr).copy()
        for name, _, _ in targets:
            bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            if bid >= 1:
                jid = int(model.body_jntadr[bid])
                qadr = int(model.jnt_qposadr[jid])
                body_map[name] = (bid, qadr)
    except Exception:
        mujoco = None
        model = None

    variants: list[RandomizedVariant] = []
    patched_checks = 0
    for idx in range(count):
        v_seed = int(seed) + idx
        rng = random.Random(v_seed)
        chosen_pert: dict[str, dict[str, Any]] | None = None
        chosen_pen = 0.0
        used_attempts = 0

        for attempt in range(1, max_attempts + 1):
            used_attempts = attempt
            pert: dict[str, dict[str, Any]] = {}
            for name, pos, quat in targets:
                dx = rng.uniform(-xy_jitter_m, xy_jitter_m)
                dy = rng.uniform(-xy_jitter_m, xy_jitter_m)
                dyaw = rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)
                half = math.radians(dyaw) / 2.0
                qz = (math.cos(half), 0.0, 0.0, math.sin(half))
                new_quat = _quat_mul_wxyz(qz, quat)
                qnorm = math.sqrt(sum(v * v for v in new_quat)) or 1.0
                new_quat = tuple(v / qnorm for v in new_quat)
                new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
                m_scale = rng.uniform(*mass_scale_range)
                f_scale = rng.uniform(*friction_scale_range)
                pert[name] = {
                    "pos": new_pos,
                    "quat": list(new_quat),
                    "delta_xy_m": [round(dx, 5), round(dy, 5)],
                    "delta_yaw_deg": round(dyaw, 3),
                    "mass_scale": round(m_scale, 4),
                    "friction_scale": round(f_scale, 4),
                }

            if mujoco is None or model is None or data is None:
                chosen_pert = pert
                chosen_pen = 0.0
                break

            # In-place MjModel parameter patch — avoids recompiling XML per attempt.
            for arr, backup in base_arrays.items():
                getattr(model, arr)[:] = backup
            for name, info in pert.items():
                if name not in body_map:
                    continue
                bid, qadr = body_map[name]
                model.body_pos[bid] = info["pos"]
                model.body_quat[bid] = info["quat"]
                model.qpos0[qadr : qadr + 7] = [*info["pos"], *info["quat"]]
                model.body_mass[bid] = base_arrays["body_mass"][bid] * info["mass_scale"]
                model.body_inertia[bid] = base_arrays["body_inertia"][bid] * info["mass_scale"]
                gstart = int(model.body_geomadr[bid])
                gnum = int(model.body_geomnum[bid])
                for gid in range(gstart, gstart + gnum):
                    model.geom_friction[gid] = (
                        base_arrays["geom_friction"][gid] * info["friction_scale"]
                    )
            mujoco.mj_setConst(model, data)
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            patched_checks += 1

            worst_pen = 0.0
            for ci in range(int(data.ncon)):
                d = float(data.contact[ci].dist)
                if d < 0.0 and -d > worst_pen:
                    worst_pen = -d
            if worst_pen <= penetration_tol_m or attempt == max_attempts:
                chosen_pert = pert
                chosen_pen = worst_pen
                break

        assert chosen_pert is not None
        out_root = copy.deepcopy(base_root)
        out_world = out_root.find("worldbody")
        assert out_world is not None
        for body in out_world.findall("body"):
            name = body.attrib.get("name", "")
            if name not in chosen_pert:
                continue
            info = chosen_pert[name]
            body.set("pos", _fmt(info["pos"]))
            body.set("quat", _fmt(info["quat"]))
            m_scale = info["mass_scale"]
            f_scale = info["friction_scale"]
            if m_scale != 1.0:
                inertial = body.find("inertial")
                if inertial is not None:
                    if "mass" in inertial.attrib:
                        inertial.set("mass", _fmt([float(inertial.attrib["mass"]) * m_scale]))
                    if "diaginertia" in inertial.attrib:
                        diag = [float(v) * m_scale for v in inertial.attrib["diaginertia"].split()]
                        inertial.set("diaginertia", _fmt(diag))
                for geom in body.findall("geom"):
                    if "mass" in geom.attrib and float(geom.attrib["mass"]) > 0:
                        geom.set("mass", _fmt([float(geom.attrib["mass"]) * m_scale]))
            if f_scale != 1.0:
                for geom in body.findall("geom"):
                    if "friction" in geom.attrib:
                        fr = [float(v) * f_scale for v in geom.attrib["friction"].split()]
                        geom.set("friction", _fmt(fr))

        variant_xml = format_xml(out_root)
        variants.append(
            RandomizedVariant(
                index=idx,
                seed=v_seed,
                xml=variant_xml,
                xml_sha256=hashlib.sha256(variant_xml.encode("utf-8")).hexdigest(),
                perturbations=chosen_pert,
                max_penetration_m=round(chosen_pen, 6),
                attempts_used=used_attempts,
            )
        )

    stats = {
        "requested_count": count,
        "generated_count": len(variants),
        "randomized_bodies": [name for name, _, _ in targets],
        "native_compiles": 1 if model is not None else 0,
        "patched_checks": patched_checks,
        "mujoco_checked": model is not None,
    }
    return variants, stats
