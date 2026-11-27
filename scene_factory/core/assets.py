"""Lightweight asset catalog and high-fidelity MuJoCo MJCF fragment builder.

Reads the existing repository asset library (`data/assets/registry.jsonl` and
verified multi-part `geometry.json` meshes) with zero dependency on legacy v1
compilation pipelines.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# Bilingual aliases to help both Chinese and English queries hit local assets.
_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "杯": ("mug", "cup"),
    "马克杯": ("mug",),
    "碗": ("bowl",),
    "盘": ("plate", "dish"),
    "瓶": ("bottle",),
    "盒": ("box",),
    "桌": ("table", "coffee_table", "dining_table"),
    "餐桌": ("dining_table", "table"),
    "茶几": ("coffee_table",),
    "椅": ("chair",),
    "沙发": ("sofa",),
    "柜": ("cabinet", "side_cabinet", "kitchen_cabinet"),
    "架": ("shelf", "storage_shelf", "stand"),
    "台面": ("counter", "kitchen_counter"),
    "岛台": ("island", "kitchen_island"),
    "锅": ("pot", "frying_pan"),
    "刀": ("knife", "kitchen_knife"),
    "砧板": ("cutting_board",),
    "微波炉": ("microwave",),
    "炉": ("stove", "electric_stove"),
    "水壶": ("kettle", "electric_kettle"),
    "蛋糕": ("cake", "cupcake", "dessert"),
    "甜点": ("dessert", "cake", "stand"),
    "遥控器": ("remote", "remote_control"),
    "抱枕": ("cushion",),
    "靠垫": ("cushion",),
    "零食": ("snack", "snack_bag"),
    "背包": ("backpack",),
    "鞋": ("shoe",),
    "钥匙": ("key", "keys"),
    "凳": ("bench", "entry_bench"),
}


def default_assets_dir() -> Path:
    """Resolve `data/assets` from the repository or installed package."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "assets"
        if (candidate / "registry.jsonl").is_file():
            return candidate
    raise FileNotFoundError("could not locate data/assets/registry.jsonl")


def _fmt_nums(values: Any) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def sanitize_mjcf_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip()).strip("_.-")
    return cleaned or "object"


def yaw_deg_to_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(float(yaw_deg)) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


@dataclass(frozen=True)
class AssetEntry:
    """Compact, simulator-ready view of one library asset."""

    asset_id: str
    name: str
    category: str
    bbox_m: tuple[float, float, float]
    mass_kg: float
    friction: float
    primitive: str = "box"
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)
    collision_enabled: bool = True
    status: str = "validated"
    tags: tuple[str, ...] = ()
    support_surfaces: tuple[dict[str, Any], ...] = ()
    has_mesh_geometry: bool = False
    geometry_sha256: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_summary(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "category": self.category,
            "bbox_m": list(self.bbox_m),
            "mass_kg": self.mass_kg,
            "friction": self.friction,
            "primitive": self.primitive,
            "color_rgb": list(self.color),
            "has_detailed_mesh": self.has_mesh_geometry,
            "support_surfaces": [
                {
                    "name": s.get("name", "top"),
                    "center_m": s.get("center"),
                    "size_m": s.get("size"),
                }
                for s in self.support_surfaces
            ],
            "tags": list(self.tags),
        }


class AssetCatalog:
    """Read-only catalog over `data/assets/registry.jsonl` and local mesh geometry."""

    def __init__(self, assets_dir: str | Path | None = None) -> None:
        self.assets_dir = Path(assets_dir) if assets_dir else default_assets_dir()
        self.registry_path = self.assets_dir / "registry.jsonl"
        self._entries: dict[str, AssetEntry] = {}
        self._mesh_cache: dict[str, list[dict[str, Any]] | None] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.is_file():
            raise FileNotFoundError(f"asset registry not found: {self.registry_path}")
        for line in self.registry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("status") == "rejected":
                continue
            asset_id = str(raw["asset_id"])
            bbox = tuple(float(v) for v in raw.get("bbox_m", (0.1, 0.1, 0.1)))
            color = tuple(float(v) for v in raw.get("color", (0.7, 0.7, 0.7)))
            mass = float(raw.get("mass") if raw.get("mass") is not None else raw.get("mass_kg", 0.5))
            friction = float(
                raw.get("friction")
                if raw.get("friction") is not None
                else raw.get("static_friction", 0.6)
            )
            surfaces = tuple(raw.get("support_surface") or raw.get("support_surfaces") or ())
            geom_path = self.assets_dir / "source" / asset_id / "geometry.json"
            geom_sha = ""
            if geom_path.is_file():
                geom_sha = hashlib.sha256(geom_path.read_bytes()).hexdigest()
            self._entries[asset_id] = AssetEntry(
                asset_id=asset_id,
                name=str(raw.get("name", asset_id)),
                category=str(raw.get("category", "object")),
                bbox_m=(bbox[0], bbox[1], bbox[2]),
                mass_kg=max(mass, 1e-4),
                friction=max(friction, 0.0),
                primitive=str(raw.get("primitive") or "box").lower(),
                color=(color[0], color[1], color[2]),
                collision_enabled=bool(raw.get("collision_enabled", True)),
                status=str(raw.get("status", "validated")),
                tags=tuple(str(t) for t in raw.get("tags", ())),
                support_surfaces=surfaces,
                has_mesh_geometry=geom_path.is_file(),
                geometry_sha256=geom_sha,
                raw=raw,
            )

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, asset_id: str) -> AssetEntry:
        if asset_id not in self._entries:
            raise KeyError(f"unknown asset_id: {asset_id!r}")
        return self._entries[asset_id]

    def list_all(self) -> list[AssetEntry]:
        return [self._entries[key] for key in sorted(self._entries)]

    def search(
        self,
        query: str = "",
        *,
        category: str | None = None,
        detailed_only: bool = False,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Rank local assets by keyword, Chinese/English aliases, and mesh fidelity."""
        terms = [t for t in re.split(r"[\s,;/，、]+", query.strip().lower()) if t]
        expanded: set[str] = set(terms)
        for zh, aliases in _QUERY_ALIASES.items():
            if zh in query:
                expanded.update(aliases)

        scored: list[tuple[int, str, AssetEntry]] = []
        for entry in self._entries.values():
            if category and entry.category.lower() != category.lower():
                continue
            if detailed_only and not entry.has_mesh_geometry:
                continue
            haystack = " ".join(
                (entry.asset_id, entry.name, entry.category, " ".join(entry.tags))
            ).lower()
            score = 0
            if not expanded:
                score = 2 if entry.has_mesh_geometry else 1
            else:
                for token in expanded:
                    if token == entry.asset_id.lower() or token == entry.category.lower():
                        score += 6
                    elif token in haystack:
                        score += 3
                if score == 0:
                    continue
                if entry.has_mesh_geometry:
                    score += 2
            scored.append((score, entry.asset_id, entry))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [entry.to_summary() for _, _, entry in scored[: max(1, min(limit, 50))]]

    def load_mesh_parts(self, asset_id: str) -> list[dict[str, Any]] | None:
        """Load verified multi-part convex geometry for high-fidelity MJCF export."""
        if asset_id in self._mesh_cache:
            return self._mesh_cache[asset_id]
        folder = self.assets_dir / "source" / asset_id
        geom_path = folder / "geometry.json"
        if not geom_path.is_file():
            self._mesh_cache[asset_id] = None
            return None
        data = geom_path.read_bytes()
        source_path = folder / "SOURCE.json"
        if source_path.is_file():
            manifest = json.loads(source_path.read_text(encoding="utf-8"))
            expected = manifest.get("geometry_sha256")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"geometry_sha256 mismatch for asset {asset_id!r}")
        parts = json.loads(data).get("parts", [])
        if not parts:
            self._mesh_cache[asset_id] = None
            return None
        self._mesh_cache[asset_id] = parts
        return parts

    def build_mjcf_fragment(
        self,
        asset_id: str,
        *,
        name: str,
        pos: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float] | None = None,
        yaw_deg: float = 0.0,
        dynamic: bool = True,
        scale: float = 1.0,
        mass_kg: float | None = None,
        friction: float | None = None,
        rgba: tuple[float, float, float, float] | None = None,
    ) -> tuple[list[ET.Element], ET.Element, bool]:
        """Build MuJoCo `<mesh>` asset elements and one `<body/>` element.

        Returns `(asset_elements, body_element, uses_detailed_mesh)`.
        """
        entry = self.get(asset_id)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive and finite")
        body_name = sanitize_mjcf_name(name)
        quat = quat_wxyz if quat_wxyz is not None else yaw_deg_to_quat_wxyz(yaw_deg)
        norm = math.sqrt(sum(float(v) ** 2 for v in quat))
        if norm <= 0 or not math.isfinite(norm):
            raise ValueError("quaternion must be finite and non-zero")
        quat = tuple(float(v) / norm for v in quat)

        eff_mass = float(mass_kg) if mass_kg is not None else entry.mass_kg * (scale**3)
        eff_friction = float(friction) if friction is not None else entry.friction
        if not math.isfinite(eff_mass) or eff_mass <= 0:
            raise ValueError("mass_kg must be finite and positive")
        if not math.isfinite(eff_friction) or eff_friction < 0:
            raise ValueError("friction must be finite and non-negative")

        bbox = tuple(dim * scale for dim in entry.bbox_m)
        body = ET.Element(
            "body",
            {"name": body_name, "pos": _fmt_nums(pos), "quat": _fmt_nums(quat)},
        )
        if dynamic:
            ET.SubElement(body, "freejoint", {"name": f"joint__{body_name}"})

        parts = self.load_mesh_parts(asset_id)
        asset_elems: list[ET.Element] = []
        if parts:
            if dynamic:
                inertia = (
                    eff_mass * (bbox[1] ** 2 + bbox[2] ** 2) / 12.0,
                    eff_mass * (bbox[0] ** 2 + bbox[2] ** 2) / 12.0,
                    eff_mass * (bbox[0] ** 2 + bbox[1] ** 2) / 12.0,
                )
                ET.SubElement(
                    body,
                    "inertial",
                    {
                        "pos": "0 0 0",
                        "mass": _fmt_nums((eff_mass,)),
                        "diaginertia": _fmt_nums(inertia),
                    },
                )
            for idx, part in enumerate(parts):
                mesh_name = f"mesh__{body_name}_{idx}"
                verts = [[float(c) * scale for c in v] for v in part["vertices"]]
                tris = [
                    (face[0], face[i], face[i + 1])
                    for face in part["faces"]
                    for i in range(1, len(face) - 1)
                ]
                asset_elems.append(
                    ET.Element(
                        "mesh",
                        {
                            "name": mesh_name,
                            "vertex": _fmt_nums(c for v in verts for c in v),
                            "face": " ".join(str(i) for tri in tris for i in tri),
                        },
                    )
                )
                part_rgba = rgba if rgba is not None else (*part.get("color", entry.color), 1.0)
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"geom__{body_name}_{idx}",
                        "type": "mesh",
                        "mesh": mesh_name,
                        "mass": "0",
                        "rgba": _fmt_nums(part_rgba),
                        "friction": _fmt_nums((eff_friction, 0.008, 0.001)),
                        "contype": "1" if entry.collision_enabled else "0",
                        "conaffinity": "1" if entry.collision_enabled else "0",
                    },
                )
            return asset_elems, body, True

        geom_type, size = _primitive_shape(entry.primitive, bbox)
        geom_rgba = rgba if rgba is not None else (*entry.color, 1.0)
        attrs = {
            "name": f"geom__{body_name}",
            "type": geom_type,
            "size": _fmt_nums(size),
            "rgba": _fmt_nums(geom_rgba),
            "friction": _fmt_nums((eff_friction, eff_friction * 0.02, 0.001)),
            "contype": "1" if entry.collision_enabled else "0",
            "conaffinity": "1" if entry.collision_enabled else "0",
        }
        if dynamic:
            attrs["mass"] = _fmt_nums((eff_mass,))
        ET.SubElement(body, "geom", attrs)
        return asset_elems, body, False


def _primitive_shape(primitive: str, bbox: tuple[float, float, float]) -> tuple[str, tuple[float, ...]]:
    if primitive == "sphere":
        return "sphere", (max(bbox) / 2.0,)
    if primitive in {"cylinder", "capsule"}:
        return primitive, (max(bbox[0], bbox[1]) / 2.0, bbox[2] / 2.0)
    return "box", (bbox[0] / 2.0, bbox[1] / 2.0, bbox[2] / 2.0)
