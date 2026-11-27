"""Minimal, orthogonal atomic toolset exposed to VLM/LLM scene agents and MCP clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assets import AssetCatalog
from .mjcf import (
    FragmentCache,
    create_base_xml,
    inspect_xml_summary,
    merge_fragment,
    patch_xml,
)
from .randomizer import RandomizedVariant, randomize_xml
from .verifier import VerificationReport, verify_mujoco_xml


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_assets",
            "description": (
                "Search the local high-fidelity 3D asset catalog by keywords (English or Chinese) "
                "or category. Returns asset_id, dimensions (bbox_m), mass, friction, support "
                "surfaces, and whether a multi-part convex mesh is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords, e.g. 'mug', 'table', '茶几 沙发'."},
                    "category": {"type": "string", "description": "Optional exact category filter."},
                    "detailed_only": {"type": "boolean", "description": "Only return multi-part mesh assets."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_asset",
            "description": (
                "Insert (or replace by name) a local library asset into the current MuJoCo XML "
                "with full multi-part convex mesh geometry when available. Automatically reuses "
                "cached fragments for speed and low token cost."
            ),
            "parameters": {
                "type": "object",
                "required": ["asset_id", "name", "pos"],
                "properties": {
                    "asset_id": {"type": "string"},
                    "name": {"type": "string", "description": "Unique MuJoCo body name in the scene."},
                    "pos": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[x, y, z] body origin in meters.",
                    },
                    "yaw_deg": {"type": "number", "default": 0.0},
                    "dynamic": {"type": "boolean", "default": True},
                    "scale": {"type": "number", "default": 1.0},
                    "mass_kg": {"type": "number"},
                    "friction": {"type": "number"},
                    "rgba": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_xml",
            "description": (
                "Write or completely replace the current MuJoCo MJCF XML string. Use freely for "
                "custom geometry, lighting, walls, cameras, joints, or full hand-authored scenes."
            ),
            "parameters": {
                "type": "object",
                "required": ["xml"],
                "properties": {
                    "xml": {"type": "string", "description": "Complete MuJoCo <mujoco>...</mujoco> XML."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_xml",
            "description": (
                "Apply a targeted substring replacement to the current MuJoCo XML (low token cost "
                "for adjusting poses, sizes, colors, lights, cameras, or physical parameters)."
            ),
            "parameters": {
                "type": "object",
                "required": ["old_text", "new_text"],
                "properties": {
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_xml",
            "description": (
                "Compile the current MuJoCo XML, run settling physics, and render multi-view RGB "
                "images. Returns objective compilation errors, penetration depth, drift/falls, "
                "and rendered views for visual self-critique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "settle_steps": {"type": "integer", "minimum": 0, "maximum": 1000, "default": 150},
                    "render": {"type": "boolean", "default": True},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "randomize_xml",
            "description": (
                "Generate deterministic collision-checked layout/physics randomizations directly "
                "from the current MuJoCo XML using fast in-place MjModel parameter patching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
                    "seed": {"type": "integer", "default": 42},
                    "xy_jitter_m": {"type": "number", "default": 0.08},
                    "yaw_jitter_deg": {"type": "number", "default": 25.0},
                    "mass_scale_min": {"type": "number", "default": 1.0},
                    "mass_scale_max": {"type": "number", "default": 1.0},
                    "friction_scale_min": {"type": "number", "default": 1.0},
                    "friction_scale_max": {"type": "number", "default": 1.0},
                },
            },
        },
    },
]


@dataclass
class SceneWorkspace:
    """Mutable working XML and latest verification/randomization artifacts for one run."""

    xml: str = field(default_factory=create_base_xml)
    render_root: Path | None = None
    verify_count: int = 0
    last_verification: VerificationReport | None = None
    variants: list[RandomizedVariant] = field(default_factory=list)


class SceneToolbox:
    """Executes atomic Scene-to-MJCF tools without hardcoded semantic workflow gates."""

    def __init__(
        self,
        catalog: AssetCatalog | None = None,
        workspace: SceneWorkspace | None = None,
    ) -> None:
        self.catalog = catalog or AssetCatalog()
        self.fragments = FragmentCache(self.catalog)
        self.workspace = workspace or SceneWorkspace()

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run one tool call. Returns `(json_result, rendered_images_for_vlm)`."""
        try:
            if name == "search_assets":
                matches = self.catalog.search(
                    str(arguments.get("query", "")),
                    category=arguments.get("category"),
                    detailed_only=bool(arguments.get("detailed_only", False)),
                    limit=int(arguments.get("limit", 12)),
                )
                return {"ok": True, "count": len(matches), "assets": matches}, []

            if name == "insert_asset":
                pos_raw = arguments["pos"]
                if not isinstance(pos_raw, (list, tuple)) or len(pos_raw) != 3:
                    raise ValueError("pos must be a 3-element [x, y, z] list")
                pos = (float(pos_raw[0]), float(pos_raw[1]), float(pos_raw[2]))
                rgba_raw = arguments.get("rgba")
                rgba = tuple(float(v) for v in rgba_raw) if rgba_raw is not None else None
                assets, body, detailed, hit = self.fragments.get_or_build(
                    str(arguments["asset_id"]),
                    name=str(arguments["name"]),
                    pos=pos,
                    yaw_deg=float(arguments.get("yaw_deg", 0.0)),
                    dynamic=bool(arguments.get("dynamic", True)),
                    scale=float(arguments.get("scale", 1.0)),
                    mass_kg=arguments.get("mass_kg"),
                    friction=arguments.get("friction"),
                    rgba=rgba,
                )
                self.workspace.xml = merge_fragment(
                    self.workspace.xml,
                    assets,
                    body,
                    enable_multiccd=detailed,
                    replace_existing=True,
                )
                summary = inspect_xml_summary(self.workspace.xml)
                return {
                    "ok": True,
                    "inserted": body.attrib.get("name"),
                    "detailed_mesh": detailed,
                    "fragment_cache_hit": hit,
                    "body_count": summary["body_count"],
                    "xml_sha256": summary["xml_sha256"],
                }, []

            if name == "write_xml":
                xml = str(arguments["xml"])
                self.workspace.xml = xml
                summary = inspect_xml_summary(xml)
                return {"ok": True, "summary": summary}, []

            if name == "edit_xml":
                updated = patch_xml(
                    self.workspace.xml,
                    str(arguments["old_text"]),
                    str(arguments["new_text"]),
                )
                self.workspace.xml = updated
                summary = inspect_xml_summary(updated)
                return {"ok": True, "summary": summary}, []

            if name == "verify_xml":
                self.workspace.verify_count += 1
                rdir = (
                    self.workspace.render_root / f"verify_{self.workspace.verify_count:02d}"
                    if self.workspace.render_root
                    else None
                )
                report = verify_mujoco_xml(
                    self.workspace.xml,
                    settle_steps=int(arguments.get("settle_steps", 150)),
                    render=bool(arguments.get("render", True)),
                    render_dir=rdir,
                )
                self.workspace.last_verification = report
                images = [
                    {
                        "camera": item["camera"],
                        "path": item.get("path"),
                        "sha256": item["sha256"],
                        "png_base64": item["png_base64"],
                    }
                    for item in report.renders
                    if "png_base64" in item
                ]
                return {"ok": True, "verification": report.to_dict(include_base64=False)}, images

            if name == "randomize_xml":
                variants, stats = randomize_xml(
                    self.workspace.xml,
                    count=int(arguments.get("count", 5)),
                    seed=int(arguments.get("seed", 42)),
                    xy_jitter_m=float(arguments.get("xy_jitter_m", 0.08)),
                    yaw_jitter_deg=float(arguments.get("yaw_jitter_deg", 25.0)),
                    mass_scale_range=(
                        float(arguments.get("mass_scale_min", 1.0)),
                        float(arguments.get("mass_scale_max", 1.0)),
                    ),
                    friction_scale_range=(
                        float(arguments.get("friction_scale_min", 1.0)),
                        float(arguments.get("friction_scale_max", 1.0)),
                    ),
                )
                self.workspace.variants = variants
                return {
                    "ok": True,
                    "stats": stats,
                    "variants": [
                        {
                            "index": v.index,
                            "seed": v.seed,
                            "xml_sha256": v.xml_sha256,
                            "max_penetration_m": v.max_penetration_m,
                            "attempts_used": v.attempts_used,
                        }
                        for v in variants
                    ],
                }, []

            return {"ok": False, "error": f"unknown tool: {name!r}"}, []
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, []
