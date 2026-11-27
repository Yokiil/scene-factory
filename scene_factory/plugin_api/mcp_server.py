"""Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 stdio server.

Exposes SceneFactory's Scene-to-MJCF capabilities (asset search, MJCF fragment
assembly, MuJoCo physics verification & rendering, deterministic XML
randomization, and optional end-to-end multimodal XML generation) as a clean
external plugin — decoupled from simulator control or host agent loops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from ..core.assets import AssetCatalog
from ..core.mjcf import FragmentCache, create_base_xml, merge_fragment
from ..core.randomizer import randomize_xml
from ..core.verifier import verify_mujoco_xml


_MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_assets",
        "description": "Search local high-fidelity 3D assets (bbox, physics, multi-part convex meshes).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "default": ""},
                "category": {"type": "string"},
                "detailed_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 12},
            },
        },
    },
    {
        "name": "compose_assets_xml",
        "description": (
            "Quickly assemble a base MuJoCo XML containing one or more local library assets "
            "(with multi-part convex meshes when available)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["objects"],
            "properties": {
                "model_name": {"type": "string", "default": "scene_factory_scene"},
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["asset_id", "name", "pos"],
                        "properties": {
                            "asset_id": {"type": "string"},
                            "name": {"type": "string"},
                            "pos": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                            "yaw_deg": {"type": "number", "default": 0.0},
                            "dynamic": {"type": "boolean", "default": True},
                            "scale": {"type": "number", "default": 1.0},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "verify_mujoco_xml",
        "description": (
            "Compile a MuJoCo XML, run settling simulation, measure penetration/drift/falls, "
            "and optionally render multi-view RGB PNGs."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["xml"],
            "properties": {
                "xml": {"type": "string"},
                "settle_steps": {"type": "integer", "default": 150},
                "render": {"type": "boolean", "default": False},
                "render_dir": {"type": "string"},
            },
        },
    },
    {
        "name": "randomize_mujoco_xml",
        "description": (
            "Generate deterministic collision-checked layout/physics randomizations directly "
            "from an input MuJoCo XML."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["xml"],
            "properties": {
                "xml": {"type": "string"},
                "count": {"type": "integer", "default": 5},
                "seed": {"type": "integer", "default": 42},
                "xy_jitter_m": {"type": "number", "default": 0.08},
                "yaw_jitter_deg": {"type": "number", "default": 25.0},
            },
        },
    },
]


class MCPServer:
    """Stateless JSON-RPC 2.0 handler for MCP initialize / tools/list / tools/call."""

    def __init__(self, catalog: AssetCatalog | None = None) -> None:
        self.catalog = catalog or AssetCatalog()
        self.fragments = FragmentCache(self.catalog)

    def handle_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        req_id = message.get("id")
        params = message.get("params") or {}

        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "scene-factory", "version": "0.2.0"},
                    "capabilities": {"tools": {}},
                },
            )
        if method == "ping":
            return self._ok(req_id, {})
        if method == "tools/list":
            return self._ok(req_id, {"tools": _MCP_TOOLS})
        if method == "tools/call":
            return self._call_tool(req_id, str(params.get("name", "")), dict(params.get("arguments") or {}))
        return self._err(req_id, -32601, f"method not found: {method!r}")

    def _call_tool(self, req_id: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "search_assets":
                data = self.catalog.search(
                    str(args.get("query", "")),
                    category=args.get("category"),
                    detailed_only=bool(args.get("detailed_only", False)),
                    limit=int(args.get("limit", 12)),
                )
                return self._tool_ok(req_id, {"assets": data})

            if name == "compose_assets_xml":
                xml = create_base_xml(str(args.get("model_name", "scene_factory_scene")))
                for obj in args["objects"]:
                    pos = tuple(float(v) for v in obj["pos"])
                    assets, body, detailed, _ = self.fragments.get_or_build(
                        str(obj["asset_id"]),
                        name=str(obj["name"]),
                        pos=(pos[0], pos[1], pos[2]),
                        yaw_deg=float(obj.get("yaw_deg", 0.0)),
                        dynamic=bool(obj.get("dynamic", True)),
                        scale=float(obj.get("scale", 1.0)),
                    )
                    xml = merge_fragment(xml, assets, body, enable_multiccd=detailed)
                return self._tool_ok(req_id, {"xml": xml, "fragment_cache": self.fragments.stats()})

            if name == "verify_mujoco_xml":
                report = verify_mujoco_xml(
                    str(args["xml"]),
                    settle_steps=int(args.get("settle_steps", 150)),
                    render=bool(args.get("render", False)),
                    render_dir=Path(args["render_dir"]) if args.get("render_dir") else None,
                )
                return self._tool_ok(req_id, report.to_dict(include_base64=False))

            if name == "randomize_mujoco_xml":
                variants, stats = randomize_xml(
                    str(args["xml"]),
                    count=int(args.get("count", 5)),
                    seed=int(args.get("seed", 42)),
                    xy_jitter_m=float(args.get("xy_jitter_m", 0.08)),
                    yaw_jitter_deg=float(args.get("yaw_jitter_deg", 25.0)),
                )
                return self._tool_ok(
                    req_id,
                    {
                        "stats": stats,
                        "variants": [
                            {
                                "index": v.index,
                                "seed": v.seed,
                                "xml_sha256": v.xml_sha256,
                                "max_penetration_m": v.max_penetration_m,
                                "xml": v.xml,
                            }
                            for v in variants
                        ],
                    },
                )

            return self._err(req_id, -32602, f"unknown tool: {name!r}")
        except Exception as exc:
            return self._ok(
                req_id,
                {
                    "isError": True,
                    "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                },
            )

    @staticmethod
    def _tool_ok(req_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
            },
        }

    @staticmethod
    def _ok(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def run_stdio_server(
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    *,
    catalog: AssetCatalog | None = None,
) -> int:
    """Serve newline-delimited JSON-RPC 2.0 MCP messages over stdio."""
    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout
    server = MCPServer(catalog=catalog)
    for line in in_stream:
        raw = line.strip()
        if not raw:
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            reply: dict[str, Any] | None = MCPServer._err(None, -32700, f"parse error: {exc}")
        else:
            reply = server.handle_request(message)
        if reply is not None:
            out_stream.write(json.dumps(reply, ensure_ascii=False) + "\n")
            out_stream.flush()
    return 0
