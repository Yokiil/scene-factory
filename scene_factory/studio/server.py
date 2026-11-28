"""Thin stdlib HTTP server for SceneFactory v2 Precision 3D Studio.

Bridges the web workspace directly to v2 core (`AssetCatalog`, `FragmentCache`,
`verify_mujoco_xml`, `randomize_xml`, `SceneAgent`), `CordisContext`, and
`scene_factory.spatial` without introducing any framework dependencies or
coupling UI state into core modules.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import tempfile
import threading
import xml.etree.ElementTree as ET
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from ..cordis import CordisContext, apply_cordis_config, load_cordis_config
from ..core import (
    AssetCatalog,
    FragmentCache,
    LLMClient,
    LLMResponse,
    SceneAgent,
    ToolCall,
    create_base_xml,
    load_llm_config,
    merge_fragment,
    randomize_xml,
    verify_mujoco_xml,
)
from ..spatial.benchmark import _build_level2_nested_pair
from ..spatial.geometry import (
    _EYE,
    _add,
    _element_rotation,
    _mat_mul,
    _mat_vec,
    _vec3,
    analyze_xml_spatial,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"


def extract_render_geoms(xml: str) -> dict[str, Any]:
    """Parse all MuJoCo bodies, geoms, and inline meshes into world-frame 3D viewer primitives."""
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    angle_degrees = not (
        compiler is not None and compiler.attrib.get("angle", "degree").lower() == "radian"
    )

    meshes: dict[str, dict[str, list[float] | list[int]]] = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = mesh.attrib.get("name", "")
            raw_v = mesh.attrib.get("vertex", "")
            raw_f = mesh.attrib.get("face", "")
            if not name or not raw_v:
                continue
            verts = [float(v) for v in raw_v.split()]
            faces = [int(i) for i in raw_f.split()] if raw_f else []
            meshes[name] = {"vertices": verts, "faces": faces}

    world = root.find("worldbody")
    if world is None:
        return {"bodies": [], "meshes": {}}

    bodies_out: list[dict[str, Any]] = []

    def _walk(
        node: ET.Element,
        parent_name: str | None,
        parent_pos: tuple[float, float, float],
        parent_rot: Any,
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

            geoms_out: list[dict[str, Any]] = []
            for gi, geom in enumerate(child.findall("geom")):
                gtype = geom.attrib.get("type", "box").lower()
                if gtype == "plane":
                    continue
                gpos_l = _vec3(geom.attrib.get("pos"))
                grot_l = _element_rotation(geom, angle_degrees=angle_degrees)
                gpos_w = _add(wpos, _mat_vec(wrot, gpos_l))
                grot_w = _mat_mul(wrot, grot_l)
                rgba_vals = [float(v) for v in geom.attrib.get("rgba", "0.62 0.66 0.72 1").split()]
                if len(rgba_vals) < 4:
                    rgba_vals = [0.62, 0.66, 0.72, 1.0]
                size_vals = [float(v) for v in geom.attrib.get("size", "").split()]
                geoms_out.append(
                    {
                        "name": geom.attrib.get("name") or f"{bname}_g{gi}",
                        "type": gtype,
                        "mesh": geom.attrib.get("mesh"),
                        "size": size_vals,
                        "pos": [round(v, 6) for v in gpos_w],
                        "rot": [[round(c, 6) for c in row] for row in grot_w],
                        "rgba": rgba_vals[:4],
                    }
                )

            bodies_out.append(
                {
                    "name": bname,
                    "parent": parent_name,
                    "depth": depth,
                    "dynamic": dynamic,
                    "world_pos": [round(v, 6) for v in wpos],
                    "world_rot": [[round(c, 6) for c in row] for row in wrot],
                    "geoms": geoms_out,
                }
            )
            _walk(child, bname, wpos, wrot, depth + 1, counter)

    _walk(world, None, (0.0, 0.0, 0.0), _EYE, 0, [0])
    return {"bodies": bodies_out, "meshes": meshes}


class _DemoSelfCorrectingVLM:
    """Offline showcase VLM demonstrating real multi-turn MuJoCo visual/physical self-correction."""

    model = "demo-self-correct-vlm"

    def __init__(self, catalog: AssetCatalog, prompt: str) -> None:
        self.catalog = catalog
        self.prompt = prompt
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.calls += 1
        table_h = self.catalog.get("kd_table_01").bbox_m[2]
        mug_h = self.catalog.get("kd_mug_01").bbox_m[2]
        bowl_h = self.catalog.get("kd_bowl_01").bbox_m[2]
        plate_h = self.catalog.get("kd_plate_01").bbox_m[2]
        bottle_h = self.catalog.get("kd_bottle_01").bbox_m[2]

        if self.calls == 1:
            return LLMResponse(
                content="先检索本地高保真多凸包餐厨资产，获取精确尺寸与支撑面高度。",
                tool_calls=(
                    ToolCall("call_1", "search_assets", {"query": "table mug bowl plate bottle", "detailed_only": True}),
                ),
                usage={"prompt_tokens": 310, "completion_tokens": 42},
            )
        if self.calls == 2:
            return LLMResponse(
                content="先组装餐桌和 4 件桌面器物；故意将马克杯初始高度偏低 3.5 cm，模拟初次布局穿模并送入 MuJoCo 客观验证。",
                tool_calls=(
                    ToolCall(
                        "call_2",
                        "insert_asset",
                        {"asset_id": "kd_table_01", "name": "dining_table", "pos": [0.0, 0.0, table_h / 2.0], "dynamic": False},
                    ),
                    ToolCall(
                        "call_3",
                        "insert_asset",
                        {"asset_id": "kd_mug_01", "name": "blue_mug", "pos": [-0.26, -0.06, table_h - 0.035], "yaw_deg": 24, "dynamic": True},
                    ),
                    ToolCall(
                        "call_4",
                        "insert_asset",
                        {"asset_id": "kd_bowl_01", "name": "ceramic_bowl", "pos": [0.04, 0.05, table_h + bowl_h / 2.0 + 0.002], "yaw_deg": -15, "dynamic": True},
                    ),
                    ToolCall(
                        "call_5",
                        "insert_asset",
                        {"asset_id": "kd_plate_01", "name": "dinner_plate", "pos": [0.34, -0.04, table_h + plate_h / 2.0 + 0.002], "dynamic": True},
                    ),
                    ToolCall(
                        "call_6",
                        "insert_asset",
                        {"asset_id": "kd_bottle_01", "name": "glass_bottle", "pos": [-0.08, 0.22, table_h + bottle_h / 2.0 + 0.002], "dynamic": True},
                    ),
                    ToolCall("call_7", "verify_xml", {"settle_steps": 120, "render": True}),
                ),
                usage={"prompt_tokens": 640, "completion_tokens": 168},
            )
        if self.calls == 3:
            return LLMResponse(
                content=(
                    "MuJoCo 反馈与渲染图显示 `blue_mug` 与桌面发生显著穿模并在静置时弹跳位移。"
                    "现按 `dining_table` 顶面高度 + `kd_mug_01` 半高 + 2 mm 间隙将其抬正，并重新运行 MuJoCo 静置与渲染验证。"
                ),
                tool_calls=(
                    ToolCall(
                        "call_8",
                        "insert_asset",
                        {"asset_id": "kd_mug_01", "name": "blue_mug", "pos": [-0.26, -0.06, table_h + mug_h / 2.0 + 0.002], "yaw_deg": 24, "dynamic": True},
                    ),
                    ToolCall("call_9", "verify_xml", {"settle_steps": 120, "render": True}),
                ),
                usage={"prompt_tokens": 1020, "completion_tokens": 96},
            )
        return LLMResponse(
            content="已完成多轮物理验证与视觉自纠：全部 5 个多凸包物体静置稳定、0 穿模、0 悬空，Score = 1.00。",
            tool_calls=(),
            usage={"prompt_tokens": 1240, "completion_tokens": 45},
        )


class StudioBackend:
    """Thread-safe local backend serving v2 Studio actions and Cordis plugin state."""

    def __init__(
        self,
        *,
        assets_dir: str | Path | None = None,
        cordis_config: str | Path | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.catalog = AssetCatalog(assets_dir) if assets_dir else AssetCatalog()
        self.fragments = FragmentCache(self.catalog)
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="sf_studio_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir = self.work_dir / "renders"
        self.render_dir.mkdir(exist_ok=True)

        self.cordis = CordisContext()
        default_cfg = Path("config/cordis.toml")
        self.cordis_path = Path(cordis_config) if cordis_config else (default_cfg if default_cfg.is_file() else None)
        self._plugin_catalog: list[dict[str, Any]] = []
        if self.cordis_path and self.cordis_path.is_file():
            self._plugin_catalog = load_cordis_config(self.cordis_path)
            apply_cordis_config(self.cordis, self.cordis_path)

        self._lock = threading.Lock()
        self.latest_trace: dict[str, Any] | None = None

    def _register_render_urls(self, renders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in renders:
            copy_item = {k: v for k, v in item.items() if k != "png_base64"}
            path_str = item.get("path")
            if path_str:
                p = Path(path_str)
                try:
                    rel = p.resolve().relative_to(self.render_dir.resolve()).as_posix()
                    copy_item["url"] = f"/renders/{rel}"
                except ValueError:
                    pass
            elif item.get("png_base64"):
                copy_item["url"] = f"data:image/png;base64,{item['png_base64']}"
            out.append(copy_item)
        return out

    def cordis_state(self) -> list[dict[str, Any]]:
        active = set(self.cordis.mounted_names())
        rows = []
        for decl in self._plugin_catalog:
            rows.append({**decl, "enabled": decl["name"] in active})
        return rows

    def toggle_cordis_plugin(self, name: str, enabled: bool) -> list[dict[str, Any]]:
        with self._lock:
            target = next((d for d in self._plugin_catalog if d["name"] == name), None)
            if target is None:
                raise KeyError(f"unknown Cordis plugin: {name!r}")
            target["enabled"] = bool(enabled)
            # Re-apply cleanly from in-memory catalog without mutating disk config unexpectedly.
            tmp_cfg = self.work_dir / "cordis_runtime.json"
            tmp_cfg.write_text(json.dumps({"plugins": self._plugin_catalog}), encoding="utf-8")
            apply_cordis_config(self.cordis, tmp_cfg)
            return self.cordis_state()

    def status(self) -> dict[str, Any]:
        llm_cfg = load_llm_config()
        return {
            "version": "0.2.0",
            "asset_count": len(self.catalog),
            "detailed_asset_count": sum(1 for a in self.catalog.list_all() if a.has_mesh_geometry),
            "llm": {
                "configured": bool(llm_cfg.base_url and llm_cfg.model),
                "model": llm_cfg.model or "未配置（可用 Demo Trace 体验闭环）",
                "endpoint": llm_cfg.endpoint if llm_cfg.base_url else "",
            },
            "cordis_plugins": self.cordis_state(),
            "presets": [
                {
                    "id": "kitchen_high_fidelity",
                    "name": "高保真餐厨桌面",
                    "tag": "Multi-Part Mesh · Stable",
                    "description": "5 件原创多凸包网格资产 + 完整多轮 VLM 自纠 Trace",
                },
                {
                    "id": "level2_nested_collision",
                    "name": "嵌套坐标系穿模诊断",
                    "tag": "Level 2 SE(3) · Collision",
                    "description": "3 层父子嵌套坐标系穿模案例，展示零渲染 3D OBB 红框与分离轴",
                },
                {
                    "id": "level2_nested_floating",
                    "name": "嵌套物体悬空检测",
                    "tag": "Level 2 SE(3) · Floating",
                    "description": "动态物体悬空间隙检测与支撑面定位",
                },
                {
                    "id": "dessert_display",
                    "name": "三层甜点展示架",
                    "tag": "Multi-Part Mesh · Tiered",
                    "description": "三层木质展示架与多凸包甜点精细摆放",
                },
            ],
        }

    def build_preset(self, preset_id: str) -> dict[str, Any]:
        import random

        if preset_id == "kitchen_high_fidelity":
            return self.run_demo_trace("一张餐桌，上面摆放蓝色马克杯、陶瓷碗、餐盘和玻璃瓶，要求多凸包高保真且零穿模。")

        if preset_id == "level2_nested_collision":
            xml, _, _ = _build_level2_nested_pair(random.Random(42), "colliding")
            return self.inspect_xml(xml, title="Level 2 · 嵌套坐标系穿模案例", settle_steps=80, render=True)

        if preset_id == "level2_nested_floating":
            xml, _, _ = _build_level2_nested_pair(random.Random(43), "floating")
            return self.inspect_xml(xml, title="Level 2 · 嵌套物体悬空案例", settle_steps=80, render=True)

        if preset_id == "dessert_display":
            xml = create_base_xml("dessert_display")
            stand = self.catalog.get("dessert_tiered_stand_01")
            a_s, b_s, d_s, _ = self.fragments.get_or_build(
                "dessert_tiered_stand_01",
                name="tiered_stand",
                pos=(0.0, 0.0, stand.bbox_m[2] / 2.0),
                dynamic=False,
            )
            xml = merge_fragment(xml, a_s, b_s, enable_multiccd=d_s)
            placements = [
                ("dessert_cream_cupcake_01", "cupcake_bottom", (-0.20, -0.14, 0.080 + 0.055 + 0.002)),
                ("dessert_strawberry_square_01", "strawberry_bottom", (0.19, -0.15, 0.080 + 0.045 + 0.002)),
                ("dessert_pink_mini_cake_01", "pink_cake_mid", (-0.15, -0.13, 0.310 + 0.045 + 0.002)),
                ("dessert_cream_cupcake_01", "cupcake_mid", (0.15, 0.12, 0.310 + 0.055 + 0.002)),
                ("dessert_strawberry_square_01", "strawberry_top", (0.0, -0.10, 0.540 + 0.045 + 0.002)),
            ]
            for aid, name, pos in placements:
                a_i, b_i, d_i, _ = self.fragments.get_or_build(aid, name=name, pos=pos, dynamic=True)
                xml = merge_fragment(xml, a_i, b_i, enable_multiccd=d_i)
            return self.inspect_xml(xml, title="三层甜点展示架", settle_steps=80, render=True)

        raise KeyError(f"unknown preset: {preset_id!r}")

    def inspect_xml(
        self,
        xml: str,
        *,
        title: str = "Workspace Scene",
        settle_steps: int = 120,
        render: bool = True,
    ) -> dict[str, Any]:
        xml_hash = hashlib.sha256(xml.encode("utf-8")).hexdigest()[:12]
        rdir = self.render_dir / f"scene_{xml_hash}"
        analysis = analyze_xml_spatial(xml, crosscheck_mujoco=True)
        verification = verify_mujoco_xml(
            xml, settle_steps=settle_steps, render=render, render_dir=rdir if render else None
        )
        scene_3d = extract_render_geoms(xml)
        ver_dict = verification.to_dict(include_base64=False)
        ver_dict["renders"] = self._register_render_urls(ver_dict["renders"])
        return {
            "title": title,
            "xml": xml,
            "xml_sha256": verification.xml_sha256,
            "analysis": analysis.to_dict(),
            "verification": ver_dict,
            "scene_3d": scene_3d,
            "trace": None,
            "variants": [],
        }

    def run_demo_trace(self, prompt: str, *, randomize_count: int = 4, seed: int = 42) -> dict[str, Any]:
        with self._lock:
            run_id = hashlib.sha256(f"{prompt}:{seed}:{len(self.cordis.mounted_names())}".encode()).hexdigest()[:10]
            out_dir = self.work_dir / f"demo_{run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            vlm = _DemoSelfCorrectingVLM(self.catalog, prompt)
            agent = SceneAgent(vlm, catalog=self.catalog, cordis=self.cordis, max_turns=6)
            result = agent.run(
                prompt,
                output_dir=out_dir,
                randomize_count=randomize_count,
                seed=seed,
            )
            return self._package_agent_result(result, title="高保真餐厨桌面 · 自纠闭环")

    def run_generate(
        self,
        prompt: str,
        *,
        images_data_urls: list[str] | None = None,
        randomize_count: int = 0,
        seed: int = 42,
        no_plugins: bool = False,
        demo_mode: bool = False,
    ) -> dict[str, Any]:
        if demo_mode:
            return self.run_demo_trace(prompt, randomize_count=max(randomize_count, 4), seed=seed)

        cfg = load_llm_config()
        if not cfg.base_url or not cfg.model:
            return {
                "ok": False,
                "needs_llm_config": True,
                "error": "尚未配置多模态 LLM（config/llm.json 或 SCENE_FACTORY_LLM_BASE_URL / MODEL）。可点击「离线 Demo 闭环」体验完整多轮自纠与 MuJoCo 渲染。",
            }

        saved_images: list[Path] = []
        img_dir = self.work_dir / "uploads"
        img_dir.mkdir(exist_ok=True)
        for idx, data_url in enumerate(images_data_urls or ()):
            if "," in data_url:
                _, b64 = data_url.split(",", 1)
                raw = base64.b64decode(b64)
                p = img_dir / f"ref_{idx}_{hashlib.sha256(raw).hexdigest()[:8]}.png"
                p.write_bytes(raw)
                saved_images.append(p)

        with self._lock:
            ctx = CordisContext() if no_plugins else self.cordis
            out_dir = self.work_dir / f"gen_{hashlib.sha256(prompt.encode()).hexdigest()[:10]}"
            agent = SceneAgent(LLMClient(cfg), catalog=self.catalog, cordis=ctx, max_turns=10)
            result = agent.run(
                prompt,
                images=saved_images,
                output_dir=out_dir,
                randomize_count=randomize_count,
                seed=seed,
            )
            return self._package_agent_result(result, title=prompt[:32])

    def _package_agent_result(self, result: Any, *, title: str) -> dict[str, Any]:
        # Copy generated renders under self.render_dir so /renders/<rel> serves them cleanly.
        run_tag = result.xml_sha256[:10]
        target_rdir = self.render_dir / f"trace_{run_tag}"
        target_rdir.mkdir(parents=True, exist_ok=True)
        if result.output_dir and (result.output_dir / "renders").is_dir():
            for png in (result.output_dir / "renders").rglob("*.png"):
                rel = png.relative_to(result.output_dir / "renders")
                dest = target_rdir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(png.read_bytes())

        trace_dict = result.trace.to_dict()
        # Rewrite image_ref paths in trace messages into browser-loadable /renders/... URLs.
        for msg in trace_dict.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_ref" and part.get("path") and result.output_dir:
                        try:
                            rel = Path(part["path"]).resolve().relative_to(
                                (result.output_dir / "renders").resolve()
                            )
                            part["url"] = f"/renders/trace_{run_tag}/{rel.as_posix()}"
                        except ValueError:
                            pass

        self.latest_trace = trace_dict
        analysis = analyze_xml_spatial(result.xml, crosscheck_mujoco=True)
        ver_dict = result.verification.to_dict(include_base64=False)
        for item in ver_dict.get("renders", []):
            if item.get("path") and result.output_dir:
                try:
                    rel = Path(item["path"]).resolve().relative_to(
                        (result.output_dir / "renders").resolve()
                    )
                    item["url"] = f"/renders/trace_{run_tag}/{rel.as_posix()}"
                except ValueError:
                    pass

        variant_payloads = []
        for var in result.variants:
            vrdir = self.render_dir / f"var_{var.xml_sha256[:10]}"
            vrep = verify_mujoco_xml(var.xml, settle_steps=0, render=True, render_dir=vrdir, width=320, height=240)
            vrenders = self._register_render_urls(vrep.to_dict(include_base64=False)["renders"])
            variant_payloads.append(
                {
                    "index": var.index,
                    "seed": var.seed,
                    "xml": var.xml,
                    "xml_sha256": var.xml_sha256,
                    "max_penetration_m": var.max_penetration_m,
                    "attempts_used": var.attempts_used,
                    "thumb_url": vrenders[0]["url"] if vrenders else None,
                }
            )

        return {
            "ok": True,
            "title": title,
            "xml": result.xml,
            "xml_sha256": result.xml_sha256,
            "analysis": analysis.to_dict(),
            "verification": ver_dict,
            "scene_3d": extract_render_geoms(result.xml),
            "trace": trace_dict,
            "variants": variant_payloads,
        }

    def run_randomize(self, payload: dict[str, Any]) -> dict[str, Any]:
        xml = str(payload["xml"])
        count = max(1, min(int(payload.get("count", 6)), 24))
        seed = int(payload.get("seed", 42))
        xy_jitter = float(payload.get("xy_jitter_m", 0.08))
        yaw_jitter = float(payload.get("yaw_jitter_deg", 30.0))
        m_min = float(payload.get("mass_scale_min", 0.9))
        m_max = float(payload.get("mass_scale_max", 1.1))
        variants, stats = randomize_xml(
            xml,
            count=count,
            seed=seed,
            xy_jitter_m=xy_jitter,
            yaw_jitter_deg=yaw_jitter,
            mass_scale_range=(m_min, m_max),
        )
        items = []
        for var in variants:
            vrdir = self.render_dir / f"var_{var.xml_sha256[:10]}"
            vrep = verify_mujoco_xml(var.xml, settle_steps=0, render=True, render_dir=vrdir, width=320, height=240)
            vrenders = self._register_render_urls(vrep.to_dict(include_base64=False)["renders"])
            items.append(
                {
                    "index": var.index,
                    "seed": var.seed,
                    "xml": var.xml,
                    "xml_sha256": var.xml_sha256,
                    "max_penetration_m": var.max_penetration_m,
                    "attempts_used": var.attempts_used,
                    "thumb_url": vrenders[0]["url"] if vrenders else None,
                }
            )
        return {"ok": True, "stats": stats, "variants": items}


def create_studio_server(
    host: str = "127.0.0.1",
    port: int = 8788,
    *,
    backend: StudioBackend | None = None,
) -> ThreadingHTTPServer:
    app = backend or StudioBackend()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime, _ = mimetypes.guess_type(path.name)
            if path.suffix == ".js":
                mime = "application/javascript; charset=utf-8"
            elif path.suffix == ".css":
                mime = "text/css; charset=utf-8"
            elif path.suffix == ".html":
                mime = "text/html; charset=utf-8"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16 * 1024 * 1024:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path == "/api/status":
                return self._send_json(app.status())
            if path == "/api/assets":
                qs = parse_qs(parsed.query)
                query = qs.get("query", [""])[0]
                limit = int(qs.get("limit", ["24"])[0])
                return self._send_json({"assets": app.catalog.search(query, limit=limit)})
            if path.startswith("/api/preset/"):
                pid = path.removeprefix("/api/preset/").strip("/")
                try:
                    return self._send_json(app.build_preset(pid))
                except Exception as exc:
                    return self._send_json({"ok": False, "error": str(exc)}, status=400)
            if path.startswith("/renders/"):
                rel = path.removeprefix("/renders/")
                candidate = (app.render_dir / rel).resolve()
                if candidate.is_relative_to(app.render_dir.resolve()):
                    return self._send_file(candidate)
                return self.send_error(HTTPStatus.FORBIDDEN)

            rel_static = "index.html" if path in {"", "/"} else path.lstrip("/")
            candidate = (STATIC_DIR / rel_static).resolve()
            if candidate.is_relative_to(STATIC_DIR.resolve()):
                return self._send_file(candidate)
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            try:
                body = self._read_json()
                if parsed.path == "/api/analyze":
                    res = app.inspect_xml(
                        str(body["xml"]),
                        title=str(body.get("title", "Edited XML")),
                        settle_steps=int(body.get("settle_steps", 120)),
                        render=bool(body.get("render", True)),
                    )
                    return self._send_json({"ok": True, **res})
                if parsed.path == "/api/randomize":
                    return self._send_json(app.run_randomize(body))
                if parsed.path == "/api/generate":
                    res = app.run_generate(
                        str(body.get("prompt", "")),
                        images_data_urls=body.get("images") or [],
                        randomize_count=int(body.get("randomize_count", 0)),
                        seed=int(body.get("seed", 42)),
                        no_plugins=bool(body.get("no_plugins", False)),
                        demo_mode=bool(body.get("demo_mode", False)),
                    )
                    return self._send_json(res)
                if parsed.path == "/api/cordis/toggle":
                    plugins = app.toggle_cordis_plugin(str(body["name"]), bool(body["enabled"]))
                    return self._send_json({"ok": True, "cordis_plugins": plugins})
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    return ThreadingHTTPServer((host, port), Handler)


def run_studio(host: str = "127.0.0.1", port: int = 8788, *, assets_dir: Path | None = None) -> int:
    backend = StudioBackend(assets_dir=assets_dir)
    server = create_studio_server(host, port, backend=backend)
    print(f"SceneFactory v2 Studio running at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
