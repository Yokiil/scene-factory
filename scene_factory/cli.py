"""Clean v2 CLI for Scene-to-MJCF generation, verification, randomization, traces, and MCP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .cordis import CordisContext, apply_cordis_config, load_cordis_config
from .core import (
    AssetCatalog,
    LLMClient,
    SceneAgent,
    export_sft_jsonl,
    load_llm_config,
    randomize_xml,
    verify_mujoco_xml,
)
from .core.mjcf import write_xml_atomic
from .plugin_api import run_stdio_server
from .spatial import analyze_xml_spatial, generate_xml_spatial_benchmark
from .studio import run_studio


def _default_cordis_path() -> Path | None:
    candidate = Path("config/cordis.toml")
    return candidate if candidate.is_file() else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scene-factory",
        description="Image + text to high-fidelity MuJoCo MJCF XML with Cordis plugin governance.",
    )
    parser.add_argument("--assets", type=Path, help="Override data/assets directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run the multimodal VLM/LLM agent to generate MuJoCo XML.")
    gen.add_argument("--prompt", required=True, help="Scene description text.")
    gen.add_argument("--image", action="append", default=[], help="Reference image path (repeatable).")
    gen.add_argument("--output", type=Path, required=True, help="Output directory for scene.xml, renders, trace.json.")
    gen.add_argument("--max-turns", type=int, default=12)
    gen.add_argument("--randomize", type=int, default=0, help="Also emit N random layout variants.")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--cordis", type=Path, help="Cordis plugin config (.toml or .json).")
    gen.add_argument("--no-plugins", action="store_true", help="Disable all Cordis plugins (naked LLM baseline).")
    gen.add_argument("--llm-config", type=Path, help="Optional config/llm.json override.")

    ver = sub.add_parser("verify", help="Compile, settle, and render a MuJoCo XML file.")
    ver.add_argument("xml", type=Path, help="Path to MuJoCo XML file.")
    ver.add_argument("--settle-steps", type=int, default=150)
    ver.add_argument("--render-dir", type=Path, help="Directory to save rendered PNG views.")
    ver.add_argument("--no-render", action="store_true")

    rnd = sub.add_parser("randomize", help="Generate deterministic random layout/physics XML variants.")
    rnd.add_argument("xml", type=Path, help="Base MuJoCo XML file.")
    rnd.add_argument("--output", type=Path, required=True, help="Output directory for randomized XMLs.")
    rnd.add_argument("--count", type=int, default=5)
    rnd.add_argument("--seed", type=int, default=42)
    rnd.add_argument("--xy-jitter", type=float, default=0.08)
    rnd.add_argument("--yaw-jitter", type=float, default=25.0)
    rnd.add_argument("--mass-scale", type=float, nargs=2, default=[1.0, 1.0], metavar=("MIN", "MAX"))
    rnd.add_argument("--friction-scale", type=float, nargs=2, default=[1.0, 1.0], metavar=("MIN", "MAX"))

    assets = sub.add_parser("assets", help="Search or list local high-fidelity assets.")
    assets.add_argument("--query", default="")
    assets.add_argument("--category")
    assets.add_argument("--detailed-only", action="store_true")
    assets.add_argument("--limit", type=int, default=20)

    plugins = sub.add_parser("plugins", help="Inspect Cordis plugin admission/exit configuration.")
    plugins.add_argument("--cordis", type=Path, help="Cordis config path (default: config/cordis.toml).")

    sft = sub.add_parser("export-sft", help="Export verified agent traces into SFT/RL JSONL.")
    sft.add_argument("traces", nargs="+", type=Path, help="Trace JSON files or directories containing trace.json.")
    sft.add_argument("--output", type=Path, required=True)
    sft.add_argument("--min-score", type=float, default=0.8)

    sp = sub.add_parser("spatial-check", help="Zero-render analytical 3D coordinate & OBB collision check on XML.")
    sp.add_argument("xml", type=Path, help="Path to MuJoCo XML file.")
    sp.add_argument("--crosscheck-mujoco", action="store_true", help="Also compare pairwise distances with MuJoCo.")

    bm = sub.add_parser("benchmark-xml", help="Generate 3-level XML-Spatial-Collision benchmark for S-Space probing.")
    bm.add_argument("--output", type=Path, required=True, help="Output directory for samples.jsonl and manifest.json.")
    bm.add_argument("--per-level", type=int, default=30, help="Minimal-pair groups per difficulty level.")
    bm.add_argument("--seed", type=int, default=42)
    bm.add_argument("--render-images", action="store_true", help="Also render PNG views for cross-modal S-Space alignment.")
    bm.add_argument("--no-mujoco-crosscheck", action="store_true")

    sub.add_parser("mcp", help="Run zero-dependency MCP stdio plugin server for external agents.")

    st = sub.add_parser("studio", help="Launch Precision 3D Studio web UI for SceneFactory v2.")
    st.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    st.add_argument("--port", type=int, default=8788, help="Bind port (default: 8788).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "mcp":
        catalog = AssetCatalog(args.assets) if args.assets else AssetCatalog()
        return run_stdio_server(catalog=catalog)

    if args.command == "studio":
        return run_studio(host=args.host, port=args.port, assets_dir=args.assets)

    if args.command == "assets":
        catalog = AssetCatalog(args.assets) if args.assets else AssetCatalog()
        results = catalog.search(
            args.query,
            category=args.category,
            detailed_only=args.detailed_only,
            limit=args.limit,
        )
        print(json.dumps({"total_catalog": len(catalog), "matches": results}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "plugins":
        path = args.cordis or _default_cordis_path()
        if path is None:
            print(json.dumps({"config": None, "plugins": []}, indent=2))
            return 0
        declarations = load_cordis_config(path)
        print(json.dumps({"config": str(path), "plugins": declarations}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "verify":
        xml = args.xml.read_text(encoding="utf-8")
        report = verify_mujoco_xml(
            xml,
            settle_steps=args.settle_steps,
            render=not args.no_render,
            render_dir=args.render_dir,
        )
        print(json.dumps(report.to_dict(include_base64=False), ensure_ascii=False, indent=2))
        return 0 if report.compiled and report.settled_stable else 2

    if args.command == "randomize":
        xml = args.xml.read_text(encoding="utf-8")
        variants, stats = randomize_xml(
            xml,
            count=args.count,
            seed=args.seed,
            xy_jitter_m=args.xy_jitter,
            yaw_jitter_deg=args.yaw_jitter,
            mass_scale_range=(args.mass_scale[0], args.mass_scale[1]),
            friction_scale_range=(args.friction_scale[0], args.friction_scale[1]),
        )
        args.output.mkdir(parents=True, exist_ok=True)
        written = []
        for var in variants:
            out_path = write_xml_atomic(args.output / f"variant_{var.index:03d}_seed{var.seed}.xml", var.xml)
            written.append(
                {
                    "index": var.index,
                    "seed": var.seed,
                    "path": str(out_path.resolve()),
                    "xml_sha256": var.xml_sha256,
                    "max_penetration_m": var.max_penetration_m,
                }
            )
        print(json.dumps({"stats": stats, "variants": written}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-sft":
        trace_files: list[Path] = []
        for entry in args.traces:
            if entry.is_dir():
                trace_files.extend(sorted(entry.rglob("trace.json")))
            elif entry.is_file():
                trace_files.append(entry)
        summary = export_sft_jsonl(trace_files, args.output, min_score=args.min_score)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "spatial-check":
        xml = args.xml.read_text(encoding="utf-8")
        analysis = analyze_xml_spatial(xml, crosscheck_mujoco=args.crosscheck_mujoco)
        print(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2))
        return 0 if analysis.assembly_valid else 2

    if args.command == "benchmark-xml":
        catalog = AssetCatalog(args.assets) if args.assets else AssetCatalog()
        manifest = generate_xml_spatial_benchmark(
            args.output,
            per_level=args.per_level,
            seed=args.seed,
            render_images=args.render_images,
            crosscheck_mujoco=not args.no_mujoco_crosscheck,
            catalog=catalog,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate":
        catalog = AssetCatalog(args.assets) if args.assets else AssetCatalog()
        cordis = CordisContext()
        cordis_path = None if args.no_plugins else (args.cordis or _default_cordis_path())
        apply_cordis_config(cordis, cordis_path, disable_all=args.no_plugins)
        client = LLMClient(load_llm_config(args.llm_config))
        agent = SceneAgent(client, catalog=catalog, cordis=cordis, max_turns=args.max_turns)
        result = agent.run(
            args.prompt,
            images=args.image,
            output_dir=args.output,
            randomize_count=args.randomize,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "xml_sha256": result.xml_sha256,
                    "compiled": result.verification.compiled,
                    "settled_stable": result.verification.settled_stable,
                    "score": result.verification.score,
                    "turns": result.trace.turns,
                    "variant_count": len(result.variants),
                    "cordis_plugins": [p["name"] for p in result.trace.cordis_plugins],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.verification.compiled else 2

    parser.print_help(sys.stderr)
    return 1
