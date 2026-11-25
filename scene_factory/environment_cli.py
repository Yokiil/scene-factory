"""CLI entry points for environment packages and constrained scene tooling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import write_json_atomic
from .paths import default_registry_path
from .session_contracts import finite_json


ENVIRONMENT_COMMANDS = {"build", "verify", "archive", "randomize", "assembly-evaluate", "room-benchmark", "dependency-diff", "incremental-compile"}


def add_environment_commands(commands: Any) -> None:
    build = commands.add_parser("build", help="Build a frozen MuJoCo or Isaac environment package")
    build.add_argument("scene", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--backend", choices=("mujoco", "isaac"), default="mujoco")
    build.add_argument("--cache", type=Path)
    build.add_argument("--usd", type=Path, help="Isaac scene USD")
    build.add_argument("--robot-asset-root", type=Path, help="Local Isaac asset root for the Franka USD closure")
    build.add_argument("--backend-options", type=Path, help="JSON file with locked Isaac backend settings")
    verify = commands.add_parser("verify", help="Verify environment bytes and runtime compatibility")
    verify.add_argument("package", type=Path)
    verify.add_argument("--content-only", action="store_true", help="Skip local runtime compatibility/re-export checks")
    archive = commands.add_parser("archive", help="Create a ZIP from a verified environment")
    archive.add_argument("package", type=Path)
    archive.add_argument("--output", type=Path, required=True)
    randomize = commands.add_parser("randomize", help="Sample instance physics without mutating shared assets")
    randomize.add_argument("scene", type=Path)
    randomize.add_argument("config", type=Path)
    randomize.add_argument("--output", type=Path, required=True)
    randomize.add_argument("--seed", type=int, default=0)
    randomize.add_argument("--count", type=int, default=1)
    evaluate = commands.add_parser("assembly-evaluate", help="Evaluate a measured assembly JSON trace")
    evaluate.add_argument("spec", type=Path)
    evaluate.add_argument("trace", type=Path)
    evaluate.add_argument("--output", type=Path)
    room = commands.add_parser("room-benchmark", help="Measure 20/50/100-object proxy background fixtures")
    room.add_argument("--output", type=Path, required=True)
    room.add_argument("--physics-steps", type=int, default=1000)
    diff = commands.add_parser("dependency-diff", help="Plan conservative invalidation after scene changes")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--output", type=Path)
    incremental = commands.add_parser("incremental-compile", help="Compile scene variants with safe episode-boundary MuJoCo updates")
    incremental.add_argument("scenes", type=Path, nargs="+")
    incremental.add_argument("--output", type=Path, required=True)
    incremental.add_argument("--cache", type=Path)
    incremental.add_argument("--mode", choices=("full", "file_cache", "incremental", "partition"), default="incremental")
    incremental.add_argument("--background-id", action="append", default=[])
    incremental.add_argument("--horizon-s", type=float, default=.2)
    incremental.add_argument("--max-linear-speed-mps", type=float, default=2.)


def _read(path: Path) -> Any:
    return finite_json(json.loads(path.read_text(encoding="utf-8-sig")))


def _safe_report_output(output: Path | None, inputs: list[Path]) -> None:
    if output is not None and any(output.resolve() == source.resolve() for source in inputs):
        raise ValueError("report output must not overwrite an input")


def run_environment_command(args: Any) -> int:
    command = args.sandbox_command
    if command == "build":
        from .environment_package import build_environment

        report = build_environment(args.scene, args.output, registry_path=args.registry,
                                   backend=args.backend, cache_dir=args.cache,
                                   usd_path=args.usd, robot_asset_root=args.robot_asset_root,
                                   backend_options=_read(args.backend_options) if args.backend_options else None)
    elif command == "verify":
        from .environment_package import verify_environment

        report = verify_environment(args.package, check_runtime=not args.content_only)
    elif command == "archive":
        from .environment_package import archive_environment

        report = {"archive": str(archive_environment(args.package, args.output)), "valid": True}
    elif command == "randomize":
        from .constrained_randomization import sample_candidate_seeds, sample_instance_physics
        from .registry import AssetRegistry

        if not 1 <= args.count <= 10000 or not 0 <= args.seed < 2**64 or args.seed + args.count > 2**64:
            raise ValueError("randomization requires count in [1,10000] and uint64 seed range")
        if args.output.exists():
            raise FileExistsError("randomization output must be a new directory")
        scene, config = _read(args.scene), _read(args.config)
        registry_path = args.registry or default_registry_path()
        registry = AssetRegistry.load(registry_path)
        if args.count == 1:
            sampled, report = sample_instance_physics(scene, config, args.seed, registry=registry)
        else:
            sampled = None
            report = sample_candidate_seeds(scene, config, list(range(args.seed, args.seed + args.count)), registry=registry)
        args.output.mkdir(parents=True)
        write_json_atomic(args.output / "source_layout.json", scene)
        write_json_atomic(args.output / "config.json", config)
        if sampled is not None:
            write_json_atomic(args.output / "layout.json", sampled)
        write_json_atomic(args.output / "sampling.json", report)
        invalid = not report.get("valid", True) if args.count == 1 else report["counts"]["parameter_invalid"] > 0
        summary = {"output": str(args.output.resolve()), "candidate_count": args.count,
                   "parameter_constraints_passed": not invalid, "physics_validity": "not_verified", "controller_outcome": "not_run"}
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if invalid else 0
    elif command == "assembly-evaluate":
        from .assembly import evaluate_assembly_trace

        _safe_report_output(args.output, [args.spec, args.trace])
        report = evaluate_assembly_trace(_read(args.spec), _read(args.trace))
        if args.output is not None:
            write_json_atomic(args.output, report)
    elif command == "room-benchmark":
        from .room_benchmarks import run_room_benchmarks

        report = run_room_benchmarks(args.output, physics_steps=args.physics_steps)
        report = {"output": str(args.output.resolve()), "scope": report["scope"],
                  "local_robot_task": report["local_robot_task"],
                  "rows": [{key: row[key] for key in ("object_count", "dynamic_count", "static_count", "background_settling", "step_p50_s", "step_p95_s")} for row in report["rows"]]}
    elif command == "dependency-diff":
        from .physics_dependencies import plan_scene_update

        _safe_report_output(args.output, [args.before, args.after])
        report = plan_scene_update(_read(args.before), _read(args.after))
        if args.output is not None:
            write_json_atomic(args.output, report)
    elif command == "incremental-compile":
        from .incremental_compiler import IncrementalMujocoCompiler
        from .registry import AssetRegistry

        if args.output.exists():
            raise FileExistsError("incremental compilation output must be a new directory")
        scenes = [_read(path) for path in args.scenes]
        compiler = IncrementalMujocoCompiler(
            AssetRegistry.load(args.registry or default_registry_path()), mode=args.mode,
            cache_dir=args.cache, background_ids=set(args.background_id),
            horizon_s=args.horizon_s, max_linear_speed_mps=args.max_linear_speed_mps,
        )
        rows = []
        args.output.mkdir(parents=True)
        for index, scene in enumerate(scenes):
            result = compiler.compile(scene)
            directory = args.output / f"{index:04d}"
            directory.mkdir()
            (directory / "scene.xml").write_text(result.xml, encoding="utf-8")
            write_json_atomic(directory / "layout.json", scene)
            write_json_atomic(directory / "compilation.json", result.report)
            rows.append({"index": index, "directory": directory.name,
                         "kind": result.report["kind"], "xml_sha256": result.report["xml_sha256"],
                         "fragment_hits": result.report["fragment_hits"], "fragment_misses": result.report["fragment_misses"],
                         "compile_total_s": result.report["compile_total_s"]})
        report = {"schema_version": "scene_factory.incremental_variants.v1", "status": "compiled",
                  "physics_acceptance": "not_evaluated", "task_acceptance": "not_evaluated",
                  "output": str(args.output.resolve()), "mode": args.mode, "rows": rows}
        write_json_atomic(args.output / "report.json", report)
    else:
        raise ValueError(f"unknown environment command: {command}")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    failed = report.get("valid") is False or report.get("status") in {"failed", "not_verified"}
    if command == "room-benchmark":
        failed = any(row["background_settling"] != "passed" for row in report["rows"])
    return 2 if failed else 0
