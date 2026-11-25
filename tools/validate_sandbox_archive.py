"""Verify, safely extract, and HTTP-smoke a source delivery using extracted code.

The Python interpreter supplies dependencies; SceneFactory is imported only from
the extracted archive. This CPU/state-only check makes no Linux/GPU/robot claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.package_sandbox_delivery import MANIFEST, PREFIX, SCHEMA, _safe_relative, verify_delivery


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("output path must not contain symbolic links or junctions")


def _portable_path(name: str) -> None:
    relative = _safe_relative(name)
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    for part in relative.parts:
        if part.endswith((".", " ")) or part.split(".")[0].casefold() in reserved:
            raise ValueError("archive member uses a nonportable Windows path")


def extract_checked(archive: Path, destination: Path) -> dict:
    """Extract only a verified manifest into a new directory, without deleting."""
    verification = verify_delivery(archive)
    if not verification["valid"]:
        raise ValueError("delivery verification failed: " + "; ".join(verification["errors"]))
    _reject_links(destination)
    if destination.exists():
        raise FileExistsError("extraction directory must be new")
    before = file_sha256(archive)
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read(MANIFEST))
        # Validate every portable path before creating any extracted file.
        for item in source.infolist():
            _portable_path(item.filename)
        expected = {PREFIX + entry["path"]: entry for entry in manifest["files"]}
        destination.mkdir(parents=True)
        for item in source.infolist():
            target = destination.joinpath(*_safe_relative(item.filename).parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("extraction target escapes controlled directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            if item.filename != MANIFEST:
                entry = expected[item.filename]
                if target.stat().st_size != entry["size"] or file_sha256(target) != entry["sha256"]:
                    raise ValueError("extracted file differs from manifest: " + entry["path"])
    if before != file_sha256(archive):
        raise ValueError("archive changed during extraction")
    return {**verification, "archive_sha256": before, "extracted_manifest_files_match": True}


SMOKE_SOURCE = r'''
from pathlib import Path
import importlib.metadata
import json
import sys
import threading
import time

SOURCE = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(SOURCE))

import scene_factory
from scene_factory.runtime_service import RuntimeService, serve
from scene_factory.runtime_client import RuntimeClient


def run():
    WORK.mkdir(parents=True, exist_ok=False)
    module = Path(scene_factory.__file__).resolve()
    assert module.is_relative_to(SOURCE), 'SceneFactory imported outside extracted source'
    scene = {'scene_id': 'extracted-archive-http-smoke', 'room_dimensions_m': [2, 2, 3],
             'objects': [{'object_id': 'part', 'asset_id': 'mug_blue', 'dynamic': True,
                          'bbox_m': [.04, .04, .06],
                          'pose': {'position': [0, 0, .031], 'yaw_deg': 0}}], 'task': {}}
    scene_path = WORK / 'layout.json'
    scene_path.write_text(json.dumps(scene), encoding='utf-8')
    manager = RuntimeService(WORK / 'service', allowed_scene_root=WORK,
                             max_sessions=1, cpu_budget=1, memory_budget_mb=1024,
                             mujoco_worker_pool=True)
    server = serve(manager, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RuntimeClient(f'http://127.0.0.1:{server.server_port}')
    sid = None
    report = {'schema_version': 'scene_factory.extracted_archive_http_smoke.v1',
              'backend': 'mujoco', 'engine_version': importlib.metadata.version('mujoco'),
              'rendering': False, 'source_root': str(SOURCE), 'imported_module': str(module),
              'python_executable': sys.executable, 'robot_task_acceptance': 'not_evaluated'}
    try:
        build = client.build({'idempotency_key': 'build', 'scene_path': str(scene_path)})
        deadline = time.monotonic() + 30
        while build['status'] in {'queued', 'running'} and time.monotonic() < deadline:
            time.sleep(.05)
            build = client.build_status(build['build_id'])
        assert build['status'] == 'completed', build
        report['build'] = build
        request = {'environment_path': build['result']['package_dir'], 'backend': 'mujoco',
                   'history_limit': 128, 'resources': {'cpu_threads': 1, 'memory_mb': 256}}
        created = client.create({**request, 'idempotency_key': 'create-1'})
        sid = created['session_id']
        assert client.create({**request, 'idempotency_key': 'create-1'})['session_id'] == sid
        initial = client.reset(sid, {'idempotency_key': 'reset', 'seed': 42})
        action = {'episode_id': initial['episode_id'], 'sequence_id': 0,
                  'idempotency_key': 'step', 'steps': 3,
                  'action': {'mode': 'object_forces', 'payload': {'object_forces': {'part': [1, 0, 8]}}}}
        stepped = client.step(sid, action)
        assert client.step(sid, action) == stepped
        assert stepped['step_index'] == 3
        assert stepped['observation']['objects'] != initial['observation']['objects']
        assert client.observe(sid) == stepped
        saved = client.snapshot(sid, {'idempotency_key': 'snapshot', 'level': 'runtime_state'})
        client.step(sid, {**action, 'sequence_id': 1, 'idempotency_key': 'step-2'})
        restored = client.restore(sid, {'idempotency_key': 'restore', 'snapshot': saved})
        assert restored['observation']['objects'] == stepped['observation']['objects']
        assert restored['episode_id'] != initial['episode_id']
        first_closed = client.close(sid)
        assert first_closed['closed'] and first_closed['worker_disposition'] == 'pooled'
        sid = None
        second = client.create({**request, 'idempotency_key': 'create-2'})
        sid = second['session_id']
        reset = client.reset(sid, {'idempotency_key': 'reset', 'seed': 42})
        assert second['worker_reused'] and second['worker_pid'] == created['worker_pid']
        assert reset['info']['model_compile_count'] == 1
        assert reset['observation']['objects'] == initial['observation']['objects']
        report.update(status='passed', session_count=2, initial_worker_pid=created['worker_pid'],
                      reused_worker_pid=second['worker_pid'], model_compile_count=reset['info']['model_compile_count'],
                      duplicate_create_same_session=True, duplicate_step_executed_once=True,
                      external_force_changed_state=True, runtime_restore_matches_snapshot=True,
                      cross_session_state_isolated=True, pool_reused_model=True)
        imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                   if name.startswith('scene_factory') and getattr(module, '__file__', None)}
        assert all(Path(path).is_relative_to(SOURCE) for path in imports.values())
        report['scene_factory_imports'] = imports
    finally:
        if sid is not None:
            client.close(sid)
        server.shutdown()
        server.server_close()
        manager.close()
        thread.join(3)
    (WORK / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status':report['status'], 'result':str(WORK/'result.json')}))


if __name__ == '__main__':
    run()
'''


def validate(archive: Path, output: Path, *, python: Path, expected_sha256: str | None = None,
             output_created: bool = False, rehearsal: bool = False) -> dict:
    archive = archive.resolve(strict=True)
    output = output.absolute()
    _reject_links(output)
    if not output_created:
        output.mkdir(parents=True, exist_ok=False)
    digest = file_sha256(archive)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("archive SHA-256 does not match expected digest")
    verification = extract_checked(archive, output / "extracted")
    source = output / "extracted" / PREFIX.rstrip("/")
    if not (source / "scene_factory/runtime_service.py").is_file():
        raise ValueError("archive lacks runtime service source")
    runner = output / "run_extracted_http_smoke.py"
    runner.write_text(SMOKE_SOURCE, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["SCENE_FACTORY_HOME"] = str(source)
    completed = subprocess.run([str(python.resolve()), "-I", str(runner), str(source), str(output / "runtime")],
                               cwd=source, env=environment, capture_output=True, text=True, timeout=120, check=False)
    (output / "smoke_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "smoke_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result_path = output / "runtime/result.json"
    smoke = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    report = {"schema_version": "scene_factory.archive_delivery_validation.v1",
              "status": "passed" if completed.returncode == 0 and smoke and smoke.get("status") == "passed" else "failed",
              "archive": str(archive), "archive_sha256": digest, "archive_verification": verification,
              "rehearsal_only": rehearsal, "is_final_delivery_acceptance": not rehearsal,
              "subprocess_exit_code": completed.returncode, "extracted_http_smoke": smoke,
              "linux_gpu_acceptance": "not_evaluated", "dependencies": "provided by explicit Python interpreter"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def create_rehearsal_archive(root: Path, archive: Path) -> None:
    """Small source fixture, not a release: current Python modules and one primitive."""
    files = {path.relative_to(root).as_posix(): path.read_bytes()
             for path in sorted((root / "scene_factory").rglob("*.py"))
             if "__pycache__" not in path.parts}
    records = [json.loads(line) for line in (root / "data/assets/registry.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    primitive = next(record for record in records if record["asset_id"] == "mug_blue")
    assert primitive.get("source_type") == "primitive" and not primitive.get("usd_path")
    files["data/assets/registry.jsonl"] = (json.dumps(primitive) + "\n").encode("utf-8")
    entries = [{"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "category": "source"}
               for name, content in files.items()]
    manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "rehearsal_current_python_source_and_single_primitive_fixture_not_final_delivery",
                "file_count": len(entries), "total_size": sum(entry["size"] for entry in entries), "files": entries,
                "asset_dependency_closure": {"status": "complete_local_asset_dependencies", "files": ["data/assets/registry.jsonl"]}}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name, content in files.items():
            output.writestr(PREFIX + name, content)
        output.writestr(MANIFEST, json.dumps(manifest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New controlled directory; no existing content is removed")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Interpreter with project dependencies and MuJoCo")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--self-test", action="store_true", help="Rehearse on current Python sources and one primitive, without producing a release archive")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.archive or args.expected_sha256:
                parser.error("--self-test cannot be combined with archive or expected digest")
            _reject_links(args.output.absolute())
            args.output.mkdir(parents=True, exist_ok=False)
            fixture = args.output.absolute() / "rehearsal_fixture.zip"
            create_rehearsal_archive(ROOT, fixture)
            result = validate(fixture, args.output, python=args.python, output_created=True, rehearsal=True)
        else:
            if args.archive is None:
                parser.error("archive is required unless --self-test is selected")
            result = validate(args.archive, args.output, python=args.python, expected_sha256=args.expected_sha256)
        print(json.dumps({key: result[key] for key in ("status", "archive_sha256", "rehearsal_only", "subprocess_exit_code")}, indent=2))
        return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        print(f"archive validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
