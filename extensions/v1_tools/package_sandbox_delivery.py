"""Create and verify a source/evidence delivery archive from explicit workspace roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "scene-factory-sandbox/"
MANIFEST = PREFIX + "DELIVERY_MANIFEST.json"
SCHEMA = "scene_factory.sandbox_delivery.v1"
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
SOURCE_ROOTS = (
    "scene_factory", "tests", "schemas", "recipes", "data/assets", "web", "tools", "docs", "examples", "deploy",
    "config/robots", "config/cameras", "configs/robots", "configs/cameras",
)
ROOT_FILES = {
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "RELEASE_CHECKLIST.md",
    "pyproject.toml", "uv.lock", "MANIFEST.in", "package.json", "package-lock.json",
    "config/llm.example.json", "configs/assets_batch.json",
    "config/robots.json", "config/cameras.json", "configs/robots.json", "configs/cameras.json",
}
EXTENSIONS = {
    ".py", ".pyi", ".json", ".jsonl", ".md", ".toml", ".lock", ".csv", ".yaml", ".yml",
    ".xml", ".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf", ".bin", ".png", ".jpg",
    ".jpeg", ".svg", ".txt", ".js", ".mjs", ".cjs", ".css", ".html", ".ps1", ".cmd",
    ".bat", ".sh", ".obj", ".mtl", ".stl", ".npy", ".npz", ".pdf", ".dockerignore",
    ".urdf", ".xacro", ".dae", ".tga", ".exr", ".dds", ".tif", ".tiff", ".hdr",
}
FORBIDDEN_COMPONENTS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", ".pytest_cache", ".ruff_cache",
    "local_resources", "local_assets", "isaac_local_assets", ".ssh", ".aws", ".azure", ".kube",
}
EVIDENCE_EXTENSIONS = {".json", ".jsonl", ".png", ".jpg", ".svg", ".csv", ".md", ".npz", ".npy", ".zip", ".txt", ".xml", ".usda", ".usd"}


def _safe_relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def _forbidden(relative: PurePosixPath) -> bool:
    parts = [part.lower() for part in relative.parts]
    name = parts[-1]
    return (
        any(part in FORBIDDEN_COMPONENTS for part in parts)
        or name.startswith(".env")
        or name in {"llm.json", "credentials", "credentials.json", "secrets.json", "gcm-diagnose.log"}
        or name.endswith((".pem", ".key", ".pfx", ".p12", ".log", ".tmp", ".pyc", ".sqlite3", ".db"))
    )


def _source_allowed(relative: str) -> bool:
    path = _safe_relative(relative)
    if _forbidden(path):
        return False
    if relative in ROOT_FILES:
        return True
    if not any(relative.startswith(base + "/") for base in SOURCE_ROOTS):
        return False
    return path.suffix.lower() in EXTENSIONS or path.name.startswith("Dockerfile") or path.name in {"LICENSE", "NOTICE"}


def _checked_file(root: Path, path: Path) -> Path:
    path = path.absolute()
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"delivery input escapes workspace: {path}")
    current = path
    while current != root and current.is_relative_to(root):
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError(f"delivery input must not be a link: {current}")
        current = current.parent
    if not path.is_file():
        raise ValueError(f"delivery input is not a file: {path}")
    return path


def collect_sources(root: Path) -> list[Path]:
    root = root.resolve()
    names: set[str] = set()
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("git source inventory failed")
        names.update(name for name in result.stdout.decode("utf-8").split("\x00") if name)
    # Explicit source roots also include the new deliverables before a commit.
    for base in SOURCE_ROOTS:
        directory = root / base
        if directory.exists():
            names.update(path.relative_to(root).as_posix() for path in directory.rglob("*") if path.is_file())
    names.update(name for name in ROOT_FILES if (root / name).is_file())
    sources = [_checked_file(root, root / name) for name in sorted(names) if _source_allowed(name)]
    for path in sources:
        _reject_private_isaac_package(path, root)
    return sources


def asset_dependency_inventory(root: Path, *, usd_python: Path | None = None) -> dict:
    """Resolve packaged registry/source assets; reject unresolved or external files.

    Binary USD needs a real OpenUSD interpreter. It is never treated as opaque
    proof of a complete dependency tree. No remote resources are downloaded.
    """
    root = root.resolve()
    asset_root = root / "data/assets"
    registry = asset_root / "registry.jsonl"
    if not registry.exists():
        return {"status": "registry_absent", "files": [], "usd_resolver": None}
    required: set[Path] = set()

    def require(value: str, base: Path) -> Path:
        if not isinstance(value, str) or not value or ":" in value or "\\" in value or "\x00" in value:
            raise ValueError("asset dependency must use a relative local path")
        if Path(value).is_absolute():
            raise ValueError("asset dependency must use a relative local path")
        path = Path(os.path.abspath(base / unquote(value)))
        if not path.is_relative_to(asset_root) or _forbidden(PurePosixPath(path.relative_to(root).as_posix())):
            raise ValueError("asset dependency escapes distributable asset directory")
        path = _checked_file(root, path)
        required.add(path)
        return path

    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("asset registry record must be an object")
        for key in ("usd_path", "source_path", "collision_path", "qa_report", "qa_report_path"):
            if record.get(key):
                require(record[key], asset_root)
    for manifest in sorted((asset_root / "source").glob("*/SOURCE.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("source_geometry", "source_collision_geometry"):
            if payload.get(key):
                require(payload[key], manifest.parent)
        for entry in payload.get("source_files", []):
            if entry.get("path"):
                require(entry["path"], manifest.parent)
    # A source pack includes all asset files, including non-registry visual GLBs.
    for path in sorted(asset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".gltf", ".glb"}:
            continue
        _checked_file(root, path)
        if path.suffix.lower() == ".gltf":
            model = json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open("rb") as handle:
                header = handle.read(20)
                if len(header) != 20:
                    raise ValueError(f"truncated GLB: {path.name}")
                magic, version, total, length, chunk = struct.unpack("<4sIIII", header)
                if magic != b"glTF" or version != 2 or chunk != 0x4E4F534A or length > MAX_FILE_BYTES or total != path.stat().st_size:
                    raise ValueError(f"unsupported GLB: {path.name}")
                model = json.loads(handle.read(length))
        for entry in model.get("buffers", []) + model.get("images", []):
            uri = entry.get("uri")
            if uri and not uri.startswith("data:"):
                require(uri, path.parent)
    usd_paths = sorted(path for path in asset_root.rglob("*")
                       if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"})
    if usd_paths:
        script = (
            "import json,sys; from pxr import Sdf,UsdUtils; paths=json.loads(sys.stdin.read()); "
            "result=[]\n"
            "for path in paths:\n"
            " layers,assets,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(path))\n"
            " result.append({'source':path,'files':[layer.realPath for layer in layers]+list(assets),'unresolved':list(unresolved)})\n"
            "print('SCENE_FACTORY_DEPENDENCIES='+json.dumps(result))\n"
        )
        completed = subprocess.run([str(usd_python or sys.executable), "-c", script],
                                   input=json.dumps([str(_checked_file(root, p)) for p in usd_paths]),
                                   text=True, capture_output=True, check=False, timeout=120)
        marker = "SCENE_FACTORY_DEPENDENCIES="
        if completed.returncode or marker not in completed.stdout:
            raise ValueError("USD closure requires working OpenUSD; pass --usd-python with an interpreter containing pxr")
        for entry in json.loads(completed.stdout.split(marker, 1)[1]):
            if entry["unresolved"]:
                raise ValueError("unresolved USD dependencies: " + repr(entry["unresolved"]))
            if not entry["files"]:
                raise ValueError("USD resolver returned no root layer")
            for value in entry["files"]:
                path = Path(value)
                if not path.is_absolute() or not path.is_relative_to(asset_root):
                    raise ValueError("USD dependency escapes distributable asset directory")
                require(path.relative_to(asset_root).as_posix(), asset_root)
    return {"status": "complete_local_asset_dependencies", "usd_resolver": "OpenUSD.ComputeAllDependencies" if usd_paths else None,
            "usd_root_count": len(usd_paths), "files": sorted(path.relative_to(root).as_posix() for path in required)}


def _evidence_files(root: Path, selections: Iterable[str | Path]) -> list[Path]:
    allowed_roots = [root / "outputs" / name for name in ("delivery_20261028", "completion_20261104")]
    found: set[Path] = set()
    for value in selections:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if not any(path.resolve().is_relative_to(base.resolve()) for base in allowed_roots):
            raise ValueError("evidence must be selected from an explicitly allowed delivery/completion root")
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            _reject_private_isaac_package(candidate, root)
            relative = PurePosixPath(candidate.relative_to(root).as_posix())
            if not _forbidden(relative) and candidate.suffix.lower() in EVIDENCE_EXTENSIONS:
                found.add(_checked_file(root, candidate))
        if not path.exists():
            raise ValueError(f"evidence selection does not exist: {value}")
    return sorted(found)


def _reject_private_isaac_package(candidate: Path, root: Path) -> None:
    """Local official robot closures are not part of the generic source ZIP."""
    for parent in (candidate.parent, *candidate.parents):
        if not parent.is_relative_to(root):
            break
        lock = parent / "runtime" / "lock.json"
        if lock.is_file():
            payload = json.loads(lock.read_text(encoding="utf-8"))
            if payload.get("backend") == "isaac":
                raise ValueError("private local Isaac asset packages must be excluded from generic source delivery")
        if parent == root:
            break
    if candidate.suffix.lower() == ".zip":
        with zipfile.ZipFile(candidate) as archive:
            for item in archive.infolist():
                if item.filename.endswith("runtime/lock.json"):
                    if item.file_size > 8 * 1024 * 1024:
                        raise ValueError("embedded runtime lock exceeds inspection limit")
                    if json.loads(archive.read(item)).get("backend") == "isaac":
                        raise ValueError("private local Isaac package ZIP must be excluded from generic source delivery")


def build_delivery(root: Path, output: Path, *, evidence: Iterable[str | Path] = (), usd_python: Path | None = None) -> dict:
    root, output = root.resolve(), output.absolute()
    if output.exists():
        raise FileExistsError(f"delivery archive already exists: {output}")
    sources = collect_sources(root)
    dependencies = asset_dependency_inventory(root, usd_python=usd_python)
    # Include valid asset extensions that did not appear in the ordinary allowlist.
    sources = sorted(set(sources) | {root / name for name in dependencies["files"]})
    evidence_files = _evidence_files(root, evidence)
    files = {path: "source" for path in sources}
    files.update({path: "evidence" for path in evidence_files})
    if not files:
        raise ValueError("delivery inventory is empty")
    if output.resolve() in {path.resolve() for path in files}:
        raise ValueError("delivery output must not overwrite an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=".sandbox-delivery-", suffix=".zip", dir=output.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    entries = []
    total = 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            used: set[str] = set()
            for path, category in sorted(files.items()):
                relative = path.relative_to(root).as_posix()
                _safe_relative(relative)
                name = PREFIX + relative
                if name.casefold() in used:
                    raise ValueError(f"case-colliding archive member: {relative}")
                used.add(name.casefold())
                before = path.stat()
                if before.st_size > MAX_FILE_BYTES:
                    raise ValueError(f"delivery file exceeds size limit: {relative}")
                content = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"input changed while packaging: {relative}")
                total += len(content)
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("delivery exceeds total size limit")
                archive.writestr(name, content)
                entries.append({"path": relative, "size": len(content),
                                "sha256": hashlib.sha256(content).hexdigest(), "category": category})
            manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "scope": "source_and_selected_evidence_not_a_verified_linux_gpu_release",
                        "file_count": len(entries), "total_size": total, "files": entries,
                        "asset_dependency_closure": dependencies,
                        "exclusions": ["credentials", "local_resources", "virtual_environments", "node_modules", "runtime_databases", "unselected_outputs"]}
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        verification = verify_delivery(temporary)
        if not verification["valid"]:
            raise ValueError("delivery verification failed: " + "; ".join(verification["errors"]))
        os.replace(temporary, output)
        checksum = hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix(output.suffix + ".sha256").write_text(checksum + "  " + output.name + "\n", encoding="utf-8")
        return {"archive": str(output), "sha256": checksum, "file_count": len(entries),
                "total_uncompressed_bytes": total, "verification": verification}
    finally:
        temporary.unlink(missing_ok=True)


def verify_delivery(path: Path) -> dict:
    errors = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len({name.casefold() for name in names}) != len(names):
                raise ValueError("duplicate or case-colliding archive member")
            total = 0
            for item in infos:
                _safe_relative(item.filename)
                if not item.filename.startswith(PREFIX) or item.is_dir():
                    raise ValueError("archive member outside delivery prefix or directory entry")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("archive contains a symbolic link")
                if item.file_size > MAX_FILE_BYTES:
                    raise ValueError("archive member exceeds size limit")
                total += item.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("archive exceeds size limit")
            manifest = json.loads(archive.read(MANIFEST))
            if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
                raise ValueError("unsupported delivery manifest")
            expected = {MANIFEST}
            expected_size = 0
            for entry in manifest["files"]:
                if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256", "category"}:
                    raise ValueError("invalid manifest entry")
                if entry["category"] not in {"source", "evidence"} or type(entry["size"]) is not int or entry["size"] < 0:
                    raise ValueError("invalid manifest category or size")
                relative = _safe_relative(entry["path"])
                if _forbidden(relative):
                    raise ValueError("manifest includes a forbidden file")
                name = PREFIX + entry["path"]
                if name in expected:
                    raise ValueError("duplicate manifest entry")
                expected.add(name)
                content = archive.read(name)
                if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"hash or size mismatch: {entry['path']}")
                expected_size += len(content)
            if expected != set(names):
                raise ValueError("unlisted or missing archive content")
            if manifest.get("file_count") != len(manifest["files"]) or manifest.get("total_size") != expected_size:
                raise ValueError("manifest totals mismatch")
            dependencies = manifest.get("asset_dependency_closure", {})
            if dependencies.get("status") not in {"registry_absent", "complete_local_asset_dependencies"}:
                raise ValueError("invalid dependency closure status")
            for relative in dependencies.get("files", []):
                _safe_relative(relative)
                if PREFIX + relative not in expected:
                    raise ValueError("dependency closure contains an unpackaged file")
        return {"valid": True, "errors": [], "file_count": len(manifest["files"])}
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        errors.append(str(exc))
        return {"valid": False, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--evidence", action="append", default=[])
    create.add_argument("--usd-python", type=Path, help="Python with pxr for binary USD dependency closure")
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        result = build_delivery(ROOT, args.output, evidence=args.evidence, usd_python=args.usd_python) if args.command == "create" else verify_delivery(args.archive)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result.get("valid", True) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"sandbox delivery: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
