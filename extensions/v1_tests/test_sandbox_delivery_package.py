from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools.package_sandbox_delivery import MANIFEST, PREFIX, build_delivery, verify_delivery


def put(root: Path, name: str, text: str = "content") -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_delivery_includes_new_sources_and_explicit_evidence_but_not_secrets(tmp_path):
    root = tmp_path / "repo"
    put(root, "README.md")
    put(root, "scene_factory/new_module.py", "result = 1\n")
    put(root, "data/assets/source/generated/geometry.json", "{}")
    put(root, "config/llm.json", '{"api_key":"do-not-export"}')
    put(root, ".env", "PASSWORD=do-not-export")
    put(root, "tools/__pycache__/cached.pyc")
    put(root, "local_resources/private_model.bin")
    put(root, "outputs/delivery_20261028/report.json", '{"status":"passed"}')
    put(root, "outputs/completion_20261104/report.json", '{"status":"passed"}')
    put(root, "outputs/delivery_20261028/service.sqlite3")
    put(root, "outputs/other/private.json")
    output = tmp_path / "delivery.zip"
    report = build_delivery(root, output, evidence=["outputs/delivery_20261028", "outputs/completion_20261104"])
    assert report["verification"]["valid"]
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert PREFIX + "scene_factory/new_module.py" in names
        assert PREFIX + "outputs/delivery_20261028/report.json" in names
        assert PREFIX + "outputs/completion_20261104/report.json" in names
        assert not any("llm.json" in n or ".env" in n or "sqlite3" in n or "local_resources" in n for n in names)
        assert not any("private.json" in n or ".pyc" in n for n in names)
    assert output.with_suffix(".zip.sha256").is_file()


def test_tampered_archive_and_unlisted_member_fail_verification(tmp_path):
    root = tmp_path / "repo"
    put(root, "README.md", "original")
    original = tmp_path / "original.zip"
    build_delivery(root, original)
    for tamper in ("body", "extra"):
        output = tmp_path / (tamper + ".zip")
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(output, "w") as dest:
            for item in source.infolist():
                content = source.read(item.filename)
                if tamper == "body" and item.filename.endswith("README.md"):
                    content = b"changed"
                dest.writestr(item.filename, content)
            if tamper == "extra":
                dest.writestr(PREFIX + "tools/unlisted.py", "unexpected")
        assert not verify_delivery(output)["valid"]


def test_archive_path_traversal_and_duplicate_members_fail(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(PREFIX + "../escape.json", "{}")
    assert not verify_delivery(archive)["valid"]
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(PREFIX + "README.md", "one")
        output.writestr(PREFIX + "readme.md", "two")
    assert not verify_delivery(archive)["valid"]


def test_evidence_outside_delivery_root_and_existing_output_are_rejected(tmp_path):
    root = tmp_path / "repo"
    put(root, "README.md")
    evidence = put(root, "outputs/private.json", "{}")
    output = tmp_path / "delivery.zip"
    with pytest.raises(ValueError, match="evidence must"):
        build_delivery(root, output, evidence=[evidence])
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_delivery(root, output)
    assert output.read_bytes() == b"existing"


def test_manifest_cannot_hide_missing_source(tmp_path):
    root = tmp_path / "repo"
    put(root, "README.md")
    original = tmp_path / "original.zip"
    build_delivery(root, original)
    modified = tmp_path / "modified.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(modified, "w") as output:
        manifest = json.loads(source.read(MANIFEST))
        output.writestr(MANIFEST, json.dumps(manifest))
    assert not verify_delivery(modified)["valid"]


def test_private_isaac_package_cannot_enter_generic_source_delivery(tmp_path):
    root = tmp_path / "repo"
    put(root, "README.md")
    package = root / "outputs/delivery_20261028/private_environment"
    put(package, "runtime/lock.json", '{"backend":"isaac"}')
    put(package, "assets/robot/franka.usd", "private-cache")
    with pytest.raises(ValueError, match="private local Isaac"):
        build_delivery(root, tmp_path / "blocked.zip", evidence=["outputs/delivery_20261028"])
    # Wrapping the private package in a ZIP cannot bypass the exclusion.
    hidden = root / "outputs/delivery_20261028/hidden.zip"
    with zipfile.ZipFile(hidden, "w") as archive:
        archive.writestr("runtime/lock.json", '{"backend":"isaac"}')
        archive.writestr("assets/robot/franka.usd", "private-cache")
    with pytest.raises(ValueError, match="private local Isaac"):
        build_delivery(root, tmp_path / "blocked2.zip", evidence=[hidden])


def test_registry_missing_or_external_dependency_prevents_delivery(tmp_path):
    root = tmp_path / "repo"
    registry = put(root, "data/assets/registry.jsonl")
    for reference in ("missing.usd", "../../local_resources/private.usd", "https://example.com/asset.usd"):
        registry.write_text(json.dumps({"asset_id": "a", "usd_path": reference}), encoding="utf-8")
        with pytest.raises(ValueError):
            build_delivery(root, tmp_path / "blocked.zip")
        assert not (tmp_path / "blocked.zip").exists()


def test_source_manifest_dependency_and_public_robot_camera_configs_are_included(tmp_path):
    root = tmp_path / "repo"
    put(root, "data/assets/registry.jsonl", '{"asset_id":"a"}\n')
    put(root, "data/assets/source/a/SOURCE.json", '{"source_files":[{"path":"texture.webp"}]}')
    put(root, "data/assets/source/a/texture.webp")
    put(root, "config/robots/franka.json", "{}")
    put(root, "configs/cameras/wrist.yaml", "camera: wrist")
    put(root, "config/robots/credentials.json", '{"secret":"excluded"}')
    put(root, "config/llm.json", '{"secret":"excluded"}')
    output = tmp_path / "delivery.zip"
    build_delivery(root, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert PREFIX + "data/assets/source/a/texture.webp" in names
        assert PREFIX + "config/robots/franka.json" in names
        assert PREFIX + "configs/cameras/wrist.yaml" in names
        assert not any("credentials.json" in n or n.endswith("/llm.json") for n in names)


def test_gltf_external_texture_must_be_local_and_present(tmp_path):
    root = tmp_path / "repo"
    put(root, "data/assets/registry.jsonl", '{"asset_id":"a"}\n')
    model = put(root, "data/assets/source/a/model.gltf", '{"images":[{"uri":"texture.webp"}]}')
    with pytest.raises(ValueError, match="not a file"):
        build_delivery(root, tmp_path / "missing.zip")
    put(root, "data/assets/source/a/texture.webp")
    assert build_delivery(root, tmp_path / "present.zip")["verification"]["valid"]
    model.write_text('{"images":[{"uri":"https://example.com/private.png"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="relative local"):
        build_delivery(root, tmp_path / "external.zip")
