from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest

from scene_factory.environment_package import archive_environment, build_environment, verify_environment
from scene_factory.factory import SceneFactory
from scene_factory.physics_dependencies import PhysicsDependencyGraph, build_dependency_graph, plan_scene_update


@pytest.fixture
def scene_path(tmp_path):
    scene = SceneFactory().build_from_recipe("living_room_returned_home", 19).scene.to_dict()
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path


def test_package_freezes_inputs_and_reproduces_mjcf_without_original_registry(scene_path, tmp_path):
    import mujoco
    from scene_factory.exporters.mujoco_mjcf import MujocoMjcfExporter
    from scene_factory.registry import AssetRegistry

    package = tmp_path / "environment"
    result = build_environment(scene_path, package)
    assert result["verification"]["valid"]
    scene = json.loads((package / "scene/layout.json").read_text(encoding="utf-8"))
    registry = AssetRegistry.load(package / "assets/registry.jsonl")
    xml = MujocoMjcfExporter(registry).to_string(scene)
    assert xml == (package / "scene/scene.xml").read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    scene_path.write_text("{}", encoding="utf-8")
    assert verify_environment(package)["valid"]
    archive = archive_environment(package, tmp_path / "environment.zip")
    assert archive.is_file()


def test_tampering_and_unsafe_lock_paths_are_rejected(scene_path, tmp_path):
    package = tmp_path / "environment"
    build_environment(scene_path, package)
    (package / "scene/layout.json").write_text("{}", encoding="utf-8")
    result = verify_environment(package)
    assert not result["valid"] and "dependency_mismatch:scene/layout.json" in result["errors"]
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["lock_path"] = "../outside.json"
    (package / "manifest.json").write_text(json.dumps(manifest))
    assert not verify_environment(package)["valid"]


def test_cache_reuses_compiled_artifact_but_not_runtime_evidence(scene_path, tmp_path):
    from scene_factory.exporters.mujoco_mjcf import MujocoMjcfExporter

    cache = tmp_path / "cache"
    first = build_environment(scene_path, tmp_path / "first", cache_dir=cache)
    with patch("scene_factory.environment_package.MujocoMjcfExporter.to_string", autospec=True, side_effect=MujocoMjcfExporter.to_string) as exporter:
        second = build_environment(scene_path, tmp_path / "second", cache_dir=cache)
    assert exporter.call_count == 1  # Revalidation only; generation used the cache.
    assert not first["cache"]["hit"] and second["cache"]["hit"]
    assert first["environment_sha256"] == second["environment_sha256"]
    assert second["evidence_status"] == "not_verified"
    record = next(cache.glob("*.json"))
    record.write_text('{"key":"tampered","xml":"bad","xml_sha256":"bad"}')
    with pytest.raises(ValueError, match="corrupt"):
        build_environment(scene_path, tmp_path / "third", cache_dir=cache)


def test_unsupported_backend_and_existing_output_fail_without_overwrite(scene_path, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        build_environment(scene_path, target)
    assert marker.read_text() == "keep"
    with pytest.raises(ValueError, match="Isaac"):
        build_environment(scene_path, tmp_path / "isaac", backend="isaac")


def test_selected_mesh_dependency_closure_is_copied_and_checked(tmp_path):
    import mujoco
    from scene_factory.registry import AssetRegistry
    from scene_factory.paths import default_registry_path

    registry = AssetRegistry.load(default_registry_path())
    asset = registry.get("kd_box_01")
    scene = {"scene_id": "mesh_dependency_fixture", "room_dimensions_m": [4, 4, 3], "task": {}, "objects": [{
        "object_id": "box", "asset_id": asset.asset_id, "dynamic": True,
        "bbox_m": list(asset.bbox_m), "pose": {"position": [0, 0, asset.bbox_m[2] / 2], "yaw_deg": 0},
    }]}
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    package = tmp_path / "package"
    build_environment(path, package)
    assert (package / "assets/source/kd_box_01/geometry.json").is_file()
    mujoco.MjModel.from_xml_string((package / "scene/scene.xml").read_text())
    (package / "assets/source/kd_box_01/geometry.json").write_text("{}")
    assert not verify_environment(package)["valid"]


def test_mass_update_invalidates_dynamics_but_preserves_visual_node(scene_path):
    before = json.loads(scene_path.read_text())
    after = copy.deepcopy(before)
    dynamic = next(item for item in after["objects"] if item["dynamic"])
    dynamic["physics"] = {"mass_kg": 1.2}
    report = plan_scene_update(before, after)
    assert report["full_dynamics_revalidation"]
    assert "artifact:visual" in report["reusable_nodes"]
    assert "artifact:physics_evidence" in report["affected_nodes"]
    graph = build_dependency_graph(before).to_dict()
    source = copy.deepcopy(graph)
    assert PhysicsDependencyGraph.from_dict(graph).to_dict() == source
    assert graph == source
    assert not plan_scene_update(before, before)["affected_nodes"]


def test_new_object_or_pose_does_not_reuse_old_contact_evidence(scene_path):
    before = json.loads(scene_path.read_text())
    after = copy.deepcopy(before)
    after["objects"][0]["pose"]["position"][0] += 1
    assert plan_scene_update(before, after)["full_dynamics_revalidation"]
    added = copy.deepcopy(before["objects"][0])
    added["object_id"] = "new_object"
    after["objects"].append(added)
    assert "scene" in plan_scene_update(before, after)["changed_nodes"]
