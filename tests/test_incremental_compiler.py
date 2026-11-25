from __future__ import annotations

import copy
import json

import pytest

from scene_factory.factory import SceneFactory
from scene_factory.incremental_compiler import (
    IncrementalMujocoCompiler, ObjectFragmentCompiler, motion_envelope_query,
)
from scene_factory.exporters.mujoco_mjcf import MujocoMjcfExporter
from scene_factory.models import AssetRecord
from scene_factory.registry import AssetRegistry


@pytest.fixture
def fixture():
    registry = AssetRegistry([AssetRecord(asset_id="cube", category="cube", bbox_m=(.1, .1, .1), mass_kg=1)])
    scene = {"scene_id": "incremental", "room_dimensions_m": [4, 4, 3], "task": {}, "objects": [
        {"object_id": "part", "asset_id": "cube", "dynamic": True, "bbox_m": [.1, .1, .1],
         "pose": {"position": [0, 0, .08], "yaw_deg": 0}, "support": "floor",
         "physics": {"mass_kg": 1, "inertia_kg_m2": [.0017, .0017, .0017], "friction": .5, "source": "test"}},
        {"object_id": "obstacle", "asset_id": "cube", "dynamic": False, "bbox_m": [.1, .1, .1],
         "pose": {"position": [2, 0, .05], "yaw_deg": 0}, "support": "floor"},
    ]}
    return registry, scene


def test_fragment_export_is_identical_and_rebuilds_only_changed_object(tmp_path):
    factory = SceneFactory()
    scene = json.loads(json.dumps(factory.build_from_recipe("living_room_returned_home", 19).scene.to_dict()))
    compiler = ObjectFragmentCompiler(factory.registry, cache_dir=tmp_path)
    xml, first = compiler.export(scene)
    assert xml == MujocoMjcfExporter(factory.registry).to_string(scene)
    assert first["fragment_misses"] == len(scene["objects"])
    changed = copy.deepcopy(scene)
    changed["objects"][0]["pose"]["yaw_deg"] += 2
    xml, report = compiler.export(changed)
    assert xml == MujocoMjcfExporter(factory.registry).to_string(changed)
    assert report["rebuilt_objects"] == [scene["objects"][0]["object_id"]]
    assert report["fragment_hits"] == len(scene["objects"]) - 1
    other = ObjectFragmentCompiler(factory.registry, cache_dir=tmp_path)
    assert other.export(changed)[1]["fragment_misses"] == 0
    entry = tmp_path / f"{compiler._key(scene['objects'][0], compiler.dependency_fingerprints(scene))}.json"
    entry.write_text("{}")
    # Test exactly the corrupted dependency rather than rely on directory order.
    bad = ObjectFragmentCompiler(factory.registry, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="corrupt"):
        bad.export(scene)


def test_native_pose_physics_patch_matches_full_compilation(fixture):
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    registry, scene = fixture
    incremental = IncrementalMujocoCompiler(registry)
    initial = incremental.compile(scene)
    model = initial.model
    changed = copy.deepcopy(scene)
    part = changed["objects"][0]
    part["physics"] = {"mass_kg": 2, "inertia_kg_m2": [.0034]*3, "friction": .25, "source": "test"}
    part["pose"] = {"position": [.05, -.1, .09], "yaw_deg": 40}
    result = incremental.compile(changed)
    assert result.model is model
    assert result.report["kind"] == "episode_boundary_parameter_patch"
    assert result.report["fragment_hits"] == 1
    full = IncrementalMujocoCompiler(registry, mode="full").compile(changed)
    for array in ("body_mass", "body_inertia", "qpos0", "geom_friction", "dof_M0"):
        np.testing.assert_allclose(getattr(result.model, array), getattr(full.model, array), rtol=0, atol=1e-13)
    for _ in range(100):
        mujoco.mj_step(result.model, result.data)
        mujoco.mj_step(full.model, full.data)
        np.testing.assert_allclose(result.data.qpos, full.data.qpos, rtol=0, atol=1e-13)
        np.testing.assert_allclose(result.data.qvel, full.data.qvel, rtol=0, atol=1e-13)


def test_explicit_three_dimensional_orientation_is_exported_and_patched(fixture):
    pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    registry, scene = fixture
    compiler = IncrementalMujocoCompiler(registry)
    initial = compiler.compile(scene)
    changed = copy.deepcopy(scene)
    changed["objects"][0]["pose"]["orientation_wxyz"] = [2**-.5, 2**-.5, 0., 0.]
    result = compiler.compile(changed)
    assert result.model is initial.model
    np.testing.assert_allclose(result.data.qpos[3:7], [2**-.5, 2**-.5, 0, 0], atol=1e-8)
    assert result.report["kind"] == "episode_boundary_parameter_patch"


def test_shape_static_pose_and_topology_reload(fixture):
    pytest.importorskip("mujoco")
    registry, scene = fixture
    compiler = IncrementalMujocoCompiler(registry)
    old = compiler.compile(scene).model
    scene["objects"][1]["pose"]["position"][0] -= .5
    new = compiler.compile(scene)
    assert new.model is not old and new.report["kind"] == "native_compile"
    old = new.model
    scene["objects"][0]["bbox_m"][0] += .01
    assert compiler.compile(scene).model is not old


def test_envelope_discovers_new_contact_and_propagates_dynamic_chain(fixture):
    _, scene = fixture
    changed = copy.deepcopy(scene)
    changed["objects"][0]["pose"]["position"][0] = 1.95
    report = motion_envelope_query(scene, changed, {"part"}, horizon_s=.1, max_linear_speed_mps=.1)
    assert ["obstacle", "part"] in report["newly_possible_pairs"]
    assert report["affected_objects"] == ["obstacle", "part"]
    assert report["evidence_policy"] == "full_dynamics_and_task_revalidation"
    with pytest.raises(ValueError):
        motion_envelope_query(scene, changed, {"part"}, horizon_s=float("nan"), max_linear_speed_mps=1)


def test_partition_refuses_background_reachable_by_dynamic(fixture):
    pytest.importorskip("mujoco")
    registry, scene = fixture
    scene["objects"][1]["pose"]["position"][0] = .11
    compiler = IncrementalMujocoCompiler(registry, mode="partition", background_ids={"obstacle"})
    with pytest.raises(ValueError, match="motion envelope"):
        compiler.compile(scene)


def test_partition_simplifies_only_exact_box_meshes_with_equal_local_rollout():
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    from scene_factory.paths import default_registry_path

    registry = AssetRegistry.load(default_registry_path())
    background = registry.get("kd_box_01")
    scene = {"scene_id": "partition", "room_dimensions_m": [10, 10, 3], "objects": [
        {"object_id": "background", "asset_id": background.asset_id, "dynamic": False,
         "bbox_m": list(background.bbox_m), "pose": {"position": [4, 0, .06], "yaw_deg": 0}},
        {"object_id": "part", "asset_id": "remote_black", "dynamic": True,
         "bbox_m": [.1, .1, .1], "pose": {"position": [0, 0, .1], "yaw_deg": 0}},
    ]}
    full = IncrementalMujocoCompiler(registry, mode="full").compile(scene)
    partition = IncrementalMujocoCompiler(registry, mode="partition", background_ids={"background"}).compile(scene)
    assert full.model.nmesh == 1 and partition.model.nmesh == 0
    assert partition.report["precision_partition"][0]["geometry_proof"] == "convex_hull_of_all_eight_box_corners"
    for _ in range(100):
        mujoco.mj_step(full.model, full.data)
        mujoco.mj_step(partition.model, partition.data)
        np.testing.assert_array_equal(full.data.qpos, partition.data.qpos)
        np.testing.assert_array_equal(full.data.qvel, partition.data.qvel)


def test_invalid_physics_fails_without_mutating_previous_model(fixture):
    pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    registry, scene = fixture
    compiler = IncrementalMujocoCompiler(registry)
    result = compiler.compile(scene)
    previous = result.model.body_mass.copy()
    changed = copy.deepcopy(scene)
    changed["objects"][0]["physics"]["mass_kg"] = -1
    with pytest.raises(ValueError, match="positive"):
        compiler.compile(changed)
    np.testing.assert_array_equal(previous, result.model.body_mass)


def test_whole_file_cache_reuses_only_exact_scene(fixture):
    pytest.importorskip("mujoco")
    registry, scene = fixture
    compiler = IncrementalMujocoCompiler(registry, mode="file_cache")
    assert not compiler.compile(scene).report["whole_file_cache_hit"]
    result = compiler.compile(scene)
    assert result.report["whole_file_cache_hit"]
    assert result.report["kind"] == "native_compile"
    changed = copy.deepcopy(scene)
    changed["objects"][0]["physics"]["mass_kg"] = 1.1
    assert not compiler.compile(changed).report["whole_file_cache_hit"]


def test_fragment_memory_cache_is_bounded(fixture):
    registry, scene = fixture
    compiler = ObjectFragmentCompiler(registry, max_cached_fragments=1)
    compiler.export(scene)
    assert len(compiler.cache) == 1


def test_cli_writes_portable_variants_and_does_not_claim_physics_acceptance(tmp_path):
    pytest.importorskip("mujoco")
    import contextlib
    import io
    from scene_factory.cli import main

    factory = SceneFactory()
    scene = json.loads(json.dumps(factory.build_from_recipe("living_room_returned_home", 19).scene.to_dict()))
    source = tmp_path / "before.json"
    source.write_text(json.dumps(scene))
    changed = copy.deepcopy(scene)
    next(item for item in changed["objects"] if item["dynamic"])["pose"]["position"][0] += .01
    after = tmp_path / "after.json"
    after.write_text(json.dumps(changed))
    output = tmp_path / "compiled"
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main(["sandbox", "incremental-compile", str(source), str(after), "--output", str(output)])
    assert code == 0
    report = json.loads(stream.getvalue())
    assert report["physics_acceptance"] == "not_evaluated"
    assert report["rows"][1]["kind"] == "episode_boundary_parameter_patch"
    assert (output / "0001/scene.xml").is_file()
