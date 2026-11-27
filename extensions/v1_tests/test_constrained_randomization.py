from __future__ import annotations

import copy

import pytest

from scene_factory.constrained_randomization import sample_candidate_seeds, sample_instance_physics
from scene_factory.models import AssetRecord
from scene_factory.registry import AssetRegistry


@pytest.fixture
def fixture():
    registry = AssetRegistry([AssetRecord(asset_id="primitive", category="part", bbox_m=(0.02, 0.02, 0.1), mass_kg=0.5, license="CC0")])
    scene = {"scene_id": "randomization-fixture", "room_dimensions_m": [2, 2, 3], "objects": [{
        "object_id": name, "asset_id": "primitive", "dynamic": True, "bbox_m": [0.02, 0.02, 0.1],
        "pose": {"position": [i * 0.2, 0, 0.05], "yaw_deg": 0}, "support": "floor",
    } for i, name in enumerate(("peg", "socket"))]}
    config = {"instances": {name: {"scale": [0.9, 1.1], "density_scale": [0.8, 1.2], "friction": [0.2, 0.8], "scale_group": "fit"} for name in ("peg", "socket")},
              "interfaces": [{"peg_id": "peg", "socket_id": "socket", "peg_radius_m": 0.01, "socket_radius_m": 0.0105, "clearance_m": [0.0004, 0.0006]}]}
    return scene, config, registry


def test_sample_reproducibility_density_scaling_and_source_nonmutation(fixture):
    scene, config, registry = fixture
    source, original_config = copy.deepcopy(scene), copy.deepcopy(config)
    first, record = sample_instance_physics(scene, config, 42, registry=registry)
    assert (first, record) == sample_instance_physics(scene, config, 42, registry=registry)
    assert scene == source and config == original_config
    assert registry.get("primitive").mass_kg == 0.5
    assert record["valid"] and record["controller_outcome"] == "not_run"
    peg = record["sampled_values"]["peg"]
    assert peg["scale"] == record["sampled_values"]["socket"]["scale"]
    assert peg["effective"]["mass_kg"] == pytest.approx(0.5 * peg["scale"]**3 * peg["density_scale"])
    assert peg["effective"]["inertia_kg_m2"][0] == pytest.approx(peg["base_inertia_kg_m2"][0] * peg["scale"]**5 * peg["density_scale"])
    assert first["objects"][0]["pose"]["position"][2] - first["objects"][0]["bbox_m"][2] / 2 == pytest.approx(0)
    other, _ = sample_instance_physics(scene, config, 43, registry=registry)
    assert other["objects"][0]["physics"] != first["objects"][0]["physics"]


def test_new_unrelated_stream_does_not_change_existing_draws(fixture):
    scene, config, registry = fixture
    _, original = sample_instance_physics(scene, config, 42, registry=registry)
    extra = copy.deepcopy(scene["objects"][0])
    extra["object_id"] = "new"
    scene["objects"].append(extra)
    config["instances"]["new"] = {"scale": [0.5, 2]}
    _, changed = sample_instance_physics(scene, config, 42, registry=registry)
    assert original["sampled_values"]["peg"] == changed["sampled_values"]["peg"]


def test_clearance_rejection_is_separate_from_controller_failure(fixture):
    scene, config, registry = fixture
    config["interfaces"][0]["socket_radius_m"] = 0.009
    _, record = sample_instance_physics(scene, config, 42, registry=registry)
    assert not record["valid"] and record["rejections"][0]["reason"] == "interface_clearance_out_of_range"
    report = sample_candidate_seeds(scene, config, list(range(100)), registry=registry)
    assert report["counts"]["parameter_invalid"] == 100
    assert report["counts"]["controller_failed"] is None
    assert report["counts"]["task_success"] is None
    assert len(report["rows"][0]["sampled_values"]) == 2


@pytest.mark.parametrize("bad", [True, -1, 2**64])
def test_invalid_seeds_fail(fixture, bad):
    scene, config, registry = fixture
    with pytest.raises(ValueError):
        sample_instance_physics(scene, config, bad, registry=registry)


def test_unsupported_changes_fail_instead_of_silent_ignoring(fixture):
    scene, config, registry = fixture
    config["instances"]["peg"]["damping"] = [0, 1]
    with pytest.raises(ValueError, match="invalid randomization fields"):
        sample_instance_physics(scene, config, 42, registry=registry)
    del config["instances"]["peg"]["damping"]
    config["instances"]["socket"]["scale"] = [1, 2]
    with pytest.raises(ValueError, match="identical"):
        sample_instance_physics(scene, config, 42, registry=registry)


def test_sampled_parameters_are_applied_by_mujoco_exporter(fixture):
    import mujoco
    from scene_factory.exporters.mujoco_mjcf import MujocoMjcfExporter

    scene, config, registry = fixture
    sampled, report = sample_instance_physics(scene, config, 42, registry=registry)
    model = mujoco.MjModel.from_xml_string(MujocoMjcfExporter(registry).to_string(sampled))
    peg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object__peg")
    assert model.body_mass[peg] == pytest.approx(report["sampled_values"]["peg"]["effective"]["mass_kg"])
    assert model.body_inertia[peg].tolist() == pytest.approx(report["sampled_values"]["peg"]["effective"]["inertia_kg_m2"])
