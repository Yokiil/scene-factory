from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scene_factory.robot_physics_overrides import normalize_physics_overrides, verify_physics_overrides


def configuration():
    return {
        "schema_version": "scene_factory.robot_physics_overrides.v1",
        "source": {"kind": "geometry_estimate", "method": "PhysX_uniform_density_collision_geometry",
                   "real_world_measured": False, "description": "Preserved mass and uniform collision volume estimate"},
        "robot": {"recompute_inertia": True, "preserve_authored_mass": True,
                  "solver_position_iterations": 64, "solver_velocity_iterations": 16},
        "contact": {"contact_offset_m": .0001, "rest_offset_m": 0},
        "task_parts": {"peg_1": {"solver_position_iterations": 64, "solver_velocity_iterations": 16}, "socket_1": {}},
        "scene_solver": {"min_position_iterations": 64, "max_position_iterations": 255,
                         "min_velocity_iterations": 16, "max_velocity_iterations": 255},
    }


def test_versioned_override_is_explicit_and_does_not_mutate_inputs():
    raw = configuration()
    before = copy.deepcopy(raw)
    result = normalize_physics_overrides(raw)
    assert raw == before and result["source"]["real_world_measured"] is False
    assert result["task_parts"]["socket_1"] == {"solver_position_iterations": None, "solver_velocity_iterations": None}
    assert normalize_physics_overrides(None) is None
    assert verify_physics_overrides(object(), {}) == {}


@pytest.mark.parametrize("field,value", [("schema_version", "old"), ("unknown", 1), ("source", None), ("robot", {}),
                                         ("contact", {"contact_offset_m": .0001, "rest_offset_m": .0001})])
def test_invalid_override_contract_is_rejected(field, value):
    raw = configuration()
    raw[field] = value
    with pytest.raises(ValueError):
        normalize_physics_overrides(raw)


@pytest.mark.parametrize("bad", [True, "0.0001", None, float("nan"), float("inf"), 10**1000])
def test_contact_offset_must_be_finite(bad):
    raw = configuration()
    raw["contact"]["contact_offset_m"] = bad
    with pytest.raises(ValueError, match="finite"):
        normalize_physics_overrides(raw)


def test_source_must_not_claim_real_world_measurement_and_solver_limits_must_match():
    raw = configuration()
    raw["source"]["real_world_measured"] = True
    with pytest.raises(ValueError, match="unmeasured"):
        normalize_physics_overrides(raw)
    raw = configuration()
    raw["scene_solver"]["max_position_iterations"] = 32
    with pytest.raises(ValueError, match="minimum"):
        normalize_physics_overrides(raw)
    raw = configuration()
    raw["robot"]["solver_velocity_iterations"] = 16.0
    with pytest.raises(ValueError, match="integer"):
        normalize_physics_overrides(raw)


def test_live_tensor_validation_rejects_old_scaled_inertia_and_preserves_evidence():
    import numpy as np
    from types import SimpleNamespace

    class View:
        body_names = ["panda_leftfinger"]
        diagonal = np.array([4.20409e-6, 3.90260e-6, 1.33479e-6])
        off_diagonal_residual = 0.
        mass = .0140552232
        center = [0., .012, .026]

        def get_body_masses(self):
            return np.array([[self.mass]])

        def get_body_inertias(self):
            matrix = np.diag(self.diagonal)
            matrix[0, 1] = self.off_diagonal_residual
            return matrix.reshape(1, 1, 9)

        def get_body_coms(self):
            return np.array([[self.center]]), np.array([[[1., 0., 0., 0.]]])

    view = View()
    report = {"source": configuration()["source"], "robot_links": [{"name": "panda_leftfinger",
              "preserved_mass_kg": view.mass, "geometry_bbox_size_m": [.021, .0264, .05377],
              "geometry_bbox_min_m": [-.0105, 0, .0001], "geometry_bbox_max_m": [.0105, .0264, .05387]}], "live_verified": False}
    robot = SimpleNamespace(_articulation_view=view)
    result = verify_physics_overrides(robot, report)
    assert result["live_verified"] and not report["live_verified"]
    assert result["robot_links"][0]["live"]["center_of_mass_m"] == [0., .012, .026]
    view.off_diagonal_residual = np.finfo(np.float32).eps * max(view.diagonal)
    rounded = verify_physics_overrides(robot, report)
    assert rounded["robot_links"][0]["live"]["inertia_kg_m2"][0][1] == view.off_diagonal_residual
    assert rounded["robot_links"][0]["live"]["inertia_kg_m2"][1][0] == 0
    view.off_diagonal_residual *= 16
    with pytest.raises(RuntimeError, match="asymmetric"):
        verify_physics_overrides(robot, report)
    view.off_diagonal_residual = 0.
    view.diagonal = view.diagonal / 10000
    with pytest.raises(RuntimeError, match="estimate invalid"):
        verify_physics_overrides(robot, report)
    view.diagonal = np.array([4.20409e-6, 3.90260e-6, 1.33479e-6])
    view.mass = report["robot_links"][0]["preserved_mass_kg"]
    view.center = [0., 0., 0.]
    with pytest.raises(RuntimeError, match="COM is outside"):
        verify_physics_overrides(robot, report)
    view.diagonal = np.array([4.20409e-6, 3.90260e-6, 1.33479e-6])
    view.mass *= 2
    with pytest.raises(RuntimeError, match="estimate invalid"):
        verify_physics_overrides(robot, report)


def test_schema_declares_the_same_version_and_estimate_provenance():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "schemas/robot_physics_overrides.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == normalize_physics_overrides(configuration())["schema_version"]
    assert schema["properties"]["source"]["properties"]["real_world_measured"]["const"] is False


def test_stage_local_estimate_applies_apis_without_writing_source_layers(tmp_path, monkeypatch):
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import Gf, UsdGeom, UsdPhysics
    import pxr
    from types import SimpleNamespace
    from scene_factory.robot_physics_overrides import apply_physics_overrides

    class API:
        def __init__(self, prim, prefix):
            self.prim, self.prefix = prim, prefix

        def __getattr__(self, name):
            operation = name[:3] if name.startswith("Get") else name[:6]
            attribute = name[len(operation):-4]
            key = self.prefix + ":" + attribute[0].lower() + attribute[1:]
            kind = Sdf.ValueTypeNames.Float if self.prefix == "physxCollision" else Sdf.ValueTypeNames.Int
            return lambda: self.prim.CreateAttribute(key, kind)

    def api_type(prefix):
        return SimpleNamespace(Apply=lambda prim: API(prim, prefix))

    # Real USD composition is exercised; this stand-in only supplies PhysX
    # attribute wrappers outside Kit. It does not simulate mass properties.
    monkeypatch.setattr(pxr, "PhysxSchema", SimpleNamespace(
        PhysxArticulationAPI=api_type("physxArticulation"), PhysxCollisionAPI=api_type("physxCollision"),
        PhysxRigidBodyAPI=api_type("physxRigidBody"), PhysxSceneAPI=api_type("physxScene")), raising=False)
    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    UsdGeom.Xform.Define(stage, "/World/Robot")
    body = UsdGeom.Xform.Define(stage, "/World/Robot/link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(.014)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-10))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Robot/link/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0), Gf.Vec3f(.02, 0, 0), Gf.Vec3f(0, .03, 0), Gf.Vec3f(0, 0, .05)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    child_mass = UsdPhysics.MassAPI.Apply(mesh.GetPrim())
    child_mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    stage.GetRootLayer().Save()
    before = path.read_bytes()
    original_layer = stage.GetRootLayer().ExportToString()
    raw = configuration()
    raw["task_parts"] = {}
    report = apply_physics_overrides(stage, "/World/Robot", {}, raw)
    assert path.read_bytes() == before
    assert stage.GetRootLayer().ExportToString() == original_layer
    assert not report["source_layers_written"] and not report["live_verified"]
    assert list(mass.GetDiagonalInertiaAttr().Get()) == [0., 0., 0.]
    assert list(mass.GetCenterOfMassAttr().Get()) == [float("-inf")]*3
    assert list(child_mass.GetCenterOfMassAttr().Get()) == [float("-inf")]*3
    assert report["contacts"][0]["effective_authored_readback"]["contact_offset_m"] == pytest.approx(.0001)
    assert report["scene_solver_authored_readback"]["min_velocity_iterations"] == 16
