"""Build a separately identified square-peg variant; never starts a simulator.

The original cylinder builder and byte-identical assembly spec are preserved.
This changes shape, volume, density and inertia, not the acceptance thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_assembly_fixture import build_fixture


FIXTURE_ID = "original_square_peg_socket_v1"
SIDE_M, LENGTH_M, MASS_KG, SOCKET_HALF_WIDTH_M = .02, .1, .08, .01075


def square_physics() -> dict:
    transverse = MASS_KG*(SIDE_M**2+LENGTH_M**2)/12
    return {"shape": "square_prism", "dimensions_m": [SIDE_M, SIDE_M, LENGTH_M],
            "mass_kg": MASS_KG, "center_of_mass_m": [0., 0., 0.],
            "inertia_kg_m2": [transverse, transverse, MASS_KG*SIDE_M**2/6],
            "density_kg_m3": MASS_KG/(SIDE_M**2*LENGTH_M),
            "nominal_face_clearance_m": SOCKET_HALF_WIDTH_M-SIDE_M/2,
            "yaw_only_clearance_at_frozen_angular_tolerance_m": SOCKET_HALF_WIDTH_M-SIDE_M/2*(math.cos(.02)+math.sin(.02))}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def build_square_fixture(output_dir: str | Path) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    output = Path(output_dir).absolute()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("square fixture output must be an empty/new directory")
    build_fixture(output)
    usd_path, layout_path, spec_path = (output/name for name in ("scene.usda", "layout.json", "assembly_spec.json"))
    original_source = json.loads((output/"SOURCE.json").read_text(encoding="utf-8"))
    original_layout = json.loads(layout_path.read_text(encoding="utf-8"))
    spec_bytes = spec_path.read_bytes()
    original_usd_sha256 = _sha(usd_path)
    stage = Usd.Stage.Open(str(usd_path))

    def non_peg_fingerprint():
        layer = Sdf.Layer.CreateAnonymous("non_peg.usda")
        if not Sdf.CopySpec(stage.GetRootLayer(), "/World", layer, "/World"):
            raise RuntimeError("cannot snapshot original workspace")
        copy = Usd.Stage.Open(layer)
        if not copy.RemovePrim("/World/Objects/peg_1"):
            raise RuntimeError("cannot exclude peg from workspace fingerprint")
        return hashlib.sha256(layer.ExportToString().encode()).hexdigest()

    original_non_peg_hash = non_peg_fingerprint()
    peg_path = "/World/Objects/peg_1"
    if not stage.RemovePrim(peg_path):
        raise RuntimeError("original peg prim missing")
    cube = UsdGeom.Cube.Define(stage, peg_path)
    cube.CreateSizeAttr(1.)
    cube.CreateDisplayColorAttr([Gf.Vec3f(.2, .55, .82)])
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(.55, -.13, .97))
    transform.AddScaleOp().Set(Gf.Vec3f(SIDE_M, SIDE_M, LENGTH_M))
    prim = cube.GetPrim()
    prim.CreateAttribute("sceneFactory:objectId", Sdf.ValueTypeNames.String).Set("peg_1")
    prim.CreateAttribute("sceneFactory:category", Sdf.ValueTypeNames.String).Set("peg")
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Materials/MetalContact")), materialPurpose="physics")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.0001)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)
    physical = square_physics()
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0., 0., 0.))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*physical["inertia_kg_m2"]))
    new_non_peg_hash = non_peg_fingerprint()
    if new_non_peg_hash != original_non_peg_hash or spec_path.read_bytes() != spec_bytes:
        raise RuntimeError("variant modified the original workspace or frozen spec")
    stage.GetRootLayer().Save()
    if stage.GetRootLayer().subLayerPaths or any(item.HasAuthoredReferences() for item in stage.Traverse()):
        raise RuntimeError("square fixture unexpectedly references external assets")
    layout = json.loads(json.dumps(original_layout))
    layout.update(scene_id=FIXTURE_ID, recipe_name="original_square_peg_socket_fixture",
                  description="Separate square-prism peg variant; geometry authoring only until measured robot acceptance.")
    peg = next(item for item in layout["objects"] if item["object_id"] == "peg_1")
    peg["asset_id"] = "assembly_square_peg_20mm"
    layout["fixture_variant"] = {"fixture_id": FIXTURE_ID, "base_fixture_id": original_layout["scene_id"],
                                 "changed": ["peg_shape", "peg_volume", "peg_density", "peg_inertia"],
                                 "unchanged": ["peg_mass", "peg_pose", "socket", "support", "robot_base", "task_spec_bytes"],
                                 "acceptance": "not_verified"}
    _write(layout_path, layout)
    source = {"source": "SceneFactory original procedural CAD square-peg variant", "license": "MIT",
              "fixture_id": FIXTURE_ID, "base_fixture_id": original_layout["scene_id"],
              "generator": "tools/build_square_assembly_fixture.py", "generator_sha256": _sha(Path(__file__)),
              "base_generator": "tools/build_assembly_fixture.py", "base_generator_sha256": _sha(Path(__file__).with_name("build_assembly_fixture.py")),
              "base_usd_sha256": original_usd_sha256, "base_source_manifest": original_source,
              "original_non_peg_prim_sha256": original_non_peg_hash, "variant_non_peg_prim_sha256": new_non_peg_hash,
              "frozen_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
              "geometry": {"peg_shape": "square_prism", "peg_width_m": SIDE_M, "peg_length_m": LENGTH_M,
                           "socket_half_width_m": SOCKET_HALF_WIDTH_M, "clearance_m": physical["nominal_face_clearance_m"],
                           "contact_offset_m": .0001, "rest_offset_m": 0.,
                           "clearance_scope": "Nominal axis-aligned face clearance; corner/yaw contact differs from cylinder."},
              "physical_parameters": physical, "usd_sha256": _sha(usd_path), "layout_sha256": _sha(layout_path),
              "validation": "USD authoring and unchanged-input checks only; no physics or insertion acceptance"}
    _write(output/"SOURCE.json", source)
    check = {"schema_version": "scene_factory.square_fixture_build.v1", "fixture_id": FIXTURE_ID,
             "primitive_type": prim.GetTypeName(), "physics": physical,
             "spec_unchanged": spec_path.read_bytes() == spec_bytes, "non_peg_prims_unchanged": new_non_peg_hash == original_non_peg_hash,
             "usd_sha256": _sha(usd_path), "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
             "external_asset_references": False, "acceptance": "not_verified"}
    _write(output/"fixture_report.json", check)
    return {"output_dir": str(output), "usd": str(usd_path), "layout": str(layout_path),
            "assembly_spec": str(spec_path), **check}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(build_square_fixture(parser.parse_args().output), indent=2))
