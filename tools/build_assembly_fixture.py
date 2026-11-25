"""Author original peg/socket USD geometry without starting a simulator.

Run in an environment providing pxr. The output is a fixture, not proof of a
successful robot insertion. No external asset references are embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_fixture(output_dir: str | Path) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)
    usd = output / "scene.usda"
    if usd.exists():
        raise FileExistsError(f"fixture output exists: {usd}")
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Objects")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(800)
    material = UsdShade.Material.Define(stage, "/World/Materials/MetalContact")
    contact_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact_material.CreateStaticFrictionAttr(0.35)
    contact_material.CreateDynamicFrictionAttr(0.25)
    contact_material.CreateRestitutionAttr(0)
    objects = []

    def collision(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.0001)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)

    def box(path, center, size, color=(0.45, 0.48, 0.52)):
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1)
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        transform = UsdGeom.Xformable(geom)
        transform.AddTranslateOp().Set(Gf.Vec3d(*center))
        transform.AddScaleOp().Set(Gf.Vec3f(*size))
        collision(geom.GetPrim())
        return geom.GetPrim()

    def register(prim, object_id, asset_id, position, bbox, dynamic, category):
        prim.CreateAttribute("sceneFactory:objectId", Sdf.ValueTypeNames.String).Set(object_id)
        prim.CreateAttribute("sceneFactory:category", Sdf.ValueTypeNames.String).Set(category)
        objects.append({"object_id": object_id, "asset_id": asset_id, "category": category,
                        "pose": {"position": list(position), "yaw_deg": 0}, "bbox_m": list(bbox),
                        "dynamic": dynamic, "support": "table_1:top" if object_id in {"peg_1", "socket_1"} else "floor"})

    box("/World/Room/Floor", (0, 0, -0.025), (4, 4, 0.05))
    mount = box("/World/Objects/robot_mount_1", (-0.05, 0, 0.39), (0.5, 0.6, 0.78))
    register(mount, "robot_mount_1", "assembly_robot_mount", (-0.05, 0, 0.39), (0.5, 0.6, 0.78), False, "support")
    table = box("/World/Objects/table_1", (0.70, 0, 0.46), (0.7, 0.8, 0.92), (0.48, 0.35, 0.23))
    register(table, "table_1", "assembly_table", (0.70, 0, 0.46), (0.7, 0.8, 0.92), False, "support")
    socket_center = (0.60, 0.13, 0.96)
    socket = UsdGeom.Xform.Define(stage, "/World/Objects/socket_1")
    UsdGeom.Xformable(socket).AddTranslateOp().Set(Gf.Vec3d(*socket_center))
    inner, outer = 0.01075, 0.035
    wall = outer - inner
    box("/World/Objects/socket_1/Base", (0, 0, -0.03), (2*outer, 2*outer, 0.02))
    for side in (-1, 1):
        box(f"/World/Objects/socket_1/X{str(side).replace('-', 'minus')}", (side * (inner + wall/2), 0, 0.01), (wall, 2*outer, 0.06))
        box(f"/World/Objects/socket_1/Y{str(side).replace('-', 'minus')}", (0, side * (inner + wall/2), 0.01), (2*inner, wall, 0.06))
    register(socket.GetPrim(), "socket_1", "assembly_square_socket", socket_center, (0.07, 0.07, 0.08), False, "socket")
    peg_position = (0.55, -0.13, 0.97)
    peg = UsdGeom.Cylinder.Define(stage, "/World/Objects/peg_1")
    peg.CreateRadiusAttr(0.01)
    peg.CreateHeightAttr(0.1)
    peg.CreateAxisAttr(UsdGeom.Tokens.z)
    peg.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.55, 0.82)])
    UsdGeom.Xformable(peg).AddTranslateOp().Set(Gf.Vec3d(*peg_position))
    collision(peg.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(peg.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(peg.GetPrim())
    mass.CreateMassAttr(0.08)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    radius, length, mass_kg = 0.01, 0.1, 0.08
    transverse = mass_kg * (3 * radius**2 + length**2) / 12
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(transverse, transverse, mass_kg * radius**2 / 2))
    register(peg.GetPrim(), "peg_1", "assembly_cylindrical_peg", peg_position, (0.02, 0.02, 0.1), True, "peg")
    stage.GetRootLayer().Save()
    spec = {"schema_version": "scene_factory.assembly_spec.v1", "spec_id": "original_peg_socket_v1",
            "peg_id": "peg_1", "socket_id": "socket_1", "peg_interface_position_m": [0, 0, -0.05],
            "socket_interface_position_m": [0, 0, 0.04], "insertion_axis_socket": [0, 0, -1],
            "target_depth_m": 0.06, "clearance_m": [0.0006, 0.0009],
            "position_tolerance_m": 0.0005, "angular_tolerance_rad": 0.02, "max_force_N": 30,
            "min_hold_s": 0.5, "release_stable_s": 0.5, "timeout_s": 60, "jam_window_s": 2,
            "min_progress_m": 0.0005, "max_linear_speed_mps": 0.3, "max_angular_speed_rps": 2,
            "drop_below_z_m": 0.89, "allowed_contact_pairs": [["peg_1", "socket_1"], ["peg_1", "robot"], ["peg_1", "table_1"]]}
    scene = {"scene_id": "original_peg_socket_v1", "seed": 0, "recipe_name": "original_peg_socket_fixture",
             "room_type": "kitchen", "room_dimensions_m": [4, 4, 3], "event": "peg_socket_assembly",
             "description": "Original rigid peg and square socket; unverified robot insertion fixture.",
             "objects": objects, "task": {"instruction": "Insert the peg into the socket using robot contacts.",
             "target_object": "peg_1", "robot": "franka", "robot_base_position_m": [-0.05, 0, 0.78],
             "robot_base_orientation_wxyz": [1, 0, 0, 0], "grasp_orientation_wxyz": [0, 1, 0, 0],
             "grasp_offset_m": [0, 0, 0], "success": {"predicate": "assembly", "spec": "assembly_spec.json"}}}
    outputs = {"layout.json": scene, "assembly_spec.json": spec,
               "SOURCE.json": {"source": "SceneFactory original procedural CAD geometry", "license": "MIT",
                 "geometry": {"peg_radius_m": radius, "peg_length_m": length, "socket_half_width_m": inner,
                   "clearance_m": inner-radius, "contact_offset_m": 0.0001, "rest_offset_m": 0},
                 "validation": "USD authoring only; no physics or insertion acceptance",
                 "usd_sha256": hashlib.sha256(usd.read_bytes()).hexdigest()}}
    for name, data in outputs.items():
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return {"output_dir": str(output), "usd": str(usd), "layout": str(output / "layout.json"),
            "assembly_spec": str(output / "assembly_spec.json"), "acceptance": "not_verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(build_fixture(parser.parse_args().output), indent=2))
