"""Extend the original assembly fixture into seeded 20/50/100-object rooms.

Only OpenUSD authoring is used: this does not start SimulationApp or establish
physics/robot-task acceptance. The original four-object workspace and frozen
assembly spec are checked byte-for-byte/canonically before and after extension.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_assembly_fixture import build_fixture


WORKSPACE_IDS = ("robot_mount_1", "table_1", "socket_1", "peg_1")
PROTECTED = {"min_m": [-1.5, -1.5, 0.], "max_m": [2., 1.5, 2.3]}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intersects(left, right):
    return all(left["min_m"][axis] <= right["max_m"][axis]
               and right["min_m"][axis] <= left["max_m"][axis] for axis in range(3))


def plan_backgrounds(object_count: int, seed: int) -> list[dict]:
    if isinstance(object_count, bool) or object_count not in {20, 50, 100}:
        raise ValueError("room fixture object_count must be 20, 50 or 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("room fixture seed must be uint64")
    rng = random.Random(seed)
    candidates = [(round(x*.8, 6), round(y*.8, 6)) for y in range(-7, 8) for x in range(-7, 8)]
    rng.shuffle(candidates)
    rows = []
    for x, y in candidates:
        x, y = x + rng.uniform(-.025, .025), y + rng.uniform(-.025, .025)
        # Full static containment cells, not just their contained body, must
        # avoid the robot/approach/assembly/withdrawal swept-volume reservation.
        bounds = {"min_m": [x-.34, y-.34, -.01], "max_m": [x+.34, y+.34, .7]}
        if intersects(bounds, PROTECTED):
            continue
        size = [round(rng.uniform(.12, .3), 5) for _ in range(3)]
        dynamic = len(rows) % 3 == 0
        mass = 600 * math.prod(size)
        inertia = [mass*(size[1]**2+size[2]**2)/12,
                   mass*(size[0]**2+size[2]**2)/12,
                   mass*(size[0]**2+size[1]**2)/12]
        rows.append({"object_id": f"room_background_{len(rows):03d}",
                     "asset_id": "original_room_background_box", "category": "background",
                     "dynamic": dynamic, "support": "floor", "bbox_m": size,
                     "pose": {"position": [x, y, size[2]/2 + (.002 if dynamic else 0)], "yaw_deg": 0},
                     "color": [rng.uniform(.2, .7) for _ in range(3)],
                     "physical_parameters": {"mass_kg": mass if dynamic else None,
                                             "inertia_kg_m2": inertia if dynamic else None,
                                             "static_friction": 1., "dynamic_friction": .8,
                                             "restitution": 0., "initial_velocity_mps": [0, 0, 0]},
                     "containment_cell": {"authored": dynamic, "bounds_m": bounds,
                                          "inner_half_width_m": .3, "wall_thickness_m": .025,
                                          "wall_height_m": .65, "lid": True},
                     "workspace_intersection": False})
        if len(rows) == object_count - 4:
            break
    if len(rows) != object_count - 4:
        raise ValueError("room cannot accommodate requested backgrounds outside protected workspace")
    return rows


def build_room_fixture(output_dir: str | Path, object_count: int, *, seed: int = 19, peg_shape: str = "cylinder") -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    if peg_shape not in {"cylinder", "square"}:
        raise ValueError("peg_shape must be cylinder or square")
    backgrounds = plan_backgrounds(object_count, seed)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("room fixture output must be an empty/new directory")
    if peg_shape == "cylinder":
        build_fixture(output)
    else:
        from tools.build_square_assembly_fixture import build_square_fixture
        build_square_fixture(output)
    usd_path = output / "scene.usda"
    layout_path = output / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    source = json.loads((output / "SOURCE.json").read_text(encoding="utf-8"))
    original_usd_hash = _sha(usd_path)
    spec_bytes = (output / "assembly_spec.json").read_bytes()
    workspace_layout = _canonical(layout["objects"])
    workspace_task = _canonical(layout["task"])
    stage = Usd.Stage.Open(str(usd_path))

    def workspace_fingerprint():
        layer = Sdf.Layer.CreateAnonymous("workspace_snapshot.usda")
        for path in [f"/World/Objects/{key}" for key in WORKSPACE_IDS] + ["/World/Materials/MetalContact", "/World/PhysicsScene"]:
            Sdf.CreatePrimInLayer(layer, Sdf.Path(path).GetParentPath())
            if not Sdf.CopySpec(stage.GetRootLayer(), path, layer, path):
                raise RuntimeError(f"cannot snapshot workspace prim {path}")
        return hashlib.sha256(layer.ExportToString().encode()).hexdigest()

    workspace_before = workspace_fingerprint()
    floor = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Room/Floor"))
    next(op for op in floor.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(14, 14, .05))
    material = UsdShade.Material.Define(stage, "/World/Materials/RoomBackground")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.)
    physics_material.CreateDynamicFrictionAttr(.8)
    physics_material.CreateRestitutionAttr(0.)

    def box(path, center, size, color):
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.)
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        transform = UsdGeom.Xformable(geom)
        transform.AddTranslateOp().Set(Gf.Vec3d(*center))
        transform.AddScaleOp().Set(Gf.Vec3f(*size))
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(.001)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.)
        return prim

    # Perimeter collisions are room infrastructure, explicitly counted below.
    for index, (center, size) in enumerate([
        ((-7, 0, 1.5), (.1, 14, 3)), ((7, 0, 1.5), (.1, 14, 3)),
        ((0, -7, 1.5), (14, .1, 3)), ((0, 7, 1.5), (14, .1, 3)),
    ]):
        box(f"/World/Room/Walls/wall_{index}", center, size, (.65, .66, .68))
    for item in backgrounds:
        object_id = item["object_id"]
        center, size = item["pose"]["position"], item["bbox_m"]
        prim = box(f"/World/Objects/{object_id}", center, size, item["color"])
        prim.CreateAttribute("sceneFactory:objectId", Sdf.ValueTypeNames.String).Set(object_id)
        prim.CreateAttribute("sceneFactory:category", Sdf.ValueTypeNames.String).Set("background")
        if item["dynamic"]:
            rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid.CreateVelocityAttr(Gf.Vec3f(0, 0, 0))
            rigid.CreateAngularVelocityAttr(Gf.Vec3f(0, 0, 0))
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(item["physical_parameters"]["mass_kg"])
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*item["physical_parameters"]["inertia_kg_m2"]))
            x, y, _ = center
            half, thickness, height = .3, .025, .65
            walls = [((x-half-thickness/2, y, height/2), (thickness, 2*(half+thickness), height)),
                     ((x+half+thickness/2, y, height/2), (thickness, 2*(half+thickness), height)),
                     ((x, y-half-thickness/2, height/2), (2*half, thickness, height)),
                     ((x, y+half+thickness/2, height/2), (2*half, thickness, height)),
                     ((x, y, height+thickness/2), (2*(half+thickness), 2*(half+thickness), thickness))]
            for index, (wall_center, wall_size) in enumerate(walls):
                wall = box(f"/World/Room/BackgroundCells/{object_id}/wall_{index}", wall_center, wall_size, (.5, .52, .55))
                UsdGeom.Imageable(wall).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        layout["objects"].append({key: value for key, value in item.items() if key not in {"color", "physical_parameters", "containment_cell", "workspace_intersection"}})
    stage.GetRootLayer().Save()
    workspace_after = workspace_fingerprint()
    if workspace_before != workspace_after:
        raise RuntimeError("original assembly USD workspace changed during room extension")
    if _canonical(layout["objects"][:4]) != workspace_layout or _canonical(layout["task"]) != workspace_task or (output / "assembly_spec.json").read_bytes() != spec_bytes:
        raise RuntimeError("original assembly contract changed during room extension")
    layout["scene_id"] = (f"original_peg_socket_room{object_count}_seed{seed}" if peg_shape == "cylinder"
                          else f"original_square_peg_socket_room{object_count}_seed{seed}")
    layout["seed"] = seed
    layout["recipe_name"] = "original_peg_socket_room_extension"
    layout["room_dimensions_m"] = [14, 14, 3]
    layout["description"] = "Original peg/socket workspace inside a seeded room; robot task acceptance pending measured replay."
    layout["room_fixture"] = {"object_count": object_count, "workspace_object_count": 4,
                              "background_count": len(backgrounds), "protected_workspace_aabb_m": PROTECTED,
                              "background_dynamic_count": sum(item["dynamic"] for item in backgrounds),
                              "background_static_count": sum(not item["dynamic"] for item in backgrounds),
                              "containment_cells": "five fixed invisible colliders per dynamic background; physics remains enabled",
                              "runtime_background_stability": "not_evaluated", "local_assembly_task": "not_evaluated"}
    if peg_shape == "square":
        layout["fixture_variant"]["room_fixture_id"] = layout["scene_id"]
        layout["room_fixture"]["peg_shape"] = "square"
    layout_path.write_text(json.dumps(layout, indent=2, allow_nan=False), encoding="utf-8")
    counts = {"scene_objects": len(layout["objects"]),
              "dynamic_scene_objects": sum(item["dynamic"] for item in layout["objects"]),
              "static_scene_objects": sum(not item["dynamic"] for item in layout["objects"]),
              "usd_rigid_bodies": sum(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse()),
              "usd_collision_prims": sum(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in stage.Traverse()),
              "background_containment_colliders": 5*sum(item["dynamic"] for item in backgrounds),
              "room_perimeter_colliders": 4, "external_geometry_assets": 0}
    source.update({"source_name": "Original SceneFactory seeded assembly room fixture", "license": "MIT",
                   "geometry_origin": "procedural", "source_url": "tools/build_assembly_room_fixture.py",
                   "source_revision": f"sha256:{_sha(__file__)}", "license_source": "LICENSE",
                   "generator_dependencies_sha256": {"tools/build_assembly_fixture.py": _sha(Path(__file__).with_name("build_assembly_fixture.py"))},
                   "usd_sha256": _sha(usd_path), "layout_sha256": _sha(layout_path),
                   "assembly_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                   "original_workspace": {"base_usd_sha256": original_usd_hash, "canonical_prim_specs_sha256": workspace_before,
                                          "canonical_layout_objects_sha256": hashlib.sha256(workspace_layout).hexdigest(),
                                          "canonical_task_sha256": hashlib.sha256(workspace_task).hexdigest(),
                                          "unchanged_after_extension": True},
                   "seed": seed, "counts": counts, "protected_workspace_aabb_m": PROTECTED,
                   "backgrounds": backgrounds,
                   "validation": "Geometry authoring and workspace identity only; dynamics, background stability and assembly task not evaluated"})
    if peg_shape == "square":
        source["fixture_id"] = layout["scene_id"]
        source["generator_dependencies_sha256"]["tools/build_square_assembly_fixture.py"] = _sha(Path(__file__).with_name("build_square_assembly_fixture.py"))
        source["background_plan_sha256"] = hashlib.sha256(_canonical(backgrounds)).hexdigest()
    (output / "SOURCE.json").write_text(json.dumps(source, indent=2, allow_nan=False), encoding="utf-8")
    report = {"schema_version": "scene_factory.assembly_room_fixture.v1", "output_dir": str(output),
              "usd": str(usd_path), "layout": str(layout_path), "assembly_spec": str(output / "assembly_spec.json"),
              "seed": seed, "counts": counts, "workspace_unchanged": True,
              "spec_unchanged": True, "acceptance": "geometry_only_not_physics_acceptance",
              "files_sha256": {name: _sha(output / name) for name in ("scene.usda", "layout.json", "assembly_spec.json", "SOURCE.json")}}
    (output / "fixture_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objects", type=int, choices=(20, 50, 100), default=20)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--peg-shape", choices=("cylinder", "square"), default="cylinder")
    args = parser.parse_args()
    print(json.dumps(build_room_fixture(args.output, args.objects, seed=args.seed, peg_shape=args.peg_shape), indent=2))
