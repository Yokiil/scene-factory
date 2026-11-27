"""Explicit, opt-in Isaac estimates derived from locked collision geometry.

Imports of USD/PhysX are confined to the application function after Kit startup.
No source asset layer is edited or saved. These are estimates, not measurements.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any


SCHEMA_VERSION = "scene_factory.robot_physics_overrides.v1"
ESTIMATION_METHOD = "PhysX_uniform_density_collision_geometry"


def _object(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError(f"{name} must be an object with supported fields {sorted(allowed)}")
    return value


def _iterations(value: Any, name: str, minimum: int, *, optional: bool = True) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 255:
        raise ValueError(f"{name} must be an integer in [{minimum}, 255]")
    return value


def normalize_physics_overrides(raw: Any) -> dict[str, Any] | None:
    """Validate the whole estimate request before starting SimulationApp."""
    if raw is None:
        return None
    raw = _object(raw, {"schema_version", "source", "robot", "contact", "task_parts", "scene_solver"}, "physics_overrides")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported physics_overrides schema_version")
    source = _object(raw.get("source"), {"kind", "method", "real_world_measured", "description"}, "physics_overrides.source")
    if (source.get("kind") != "geometry_estimate" or source.get("method") != ESTIMATION_METHOD
            or source.get("real_world_measured") is not False or not isinstance(source.get("description"), str)
            or not source["description"].strip()):
        raise ValueError("physics_overrides must explicitly identify an unmeasured collision geometry estimate")
    robot = _object(raw.get("robot"), {"recompute_inertia", "preserve_authored_mass", "solver_position_iterations", "solver_velocity_iterations"}, "physics_overrides.robot")
    if robot.get("recompute_inertia") is not True or robot.get("preserve_authored_mass") is not True:
        raise ValueError("robot estimate must recompute inertia and preserve authored mass")
    normalized_robot = {"recompute_inertia": True, "preserve_authored_mass": True}
    for kind, minimum in (("position", 1), ("velocity", 0)):
        key = f"solver_{kind}_iterations"
        normalized_robot[key] = _iterations(robot.get(key), f"robot.{key}", minimum)
    contact = _object(raw.get("contact"), {"contact_offset_m", "rest_offset_m"}, "physics_overrides.contact")
    offsets = {}
    for key in ("contact_offset_m", "rest_offset_m"):
        value = contact.get(key)
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"physics_overrides.{key} must be finite")
        offsets[key] = float(value)
    if offsets["rest_offset_m"] < 0 or offsets["contact_offset_m"] <= offsets["rest_offset_m"]:
        raise ValueError("contact_offset_m must exceed nonnegative rest_offset_m")
    parts = raw.get("task_parts", {})
    if not isinstance(parts, dict):
        raise ValueError("physics_overrides.task_parts must be an object")
    normalized_parts = {}
    for name, values in parts.items():
        if not isinstance(name, str) or not name or any(character in name for character in "/\\:"):
            raise ValueError("physics_overrides task part ID is invalid")
        values = _object(values, {"solver_position_iterations", "solver_velocity_iterations"}, f"task_parts.{name}")
        normalized_parts[name] = {f"solver_{kind}_iterations": _iterations(values.get(f"solver_{kind}_iterations"), f"task_parts.{name}.{kind}", minimum)
                                  for kind, minimum in (("position", 1), ("velocity", 0))}
    scene_raw = _object(raw.get("scene_solver", {}), {f"{limit}_{kind}_iterations" for limit in ("min", "max") for kind in ("position", "velocity")}, "physics_overrides.scene_solver")
    scene = {key: _iterations(value, f"scene_solver.{key}", 1 if "position" in key else 0, optional=False)
             for key, value in scene_raw.items()}
    for kind in ("position", "velocity"):
        if scene.get(f"min_{kind}_iterations", 1 if kind == "position" else 0) > scene.get(f"max_{kind}_iterations", 255):
            raise ValueError("scene solver minimum cannot exceed maximum")
    return {"schema_version": SCHEMA_VERSION, "source": copy.deepcopy(source), "robot": normalized_robot,
            "contact": offsets, "task_parts": normalized_parts, "scene_solver": scene}


def _serial_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except TypeError:
        number = float(value)
        return number if math.isfinite(number) else "automatic"
    return values if all(math.isfinite(item) for item in values) else "automatic"


def apply_physics_overrides(stage: Any, robot_root: str, object_prims: dict[str, Any], raw: Any) -> dict[str, Any]:
    """Author an estimate into the anonymous session layer before PhysX loads it."""
    config = normalize_physics_overrides(raw)
    if config is None:
        return {}
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0) or not math.isclose(UsdPhysics.GetStageKilogramsPerUnit(stage), 1.0):
        raise ValueError("physics estimate requires meter/kilogram stage units")
    robot = stage.GetPrimAtPath(robot_root)
    if not robot:
        raise ValueError("physics override robot root is missing")
    bodies = [prim for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies or any(prim.IsInstanceProxy() for prim in bodies):
        raise ValueError("physics estimate requires editable robot rigid-body roots")
    for object_id in config["task_parts"]:
        if object_id not in object_prims:
            raise ValueError(f"physics override task part is missing: {object_id}")
    report = {"schema_version": SCHEMA_VERSION, "configuration": config, "source": config["source"],
              "authoring_scope": "anonymous_stage_session_layer", "source_layers_written": False,
              "robot_links": [], "contacts": [], "task_parts": [], "live_verified": False}
    contact_roots = [robot, *(object_prims[name] for name in config["task_parts"])]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # Instance proxy attributes cannot be overridden directly. Deinstance
        # only selected robot/task roots in this stage's anonymous layer.
        for root in contact_roots:
            instances = [prim.GetPath() for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if prim.IsInstance()]
            for path in instances:
                prim = stage.GetPrimAtPath(path)
                if prim and not prim.IsInstanceProxy():
                    prim.SetInstanceable(False)
        for body in bodies:
            body = stage.GetPrimAtPath(body.GetPath())
            mass_api = UsdPhysics.MassAPI(body)
            mass = mass_api.GetMassAttr().Get()
            if mass is None or not math.isfinite(float(mass)) or mass <= 0:
                raise ValueError(f"robot body needs an explicit positive mass: {body.GetPath()}")
            colliders = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                         if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
            if not colliders:
                raise ValueError(f"robot body has no enabled collision geometry: {body.GetPath()}")
            geometry = []
            body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            bounds = Gf.Range3d()
            for prim in colliders:
                mesh = UsdGeom.Mesh(prim)
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if mesh else None
                if not mesh or approximation not in {"convexHull", "convexDecomposition"}:
                    raise ValueError(f"robot estimate requires convex collision meshes: {prim.GetPath()}")
                points = mesh.GetPointsAttr().Get()
                if not points or len(points) < 4:
                    raise ValueError("collision estimate has insufficient mesh vertices")
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * body_world.GetInverse()
                transformed = [list(transform.Transform(Gf.Vec3d(point))) for point in points]
                if not all(math.isfinite(component) for point in transformed for component in point):
                    raise ValueError("collision estimate geometry is nonfinite")
                for point in transformed:
                    bounds.UnionWith(Gf.Vec3d(*point))
                geometry.append({"path": str(prim.GetPath()), "approximation": str(approximation), "points_body_m": transformed})
            size = list(bounds.GetSize())
            if not all(math.isfinite(item) and item > 1e-8 for item in size):
                raise ValueError("collision estimate geometry has degenerate volume")
            original = {"mass_kg": float(mass), "diagonal_inertia_kg_m2": _serial_value(mass_api.GetDiagonalInertiaAttr().Get()),
                        "center_of_mass_m": _serial_value(mass_api.GetCenterOfMassAttr().Get())}
            geometry_digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            collision_mass_overrides = []
            for collider in colliders:
                # Nested MassAPI opinions otherwise retain their old COM even
                # when the rigid-body COM requests automatic computation.
                collider_mass = UsdPhysics.MassAPI(collider)
                collision_mass_overrides.append({"path": str(collider.GetPath()),
                    "original_mass_kg": _serial_value(collider_mass.GetMassAttr().Get()),
                    "original_density_kg_m3": _serial_value(collider_mass.GetDensityAttr().Get()),
                    "original_center_of_mass_m": _serial_value(collider_mass.GetCenterOfMassAttr().Get()),
                    "original_diagonal_inertia_kg_m2": _serial_value(collider_mass.GetDiagonalInertiaAttr().Get())})
                collider_mass = UsdPhysics.MassAPI.Apply(collider)
                collider_mass.CreateMassAttr().Set(0)
                collider_mass.CreateDensityAttr().Set(0)
                collider_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0))
                collider_mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(float("-inf")))
                collider_mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(0))
            # USD Physics defaults request engine computation from collision
            # volume. Explicit mass wins over density; COM/axes are recomputed.
            mass_api = UsdPhysics.MassAPI.Apply(body)
            mass_api.CreateMassAttr().Set(float(mass))
            mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0))
            mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(float("-inf")))
            mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(0))
            report["robot_links"].append({"name": body.GetName(), "path": str(body.GetPath()), "original": original,
                "estimate_method": ESTIMATION_METHOD, "collision_geometry_sha256": geometry_digest,
                "geometry_bbox_size_m": size, "geometry_bbox_min_m": list(bounds.GetMin()),
                "geometry_bbox_max_m": list(bounds.GetMax()), "preserved_mass_kg": float(mass),
                "collision_mass_overrides": collision_mass_overrides,
                "authored_inertia": "automatic_collision_volume", "authored_com_and_principal_axes": "automatic_collision_volume"})
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(robot)
        for kind in ("position", "velocity"):
            value = config["robot"][f"solver_{kind}_iterations"]
            if value is not None:
                getattr(articulation, f"CreateSolver{kind.title()}IterationCountAttr")().Set(value)
        for root in contact_roots:
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    continue
                if prim.IsInstanceProxy():
                    raise ValueError(f"physics contact override remains an instance proxy: {prim.GetPath()}")
                original = {key: _serial_value(prim.GetAttribute("physxCollision:" + key).Get())
                            for key in ("contactOffset", "restOffset")}
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr().Set(config["contact"]["contact_offset_m"])
                api.CreateRestOffsetAttr().Set(config["contact"]["rest_offset_m"])
                loaded_offsets = {"contact_offset_m": float(api.GetContactOffsetAttr().Get()),
                                  "rest_offset_m": float(api.GetRestOffsetAttr().Get())}
                if not all(math.isclose(loaded_offsets[key], value, rel_tol=1e-6, abs_tol=1e-9)
                           for key, value in config["contact"].items()):
                    raise RuntimeError("physics contact override USD readback differs")
                report["contacts"].append({"path": str(prim.GetPath()), "original": original,
                                           "effective_authored_readback": loaded_offsets,
                                           "readback_scope": "USD API authoring; automatic distance is not claimed"})
        for name, values in config["task_parts"].items():
            prim = object_prims[name]
            requests = {kind: values[f"solver_{kind}_iterations"] for kind in ("position", "velocity")}
            if any(value is not None for value in requests.values()) and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"solver counts require a dynamic task rigid body: {name}")
            if any(value is not None for value in requests.values()):
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                for kind, value in requests.items():
                    if value is not None:
                        getattr(api, f"CreateSolver{kind.title()}IterationCountAttr")().Set(value)
                readback = {kind: int(getattr(api, f"GetSolver{kind.title()}IterationCountAttr")().Get())
                            for kind in requests if requests[kind] is not None}
                if any(readback[kind] != value for kind, value in requests.items() if value is not None):
                    raise RuntimeError("task rigid-body solver override USD readback differs")
            else:
                readback = {}
            report["task_parts"].append({"object_id": name, "requested_solver_counts": requests,
                                         "authored_solver_readback": readback})
        scene = stage.GetPrimAtPath("/World/PhysicsScene")
        api = PhysxSchema.PhysxSceneAPI.Apply(scene)
        for key, value in config["scene_solver"].items():
            limit, kind, _ = key.split("_")
            getattr(api, f"Create{limit.title()}{kind.title()}IterationCountAttr")().Set(value)
        report["scene_solver_authored_readback"] = {}
        for key, expected in config["scene_solver"].items():
            limit, kind, _ = key.split("_")
            observed = getattr(api, f"Get{limit.title()}{kind.title()}IterationCountAttr")().Get()
            if observed != expected:
                raise RuntimeError("physics scene solver override USD readback differs")
            report["scene_solver_authored_readback"][key] = int(observed)
    report["session_layer_sha256"] = hashlib.sha256(stage.GetSessionLayer().ExportToString().encode()).hexdigest()
    return report


def verify_physics_overrides(robot: Any, report: dict[str, Any], *, stage: Any = None) -> dict[str, Any]:
    """Reject absent, degenerate or implausibly scaled live geometry estimates."""
    if not report:
        return {}
    import numpy as np

    view = robot._articulation_view
    names = list(view.body_names)
    masses = np.asarray(view.get_body_masses(), dtype=float).reshape(-1)
    inertias = np.asarray(view.get_body_inertias(), dtype=float).reshape(-1, 3, 3)
    centers, axes = view.get_body_coms()
    centers, axes = np.asarray(centers, dtype=float).reshape(-1, 3), np.asarray(axes, dtype=float).reshape(-1, 4)
    if not (len(names) == len(masses) == len(inertias) == len(centers) == len(axes)):
        raise RuntimeError("physics override tensor dimensions are inconsistent")
    result = copy.deepcopy(report)
    configuration = result.get("configuration", {})
    result["robot_solver_live_readback"] = {}
    for kind in ("position", "velocity"):
        requested = configuration.get("robot", {}).get(f"solver_{kind}_iterations")
        if requested is not None:
            observed = int(getattr(robot, f"get_solver_{kind}_iteration_count")())
            if observed != requested:
                raise RuntimeError("physics override robot solver runtime differs from requested counts")
            result["robot_solver_live_readback"][kind] = observed
    if stage is not None:
        scene = stage.GetPrimAtPath("/World/PhysicsScene")
        result["scene_solver_after_initialization"] = {}
        for key, requested in configuration.get("scene_solver", {}).items():
            limit, kind, _ = key.split("_")
            name = f"physxScene:{limit}{kind.title()}IterationCount"
            observed = scene.GetAttribute(name).Get()
            if observed != requested:
                raise RuntimeError("physics context changed the requested scene solver bounds")
            result["scene_solver_after_initialization"][key] = int(observed)
        for row in result.get("contacts", []):
            collider = stage.GetPrimAtPath(row["path"])
            values = {"contact_offset_m": collider.GetAttribute("physxCollision:contactOffset").Get(),
                      "rest_offset_m": collider.GetAttribute("physxCollision:restOffset").Get()}
            if not all(value is not None and math.isclose(float(value), configuration["contact"][key], rel_tol=1e-6, abs_tol=1e-9)
                       for key, value in values.items()):
                raise RuntimeError("physics contact overrides changed during initialization")
            row["after_initialization_USD_readback"] = values
    for row in result["robot_links"]:
        if row["name"] not in names:
            raise RuntimeError("physics override body is missing from articulation")
        index = names.index(row["name"])
        mass, inertia = masses[index], inertias[index]
        # PhysX returns FP32 tensors. Frame rotations introduce roundoff scaled
        # by the tensor, including tiny off-diagonal terms near zero. Comparing
        # each entry with a fixed 1e-12 bound rejected a live 7.28e-12 residual
        # in a 1.39e-2 tensor. Bound the antisymmetric residual at eight FP32
        # epsilons of its largest entry; preserve the exact raw matrix and never
        # symmetrize it or write a correction back into the simulator.
        tensor_scale = float(np.max(np.abs(inertia)))
        symmetry_error = float(np.max(np.abs(inertia - inertia.T)))
        symmetry_tolerance = float(8 * np.finfo(np.float32).eps * tensor_scale)
        if not np.isfinite(inertia).all() or symmetry_error > symmetry_tolerance:
            raise RuntimeError(f"physics override inertia tensor is nonfinite or asymmetric: {row['name']}; "
                               f"residual={symmetry_error}, fp32_bound={symmetry_tolerance}")
        eigenvalues = np.linalg.eigvalsh(inertia)
        scale = float(sum(item**2 for item in row["geometry_bbox_size_m"]))
        if (not math.isfinite(mass) or not np.isclose(mass, row["preserved_mass_kg"], rtol=1e-5, atol=1e-9)
                or np.any(eigenvalues <= 0) or eigenvalues[-1] > eigenvalues[:2].sum() * (1 + 1e-4)
                or eigenvalues[-1] > mass * scale or eigenvalues[-1] < mass * scale * 1e-5
                or not np.isfinite(centers[index]).all() or not np.isfinite(axes[index]).all()
                or not np.isclose(np.linalg.norm(axes[index]), 1, atol=1e-4)):
            raise RuntimeError(f"physics override live mass/inertia estimate invalid: {row['name']}")
        if "geometry_bbox_min_m" in row and (
                np.any(centers[index] < np.asarray(row["geometry_bbox_min_m"]) - 1e-5)
                or np.any(centers[index] > np.asarray(row["geometry_bbox_max_m"]) + 1e-5)):
            raise RuntimeError(f"physics override computed COM is outside collision geometry bounds: {row['name']}")
        row["live"] = {"mass_kg": float(mass), "inertia_kg_m2": inertia.tolist(),
                       "center_of_mass_m": centers[index].tolist(), "principal_axes_wxyz": axes[index].tolist(),
                       "symmetry_residual_kg_m2": symmetry_error,
                       "symmetry_fp32_bound_kg_m2": symmetry_tolerance,
                       "source": "live_PhysX_articulation_tensor", "verified": True}
    result["live_verified"] = True
    return result
