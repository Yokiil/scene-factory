"""Live PhysX object states and explicitly scoped stage contact measurements."""
from __future__ import annotations

from typing import Any


def prepare_contact_reporting(stage: Any) -> None:
    """Request reports before PhysX creates actors; preserve sleep/solver settings."""
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)


class IsaacObservationAdapter:
    def __init__(self, stage: Any, object_prims: dict[str, Any], physics_view: Any, dt: float,
                 required_dynamic_ids: set[str] | None = None):
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.api.sensors import RigidContactView
        from pxr import PhysxSchema, Usd, UsdPhysics
        from .runtime_worker import _new_rigid_view

        self.stage, self.objects, self.dt = stage, object_prims, dt
        self.motion_views = {}
        for object_id, prim in object_prims.items():
            if object_id in (required_dynamic_ids or set()) and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise NotImplementedError(f"dynamic object has no supported root PhysX body: {object_id}")
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                view = _new_rigid_view(RigidPrim, str(prim.GetPath()), "session_" + object_id)
                view.initialize(physics_view)
                self.motion_views[object_id] = view
        self.colliders = []
        self.owner_by_collider = {}
        self.sensor_roots = {}
        self.contact_views = {}
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            path = str(prim.GetPath())
            owner = next((key for key, root in object_prims.items()
                          if prim.GetPath().HasPrefix(root.GetPath())), None)
            if owner is None:
                owner = "robot" if path.startswith("/World/Robot/") else "stage:" + path
            parent = prim
            while parent and parent.IsValid() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            self.colliders.append(path)
            self.owner_by_collider[path] = owner
            # Static collision shapes are valid filters but are not rigid-body
            # sensors. Every solver contact has at least one moving rigid body.
            if parent and parent.IsValid():
                if not parent.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    raise NotImplementedError(f"rigid sensor has no contact report API: {parent.GetPath()}")
                threshold = PhysxSchema.PhysxContactReportAPI(parent).GetThresholdAttr().Get()
                if threshold is None or float(threshold) > 0:
                    raise NotImplementedError(f"rigid sensor contact threshold omits small forces: {parent.GetPath()}")
                self.sensor_roots[str(parent.GetPath())] = owner
        self.colliders.sort()
        if not self.colliders:
            raise RuntimeError("scene contains no collision shapes for contact coverage")
        for index, (path, owner) in enumerate(sorted(self.sensor_roots.items())):
            # Measurement must not alter sleep thresholds or solver stabilization.
            view = RigidContactView(prim_paths_expr=path, filter_paths_expr=self.colliders,
                                    name=f"session_general_contact_{index}",
                                    prepare_contact_sensors=False, disable_stablization=False,
                                    max_contact_count=8192)
            view.initialize(physics_view)
            self.contact_views[path] = (owner, view)

    def object_states(self) -> dict[str, Any]:
        from pxr import Gf, Usd, UsdGeom
        from .runtime_worker import _live_rigid_state

        result = {}
        for object_id, prim in self.objects.items():
            if object_id in self.motion_views:
                result[object_id] = _live_rigid_state(self.motion_views[object_id], object_id)
            else:
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                quaternion = Gf.Transform(matrix).GetRotation().GetQuat()
                result[object_id] = {
                    "position": [float(v) for v in matrix.ExtractTranslation()],
                    "orientation_wxyz": [float(quaternion.GetReal()), *map(float, quaternion.GetImaginary())],
                    "linear_velocity_mps": [0.0] * 3, "angular_velocity_rps": [0.0] * 3,
                    "pose_source": "USD static transform", "velocity_source": "static object",
                }
        return result

    def contacts(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        import numpy as np

        directional, resultants, point_counts = {}, {}, {}
        buffers = []
        for path, (owner, view) in self.contact_views.items():
            matrix = np.asarray(view.get_contact_force_matrix(dt=self.dt), dtype=float)
            if matrix.ndim != 3 or matrix.shape[1:] != (len(self.colliders), 3) or not np.isfinite(matrix).all():
                raise RuntimeError(f"invalid general contact measurement for {path}: {matrix.shape}")
            data = view.get_contact_force_data(dt=self.dt)
            if not isinstance(data, (tuple, list)) or len(data) != 6:
                raise RuntimeError(f"per-contact-point force measurements unavailable for {path}")
            force, points, normals, distances, counts, starts = map(np.asarray, data)
            if force.ndim != 2 or force.shape[1] != 1 or points.shape != (len(force), 3) or normals.shape != points.shape or distances.shape != force.shape:
                raise RuntimeError(f"invalid per-contact buffers for {path}")
            if counts.shape != matrix.shape[:2] or starts.shape != counts.shape or not np.issubdtype(counts.dtype, np.integer) or not np.issubdtype(starts.dtype, np.integer):
                raise RuntimeError(f"invalid contact pair indices for {path}")
            counts, starts = counts.astype(np.int64), starts.astype(np.int64)
            used = int(counts.sum())
            capacity = len(force)
            if np.any(counts < 0) or np.any((counts > 0) & ((starts < 0) | (starts + counts > capacity))):
                raise RuntimeError(f"contact point buffer indices out of bounds for {path}")
            # At capacity the API does not prove that additional points were not
            # truncated. Reject even an exactly full buffer conservatively.
            if capacity == 0 or used >= capacity or np.any((counts > 0) & (starts + counts >= capacity)):
                raise RuntimeError(f"contact point buffer saturated for {path}; force acceptance is unavailable")
            point_force = np.zeros(counts.shape, dtype=float)
            for row, column in zip(*np.nonzero(counts), strict=True):
                start, count = int(starts[row, column]), int(counts[row, column])
                selected = slice(start, start + count)
                if not all(np.isfinite(values[selected]).all() for values in (force, points, normals, distances)):
                    raise RuntimeError(f"non-finite active contact point data for {path}")
                point_force[row, column] = float(np.abs(force[selected]).sum())
            if np.any(point_force + 1e-4 < np.linalg.norm(matrix, axis=2)):
                raise RuntimeError(f"contact point forces do not cover the measured resultant for {path}")
            norms = np.linalg.norm(matrix, axis=2).sum(axis=0)
            magnitudes = point_force.sum(axis=0)
            counts_by_filter = counts.sum(axis=0)
            for collider, magnitude, resultant, count in zip(self.colliders, magnitudes, norms, counts_by_filter, strict=True):
                other = self.owner_by_collider[collider]
                if other == owner or (magnitude <= 1e-7 and count == 0):
                    continue
                pair = (owner, other)
                directional[pair] = directional.get(pair, 0.0) + float(magnitude)
                resultants[pair] = resultants.get(pair, 0.0) + float(resultant)
                point_counts[pair] = point_counts.get(pair, 0) + int(count)
            buffers.append({"sensor_path": path, "used_contact_points": used, "capacity": capacity,
                            "saturated": False})
        pairs, pair_resultants, pair_counts = {}, {}, {}
        for (owner, other), magnitude in directional.items():
            pair = tuple(sorted((owner, other)))
            pairs[pair] = max(pairs.get(pair, 0.0), magnitude)
            pair_resultants[pair] = max(pair_resultants.get(pair, 0.0), resultants[(owner, other)])
            pair_counts[pair] = max(pair_counts.get(pair, 0), point_counts[(owner, other)])
        contacts = [{"pair": list(pair), "normal_force_N": magnitude,
                     "normal_force_resultant_N": pair_resultants[pair], "contact_point_count": pair_counts[pair],
                     "measurement": "sum_of_absolute_per_contact_point_normal_forces",
                     "sample_window_s": self.dt, "units": "N"} for pair, magnitude in sorted(pairs.items())]
        return contacts, {
            "status": "measured", "collision_shape_count": len(self.colliders),
            "sensor_root_count": len(self.contact_views), "collider_paths": list(self.colliders),
            "complete": True, "includes_instance_proxies": True,
            "point_force_measurement": "sum_of_absolute_per_contact_point_normal_forces",
            "buffer_saturated": False, "contact_buffers": buffers,
            "scope": "all_scene_contacts", "continuous": True,
            "object_ids": sorted(set(self.owner_by_collider.values())),
            "force_aggregation": "sum_per_point_normal_magnitudes",
            "sample_dt_s": self.dt,
            "source": "PhysX RigidContactView.get_contact_force_data",
            "geometry_scope": "enabled CollisionAPI shapes on the loaded stage, including instance proxies",
            "excluded_from_pair_output": ["same_owner_internal_contacts", "static_static_pairs"],
            "unavailable": ["friction_force_vectors", "soft_body_contacts"],
            "sampling": "sum of absolute per-contact-point normal forces over one physics step",
        }
