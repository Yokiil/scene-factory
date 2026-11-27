"""Incremental MJCF generation and episode-boundary MuJoCo parameter updates.

This module has deliberately narrow, executable contracts. Object fragments are
content addressed; only free-body poses and explicitly authored mass/inertia/
friction can update an existing model. Every other change invokes the native
compiler. No saved physics or task evidence is reused by this module.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

from .asset_geometry import load_asset_parts
from .exporters.mujoco_mjcf import MujocoMjcfExporter
from .physics_dependencies import plan_scene_update
from .registry import AssetRegistry


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _xml(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"


def _unindent(element: ET.Element) -> None:
    for child in element.iter():
        if child.text is not None and not child.text.strip():
            child.text = None
        if child.tail is not None and not child.tail.strip():
            child.tail = None


def compiler_source_paths() -> list[Path]:
    """Ordered, explicit source closure used by the compilation cache key."""
    modules = [Path(__file__), Path(__file__).parent / "exporters/mujoco_mjcf.py",
               Path(__file__).parent / "asset_geometry.py", Path(__file__).parent / "models.py",
               Path(__file__).parent / "registry.py", Path(__file__).parent / "physics_dependencies.py"]
    pose_math = Path(__file__).parent / "pose_math.py"
    if pose_math.is_file():
        modules.append(pose_math)
    return modules


class ObjectFragmentCompiler:
    """Reuse untouched object fragments without regenerating their geometry.

    Full scene settings still go through the normal exporter, with no objects.
    Sanitized-name collisions use the normal full exporter, preserving its
    documented index-based naming rather than inventing a different topology.
    """

    def __init__(self, registry: AssetRegistry, *, cache_dir: str | Path | None = None, max_cached_fragments: int = 4096):
        if isinstance(max_cached_fragments, bool) or not isinstance(max_cached_fragments, int) or max_cached_fragments < 1:
            raise ValueError("max_cached_fragments must be a positive integer")
        self.registry = registry
        self.exporter = MujocoMjcfExporter(registry)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_cached_fragments = max_cached_fragments
        self.source_sha256 = _digest(b"".join(path.read_bytes() for path in compiler_source_paths()))

    def dependency_fingerprints(self, scene: dict[str, Any]) -> dict[str, str]:
        """Hash each shared asset once per request, including mutable source files."""
        result = {}
        for asset_id in sorted({item["asset_id"] for item in scene["objects"]}):
            asset = self.registry.get(asset_id)
            sources = {}
            if "kitchen_dining_pack" in asset.tags:
                directory = self.registry.base_dir / "source" / asset.asset_id
                sources = {name: _digest((directory / name).read_bytes())
                           for name in ("SOURCE.json", "geometry.json")}
            result[asset_id] = _digest(_canonical({"asset": asdict(asset), "sources": sources}))
        return result

    def _key(self, item: dict[str, Any], fingerprints: dict[str, str]) -> str:
        return _digest(_canonical({"object": item, "asset": fingerprints[item["asset_id"]],
                                   "compiler": self.source_sha256}))

    def export(self, scene: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _canonical(scene)  # Reject non-finite values before touching a cache.
        ids = [str(item["object_id"]) for item in scene["objects"]]
        names = [self.exporter.body_name(value) for value in ids]
        if len(ids) != len(set(ids)):
            raise ValueError("incremental scene object IDs must be unique")
        if len(names) != len(set(names)):
            return self.exporter.to_string(scene), {
                "fragment_hits": 0, "fragment_misses": len(ids),
                "fallback": "sanitized_name_collision", "rebuilt_objects": ids,
            }
        header = {**scene, "objects": []}
        root = ET.fromstring(self.exporter.to_string(header))
        _unindent(root)
        if any("kitchen_dining_pack" in self.registry.get(item["asset_id"]).tags
               for item in scene["objects"]):
            ET.SubElement(root.find("option"), "flag", {"multiccd": "enable"})
        assets, world = root.find("asset"), root.find("worldbody")
        hits, rebuilt = 0, []
        fingerprints = self.dependency_fingerprints(scene)
        for item in scene["objects"]:
            key = self._key(item, fingerprints)
            record = self.cache.get(key)
            path = self.cache_dir / f"{key}.json" if self.cache_dir else None
            if record is None and path and path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                content = record.get("content")
                if record.get("key") != key or _digest(_canonical(content)) != record.get("sha256"):
                    raise ValueError("incremental object cache is corrupt")
                self.cache[key] = record
            if record is None:
                standalone = ET.fromstring(self.exporter.to_string({**scene, "objects": [item]}))
                fragment_assets = standalone.find("asset")
                fragment_body = standalone.find("./worldbody/body")
                _unindent(fragment_body)
                for mesh in fragment_assets:
                    _unindent(mesh)
                content = {"body": ET.tostring(fragment_body, encoding="unicode"),
                           "assets": [ET.tostring(mesh, encoding="unicode") for mesh in fragment_assets]}
                record = {"key": key, "content": content, "sha256": _digest(_canonical(content))}
                self.cache[key] = record
                if path:
                    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(_canonical(record))
                    try:
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                rebuilt.append(item["object_id"])
            else:
                hits += 1
            self.cache.move_to_end(key)
            while len(self.cache) > self.max_cached_fragments:
                self.cache.popitem(last=False)
            world.append(ET.fromstring(record["content"]["body"]))
            for mesh in record["content"]["assets"]:
                assets.append(ET.fromstring(mesh))
        return _xml(root), {"fragment_hits": hits, "fragment_misses": len(rebuilt),
                            "rebuilt_objects": rebuilt, "fallback": None}


def motion_envelope_query(
    before: dict[str, Any], after: dict[str, Any], changed_ids: set[str], *,
    horizon_s: float, max_linear_speed_mps: float,
    geometry_radii_m: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Query both old and new geometry, including previously absent contacts.

    Bounding spheres contain every orientation of the authored box. Dynamic
    spheres expand by v_max*T. The speed bound is a caller condition, not a
    solver guarantee: consumers must monitor it and fall back on violation.
    Shared fixed scenery does not transmit motion between free-body islands.
    """
    for value in (horizon_s, max_linear_speed_mps):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("motion envelope horizon and speed must be finite and nonnegative")
    old = {item["object_id"]: item for item in before["objects"]}
    new = {item["object_id"]: item for item in after["objects"]}
    for item in [*old.values(), *new.values()]:
        size, position = item["bbox_m"], item["pose"]["position"]
        if len(size) != 3 or len(position) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in [*size, *position]
        ) or any(value <= 0 for value in size):
            raise ValueError("motion envelope requires finite poses and positive geometry bounds")
    if geometry_radii_m and any(not isinstance(value, (int, float)) or isinstance(value, bool)
                                or not math.isfinite(value) or value < 0 for value in geometry_radii_m.values()):
        raise ValueError("actual geometry radii must be finite and nonnegative")

    def pairs(objects: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
        result = set()
        keys = sorted(objects)
        for index, left_id in enumerate(keys):
            left = objects[left_id]
            for right_id in keys[index + 1:]:
                right = objects[right_id]
                if not left["dynamic"] and not right["dynamic"]:
                    continue
                left_radius = math.sqrt(sum(float(v)**2 for v in left["bbox_m"])) / 2
                right_radius = math.sqrt(sum(float(v)**2 for v in right["bbox_m"])) / 2
                if geometry_radii_m:
                    left_radius = max(left_radius, geometry_radii_m.get(left_id, 0.))
                    right_radius = max(right_radius, geometry_radii_m.get(right_id, 0.))
                travel = (int(left["dynamic"]) + int(right["dynamic"])) * horizon_s * max_linear_speed_mps
                distance = math.dist(left["pose"]["position"], right["pose"]["position"])
                if distance <= left_radius + right_radius + travel:
                    result.add((left_id, right_id))
        return result

    old_pairs, new_pairs = pairs(old), pairs(new)
    union = old_pairs | new_pairs
    closure = set(changed_ids)
    while True:
        previous = set(closure)
        for left, right in union:
            # A static support affects its touching dynamics, but two dynamics
            # do not become mechanically coupled just by sharing a fixed body.
            if left in closure and (left in changed_ids or old.get(left, new.get(left))["dynamic"]):
                closure.add(right)
            if right in closure and (right in changed_ids or old.get(right, new.get(right))["dynamic"]):
                closure.add(left)
        if closure == previous:
            break
    return {
        "schema_version": "scene_factory.motion_envelope_query.v1",
        "horizon_s": horizon_s, "max_linear_speed_mps": max_linear_speed_mps,
        "bound_status": "caller_condition_requires_runtime_monitoring",
        "old_candidate_pairs": [list(pair) for pair in sorted(old_pairs)],
        "new_candidate_pairs": [list(pair) for pair in sorted(new_pairs)],
        "newly_possible_pairs": [list(pair) for pair in sorted(new_pairs - old_pairs)],
        "affected_objects": sorted(closure),
        "evidence_policy": "full_dynamics_and_task_revalidation",
        "geometry_bound": "authored_box_plus_actual_mesh_radius" if geometry_radii_m else "authored_box_only_not_independent_geometry_proof",
    }


def _simplify_static_boxes(xml: str, object_ids: set[str]) -> tuple[str, list[dict[str, Any]]]:
    """Replace exact convex box meshes; reject other shapes or moving objects."""
    root = ET.fromstring(xml)
    assets = root.find("asset")
    meshes = {mesh.attrib["name"]: mesh for mesh in assets}
    proofs = []
    removed = set()
    selected_names = {MujocoMjcfExporter.body_name(key): key for key in object_ids}
    for body in root.findall("./worldbody/body"):
        selected = selected_names.get(body.attrib["name"])
        if selected is None:
            continue
        if body.find("freejoint") is not None:
            raise ValueError("precision partition may simplify only fixed background bodies")
        for geom in body.findall("geom"):
            if geom.attrib.get("type") != "mesh":
                continue
            mesh_name = geom.attrib["mesh"]
            mesh = meshes[mesh_name]
            numbers = [float(value) for value in mesh.attrib["vertex"].split()]
            vertices = [tuple(numbers[index:index + 3]) for index in range(0, len(numbers), 3)]
            low = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
            high = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
            corners = {(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])}
            if len(corners) != 8 or set(vertices) != corners:
                raise ValueError(f"background mesh is not an exact convex box: {selected}")
            center = [(a + b) / 2 for a, b in zip(low, high, strict=True)]
            size = [(b - a) / 2 for a, b in zip(low, high, strict=True)]
            geom.attrib.pop("mesh")
            geom.set("type", "box")
            geom.set("pos", " ".join(f"{value:.9g}" for value in center))
            geom.set("size", " ".join(f"{value:.9g}" for value in size))
            removed.add(mesh_name)
            proofs.append({"object_id": selected, "mesh": mesh_name,
                           "geometry_proof": "convex_hull_of_all_eight_box_corners",
                           "background_is_fixed": True, "contact_algorithm_changed": True})
    for name in removed:
        if not any(geom.attrib.get("mesh") == name for geom in root.iter("geom")):
            assets.remove(meshes[name])
    return _xml(root), proofs


def _patch_description(xml: str) -> tuple[str, dict[str, Any]]:
    root = ET.fromstring(xml)
    patches = {}
    for body in root.findall("./worldbody/body"):
        if body.find("freejoint") is None or body.find("body") is not None:
            continue
        patch = {"position": [float(v) for v in body.attrib.pop("pos").split()],
                 "quaternion": [float(v) for v in body.attrib.pop("quat").split()]}
        inertial = body.find("inertial")
        geoms = body.findall("geom")
        if inertial is not None and set(inertial.attrib) == {"pos", "mass", "diaginertia"} and inertial.attrib["pos"] == "0 0 0" and all(geom.attrib.get("type") != "mesh" for geom in geoms):
            patch["mass"] = float(inertial.attrib.pop("mass"))
            patch["inertia"] = [float(v) for v in inertial.attrib.pop("diaginertia").split()]
            patch["friction"] = {}
            for geom in geoms:
                patch["friction"][geom.attrib["name"]] = [float(v) for v in geom.attrib.pop("friction").split()]
                # With explicit inertial this value is unused by MuJoCo.
                geom.attrib.pop("mass", None)
        patches[body.attrib["name"]] = patch
    _unindent(root)
    return ET.tostring(root, encoding="unicode"), patches


@dataclass
class CompilationResult:
    model: Any
    data: Any
    xml: str
    report: dict[str, Any]


class IncrementalMujocoCompiler:
    """Executable full/cache/incremental/partition strategies for comparison.

    Call only at episode boundaries. Returned data starts a fresh episode;
    a model update never masquerades as restoration of a previous trajectory.
    """

    MODES = {"full", "file_cache", "incremental", "partition"}

    def __init__(
        self, registry: AssetRegistry, *, mode: str = "incremental",
        cache_dir: str | Path | None = None,
        background_ids: set[str] | None = None,
        horizon_s: float = 0.2, max_linear_speed_mps: float = 2.0,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unknown incremental compilation mode: {mode}")
        for value in (horizon_s, max_linear_speed_mps):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("motion envelope horizon and speed must be finite and nonnegative")
        self.registry, self.mode = registry, mode
        self.fragments = ObjectFragmentCompiler(registry, cache_dir=cache_dir)
        self.exporter = MujocoMjcfExporter(registry)
        self.file_cache: OrderedDict[str, str] = OrderedDict()
        self.background_ids = set(background_ids or ())
        self.horizon_s, self.max_linear_speed_mps = horizon_s, max_linear_speed_mps
        self.model = self.data = self.scene = self.signature = None
        self.compile_count = self.patch_count = 0

    def _mesh_radii(self, *scenes: dict[str, Any]) -> dict[str, float]:
        assets = {}
        radii = {}
        for scene in scenes:
            for item in scene["objects"]:
                asset_id = item["asset_id"]
                if asset_id not in assets:
                    parts = load_asset_parts(self.registry, self.registry.get(asset_id))
                    assets[asset_id] = max((math.sqrt(sum(float(value)**2 for value in vertex))
                                            for part in parts or () for vertex in part["vertices"]), default=0.)
                radii[item["object_id"]] = max(radii.get(item["object_id"], 0.), assets[asset_id])
        return radii

    def compile(self, scene: dict[str, Any]) -> CompilationResult:
        import mujoco
        import numpy as np

        started = time.perf_counter()
        scene = copy.deepcopy(scene)
        fragment_report = {"fragment_hits": 0, "fragment_misses": len(scene["objects"]),
                           "rebuilt_objects": [item["object_id"] for item in scene["objects"]], "fallback": None}
        cache_hit = False
        if self.mode in {"incremental", "partition"}:
            xml, fragment_report = self.fragments.export(scene)
        elif self.mode == "file_cache":
            # A whole-input cache cannot reuse a file if one object changes.
            key = _digest(_canonical({"scene": scene, "dependencies": self.fragments.dependency_fingerprints(scene),
                                      "compiler": self.fragments.source_sha256}))
            cache_hit = key in self.file_cache
            xml = self.file_cache.get(key)
            if xml is None:
                xml = self.exporter.to_string(scene)
                self.file_cache[key] = xml
            self.file_cache.move_to_end(key)
            while len(self.file_cache) > 128:
                self.file_cache.popitem(last=False)
        else:
            xml = self.exporter.to_string(scene)
        proofs = []
        envelope = None
        if self.scene is not None:
            old = {item["object_id"]: item for item in self.scene["objects"]}
            new = {item["object_id"]: item for item in scene["objects"]}
            changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
            envelope = motion_envelope_query(self.scene, scene, changed,
                                             horizon_s=self.horizon_s,
                                             max_linear_speed_mps=self.max_linear_speed_mps,
                                             geometry_radii_m=self._mesh_radii(self.scene, scene))
        if self.mode == "partition":
            unknown = self.background_ids - {item["object_id"] for item in scene["objects"]}
            if unknown:
                raise ValueError(f"unknown fixed background object IDs: {sorted(unknown)}")
            # Partition is chosen before the rollout, never switched in contact.
            query = motion_envelope_query(scene, scene, set(), horizon_s=self.horizon_s,
                                          max_linear_speed_mps=self.max_linear_speed_mps,
                                          geometry_radii_m=self._mesh_radii(scene))
            candidates = {key for pair in query["new_candidate_pairs"] for key in pair}
            unsafe = self.background_ids & candidates
            if unsafe:
                raise ValueError(f"precision background intersects a dynamic motion envelope: {sorted(unsafe)}")
            xml, proofs = _simplify_static_boxes(xml, self.background_ids)
        generation_s = time.perf_counter() - started
        signature, patches = _patch_description(xml)
        engine_start = time.perf_counter()
        patch_allowed = self.mode in {"incremental", "partition"} and self.model is not None and self.signature == signature
        kind = "native_compile"
        fallback_reason = ("strategy_requires_native_compile" if self.mode in {"full", "file_cache"}
                           else "initial_model" if self.model is None
                           else "non_patchable_geometry_topology_or_settings_changed" if not patch_allowed else None)
        if patch_allowed:
            # Keep every edited primary array so an unexpected failure can roll
            # back without leaving a partially changed model in the caller.
            arrays = ("body_mass", "body_inertia", "body_pos", "body_quat", "qpos0", "geom_friction")
            backup = {name: getattr(self.model, name).copy() for name in arrays}
            try:
                for name, patch in patches.items():
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    if body_id < 1:
                        raise ValueError(f"patch references missing body: {name}")
                    position, quaternion = np.asarray(patch["position"]), np.asarray(patch["quaternion"])
                    quaternion /= np.linalg.norm(quaternion)
                    joint_id = self.model.body_jntadr[body_id]
                    start = self.model.jnt_qposadr[joint_id]
                    self.model.body_pos[body_id] = position
                    self.model.body_quat[body_id] = quaternion
                    self.model.qpos0[start:start + 7] = [*position, *quaternion]
                    if "mass" in patch:
                        self.model.body_mass[body_id] = patch["mass"]
                        self.model.body_inertia[body_id] = patch["inertia"]
                        for geom_name, friction in patch["friction"].items():
                            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                            self.model.geom_friction[geom_id] = friction
                candidate_data = mujoco.MjData(self.model)
                mujoco.mj_setConst(self.model, candidate_data)
                mujoco.mj_resetData(self.model, candidate_data)
                mujoco.mj_forward(self.model, candidate_data)
                if not np.isfinite(candidate_data.qpos).all() or not np.isfinite(candidate_data.qvel).all():
                    raise ValueError("non-finite state after parameter patch")
                self.data = candidate_data
            except Exception:
                for name, value in backup.items():
                    getattr(self.model, name)[:] = value
                mujoco.mj_setConst(self.model, self.data)
                raise
            kind = "episode_boundary_parameter_patch"
            self.patch_count += 1
        else:
            candidate_model = mujoco.MjModel.from_xml_string(xml)
            candidate_data = mujoco.MjData(candidate_model)
            mujoco.mj_forward(candidate_model, candidate_data)
            self.model, self.data = candidate_model, candidate_data
            self.compile_count += 1
        engine_s = time.perf_counter() - engine_start
        invalidation = plan_scene_update(self.scene, scene) if self.scene else None
        self.scene, self.signature = scene, signature
        report = {
            "schema_version": "scene_factory.incremental_compilation.v1", "mode": self.mode,
            "mujoco_version": mujoco.__version__, "compiler_source_sha256": self.fragments.source_sha256,
            "kind": kind, "xml_sha256": _digest(xml.encode()), "generation_s": generation_s,
            "engine_fallback_reason": fallback_reason,
            "engine_update_s": engine_s, "compile_total_s": time.perf_counter() - started,
            "whole_file_cache_hit": cache_hit, **fragment_report,
            "native_compile_count": self.compile_count, "parameter_patch_count": self.patch_count,
            "invalidation": invalidation, "motion_envelope": envelope,
            "precision_partition": proofs, "dynamics_evidence": "invalid_requires_full_replay",
            "task_evidence": "invalid_requires_task_evaluation",
        }
        return CompilationResult(self.model, self.data, xml, report)
