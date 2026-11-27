"""Portable frozen inputs and stepwise contact comparison for proxy benchmarks."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .dataset import write_json_atomic
from .incremental_compiler import compiler_source_paths
from .registry import AssetRegistry


INPUT_SCHEMA = "scene_factory.incremental_benchmark_inputs.v1"
CONTACT_SCHEMA = "scene_factory.incremental_contact_trace.v2"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_inputs(registry_path: Path, asset_ids: set[str], destination: Path) -> tuple[AssetRegistry, dict[str, Any]]:
    """Consume copied selected raw rows and copied dependencies in all arms."""
    registry_path, destination = registry_path.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError("frozen input directory already exists")
    destination.mkdir(parents=True)
    original = AssetRegistry.load(registry_path)
    entries: list[dict[str, Any]] = []

    def put(relative: str, data: bytes, kind: str, **fields: Any) -> None:
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({"path": relative, "sha256": _digest(data), "size_bytes": len(data), "kind": kind, **fields})

    full = registry_path.read_bytes()
    put("original_registry.jsonl", full, "raw_full_registry")
    selected: list[bytes] = []
    raw_records: dict[str, Any] = {}
    for line in full.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith(b"#"):
            continue
        raw = json.loads(stripped)
        if raw["asset_id"] in asset_ids:
            selected.append(line if line.endswith((b"\n", b"\r")) else line+b"\n")
            raw_records[raw["asset_id"]] = raw
    if set(raw_records) != asset_ids:
        raise ValueError("requested asset rows missing from raw registry")
    put("registry.jsonl", b"".join(selected), "selected_raw_registry_rows", asset_ids=sorted(asset_ids))
    normalized = {asset_id: asdict(original.get(asset_id)) for asset_id in sorted(asset_ids)}
    put("normalized_assets.json", json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False).encode(), "normalized_asset_records")
    for asset_id in sorted(asset_ids):
        asset = original.get(asset_id)
        if "kitchen_dining_pack" in asset.tags:
            for name in ("SOURCE.json", "geometry.json"):
                relative = f"source/{asset_id}/{name}"
                put(relative, (registry_path.parent / relative).read_bytes(), "asset_dependency", asset_id=asset_id)
    project = Path(__file__).resolve().parents[1]
    source_order = []
    for path in compiler_source_paths():
        relative = path.relative_to(project).as_posix()
        source_order.append(f"compiler_sources/{relative}")
        put(source_order[-1], path.read_bytes(), "compiler_source", original_path=relative)
    for path in (Path(__file__), project / "tools/benchmark_incremental_compiler.py", project / "tools/verify_incremental_benchmark.py"):
        relative = path.relative_to(project).as_posix()
        put(f"experiment_sources/{relative}", path.read_bytes(), "experiment_source", original_path=relative)
    manifest = {"schema_version": INPUT_SCHEMA, "original_registry_sha256": _digest(full),
                "selected_asset_ids": sorted(asset_ids), "compiler_source_order": source_order,
                "compiler_source_sha256": _digest(b"".join((destination/path).read_bytes() for path in source_order)),
                "registry_consumed_by_all_arms": "registry.jsonl", "entries": entries,
                "scope": "proxy_exporter_used_dependencies; external_robot_or_USD_assets_not_used"}
    write_json_atomic(destination / "manifest.json", manifest)
    frozen = AssetRegistry.load(destination / "registry.jsonl")
    if {asset_id: asdict(frozen.get(asset_id)) for asset_id in sorted(asset_ids)} != normalized:
        raise ValueError("freezing changed normalized registry records")
    return frozen, manifest


def verify_inputs(directory: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    errors = []
    if _digest(manifest_path.read_bytes()) != expected_manifest_sha256:
        errors.append("input_manifest_hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INPUT_SCHEMA:
        errors.append("input_manifest_schema")
    for entry in manifest["entries"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("frozen input path escapes directory")
        if not path.is_file() or _digest(path.read_bytes()) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            errors.append(f"input_hash:{entry['path']}")
    full = (directory / "original_registry.jsonl").read_bytes()
    if _digest(full) != manifest["original_registry_sha256"]:
        errors.append("full_registry_hash")
    source_paths = [(directory/path).resolve() for path in manifest["compiler_source_order"]]
    if any(not path.is_relative_to(directory) for path in source_paths):
        raise ValueError("compiler source path escapes frozen directory")
    if _digest(b"".join(path.read_bytes() for path in source_paths)) != manifest["compiler_source_sha256"]:
        errors.append("compiler_source_closure_hash")
    expected_rows = []
    for line in full.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#") and json.loads(stripped)["asset_id"] in manifest["selected_asset_ids"]:
            expected_rows.append(line if line.endswith((b"\n", b"\r")) else line+b"\n")
    if (directory/"registry.jsonl").read_bytes() != b"".join(expected_rows):
        errors.append("selected_registry_rows_differ_from_original")
    frozen_registry = AssetRegistry.load(directory/"registry.jsonl")
    normalized = json.loads((directory/"normalized_assets.json").read_text(encoding="utf-8"))
    consumed = json.loads(json.dumps({asset_id: asdict(frozen_registry.get(asset_id)) for asset_id in manifest["selected_asset_ids"]}, allow_nan=False))
    if normalized != consumed:
        errors.append("normalized_registry_differs_from_consumed")
    return {"status": "passed" if not errors else "failed", "errors": errors,
            "checked_inputs": len(manifest["entries"]), "original_registry_sha256": manifest["original_registry_sha256"],
            "compiler_source_sha256": manifest["compiler_source_sha256"]}


def validate_contacts(trace: dict[str, Any], *, expected_steps: int, dt_s: float) -> None:
    if trace.get("schema_version") != CONTACT_SCHEMA or len(trace.get("steps", [])) != expected_steps:
        raise ValueError("contact trace schema or step count mismatch")
    for index, step in enumerate(trace["steps"]):
        if step.get("step_index") != index+1 or not math.isclose(step["sim_time_s"], (index+1)*dt_s, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("contact trace time grid mismatch")
        if step.get("contact_count") != len(step.get("contacts", [])):
            raise ValueError("contact trace incomplete declared count")
        for row in step["contacts"]:
            if not isinstance(row.get("geom_pair"), list) or len(row["geom_pair"]) != 2 or not all(isinstance(value, str) and value for value in row["geom_pair"]):
                raise ValueError("contact trace geom pair invalid")
            force = row["normal_force_N"]
            if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or force < 0:
                raise ValueError("contact trace normal force invalid")
            for field, length in (("position_m", 3), ("wrench_contact_frame_N_Nm", 6)):
                values = row[field]
                if not isinstance(values, list) or len(values) != length or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                    raise ValueError(f"contact trace {field} invalid")
            if not math.isclose(force, abs(row["wrench_contact_frame_N_Nm"][0]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("normal force does not match measured contact-frame force")


def compare_contacts(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare ordered point records and pair sums at every physics step."""
    if reference.get("schema_version") != CONTACT_SCHEMA or candidate.get("schema_version") != CONTACT_SCHEMA:
        raise ValueError("stepwise force comparison requires contact trace v2")
    left, right = reference["steps"], candidate["steps"]
    pairs_identical = len(left) == len(right)
    normal_errors, wrench_errors, position_errors = [], [], []
    pair_sum_errors = []
    point_count = 0
    mismatched_steps = []
    for step_index, (full_step, other_step) in enumerate(zip(left, right, strict=False)):
        full_contacts, other_contacts = full_step["contacts"], other_step["contacts"]
        keys_left = [row["geom_pair"] for row in full_contacts]
        keys_right = [row["geom_pair"] for row in other_contacts]
        if full_step["step_index"] != other_step["step_index"] or full_step["sim_time_s"] != other_step["sim_time_s"] or keys_left != keys_right:
            pairs_identical = False
            mismatched_steps.append(step_index)
            continue
        for full, other in zip(full_contacts, other_contacts, strict=True):
            point_count += 1
            normal_errors.append(abs(full["normal_force_N"]-other["normal_force_N"]))
            wrench_errors.extend(abs(a-b) for a, b in zip(full["wrench_contact_frame_N_Nm"], other["wrench_contact_frame_N_Nm"], strict=True))
            position_errors.extend(abs(a-b) for a, b in zip(full["position_m"], other["position_m"], strict=True))
        def sums(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
            values: dict[tuple[str, str], float] = {}
            for row in rows:
                key = tuple(row["geom_pair"])
                values[key] = values.get(key, 0.)+row["normal_force_N"]
            return values
        sum_left, sum_right = sums(full_contacts), sums(other_contacts)
        pair_sum_errors.extend(abs(sum_left.get(pair, 0.)-sum_right.get(pair, 0.)) for pair in sum_left.keys() | sum_right.keys())
    all_values = [*normal_errors, *wrench_errors, *position_errors, *pair_sum_errors]
    if any(not math.isfinite(value) for value in all_values):
        raise ValueError("contact trace contains nonfinite differences")
    return {"contact_pairs_identical": pairs_identical, "compared_steps": min(len(left), len(right)),
            "compared_contact_points": point_count, "mismatched_steps": mismatched_steps,
            "max_normal_force_difference_N": max(normal_errors, default=0.),
            "max_pair_normal_sum_difference_N": max(pair_sum_errors, default=0.),
            "max_wrench_component_difference": max(wrench_errors, default=0.),
            "max_contact_position_difference_m": max(position_errors, default=0.)}
