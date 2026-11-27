"""Export measured policy rollouts as explicit behavior-supervision examples.

An executed choice is a behavior label, never an oracle or optimality label.
The episode's measured physics outcome is retained as a separate target.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .agent_adapter import RegisteredSkill
from .dataset import write_json_atomic
from .episode_randomization import assign_family_splits
from .session_contracts import finite_json

SAMPLE_SCHEMA = "scene_factory.decision_training_sample.v1"
DATASET_SCHEMA = "scene_factory.decision_dataset.v1"
_OUTCOMES = {"task_success", "task_not_evaluated", "engine_time_limit", "policy_budget_exhausted", "runtime_or_policy_failure"}
_POLICIES = {"contract_test_double", "typesafe_endpoint", "self_hosted_jev", "other_external_policy"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def validate_decision_sample(sample: Any) -> dict[str, Any]:
    """Strict semantic validator used for writes and offline dataset loading."""
    sample = finite_json(sample, "decision sample")
    _exact(sample, {"schema_version", "sample_id", "state", "rubric", "label", "observed_action", "outcome", "source", "split"}, "sample")
    if sample["schema_version"] != SAMPLE_SCHEMA:
        raise ValueError("unsupported decision sample schema")
    _hash(sample["sample_id"], "sample_id")
    state = _exact(sample["state"], {"observation", "task", "step_index", "sim_time_s", "sensor_truth_policy"}, "state")
    if not isinstance(state["observation"], dict) or not isinstance(state["task"], dict):
        raise ValueError("state observation and task must be objects")
    _integer(state["step_index"], "state.step_index")
    if isinstance(state["sim_time_s"], bool) or not isinstance(state["sim_time_s"], (int, float)) or state["sim_time_s"] < 0:
        raise ValueError("state.sim_time_s must be nonnegative")
    if state["sensor_truth_policy"] != "ground_truth_and_noisy_measurements_are_separate":
        raise ValueError("sensor truth policy is missing")
    rubric = _exact(sample["rubric"], {"type", "question_id", "instructions", "criteria"}, "rubric")
    if rubric["type"] != "choice" or rubric["question_id"] != "next_skill":
        raise ValueError("only next_skill choice rubrics are supported")
    _text(rubric["instructions"], "rubric instructions")
    criteria = rubric["criteria"]
    if not isinstance(criteria, dict) or not criteria or len(criteria) > 255:
        raise ValueError("rubric requires 1..255 registered choices")
    for key, description in criteria.items():
        _text(key, "skill id")
        _text(description, "skill description")
    label = _exact(sample["label"], {"kind", "choice", "optimality"}, "label")
    if label["kind"] != "observed_behavior" or label["optimality"] != "not_established" or label["choice"] not in criteria:
        raise ValueError("labels must be an observed behavior within the rubric; oracle claims are unsupported")
    if not isinstance(sample["observed_action"], dict) or not sample["observed_action"]:
        raise ValueError("observed_action must be an executed action")
    outcome = _exact(sample["outcome"], {"episode_outcome", "task_success", "terminated", "truncated", "failure_flags", "authority"}, "outcome")
    if outcome["episode_outcome"] not in _OUTCOMES or type(outcome["terminated"]) is not bool or type(outcome["truncated"]) is not bool:
        raise ValueError("invalid measured episode outcome")
    if outcome["task_success"] is not None and type(outcome["task_success"]) is not bool:
        raise ValueError("task_success must be boolean or null")
    if outcome["authority"] != "recorded_physics_task_evaluator":
        raise ValueError("task labels must name the physics evaluator")
    if not isinstance(outcome["failure_flags"], list) or len(set(outcome["failure_flags"])) != len(outcome["failure_flags"]):
        raise ValueError("failure_flags must be a unique list")
    for flag in outcome["failure_flags"]:
        _text(flag, "failure flag")
    if outcome["episode_outcome"] == "task_success" and (outcome["task_success"] is not True or not outcome["terminated"]):
        raise ValueError("success outcome contradicts recorded task evidence")
    source = _exact(sample["source"], {"rollout_file", "rollout_sha256", "episode_id", "session_id", "scene_sha256", "backend", "fidelity", "seed", "transition_index", "policy_source", "model_resolved", "model_inference_evidence", "selector_state_sha256", "action_sha256"}, "source")
    for key in ("rollout_file", "episode_id", "session_id", "fidelity", "model_resolved"):
        _text(source[key], "source." + key)
    for key in ("rollout_sha256", "scene_sha256", "selector_state_sha256", "action_sha256"):
        _hash(source[key], "source." + key)
    if source["backend"] not in {"mujoco", "isaac"}:
        raise ValueError("training export requires a measured physics backend; dry-run is unsupported")
    _integer(source["seed"], "source.seed")
    _integer(source["transition_index"], "source.transition_index")
    if source["policy_source"] not in _POLICIES or source["model_inference_evidence"] not in {"test_double_no_live_inference", "reported_in_rollout_not_independently_verified"}:
        raise ValueError("invalid model provenance")
    if source["policy_source"] == "contract_test_double" and source["model_inference_evidence"] != "test_double_no_live_inference":
        raise ValueError("test-double decisions cannot claim live inference")
    if "test_double" in source["model_resolved"] and source["policy_source"] != "contract_test_double":
        raise ValueError("resolved test-double model conflicts with source provenance")
    if source["action_sha256"] != _digest(sample["observed_action"]):
        raise ValueError("observed action checksum mismatch")
    if sample["sample_id"] != _digest({"rollout_sha256": source["rollout_sha256"], "transition_index": source["transition_index"]}):
        raise ValueError("sample identity does not match its measured source transition")
    split = _exact(sample["split"], {"name", "asset_families", "task_structure", "policy"}, "split")
    if split["name"] not in {"train", "validation", "test"} or split["policy"] != "connected_components_shared_asset_family_or_task_structure":
        raise ValueError("invalid family split")
    if not isinstance(split["asset_families"], list) or not split["asset_families"] or len(set(split["asset_families"])) != len(split["asset_families"]):
        raise ValueError("asset families must be a nonempty unique list")
    for family in split["asset_families"]:
        _text(family, "asset family")
    _text(split["task_structure"], "task structure")
    return sample


def _observation(value: Any, *, episode: str | None = None, session: str | None = None, scene: str | None = None) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != "scene_factory.session_observation.v1":
        raise ValueError("rollout observation must use the measured session schema")
    if value.get("backend") not in {"mujoco", "isaac"} or not isinstance(value.get("observation"), dict):
        raise ValueError("only actual physics backend observations can be exported")
    for key, expected in (("episode_id", episode), ("session_id", session), ("scene_sha256", scene)):
        _text(value.get(key), key)
        if expected is not None and value[key] != expected:
            raise ValueError(f"rollout {key} changed within one episode")
    _hash(value["scene_sha256"], "scene_sha256")
    _integer(value.get("state_version"), "state_version", 1)
    _integer(value.get("step_index"), "step_index")
    if type(value.get("terminated")) is not bool or type(value.get("truncated")) is not bool:
        raise ValueError("observation must contain termination flags")
    if isinstance(value.get("sim_time"), bool) or not isinstance(value.get("sim_time"), (int, float)) or value["sim_time"] < 0:
        raise ValueError("observation sim_time must be nonnegative")
    _text(value.get("fidelity"), "fidelity")
    return value


def export_decision_dataset(sources: list[dict[str, Any]], output_dir: str | Path, *,
                            skills: list[RegisteredSkill], split_seed: int = 0) -> dict[str, Any]:
    """Export existing measured rollouts, preserving failures and provenance.

    Each source has exactly path/asset_families/task_structure/policy_source.
    The split is computed from shared asset families OR task structures before
    any transition rows are emitted. No simulator or model inference is run.
    """
    if not sources or not skills or len({skill.skill_id for skill in skills}) != len(skills):
        raise ValueError("nonempty rollout sources and unique registered skills are required")
    _integer(split_seed, "split_seed")
    catalog = {skill.skill_id: skill for skill in skills}
    for skill in skills:
        _text(skill.skill_id, "skill id")
        _text(skill.description, "skill description")
        if not isinstance(skill.action, dict) or not skill.action:
            raise ValueError("registered skills require a fixed action")
        finite_json(skill.action)
    loaded, identities = [], set()
    for index, source in enumerate(sources):
        _exact(source, {"path", "asset_families", "task_structure", "policy_source"}, "rollout source")
        path = Path(source["path"]).resolve()
        content = path.read_bytes()
        if len(content) > 256 * 1024 * 1024:
            raise ValueError("rollout exceeds 256 MiB")
        rollout = finite_json(json.loads(content), "rollout")
        if rollout.get("schema_version") != "scene_factory.agent_rollout.v1" or rollout.get("status") not in {"completed", "failed"}:
            raise ValueError("only completed or recorded failed agent rollouts can be exported")
        if rollout.get("outcome") not in _OUTCOMES:
            raise ValueError("rollout outcome must be explicitly classified")
        _integer(rollout.get("seed"), "rollout seed")
        initial = _observation(rollout.get("initial"))
        identity = (initial["session_id"], initial["episode_id"])
        if identity in identities:
            raise ValueError("duplicate measured episode")
        identities.add(identity)
        if source["policy_source"] not in _POLICIES:
            raise ValueError("unknown policy source")
        loaded.append({"source": source, "path": path, "sha256": hashlib.sha256(content).hexdigest(), "rollout": rollout,
                       "split_id": str(index), "episode": initial["episode_id"]})
    splits = assign_family_splits([{ "episode_id": row["split_id"], "asset_families": row["source"]["asset_families"],
                                    "task_structure": row["source"]["task_structure"]} for row in loaded], seed=split_seed)
    split_by_id = {row["episode_id"]: row for row in splits["rows"]}
    samples, unexecuted, episode_counts = [], 0, Counter()
    for row in loaded:
        rollout, source = row["rollout"], row["source"]
        previous = _observation(rollout["initial"])
        transitions, decisions = rollout.get("transitions"), rollout.get("decision_records")
        if not isinstance(transitions, list) or not transitions or not isinstance(decisions, list) or len(decisions) < len(transitions):
            raise ValueError("rollout needs measured transitions paired with recorded decisions")
        unexecuted += len(decisions) - len(transitions)
        final = _observation(rollout.get("final", transitions[-1]["result"]), episode=previous["episode_id"], session=previous["session_id"], scene=previous["scene_sha256"])
        if final != transitions[-1]["result"]:
            raise ValueError("final observation differs from the last committed measured transition")
        task = final.get("info", {}).get("task", {})
        success = task.get("task_success") if task.get("evaluation_status") not in {"unsupported", "not_evaluated"} else None
        flags = [] if rollout["outcome"] == "task_success" else [rollout["outcome"]]
        if rollout["status"] == "failed" and "runtime_or_policy_failure" not in flags:
            flags.append("runtime_or_policy_failure")
        outcome = {"episode_outcome": rollout["outcome"], "task_success": success, "terminated": final["terminated"], "truncated": final["truncated"],
                   "failure_flags": flags, "authority": "recorded_physics_task_evaluator"}
        episode_counts[rollout["outcome"]] += 1
        split_row = split_by_id[row["split_id"]]
        for index, (transition, decision) in enumerate(zip(transitions, decisions, strict=False)):
            result = _observation(transition.get("result"), episode=previous["episode_id"], session=previous["session_id"], scene=previous["scene_sha256"])
            request = transition.get("request", {})
            if (transition.get("sequence_id") != index or request.get("sequence_id") != index
                    or request.get("episode_id") != previous["episode_id"] or transition.get("previous_state_version") != previous["state_version"]
                    or result["state_version"] <= previous["state_version"] or result["step_index"] <= previous["step_index"]
                    or result["sim_time"] <= previous["sim_time"]):
                raise ValueError("rollout transition order or physical time is inconsistent")
            if result["backend"] != previous["backend"] or result["fidelity"] != previous["fidelity"]:
                raise ValueError("backend or fidelity changed within one measured episode")
            if previous["terminated"] or previous["truncated"]:
                raise ValueError("rollout executes an action after episode termination")
            if not isinstance(decision, dict) or decision.get("schema_version") != "scene_factory.jev_skill_decision.v1" or decision.get("status") != "selected":
                raise ValueError("only a recorded selected skill with a measured execution can receive a behavior label")
            selected = decision.get("selected_skill")
            candidates = decision.get("candidate_skills")
            if not isinstance(candidates, list) or not candidates or len(set(candidates)) != len(candidates) or selected not in candidates or any(skill not in catalog for skill in candidates):
                raise ValueError("decision candidates must match the supplied registered skill catalog")
            action = request.get("action")
            if action != decision.get("action") or action != catalog[selected].action:
                raise ValueError("recorded decision/action differs from the executed registered skill")
            sample = {"schema_version": SAMPLE_SCHEMA,
                      "sample_id": _digest({"rollout_sha256": row["sha256"], "transition_index": index}),
                      "state": {"observation": previous["observation"], "task": previous.get("info", {}).get("task", {}), "step_index": previous["step_index"],
                                "sim_time_s": previous["sim_time"], "sensor_truth_policy": "ground_truth_and_noisy_measurements_are_separate"},
                      "rubric": {"type": "choice", "question_id": "next_skill", "instructions": "Predict the registered skill executed by the recorded behavior policy. This is not an optimal-action label.",
                                 "criteria": {skill: catalog[skill].description for skill in candidates}},
                      "label": {"kind": "observed_behavior", "choice": selected, "optimality": "not_established"}, "observed_action": action,
                      "outcome": outcome,
                      "source": {"rollout_file": row["path"].name, "rollout_sha256": row["sha256"], "episode_id": previous["episode_id"], "session_id": previous["session_id"],
                                 "scene_sha256": previous["scene_sha256"], "backend": previous["backend"], "fidelity": previous["fidelity"], "seed": rollout["seed"], "transition_index": index,
                                 "policy_source": source["policy_source"], "model_resolved": decision.get("model_resolved"),
                                 "model_inference_evidence": "test_double_no_live_inference" if source["policy_source"] == "contract_test_double" else "reported_in_rollout_not_independently_verified",
                                 "selector_state_sha256": decision.get("state_sha256"), "action_sha256": _digest(action)},
                      "split": {"name": split_row["split"], "asset_families": split_row["asset_families"], "task_structure": split_row["task_structure"], "policy": splits["policy"]}}
            samples.append(validate_decision_sample(sample))
            previous = result
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("dataset output must be empty; preserve previous evidence")
    content = b"".join(json.dumps(sample, sort_keys=True, ensure_ascii=False, allow_nan=False).encode() + b"\n" for sample in samples)
    (output / "samples.jsonl").write_bytes(content)
    write_json_atomic(output / "family_splits.json", splits)
    manifest = {"schema_version": DATASET_SCHEMA, "sample_schema": SAMPLE_SCHEMA, "sample_count": len(samples), "episode_count": len(loaded),
                "split_sample_counts": dict(Counter(sample["split"]["name"] for sample in samples)), "episode_outcomes": dict(episode_counts),
                "source_rollouts": [{"file": row["path"].name, "sha256": row["sha256"]} for row in loaded],
                "samples_sha256": hashlib.sha256(content).hexdigest(),
                "family_splits_sha256": hashlib.sha256((output / "family_splits.json").read_bytes()).hexdigest(),
                "ignored_unexecuted_decisions": unexecuted,
                "label_kind": "observed_behavior", "optimality": "not_established", "training_performed": False,
                "live_model_inference_performed_by_export": False, "empty_splits": splits["empty_splits"],
                "limitations": ["Task outcomes do not establish the optimal next skill.", "An empty held-out split provides no held-out generalization estimate.",
                                "Model inference provenance is retained from source evidence; export performs no model inference."]}
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def verify_decision_dataset(directory: str | Path) -> dict[str, Any]:
    """Check file checksum, strict row semantics, and cross-row split leakage."""
    directory = Path(directory)
    manifest = finite_json(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
    _exact(manifest, {"schema_version", "sample_schema", "sample_count", "episode_count", "split_sample_counts", "episode_outcomes", "source_rollouts", "samples_sha256",
                      "family_splits_sha256", "ignored_unexecuted_decisions", "label_kind", "optimality", "training_performed", "live_model_inference_performed_by_export", "empty_splits", "limitations"}, "manifest")
    if manifest.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported dataset manifest")
    if (manifest["sample_schema"] != SAMPLE_SCHEMA or manifest["label_kind"] != "observed_behavior" or manifest["optimality"] != "not_established"
            or manifest["training_performed"] is not False or manifest["live_model_inference_performed_by_export"] is not False):
        raise ValueError("manifest makes unsupported training or optimality claims")
    splits_content = (directory / "family_splits.json").read_bytes()
    if hashlib.sha256(splits_content).hexdigest() != manifest["family_splits_sha256"]:
        raise ValueError("family split checksum mismatch")
    content = (directory / "samples.jsonl").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.get("samples_sha256"):
        raise ValueError("dataset sample checksum mismatch")
    samples = [validate_decision_sample(json.loads(line)) for line in content.splitlines() if line.strip()]
    if len(samples) != manifest.get("sample_count") or len({sample["sample_id"] for sample in samples}) != len(samples):
        raise ValueError("sample count or unique identity mismatch")
    identities = {(sample["source"]["session_id"], sample["source"]["episode_id"]) for sample in samples}
    if len(identities) != manifest["episode_count"] or dict(Counter(sample["split"]["name"] for sample in samples)) != manifest["split_sample_counts"]:
        raise ValueError("manifest episode or split counts differ from measured samples")
    source_refs = {(row["file"], row["sha256"]) for row in manifest["source_rollouts"]}
    if source_refs != {(sample["source"]["rollout_file"], sample["source"]["rollout_sha256"]) for sample in samples}:
        raise ValueError("manifest source references differ from measured samples")
    groups: dict[tuple[str, str], set[str]] = {}
    for sample in samples:
        split = sample["split"]
        for key in [("asset", family) for family in split["asset_families"]] + [("task", split["task_structure"])]:
            groups.setdefault(key, set()).add(split["name"])
    if any(len(names) != 1 for names in groups.values()):
        raise ValueError("asset family or task structure leaks across dataset splits")
    return {"valid": True, "sample_count": len(samples), "samples_sha256": manifest["samples_sha256"], "asset_family_leakage": False, "task_structure_leakage": False}
