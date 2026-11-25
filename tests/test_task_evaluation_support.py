from __future__ import annotations

import pytest

from scene_factory.tasks import TaskEvaluator


@pytest.mark.parametrize("state", [{}, {"peg": (0, 0, 0), "socket": (0, 0, 0)}])
def test_assembly_is_explicitly_unsupported_by_pose_only_evaluator(state):
    evaluator = TaskEvaluator(
        {"success": {"predicate": "assembly", "subject": "peg", "target": "socket"}}, state
    )
    # Even a claimed success and apparently perfect final pose cannot stand in
    # for the dedicated trajectory/contact/release acceptance procedure.
    result = evaluator.status(state, {"task_success": True, "inserted": True})
    assert result["task_success"] is False
    assert result["evaluation_status"] == "unsupported"
    assert result["failure_reason"] == "assembly_requires_trajectory_contact_and_release_evidence"
    assert evaluator.evaluate(state) is False


def test_unknown_predicate_has_consistent_status_with_or_without_referenced_objects():
    task = {"success": {"predicate": "future_task", "subject": "a", "target": "b"}}
    empty = TaskEvaluator(task, {}).status({})
    populated = TaskEvaluator(task, {"a": (0, 0, 0), "b": (0, 0, 0)}).status(
        {"a": (0, 0, 0), "b": (0, 0, 0)}
    )
    assert empty == populated
    assert empty["evaluation_status"] == "unsupported"
    assert empty["task_success"] is False


def test_missing_success_predicate_is_not_an_evaluated_task_failure():
    result = TaskEvaluator({}, {}).status({})
    assert result["task_success"] is False
    assert result["evaluation_status"] == "not_evaluated"


def test_existing_supported_predicates_still_evaluate():
    task = {"success": {"predicate": "near", "subject": "a", "target": "b", "max_distance_m": 0.1}}
    initial = {"a": (0, 0, 0), "b": (1, 0, 0)}
    evaluator = TaskEvaluator(task, initial)
    assert evaluator.evaluate(initial) is False
    assert evaluator.evaluate({"a": (0, 0, 0), "b": (0.01, 0, 0)}) is True
