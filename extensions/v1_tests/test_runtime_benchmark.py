from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scene_factory.agent import DryRunBackend
from scene_factory.runtime_benchmark import (
    main,
    run_benchmark,
    summarize_samples,
)


@pytest.fixture
def scene_path(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"scene_id": "bench", "objects": [], "seed": 12}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_gpu_probe():
    with patch("scene_factory.runtime_benchmark._gpu_inventory", return_value={"status": "test", "devices": []}):
        yield


def small_benchmark(scene_path, **kwargs):
    return run_benchmark(scene_path, creation_trials=2, reset_trials=3, step_trials=4, **kwargs)


def test_percentiles_exclude_failures_but_keep_attempts():
    rows = [{"status": "ok", "elapsed_s": value} for value in (1, 2, 3, 4)]
    rows.append({"status": "failed", "elapsed_s": 1000})
    summary = summarize_samples(rows)
    assert summary["p50_s"] == 2.5
    assert summary["p95_s"] == pytest.approx(3.85)
    assert summary["sample_count"] == 4
    assert summary["failure_count"] == 1
    assert summary["attempt_count"] == 5
    assert summary["samples"] == rows
    assert summarize_samples([])["p95_s"] is None


def test_baseline_records_actual_sample_count_and_roundtrips(scene_path, tmp_path):
    output = tmp_path / "benchmark.json"
    report = small_benchmark(scene_path, output=output)
    assert report["status"] == "completed"
    assert report["physics_acceptance"] == "not_evaluated"
    assert report["stages"]["in_process_creation"]["sample_count"] == 2
    assert report["stages"]["reset"]["sample_count"] == 3
    assert report["stages"]["step"]["sample_count"] == 4
    assert report["stages"]["close"]["sample_count"] == 2
    assert report["stages"]["render"]["sample_count"] == 0
    assert report["stages"]["output_write"]["sample_count"] == 1
    assert report["runtime"]["fidelity"] == "no_physics"
    assert report["throughput"]["physics_steps_per_s"] is None
    assert "not a fresh process" in report["unmeasured"]["process_cold_start"]
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert len(report["inputs"]["scene_sha256"]) == 64
    assert len(report["inputs"]["registry_sha256"]) == 64


def test_backend_reset_failure_closes_partial_instances_and_preserves_report(scene_path, tmp_path):
    instances = []
    class BrokenBackend(DryRunBackend):
        closed = False
        def reset(self, scene):
            raise RuntimeError("engine load failed")
        def close(self):
            self.closed = True
    def factory():
        instance = BrokenBackend()
        instances.append(instance)
        return instance
    report = small_benchmark(scene_path, output=tmp_path / "failed.json", backend_factory=factory)
    assert report["status"] == "failed"
    assert len(instances) == 2
    assert all(instance.closed for instance in instances)
    assert report["stages"]["first_reset"]["failure_count"] == 2
    assert report["stages"]["step"]["sample_count"] == 0
    assert json.loads((tmp_path / "failed.json").read_text())["status"] == "failed"


def test_terminal_reset_occurs_before_next_step(scene_path):
    class TerminalBackend(DryRunBackend):
        def step(self, action):
            if self.steps:
                raise RuntimeError("stepped after terminal")
            return super().step(action)
    report = small_benchmark(scene_path, backend_factory=lambda: TerminalBackend(max_steps=1))
    assert report["status"] == "completed"
    assert report["stages"]["step"]["sample_count"] == 4
    assert report["stages"]["terminal_reset"]["sample_count"] == 3
    assert report["outcomes"]["truncated"] == 4


def test_none_render_is_failure_not_image_measurement(scene_path):
    report = small_benchmark(scene_path, render=True)
    assert report["status"] == "failed"
    assert report["stages"]["render"]["sample_count"] == 0
    assert report["stages"]["render"]["failure_count"] == 1
    assert "unsupported" in report["stages"]["render"]["samples"][0]["error"]


def test_invalid_step_is_not_included_in_throughput(scene_path):
    class InvalidStep(DryRunBackend):
        def step(self, action):
            return {}, float("nan"), False, False, {}
    report = small_benchmark(scene_path, backend_factory=InvalidStep)
    assert report["status"] == "failed"
    assert report["stages"]["step"]["failure_count"] == 1
    assert report["throughput"]["environment_steps_per_s"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("stage", ["reset", "step"])
@pytest.mark.parametrize("field", ["observation", "info"])
def test_nonfinite_nested_state_fails_benchmark(scene_path, stage, field):
    class NonfiniteState(DryRunBackend):
        def reset(self, scene):
            observation, info = super().reset(scene)
            if stage == "reset":
                target = observation if field == "observation" else info
                target["nested"] = {"qpos": [0.0, float("nan")]}
            return observation, info
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if stage == "step":
                target = observation if field == "observation" else info
                target["nested"] = {"qvel": [float("inf")]}
            return observation, reward, terminated, truncated, info
    report = small_benchmark(scene_path, backend_factory=NonfiniteState)
    assert report["status"] == "failed"
    assert report["stages"]["first_reset" if stage == "reset" else "step"]["failure_count"] > 0
    assert report["stages"]["step"]["sample_count"] == 0
    assert report["throughput"]["environment_steps_per_s"] is None
    assert any("finite numbers" in failure["error"] for failure in report["failures"])
    json.dumps(report, allow_nan=False)


def test_close_failure_is_retained(scene_path):
    class BrokenClose(DryRunBackend):
        def close(self):
            raise RuntimeError("cleanup failed")
    report = small_benchmark(scene_path, backend_factory=BrokenClose)
    assert report["status"] == "failed"
    assert report["stages"]["close"]["failure_count"] == 2


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"step_trials": 0}, {"reset_trials": -1}, {"creation_trials": True}])
def test_invalid_configuration_does_not_start_backend(scene_path, kwargs):
    with patch("scene_factory.runtime_benchmark.create_benchmark_backend") as create:
        with pytest.raises(ValueError):
            run_benchmark(scene_path, **kwargs)
        create.assert_not_called()


def test_output_cannot_overwrite_scene(scene_path):
    original = scene_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        small_benchmark(scene_path, output=scene_path)
    assert scene_path.read_bytes() == original


def test_write_failure_returns_failed_report(scene_path, tmp_path):
    with patch("scene_factory.runtime_benchmark.write_json_atomic", side_effect=OSError("disk full")):
        report = small_benchmark(scene_path, output=tmp_path / "report.json")
    assert report["status"] == "failed"
    assert report["stages"]["output_write"]["failure_count"] == 1
    assert report["failures"][-1]["error"] == "disk full"


def test_finalization_failure_does_not_leave_completed_report(scene_path, tmp_path):
    from scene_factory.dataset import write_json_atomic
    output = tmp_path / "report.json"
    calls = 0
    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("final write failed")
        write_json_atomic(path, payload)
    with patch("scene_factory.runtime_benchmark.write_json_atomic", side_effect=fail_second):
        report = small_benchmark(scene_path, output=output)
    assert report["status"] == "failed"
    assert report["failures"][-1]["stage"] == "output_write_finalize"
    assert json.loads(output.read_text())["status"] == "writing_report"


def test_cli_failure_is_nonzero_and_has_report(scene_path, tmp_path, capsys):
    output = tmp_path / "failed.json"
    code = main([str(scene_path), "--output", str(output), "--creation-trials", "1",
                 "--reset-trials", "1", "--step-trials", "1", "--render"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert json.loads(output.read_text())["status"] == "failed"


def test_input_changes_are_reported_as_failure(scene_path):
    class MutatingBackend(DryRunBackend):
        def step(self, action):
            scene_path.write_text(scene_path.read_text() + " ", encoding="utf-8")
            return super().step(action)
    report = small_benchmark(scene_path, backend_factory=MutatingBackend)
    assert report["status"] == "failed"
    assert any(item["stage"] == "input_verification" for item in report["failures"])


def test_selected_mujoco_external_geometry_is_hashed_and_changes_fail(tmp_path):
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({"asset_id": "part", "category": "cup", "bbox_m": [1, 1, 1],
                                    "tags": ["kitchen_dining_pack"]}), encoding="utf-8")
    folder = tmp_path / "source" / "part"
    folder.mkdir(parents=True)
    manifest = folder / "SOURCE.json"
    geometry = folder / "geometry.json"
    manifest.write_text("{}", encoding="utf-8")
    geometry.write_text("{}", encoding="utf-8")
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps({"scene_id": "bench", "seed": 1, "objects": [
        {"object_id": "part_1", "asset_id": "part", "dynamic": True,
         "pose": {"position": [0, 0, 1], "yaw_deg": 0}},
    ]}), encoding="utf-8")
    class MutatingGeometry(DryRunBackend):
        def step(self, action):
            geometry.write_text('{"changed":true}', encoding="utf-8")
            return super().step(action)
    report = small_benchmark(scene_path, backend="mujoco", registry_path=registry,
                             backend_factory=MutatingGeometry)
    dependencies = report["inputs"]["mujoco_geometry_dependencies"]
    assert dependencies["status"] == "complete_for_current_mjcf_exporter"
    assert len(dependencies["files"]) == 2
    assert report["status"] == "failed"
    assert any("geometry dependency changed" in item["error"] for item in report["failures"])


def test_real_mujoco_state_baseline(tmp_path):
    pytest.importorskip("mujoco")
    from scene_factory.factory import SceneFactory
    with patch.dict("os.environ", {"SCENE_FACTORY_LLM_MODE": "off"}):
        factory = SceneFactory()
        built = factory.build_from_recipe("living_room_returned_home", 19)
        files = factory.write_result(built, tmp_path / "scene", export_mjcf=True)
    report = run_benchmark(files["layout"], backend="mujoco", creation_trials=1,
                           reset_trials=2, step_trials=3)
    assert report["status"] == "completed", report["failures"]
    assert report["runtime"]["engine_version"]
    assert report["runtime"]["physics_dt_s"] == pytest.approx(0.002)
    assert report["runtime"]["action_repeat"] == 5
    assert report["runtime"]["geom_count"] > 0
    assert report["peak_observed_contact_count"] is not None
    assert report["throughput"]["physics_steps_per_s"] == 5 * report["throughput"]["environment_steps_per_s"]
