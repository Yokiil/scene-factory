from __future__ import annotations

import contextlib
import copy
import io
import json

from scene_factory.cli import main
from scene_factory.factory import SceneFactory


def invoke(args):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(args)
    return code, json.loads(output.getvalue())


def scene_input(tmp_path):
    scene = SceneFactory().build_from_recipe("living_room_returned_home", 19).scene.to_dict()
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path, scene


def test_build_verify_archive_and_tampered_verification(tmp_path):
    path, _ = scene_input(tmp_path)
    package = tmp_path / "environment"
    code, result = invoke(["sandbox", "build", str(path), "--output", str(package)])
    assert code == 0 and result["verification"]["valid"]
    code, result = invoke(["sandbox", "verify", str(package)])
    assert code == 0 and result["runtime_checked"]
    archive = tmp_path / "environment.zip"
    code, result = invoke(["sandbox", "archive", str(package), "--output", str(archive)])
    assert code == 0 and archive.is_file()
    (package / "scene/layout.json").write_text("{}")
    code, result = invoke(["sandbox", "verify", str(package)])
    assert code == 2 and not result["valid"]


def test_randomization_writes_source_actual_values_and_scene(tmp_path):
    path, scene = scene_input(tmp_path)
    target = next(item["object_id"] for item in scene["objects"] if item["dynamic"])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"instances": {target: {"scale": [0.95, 1.05]}}}))
    output = tmp_path / "sample"
    code, result = invoke(["sandbox", "randomize", str(path), str(config), "--output", str(output), "--seed", "42"])
    assert code == 0 and result["physics_validity"] == "not_verified"
    sampled = json.loads((output / "layout.json").read_text(encoding="utf-8"))
    assert target in sampled["randomization"]["sampled_values"]
    many = tmp_path / "many"
    code, result = invoke(["sandbox", "randomize", str(path), str(config), "--output", str(many), "--count", "100"])
    assert code == 0 and result["candidate_count"] == 100
    report = json.loads((many / "sampling.json").read_text(encoding="utf-8"))
    assert len(report["rows"]) == 100
    assert report["counts"]["task_success"] is None


def test_dependency_diff_and_input_overwrite_guard(tmp_path):
    path, scene = scene_input(tmp_path)
    changed = copy.deepcopy(scene)
    target = next(item for item in changed["objects"] if item["dynamic"])
    target["physics"] = {"mass_kg": 1.0}
    after = tmp_path / "after.json"
    after.write_text(json.dumps(changed))
    code, result = invoke(["sandbox", "dependency-diff", str(path), str(after)])
    assert code == 0 and result["full_dynamics_revalidation"]
    assert "artifact:visual" in result["reusable_nodes"]
    original = path.read_bytes()
    with contextlib.redirect_stderr(io.StringIO()):
        assert main(["sandbox", "dependency-diff", str(path), str(after), "--output", str(path)]) == 1
    assert path.read_bytes() == original
