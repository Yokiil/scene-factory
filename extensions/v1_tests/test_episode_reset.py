from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scene_factory.agent import DryRunBackend, SceneFactoryEnv
from scene_factory.episode_reset import prepare_episode_reset


def source_scene():
    return {
        "scene_id": "reset_fixture", "seed": 42,
        "task": {"instruction": "Move the cup"},
        "objects": [
            {"object_id": "table", "dynamic": False,
             "pose": {"position": [0.0, 0.0, 0.4], "yaw_deg": 0.0}},
            {"object_id": "cup", "dynamic": True, "support": "table:top",
             "pose": {"position": [0.1, 0.2, 0.85], "yaw_deg": 5.0}},
            {"object_id": "other", "dynamic": True, "support": "table:top",
             "pose": {"position": [0.3, 0.4, 0.85], "yaw_deg": 10.0}},
        ],
    }


JITTER = {"pose_jitter": {"cup": {"x_m": [-0.01, 0.01], "yaw_deg": [-2, 2]}}}


class CountingBackend(DryRunBackend):
    def __init__(self):
        super().__init__()
        self.reset_count = 0
        self.fail = False

    def reset(self, scene):
        self.reset_count += 1
        if self.fail:
            self.scene = scene
            raise RuntimeError("initialization failed")
        return super().reset(scene)


class EpisodeResetTests(unittest.TestCase):
    def test_seed_reproduces_draws_and_final_scene_without_mutating_inputs(self):
        original, options = source_scene(), copy.deepcopy(JITTER)
        expected = copy.deepcopy(original)
        first, first_record = prepare_episode_reset(original, seed=7, options=options)
        second, second_record = prepare_episode_reset(original, seed=7, options=options)
        different, _ = prepare_episode_reset(original, seed=8, options=options)
        self.assertEqual(first, second)
        self.assertEqual(first_record, second_record)
        self.assertNotEqual(first, different)
        self.assertEqual(original, expected)
        self.assertEqual(options, JITTER)
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["objects"][1]["pose"]["position"][2], 0.85)
        offsets = first_record["sampled_values"]["cup"]["sampled_offsets"]
        self.assertAlmostEqual(first["objects"][1]["pose"]["position"][0], 0.1 + offsets["x_m"])
        self.assertTrue(first_record["randomization_applied"])

    def test_independent_streams_survive_reordering_and_added_fields(self):
        _, initial = prepare_episode_reset(source_scene(), seed=9, options=JITTER)
        options = copy.deepcopy(JITTER)
        options["pose_jitter"]["other"] = {"x_m": [-1, 1]}
        options["pose_jitter"]["cup"]["y_m"] = [-0.05, 0.05]
        scene = source_scene()
        scene["objects"].reverse()
        _, expanded = prepare_episode_reset(scene, seed=9, options=options)
        for field in ("x_m", "yaw_deg"):
            self.assertEqual(initial["sampled_values"]["cup"]["sampled_offsets"][field],
                             expanded["sampled_values"]["cup"]["sampled_offsets"][field])

    def test_absolute_overrides_then_jitter_and_implicit_seed_replay(self):
        options = {
            "pose_overrides": {"cup": {"position_xy_m": [0.5, -0.2], "yaw_deg": 90}},
            "pose_jitter": {"cup": {"x_m": [0.02, 0.02], "yaw_deg": [1, 1]}},
        }
        scene, record = prepare_episode_reset(source_scene(), options=options)
        self.assertEqual(scene["objects"][1]["pose"],
                         {"position": [0.52, -0.2, 0.85], "yaw_deg": 91.0})
        replay, replay_record = prepare_episode_reset(source_scene(), seed=record["seed"], options=options)
        self.assertEqual(scene, replay)
        self.assertEqual(record["episode_scene_sha256"], replay_record["episode_scene_sha256"])
        self.assertEqual(record["seed_source"], "system_entropy")

    def test_seed_alone_preserves_scene_and_reports_no_randomization(self):
        scene, record = prepare_episode_reset(source_scene(), seed=3)
        self.assertEqual(scene, source_scene())
        self.assertFalse(record["randomization_applied"])
        self.assertFalse(record["scene_modified"])
        self.assertEqual(record["sampled_values"], {})
        self.assertEqual(record["source_scene_sha256"], record["episode_scene_sha256"])

    def test_rejects_bad_seed_unknown_options_objects_ranges_and_non_leaf_objects(self):
        for seed in (True, -1, 2**64, 1.5, "1"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                prepare_episode_reset(source_scene(), seed=seed)
        cases = [
            {"physics": {}}, {"pose_jitter": {"missing": {"x_m": [0, 1]}}},
            {"pose_jitter": {"cup": {"x_m": [1, -1]}}},
            {"pose_jitter": {"cup": {"x_m": [0, float("nan")]}}},
            {"pose_jitter": {"cup": {"x_m": [False, 1]}}},
            {"pose_jitter": {"cup": {"z_m": [0, 1]}}},
            {"pose_overrides": {"cup": {"position_xy_m": [1, 2, 3]}}},
            {"pose_overrides": {"table": {"yaw_deg": 5}}},
            {"pose_jitter": {"cup": {}}},
        ]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_episode_reset(source_scene(), options=options)
        scene = source_scene()
        scene["objects"][2]["support"] = "cup:top"
        with self.assertRaisesRegex(ValueError, "dynamic leaf"):
            prepare_episode_reset(scene, seed=4, options=JITTER)

    def test_invalid_options_leave_active_episode_and_backend_failure_invalidates_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text(json.dumps(source_scene()), encoding="utf-8")
            backend = CountingBackend()
            env = SceneFactoryEnv(path, backend)
            _, info = env.reset(seed=1)
            digest = env.scene_sha256
            self.assertEqual(digest, info["episode_reset"]["episode_scene_sha256"])
            with self.assertRaises(ValueError):
                env.reset(options={"unrecognized": 4})
            self.assertEqual(backend.reset_count, 1)
            self.assertEqual(env.scene_sha256, digest)
            env.step(None)
            backend.fail = True
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                env.reset(seed=2)
            self.assertIsNone(env.scene_sha256)
            self.assertIsNone(backend.scene)
            with self.assertRaises(RuntimeError):
                env.step(None)
            with self.assertRaises(RuntimeError):
                env.render()
            backend.fail = False
            env.reset(seed=2)
            env.step(None)
            env.close()

    def test_repeated_reset_does_not_accumulate_jitter_or_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            source = json.dumps(source_scene())
            path.write_text(source, encoding="utf-8")
            env = SceneFactoryEnv(path, DryRunBackend())
            first, first_info = env.reset(seed=30, options=JITTER)
            expected = copy.deepcopy(first)
            first["scene_graph"][1]["pose"]["position"][0] = 999
            env.step(None)
            second, second_info = env.reset(seed=30, options=JITTER)
            self.assertEqual(second, expected)
            self.assertEqual(first_info, second_info)
            self.assertEqual(path.read_text(encoding="utf-8"), source)
            env.close()

    def test_legacy_backend_accepts_no_options_but_rejects_pose_changes(self):
        class LegacyBackend(CountingBackend):
            supports_scene_reset_options = False

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text(json.dumps(source_scene()), encoding="utf-8")
            backend = LegacyBackend()
            env = SceneFactoryEnv(path, backend)
            env.reset(seed=5)
            with self.assertRaisesRegex(ValueError, "does not support reset pose changes"):
                env.reset(seed=5, options=JITTER)
            self.assertEqual(backend.reset_count, 1)
            env.close()

    def test_isaac_refuses_pose_changes_before_loading_simulator(self):
        from scene_factory.backends.isaac import IsaacSimBackend

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text(json.dumps(source_scene()), encoding="utf-8")
            backend = IsaacSimBackend(Path(directory) / "not-exported.usd")
            env = SceneFactoryEnv(path, backend)
            with self.assertRaisesRegex(ValueError, "does not support reset pose changes"):
                env.reset(seed=5, options=JITTER)
            self.assertIsNone(backend.scene)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo extra is not installed")
    def test_mujoco_initial_state_reproduces_after_simulation_steps(self):
        from scene_factory.backends.mujoco import MujocoBackend
        from scene_factory.factory import SceneFactory

        factory = SceneFactory()
        result = factory.build_from_recipe("kitchen_after_cooking", 42)
        with tempfile.TemporaryDirectory() as directory:
            files = factory.write_result(result, directory, export_mjcf=False)
            scene = json.loads(Path(files["layout"]).read_text(encoding="utf-8"))
            supported_ids = {(item.get("support") or "").partition(":")[0] for item in scene["objects"]}
            target = next(item["object_id"] for item in scene["objects"]
                          if item["dynamic"] and item["object_id"] not in supported_ids)
            options = {"pose_jitter": {target: {"x_m": [-0.001, 0.001]}}}
            env = SceneFactoryEnv(files["layout"], MujocoBackend(registry=factory.registry))
            try:
                first, first_info = env.reset(seed=16, options=options)
                for _ in range(5):
                    env.step(None)
                second, second_info = env.reset(seed=16, options=options)
                self.assertEqual(first["proprioception"], second["proprioception"])
                self.assertEqual(first["object_poses"], second["object_poses"])
                self.assertEqual(first_info["episode_reset"], second_info["episode_reset"])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
