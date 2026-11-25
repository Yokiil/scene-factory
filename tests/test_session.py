from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

from scene_factory.agent import DryRunBackend, SceneFactoryEnv
from scene_factory.session import RuntimeSession


class CountingBackend(DryRunBackend):
    def __init__(self, max_steps=100):
        super().__init__(max_steps=max_steps)
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.fail_reset = False
        self.fail_step = False

    def reset(self, scene):
        self.reset_calls += 1
        if self.fail_reset:
            self.scene = scene
            raise RuntimeError("backend reset failed after mutation")
        return super().reset(scene)

    def step(self, action):
        self.step_calls += 1
        if self.fail_step:
            self.steps += 1
            raise RuntimeError("backend step failed after mutation")
        return super().step(action)

    def close(self):
        self.close_calls += 1
        super().close()


class RuntimeSessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.scene_path = Path(directory.name) / "layout.json"
        self.scene_path.write_text(json.dumps({
            "scene_id": "session_fixture", "seed": 42,
            "task": {"instruction": "Move the cup"},
            "objects": [{
                "object_id": "cup", "dynamic": True,
                "pose": {"position": [0.1, 0.2, 0.85], "yaw_deg": 5.0},
            }],
        }), encoding="utf-8")

    def session(self, *, max_steps=100, history_limit=4096):
        backend = CountingBackend(max_steps=max_steps)
        session = RuntimeSession(SceneFactoryEnv(self.scene_path, backend), history_limit=history_limit)
        self.addCleanup(session.close)
        return session, backend

    @staticmethod
    def reset_request(key="reset-0", seed=1, options=None):
        return {"idempotency_key": key, "seed": seed, "options": options or {}}

    @staticmethod
    def step_request(session, *, key="step-0", sequence=0, steps=1):
        return {
            "episode_id": session.episode_id, "idempotency_key": key,
            "sequence_id": sequence, "steps": steps,
            "action": {"mode": "noop", "payload": {}},
        }

    def test_reset_retry_is_exactly_once_and_responses_are_detached(self):
        session, backend = self.session()
        request = self.reset_request()
        first = session.reset(request)
        expected = copy.deepcopy(first)
        first["observation"]["scene_graph"][0]["pose"]["position"][0] = 999
        capabilities = session.capabilities
        capabilities["action_modes"].append("torque")
        retry = session.reset(request)
        self.assertEqual(retry, expected)
        self.assertEqual(session.observe(), expected)
        self.assertNotIn("torque", session.capabilities["action_modes"])
        self.assertEqual(backend.reset_calls, 1)
        with self.assertRaisesRegex(ValueError, "different reset"):
            session.reset(self.reset_request(seed=2))

    def test_initial_reset_validation_does_not_fault_session_or_consume_key(self):
        session, backend = self.session(history_limit=1)
        with self.assertRaises(ValueError):
            session.reset(self.reset_request(options={"unknown": True}))
        self.assertFalse(session.describe()["faulted"])
        self.assertEqual(backend.reset_calls, 0)
        self.assertEqual(session.env.reset_attempts, 0)
        session.reset(self.reset_request())
        self.assertEqual(backend.reset_calls, 1)
        self.assertEqual(session.env.reset_attempts, 1)

    def test_concurrent_duplicate_step_executes_once_and_cached_response_is_detached(self):
        session, backend = self.session()
        session.reset(self.reset_request())
        request = self.step_request(session, steps=3)
        barrier = threading.Barrier(6)

        def invoke():
            barrier.wait(timeout=5)
            return session.step(request)

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(lambda _: invoke(), range(6)))
        self.assertEqual(backend.step_calls, 3)
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(results[0]["step_index"], 3)
        self.assertEqual(session.describe()["next_sequence_id"], 1)
        expected = copy.deepcopy(results[0])
        results[0]["info"]["executed_steps"] = 999
        self.assertEqual(session.step(request), expected)
        self.assertEqual(session.observe(), expected)

    def test_sequence_conflicting_keys_and_old_episodes_never_reexecute(self):
        session, backend = self.session()
        session.reset(self.reset_request())
        request = self.step_request(session)
        session.step(request)
        with self.assertRaisesRegex(ValueError, "different step"):
            session.step({**request, "steps": 2})
        with self.assertRaisesRegex(ValueError, "expected sequence_id 1"):
            session.step(self.step_request(session, key="out-of-order", sequence=2))
        self.assertEqual(backend.step_calls, 1)
        session.reset(self.reset_request(key="reset-1"))
        with self.assertRaisesRegex(ValueError, "stale or unknown episode"):
            session.step(request)
        with self.assertRaisesRegex(ValueError, "stale episode"):
            session.reset(self.reset_request())
        session.step(self.step_request(session))
        self.assertEqual(backend.step_calls, 2)

    def test_repeat_stops_at_terminal_and_duplicate_terminal_request_is_replayed(self):
        session, backend = self.session(max_steps=2)
        session.reset(self.reset_request())
        request = self.step_request(session, steps=10)
        response = session.step(request)
        self.assertTrue(response["truncated"])
        self.assertEqual(response["info"]["executed_steps"], 2)
        self.assertEqual(response["info"]["requested_steps"], 10)
        self.assertEqual(response["step_index"], 2)
        self.assertEqual(session.step(request), response)
        with self.assertRaisesRegex(RuntimeError, "episode ended"):
            session.step(self.step_request(session, key="next", sequence=1))
        self.assertEqual(backend.step_calls, 2)

    def test_failed_step_is_never_retried_and_new_reset_recovers(self):
        session, backend = self.session()
        session.reset(self.reset_request())
        request = self.step_request(session, steps=3)
        backend.fail_step = True
        with self.assertRaisesRegex(RuntimeError, "backend step failed"):
            session.step(request)
        self.assertTrue(session.describe()["faulted"])
        with self.assertRaisesRegex(RuntimeError, "reset is required"):
            session.step(request)
        with self.assertRaisesRegex(RuntimeError, "reset is required"):
            session.observe()
        with self.assertRaisesRegex(RuntimeError, "new idempotency_key"):
            session.reset(self.reset_request())
        self.assertEqual(backend.step_calls, 1)
        self.assertEqual(backend.reset_calls, 1)
        backend.fail_step = False
        session.reset(self.reset_request(key="recover"))
        session.step(self.step_request(session))
        self.assertEqual(backend.step_calls, 2)
        self.assertFalse(session.describe()["faulted"])

    def test_invalid_reset_preserves_active_episode_and_can_be_corrected_with_same_key(self):
        session, backend = self.session()
        session.reset(self.reset_request())
        session.step(self.step_request(session))
        expected = session.observe()
        with self.assertRaisesRegex(ValueError, "unsupported reset options"):
            session.reset(self.reset_request(key="bad-option", options={"unknown": 1}))
        self.assertEqual(session.observe(), expected)
        self.assertFalse(session.describe()["faulted"])
        self.assertEqual(backend.reset_calls, 1)
        session.step(self.step_request(session, key="still-active", sequence=1))
        session.reset(self.reset_request(key="bad-option"))
        self.assertEqual(backend.reset_calls, 2)

    def test_failed_backend_reset_blocks_old_state_and_cannot_be_reexecuted(self):
        session, backend = self.session()
        session.reset(self.reset_request())
        backend.fail_reset = True
        request = self.reset_request(key="failed")
        with self.assertRaisesRegex(RuntimeError, "backend reset failed"):
            session.reset(request)
        self.assertIsNone(session.env.scene_sha256)
        self.assertTrue(session.describe()["faulted"])
        with self.assertRaisesRegex(RuntimeError, "previous reset failed"):
            session.reset(request)
        with self.assertRaisesRegex(RuntimeError, "reset is required"):
            session.observe()
        self.assertEqual(backend.reset_calls, 2)
        backend.fail_reset = False
        session.reset(self.reset_request(key="recovered"))
        self.assertFalse(session.describe()["faulted"])

    def test_unsupported_action_and_nonfinite_payload_do_not_consume_sequence(self):
        session, backend = self.session()
        session.reset(self.reset_request())
        request = self.step_request(session)
        with self.assertRaisesRegex(ValueError, "not supported"):
            session.step({**request, "action": {"mode": "joint_position", "payload": {}}})
        with self.assertRaisesRegex(ValueError, "finite"):
            session.step({**request, "action": {"mode": "dry_run", "payload": {"v": float("inf")}}})
        self.assertEqual(backend.step_calls, 0)
        self.assertEqual(session.describe()["next_sequence_id"], 0)
        session.step(request)
        self.assertEqual(backend.step_calls, 1)

    def test_bounded_history_keeps_old_retry_receipt_and_rejects_new_mutations(self):
        session, backend = self.session(history_limit=1)
        session.reset(self.reset_request())
        request = self.step_request(session)
        response = session.step(request)
        with self.assertRaisesRegex(RuntimeError, "step history limit"):
            session.step(self.step_request(session, key="next", sequence=1))
        with self.assertRaisesRegex(RuntimeError, "reset history limit"):
            session.reset(self.reset_request(key="next-reset"))
        self.assertEqual(session.step(request), response)
        self.assertEqual(backend.step_calls, 1)
        self.assertEqual(backend.reset_calls, 1)

    def test_lifecycle_requires_reset_and_close_is_idempotent(self):
        session, backend = self.session()
        with self.assertRaisesRegex(RuntimeError, "reset must be called first"):
            session.observe()
        session.reset(self.reset_request())
        session.close()
        session.close()
        self.assertEqual(backend.close_calls, 1)
        self.assertIsNone(session.env.scene_sha256)
        with self.assertRaisesRegex(RuntimeError, "session is closed"):
            session.reset(self.reset_request())
        with self.assertRaisesRegex(RuntimeError, "session is closed"):
            session.observe()

    def test_cli_smoke_runs_session_and_writes_same_report_as_stdout(self):
        from scene_factory.cli import main

        output = self.scene_path.parent / "smoke.json"
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = main([
                "sandbox", "smoke", str(self.scene_path), "--backend", "dry-run",
                "--steps", "3", "--output", str(output),
            ])
        self.assertEqual(result, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report, json.loads(captured.getvalue()))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["scope"], "session_protocol_smoke_not_physics_acceptance")
        self.assertTrue(report["retry_response_identical"])
        self.assertEqual(report["final"]["step_index"], 3)
        self.assertEqual(report["initial"]["episode_id"], report["final"]["episode_id"])

    def test_cli_invalid_json_preserves_input_and_writes_failure_evidence(self):
        from scene_factory.cli import main

        options = self.scene_path.parent / "invalid.json"
        options.write_text("{invalid", encoding="utf-8")
        output = self.scene_path.parent / "failed-smoke.json"
        with redirect_stdout(io.StringIO()):
            result = main([
                "sandbox", "smoke", str(self.scene_path), "--reset-options", str(options),
                "--output", str(output),
            ])
        self.assertEqual(result, 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "JSONDecodeError")
        self.assertEqual(options.read_text(encoding="utf-8"), "{invalid")

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo extra is not installed")
    def test_real_mujoco_rejects_all_invalid_force_targets_before_mutation(self):
        import numpy as np

        from scene_factory.backends.mujoco import MujocoBackend
        from scene_factory.factory import SceneFactory

        factory = SceneFactory()
        result = factory.build_from_recipe("kitchen_after_cooking", 42)
        files = factory.write_result(result, self.scene_path.parent / "physical", export_mjcf=False)
        backend = MujocoBackend(registry=factory.registry)
        session = RuntimeSession(SceneFactoryEnv(files["layout"], backend))
        self.addCleanup(session.close)
        session.reset(self.reset_request())
        dynamic_id = next(item["object_id"] for item in backend.scene["objects"] if item["dynamic"])
        static_id = next(item["object_id"] for item in backend.scene["objects"] if not item["dynamic"])
        request = self.step_request(session)
        request["action"] = {"mode": "object_forces", "payload": {"object_forces": {
            dynamic_id: [0.01, 0, 0],
        }}}
        response = session.step(request)
        expected_qpos = backend._data.qpos.copy()
        expected_force = backend._data.xfrc_applied.copy()
        self.assertAlmostEqual(response["sim_time"], backend._model.opt.timestep * backend.frame_skip)
        for invalid_id in (static_id, "missing-object"):
            invalid = self.step_request(session, key="invalid", sequence=1)
            invalid["action"] = {"mode": "object_forces", "payload": {"object_forces": {
                dynamic_id: [123, 0, 0], invalid_id: [1, 0, 0],
            }}}
            with self.subTest(target=invalid_id), self.assertRaisesRegex(ValueError, "dynamic scene object"):
                session.step(invalid)
            np.testing.assert_array_equal(backend._data.qpos, expected_qpos)
            np.testing.assert_array_equal(backend._data.xfrc_applied, expected_force)
            self.assertEqual(backend.steps, 1)
            self.assertEqual(session.describe()["next_sequence_id"], 1)
            self.assertEqual(session.observe(), response)


if __name__ == "__main__":
    unittest.main()
