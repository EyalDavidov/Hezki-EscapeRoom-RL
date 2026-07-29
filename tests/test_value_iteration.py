from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room1_artifact, import_room1_artifact
from escape_room_rl.evaluation import evaluate_policy
from escape_room_rl.room1 import Room1Environment, default_room1_config, generate_random_slippery_cells
from escape_room_rl.value_iteration import (
    ValueIterationConfig,
    ValueIterationResult,
    run_value_iteration,
)


class ValueIterationTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        ValueIterationConfig(gamma=0.95, theta=1e-6, max_iterations=500)

        with self.assertRaises(ValueError):
            ValueIterationConfig(gamma=1.5)

        with self.assertRaises(ValueError):
            ValueIterationConfig(theta=-1e-6)

        with self.assertRaises(ValueError):
            ValueIterationConfig(max_iterations=0)

    def test_value_iteration_converges_and_solves_default_room(self) -> None:
        environment = Room1Environment(default_room1_config())
        config = ValueIterationConfig(gamma=0.95, theta=1e-6, seed=42)
        result = run_value_iteration(environment, config)

        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.iterations, config.max_iterations)
        self.assertIn(environment.start, result.policy)

        episodes = evaluate_policy(
            environment, result.policy, episodes=10, max_timesteps=250, seed=42
        )
        self.assertTrue(all(episode.success for episode in episodes))

    def test_value_iteration_callback_is_invoked(self) -> None:
        environment = Room1Environment(default_room1_config())
        config = ValueIterationConfig(gamma=0.9, max_iterations=5)
        metrics_logged = []

        def callback(metric, values, policy):
            metrics_logged.append(metric)

        result = run_value_iteration(environment, config, callback=callback)
        self.assertEqual(len(metrics_logged), result.iterations)

    def test_artifact_round_trip_for_value_iteration(self) -> None:
        config_env = default_room1_config()
        config_env.slippery = generate_random_slippery_cells(config_env, count=5, seed=10)
        environment = Room1Environment(config_env)

        alg_config = ValueIterationConfig(gamma=0.95, theta=1e-6, seed=99)
        result = run_value_iteration(environment, alg_config)

        artifact_json = export_room1_artifact(environment, alg_config, result)
        loaded_env, loaded_config, loaded_res = import_room1_artifact(artifact_json)

        self.assertIsInstance(loaded_config, ValueIterationConfig)
        self.assertIsInstance(loaded_res, ValueIterationResult)
        self.assertEqual(loaded_config.gamma, alg_config.gamma)
        self.assertEqual(loaded_config.theta, alg_config.theta)
        self.assertEqual(loaded_res.iterations, result.iterations)
        self.assertEqual(loaded_res.policy, result.policy)
        for state, value in result.values.items():
            self.assertAlmostEqual(loaded_res.values[state], value)


if __name__ == "__main__":
    unittest.main()
