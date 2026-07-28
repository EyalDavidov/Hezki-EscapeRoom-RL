from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room3_artifact, import_room3_artifact
from escape_room_rl.evaluation import evaluate_policy
from escape_room_rl.q_learning import QLearningConfig, run_q_learning
from escape_room_rl.room1 import Action, SlipperyCell
from escape_room_rl.room3 import Room3Config, Room3Environment, default_room3_config


class Room3QLearningTests(unittest.TestCase):
    def test_room3_environment_legality(self) -> None:
        env = Room3Environment(default_room3_config())
        self.assertEqual(env.start, (0, 0))
        self.assertEqual(env.goal, (9, 9))

    def test_q_learning_solves_room3(self) -> None:
        env = Room3Environment(default_room3_config())
        config = QLearningConfig(
            alpha=0.2,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            episodes=300,
            max_timesteps=200,
            seed=42,
        )
        result = run_q_learning(env, config)
        self.assertTrue(len(result.metrics) == 300)

        episodes = evaluate_policy(env, result.policy, episodes=10, max_timesteps=200, seed=42)
        success_rate = sum(ep.success for ep in episodes) / len(episodes)
        self.assertGreaterEqual(success_rate, 0.8)

    def test_room3_artifact_roundtrip(self) -> None:
        config_env = default_room3_config()
        config_env.slippery = {(1, 0): SlipperyCell()}
        env = Room3Environment(config_env)
        config = QLearningConfig(episodes=20, seed=15)
        result = run_q_learning(env, config)

        artifact = export_room3_artifact(env, config, result)
        loaded_env, loaded_config, loaded_result = import_room3_artifact(artifact)

        self.assertEqual(loaded_env.config.walls, config_env.walls)
        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_result.policy, result.policy)


if __name__ == "__main__":
    unittest.main()
