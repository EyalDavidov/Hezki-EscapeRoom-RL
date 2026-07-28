from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room2_artifact, import_room2_artifact
from escape_room_rl.evaluation import evaluate_policy
from escape_room_rl.room1 import Action, SlipperyCell
from escape_room_rl.room2 import Room2Config, Room2Environment, default_room2_config
from escape_room_rl.sarsa import SarsaConfig, run_sarsa


class Room2SarsaTests(unittest.TestCase):
    def test_room2_environment_legality_and_boundaries(self) -> None:
        env = Room2Environment(default_room2_config())
        self.assertEqual(env.start, (0, 0))
        self.assertEqual(env.goal, (9, 9))
        self.assertNotIn(Action.DOWN, env.legal_actions((0, 0)))
        self.assertNotIn(Action.RIGHT, env.legal_actions((0, 0)))

    def test_sarsa_learns_and_solves_room2(self) -> None:
        env = Room2Environment(default_room2_config())
        config = SarsaConfig(
            alpha=0.2,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.99,
            episodes=300,
            max_timesteps=200,
            seed=42,
        )
        result = run_sarsa(env, config)
        self.assertTrue(len(result.metrics) == 300)
        self.assertIsNotNone(result.policy)

        # Evaluate policy
        episodes = evaluate_policy(env, result.policy, episodes=10, max_timesteps=200, seed=42)
        success_rate = sum(ep.success for ep in episodes) / len(episodes)
        self.assertGreaterEqual(success_rate, 0.8)

    def test_room2_artifact_roundtrip(self) -> None:
        config_env = default_room2_config()
        config_env.slippery = {(1, 0): SlipperyCell()}
        config_env.terminal_states = frozenset({config_env.goal, (4, 4)})
        config_env.cell_rewards = {(1, 1): 1.25}
        env = Room2Environment(config_env)
        config = SarsaConfig(episodes=20, seed=7)
        result = run_sarsa(env, config)

        artifact = export_room2_artifact(env, config, result)
        loaded_env, loaded_config, loaded_result = import_room2_artifact(artifact)

        self.assertEqual(loaded_env.config.walls, config_env.walls)
        self.assertEqual(loaded_env.config.terminal_states, config_env.terminal_states)
        self.assertEqual(loaded_env.config.cell_rewards, config_env.cell_rewards)
        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_result.policy, result.policy)


if __name__ == "__main__":
    unittest.main()
