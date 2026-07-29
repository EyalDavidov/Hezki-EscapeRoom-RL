from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room4_artifact, import_room4_artifact
from escape_room_rl.dqn import DQNConfig, run_dqn
from escape_room_rl.evaluation import evaluate_room4_dqn
from escape_room_rl.room4 import Action4, PipeObstacle, Room4Config, Room4Environment


class Room4DQNTests(unittest.TestCase):
    def test_room4_environment_physics(self) -> None:
        config = Room4Config(
            width=10.0,
            height=10.0,
            dt=0.02,
            start=(0.5, 5.0, 0.0, 0.0),
            pipes=[PipeObstacle(x=5.0, width=1.0, gap_start=4.0, gap_size=2.0)],
        )
        env = Room4Environment(config)

        # Reset check
        state = env.reset()
        self.assertEqual(state, (0.5, 5.0, 0.0, 0.0))

        # Move right (Action RIGHT: Vx=1.0, Vy=0.0)
        tr = env.step(state, Action4.RIGHT)
        self.assertAlmostEqual(tr.next_state[0], 0.52, places=4)
        self.assertAlmostEqual(tr.next_state[1], 5.0, places=4)
        self.assertFalse(tr.done)

        # Test pipe collision
        # Pipe is at x=5.0 with gap [4.0, 6.0]. Position (5.0, 2.0) should collide!
        self.assertTrue(env.is_terminal((5.0, 2.0, 0.0, 0.0)))
        # Position inside gap (5.0, 5.0) should NOT collide!
        self.assertFalse(env.is_terminal((5.0, 5.0, 0.0, 0.0)))

    def test_dqn_training_and_eval(self) -> None:
        config_env = Room4Config(
            pipes=[PipeObstacle(x=4.0, width=0.6, gap_start=2.0, gap_size=6.0)],
        )
        env = Room4Environment(config_env)
        config_algo = DQNConfig(
            alpha=0.001,
            episodes=15,
            max_timesteps=100,
            batch_size=16,
            buffer_capacity=500,
            hidden_dims=(32, 32),
            seed=42,
        )
        result = run_dqn(env, config_algo)
        self.assertEqual(len(result.metrics), 15)

        # Evaluation test
        eval_results = evaluate_room4_dqn(env, result.policy_net, episodes=3, max_timesteps=100, seed=42)
        self.assertEqual(len(eval_results), 3)

    def test_room4_artifact_roundtrip(self) -> None:
        config_env = Room4Config(
            pipes=[
                PipeObstacle(x=3.0, width=0.5, gap_start=3.0, gap_size=4.0),
                PipeObstacle(x=6.0, width=0.5, gap_start=1.5, gap_size=4.0),
            ],
        )
        env = Room4Environment(config_env)
        config_algo = DQNConfig(episodes=5, hidden_dims=(32, 32), seed=12)
        result = run_dqn(env, config_algo)

        artifact_json = export_room4_artifact(env, config_algo, result)
        loaded_env, loaded_algo, loaded_result = import_room4_artifact(artifact_json)

        self.assertEqual(len(loaded_env.config.pipes), 2)
        self.assertEqual(loaded_env.config.pipes[0].x, 3.0)
        self.assertEqual(loaded_algo.hidden_dims, (32, 32))
        self.assertEqual(len(loaded_result.metrics), 5)


if __name__ == "__main__":
    unittest.main()
