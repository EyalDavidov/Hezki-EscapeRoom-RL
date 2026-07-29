from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room4_artifact, import_room4_artifact
from escape_room_rl.dqn import DQNConfig, run_dqn
from escape_room_rl.evaluation import evaluate_room4_dqn
from escape_room_rl.room4 import (
    Action4,
    PipeObstacle,
    Room4Config,
    Room4Environment,
    distribute_pipes_evenly,
)


class Room4DQNTests(unittest.TestCase):
    def test_five_pipes_are_evenly_spaced_and_valid(self) -> None:
        pipes = distribute_pipes_evenly(
            [
                PipeObstacle(x=2.5, gap_start=3.5),
                PipeObstacle(x=5.0, gap_start=2.0),
                PipeObstacle(x=7.5, gap_start=4.5),
            ],
            5,
        )

        self.assertEqual([pipe.x for pipe in pipes], [2.0, 3.5, 5.0, 6.5, 8.0])
        self.assertEqual([pipe.gap_start for pipe in pipes[:3]], [3.5, 2.0, 4.5])
        Room4Environment(Room4Config(pipes=pipes))

    def test_backward_movement_reward_applies_to_left_actions(self) -> None:
        env = Room4Environment(
            Room4Config(
                pipes=[],
                rewards={
                    "step": 0.0,
                    "progress": 0.0,
                    "backward": -2.5,
                    "pipe_passed": 0.0,
                    "goal_reached": 0.0,
                    "collision": 0.0,
                },
            )
        )

        for action in (Action4.DOWN_LEFT, Action4.LEFT, Action4.UP_LEFT):
            transition = env.step((5.0, 5.0, 0.0, 0.0), action)
            self.assertEqual(transition.reward, -2.5)
            self.assertIn("backward_move", transition.events)

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
        self.assertGreaterEqual(result.training_duration_seconds, 0.0)
        self.assertEqual(
            sum(result.action_counts.values()),
            sum(metric.timesteps for metric in result.metrics),
        )
        self.assertEqual(set(result.action_counts), {action.name for action in Action4})

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
        self.assertEqual(loaded_result.action_counts, result.action_counts)
        self.assertAlmostEqual(
            loaded_result.training_duration_seconds,
            result.training_duration_seconds,
        )


    def test_dqn_activation_functions(self) -> None:
        for act in ["ReLU", "LeakyReLU", "Tanh", "ELU", "SiLU"]:
            config_algo = DQNConfig(episodes=2, max_timesteps=10, batch_size=8, buffer_capacity=100, activation_fn=act, seed=42)
            env = Room4Environment(Room4Config())
            result = run_dqn(env, config_algo)
            self.assertEqual(result.config.activation_fn, act)
            self.assertEqual(len(result.metrics), 2)

        with self.assertRaises(ValueError):
            DQNConfig(activation_fn="InvalidActivation")


if __name__ == "__main__":
    unittest.main()

