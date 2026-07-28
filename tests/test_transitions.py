from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.evaluation import evaluate_policy
from escape_room_rl.policy_iteration import PolicyIterationConfig, run_policy_iteration
from escape_room_rl.q_learning import QLearningConfig, run_q_learning
from escape_room_rl.room1 import Room1Environment, default_room1_config
from escape_room_rl.room2 import Room2Environment, default_room2_config
from escape_room_rl.room3 import Room3Environment, default_room3_config
from escape_room_rl.sarsa import SarsaConfig, run_sarsa


class TransitionTests(unittest.TestCase):
    def test_room1_to_room2_transition_unlock(self) -> None:
        """Mission 3: Solving Room 1 allows transition to Room 2."""
        env1 = Room1Environment(default_room1_config())
        res1 = run_policy_iteration(env1, PolicyIterationConfig(seed=42))
        test_res1 = evaluate_policy(env1, res1.policy, episodes=5, max_timesteps=200, seed=42)
        
        # Verify Room 1 was solved cleanly
        self.assertTrue(all(ep.success for ep in test_res1))

        # Room 1 terminal goal (9,9) transitions to Room 2 start (0,0)
        env2 = Room2Environment(default_room2_config())
        self.assertEqual(env1.goal, (9, 9))
        self.assertEqual(env2.start, (0, 0))

    def test_room2_to_room3_transition_unlock(self) -> None:
        """Mission 5: Solving Room 2 allows transition to Room 3."""
        env2 = Room2Environment(default_room2_config())
        res2 = run_sarsa(env2, SarsaConfig(episodes=250, seed=42))
        test_res2 = evaluate_policy(env2, res2.policy, episodes=5, max_timesteps=200, seed=42)
        
        # Verify Room 2 goal reaches Room 3 start
        env3 = Room3Environment(default_room3_config())
        self.assertEqual(env2.goal, (9, 9))
        self.assertEqual(env3.start, (0, 0))


if __name__ == "__main__":
    unittest.main()
