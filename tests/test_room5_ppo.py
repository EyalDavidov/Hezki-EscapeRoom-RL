from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room5_artifact, import_room5_artifact
from escape_room_rl.evaluation import evaluate_room5_ppo
from escape_room_rl.ppo import PPOConfig, run_ppo
from escape_room_rl.room5 import (
    OBSERVATION_SIZE,
    Action5,
    Room5Config,
    Room5Environment,
    TrafficCar,
)
from escape_room_rl.visualization import render_room5_html


class Room5PPOTests(unittest.TestCase):
    def test_lane_count_and_field_of_view_shape_the_observation(self) -> None:
        environment = Room5Environment(
            Room5Config(lane_count=2, vision_distance=50.0, traffic_count=4)
        )
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=0, lane=0, distance=25.0, speed=15.0),
            TrafficCar(car_id=1, lane=1, distance=80.0, speed=15.0),
        ]

        observation = environment.observation()
        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertEqual(observation[1], 1.0)
        self.assertAlmostEqual(observation[6], 0.5)
        self.assertEqual(observation[7], 1.0)
        self.assertEqual(observation[8], -1.0)

    def test_collision_and_overtake_rewards(self) -> None:
        rewards = {
            "step": 0.0,
            "forward_progress": 0.0,
            "overtake": 7.0,
            "lane_change": 0.0,
            "invalid_lane_change": -1.0,
            "collision": -20.0,
            "goal_reached": 0.0,
        }
        environment = Room5Environment(
            Room5Config(lane_count=2, traffic_count=1, rewards=rewards)
        )
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=0, lane=1, distance=6.4, speed=20.0)
        ]
        collision = environment.step(Action5.KEEP_LANE)
        self.assertTrue(collision.done)
        self.assertIn("collision", collision.events)
        self.assertEqual(collision.reward, -20.0)

        environment.reset(7)
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=1, lane=0, distance=-4.4, speed=20.0)
        ]
        overtake = environment.step(Action5.KEEP_LANE)
        self.assertFalse(overtake.done)
        self.assertIn("overtake", overtake.events)
        self.assertEqual(overtake.reward, 7.0)

    def test_lane_boundaries_and_visualization(self) -> None:
        environment = Room5Environment(Room5Config(lane_count=6, traffic_count=5))
        environment.ego_lane = 0
        transition = environment.step(Action5.LEFT)
        self.assertIn("invalid_lane_change", transition.events)
        self.assertEqual(environment.ego_lane, 0)

        html = render_room5_html(environment)
        self.assertIn("one-way road with 6 lanes", html)
        self.assertIn("AGENT", html)
        self.assertIn("VISION", html)

    def test_ppo_training_evaluation_and_artifact_roundtrip(self) -> None:
        environment = Room5Environment(
            Room5Config(
                lane_count=3,
                vision_distance=80.0,
                road_length=120.0,
                traffic_count=5,
                seed=11,
            )
        )
        config = PPOConfig(
            episodes=3,
            max_timesteps=25,
            update_epochs=2,
            mini_batch_size=16,
            hidden_dims=(32, 32),
            seed=11,
        )
        result = run_ppo(environment, config)

        self.assertEqual(len(result.metrics), 3)
        self.assertGreaterEqual(result.training_duration_seconds, 0.0)
        self.assertEqual(
            sum(result.action_counts.values()),
            sum(metric.timesteps for metric in result.metrics),
        )
        evaluation = evaluate_room5_ppo(
            environment,
            result.policy_net,
            episodes=2,
            max_timesteps=20,
            seed=99,
        )
        self.assertEqual(len(evaluation), 2)
        self.assertTrue(all(item.trajectory for item in evaluation))

        artifact = export_room5_artifact(environment, config, result)
        loaded_environment, loaded_config, loaded_result = import_room5_artifact(artifact)
        self.assertEqual(loaded_environment.config.lane_count, 3)
        self.assertEqual(loaded_environment.config.vision_distance, 80.0)
        self.assertEqual(loaded_config.hidden_dims, (32, 32))
        self.assertEqual(loaded_result.action_counts, result.action_counts)
        self.assertEqual(len(loaded_result.metrics), 3)


if __name__ == "__main__":
    unittest.main()
