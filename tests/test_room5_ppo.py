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
    MIN_TRAFFIC_CLEARANCE_METERS,
    OBSERVATION_SIZE,
    Action5,
    Room5Config,
    Room5Environment,
    TrafficCar,
)
from escape_room_rl.visualization import render_room5_html


class Room5PPOTests(unittest.TestCase):
    def test_zero_traffic_is_a_valid_empty_road(self) -> None:
        env = Room5Environment(Room5Config(traffic_count=0))
        env.reset(seed=7)
        for _ in range(20):
            env.step(Action5.KEEP_LANE)
        self.assertEqual(env.traffic, [])

    def test_episode_starts_empty_and_traffic_arrives_progressively(self) -> None:
        environment = Room5Environment(
            Room5Config(
                lane_count=3,
                vision_distance=60.0,
                road_length=1000.0,
                traffic_count=3,
                seed=9,
            )
        )

        environment.reset(9)
        self.assertEqual(environment.traffic, [])
        self.assertEqual(environment.observation()[6:], (1.0, 0.0, 0.0))

        for _ in range(6):
            environment.step(Action5.KEEP_LANE)

        self.assertGreater(len(environment.traffic), 0)
        self.assertLess(len(environment.traffic), environment.config.traffic_count + 1)

    def test_lane_count_and_field_of_view_shape_the_observation(self) -> None:
        environment = Room5Environment(
            Room5Config(lane_count=2, vision_distance=50.0, traffic_count=4)
        )
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=0, lane=0, distance=25.0, speed=15.0),
            TrafficCar(car_id=1, lane=1, distance=40.0, speed=15.0),
        ]

        observation = environment.observation()
        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertEqual(observation[1], 1.0)
        self.assertAlmostEqual(observation[6], 35.5 / 50.0)
        self.assertAlmostEqual(observation[7], 0.5)
        self.assertEqual(observation[8], 0.0)

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

        environment.traffic = [
            TrafficCar(car_id=20, lane=0, distance=42.0, speed=18.0),
            TrafficCar(car_id=21, lane=2, distance=18.25, speed=18.0),
        ]

        html = render_room5_html(environment)
        self.assertIn("one-way road with 6 lanes", html)
        self.assertIn("AGENT", html)
        self.assertIn("CURRENT-LANE VIEW", html)
        self.assertIn("37.5 m", html)
        self.assertNotIn("18.2 m", html)
        self.assertIn("Nearest car distance", html)

    def test_vehicle_distance_uses_physical_edges(self) -> None:
        environment = Room5Environment(
            Room5Config(lane_count=2, vision_distance=100.0, traffic_count=0)
        )
        car = TrafficCar(car_id=1, lane=1, distance=40.0, speed=18.0)
        environment.ego_lane = 1
        environment.traffic = [car]

        self.assertAlmostEqual(environment.forward_clearance(car), 35.5)
        self.assertAlmostEqual(environment.nearest_ahead_distance(1), 35.5)
        self.assertAlmostEqual(environment.observation()[6], 0.355)

    def test_same_lane_traffic_keeps_three_meter_clearance(self) -> None:
        environment = Room5Environment(
            Room5Config(lane_count=2, traffic_count=0, ego_speed=30.0)
        )
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=0, lane=0, distance=30.0, speed=12.0),
            TrafficCar(car_id=1, lane=0, distance=23.0, speed=24.0),
        ]

        environment.step(Action5.KEEP_LANE)
        ordered = sorted(environment.traffic, key=lambda car: car.distance)
        edge_clearance = (
            ordered[1].distance
            - ordered[0].distance
            - environment.config.car_length
        )
        self.assertGreaterEqual(
            edge_clearance + 1e-9,
            MIN_TRAFFIC_CLEARANCE_METERS,
        )

    def test_lane_change_clearance_reward_and_penalty(self) -> None:
        rewards = {
            "step": 0.0,
            "forward_progress": 0.0,
            "overtake": 0.0,
            "lane_change": 0.0,
            "safer_lane_change": 3.0,
            "riskier_lane_change": -4.0,
            "invalid_lane_change": 0.0,
            "collision": 0.0,
            "goal_reached": 0.0,
        }
        environment = Room5Environment(
            Room5Config(
                lane_count=3,
                vision_distance=100.0,
                road_length=1000.0,
                traffic_count=2,
                rewards=rewards,
            )
        )
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=0, lane=1, distance=20.0, speed=20.0),
            TrafficCar(car_id=1, lane=0, distance=60.0, speed=20.0),
        ]
        safer = environment.step(Action5.LEFT)
        self.assertEqual(safer.reward, 3.0)
        self.assertIn("safer_lane_change", safer.events)

        environment.reset(5)
        environment.ego_lane = 1
        environment.traffic = [
            TrafficCar(car_id=2, lane=1, distance=60.0, speed=20.0),
            TrafficCar(car_id=3, lane=0, distance=20.0, speed=20.0),
        ]
        riskier = environment.step(Action5.LEFT)
        self.assertEqual(riskier.reward, -4.0)
        self.assertIn("riskier_lane_change", riskier.events)

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
