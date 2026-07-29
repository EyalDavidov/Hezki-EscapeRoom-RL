from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from escape_room_rl.artifacts import export_room1_artifact, import_room1_artifact
from escape_room_rl.evaluation import evaluate_policy
from escape_room_rl.policy_iteration import PolicyIterationConfig, run_policy_iteration
from escape_room_rl.room1 import (
    Action,
    Room1Config,
    Room1Environment,
    SlipperyCell,
    default_room1_config,
    generate_random_grid_layout,
    generate_random_slippery_cells,
)
from escape_room_rl.display_formatting import format_reward_label
from escape_room_rl.visualization import render_grid_html


class Room1EnvironmentTests(unittest.TestCase):
    def test_reward_labels_use_signed_colored_numbers(self) -> None:
        config = default_room1_config()
        config.slippery = {(1, 0): SlipperyCell()}
        config.terminal_states = frozenset({config.goal, (4, 4)})
        config.cell_rewards = {
            (1, 0): 1.5,
            config.goal: 2.5,
            (4, 4): -0.75,
        }
        environment = Room1Environment(config)

        rendered = render_grid_html(environment)

        self.assertEqual(format_reward_label(2.5), "+2.5")
        self.assertEqual(format_reward_label(-0.75), "-0.75")
        self.assertIn("❄️", rendered)
        self.assertIn('class="cell-reward reward-positive">+1.5</span>', rendered)
        self.assertIn('class="cell-reward reward-positive">+2.5</span>', rendered)
        self.assertIn('class="cell-reward reward-negative">-0.75</span>', rendered)
        self.assertIn("🚪", rendered)
        self.assertIn("🛑", rendered)
        self.assertNotIn("🎁", rendered)

    def test_coordinate_convention_and_boundary_actions(self) -> None:
        environment = Room1Environment(default_room1_config())
        self.assertEqual(environment.start, (0, 0))
        self.assertEqual(environment.goal, (9, 9))
        self.assertNotIn(Action.DOWN, environment.legal_actions((0, 0)))
        self.assertNotIn(Action.RIGHT, environment.legal_actions((0, 0)))
        self.assertIn(Action.UP, environment.legal_actions((0, 0)))
        self.assertIn(Action.LEFT, environment.legal_actions((0, 0)))

    def test_known_model_probabilities_sum_to_one(self) -> None:
        config = default_room1_config()
        config.slippery = {(1, 0): SlipperyCell()}
        environment = Room1Environment(config)
        for state in environment.non_terminal_states:
            for action in environment.legal_actions(state):
                total = sum(
                    transition.probability
                    for transition in environment.transition_model(state, action)
                )
                self.assertAlmostEqual(total, 1.0)

    def test_slip_is_triggered_by_the_target_cell(self) -> None:
        config = default_room1_config()
        config.slippery = {
            (1, 0): SlipperyCell(reach=0.0, up=1.0, down=0.0, right=0.0, left=0.0)
        }
        environment = Room1Environment(config)
        transitions = environment.transition_model((0, 0), Action.LEFT)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].next_state, (1, 1))
        self.assertIn("entered_slippery", transitions[0].events)
        self.assertIn("slipped", transitions[0].events)

    def test_blocked_slip_ends_on_the_slippery_target(self) -> None:
        config = default_room1_config()
        config.slippery = {
            (1, 0): SlipperyCell(reach=0.0, up=0.0, down=1.0, right=0.0, left=0.0)
        }
        environment = Room1Environment(config)
        transition = environment.transition_model((0, 0), Action.LEFT)[0]
        self.assertEqual(transition.next_state, (1, 0))
        self.assertIn("blocked_slip", transition.events)

    def test_random_generator_is_reproducible_and_excludes_terminals(self) -> None:
        config = default_room1_config()
        first = generate_random_slippery_cells(config, count=12, seed=77)
        second = generate_random_slippery_cells(config, count=12, seed=77)
        self.assertEqual(first, second)
        self.assertNotIn(config.start, first)
        self.assertNotIn(config.goal, first)
        self.assertEqual(len(first), 12)

    def test_fixed_map_has_a_path_from_start_to_goal(self) -> None:
        environment = Room1Environment(default_room1_config())
        frontier = [environment.start]
        visited = {environment.start}
        while frontier:
            state = frontier.pop()
            for action in environment.legal_actions(state):
                next_state = environment.move(state, action)
                if next_state not in visited:
                    visited.add(next_state)
                    frontier.append(next_state)
        self.assertIn(environment.goal, visited)

    def test_custom_wall_changes_the_available_actions(self) -> None:
        config = default_room1_config()
        config.walls = frozenset(set(config.walls) | {(1, 0)})
        environment = Room1Environment(config)
        self.assertNotIn(Action.LEFT, environment.legal_actions((0, 0)))
        self.assertIn(Action.UP, environment.legal_actions((0, 0)))

    def test_grid_rejects_a_wall_barrier_that_blocks_the_goal(self) -> None:
        config = default_room1_config()
        config.walls = frozenset({(1, y) for y in range(10)})
        with self.assertRaisesRegex(ValueError, "goal is unreachable"):
            Room1Environment(config)

    def test_custom_start_goal_termination_and_cell_reward(self) -> None:
        config = default_room1_config()
        config.slippery = {}
        config.start = (1, 0)
        config.goal = (8, 9)
        config.terminal_states = frozenset({(8, 9), (4, 4)})
        config.cell_rewards = {(1, 1): 2.5}
        environment = Room1Environment(config)

        rewarded = environment.transition_model((0, 1), Action.LEFT)[0]
        self.assertAlmostEqual(rewarded.reward, 2.4)
        termination = environment.transition_model((3, 4), Action.LEFT)[0]
        self.assertTrue(termination.done)
        self.assertNotEqual(termination.next_state, environment.goal)

    def test_full_grid_generator_is_reproducible_and_uses_integer_percentages(self) -> None:
        first = generate_random_grid_layout(20, 8, 91)
        second = generate_random_grid_layout(20, 8, 91)
        self.assertEqual(first, second)
        self.assertEqual(len(first.walls), 20)
        self.assertEqual(len(first.slippery), 8)
        self.assertNotEqual(first.start, first.goal)
        for slippery in first.slippery.values():
            percentages = [value * 100 for value in slippery.as_dict().values()]
            self.assertTrue(all(value.is_integer() for value in percentages))
            self.assertEqual(sum(percentages), 100)
        Room1Environment(
            Room1Config(
                start=first.start,
                goal=first.goal,
                walls=first.walls,
                slippery=first.slippery,
                terminal_states=frozenset({first.goal}),
            )
        )


class PolicyIterationTests(unittest.TestCase):
    def test_policy_iteration_converges_and_solves_default_room(self) -> None:
        environment = Room1Environment(default_room1_config())
        algorithm_config = PolicyIterationConfig(seed=9)
        result = run_policy_iteration(environment, algorithm_config)
        self.assertTrue(result.converged)
        episodes = evaluate_policy(
            environment, result.policy, episodes=10, max_timesteps=250, seed=9
        )
        self.assertTrue(all(episode.success for episode in episodes))
        self.assertTrue(all(episode.timesteps <= 250 for episode in episodes))

    def test_test_run_does_not_mutate_policy(self) -> None:
        environment = Room1Environment(default_room1_config())
        result = run_policy_iteration(environment, PolicyIterationConfig(seed=3))
        policy_before = result.policy.copy()
        evaluate_policy(environment, result.policy, 5, 250, 4)
        self.assertEqual(result.policy, policy_before)

    def test_artifact_round_trip_preserves_model(self) -> None:
        config = default_room1_config()
        config.slippery = generate_random_slippery_cells(config, 5, 12)
        config.cell_rewards = {(1, 1): 1.25}
        config.terminal_states = frozenset({config.goal, (4, 4)})
        environment = Room1Environment(config)
        algorithm_config = PolicyIterationConfig(seed=12)
        result = run_policy_iteration(environment, algorithm_config)
        artifact = export_room1_artifact(environment, algorithm_config, result)
        loaded_environment, loaded_config, loaded_result = import_room1_artifact(artifact)
        self.assertEqual(loaded_environment.config.walls, config.walls)
        self.assertEqual(loaded_environment.config.slippery, config.slippery)
        self.assertEqual(loaded_environment.config.rewards, config.rewards)
        self.assertEqual(loaded_environment.config.terminal_states, config.terminal_states)
        self.assertEqual(loaded_environment.config.cell_rewards, config.cell_rewards)
        self.assertEqual(loaded_config, algorithm_config)
        self.assertEqual(loaded_result.policy, result.policy)
        for state, value in result.values.items():
            self.assertAlmostEqual(loaded_result.values[state], value)


if __name__ == "__main__":
    unittest.main()
