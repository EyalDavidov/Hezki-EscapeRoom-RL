"""Synchronous Value Iteration for a finite, known MDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .policy_iteration import action_value
from .room1 import Action, Room1Environment, State


@dataclass(frozen=True)
class ValueIterationConfig:
    gamma: float = 0.95
    theta: float = 1e-6
    max_iterations: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")


@dataclass(frozen=True)
class ValueIterationMetric:
    global_step: int
    iteration: int
    delta: float
    mean_value: float
    policy_changes: int


from .evaluation import EpisodeResult, evaluate_policy


@dataclass
class ValueIterationResult:
    values: dict[State, float]
    policy: dict[State, Action]
    metrics: list[ValueIterationMetric] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    training_episodes: list[EpisodeResult] = field(default_factory=list)


MetricCallback = Callable[
    [ValueIterationMetric, dict[State, float], dict[State, Action]], None
]


def run_value_iteration(
    environment: Room1Environment,
    config: ValueIterationConfig,
    callback: MetricCallback | None = None,
) -> ValueIterationResult:
    rng = np.random.default_rng(config.seed)
    values = {state: 0.0 for state in environment.states}
    policy: dict[State, Action] = {}

    metrics: list[ValueIterationMetric] = []
    prev_policy: dict[State, Action] = {}

    for iteration in range(1, config.max_iterations + 1):
        old_values = values.copy()
        new_values = old_values.copy()
        for terminal_state in environment.terminal_states:
            new_values[terminal_state] = 0.0

        new_policy: dict[State, Action] = {}
        for state in environment.non_terminal_states:
            legal = environment.legal_actions(state)
            action_values = {
                action: action_value(environment, old_values, state, action, config.gamma)
                for action in legal
            }
            best_value = max(action_values.values())
            new_values[state] = best_value

            best_actions = [
                action
                for action, val in action_values.items()
                if np.isclose(val, best_value, rtol=1e-10, atol=1e-12)
            ]
            current = prev_policy.get(state)
            selected = (
                current
                if current in best_actions
                else best_actions[int(rng.integers(0, len(best_actions)))]
            )
            new_policy[state] = selected

        last_delta = max(
            abs(new_values[state] - old_values[state])
            for state in environment.non_terminal_states
        )
        policy_changes = sum(
            1 for state in environment.non_terminal_states if new_policy.get(state) != prev_policy.get(state)
        )

        values = new_values
        policy = new_policy
        prev_policy = policy.copy()

        metric = ValueIterationMetric(
            global_step=iteration,
            iteration=iteration,
            delta=float(last_delta),
            mean_value=float(np.mean(list(values.values()))),
            policy_changes=policy_changes,
        )
        metrics.append(metric)
        if callback is not None:
            callback(metric, values.copy(), policy.copy())

        if last_delta < config.theta:
            sample_eps = evaluate_policy(environment, policy, episodes=5, max_timesteps=100, seed=config.seed)
            return ValueIterationResult(
                values=values,
                policy=policy,
                metrics=metrics,
                converged=True,
                iterations=iteration,
                training_episodes=sample_eps,
            )

    sample_eps = evaluate_policy(environment, policy, episodes=5, max_timesteps=100, seed=config.seed)
    return ValueIterationResult(
        values=values,
        policy=policy,
        metrics=metrics,
        converged=False,
        iterations=config.max_iterations,
        training_episodes=sample_eps,
    )

