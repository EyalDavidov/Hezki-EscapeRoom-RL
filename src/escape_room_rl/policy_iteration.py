"""Synchronous Policy Iteration for a finite, known MDP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .room1 import Action, Room1Environment, State


@dataclass(frozen=True)
class PolicyIterationConfig:
    gamma: float = 0.95
    theta: float = 1e-6
    max_policy_iterations: int = 100
    max_evaluation_sweeps: int = 10_000
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive.")
        if self.max_policy_iterations <= 0 or self.max_evaluation_sweeps <= 0:
            raise ValueError("Iteration limits must be positive.")


@dataclass(frozen=True)
class TrainingMetric:
    global_step: int
    policy_iteration: int
    evaluation_sweep: int
    phase: str
    delta: float
    mean_value: float
    policy_changes: int


@dataclass
class PolicyIterationResult:
    values: dict[State, float]
    policy: dict[State, Action]
    metrics: list[TrainingMetric] = field(default_factory=list)
    converged: bool = False
    policy_iterations: int = 0
    evaluation_sweeps: int = 0


MetricCallback = Callable[
    [TrainingMetric, dict[State, float], dict[State, Action]], None
]


def action_value(
    environment: Room1Environment,
    values: dict[State, float],
    state: State,
    action: Action,
    gamma: float,
) -> float:
    total = 0.0
    for transition in environment.transition_model(state, action):
        future = 0.0 if transition.done else gamma * values[transition.next_state]
        total += transition.probability * (transition.reward + future)
    return float(total)


def run_policy_iteration(
    environment: Room1Environment,
    config: PolicyIterationConfig,
    callback: MetricCallback | None = None,
) -> PolicyIterationResult:
    rng = np.random.default_rng(config.seed)
    values = {state: 0.0 for state in environment.states}
    policy: dict[State, Action] = {}
    for state in environment.non_terminal_states:
        legal = environment.legal_actions(state)
        policy[state] = legal[int(rng.integers(0, len(legal)))]

    metrics: list[TrainingMetric] = []
    global_step = 0
    total_evaluation_sweeps = 0

    for policy_iteration in range(1, config.max_policy_iterations + 1):
        last_delta = float("inf")
        for evaluation_sweep in range(1, config.max_evaluation_sweeps + 1):
            old_values = values.copy()
            new_values = old_values.copy()
            for terminal_state in environment.terminal_states:
                new_values[terminal_state] = 0.0
            for state in environment.non_terminal_states:
                new_values[state] = action_value(
                    environment,
                    old_values,
                    state,
                    policy[state],
                    config.gamma,
                )
            last_delta = max(
                abs(new_values[state] - old_values[state])
                for state in environment.non_terminal_states
            )
            values = new_values
            global_step += 1
            total_evaluation_sweeps += 1
            metric = TrainingMetric(
                global_step=global_step,
                policy_iteration=policy_iteration,
                evaluation_sweep=evaluation_sweep,
                phase="evaluation",
                delta=float(last_delta),
                mean_value=float(np.mean(list(values.values()))),
                policy_changes=0,
            )
            metrics.append(metric)
            if callback is not None:
                callback(metric, values.copy(), policy.copy())
            if last_delta < config.theta:
                break

        policy_changes = 0
        new_policy = policy.copy()
        for state in environment.non_terminal_states:
            legal = environment.legal_actions(state)
            action_values = {
                action: action_value(environment, values, state, action, config.gamma)
                for action in legal
            }
            best_value = max(action_values.values())
            best_actions = [
                action
                for action, value in action_values.items()
                if np.isclose(value, best_value, rtol=1e-10, atol=1e-12)
            ]
            current = policy[state]
            selected = (
                current
                if current in best_actions
                else best_actions[int(rng.integers(0, len(best_actions)))]
            )
            new_policy[state] = selected
            if selected != current:
                policy_changes += 1
        policy = new_policy
        global_step += 1
        improvement_metric = TrainingMetric(
            global_step=global_step,
            policy_iteration=policy_iteration,
            evaluation_sweep=evaluation_sweep,
            phase="improvement",
            delta=float(last_delta),
            mean_value=float(np.mean(list(values.values()))),
            policy_changes=policy_changes,
        )
        metrics.append(improvement_metric)
        if callback is not None:
            callback(improvement_metric, values.copy(), policy.copy())

        if policy_changes == 0 and last_delta < config.theta:
            return PolicyIterationResult(
                values=values,
                policy=policy,
                metrics=metrics,
                converged=True,
                policy_iterations=policy_iteration,
                evaluation_sweeps=total_evaluation_sweeps,
            )

    return PolicyIterationResult(
        values=values,
        policy=policy,
        metrics=metrics,
        converged=False,
        policy_iterations=config.max_policy_iterations,
        evaluation_sweeps=total_evaluation_sweeps,
    )
