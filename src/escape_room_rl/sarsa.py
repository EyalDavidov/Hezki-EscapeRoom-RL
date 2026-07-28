"""SARSA (On-Policy TD Control) for Room 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .room1 import Action, State
from .room2 import Room2Environment


@dataclass(frozen=True)
class SarsaConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 500
    max_timesteps: int = 250
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("Invalid epsilon parameters.")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if self.episodes <= 0 or self.max_timesteps <= 0:
            raise ValueError("episodes and max_timesteps must be positive.")


@dataclass(frozen=True)
class SarsaTrainingMetric:
    episode: int
    total_reward: float
    timesteps: int
    success: bool
    epsilon: float
    max_q_delta: float
    mean_q_value: float


@dataclass
class SarsaResult:
    q_table: dict[tuple[State, Action], float]
    policy: dict[State, Action]
    values: dict[State, float]
    metrics: list[SarsaTrainingMetric] = field(default_factory=list)
    converged: bool = False
    episodes_run: int = 0


SarsaCallback = Callable[[SarsaTrainingMetric, dict[State, float], dict[State, Action]], None]


def select_epsilon_greedy_action(
    q_table: dict[tuple[State, Action], float],
    legal_actions: tuple[Action, ...],
    state: State,
    epsilon: float,
    rng: np.random.Generator,
) -> Action:
    if not legal_actions:
        raise ValueError(f"No legal actions available from state {state}")
    if rng.random() < epsilon:
        return legal_actions[int(rng.integers(0, len(legal_actions)))]

    q_vals = [q_table.get((state, a), 0.0) for a in legal_actions]
    max_val = max(q_vals)
    best_actions = [a for a, val in zip(legal_actions, q_vals) if np.isclose(val, max_val, atol=1e-12)]
    return best_actions[int(rng.integers(0, len(best_actions)))]


def derive_policy_and_values(
    environment: Room2Environment,
    q_table: dict[tuple[State, Action], float],
) -> tuple[dict[State, Action], dict[State, float]]:
    policy: dict[State, Action] = {}
    values: dict[State, float] = {}

    for state in environment.states:
        if environment.is_terminal(state):
            values[state] = 0.0
            continue
        legal = environment.legal_actions(state)
        if not legal:
            values[state] = 0.0
            continue
        q_vals = {action: q_table.get((state, action), 0.0) for action in legal}
        best_val = max(q_vals.values())
        best_actions = [a for a, val in q_vals.items() if np.isclose(val, best_val, atol=1e-12)]
        policy[state] = best_actions[0]
        values[state] = float(best_val)

    return policy, values


def run_sarsa(
    environment: Room2Environment,
    config: SarsaConfig,
    callback: SarsaCallback | None = None,
) -> SarsaResult:
    rng = np.random.default_rng(config.seed)
    q_table: dict[tuple[State, Action], float] = {}

    # Initialize Q-table to 0.0 for all valid state-action pairs
    for state in environment.states:
        for action in environment.legal_actions(state):
            q_table[(state, action)] = 0.0

    metrics: list[SarsaTrainingMetric] = []
    epsilon = config.epsilon_start

    for episode in range(1, config.episodes + 1):
        state = environment.start
        legal = environment.legal_actions(state)
        if not legal:
            break
        action = select_epsilon_greedy_action(q_table, legal, state, epsilon, rng)

        total_reward = 0.0
        max_delta = 0.0
        timesteps = 0
        success = False

        for _ in range(config.max_timesteps):
            timesteps += 1
            transition = environment.step(state, action, rng)
            next_state = transition.next_state
            reward = transition.reward
            total_reward += reward

            old_q = q_table.get((state, action), 0.0)

            if transition.done:
                target = reward
                next_action = None
            else:
                next_legal = environment.legal_actions(next_state)
                next_action = select_epsilon_greedy_action(
                    q_table, next_legal, next_state, epsilon, rng
                )
                target = reward + config.gamma * q_table.get((next_state, next_action), 0.0)

            new_q = old_q + config.alpha * (target - old_q)
            q_table[(state, action)] = float(new_q)
            max_delta = max(max_delta, abs(new_q - old_q))

            if transition.done:
                success = next_state == environment.goal
                break

            state = next_state
            action = next_action  # type: ignore

        # Decay epsilon after episode
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        policy, values = derive_policy_and_values(environment, q_table)
        mean_q = float(np.mean(list(q_table.values()))) if q_table else 0.0

        metric = SarsaTrainingMetric(
            episode=episode,
            total_reward=float(total_reward),
            timesteps=timesteps,
            success=success,
            epsilon=float(epsilon),
            max_q_delta=float(max_delta),
            mean_q_value=mean_q,
        )
        metrics.append(metric)

        if callback is not None:
            callback(metric, values, policy)

    policy, values = derive_policy_and_values(environment, q_table)
    return SarsaResult(
        q_table=q_table,
        policy=policy,
        values=values,
        metrics=metrics,
        converged=any(m.success for m in metrics[-20:]),  # High recent success rate
        episodes_run=config.episodes,
    )
