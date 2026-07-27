"""Policy TEST execution and trajectory capture."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .room1 import Action, Room1Environment, State


@dataclass(frozen=True)
class EpisodeStep:
    timestep: int
    state: State
    action: Action
    next_state: State
    reward: float
    cumulative_reward: float
    events: tuple[str, ...]
    outcome: str


@dataclass
class EpisodeResult:
    episode: int
    success: bool
    timesteps: int
    total_reward: float
    slipped_count: int
    slippery_entries: int
    trajectory: list[EpisodeStep] = field(default_factory=list)


def evaluate_policy(
    environment: Room1Environment,
    policy: dict[State, Action],
    episodes: int,
    max_timesteps: int,
    seed: int,
) -> list[EpisodeResult]:
    if episodes <= 0 or max_timesteps <= 0:
        raise ValueError("episodes and max_timesteps must be positive.")
    rng = np.random.default_rng(seed)
    results: list[EpisodeResult] = []
    for episode in range(1, episodes + 1):
        state = environment.start
        total_reward = 0.0
        slipped_count = 0
        slippery_entries = 0
        trajectory: list[EpisodeStep] = []
        for timestep in range(1, max_timesteps + 1):
            action = policy.get(state)
            if action is None:
                break
            transition = environment.step(state, action, rng)
            total_reward += transition.reward
            slipped_count += int("slipped" in transition.events)
            slippery_entries += int("entered_slippery" in transition.events)
            trajectory.append(
                EpisodeStep(
                    timestep=timestep,
                    state=state,
                    action=action,
                    next_state=transition.next_state,
                    reward=transition.reward,
                    cumulative_reward=total_reward,
                    events=transition.events,
                    outcome=transition.outcome,
                )
            )
            state = transition.next_state
            if transition.done:
                break
        results.append(
            EpisodeResult(
                episode=episode,
                success=state == environment.goal,
                timesteps=len(trajectory),
                total_reward=float(total_reward),
                slipped_count=slipped_count,
                slippery_entries=slippery_entries,
                trajectory=trajectory,
            )
        )
    return results
