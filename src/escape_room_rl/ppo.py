"""A compact discrete-action PPO implementation for Room 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .room5 import Action5, OBSERVATION_SIZE, RoadObservation, Room5Environment


ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "SiLU": nn.SiLU,
}


@dataclass(frozen=True)
class PPOConfig:
    alpha: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    update_epochs: int = 4
    mini_batch_size: int = 64
    episodes: int = 300
    max_timesteps: int = 300
    hidden_dims: tuple[int, ...] = (64, 64)
    activation_fn: str = "Tanh"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1).")
        if self.entropy_coefficient < 0 or self.value_coefficient < 0:
            raise ValueError("Loss coefficients cannot be negative.")
        if self.update_epochs <= 0 or self.mini_batch_size <= 0:
            raise ValueError("PPO update settings must be positive.")
        if self.episodes <= 0 or self.max_timesteps <= 0:
            raise ValueError("episodes and max_timesteps must be positive.")
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer sizes.")
        if self.activation_fn not in ACTIVATION_MAP:
            raise ValueError(f"Unknown activation function: {self.activation_fn}.")


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int = OBSERVATION_SIZE,
        action_dim: int = len(Action5),
        hidden_dims: tuple[int, ...] = (64, 64),
        activation_fn: str = "Tanh",
    ):
        super().__init__()
        activation = ACTIVATION_MAP[activation_fn]
        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), activation()))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_dim)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


@dataclass(frozen=True)
class PPOTrainingMetric:
    episode: int
    total_reward: float
    timesteps: int
    success: bool
    overtakes: int = 0
    pipes_passed: int = 0
    collision: bool = False
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    mean_value: float = 0.0


from .evaluation import (
    Room4EpisodeResult,
    Room4EpisodeStep,
    Room5EpisodeResult,
    Room5EpisodeStep,
)


@dataclass
class PPOResult:
    policy_net: ActorCriticNetwork
    metrics: list[PPOTrainingMetric] = field(default_factory=list)
    converged: bool = False
    episodes_run: int = 0
    training_duration_seconds: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    config: PPOConfig = field(default_factory=PPOConfig)
    training_episodes: list[Any] = field(default_factory=list)



PPOCallback = Callable[[PPOTrainingMetric, ActorCriticNetwork], None]


def observation_tensor(observation: Any) -> torch.Tensor:
    if isinstance(observation, torch.Tensor):
        t = observation.float()
    else:
        t = torch.tensor(observation, dtype=torch.float32)

    # Normalize 4D state for Room 4 (x, y, vx, vy)
    if t.ndim == 1 and t.shape[0] == 4:
        norm = t.clone()
        norm[0] = (norm[0] / 5.0) - 1.0
        norm[1] = (norm[1] / 5.0) - 1.0
        norm[2] = norm[2] / 3.0
        norm[3] = norm[3] / 3.0
        return norm
    return t


def select_ppo_action(
    policy_net: ActorCriticNetwork,
    observation: Any,
    *,
    deterministic: bool,
    action_enum: Any = None,
) -> Any:
    with torch.no_grad():
        logits, _ = policy_net(observation_tensor(observation).unsqueeze(0))
        if deterministic:
            action_index = int(torch.argmax(logits, dim=1).item())
        else:
            action_index = int(Categorical(logits=logits).sample().item())
    if action_enum is not None:
        return action_enum(action_index)
    return Action5(action_index)


def _generalized_advantages(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for index in reversed(range(len(rewards))):
        not_done = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * not_done - values[index]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + np.asarray(values, dtype=np.float32)
    return torch.from_numpy(advantages), torch.from_numpy(returns)


def run_ppo(
    environment: Any,
    config: PPOConfig,
    callback: PPOCallback | None = None,
) -> PPOResult:
    started_at = perf_counter()
    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)

    obs_dim = getattr(environment, "observation_size", OBSERVATION_SIZE)
    act_dim = getattr(environment, "action_size", len(Action5))
    action_enum = getattr(environment, "action_enum", None)
    if action_enum is None:
        if act_dim == 9:
            from .room4 import Action4
            action_enum = Action4
        else:
            action_enum = Action5

    policy_net = ActorCriticNetwork(
        observation_dim=obs_dim,
        action_dim=act_dim,
        hidden_dims=config.hidden_dims,
        activation_fn=config.activation_fn,
    )
    optimizer = optim.Adam(policy_net.parameters(), lr=config.alpha)

    metrics: list[PPOTrainingMetric] = []
    training_episodes: list[Any] = []
    action_counts = {action.name: 0 for action in action_enum}

    for episode in range(1, config.episodes + 1):
        try:
            observation = environment.reset(config.seed + episode)
        except TypeError:
            observation = environment.reset()

        states: list[torch.Tensor] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        rewards: list[float] = []
        dones: list[bool] = []
        values: list[float] = []
        total_reward = 0.0
        overtakes = 0
        pipes_passed = 0
        collision = False
        success = False

        is_room5 = hasattr(environment, "snapshot")
        trajectory: list[Any] = []

        for timestep in range(1, config.max_timesteps + 1):
            state_tensor = observation_tensor(observation)
            with torch.no_grad():
                logits, value = policy_net(state_tensor.unsqueeze(0))
                distribution = Categorical(logits=logits)
                action_tensor = distribution.sample()
                log_probability = distribution.log_prob(action_tensor)

            action_idx = int(action_tensor.item())
            action = action_enum(action_idx)
            action_counts[action.name] += 1

            if is_room5:
                before_snap = environment.snapshot()
                transition = environment.step(action)
                total_reward += transition.reward
                overtakes += transition.events.count("overtake")
                collision = collision or "collision" in transition.events
                success = success or "goal_reached" in transition.events
                trajectory.append(
                    Room5EpisodeStep(
                        timestep=timestep,
                        state=observation,
                        action=action,
                        next_state=transition.next_state,
                        reward=float(transition.reward),
                        cumulative_reward=float(total_reward),
                        events=transition.events,
                        before_snapshot=before_snap,
                        after_snapshot=transition.snapshot,
                    )
                )
            else:
                prev_obs = observation
                try:
                    transition = environment.step(action)
                except TypeError:
                    transition = environment.step(observation, action)
                total_reward += transition.reward
                pipes_passed += transition.events.count("pipe_passed")
                collision = collision or "collision" in transition.events
                success = success or "goal_reached" in transition.events
                trajectory.append(
                    Room4EpisodeStep(
                        timestep=timestep,
                        state=prev_obs,
                        action=action,
                        next_state=transition.next_state,
                        reward=float(transition.reward),
                        cumulative_reward=float(total_reward),
                        events=transition.events,
                    )
                )

            states.append(state_tensor)
            actions.append(action_idx)
            old_log_probs.append(float(log_probability.item()))
            rewards.append(float(transition.reward))
            dones.append(bool(transition.done))
            values.append(float(value.item()))

            observation = transition.next_state
            if transition.done:
                break

        if is_room5:
            training_episodes.append(
                Room5EpisodeResult(
                    episode=episode,
                    success=success,
                    timesteps=len(trajectory),
                    total_reward=float(total_reward),
                    overtakes=overtakes,
                    collision=collision,
                    trajectory=trajectory,
                )
            )
        else:
            training_episodes.append(
                Room4EpisodeResult(
                    episode=episode,
                    success=success,
                    timesteps=len(trajectory),
                    total_reward=float(total_reward),
                    pipes_passed=pipes_passed,
                    trajectory=trajectory,
                )
            )

        with torch.no_grad():
            _, bootstrap_value = policy_net(
                observation_tensor(observation).unsqueeze(0)
            )
        last_value = 0.0 if dones[-1] else float(bootstrap_value.item())
        advantages, returns = _generalized_advantages(
            rewards,
            values,
            dones,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            last_value=last_value,
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        state_batch = torch.stack(states)
        action_batch = torch.tensor(actions, dtype=torch.long)
        old_log_batch = torch.tensor(old_log_probs, dtype=torch.float32)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        rollout_size = len(states)
        for _ in range(config.update_epochs):
            permutation = np_rng.permutation(rollout_size)
            for start in range(0, rollout_size, config.mini_batch_size):
                indices = torch.tensor(
                    permutation[start : start + config.mini_batch_size],
                    dtype=torch.long,
                )
                logits, predicted_values = policy_net(state_batch[indices])
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(action_batch[indices])
                probability_ratio = torch.exp(new_log_probs - old_log_batch[indices])

                unclipped = probability_ratio * advantages[indices]
                clipped = torch.clamp(
                    probability_ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon,
                ) * advantages[indices]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(
                    predicted_values, returns[indices]
                )
                entropy = distribution.entropy().mean()
                total_loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        metric = PPOTrainingMetric(
            episode=episode,
            total_reward=float(total_reward),
            timesteps=rollout_size,
            success=success,
            overtakes=overtakes,
            pipes_passed=pipes_passed,
            collision=collision,
            policy_loss=float(np.mean(policy_losses)) if policy_losses else 0.0,
            value_loss=float(np.mean(value_losses)) if value_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            mean_value=float(np.mean(values)) if values else 0.0,
        )
        metrics.append(metric)
        if callback is not None:
            callback(metric, policy_net)

    recent = metrics[-20:]
    return PPOResult(
        policy_net=policy_net,
        metrics=metrics,
        converged=bool(recent and np.mean([metric.success for metric in recent]) >= 0.8),
        episodes_run=config.episodes,
        training_duration_seconds=float(perf_counter() - started_at),
        action_counts=action_counts,
        config=config,
        training_episodes=training_episodes,
    )

