"""DQN (Deep Q-Network) Algorithm for Room 4 with vectorized ReplayBuffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .room4 import Action4, Room4Environment, State4, ACTION_VELOCITIES


ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "SiLU": nn.SiLU,
}


@dataclass(frozen=True)
class DQNConfig:
    alpha: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 300
    max_timesteps: int = 500
    buffer_capacity: int = 10000
    batch_size: int = 64
    target_update_freq: int = 100
    train_freq: int = 2
    hidden_dims: tuple[int, ...] = (32, 32)
    activation_fn: str = "ReLU"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("Invalid epsilon parameters.")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if self.episodes <= 0 or self.max_timesteps <= 0:
            raise ValueError("episodes and max_timesteps must be positive.")
        if self.buffer_capacity < self.batch_size:
            raise ValueError("buffer_capacity must be at least batch_size.")
        if self.train_freq <= 0:
            raise ValueError("train_freq must be positive.")
        if self.activation_fn not in ACTIVATION_MAP:
            raise ValueError(f"Unknown activation_fn '{self.activation_fn}'. Choice must be one of {sorted(ACTIVATION_MAP)}.")


class DQNNetwork(nn.Module):
    """Multi-Layer Perceptron Q-Network mapping 4D state to 9 Q-values."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 9,
        hidden_dims: tuple[int, ...] = (32, 32),
        activation_fn: str = "ReLU",
    ):
        super().__init__()
        act_cls = ACTIVATION_MAP.get(activation_fn, nn.ReLU)
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_cls())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)



class ReplayBuffer:
    """Fast NumPy-backed Experience Replay Buffer."""

    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.states = np.zeros((capacity, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.rng = np.random.default_rng(seed)

    def push(self, state: State4, action: int, reward: float, next_state: State4, done: bool) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self) -> int:
        return self.size


@dataclass(frozen=True)
class DQNTrainingMetric:
    episode: int
    total_reward: float
    timesteps: int
    success: bool
    epsilon: float
    loss: float
    mean_q_value: float


@dataclass
class DQNResult:
    policy_net: DQNNetwork
    metrics: list[DQNTrainingMetric] = field(default_factory=list)
    converged: bool = False
    episodes_run: int = 0
    training_duration_seconds: float = 0.0
    action_counts: dict[str, int] = field(default_factory=dict)
    config: DQNConfig = field(default_factory=DQNConfig)


DQNCallback = Callable[[DQNTrainingMetric, DQNNetwork], None]


def normalize_batch(states: torch.Tensor) -> torch.Tensor:
    """Vectorized batch state normalization to standard range [-1, 1]."""
    if states.ndim == 1:
        states = states.unsqueeze(0)
    norm = states.clone()
    norm[:, 0] = (norm[:, 0] / 5.0) - 1.0
    norm[:, 1] = (norm[:, 1] / 5.0) - 1.0
    norm[:, 2] = norm[:, 2] / 3.0
    norm[:, 3] = norm[:, 3] / 3.0
    return norm


def normalize_state(state: State4) -> torch.Tensor:
    """Normalize single state vector."""
    t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return normalize_batch(t).squeeze(0)


def select_dqn_action(
    policy_net: DQNNetwork,
    state: State4,
    epsilon: float,
    rng: np.random.Generator,
) -> Action4:
    if rng.random() < epsilon:
        return Action4(int(rng.integers(0, 9)))

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        norm_t = normalize_batch(state_t)
        q_values = policy_net(norm_t)
        action_idx = int(torch.argmax(q_values, dim=1).item())
        return Action4(action_idx)


def run_dqn(
    environment: Room4Environment,
    config: DQNConfig,
    callback: DQNCallback | None = None,
) -> DQNResult:
    training_started_at = perf_counter()

    # Reproducibility
    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)

    policy_net = DQNNetwork(state_dim=4, action_dim=9, hidden_dims=config.hidden_dims, activation_fn=config.activation_fn)
    target_net = DQNNetwork(state_dim=4, action_dim=9, hidden_dims=config.hidden_dims, activation_fn=config.activation_fn)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


    optimizer = optim.Adam(policy_net.parameters(), lr=config.alpha)
    criterion = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=config.buffer_capacity, seed=config.seed)

    metrics: list[DQNTrainingMetric] = []
    epsilon = config.epsilon_start
    total_steps = 0
    action_counts = {action.name: 0 for action in Action4}

    for episode in range(1, config.episodes + 1):
        state = environment.reset()
        total_reward = 0.0
        timesteps = 0
        success = False
        episode_losses: list[float] = []
        episode_q_vals: list[float] = []

        for _ in range(config.max_timesteps):
            action = select_dqn_action(policy_net, state, epsilon, np_rng)
            action_counts[action.name] += 1
            transition = environment.step(state, action)
            next_state = transition.next_state
            reward = transition.reward
            done = transition.done

            total_reward += reward
            timesteps += 1
            total_steps += 1

            replay_buffer.push(state, int(action), reward, next_state, done)

            # Optimizing training frequency (e.g. step update every train_freq steps)
            if len(replay_buffer) >= config.batch_size and total_steps % config.train_freq == 0:
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(config.batch_size)

                # Vectorized state normalization
                b_states_norm = normalize_batch(b_states)
                b_next_states_norm = normalize_batch(b_next_states)

                # Current Q-values
                q_eval = policy_net(b_states_norm).gather(1, b_actions.unsqueeze(1)).squeeze(1)

                # Target Q-values
                with torch.no_grad():
                    max_next_q = target_net(b_next_states_norm).max(dim=1)[0]
                    q_target = b_rewards + (1.0 - b_dones) * config.gamma * max_next_q

                loss = criterion(q_eval, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_losses.append(float(loss.item()))
                episode_q_vals.append(float(q_eval.mean().item()))

            # Sync target network
            if total_steps % config.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                success = "goal_reached" in transition.events
                break

            state = next_state

        # Decay epsilon
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        mean_q = float(np.mean(episode_q_vals)) if episode_q_vals else 0.0

        metric = DQNTrainingMetric(
            episode=episode,
            total_reward=float(total_reward),
            timesteps=timesteps,
            success=success,
            epsilon=float(epsilon),
            loss=mean_loss,
            mean_q_value=mean_q,
        )
        metrics.append(metric)

        if callback is not None:
            callback(metric, policy_net)

    return DQNResult(
        policy_net=policy_net,
        metrics=metrics,
        converged=any(m.success for m in metrics[-20:]) if len(metrics) >= 20 else False,
        episodes_run=config.episodes,
        training_duration_seconds=float(perf_counter() - training_started_at),
        action_counts=action_counts,
        config=config,
    )
