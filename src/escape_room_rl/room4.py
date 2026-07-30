"""Room 4: Continuous 10x10m Flappy Bird style environment for DQN."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
import numpy as np

# Continuous state vector: (x, y, vx, vy)
State4 = tuple[float, float, float, float]


class Action4(IntEnum):
    DOWN_LEFT = 0
    LEFT = 1
    UP_LEFT = 2
    DOWN = 3
    HOVER = 4
    UP = 5
    DOWN_RIGHT = 6
    RIGHT = 7
    UP_RIGHT = 8


ACTION_VELOCITIES: dict[Action4, tuple[float, float]] = {
    Action4.DOWN_LEFT: (-1.0, -1.0),
    Action4.LEFT: (-1.0, 0.0),
    Action4.UP_LEFT: (-1.0, 1.0),
    Action4.DOWN: (0.0, -1.0),
    Action4.HOVER: (0.0, 0.0),
    Action4.UP: (0.0, 1.0),
    Action4.DOWN_RIGHT: (1.0, -1.0),
    Action4.RIGHT: (1.0, 0.0),
    Action4.UP_RIGHT: (1.0, 1.0),
}


@dataclass
class PipeObstacle:
    x: float
    width: float = 0.6
    gap_start: float = 3.5  # Bottom of vertical opening gap
    gap_size: float = 3.0   # Height of vertical opening gap

    @property
    def x_min(self) -> float:
        return self.x - self.width / 2.0

    @property
    def x_max(self) -> float:
        return self.x + self.width / 2.0

    @property
    def gap_end(self) -> float:
        return self.gap_start + self.gap_size

    def collides_with(self, px: float, py: float, radius: float = 0.15) -> bool:
        """Check collision between circular bird agent (center px, py) and top/bottom pipes."""
        # Horizontal intersection
        if (px + radius) < self.x_min or (px - radius) > self.x_max:
            return False
        # Inside horizontal span: collides if NOT inside vertical gap
        if (py - radius) < self.gap_start or (py + radius) > self.gap_end:
            return True
        return False


DEFAULT_ROOM4_PIPES: list[PipeObstacle] = [
    PipeObstacle(x=2.5, width=0.6, gap_start=3.5, gap_size=3.0),
    PipeObstacle(x=5.0, width=0.6, gap_start=2.0, gap_size=3.0),
    PipeObstacle(x=7.5, width=0.6, gap_start=4.5, gap_size=3.0),
]

DEFAULT_ROOM4_REWARDS: dict[str, float] = {
    "step": -0.05,
    "progress": 1.0,
    "backward": -0.1,
    "hover": -0.2,
    "non_right": -0.1,
    "pipe_passed": 8.0,
    "goal_reached": 30.0,
    "collision": -30.0,
}


def distribute_pipes_evenly(
    pipes: list[PipeObstacle],
    count: int,
    *,
    first_x: float = 2.0,
    last_x: float = 8.0,
) -> list[PipeObstacle]:
    """Return ``count`` pipes with equal horizontal spacing inside the room.

    Existing pipe width and gap settings are retained in their current order.
    Newly added pipes receive the default obstacle settings. The number of
    pipes is unrestricted as long as it is positive; dense configurations may
    intentionally overlap and create a more difficult environment.
    """
    if count <= 0:
        raise ValueError("Pipe count must be positive.")
    if first_x >= last_x and count > 1:
        raise ValueError("first_x must be smaller than last_x for multiple pipes.")

    templates = list(pipes[:count])
    while len(templates) < count:
        templates.append(PipeObstacle(x=0.0, width=0.6, gap_start=3.0, gap_size=3.0))

    positions = (
        [float((first_x + last_x) / 2.0)]
        if count == 1
        else np.linspace(first_x, last_x, count).tolist()
    )
    return [replace(pipe, x=float(x)) for pipe, x in zip(templates, positions)]


@dataclass
class Room4Config:
    width: float = 10.0
    height: float = 10.0
    dt: float = 0.02  # 50 Hz physics updates
    start: State4 = (0.5, 5.0, 0.0, 0.0)
    goal_x: float = 9.5
    pipes: list[PipeObstacle] = field(default_factory=lambda: list(DEFAULT_ROOM4_PIPES))
    rewards: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ROOM4_REWARDS))
    max_velocity: float = 3.0
    bird_radius: float = 0.15

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Room dimensions must be positive.")
        if self.dt <= 0:
            raise ValueError("Time step dt must be positive.")
        if not (0 <= self.start[0] < self.goal_x <= self.width):
            raise ValueError("Invalid start or goal positions.")
        for p in self.pipes:
            if p.x_min < 0 or p.x_max > self.width:
                raise ValueError(f"Pipe at x={p.x} is out of bounds.")
            if p.gap_start < 0 or p.gap_end > self.height:
                raise ValueError(f"Pipe gap at x={p.x} is out of bounds.")


@dataclass
class Room4Transition:
    next_state: State4
    reward: float
    done: bool
    events: tuple[str, ...]


class Room4Environment:
    """Continuous 10x10m Flappy Bird Environment."""

    observation_size = 4
    action_size = len(Action4)
    action_enum = Action4

    def __init__(self, config: Room4Config | None = None):
        self.config = config or Room4Config()
        self.config.validate()
        self.current_state: State4 = self.config.start
        self.passed_pipes: set[int] = set()

    def reset(self, seed: int | None = None) -> State4:
        self.current_state = self.config.start
        self.passed_pipes = set()
        return self.current_state

    def is_terminal(self, state: State4) -> bool:
        x, y, _, _ = state
        if x >= self.config.goal_x:
            return True
        if x < 0 or x > self.config.width or y <= 0 or y >= self.config.height:
            return True
        for pipe in self.config.pipes:
            if pipe.collides_with(x, y, self.config.bird_radius):
                return True
        return False

    def step(self, arg1: Any, arg2: Any = None) -> Room4Transition:
        if arg2 is None:
            state = self.current_state
            action = Action4(arg1)
        else:
            state = arg1
            action = Action4(arg2)

        x, y, vx, vy = state
        target_vx, target_vy = ACTION_VELOCITIES[action]

        # Velocity update
        new_vx = float(target_vx)
        new_vy = float(target_vy)

        # Position update: s' = s + v * dt
        new_x = x + new_vx * self.config.dt
        new_y = y + new_vy * self.config.dt

        next_state: State4 = (new_x, new_y, new_vx, new_vy)
        self.current_state = next_state
        events: list[str] = ["step"]

        # Check collisions & termination
        done = False
        reward = self.config.rewards.get("step", -0.05)

        # Progress reward
        dx = new_x - x
        if dx > 0:
            reward += dx * self.config.rewards.get("progress", 0.5)
        elif dx < 0:
            events.append("backward_move")
            reward += self.config.rewards.get("backward", 0.0)

        if action is Action4.HOVER:
            events.append("hover")
            reward += self.config.rewards.get("hover", -0.1)

        # Vertical, leftward, and hover actions do not advance toward the goal.
        # Rightward diagonals are considered forward actions and are not penalized.
        if target_vx <= 0.0:
            events.append("non_right_action")
            reward += self.config.rewards.get("non_right", -0.1)

        # Boundary collision
        if new_x < 0 or new_x > self.config.width or new_y <= 0 or new_y >= self.config.height:
            done = True
            events.append("collision")
            reward += self.config.rewards.get("collision", -20.0)

        # Pipe collision & passing check
        if not done:
            for i, pipe in enumerate(self.config.pipes):
                if pipe.collides_with(new_x, new_y, self.config.bird_radius):
                    done = True
                    events.append("collision")
                    reward += self.config.rewards.get("collision", -20.0)
                    break
                # Check if passed pipe in this step
                if x < pipe.x <= new_x and i not in self.passed_pipes:
                    self.passed_pipes.add(i)
                    events.append("pipe_passed")
                    reward += self.config.rewards.get("pipe_passed", 5.0)

        # Goal check
        if not done and new_x >= self.config.goal_x:
            done = True
            events.append("goal_reached")
            reward += self.config.rewards.get("goal_reached", 20.0)

        return Room4Transition(
            next_state=next_state,
            reward=float(reward),
            done=done,
            events=tuple(events),
        )
