"""Room 1: a known-model stochastic 10x10 grid world."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose
from typing import Iterable, Mapping

import numpy as np

State = tuple[int, int]


class Action(str, Enum):
    """Actions use the visual convention: x grows left and y grows up."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


ACTION_DELTAS: dict[Action, State] = {
    Action.UP: (0, 1),
    Action.DOWN: (0, -1),
    Action.LEFT: (1, 0),
    Action.RIGHT: (-1, 0),
}

SLIP_OUTCOMES = ("reach", "up", "down", "right", "left")
SLIP_ACTIONS: dict[str, Action] = {
    "up": Action.UP,
    "down": Action.DOWN,
    "right": Action.RIGHT,
    "left": Action.LEFT,
}

SUPPORTED_REWARD_EVENTS: dict[str, str] = {
    "step": "Every timestep",
    "goal_reached": "Goal reached",
    "entered_slippery": "Attempted entry into an icy cell",
    "slipped": "A slip occurred",
    "blocked_slip": "A slide was blocked by a wall or boundary",
}


@dataclass(frozen=True)
class SlipperyCell:
    """Absolute outcome probabilities for an attempted entry into an icy cell."""

    reach: float = 0.8
    up: float = 0.05
    down: float = 0.05
    right: float = 0.05
    left: float = 0.05

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(value < 0.0 or value > 1.0 for value in values.values()):
            raise ValueError("Slip probabilities must be between 0 and 1.")
        if not isclose(sum(values.values()), 1.0, abs_tol=1e-9):
            raise ValueError("Slip probabilities must sum to 1.")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in SLIP_OUTCOMES}

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "SlipperyCell":
        return cls(**{name: float(values[name]) for name in SLIP_OUTCOMES})


DEFAULT_WALLS: frozenset[State] = frozenset(
    {
        # Three fixed barriers with gaps create alternate routes across the room.
        *((2, y) for y in (0, 1, 2, 4, 5, 6, 7)),
        *((5, y) for y in (2, 3, 4, 5, 7, 8, 9)),
        *((7, y) for y in (0, 1, 3, 4, 5, 6)),
    }
)

DEFAULT_REWARDS: dict[str, float] = {
    "step": -0.1,
    "goal_reached": 10.0,
    "entered_slippery": 0.0,
    "slipped": 0.0,
    "blocked_slip": 0.0,
}


@dataclass
class Room1Config:
    width: int = 10
    height: int = 10
    start: State = (0, 0)
    goal: State = (9, 9)
    walls: frozenset[State] = DEFAULT_WALLS
    slippery: dict[State, SlipperyCell] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARDS))


@dataclass(frozen=True)
class Transition:
    probability: float
    next_state: State
    reward: float
    done: bool
    events: tuple[str, ...]
    outcome: str


class Room1Environment:
    """Known transition and reward model used by Policy Iteration."""

    def __init__(self, config: Room1Config):
        self.config = config
        self._validate_config()

    @property
    def start(self) -> State:
        return self.config.start

    @property
    def goal(self) -> State:
        return self.config.goal

    @property
    def states(self) -> tuple[State, ...]:
        return tuple(
            (x, y)
            for x in range(self.config.width)
            for y in range(self.config.height)
            if (x, y) not in self.config.walls
        )

    @property
    def non_terminal_states(self) -> tuple[State, ...]:
        return tuple(state for state in self.states if state != self.goal)

    def in_bounds(self, state: State) -> bool:
        x, y = state
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def is_walkable(self, state: State) -> bool:
        return self.in_bounds(state) and state not in self.config.walls

    def move(self, state: State, action: Action) -> State:
        dx, dy = ACTION_DELTAS[action]
        return state[0] + dx, state[1] + dy

    def legal_actions(self, state: State) -> tuple[Action, ...]:
        if not self.is_walkable(state) or state == self.goal:
            return ()
        return tuple(
            action
            for action in Action
            if self.is_walkable(self.move(state, action))
        )

    def transition_model(self, state: State, action: Action) -> tuple[Transition, ...]:
        if action not in self.legal_actions(state):
            raise ValueError(f"Illegal action {action.value!r} from state {state}.")

        target = self.move(state, action)
        slippery = self.config.slippery.get(target)
        if slippery is None:
            return (self._make_transition(1.0, target, ("step",), "direct"),)

        transitions: list[Transition] = []
        for outcome, probability in slippery.as_dict().items():
            if probability <= 0.0:
                continue
            events = ["step", "entered_slippery"]
            next_state = target
            if outcome != "reach":
                events.append("slipped")
                candidate = self.move(target, SLIP_ACTIONS[outcome])
                if self.is_walkable(candidate):
                    next_state = candidate
                else:
                    # A blocked slide ends on the icy target cell.
                    events.append("blocked_slip")
            transitions.append(
                self._make_transition(probability, next_state, tuple(events), outcome)
            )
        return tuple(transitions)

    def step(
        self, state: State, action: Action, rng: np.random.Generator
    ) -> Transition:
        transitions = self.transition_model(state, action)
        probabilities = np.asarray(
            [transition.probability for transition in transitions], dtype=float
        )
        index = int(rng.choice(len(transitions), p=probabilities))
        return transitions[index]

    def _make_transition(
        self,
        probability: float,
        next_state: State,
        events: tuple[str, ...],
        outcome: str,
    ) -> Transition:
        complete_events = list(events)
        if next_state == self.goal:
            complete_events.append("goal_reached")
        reward = sum(self.config.rewards.get(event, 0.0) for event in complete_events)
        return Transition(
            probability=float(probability),
            next_state=next_state,
            reward=float(reward),
            done=next_state == self.goal,
            events=tuple(complete_events),
            outcome=outcome,
        )

    def _validate_config(self) -> None:
        if self.config.width != 10 or self.config.height != 10:
            raise ValueError("Room 1 must be a 10x10 grid.")
        if self.config.start != (0, 0) or self.config.goal != (9, 9):
            raise ValueError("Room 1 start and goal must be (0, 0) and (9, 9).")
        if not self.is_walkable(self.config.start) or not self.is_walkable(
            self.config.goal
        ):
            raise ValueError("Start and goal must be walkable.")
        invalid_walls = [wall for wall in self.config.walls if not self.in_bounds(wall)]
        if invalid_walls:
            raise ValueError(f"Walls outside the grid: {invalid_walls}")
        for state in self.config.slippery:
            if not self.is_walkable(state):
                raise ValueError(f"Slippery cell {state} is not walkable.")
            if state in (self.config.start, self.config.goal):
                raise ValueError("Start and goal cannot be slippery.")
        unknown_rewards = set(self.config.rewards) - set(SUPPORTED_REWARD_EVENTS)
        if unknown_rewards:
            raise ValueError(f"Unknown reward events: {sorted(unknown_rewards)}")
        unreachable = [state for state in self.non_terminal_states if not self.legal_actions(state)]
        if unreachable:
            raise ValueError(f"Walkable states without legal actions: {unreachable}")
        frontier = [self.start]
        visited = {self.start}
        while frontier:
            state = frontier.pop()
            for action in self.legal_actions(state):
                neighbor = self.move(state, action)
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        if self.goal not in visited:
            raise ValueError("The goal is unreachable from the start cell.")


def default_room1_config() -> Room1Config:
    return Room1Config()


def slippery_candidates(config: Room1Config) -> tuple[State, ...]:
    return tuple(
        (x, y)
        for x in range(config.width)
        for y in range(config.height)
        if (x, y) not in config.walls and (x, y) not in (config.start, config.goal)
    )


def generate_random_slippery_cells(
    config: Room1Config, count: int, seed: int
) -> dict[State, SlipperyCell]:
    """Generate reproducible icy cells and normalized five-way probabilities."""

    candidates = slippery_candidates(config)
    if count < 0 or count > len(candidates):
        raise ValueError(f"count must be between 0 and {len(candidates)}")
    rng = np.random.default_rng(seed)
    if count == 0:
        return {}
    indexes = rng.choice(len(candidates), size=count, replace=False)
    result: dict[State, SlipperyCell] = {}
    for index in np.atleast_1d(indexes):
        state = candidates[int(index)]
        probabilities = rng.dirichlet(np.ones(len(SLIP_OUTCOMES)))
        result[state] = SlipperyCell(
            **dict(zip(SLIP_OUTCOMES, probabilities, strict=True))
        )
    return result
