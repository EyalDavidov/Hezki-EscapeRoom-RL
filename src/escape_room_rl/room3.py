"""Room 3: an unknown-model 10x10 grid world used for Q-Learning."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .room1 import (
    Action,
    ACTION_DELTAS,
    SLIP_ACTIONS,
    SLIP_OUTCOMES,
    SUPPORTED_REWARD_EVENTS,
    SlipperyCell,
    State,
    Transition,
)

# Distinct default wall layout for Room 3 (central block with multi-chamber passages)
DEFAULT_ROOM3_WALLS: frozenset[State] = frozenset(
    {
        *((x, 3) for x in (1, 2, 3, 4, 5, 6, 7)),
        *((x, 6) for x in (2, 3, 4, 5, 6, 7, 8)),
    }
)

DEFAULT_ROOM3_REWARDS: dict[str, float] = {
    "step": -0.1,
    "goal_reached": 10.0,
    "entered_slippery": 0.0,
    "slipped": 0.0,
    "blocked_slip": 0.0,
}


@dataclass
class Room3Config:
    width: int = 10
    height: int = 10
    start: State = (0, 0)
    goal: State = (9, 9)
    walls: frozenset[State] = DEFAULT_ROOM3_WALLS
    slippery: dict[State, SlipperyCell] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ROOM3_REWARDS))
    terminal_states: frozenset[State] = field(default_factory=lambda: frozenset({(9, 9)}))
    cell_rewards: dict[State, float] = field(default_factory=dict)


class Room3Environment:
    """Model-free 10x10 grid environment for Room 3."""

    def __init__(self, config: Room3Config):
        self.config = config
        self._validate_config()

    @property
    def start(self) -> State:
        return self.config.start

    @property
    def goal(self) -> State:
        return self.config.goal

    @property
    def terminal_states(self) -> frozenset[State]:
        return self.config.terminal_states

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

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
        return tuple(state for state in self.states if not self.is_terminal(state))

    def in_bounds(self, state: State) -> bool:
        x, y = state
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def is_walkable(self, state: State) -> bool:
        return self.in_bounds(state) and state not in self.config.walls

    def move(self, state: State, action: Action) -> State:
        dx, dy = ACTION_DELTAS[action]
        return state[0] + dx, state[1] + dy

    def legal_actions(self, state: State) -> tuple[Action, ...]:
        if not self.is_walkable(state) or self.is_terminal(state):
            return ()
        return tuple(
            action
            for action in Action
            if self.is_walkable(self.move(state, action))
        )

    def step(
        self, state: State, action: Action, rng: np.random.Generator
    ) -> Transition:
        """Execute a step in the environment without exposing transition probabilities."""
        if action not in self.legal_actions(state):
            raise ValueError(f"Illegal action {action.value!r} from state {state}.")

        target = self.move(state, action)
        slippery = self.config.slippery.get(target)

        if slippery is None:
            return self._make_transition(1.0, target, ("step",), "direct")

        probabilities = [float(getattr(slippery, outcome)) for outcome in SLIP_OUTCOMES]
        outcome_idx = int(rng.choice(len(SLIP_OUTCOMES), p=probabilities))
        outcome = SLIP_OUTCOMES[outcome_idx]

        events = ["step", "entered_slippery"]
        next_state = target
        if outcome != "reach":
            events.append("slipped")
            candidate = self.move(target, SLIP_ACTIONS[outcome])
            if self.is_walkable(candidate):
                next_state = candidate
            else:
                events.append("blocked_slip")

        return self._make_transition(
            probabilities[outcome_idx], next_state, tuple(events), outcome
        )

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
        if self.is_terminal(next_state):
            complete_events.append("termination_reached")
        reward = sum(self.config.rewards.get(event, 0.0) for event in complete_events)
        reward += self.config.cell_rewards.get(next_state, 0.0)
        return Transition(
            probability=float(probability),
            next_state=next_state,
            reward=float(reward),
            done=self.is_terminal(next_state),
            events=tuple(complete_events),
            outcome=outcome,
        )

    def _validate_config(self) -> None:
        if self.config.width != 10 or self.config.height != 10:
            raise ValueError("Room 3 must be a 10x10 grid.")
        if not self.in_bounds(self.config.start) or not self.in_bounds(self.config.goal):
            raise ValueError("Start and goal must be inside the grid.")
        if self.config.start == self.config.goal:
            raise ValueError("Start and goal must be different cells.")
        if not self.is_walkable(self.config.start) or not self.is_walkable(self.config.goal):
            raise ValueError("Start and goal must be walkable.")
        if self.config.goal not in self.terminal_states:
            raise ValueError("The goal cell must be a termination state.")
        if self.config.start in self.terminal_states:
            raise ValueError("The start cell cannot be a termination state.")
        invalid_terminals = [state for state in self.terminal_states if not self.is_walkable(state)]
        if invalid_terminals:
            raise ValueError(f"Termination states must be walkable: {invalid_terminals}")
        invalid_walls = [wall for wall in self.config.walls if not self.in_bounds(wall)]
        if invalid_walls:
            raise ValueError(f"Walls outside the grid: {invalid_walls}")
        for state in self.config.slippery:
            if not self.is_walkable(state):
                raise ValueError(f"Slippery cell {state} is not walkable.")
            if state in {self.config.start, *self.terminal_states}:
                raise ValueError("Start and termination cells cannot be slippery.")
        invalid_cell_rewards = [state for state in self.config.cell_rewards if not self.is_walkable(state)]
        if invalid_cell_rewards:
            raise ValueError(f"Cell rewards must belong to walkable cells: {invalid_cell_rewards}")
        unknown_rewards = set(self.config.rewards) - set(SUPPORTED_REWARD_EVENTS)
        if unknown_rewards:
            raise ValueError(f"Unknown reward events: {sorted(unknown_rewards)}")

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


def default_room3_config() -> Room3Config:
    return Room3Config()
