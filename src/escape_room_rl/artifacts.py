"""Portable JSON artifacts for Room 1, Room 2, and Room 3 models."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .policy_iteration import (
    PolicyIterationConfig,
    PolicyIterationResult,
    TrainingMetric,
)
from .value_iteration import (
    ValueIterationConfig,
    ValueIterationMetric,
    ValueIterationResult,
)
from .q_learning import (
    QLearningConfig,
    QLearningResult,
    QLearningTrainingMetric,
)
from .room1 import Action, Room1Config, Room1Environment, SlipperyCell, State
from .room2 import Room2Config, Room2Environment
from .room3 import Room3Config, Room3Environment
from .sarsa import (
    SarsaConfig,
    SarsaResult,
    SarsaTrainingMetric,
)

ARTIFACT_VERSION = 1


def state_key(state: State) -> str:
    return f"{state[0]},{state[1]}"


def parse_state(value: str) -> State:
    x, y = value.split(",", maxsplit=1)
    return int(x), int(y)


def state_action_key(state: State, action: Action) -> str:
    return f"{state[0]},{state[1]}:{action.value}"


def parse_state_action(value: str) -> tuple[State, Action]:
    state_str, action_str = value.split(":", maxsplit=1)
    return parse_state(state_str), Action(action_str)


# =====================================================================
# Room 1 (Policy Iteration / Value Iteration)
# =====================================================================

def export_room1_artifact(
    environment: Room1Environment,
    algorithm_config: PolicyIterationConfig | ValueIterationConfig,
    result: PolicyIterationResult | ValueIterationResult,
) -> str:
    is_vi = isinstance(algorithm_config, ValueIterationConfig)
    algorithm_name = "value_iteration" if is_vi else "policy_iteration"

    result_dict: dict[str, Any] = {
        "values": {
            state_key(state): value for state, value in result.values.items()
        },
        "policy": {
            state_key(state): action.value for state, action in result.policy.items()
        },
        "metrics": [asdict(metric) for metric in result.metrics],
        "converged": result.converged,
    }
    if is_vi:
        result_dict["iterations"] = result.iterations
    else:
        result_dict["policy_iterations"] = result.policy_iterations
        result_dict["evaluation_sweeps"] = result.evaluation_sweeps

    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "room": 1,
        "algorithm": algorithm_name,
        "environment": {
            "width": environment.config.width,
            "height": environment.config.height,
            "start": list(environment.start),
            "goal": list(environment.goal),
            "walls": [list(state) for state in sorted(environment.config.walls)],
            "slippery": {
                state_key(state): slippery.as_dict()
                for state, slippery in sorted(environment.config.slippery.items())
            },
            "rewards": environment.config.rewards,
            "terminal_states": [
                list(state) for state in sorted(environment.config.terminal_states)
            ],
            "cell_rewards": {
                state_key(state): reward
                for state, reward in sorted(environment.config.cell_rewards.items())
            },
        },
        "algorithm_config": asdict(algorithm_config),
        "result": result_dict,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_room1_artifact(
    raw_json: str,
) -> tuple[
    Room1Environment,
    PolicyIterationConfig | ValueIterationConfig,
    PolicyIterationResult | ValueIterationResult,
]:
    payload = json.loads(raw_json)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version.")
    if payload.get("room") != 1 or payload.get("algorithm") not in (
        "policy_iteration",
        "value_iteration",
    ):
        raise ValueError(
            "The uploaded artifact is not a Room 1 Policy Iteration or Value Iteration model."
        )

    algorithm_name = payload.get("algorithm", "policy_iteration")

    environment_data = payload["environment"]
    room_config = Room1Config(
        width=int(environment_data["width"]),
        height=int(environment_data["height"]),
        start=tuple(environment_data["start"]),
        goal=tuple(environment_data["goal"]),
        walls=frozenset(tuple(state) for state in environment_data["walls"]),
        slippery={
            parse_state(state): SlipperyCell.from_mapping(probabilities)
            for state, probabilities in environment_data["slippery"].items()
        },
        rewards={
            str(event): float(value)
            for event, value in environment_data["rewards"].items()
        },
        terminal_states=frozenset(
            tuple(state)
            for state in environment_data.get("terminal_states", [environment_data["goal"]])
        ),
        cell_rewards={
            parse_state(state): float(reward)
            for state, reward in environment_data.get("cell_rewards", {}).items()
        },
    )
    environment = Room1Environment(room_config)

    result_data = payload["result"]
    values = {
        parse_state(state): float(value)
        for state, value in result_data["values"].items()
    }
    policy = {
        parse_state(state): Action(action)
        for state, action in result_data["policy"].items()
    }

    if algorithm_name == "value_iteration":
        algorithm_config = ValueIterationConfig(**payload["algorithm_config"])
        result = ValueIterationResult(
            values=values,
            policy=policy,
            metrics=[ValueIterationMetric(**metric) for metric in result_data["metrics"]],
            converged=bool(result_data["converged"]),
            iterations=int(result_data.get("iterations", len(result_data["metrics"]))),
        )
    else:
        algorithm_config = PolicyIterationConfig(**payload["algorithm_config"])
        result = PolicyIterationResult(
            values=values,
            policy=policy,
            metrics=[TrainingMetric(**metric) for metric in result_data["metrics"]],
            converged=bool(result_data["converged"]),
            policy_iterations=int(result_data["policy_iterations"]),
            evaluation_sweeps=int(result_data["evaluation_sweeps"]),
        )

    return environment, algorithm_config, result



# =====================================================================
# Room 2 (SARSA)
# =====================================================================

def export_room2_artifact(
    environment: Room2Environment,
    algorithm_config: SarsaConfig,
    result: SarsaResult,
) -> str:
    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "room": 2,
        "algorithm": "sarsa",
        "environment": {
            "width": environment.config.width,
            "height": environment.config.height,
            "start": list(environment.start),
            "goal": list(environment.goal),
            "walls": [list(state) for state in sorted(environment.config.walls)],
            "slippery": {
                state_key(state): slippery.as_dict()
                for state, slippery in sorted(environment.config.slippery.items())
            },
            "rewards": environment.config.rewards,
            "terminal_states": [
                list(state) for state in sorted(environment.config.terminal_states)
            ],
            "cell_rewards": {
                state_key(state): reward
                for state, reward in sorted(environment.config.cell_rewards.items())
            },
        },
        "algorithm_config": asdict(algorithm_config),
        "result": {
            "q_table": {
                state_action_key(sa[0], sa[1]): value for sa, value in result.q_table.items()
            },
            "values": {
                state_key(state): value for state, value in result.values.items()
            },
            "policy": {
                state_key(state): action.value for state, action in result.policy.items()
            },
            "metrics": [asdict(metric) for metric in result.metrics],
            "converged": result.converged,
            "episodes_run": result.episodes_run,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_room2_artifact(
    raw_json: str,
) -> tuple[Room2Environment, SarsaConfig, SarsaResult]:
    payload = json.loads(raw_json)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version.")
    if payload.get("room") != 2 or payload.get("algorithm") != "sarsa":
        raise ValueError("The uploaded artifact is not a Room 2 SARSA model.")

    environment_data = payload["environment"]
    room_config = Room2Config(
        width=int(environment_data["width"]),
        height=int(environment_data["height"]),
        start=tuple(environment_data["start"]),
        goal=tuple(environment_data["goal"]),
        walls=frozenset(tuple(state) for state in environment_data["walls"]),
        slippery={
            parse_state(state): SlipperyCell.from_mapping(probabilities)
            for state, probabilities in environment_data["slippery"].items()
        },
        rewards={
            str(event): float(value)
            for event, value in environment_data["rewards"].items()
        },
        terminal_states=frozenset(
            tuple(state)
            for state in environment_data.get("terminal_states", [environment_data["goal"]])
        ),
        cell_rewards={
            parse_state(state): float(reward)
            for state, reward in environment_data.get("cell_rewards", {}).items()
        },
    )
    environment = Room2Environment(room_config)
    algorithm_config = SarsaConfig(**payload["algorithm_config"])
    result_data = payload["result"]

    q_table = {
        parse_state_action(sa): float(value)
        for sa, value in result_data["q_table"].items()
    }
    values = {
        parse_state(state): float(value)
        for state, value in result_data["values"].items()
    }
    policy = {
        parse_state(state): Action(action)
        for state, action in result_data["policy"].items()
    }

    result = SarsaResult(
        q_table=q_table,
        values=values,
        policy=policy,
        metrics=[SarsaTrainingMetric(**metric) for metric in result_data["metrics"]],
        converged=bool(result_data["converged"]),
        episodes_run=int(result_data["episodes_run"]),
    )
    return environment, algorithm_config, result


# =====================================================================
# Room 3 (Q-Learning)
# =====================================================================

def export_room3_artifact(
    environment: Room3Environment,
    algorithm_config: QLearningConfig,
    result: QLearningResult,
) -> str:
    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "room": 3,
        "algorithm": "q_learning",
        "environment": {
            "width": environment.config.width,
            "height": environment.config.height,
            "start": list(environment.start),
            "goal": list(environment.goal),
            "walls": [list(state) for state in sorted(environment.config.walls)],
            "slippery": {
                state_key(state): slippery.as_dict()
                for state, slippery in sorted(environment.config.slippery.items())
            },
            "rewards": environment.config.rewards,
            "terminal_states": [
                list(state) for state in sorted(environment.config.terminal_states)
            ],
            "cell_rewards": {
                state_key(state): reward
                for state, reward in sorted(environment.config.cell_rewards.items())
            },
        },
        "algorithm_config": asdict(algorithm_config),
        "result": {
            "q_table": {
                state_action_key(sa[0], sa[1]): value for sa, value in result.q_table.items()
            },
            "values": {
                state_key(state): value for state, value in result.values.items()
            },
            "policy": {
                state_key(state): action.value for state, action in result.policy.items()
            },
            "metrics": [asdict(metric) for metric in result.metrics],
            "converged": result.converged,
            "episodes_run": result.episodes_run,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_room3_artifact(
    raw_json: str,
) -> tuple[Room3Environment, QLearningConfig, QLearningResult]:
    payload = json.loads(raw_json)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version.")
    if payload.get("room") != 3 or payload.get("algorithm") != "q_learning":
        raise ValueError("The uploaded artifact is not a Room 3 Q-Learning model.")

    environment_data = payload["environment"]
    room_config = Room3Config(
        width=int(environment_data["width"]),
        height=int(environment_data["height"]),
        start=tuple(environment_data["start"]),
        goal=tuple(environment_data["goal"]),
        walls=frozenset(tuple(state) for state in environment_data["walls"]),
        slippery={
            parse_state(state): SlipperyCell.from_mapping(probabilities)
            for state, probabilities in environment_data["slippery"].items()
        },
        rewards={
            str(event): float(value)
            for event, value in environment_data["rewards"].items()
        },
        terminal_states=frozenset(
            tuple(state)
            for state in environment_data.get("terminal_states", [environment_data["goal"]])
        ),
        cell_rewards={
            parse_state(state): float(reward)
            for state, reward in environment_data.get("cell_rewards", {}).items()
        },
    )
    environment = Room3Environment(room_config)
    algorithm_config = QLearningConfig(**payload["algorithm_config"])
    result_data = payload["result"]

    q_table = {
        parse_state_action(sa): float(value)
        for sa, value in result_data["q_table"].items()
    }
    values = {
        parse_state(state): float(value)
        for state, value in result_data["values"].items()
    }
    policy = {
        parse_state(state): Action(action)
        for state, action in result_data["policy"].items()
    }

    result = QLearningResult(
        q_table=q_table,
        values=values,
        policy=policy,
        metrics=[QLearningTrainingMetric(**metric) for metric in result_data["metrics"]],
        converged=bool(result_data["converged"]),
        episodes_run=int(result_data["episodes_run"]),
    )
    return environment, algorithm_config, result
