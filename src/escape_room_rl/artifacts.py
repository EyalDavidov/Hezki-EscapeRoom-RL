"""Portable JSON artifacts for Room 1 policies and their training history."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .policy_iteration import (
    PolicyIterationConfig,
    PolicyIterationResult,
    TrainingMetric,
)
from .room1 import Action, Room1Config, Room1Environment, SlipperyCell, State

ARTIFACT_VERSION = 1


def state_key(state: State) -> str:
    return f"{state[0]},{state[1]}"


def parse_state(value: str) -> State:
    x, y = value.split(",", maxsplit=1)
    return int(x), int(y)


def export_room1_artifact(
    environment: Room1Environment,
    algorithm_config: PolicyIterationConfig,
    result: PolicyIterationResult,
) -> str:
    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "room": 1,
        "algorithm": "policy_iteration",
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
        },
        "algorithm_config": asdict(algorithm_config),
        "result": {
            "values": {
                state_key(state): value for state, value in result.values.items()
            },
            "policy": {
                state_key(state): action.value for state, action in result.policy.items()
            },
            "metrics": [asdict(metric) for metric in result.metrics],
            "converged": result.converged,
            "policy_iterations": result.policy_iterations,
            "evaluation_sweeps": result.evaluation_sweeps,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_room1_artifact(
    raw_json: str,
) -> tuple[Room1Environment, PolicyIterationConfig, PolicyIterationResult]:
    payload = json.loads(raw_json)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported artifact version.")
    if payload.get("room") != 1 or payload.get("algorithm") != "policy_iteration":
        raise ValueError("The uploaded artifact is not a Room 1 Policy Iteration model.")

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
    )
    environment = Room1Environment(room_config)
    algorithm_config = PolicyIterationConfig(**payload["algorithm_config"])
    result_data = payload["result"]
    result = PolicyIterationResult(
        values={
            parse_state(state): float(value)
            for state, value in result_data["values"].items()
        },
        policy={
            parse_state(state): Action(action)
            for state, action in result_data["policy"].items()
        },
        metrics=[TrainingMetric(**metric) for metric in result_data["metrics"]],
        converged=bool(result_data["converged"]),
        policy_iterations=int(result_data["policy_iterations"]),
        evaluation_sweeps=int(result_data["evaluation_sweeps"]),
    )
    return environment, algorithm_config, result
