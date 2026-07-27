"""Escape-room reinforcement-learning environments and algorithms."""

from .evaluation import EpisodeResult, EpisodeStep, evaluate_policy
from .policy_iteration import (
    PolicyIterationConfig,
    PolicyIterationResult,
    TrainingMetric,
    run_policy_iteration,
)
from .room1 import (
    Action,
    Room1Config,
    Room1Environment,
    SlipperyCell,
    default_room1_config,
    generate_random_slippery_cells,
)

__all__ = [
    "Action",
    "EpisodeResult",
    "EpisodeStep",
    "PolicyIterationConfig",
    "PolicyIterationResult",
    "Room1Config",
    "Room1Environment",
    "SlipperyCell",
    "TrainingMetric",
    "default_room1_config",
    "evaluate_policy",
    "generate_random_slippery_cells",
    "run_policy_iteration",
]
