"""Escape-room reinforcement-learning environments and algorithms."""

from .evaluation import EpisodeResult, EpisodeStep, evaluate_policy
from .policy_iteration import (
    PolicyIterationConfig,
    PolicyIterationResult,
    TrainingMetric,
    run_policy_iteration,
)
from .q_learning import (
    QLearningConfig,
    QLearningResult,
    QLearningTrainingMetric,
    run_q_learning,
)
from .room1 import (
    Action,
    Room1Config,
    Room1Environment,
    SlipperyCell,
    default_room1_config,
    generate_random_slippery_cells,
)
from .room2 import (
    Room2Config,
    Room2Environment,
    default_room2_config,
)
from .room3 import (
    Room3Config,
    Room3Environment,
    default_room3_config,
)
from .sarsa import (
    SarsaConfig,
    SarsaResult,
    SarsaTrainingMetric,
    run_sarsa,
)

__all__ = [
    "Action",
    "EpisodeResult",
    "EpisodeStep",
    "PolicyIterationConfig",
    "PolicyIterationResult",
    "QLearningConfig",
    "QLearningResult",
    "QLearningTrainingMetric",
    "Room1Config",
    "Room1Environment",
    "Room2Config",
    "Room2Environment",
    "Room3Config",
    "Room3Environment",
    "SarsaConfig",
    "SarsaResult",
    "SarsaTrainingMetric",
    "SlipperyCell",
    "TrainingMetric",
    "default_room1_config",
    "default_room2_config",
    "default_room3_config",
    "evaluate_policy",
    "generate_random_slippery_cells",
    "run_policy_iteration",
    "run_q_learning",
    "run_sarsa",
]
