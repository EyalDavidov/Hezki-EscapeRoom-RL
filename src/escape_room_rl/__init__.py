"""Escape-room reinforcement-learning environments and algorithms."""

from .evaluation import (
    EpisodeResult,
    EpisodeStep,
    Room4EpisodeResult,
    Room4EpisodeStep,
    evaluate_policy,
    evaluate_room4_dqn,
)
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
from .room4 import (
    Action4,
    PipeObstacle,
    Room4Config,
    Room4Environment,
)
from .dqn import (
    DQNConfig,
    DQNNetwork,
    DQNResult,
    DQNTrainingMetric,
    run_dqn,
)
from .sarsa import (
    SarsaConfig,
    SarsaResult,
    SarsaTrainingMetric,
    run_sarsa,
)

from .value_iteration import (
    ValueIterationConfig,
    ValueIterationMetric,
    ValueIterationResult,
    run_value_iteration,
)

__all__ = [
    "Action",
    "Action4",
    "DQNConfig",
    "DQNNetwork",
    "DQNResult",
    "DQNTrainingMetric",
    "EpisodeResult",
    "EpisodeStep",
    "PipeObstacle",
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
    "Room4Config",
    "Room4Environment",
    "Room4EpisodeResult",
    "Room4EpisodeStep",
    "SarsaConfig",
    "SarsaResult",
    "SarsaTrainingMetric",
    "SlipperyCell",
    "TrainingMetric",
    "ValueIterationConfig",
    "ValueIterationMetric",
    "ValueIterationResult",
    "default_room1_config",
    "default_room2_config",
    "default_room3_config",
    "evaluate_policy",
    "evaluate_room4_dqn",
    "generate_random_slippery_cells",
    "run_policy_iteration",
    "run_dqn",
    "run_q_learning",
    "run_sarsa",
    "run_value_iteration",
]

