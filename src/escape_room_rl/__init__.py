"""Escape-room reinforcement-learning environments and algorithms."""

from .evaluation import (
    EpisodeResult,
    EpisodeStep,
    Room4EpisodeResult,
    Room4EpisodeStep,
    evaluate_policy,
    evaluate_room4_dqn,
    evaluate_room4_ppo,
    Room5EpisodeResult,
    Room5EpisodeStep,
    evaluate_room5_ppo,
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
    DEFAULT_SLIPPERY,
    DEFAULT_WALLS,
    Action,
    Room1Config,
    Room1Environment,
    SlipperyCell,
    default_room1_config,
    generate_random_slippery_cells,
)
from .room2 import (
    DEFAULT_ROOM2_SLIPPERY,
    DEFAULT_ROOM2_WALLS,
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
    distribute_pipes_evenly,
)
from .room5 import (
    Action5,
    RoadSnapshot,
    Room5Config,
    Room5Environment,
    TrafficCar,
)
from .dqn import (
    DQNConfig,
    DQNNetwork,
    DQNResult,
    DQNTrainingMetric,
    run_dqn,
)
from .ppo import (
    ActorCriticNetwork,
    PPOConfig,
    PPOResult,
    PPOTrainingMetric,
    run_ppo,
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
    "Action5",
    "ActorCriticNetwork",
    "DEFAULT_ROOM2_SLIPPERY",
    "DEFAULT_ROOM2_WALLS",
    "DEFAULT_SLIPPERY",
    "DEFAULT_WALLS",
    "DQNConfig",
    "DQNNetwork",
    "DQNResult",
    "DQNTrainingMetric",
    "EpisodeResult",
    "EpisodeStep",
    "PipeObstacle",
    "PolicyIterationConfig",
    "PolicyIterationResult",
    "PPOConfig",
    "PPOResult",
    "PPOTrainingMetric",
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
    "Room5Config",
    "Room5Environment",
    "Room5EpisodeResult",
    "Room5EpisodeStep",
    "RoadSnapshot",
    "SarsaConfig",
    "SarsaResult",
    "SarsaTrainingMetric",
    "SlipperyCell",
    "TrainingMetric",
    "TrafficCar",
    "ValueIterationConfig",
    "ValueIterationMetric",
    "ValueIterationResult",
    "default_room1_config",
    "default_room2_config",
    "default_room3_config",
    "evaluate_policy",
    "evaluate_room4_dqn",
    "evaluate_room4_ppo",
    "evaluate_room5_ppo",
    "generate_random_slippery_cells",
    "distribute_pipes_evenly",
    "run_policy_iteration",
    "run_ppo",
    "run_dqn",
    "run_q_learning",
    "run_sarsa",
    "run_value_iteration",
]

