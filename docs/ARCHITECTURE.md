# System Architecture

## Runtime structure

The system is a single Streamlit application:

1. The browser renders the English dashboard and grid visualization as HTML.
2. A full-width sticky button bar switches between Rooms 1–4 in right-to-left visual order and highlights the active room.
3. A room-specific vertical control bar appears on the left and contains the relevant environment, reward, hyperparameter, training, testing, and model controls.
4. Rooms 1–3 render every grid cell as a popover editor for Normal, Icy, and Wall types. The same editor assigns the start, main goal, extra termination states, and per-cell rewards. Icy cells expose a whole-number percentage distribution totaling 100%.
5. Streamlit sends user actions to the server-side Python application.
6. The environments and reinforcement-learning algorithms run in Python and remain independent of the UI.
7. Streamlit receives the results and renders the dog, policies, labeled live training charts, labeled test charts, and automatic episode replays at five base timesteps per second.

Rooms 1–3 are fully connected. Room 4 remains a navigation-ready placeholder until its continuous environment and algorithm are implemented.

## Directory structure

```text
Hezki-EscapeRoom-RL/
|-- assets/
|   `-- images/                 # Original formula images
|-- data/
|   |-- models/                 # Local model files, excluded from commits
|   `-- runs/                   # Local run results and server logs
|-- docs/
|   |-- tasks/
|   |   |-- task_0/             # Shared system specification
|   |   `-- task_1/             # Room 1 specification
|   |-- ARCHITECTURE.md
|   |-- PROJECT_PLAN.md
|   `-- project-instructions.txt
|-- src/escape_room_rl/
|   |-- room1.py                # Policy Iteration room and full-grid generator
|   |-- room2.py                # SARSA grid environment
|   |-- room3.py                # Q-Learning grid environment
|   |-- policy_iteration.py     # Dynamic Programming algorithm
|   |-- sarsa.py                # On-policy TD control
|   |-- q_learning.py           # Off-policy TD control
|   |-- evaluation.py           # Test execution and episode trajectories
|   |-- artifacts.py            # JSON model persistence
|   `-- visualization.py        # HTML grid generation
|-- tests/                      # Automated tests
|-- requirements.txt
`-- streamlit_app.py            # Streamlit application entry point
```

## Separation of responsibilities

- The room environments do not import Streamlit and can be tested independently. They own validation for dynamic starts, goals, termination states, cell rewards, walls, and ice.
- `policy_iteration.py` receives a known environment model and returns state values, a policy, and convergence metrics.
- `evaluation.py` runs tests without modifying the trained policy.
- `artifacts.py` stores the environment, algorithm settings, and trained result together to prevent loading a policy into an incompatible room.
- `visualization.py` generates the grid visualization independently of the page controls.
- `streamlit_app.py` owns room navigation, contextual field help, the full cell editor, seeded full-layout generation, runtime control, labeled charts, automatic replay, and model upload/download.
