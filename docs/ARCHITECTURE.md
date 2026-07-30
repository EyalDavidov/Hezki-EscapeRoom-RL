# System Architecture

## Runtime structure

The system is a single Streamlit application:

1. The browser renders the English dashboard and grid visualization as HTML.
2. A full-width sticky top bar keeps the project brand on the left, Room 1–5 navigation centered from left to right, and Streamlit's Deploy and menu controls aligned on the right.
3. Each room renders one long-form dashboard page. Environment, Training, Testing, and Model panels are always present; four compact sticky tabs share one sidebar row and switch only the sidebar controls client-side, without rerunning or scrolling the main page.
4. High-frequency interactions are fragment-scoped: sidebar values, grid-cell edits, and replay controls rerun only their own UI region. Expensive Train/Test/Reset actions are queued as one-shot requests and trigger a single full app run so the Python server can update the main dashboard.
5. A room-specific vertical control bar appears on the left and contains the relevant environment, reward, hyperparameter, training, testing, and model controls. Each reward uses a compact same-row checkbox to enable or disable its contribution.
6. Rooms 1–3 render every grid cell as an atomic Submit form inside a popover. The form batches Normal/Icy/Wall type, start and goal roles, termination, per-cell reward, and a directional cross of whole-number icy probabilities totaling 100%.
7. Every room's Environment view presents the algorithm contract before the visualization: observation/input, learned output, and available actions.
8. Training and test plots are rendered as fixed-axis Altair charts, preserving tooltips while disabling pan and zoom.
9. Streamlit sends user actions to the server-side Python application.
10. The environments and reinforcement-learning algorithms run in Python and remain independent of the UI.
11. Streamlit receives the results and renders the agents, policies, labeled live training charts, labeled test charts, and configurable-speed episode replays.

Rooms 1–3 are grid environments. Rooms 4 & 5 use continuous and driving environments trained with PPO.

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
|   |-- room4.py                # Continuous Flappy Bird environment
|   |-- room5.py                # One-way multi-lane driving environment
|   |-- policy_iteration.py     # Dynamic Programming algorithm
|   |-- sarsa.py                # On-policy TD control
|   |-- q_learning.py           # Off-policy TD control
|   |-- dqn.py                  # Deep Q-Network for Room 4
|   |-- ppo.py                  # Actor-critic PPO for Room 4 and Room 5
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
- `visualization.py` generates grid, continuous-room, and road visualizations independently of the page controls.
- `streamlit_app.py` owns room navigation, contextual field help, the full cell editor, seeded full-layout generation, runtime control, labeled charts, automatic replay, and model upload/download.
