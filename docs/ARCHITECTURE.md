# System Architecture

## Runtime structure

The system is a single Streamlit application:

1. The browser renders the English dashboard and grid visualization as HTML.
2. The top navigation bar switches between Rooms 1–4.
3. A room-specific vertical control bar appears on the left and contains the relevant environment, reward, hyperparameter, training, testing, and model controls.
4. Room 1 renders every grid cell as a popover editor for Normal, Icy, and Wall types. Icy cells expose their complete outcome distribution in percentages.
5. Streamlit sends user actions to the server-side Python application.
6. The environments and reinforcement-learning algorithms run in Python and remain independent of the UI.
7. Streamlit receives the results and renders the dog, policies, live training charts, test charts, and episode replays.

Room 1 is fully connected. Rooms 2–4 currently expose navigation-ready placeholders until their environments and algorithms are implemented.

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
|   |-- room1.py                # Grid environment and known transition model
|   |-- policy_iteration.py     # Dynamic Programming algorithm
|   |-- evaluation.py           # Test execution and episode trajectories
|   |-- artifacts.py            # JSON model persistence
|   `-- visualization.py        # HTML grid generation
|-- tests/                      # Automated tests
|-- requirements.txt
`-- streamlit_app.py            # Streamlit application entry point
```

## Separation of responsibilities

- `room1.py` does not import Streamlit and can be tested independently.
- `policy_iteration.py` receives a known environment model and returns state values, a policy, and convergence metrics.
- `evaluation.py` runs tests without modifying the trained policy.
- `artifacts.py` stores the environment, algorithm settings, and trained result together to prevent loading a policy into an incompatible room.
- `visualization.py` generates the grid visualization independently of the page controls.
- `streamlit_app.py` owns room navigation, user input, runtime control, charts, and model upload/download.
