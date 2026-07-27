# Hezki Escape Room RL

A reinforcement-learning final project built as a four-room escape game. Each room uses a different environment and learning algorithm.

## Current status

- Task 0: shared system specification — complete.
- Task 1: known-model grid room with Policy Iteration — initial implementation complete.
- The application UI and all runtime controls are in English.
- A full-width button navigation bar switches between Rooms 1–4 from right to left, with no radio-button controls.
- Each room has a dedicated left control bar for environment settings, rewards, hyperparameters, training, testing, and model management.
- Room 1 includes a clickable 10x10 grid editor. Every non-terminal cell can be changed to Normal, Icy, or Wall; icy cells expose a five-outcome probability editor.

Rooms 2–4 currently provide navigation-ready placeholder pages and will be connected as their environments and algorithms are implemented.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in a browser.

## Run tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Documentation

- [Project plan](docs/PROJECT_PLAN.md)
- [System architecture](docs/ARCHITECTURE.md)
- [Task 0 system specification](docs/tasks/task_0/SYSTEM_SPECIFICATION.md)
- [Room 1 specification](docs/tasks/task_1/ROOM_1_SPECIFICATION.md)
- [Original project instructions](docs/project-instructions.txt)

## Source images

The original formula images are stored under `assets/images/`.
