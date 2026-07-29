# Hezki Escape Room RL

A reinforcement-learning final project built as a five-room escape game. Each room uses a different environment and learning algorithm.

## Current status

- Task 0: shared system specification — complete.
- Task 1: known-model grid room with Policy Iteration — implemented and tested.
- Rooms 2 and 3 are implemented with SARSA and Q-Learning.
- The application UI and all runtime controls are in English.
- A full-width button navigation bar switches between Rooms 1–5 from right to left, with no radio-button controls.
- Each room has a dedicated left control bar for environment settings, rewards, hyperparameters, training, testing, and model management.
- Rooms 1–3 include a clickable 10x10 grid editor. A cell popover controls its type, custom reward, termination behavior, and assignment as the dog start or main goal.
- Icy transition probabilities are edited as whole percentages that must total 100%.
- A single seeded generator creates the start, goal, walls, icy cells, and integer ice distributions together.
- Every control includes contextual help, all charts identify both axes, and training charts are displayed side by side.
- Test replays play automatically at a base rate of five timesteps per second with a selectable speed multiplier.

- Room 4 is a continuous Flappy Bird environment trained with DQN.
- Room 5 is a configurable 2–6 lane, one-way driving environment trained with PPO. It includes adjustable forward vision, same-direction traffic, collision avoidance, overtaking rewards, live charts, testing, replay, and JSON model artifacts.

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
- [Room 5 specification](docs/tasks/task_8/ROOM_5_SPECIFICATION.md)
- [Original project instructions](docs/project-instructions.txt)

## Source images

The original formula images are stored under `assets/images/`.
