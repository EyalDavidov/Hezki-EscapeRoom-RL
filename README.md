# Hezki Escape Room RL

A reinforcement-learning final project built as a five-room escape game. Each room uses a different environment and learning algorithm.

## Current status

- Task 0: shared system specification — complete.
- Task 1: known-model grid room with Policy Iteration — implemented and tested.
- Rooms 2 and 3 are implemented with SARSA and Q-Learning.
- The application UI and all runtime controls are in English.
- A full-width top bar keeps the logo vertically centered, centers Room 1–5 navigation from left to right, and hosts Streamlit's Deploy and menu controls on the right.
- Each room has a dedicated left control bar for environment settings, rewards, hyperparameters, training, testing, and model management.
- Every Environment page begins with a compact explanation of the room's RL algorithm, observation/input, model output, and available actions.
- Training and test charts use fixed axes: hover tooltips remain available, while pan and zoom are disabled.
- Rooms 1–3 include a clickable 10x10 grid editor. A cell popover controls its type, custom reward, termination behavior, and assignment as the dog start or main goal.
- Icy transition probabilities are edited as whole percentages that must total 100%.
- A single seeded generator creates the start, goal, walls, icy cells, and integer ice distributions together.
- Every control includes contextual help, all charts identify both axes, and training charts are displayed side by side.
- Reward controls accept unrestricted positive or negative values. Every reward has a compact same-row checkbox that enables or disables its effect, while the label and numeric value remain on one line where space permits.
- Every room is a single continuous dashboard page with Environment, Training, Testing, and Model sections. Four compact sticky sidebar buttons share one row, change the visible controls, and smoothly jump to the matching page section.
- Test replays play automatically at a base rate of five timesteps per second with a selectable speed multiplier.

- Room 4 is a continuous Flappy Bird environment trained with PPO. Every action selects discrete X/Y velocity components from `{-1, 0, +1}`; its configurable reward model includes dedicated penalties for HOVER and for any action without rightward movement.
- Room 5 is a configurable 2–6 lane, one-way driving environment trained with PPO. Episodes start on an empty road before traffic arrives progressively, with at least 3 meters between same-lane traffic cars. The agent observes only its current lane and measures the physical gap from its front edge to the nearest car's rear edge. It includes clearance-aware lane-change rewards, collision avoidance, overtaking rewards, live charts, testing, replay, and JSON model artifacts.

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
