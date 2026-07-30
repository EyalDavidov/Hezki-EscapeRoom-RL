# Rooms – State and Transition Structure Summary

## Overview
This document briefly outlines the **state representation**, **room configuration**, and **transition (`Transition`)** details for each of the five rooms in the **Hezki‑EscapeRoom‑RL** project. Default values (`DEFAULT_*`) were chosen to drive rapid convergence of the learning algorithms, except for **Room 5**, which has not been fully investigated for an optimal configuration.

---

### Room 1 – Known‑model (Policy Iteration)
- **Source:** `room1.py`
- **State:** `State = tuple[int, int]` – grid coordinates `(x, y)` on a 10×10 board.
- **Config (`Room1Config`):**
  - `width = height = 10`
  - `start = (0, 0)`, `goal = (9, 9)`
  - `walls = DEFAULT_WALLS` – three fixed barrier patterns with gaps.
  - `slippery = DEFAULT_SLIPPERY` – `SlipperyCell` with probabilities `reach=0.8`, others `0.05`.
  - `rewards = DEFAULT_REWARDS` – `step = -0.1`, `goal_reached = 10.0`, other events `0`.
  - `terminal_states = {(9,9)}`.
- **Transition:**
  - Probability derived from `SlipperyCell` when present, otherwise `1.0`.
  - Events include `step`, `entered_slippery`, `slipped` (if applicable), `blocked_slip` (if slide blocked), plus `goal_reached` / `termination_reached`.
  - Reward = sum of `rewards[event]` + any cell‑specific reward.
- **Note:** Default values lead to fast convergence of Policy Iteration and yield the optimal solution.

---

### Room 2 – Model‑free (SARSA)
- **Source:** `room2.py`
- **State:** Same as Room 1.
- **Config (`Room2Config`):**
  - Walls `DEFAULT_ROOM2_WALLS` – zig‑zag pattern across two columns.
  - Slippery cells `DEFAULT_ROOM2_SLIPPERY` – four fixed icy cells with standard `SlipperyCell` probabilities.
  - Rewards identical to Room 1 (`DEFAULT_ROOM2_REWARDS`).
- **Transition:** Mirrors Room 1 logic; outcome probabilities are drawn from the `SLIP_OUTCOMES` vector.
- **Note:** Defaults enable SARSA to converge quickly to an optimal policy.

---

### Room 3 – Model‑free (Q‑Learning)
- **Source:** `room3.py`
- **State:** Grid coordinates on a 10×10 board.
- **Config (`Room3Config`):**
  - Walls `DEFAULT_ROOM3_WALLS` – central block creating multiple chambers.
  - `slippery` default is empty (no ice), providing a clean grid for Q‑Learning.
  - Rewards identical to Room 1 (`DEFAULT_ROOM3_REWARDS`).
- **Transition:** Direct transition (`probability = 1.0`) when no ice; otherwise follows the same `SlipperyCell` logic as other rooms.
- **Note:** The lack of ice gives Q‑Learning a straightforward environment, facilitating rapid optimal convergence.

---

### Room 4 – Experimental Patch
- **Source:** `room4.py`
- **Purpose:** Used by `patch_room4.py` to apply dynamic modifications (walls, slippery cells) at runtime.
- **Structure:** Mirrors the configuration style of Rooms 2/3, allowing on‑the‑fly adjustments before environment instantiation.
- **Note:** Default settings are aligned with Room 2 to preserve optimal convergence behavior.

---

### Room 5 – Complex Model (Research Incomplete)
- **Source:** `room5.py`
- **Config:** Contains a custom layout with more intricate wall patterns, slippery cells, and additional reward events.
- **Note:** We have **not been able to explore sufficiently** to identify a default configuration that guarantees optimal convergence. Further experimentation with hyper‑parameters (wall count, ice cells, reward weights) is required.

---

## Summary
All rooms use `DEFAULT_*` constants for walls, slippery cells, and rewards that were deliberately selected to **drive fast convergence** of their respective learning algorithms (Policy Iteration, SARSA, Q‑Learning). Only **Room 5** lacks a fully researched optimal default configuration and should be the focus of future tuning.
