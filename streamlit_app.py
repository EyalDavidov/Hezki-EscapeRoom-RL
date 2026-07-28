"""Streamlit dashboard for the Escape Room RL project."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from escape_room_rl.artifacts import (  # noqa: E402
    export_room1_artifact,
    import_room1_artifact,
    export_room2_artifact,
    import_room2_artifact,
    export_room3_artifact,
    import_room3_artifact,
)
from escape_room_rl.evaluation import evaluate_policy  # noqa: E402
from escape_room_rl.policy_iteration import (  # noqa: E402
    PolicyIterationConfig,
    run_policy_iteration,
)
from escape_room_rl.sarsa import (  # noqa: E402
    SarsaConfig,
    run_sarsa,
)
from escape_room_rl.q_learning import (  # noqa: E402
    QLearningConfig,
    run_q_learning,
)
from escape_room_rl.room1 import (  # noqa: E402
    DEFAULT_REWARDS,
    DEFAULT_WALLS,
    SLIP_OUTCOMES,
    SUPPORTED_REWARD_EVENTS,
    Room1Config,
    Room1Environment,
    SlipperyCell,
    generate_random_grid_layout,
)
from escape_room_rl.room2 import (  # noqa: E402
    DEFAULT_ROOM2_REWARDS,
    DEFAULT_ROOM2_WALLS,
    Room2Config,
    Room2Environment,
)
from escape_room_rl.room3 import (  # noqa: E402
    DEFAULT_ROOM3_REWARDS,
    DEFAULT_ROOM3_WALLS,
    Room3Config,
    Room3Environment,
)
from escape_room_rl.visualization import render_grid_html  # noqa: E402


st.set_page_config(
    page_title="Escape Room RL",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {
        display: none !important;
      }
      [data-testid="stAppViewContainer"] {
        overflow: auto !important;
      }
      [data-testid="stMain"] {
        overflow: visible !important;
      }
      [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        padding-top: 0rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        overflow: visible !important;
      }
      [data-testid="stSidebar"] {
        min-width: 340px;
        max-width: 340px;
      }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 { letter-spacing: -0.02em; }

      /* Sticky Top Navigation Bar */
      div:has(> .st-key-main_top_nav) {
        position: sticky !important;
        top: 0 !important;
        z-index: 9999 !important;
        overflow: visible !important;
      }
      .st-key-main_top_nav {
        position: sticky !important;
        top: 0 !important;
        z-index: 9999 !important;
        margin-left: -2.5rem !important;
        margin-right: -2.5rem !important;
        margin-top: 0 !important;
        padding: 0.85rem 2.5rem !important;
        background: #121620 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(16px) !important;
        margin-bottom: 1.75rem !important;
      }
      .st-key-main_top_nav [data-testid="stHorizontalBlock"] {
        gap: 1rem !important;
        align-items: center !important;
      }
      .st-key-main_top_nav button {
        width: 100% !important;
        min-height: 3.1rem !important;
        border-radius: 0.65rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease-in-out !important;
      }
      .st-key-main_top_nav button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
      }
      .st-key-main_top_nav button[data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
      }
      .st-key-main_top_nav button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.14) !important;
        color: #ffffff !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-1px) !important;
      }
      .room-header {
        padding: 0.4rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
      }
      .transition-banner {
        padding: 0.85rem 1.25rem;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(16, 185, 129, 0.2));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 0.75rem;
        margin-bottom: 1.25rem;
        color: #ecfdf5;
      }
      .legend-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
      .legend-item { padding: 0.25rem 0.65rem; border-radius: 999px; background: rgba(128,128,128,0.12); }

      /* Grid Popover Editor styling */
      .st-key-grid_editor {
        width: 100%;
        max-width: 720px;
        border: 3px solid #263238;
        border-radius: 8px;
        overflow: hidden;
        background: #f7f9fb;
        padding-top: 1rem;
        box-sizing: border-box;
      }
      .st-key-grid_editor [data-testid="stHorizontalBlock"] {
        column-gap: 0 !important;
        margin-top: -1rem !important;
      }
      .st-key-grid_editor [data-testid="stElementContainer"] {
        margin: 0 !important;
      }
      .st-key-grid_editor [data-testid="stPopover"] button {
        position: relative;
        width: 100%;
        min-height: 0;
        height: auto;
        aspect-ratio: 1;
        padding: 0;
        border: 1px solid #aeb8c2;
        border-radius: 0;
        box-shadow: none;
        background: #f7f9fb;
      }
      .st-key-grid_editor [data-testid="stPopover"] button:hover {
        filter: brightness(0.94);
        z-index: 1;
        box-shadow: inset 0 0 0 2px #ff4b4b;
      }
      .st-key-grid_editor [data-testid="stPopover"] button p { font-size: 0; }
      .st-key-grid_editor [data-testid="stPopover"] button::before {
        font-size: clamp(18px, 2.2vw, 30px);
        line-height: 1;
      }
      .st-key-grid_editor [data-testid="stPopover"] button::after {
        position: absolute;
        right: 3px;
        bottom: 1px;
        font-size: 9px;
        line-height: 1;
        color: #263238;
        opacity: 0.75;
      }
      /* Hide popover dropdown chevron arrows in grid cells */
      .st-key-grid_editor [data-testid="stPopover"] button svg,
      .st-key-grid_editor [data-testid="stPopover"] button [data-testid="stIconMaterial"],
      .st-key-grid_editor [data-testid="stPopover"] button [data-testid="stBaseButton-icon"],
      .st-key-grid_editor [data-testid="stPopover"] button div[data-testid="stPopoverToggleIcon"],
      .st-key-grid_editor [data-testid="stPopover"] button i {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
      }
      .st-key-grid_editor [data-testid="column"] { min-width: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "active_room": "Room 1",
        # Room 1
        "room1_walls": set(DEFAULT_WALLS),
        "room1_slippery": {},
        "room1_start": (0, 0),
        "room1_goal": (9, 9),
        "room1_terminal_states": {(9, 9)},
        "room1_cell_rewards": {},
        "room1_probability_errors": set(),
        "room1_reward_values": dict(DEFAULT_REWARDS),
        "room1_reward_enabled": {"step", "goal_reached"},
        "room1_result": None,
        "room1_result_environment": None,
        "room1_algorithm_config": None,
        "room1_test_results": None,
        "room1_editor_nonce": 0,
        "room1_training_controls": {
            "gamma": 0.95,
            "theta": 1e-6,
            "max_policy_iterations": 100,
            "max_evaluation_sweeps": 10_000,
            "seed": 42,
            "live_update_every": 5,
        },
        "room1_test_controls": {
            "episodes": 100,
            "max_timesteps": 250,
            "seed": 123,
        },
        "room1_random_controls": {"wall_count": 20, "icy_count": 8, "seed": 42},
        # Room 2 (SARSA)
        "room2_walls": set(DEFAULT_ROOM2_WALLS),
        "room2_slippery": {},
        "room2_start": (0, 0),
        "room2_goal": (9, 9),
        "room2_terminal_states": {(9, 9)},
        "room2_cell_rewards": {},
        "room2_probability_errors": set(),
        "room2_reward_values": dict(DEFAULT_ROOM2_REWARDS),
        "room2_reward_enabled": {"step", "goal_reached"},
        "room2_result": None,
        "room2_result_environment": None,
        "room2_algorithm_config": None,
        "room2_test_results": None,
        "room2_editor_nonce": 0,
        "room2_training_controls": {
            "alpha": 0.2,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.99,
            "episodes": 300,
            "max_timesteps": 200,
            "seed": 42,
            "live_update_every": 10,
        },
        "room2_test_controls": {
            "episodes": 100,
            "max_timesteps": 200,
            "seed": 123,
        },
        "room2_random_controls": {"wall_count": 16, "icy_count": 6, "seed": 42},
        # Room 3 (Q-Learning)
        "room3_walls": set(DEFAULT_ROOM3_WALLS),
        "room3_slippery": {},
        "room3_start": (0, 0),
        "room3_goal": (9, 9),
        "room3_terminal_states": {(9, 9)},
        "room3_cell_rewards": {},
        "room3_probability_errors": set(),
        "room3_reward_values": dict(DEFAULT_ROOM3_REWARDS),
        "room3_reward_enabled": {"step", "goal_reached"},
        "room3_result": None,
        "room3_result_environment": None,
        "room3_algorithm_config": None,
        "room3_test_results": None,
        "room3_editor_nonce": 0,
        "room3_training_controls": {
            "alpha": 0.2,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.99,
            "episodes": 300,
            "max_timesteps": 200,
            "seed": 42,
            "live_update_every": 10,
        },
        "room3_test_controls": {
            "episodes": 100,
            "max_timesteps": 200,
            "seed": 123,
        },
        "room3_random_controls": {"wall_count": 14, "icy_count": 6, "seed": 42},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for room_num, wall_default, icy_default in ((1, 20, 8), (2, 16, 6), (3, 14, 6)):
        random_key = f"room{room_num}_random_controls"
        if "wall_count" not in st.session_state[random_key]:
            old = st.session_state[random_key]
            st.session_state[random_key] = {
                "wall_count": wall_default,
                "icy_count": int(old.get("count", icy_default)),
                "seed": int(old.get("seed", 42)),
            }


def state_label(state: tuple[int, int]) -> str:
    return f"({state[0]}, {state[1]})"


def metrics_dataframe(result) -> pd.DataFrame:
    return pd.DataFrame([asdict(metric) for metric in result.metrics])


def test_dataframe(results) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": episode.episode,
                "success": episode.success,
                "timesteps": episode.timesteps,
                "total_reward": episode.total_reward,
                "icy_entries": episode.slippery_entries,
                "slips": episode.slipped_count,
            }
            for episode in results
        ]
    )


def render_room_navigation() -> str:
    rooms = ["Room 1", "Room 2", "Room 3", "Room 4"]
    active_room = st.session_state.active_room
    with st.container(key="main_top_nav"):
        columns = st.columns(4, gap="small")
        for column, room in zip(columns, reversed(rooms), strict=True):
            with column:
                selected = st.button(
                    room,
                    key=f"navigate_{room.lower().replace(' ', '_')}",
                    type="primary" if room == active_room else "secondary",
                    use_container_width=True,
                )
                if selected and room != active_room:
                    st.session_state.active_room = room
                    st.rerun()
    return st.session_state.active_room


# =====================================================================
# COMMON ENVIRONMENT BUILDERS & GRID EDITOR
# =====================================================================

def room_prefix(room_num: int) -> str:
    return f"room{room_num}"


def invalidate_room_model(room_num: int) -> None:
    p = room_prefix(room_num)
    st.session_state[f"{p}_result"] = None
    st.session_state[f"{p}_result_environment"] = None
    st.session_state[f"{p}_algorithm_config"] = None
    st.session_state[f"{p}_test_results"] = None


def current_cell_type(room_num: int, state: tuple[int, int]) -> str:
    p = room_prefix(room_num)
    if state in st.session_state[f"{p}_walls"]:
        return "Wall"
    if state in st.session_state[f"{p}_slippery"]:
        return "Icy"
    return "Normal"


def apply_cell_type(state: tuple[int, int], widget_key: str, room_num: int) -> None:
    p = room_prefix(room_num)
    selected_type = st.session_state[widget_key]
    if state == st.session_state[f"{p}_start"] or state in st.session_state[f"{p}_terminal_states"]:
        return
    previous_type = current_cell_type(room_num, state)
    if selected_type == previous_type:
        return

    st.session_state[f"{p}_walls"].discard(state)
    st.session_state[f"{p}_slippery"].pop(state, None)
    st.session_state[f"{p}_probability_errors"].discard(state)
    if selected_type == "Wall":
        st.session_state[f"{p}_cell_rewards"].pop(state, None)
        st.session_state[f"{p}_walls"].add(state)
    elif selected_type == "Icy":
        st.session_state[f"{p}_slippery"][state] = SlipperyCell()
    invalidate_room_model(room_num)


def set_cell_as_start(room_num: int, state: tuple[int, int]) -> None:
    p = room_prefix(room_num)
    if state == st.session_state[f"{p}_goal"]:
        return
    st.session_state[f"{p}_walls"].discard(state)
    st.session_state[f"{p}_slippery"].pop(state, None)
    st.session_state[f"{p}_terminal_states"].discard(state)
    st.session_state[f"{p}_probability_errors"].discard(state)
    st.session_state[f"{p}_start"] = state
    st.session_state[f"{p}_editor_nonce"] += 1
    invalidate_room_model(room_num)


def set_cell_as_goal(room_num: int, state: tuple[int, int]) -> None:
    p = room_prefix(room_num)
    if state == st.session_state[f"{p}_start"]:
        return
    previous_goal = st.session_state[f"{p}_goal"]
    st.session_state[f"{p}_walls"].discard(state)
    st.session_state[f"{p}_slippery"].pop(state, None)
    st.session_state[f"{p}_probability_errors"].discard(state)
    st.session_state[f"{p}_terminal_states"].discard(previous_goal)
    st.session_state[f"{p}_terminal_states"].add(state)
    st.session_state[f"{p}_goal"] = state
    st.session_state[f"{p}_editor_nonce"] += 1
    invalidate_room_model(room_num)


def update_cell_termination(room_num: int, state: tuple[int, int], widget_key: str) -> None:
    p = room_prefix(room_num)
    enabled = bool(st.session_state[widget_key])
    if state in {st.session_state[f"{p}_start"], st.session_state[f"{p}_goal"]}:
        return
    if enabled:
        st.session_state[f"{p}_walls"].discard(state)
        st.session_state[f"{p}_slippery"].pop(state, None)
        st.session_state[f"{p}_probability_errors"].discard(state)
        st.session_state[f"{p}_terminal_states"].add(state)
    else:
        st.session_state[f"{p}_terminal_states"].discard(state)
    invalidate_room_model(room_num)


def update_cell_reward(room_num: int, state: tuple[int, int], enabled: bool, value: float) -> None:
    p = room_prefix(room_num)
    previous = st.session_state[f"{p}_cell_rewards"].get(state)
    if enabled:
        st.session_state[f"{p}_cell_rewards"][state] = float(value)
    else:
        st.session_state[f"{p}_cell_rewards"].pop(state, None)
    current = st.session_state[f"{p}_cell_rewards"].get(state)
    if current != previous:
        invalidate_room_model(room_num)


def build_environment(room_num: int) -> Any:
    p = room_prefix(room_num)
    walls = frozenset(st.session_state[f"{p}_walls"])
    slippery = dict(st.session_state[f"{p}_slippery"])
    rewards = dict(st.session_state[f"{p}_reward_values"])
    common = {
        "start": st.session_state[f"{p}_start"],
        "goal": st.session_state[f"{p}_goal"],
        "walls": walls,
        "slippery": slippery,
        "rewards": rewards,
        "terminal_states": frozenset(st.session_state[f"{p}_terminal_states"]),
        "cell_rewards": dict(st.session_state[f"{p}_cell_rewards"]),
    }
    if room_num == 1:
        return Room1Environment(Room1Config(**common))
    elif room_num == 2:
        return Room2Environment(Room2Config(**common))
    else:
        return Room3Environment(Room3Config(**common))


def room_configuration_error(room_num: int) -> str | None:
    p = room_prefix(room_num)
    if st.session_state[f"{p}_probability_errors"]:
        cells = ", ".join(
            state_label(state)
            for state in sorted(st.session_state[f"{p}_probability_errors"])
        )
        return f"Icy-cell probabilities must total 100%: {cells}."
    try:
        build_environment(room_num)
    except ValueError as exc:
        return str(exc)
    return None


def render_grid_editor(room_num: int) -> None:
    """Render a full per-cell editor for layout, roles, rewards and ice."""
    p = room_prefix(room_num)
    outcome_labels = {
        "reach": "Reach the icy cell (no slide)",
        "up": "Slide up",
        "down": "Slide down",
        "right": "Slide right",
        "left": "Slide left",
    }
    icon_by_type = {"Normal": "⬜", "Icy": "❄️", "Wall": "🧱"}
    nonce = st.session_state[f"{p}_editor_nonce"]
    start = st.session_state[f"{p}_start"]
    goal = st.session_state[f"{p}_goal"]
    terminals = st.session_state[f"{p}_terminal_states"]
    cell_rewards = st.session_state[f"{p}_cell_rewards"]

    cell_styles: list[str] = []
    for y in reversed(range(10)):
        for x in reversed(range(10)):
            state = (x, y)
            cell_type = current_cell_type(room_num, state)
            icon = ""
            background = "#f7f9fb"
            coordinate_color = "#263238"
            outline = "none"
            if state == start:
                icon = "🐕"
                background = "#fff3cd"
                outline = "inset 0 0 0 3px #43a047"
            elif state == goal:
                icon = "🚪"
                outline = "inset 0 0 0 3px #f9a825"
            elif state in terminals:
                icon = "🛑"
                background = "#ffe4e6"
                outline = "inset 0 0 0 3px #e11d48"
            elif cell_type == "Wall":
                icon = "🧱"
                background = "#455a64"
                coordinate_color = "white"
            elif cell_type == "Icy":
                icon = "❄️"
                background = "#dff6ff"
            elif state in cell_rewards:
                icon = "🎁"
                background = "#f3e8ff"

            selector = f'.st-key-{p}_cell_{x}_{y} [data-testid="stPopover"] button'
            cell_styles.extend(
                [
                    f'{selector} {{ background: {background} !important; box-shadow: {outline}; }}',
                    f'{selector}::before {{ content: "{icon}"; }}',
                    f'{selector}::after {{ content: "{x},{y}"; color: {coordinate_color}; }}',
                ]
            )
    st.markdown(f"<style>{''.join(cell_styles)}</style>", unsafe_allow_html=True)

    with st.container(key="grid_editor"):
        for y in reversed(range(10)):
            columns = st.columns(10, gap=None)
            for column, x in zip(columns, reversed(range(10)), strict=True):
                state = (x, y)
                if state == start:
                    cell_label = f"🐕 {x},{y}"
                elif state == goal:
                    cell_label = f"🚪 {x},{y}"
                elif state in terminals:
                    cell_label = f"🛑 {x},{y}"
                elif state in cell_rewards and current_cell_type(room_num, state) == "Normal":
                    cell_label = f"🎁 {x},{y}"
                else:
                    cell_label = f"{icon_by_type[current_cell_type(room_num, state)]} {x},{y}"

                with column:
                    with st.container(key=f"{p}_cell_{x}_{y}"):
                        with st.popover(cell_label, use_container_width=True):
                            st.markdown(f"**Cell {state_label(state)}**")
                            type_key = f"cell_type_{p}_{nonce}_{x}_{y}"
                            cell_type = current_cell_type(room_num, state)
                            selected_type = st.radio(
                                "Cell type",
                                options=["Normal", "Icy", "Wall"],
                                index=["Normal", "Icy", "Wall"].index(cell_type),
                                horizontal=True,
                                key=type_key,
                                disabled=state == start or state in terminals,
                                help=(
                                    "Controls whether the cell is walkable normally, uses a stochastic "
                                    "ice transition, or blocks movement as a wall. Start and termination "
                                    "cells must remain walkable."
                                ),
                                on_change=apply_cell_type,
                                args=(state, type_key, room_num),
                            )

                            st.caption("Cell roles")
                            start_col, goal_col = st.columns(2)
                            with start_col:
                                if st.button(
                                    "🐕 Set as start",
                                    key=f"set_start_{p}_{nonce}_{x}_{y}",
                                    disabled=state == start or state == goal,
                                    help="Moves the dog's episode start position to this cell.",
                                    use_container_width=True,
                                ):
                                    set_cell_as_start(room_num, state)
                                    st.rerun()
                            with goal_col:
                                if st.button(
                                    "🚪 Set as goal",
                                    key=f"set_goal_{p}_{nonce}_{x}_{y}",
                                    disabled=state == goal or state == start,
                                    help="Moves the main success target here and makes it a termination state.",
                                    use_container_width=True,
                                ):
                                    set_cell_as_goal(room_num, state)
                                    st.rerun()

                            termination_key = f"termination_{p}_{nonce}_{x}_{y}"
                            st.checkbox(
                                "Termination state",
                                value=state in terminals,
                                key=termination_key,
                                disabled=state in {start, goal} or selected_type == "Wall",
                                help=(
                                    "Ends the episode immediately when the agent enters this cell. "
                                    "The main goal is always a termination state."
                                ),
                                on_change=update_cell_termination,
                                args=(room_num, state, termination_key),
                            )

                            reward_enabled = state in cell_rewards
                            reward_toggle = st.checkbox(
                                "Custom reward on entry",
                                value=reward_enabled,
                                key=f"cell_reward_enabled_{p}_{nonce}_{x}_{y}",
                                disabled=selected_type == "Wall",
                                help=(
                                    "Adds the configured reward or penalty whenever the agent enters "
                                    "this specific cell, in addition to global event rewards."
                                ),
                            )
                            reward_value = st.number_input(
                                "Cell reward value",
                                value=float(cell_rewards.get(state, 0.0)),
                                step=0.1,
                                format="%.3f",
                                key=f"cell_reward_value_{p}_{nonce}_{x}_{y}",
                                disabled=not reward_toggle or selected_type == "Wall",
                                help="Positive values attract the agent; negative values discourage entry.",
                            )
                            update_cell_reward(
                                room_num,
                                state,
                                reward_toggle and selected_type != "Wall",
                                float(reward_value),
                            )

                            if selected_type != "Icy":
                                continue

                            current = st.session_state[f"{p}_slippery"][state]
                            st.caption(
                                "Set the complete outcome distribution. The five values must total 100%."
                            )
                            percentages: dict[str, int] = {}
                            for outcome in SLIP_OUTCOMES:
                                percentages[outcome] = st.number_input(
                                    f"{outcome_labels[outcome]} (%)",
                                    min_value=0,
                                    max_value=100,
                                    value=int(round(getattr(current, outcome) * 100)),
                                    step=1,
                                    format="%d",
                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_{outcome}",
                                    help="Integer percentage assigned to this outcome when entering the icy cell.",
                                )
                            total = sum(percentages.values())
                            st.metric("Probability total", f"{total}%")
                            if total != 100:
                                st.session_state[f"{p}_probability_errors"].add(state)
                                st.error("The probabilities must total exactly 100%.")
                                continue

                            st.session_state[f"{p}_probability_errors"].discard(state)
                            updated = SlipperyCell(
                                **{outcome: percentage / 100.0 for outcome, percentage in percentages.items()}
                            )
                            if updated != current:
                                st.session_state[f"{p}_slippery"][state] = updated
                                invalidate_room_model(room_num)


def render_environment_page(room_num: int) -> None:
    p = room_prefix(room_num)
    algo_names = {1: "Policy Iteration", 2: "SARSA", 3: "Q-Learning"}
    st.markdown(
        f'<div class="room-header"><h1>Room {room_num} — Environment ({algo_names[room_num]})</h1>'
        '<p>Click any grid cell below to configure it as normal, icy, or a wall.</p></div>',
        unsafe_allow_html=True,
    )
    walls = st.session_state[f"{p}_walls"]
    slippery = st.session_state[f"{p}_slippery"]
    terminals = st.session_state[f"{p}_terminal_states"]
    cell_rewards = st.session_state[f"{p}_cell_rewards"]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Grid cells", 100)
    m2.metric("Walkable states", 100 - len(walls))
    m3.metric("Walls", len(walls))
    m4.metric("Icy cells", len(slippery))
    m5.metric("Terminations", len(terminals))
    m6.metric("Cell rewards", len(cell_rewards))
    st.markdown(
        '<div class="legend-row">'
        '<span class="legend-item">🐕 Agent / start</span>'
        '<span class="legend-item">🚪 Goal</span>'
        '<span class="legend-item">🧱 Wall</span>'
        '<span class="legend-item">❄️ Icy cell</span>'
        '<span class="legend-item">🛑 Termination</span>'
        '<span class="legend-item">🎁 Custom reward</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Select any cell to edit it. Current start: {state_label(st.session_state[f'{p}_start'])}; "
        f"main goal: {state_label(st.session_state[f'{p}_goal'])}."
    )
    render_grid_editor(room_num)

    config_err = room_configuration_error(room_num)
    if config_err is not None:
        st.error(f"Grid configuration is not trainable: {config_err}")
    else:
        st.success("The grid configuration is valid and ready for training.")

    if slippery:
        rows = []
        for (x, y), probabilities in sorted(slippery.items()):
            rows.append({"x": x, "y": y, **probabilities.as_dict()})
        st.subheader("Icy-cell transition probabilities")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_training_page(room_num: int, requests: dict[str, bool]) -> None:
    p = room_prefix(room_num)
    algo_names = {1: "Policy Iteration", 2: "SARSA", 3: "Q-Learning"}
    st.markdown(
        f'<div class="room-header"><h1>Room {room_num} — Training ({algo_names[room_num]})</h1></div>',
        unsafe_allow_html=True,
    )
    if requests["reset"]:
        invalidate_room_model(room_num)
        st.rerun()

    if requests["train"]:
        env = build_environment(room_num)
        controls = st.session_state[f"{p}_training_controls"]
        status_slot = st.empty()
        live_chart_col_1, live_chart_col_2 = st.columns(2)
        with live_chart_col_1:
            chart_slot_1 = st.empty()
        with live_chart_col_2:
            chart_slot_2 = st.empty()
        live_rows: list[dict] = []

        if room_num == 1:
            alg_config = PolicyIterationConfig(
                gamma=controls["gamma"],
                theta=controls["theta"],
                max_policy_iterations=controls["max_policy_iterations"],
                max_evaluation_sweeps=controls["max_evaluation_sweeps"],
                seed=controls["seed"],
            )

            def callback_pi(metric, _values, _policy) -> None:
                live_rows.append(asdict(metric))
                if metric.global_step % controls["live_update_every"] != 0 and metric.phase != "improvement":
                    return
                df = pd.DataFrame(live_rows)
                status_slot.info(f"Policy Iteration {metric.policy_iteration} • sweep {metric.evaluation_sweep} • delta={metric.delta:.3e}")
                eval_df = df[df["phase"] == "evaluation"]
                if not eval_df.empty:
                    chart_slot_1.line_chart(eval_df.set_index("global_step")[["delta"]], x_label="Sweep", y_label="Delta")
                    chart_slot_2.line_chart(eval_df.set_index("global_step")[["mean_value"]], x_label="Sweep", y_label="Mean V(s)")

            with st.spinner("Computing Policy Iteration..."):
                res = run_policy_iteration(env, alg_config, callback=callback_pi)
        elif room_num == 2:
            alg_config = SarsaConfig(
                alpha=controls["alpha"],
                gamma=controls["gamma"],
                epsilon_start=controls["epsilon_start"],
                epsilon_min=controls["epsilon_min"],
                epsilon_decay=controls["epsilon_decay"],
                episodes=controls["episodes"],
                max_timesteps=controls["max_timesteps"],
                seed=controls["seed"],
            )

            def callback_sarsa(metric, _values, _policy) -> None:
                live_rows.append(asdict(metric))
                if metric.episode % controls["live_update_every"] != 0:
                    return
                df = pd.DataFrame(live_rows)
                status_slot.info(f"SARSA Episode {metric.episode}/{controls['episodes']} • reward={metric.total_reward:.2f} • eps={metric.epsilon:.3f}")
                chart_slot_1.line_chart(df.set_index("episode")[["total_reward"]], x_label="Episode", y_label="Total Reward")
                chart_slot_2.line_chart(df.set_index("episode")[["epsilon"]], x_label="Episode", y_label="Epsilon")

            with st.spinner("Training SARSA agent..."):
                res = run_sarsa(env, alg_config, callback=callback_sarsa)
        else:
            alg_config = QLearningConfig(
                alpha=controls["alpha"],
                gamma=controls["gamma"],
                epsilon_start=controls["epsilon_start"],
                epsilon_min=controls["epsilon_min"],
                epsilon_decay=controls["epsilon_decay"],
                episodes=controls["episodes"],
                max_timesteps=controls["max_timesteps"],
                seed=controls["seed"],
            )

            def callback_ql(metric, _values, _policy) -> None:
                live_rows.append(asdict(metric))
                if metric.episode % controls["live_update_every"] != 0:
                    return
                df = pd.DataFrame(live_rows)
                status_slot.info(f"Q-Learning Episode {metric.episode}/{controls['episodes']} • reward={metric.total_reward:.2f} • eps={metric.epsilon:.3f}")
                chart_slot_1.line_chart(df.set_index("episode")[["total_reward"]], x_label="Episode", y_label="Total Reward")
                chart_slot_2.line_chart(df.set_index("episode")[["epsilon"]], x_label="Episode", y_label="Epsilon")

            with st.spinner("Training Q-Learning agent..."):
                res = run_q_learning(env, alg_config, callback=callback_ql)

        st.session_state[f"{p}_result"] = res
        st.session_state[f"{p}_result_environment"] = env
        st.session_state[f"{p}_algorithm_config"] = alg_config
        st.session_state[f"{p}_test_results"] = None
        status_slot.success(f"Training completed successfully!")

    res = st.session_state[f"{p}_result"]
    env = st.session_state[f"{p}_result_environment"]
    if res is None or env is None:
        st.info("Configure training parameters in the left bar and click ▶ Train.")
        return

    m1, m2, m3 = st.columns(3)
    if room_num == 1:
        m1.metric("Converged", "Yes" if res.converged else "No")
        m2.metric("Policy Iterations", res.policy_iterations)
        m3.metric("Evaluation Sweeps", res.evaluation_sweeps)
    else:
        m1.metric("Episodes Run", res.episodes_run)
        m2.metric("Converged", "Yes" if res.converged else "No")
        m3.metric("Icy Cells", len(env.config.slippery))

    st.markdown(render_grid_html(env, policy=res.policy, values=res.values), unsafe_allow_html=True)
    frame = metrics_dataframe(res)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Value / Delta Convergence")
        y_col = "delta" if room_num == 1 else "total_reward"
        x_col = "global_step" if room_num == 1 else "episode"
        st.line_chart(
            frame.set_index(x_col)[[y_col]],
            x_label="Bellman sweep" if room_num == 1 else "Training episode",
            y_label="Maximum value delta" if room_num == 1 else "Episode total reward",
        )
    with c2:
        st.subheader("Policy Changes / Epsilon Decay")
        y_col2 = "policy_changes" if room_num == 1 else "epsilon"
        x_col2 = "policy_iteration" if room_num == 1 else "episode"
        st.line_chart(
            frame.set_index(x_col2)[[y_col2]],
            x_label="Policy iteration" if room_num == 1 else "Training episode",
            y_label="Changed actions" if room_num == 1 else "Exploration epsilon",
        )

    if room_num == 1 and res.converged:
        st.markdown(
            '<div class="transition-banner">🎉 <b>Mission 3 Unlocked!</b> Room 1 solved! '
            'Click Room 2 in the top menu to proceed to Room 2!</div>',
            unsafe_allow_html=True,
        )
    elif room_num == 2 and res.converged:
        st.markdown(
            '<div class="transition-banner">🎉 <b>Mission 5 Unlocked!</b> Room 2 solved! '
            'Click Room 3 in the top menu to proceed to Room 3!</div>',
            unsafe_allow_html=True,
        )


def render_test_page(room_num: int, run_requested: bool) -> None:
    p = room_prefix(room_num)
    st.markdown(
        f'<div class="room-header"><h1>Room {room_num} — Testing</h1>'
        '<p>Evaluate the trained policy without updating values or Q-tables.</p></div>',
        unsafe_allow_html=True,
    )
    res = st.session_state[f"{p}_result"]
    env = st.session_state[f"{p}_result_environment"]
    if res is None or env is None:
        st.info("Train or load a model before running test episodes.")
        return

    if run_requested:
        controls = st.session_state[f"{p}_test_controls"]
        with st.spinner("Running test episodes..."):
            st.session_state[f"{p}_test_results"] = evaluate_policy(
                env, res.policy, controls["episodes"], controls["max_timesteps"], controls["seed"]
            )

    test_results = st.session_state[f"{p}_test_results"]
    if not test_results:
        st.info("Set test parameters in the left bar and click 🧪 Run test.")
        return

    frame = test_dataframe(test_results)
    successful = frame[frame["success"]]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Success rate", f"{frame['success'].mean():.1%}")
    m2.metric("Mean timesteps", f"{frame['timesteps'].mean():.2f}")
    m3.metric("Mean reward", f"{frame['total_reward'].mean():.3f}")
    m4.metric("Mean slips", f"{frame['slips'].mean():.2f}")

    if not successful.empty:
        st.caption(
            f"Successful episodes — median: {successful['timesteps'].median():.1f}, "
            f"minimum: {successful['timesteps'].min()}, "
            f"maximum: {successful['timesteps'].max()} timesteps."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Timesteps by episode")
        st.bar_chart(
            frame.set_index("episode")[["timesteps"]],
            x_label="Test episode",
            y_label="Timesteps used",
        )
    with c2:
        st.subheader("Reward by episode")
        st.line_chart(
            frame.set_index("episode")[["total_reward"]],
            x_label="Test episode",
            y_label="Total reward",
        )

    st.subheader("Detailed Episode Results")
    st.dataframe(frame, hide_index=True, use_container_width=True)

    st.subheader("Episode Replay & Animation")
    selected_number = st.selectbox(
        "Select episode to replay",
        options=[episode.episode for episode in test_results],
        key=f"{p}_replay_ep_select",
        help="Choose which recorded test episode will be animated.",
    )
    selected_episode = next(
        episode for episode in test_results if episode.episode == selected_number
    )
    play_col, speed_col, info_col = st.columns([1, 1, 3])
    with play_col:
        play_requested = st.button(
            "▶ Play replay",
            type="primary",
            use_container_width=True,
            key=f"{p}_play_replay",
            help="Automatically animates the episode from start to finish.",
        )
    with speed_col:
        playback_speed = st.selectbox(
            "Playback speed",
            options=[0.5, 1.0, 2.0, 4.0],
            index=1,
            format_func=lambda value: f"{value:g}×",
            key=f"{p}_replay_speed",
            help="Base speed is 5 timesteps per second. The multiplier changes that rate.",
        )
    with info_col:
        st.caption(
            f"{len(selected_episode.trajectory)} recorded timesteps • "
            f"{5 * playback_speed:g} steps/second"
        )

    replay_status = st.empty()
    replay_grid = st.empty()
    replay_status.info(f"Ready at start cell {state_label(env.start)}.")
    replay_grid.markdown(
        render_grid_html(env, agent_state=env.start, policy=res.policy, values=res.values),
        unsafe_allow_html=True,
    )
    if play_requested:
        delay = 1.0 / (5.0 * playback_speed)
        for index, step in enumerate(selected_episode.trajectory, start=1):
            replay_status.info(
                f"Timestep {index}/{len(selected_episode.trajectory)} • "
                f"action `{step.action.value}` • outcome `{step.outcome}` • "
                f"reward {step.reward:.3f} • cumulative {step.cumulative_reward:.3f}"
            )
            replay_grid.markdown(
                render_grid_html(
                    env,
                    agent_state=step.next_state,
                    policy=res.policy,
                    values=res.values,
                ),
                unsafe_allow_html=True,
            )
            time.sleep(delay)
        replay_status.success(
            "Replay complete — goal reached."
            if selected_episode.success
            else "Replay complete — episode ended without reaching the main goal."
        )


def render_models_page(room_num: int) -> None:
    p = room_prefix(room_num)
    algo_names = {1: "Policy Iteration", 2: "SARSA", 3: "Q-Learning"}
    st.markdown(
        f'<div class="room-header"><h1>Room {room_num} — Models ({algo_names[room_num]})</h1>'
        '<p>Download or upload a portable JSON model artifact.</p></div>',
        unsafe_allow_html=True,
    )
    res = st.session_state[f"{p}_result"]
    env = st.session_state[f"{p}_result_environment"]
    config = st.session_state[f"{p}_algorithm_config"]
    if res is None or env is None or config is None:
        st.info("No trained model is currently loaded.")
        return

    st.success(f"A Room {room_num} {algo_names[room_num]} model is loaded.")
    summary = {
        "room": room_num,
        "algorithm": algo_names[room_num],
        "converged": res.converged,
        "icy_cells": len(env.config.slippery),
        **asdict(config),
    }
    st.json(summary)


# =====================================================================
# CONTROL SIDEBARS AND MAIN ROOM ENTRY POINTS
# =====================================================================

def render_room_controls(room_num: int) -> tuple[str, dict[str, bool], bool]:
    p = room_prefix(room_num)
    algo_names = {1: "Policy Iteration", 2: "SARSA", 3: "Q-Learning"}
    st.sidebar.title(f"Room {room_num} Controls")
    st.sidebar.caption(f"Model • {algo_names[room_num]}")
    section = st.sidebar.radio(
        "Control section",
        options=["Environment", "Training", "Testing", "Models"],
        key=f"{p}_control_section",
        help="Choose which group of room controls is shown in the left sidebar.",
    )
    st.sidebar.divider()

    # Environment controls in sidebar
    defaults_walls = {1: DEFAULT_WALLS, 2: DEFAULT_ROOM2_WALLS, 3: DEFAULT_ROOM3_WALLS}
    random_controls = st.session_state[f"{p}_random_controls"]

    prob_err = room_configuration_error(room_num)
    requests = {"train": False, "reset": False}
    run_test = False

    if section == "Environment":
        st.sidebar.subheader("Grid layout")
        if st.sidebar.button(
            "Reset grid to default",
            key=f"{p}_reset_grid",
            use_container_width=True,
            help="Restores the room's original walls, start, goal, and removes ice and custom cell rewards.",
        ):
            st.session_state[f"{p}_walls"] = set(defaults_walls[room_num])
            st.session_state[f"{p}_slippery"] = {}
            st.session_state[f"{p}_start"] = (0, 0)
            st.session_state[f"{p}_goal"] = (9, 9)
            st.session_state[f"{p}_terminal_states"] = {(9, 9)}
            st.session_state[f"{p}_cell_rewards"] = {}
            st.session_state[f"{p}_probability_errors"] = set()
            st.session_state[f"{p}_editor_nonce"] += 1
            invalidate_room_model(room_num)
            st.rerun()

        st.sidebar.subheader("Random full-grid generator")
        random_walls = st.sidebar.number_input(
            "Number of walls",
            0,
            80,
            int(random_controls["wall_count"]),
            key=f"{p}_random_walls",
            help="Number of impassable wall cells placed by the generator.",
        )
        random_ice = st.sidebar.number_input(
            "Number of icy cells",
            0,
            98,
            int(random_controls["icy_count"]),
            key=f"{p}_random_ice",
            help="Number of stochastic icy cells, each with an integer probability distribution.",
        )
        random_seed = st.sidebar.number_input(
            "Generator seed",
            0,
            value=int(random_controls["seed"]),
            key=f"{p}_random_seed",
            help="The same seed and counts reproduce the same start, goal, walls, and icy cells.",
        )
        st.session_state[f"{p}_random_controls"] = {
            "wall_count": int(random_walls),
            "icy_count": int(random_ice),
            "seed": int(random_seed),
        }
        random_layout_error = int(random_walls) + int(random_ice) + 2 > 100
        if random_layout_error:
            st.sidebar.error("Walls + icy cells must leave two cells for start and goal.")
        if st.sidebar.button(
            "🎲 Generate entire grid",
            key=f"{p}_gen_grid",
            use_container_width=True,
            disabled=random_layout_error,
            help="Generates a valid map including start, goal, walls, icy cells, and their probabilities.",
        ):
            layout = generate_random_grid_layout(
                int(random_walls), int(random_ice), int(random_seed)
            )
            st.session_state[f"{p}_start"] = layout.start
            st.session_state[f"{p}_goal"] = layout.goal
            st.session_state[f"{p}_walls"] = set(layout.walls)
            st.session_state[f"{p}_slippery"] = dict(layout.slippery)
            st.session_state[f"{p}_terminal_states"] = {layout.goal}
            st.session_state[f"{p}_cell_rewards"] = {}
            st.session_state[f"{p}_probability_errors"] = set()
            st.session_state[f"{p}_editor_nonce"] += 1
            invalidate_room_model(room_num)
            st.rerun()

        st.sidebar.subheader("Reward configuration")
        for event, description in SUPPORTED_REWARD_EVENTS.items():
            enabled = st.sidebar.checkbox(
                description,
                value=event in st.session_state[f"{p}_reward_enabled"],
                key=f"{p}_reward_enabled_{event}",
                help=f"Enable or disable the global reward event: {description.lower()}.",
            )
            previous_enabled = event in st.session_state[f"{p}_reward_enabled"]
            previous_value = st.session_state[f"{p}_reward_values"].get(event, 0.0)
            value = st.sidebar.number_input(
                f"{description} value",
                value=float(st.session_state[f"{p}_reward_values"].get(event, 0.0)),
                step=0.1,
                format="%.3f",
                disabled=not enabled,
                key=f"{p}_reward_value_{event}",
                help="The reward added whenever this event occurs. Positive attracts; negative penalizes.",
            )
            if enabled:
                st.session_state[f"{p}_reward_enabled"].add(event)
                st.session_state[f"{p}_reward_values"][event] = float(value)
            else:
                st.session_state[f"{p}_reward_enabled"].discard(event)
                st.session_state[f"{p}_reward_values"][event] = 0.0
            if previous_enabled != enabled or previous_value != st.session_state[f"{p}_reward_values"][event]:
                invalidate_room_model(room_num)

    elif section == "Training":
        controls = st.session_state[f"{p}_training_controls"]
        st.sidebar.subheader(algo_names[room_num])
        if room_num == 1:
            gamma = st.sidebar.slider(
                "Gamma", 0.0, 0.999, float(controls["gamma"]), 0.001,
                key=f"{p}_gamma",
                help="Discount factor: higher values make future rewards more important.",
            )
            theta = st.sidebar.number_input(
                "Theta", 1e-12, 1.0, float(controls["theta"]), format="%.8f",
                key=f"{p}_theta",
                help="Policy Evaluation stops when the largest value change falls below this threshold.",
            )
            max_pi = st.sidebar.number_input(
                "Max policy iterations", 1, 1000, int(controls["max_policy_iterations"]),
                key=f"{p}_max_pi",
                help="Safety limit on complete policy evaluation-and-improvement cycles.",
            )
            max_sweeps = st.sidebar.number_input(
                "Max evaluation sweeps", 1, 100000, int(controls["max_evaluation_sweeps"]),
                key=f"{p}_max_sweeps",
                help="Maximum Bellman sweeps allowed during each policy evaluation phase.",
            )
            seed = st.sidebar.number_input(
                "Seed", 0, value=int(controls["seed"]), key=f"{p}_seed",
                help="Controls reproducible policy initialization and tie-breaking.",
            )
            live_update = st.sidebar.number_input(
                "Update charts every N sweeps", 1, 1000, int(controls["live_update_every"]),
                key=f"{p}_live_update",
                help="Lower values refresh live graphs more often but add UI overhead.",
            )
            st.session_state[f"{p}_training_controls"] = {
                "gamma": float(gamma), "theta": float(theta),
                "max_policy_iterations": int(max_pi), "max_evaluation_sweeps": int(max_sweeps),
                "seed": int(seed), "live_update_every": int(live_update),
            }
        else:
            alpha = st.sidebar.slider(
                "Alpha (learning rate)", 0.01, 1.0, float(controls["alpha"]), 0.01,
                key=f"{p}_alpha", help="Controls how strongly each new experience changes the Q-value.",
            )
            gamma = st.sidebar.slider(
                "Gamma (discount)", 0.0, 0.999, float(controls["gamma"]), 0.001,
                key=f"{p}_gamma", help="Higher values give more weight to rewards farther in the future.",
            )
            eps_start = st.sidebar.slider(
                "Epsilon start", 0.05, 1.0, float(controls["epsilon_start"]), 0.05,
                key=f"{p}_eps_start", help="Initial probability of choosing a random exploratory action.",
            )
            eps_min = st.sidebar.slider(
                "Epsilon min", 0.01, 0.5, float(controls["epsilon_min"]), 0.01,
                key=f"{p}_eps_min", help="Minimum exploration probability retained late in training.",
            )
            eps_decay = st.sidebar.number_input(
                "Epsilon decay rate", 0.8, 1.0, float(controls["epsilon_decay"]),
                format="%.4f", key=f"{p}_eps_decay",
                help="Multiplier applied to epsilon after each episode; closer to 1 decays more slowly.",
            )
            episodes = st.sidebar.number_input(
                "Training episodes", 10, 10000, int(controls["episodes"]), key=f"{p}_episodes",
                help="Number of complete learning episodes to run.",
            )
            max_steps = st.sidebar.number_input(
                "Max timesteps per episode", 10, 5000, int(controls["max_timesteps"]),
                key=f"{p}_max_steps", help="Stops an episode that has not reached a termination state.",
            )
            seed = st.sidebar.number_input(
                "Seed", 0, value=int(controls["seed"]), key=f"{p}_seed",
                help="Controls reproducible exploration and stochastic transitions.",
            )
            live_update = st.sidebar.number_input(
                "Update charts every N episodes", 1, 1000, int(controls["live_update_every"]),
                key=f"{p}_live_update", help="Lower values refresh live charts more frequently.",
            )
            st.session_state[f"{p}_training_controls"] = {
                "alpha": float(alpha), "gamma": float(gamma),
                "epsilon_start": float(eps_start), "epsilon_min": float(eps_min),
                "epsilon_decay": float(eps_decay), "episodes": int(episodes),
                "max_timesteps": int(max_steps), "seed": int(seed),
                "live_update_every": int(live_update),
            }

        if prob_err is not None:
            st.sidebar.error(f"Fix grid error: {prob_err}")
        requests["train"] = st.sidebar.button(
            "▶ Train / compute policy", type="primary", use_container_width=True,
            disabled=prob_err is not None, key=f"{p}_train_btn",
            help="Starts training with the current grid, rewards, and hyperparameters.",
        )
        requests["reset"] = st.sidebar.button(
            "Reset trained model", use_container_width=True, key=f"{p}_reset_btn",
            help="Clears the trained policy and test results without changing the grid.",
        )

    elif section == "Testing":
        controls = st.session_state[f"{p}_test_controls"]
        st.sidebar.subheader("Test configuration")
        episodes = st.sidebar.number_input(
            "Test episodes", 1, 10000, int(controls["episodes"]), key=f"{p}_test_episodes",
            help="Number of evaluation episodes run without learning.",
        )
        max_steps = st.sidebar.number_input(
            "Max timesteps per episode", 1, 50000, int(controls["max_timesteps"]),
            key=f"{p}_test_max_steps", help="Maximum length of each evaluation episode.",
        )
        seed = st.sidebar.number_input(
            "Test seed", 0, value=int(controls["seed"]), key=f"{p}_test_seed",
            help="Reproduces the same stochastic test outcomes when settings are unchanged.",
        )
        st.session_state[f"{p}_test_controls"] = {"episodes": int(episodes), "max_timesteps": int(max_steps), "seed": int(seed)}
        run_test = st.sidebar.button(
            "🧪 Run test", type="primary", use_container_width=True,
            disabled=st.session_state[f"{p}_result"] is None, key=f"{p}_run_test_btn",
            help="Evaluates the current trained model without updating it.",
        )

    else: # Models
        st.sidebar.subheader("Model artifact")
        res = st.session_state[f"{p}_result"]
        env = st.session_state[f"{p}_result_environment"]
        config = st.session_state[f"{p}_algorithm_config"]
        if res and env and config:
            if room_num == 1:
                art = export_room1_artifact(env, config, res)
            elif room_num == 2:
                art = export_room2_artifact(env, config, res)
            else:
                art = export_room3_artifact(env, config, res)
            st.sidebar.download_button(
                f"⬇ Download Room {room_num} model",
                data=art,
                file_name=f"room{room_num}_model.json",
                mime="application/json",
                use_container_width=True,
                help="Downloads the trained policy, environment layout, rewards, and hyperparameters as JSON.",
            )

        uploaded = st.sidebar.file_uploader(
            "Upload model JSON", type=["json"], key=f"{p}_upload",
            help="Select a previously exported model artifact for this room and algorithm.",
        )
        if uploaded is not None and st.sidebar.button(
            "Load model",
            use_container_width=True,
            key=f"{p}_load_btn",
            help="Loads the uploaded artifact and restores its model and complete room configuration.",
        ):
            try:
                if room_num == 1:
                    env_l, config_l, res_l = import_room1_artifact(uploaded.getvalue().decode("utf-8"))
                elif room_num == 2:
                    env_l, config_l, res_l = import_room2_artifact(uploaded.getvalue().decode("utf-8"))
                else:
                    env_l, config_l, res_l = import_room3_artifact(uploaded.getvalue().decode("utf-8"))
            except Exception as exc:
                st.sidebar.error(f"Invalid artifact: {exc}")
            else:
                st.session_state[f"{p}_result_environment"] = env_l
                st.session_state[f"{p}_algorithm_config"] = config_l
                st.session_state[f"{p}_result"] = res_l
                st.session_state[f"{p}_start"] = env_l.config.start
                st.session_state[f"{p}_goal"] = env_l.config.goal
                st.session_state[f"{p}_walls"] = set(env_l.config.walls)
                st.session_state[f"{p}_slippery"] = dict(env_l.config.slippery)
                st.session_state[f"{p}_terminal_states"] = set(env_l.config.terminal_states)
                st.session_state[f"{p}_cell_rewards"] = dict(env_l.config.cell_rewards)
                st.session_state[f"{p}_probability_errors"] = set()
                st.session_state[f"{p}_editor_nonce"] += 1
                st.sidebar.success("Model loaded successfully!")
                st.rerun()

    return section, requests, run_test


def render_room(room_num: int) -> None:
    section, requests, run_test = render_room_controls(room_num)
    if section == "Environment":
        render_environment_page(room_num)
    elif section == "Training":
        render_training_page(room_num, requests)
    elif section == "Testing":
        render_test_page(room_num, run_test)
    else:
        render_models_page(room_num)


def render_future_room(room_name: str) -> None:
    with st.sidebar:
        st.title(f"{room_name} Controls")
        st.info("This room has not been implemented yet.")
        st.radio(
            "Control section",
            options=["Environment", "Training", "Testing", "Models"],
            disabled=True,
            key=f"{room_name.lower().replace(' ', '_')}_control_section",
        )
    st.title(f"{room_name} — Coming soon")
    st.write(
        "The top navigation and room-specific left control bar are ready. "
        "This room will be connected when its continuous environment and algorithm are implemented."
    )


# Execution Entry Point
initialize_state()
active_room = render_room_navigation()
if active_room == "Room 1":
    render_room(1)
elif active_room == "Room 2":
    render_room(2)
elif active_room == "Room 3":
    render_room(3)
else:
    render_future_room(active_room)
