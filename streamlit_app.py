"""Streamlit dashboard for the Escape Room RL project."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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
    export_room4_artifact,
    import_room4_artifact,
    export_room5_artifact,
    import_room5_artifact,
)
from escape_room_rl.evaluation import (  # noqa: E402
    evaluate_policy,
    evaluate_room4_ppo,
    evaluate_room5_ppo,
)
from escape_room_rl.policy_iteration import (  # noqa: E402
    PolicyIterationConfig,
    run_policy_iteration,
)
from escape_room_rl.value_iteration import (  # noqa: E402
    ValueIterationConfig,
    run_value_iteration,
)

from escape_room_rl.sarsa import (  # noqa: E402
    SarsaConfig,
    run_sarsa,
)
from escape_room_rl.q_learning import (  # noqa: E402
    QLearningConfig,
    run_q_learning,
)
from escape_room_rl.dqn import (  # noqa: E402
    DQNConfig,
    DQNNetwork,
    run_dqn,
)
from escape_room_rl.ppo import PPOConfig, run_ppo  # noqa: E402
from escape_room_rl.room1 import (  # noqa: E402
    DEFAULT_REWARDS,
    DEFAULT_SLIPPERY,
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
    DEFAULT_ROOM2_SLIPPERY,
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
from escape_room_rl.room4 import (  # noqa: E402
    Action4,
    DEFAULT_ROOM4_PIPES,
    DEFAULT_ROOM4_REWARDS,
    PipeObstacle,
    Room4Config,
    Room4Environment,
    distribute_pipes_evenly,
)
from escape_room_rl.room5 import (  # noqa: E402
    Action5,
    DEFAULT_ROOM5_REWARDS,
    Room5Config,
    Room5Environment,
)
from escape_room_rl.display_formatting import (  # noqa: E402
    format_reward_label,
    reward_sign_class,
)
from escape_room_rl.visualization import (  # noqa: E402
    render_grid_html,
    render_room4_html,
    render_room5_html,
)



st.set_page_config(
    page_title="Escape Room RL",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --app-nav-height: 3.75rem;
      }
      header[data-testid="stHeader"] {
        top: var(--app-nav-height) !important;
      }
      [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        padding-top: 8rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
      }
      [data-testid="stSidebar"] {
        min-width: 340px;
        max-width: 340px;
        top: var(--app-nav-height) !important;
        height: calc(100vh - var(--app-nav-height)) !important;
      }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 { letter-spacing: -0.02em; }

      /* Full-viewport application navigation above both app columns. */
      div:has(> .st-key-main_top_nav) {
        position: fixed !important;
        inset: 0 0 auto 0 !important;
        width: 100vw !important;
        height: var(--app-nav-height) !important;
        z-index: 1000000 !important;
        margin: 0 !important;
        overflow: visible !important;
      }
      .st-key-main_top_nav {
        position: static !important;
        width: 100% !important;
        height: var(--app-nav-height) !important;
        margin: 0 !important;
        padding: 0.55rem 1.25rem !important;
        background: linear-gradient(90deg, #0f172a, #111827) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.32) !important;
        backdrop-filter: blur(16px) !important;
        box-sizing: border-box !important;
      }
      .st-key-main_top_nav [data-testid="stHorizontalBlock"] {
        height: 100% !important;
        gap: 0.55rem !important;
        align-items: center !important;
      }
      .st-key-nav_brand {
        flex: 1 1 auto !important;
        min-width: 250px !important;
      }
      .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        min-height: 2.55rem;
        color: #f8fafc;
        white-space: nowrap;
      }
      .nav-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.15rem;
        height: 2.15rem;
        border-radius: 0.65rem;
        background: linear-gradient(135deg, #2563eb, #14b8a6);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        font-size: 1.15rem;
        line-height: 1;
      }
      .nav-project-name {
        font-size: 1rem;
        font-weight: 750;
        letter-spacing: -0.01em;
      }
      .st-key-main_top_nav button {
        width: auto !important;
        min-width: 5.2rem !important;
        min-height: 2.35rem !important;
        padding: 0.35rem 0.9rem !important;
        border-radius: 999px !important;
        font-size: 0.88rem !important;
        font-weight: 650 !important;
        letter-spacing: 0.01em !important;
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
        "room1_algorithm": "Policy Iteration",
        "room1_walls": set(DEFAULT_WALLS),
        "room1_slippery": dict(DEFAULT_SLIPPERY),
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
            "algorithm": "Policy Iteration",
            "gamma": 0.95,
            "theta": 1e-6,
            "max_policy_iterations": 100,
            "max_evaluation_sweeps": 10_000,
            "max_iterations": 1000,
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
        "room2_slippery": dict(DEFAULT_ROOM2_SLIPPERY),
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
        # Room 4 (PPO - Flappy Bird)
        "room4_pipes": [
            PipeObstacle(x=2.5, width=0.6, gap_start=3.5, gap_size=3.0),
            PipeObstacle(x=5.0, width=0.6, gap_start=2.0, gap_size=3.0),
            PipeObstacle(x=7.5, width=0.6, gap_start=4.5, gap_size=3.0),
        ],
        "room4_pipe_count_v2": 3,
        "room4_reward_values": dict(DEFAULT_ROOM4_REWARDS),
        "room4_result": None,
        "room4_result_environment": None,
        "room4_algorithm_config": None,
        "room4_test_results": None,
        "room4_training_controls": {
            "alpha": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "update_epochs": 4,
            "mini_batch_size": 64,
            "episodes": 300,
            "max_timesteps": 500,
            "hidden_layers": 2,
            "hidden_units": 64,
            "activation_fn": "Tanh",
            "seed": 42,
            "live_update_every": 10,
        },
        "room4_test_controls": {
            "episodes": 50,
            "max_timesteps": 500,
            "seed": 123,
        },
        # Room 5 (PPO - one-way traffic avoidance)
        "room5_environment_controls": {
            "lane_count": 4,
            "vision_distance": 120.0,
            "road_length": 600.0,
            "ego_speed": 30.0,
            "traffic_speed_min": 12.0,
            "traffic_speed_max": 24.0,
            "traffic_count": 10,
            "seed": 42,
        },
        "room5_reward_values": dict(DEFAULT_ROOM5_REWARDS),
        "room5_result": None,
        "room5_result_environment": None,
        "room5_algorithm_config": None,
        "room5_test_results": None,
        "room5_training_controls": {
            "alpha": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "update_epochs": 4,
            "mini_batch_size": 64,
            "episodes": 300,
            "max_timesteps": 300,
            "hidden_layers": 2,
            "hidden_units": 64,
            "activation_fn": "Tanh",
            "seed": 42,
            "live_update_every": 10,
        },
        "room5_test_controls": {
            "episodes": 50,
            "max_timesteps": 300,
            "seed": 123,
        },
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
    rooms = ["Room 1", "Room 2", "Room 3", "Room 4", "Room 5"]
    active_room = st.session_state.active_room
    with st.container(
        key="main_top_nav",
        horizontal=True,
        vertical_alignment="center",
        gap="small",
    ):
        with st.container(key="nav_brand"):
            st.markdown(
                '<div class="nav-brand">'
                '<span class="nav-logo" aria-hidden="true">🐾</span>'
                '<span class="nav-project-name">Hezki Escape Room RL</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        for room in reversed(rooms):
            selected = st.button(
                room,
                key=f"navigate_{room.lower().replace(' ', '_')}",
                type="primary" if room == active_room else "secondary",
                width="content",
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
            main_color = "inherit"
            main_font_size = "clamp(18px, 2.2vw, 30px)"
            main_font_weight = "400"
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
                reward_value = cell_rewards[state]
                icon = format_reward_label(reward_value)
                sign_class = reward_sign_class(reward_value)
                main_color = {
                    "reward-positive": "#15803d",
                    "reward-negative": "#dc2626",
                    "reward-neutral": "#475569",
                }[sign_class]
                background = {
                    "reward-positive": "#ecfdf5",
                    "reward-negative": "#fef2f2",
                    "reward-neutral": "#f8fafc",
                }[sign_class]
                main_font_size = "clamp(13px, 1.55vw, 20px)"
                main_font_weight = "800"

            selector = f'.st-key-{p}_cell_{x}_{y} [data-testid="stPopover"] button'
            styles_for_cell = [
                f'{selector} {{ background: {background} !important; box-shadow: {outline}; }}',
                (
                    f'{selector}::before {{ content: "{icon}"; color: {main_color}; '
                    f'font-size: {main_font_size} !important; font-weight: {main_font_weight}; }}'
                ),
                f'{selector}::after {{ content: "{x},{y}"; color: {coordinate_color}; }}',
            ]
            reward_has_separate_state_marker = state in cell_rewards and (
                state in {start, goal}
                or state in terminals
                or cell_type == "Icy"
            )
            if reward_has_separate_state_marker:
                reward_value = cell_rewards[state]
                reward_color = {
                    "reward-positive": "#15803d",
                    "reward-negative": "#dc2626",
                    "reward-neutral": "#475569",
                }[reward_sign_class(reward_value)]
                styles_for_cell.append(
                    f'{selector} p::before {{ content: "{format_reward_label(reward_value)}"; '
                    f'position: absolute; top: 4px; left: 4px; z-index: 3; '
                    f'padding: 2px 4px; border-radius: 4px; '
                    f'background: rgba(255, 255, 255, 0.9); color: {reward_color}; '
                    f'font-size: clamp(10px, 1.15vw, 14px) !important; '
                    f'font-weight: 800; line-height: 1; }}'
                )
            cell_styles.extend(styles_for_cell)
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
                    cell_label = f"{format_reward_label(cell_rewards[state])} reward {x},{y}"
                else:
                    cell_label = f"{icon_by_type[current_cell_type(room_num, state)]} {x},{y}"
                if state in cell_rewards and (
                    state in {start, goal}
                    or state in terminals
                    or current_cell_type(room_num, state) == "Icy"
                ):
                    cell_label += f" {format_reward_label(cell_rewards[state])} reward"

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
    room1_algo = st.session_state.get("room1_algorithm", "Policy Iteration")
    algo_names = {1: room1_algo, 2: "SARSA", 3: "Q-Learning"}
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
        '<span class="legend-item"><span style="color:#15803d;font-weight:800">+R</span> / '
        '<span style="color:#dc2626;font-weight:800">−R</span> Cell reward</span>'
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
def render_episode_replay_visualizer(
    environment: Any,
    episodes: list[Any],
    key_prefix: str,
    room_num: int,
    policy: dict[State, Action] | None = None,
    values: dict[State, float] | None = None,
    title: str = "Episode Replay & Animation",
) -> None:
    if not episodes:
        st.info("No episode replays recorded.")
        return

    st.subheader(title)

    def format_ep_option(index: int) -> str:
        ep = episodes[index]
        reward_val = getattr(ep, "total_reward", 0.0)
        success_val = getattr(ep, "success", False)
        status_str = "Success ✅" if success_val else "Failed ❌"
        extra = ""
        if hasattr(ep, "pipes_passed"):
            extra = f", Pipes: {ep.pipes_passed}"
        elif hasattr(ep, "overtakes"):
            extra = f", Overtakes: {ep.overtakes}"
        elif hasattr(ep, "slipped_count") and ep.slipped_count > 0:
            extra = f", Slips: {ep.slipped_count}"
        return f"Episode {getattr(ep, 'episode', index + 1)} ({status_str}, Reward: {reward_val:.2f}{extra})"

    def on_ep_change() -> None:
        st.session_state[f"{key_prefix}_is_playing"] = False
        st.session_state[f"{key_prefix}_step"] = 0

    ep_select_key = f"{key_prefix}_select"
    selected_idx = st.selectbox(
        "Select episode to replay",
        options=list(range(len(episodes))),
        format_func=format_ep_option,
        key=ep_select_key,
        on_change=on_ep_change,
        help="Choose which recorded episode to replay.",
    )

    selected_ep = episodes[selected_idx]
    trajectory = getattr(selected_ep, "trajectory", [])
    if not trajectory:
        st.info("This episode has an empty trajectory.")
        return

    total_steps = len(trajectory)

    step_key = f"{key_prefix}_step"
    playing_key = f"{key_prefix}_is_playing"

    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False

    if st.session_state[step_key] >= total_steps:
        st.session_state[step_key] = 0

    is_playing = st.session_state[playing_key]

    col_prev, col_toggle, col_next, col_speed = st.columns([1, 1, 1, 2], vertical_alignment="bottom")

    with col_prev:
        if st.button("◀ Step Back", key=f"{key_prefix}_btn_prev", use_container_width=True, help="Step back 1 timestep (Keyboard: Left Arrow ←)"):
            st.session_state[playing_key] = False
            st.session_state[step_key] = max(0, st.session_state[step_key] - 1)
            st.rerun()

    with col_toggle:
        toggle_label = "⏸ Pause" if is_playing else "▶ Play"
        if st.button(toggle_label, key=f"{key_prefix}_btn_toggle", type="primary" if not is_playing else "secondary", use_container_width=True, help="Play/Pause automatic replay (Keyboard: Spacebar)"):
            st.session_state[playing_key] = not is_playing
            if st.session_state[playing_key] and st.session_state[step_key] >= total_steps - 1:
                st.session_state[step_key] = 0
            st.rerun()

    with col_next:
        if st.button("Step Next ▶", key=f"{key_prefix}_btn_next", use_container_width=True, help="Step forward 1 timestep (Keyboard: Right Arrow →)"):
            st.session_state[playing_key] = False
            st.session_state[step_key] = min(total_steps - 1, st.session_state[step_key] + 1)
            st.rerun()

    with col_speed:
        playback_speed = st.select_slider(
            "Playback speed",
            options=[0.5, 1.0, 2.0, 4.0],
            value=1.0,
            format_func=lambda val: f"{val:g}×",
            key=f"{key_prefix}_speed",
            help="Controls auto-play speed.",
        )

    def on_slider_change() -> None:
        st.session_state[playing_key] = False
        st.session_state[step_key] = int(st.session_state[f"{key_prefix}_slider_widget"])

    st.slider(
        "Timeline step",
        min_value=0,
        max_value=total_steps - 1,
        value=st.session_state[step_key],
        key=f"{key_prefix}_slider_widget",
        on_change=on_slider_change,
        help="Drag to jump to any recorded timestep.",
    )

    st.caption("💡 Keyboard shortcuts: **← Left Arrow** (Step Back) | **→ Right Arrow** (Step Forward) | **Space** (Play/Pause)")

    # Inject Keyboard Shortcuts JS Script
    js_listener = f"""
    <script>
    (function() {{
      const doc = window.parent.document;
      if (doc._replay_key_listener_{key_prefix}) return;
      doc._replay_key_listener_{key_prefix} = true;

      doc.addEventListener('keydown', function(e) {{
        const active = doc.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {{
          return;
        }}

        if (e.key === 'ArrowLeft') {{
          const buttons = Array.from(doc.querySelectorAll('button'));
          const btn = buttons.find(b => b.textContent.includes('◀ Step Back') || b.textContent.includes('Step Back'));
          if (btn) {{
            e.preventDefault();
            btn.click();
          }}
        }} else if (e.key === 'ArrowRight') {{
          const buttons = Array.from(doc.querySelectorAll('button'));
          const btn = buttons.find(b => b.textContent.includes('Step Next ▶') || b.textContent.includes('Step Next'));
          if (btn) {{
            e.preventDefault();
            btn.click();
          }}
        }} else if (e.key === ' ' || e.code === 'Space') {{
          const buttons = Array.from(doc.querySelectorAll('button'));
          const btn = buttons.find(b => b.textContent.includes('▶ Play') || b.textContent.includes('⏸ Pause'));
          if (btn) {{
            e.preventDefault();
            btn.click();
          }}
        }}
      }});
    }})();
    </script>
    """
    components.html(js_listener, height=0, width=0)

    replay_status = st.empty()
    replay_frame = st.empty()

    def render_single_step(step_idx: int) -> None:
        step_info = trajectory[step_idx]

        if room_num in (1, 2, 3):
            action_name = getattr(step_info.action, "value", str(step_info.action))
            replay_status.info(
                f"Timestep {step_idx + 1}/{total_steps} • "
                f"Action `{action_name}` • Outcome `{getattr(step_info, 'outcome', 'normal')}` • "
                f"Reward {step_info.reward:.3f} • Cumulative {step_info.cumulative_reward:.3f}"
            )
            replay_frame.markdown(
                render_grid_html(
                    environment,
                    agent_state=step_info.next_state,
                    policy=policy,
                    values=values,
                ),
                unsafe_allow_html=True,
            )
        elif room_num == 4:
            st_val = step_info.state
            replay_status.info(
                f"Step {step_info.timestep}/{total_steps} • "
                f"State: (x={st_val[0]:.2f}, y={st_val[1]:.2f}, Vx={st_val[2]:.1f}, Vy={st_val[3]:.1f}) • "
                f"Action: {step_info.action.name} • Step reward: {step_info.reward:.2f} • "
                f"Cumulative reward: {step_info.cumulative_reward:.2f}"
            )
            trajectory_states = [s.state for s in trajectory[: step_idx + 1]] + [step_info.next_state]
            replay_frame.markdown(
                render_room4_html(
                    environment,
                    agent_state=step_info.next_state,
                    trajectory=trajectory_states,
                ),
                unsafe_allow_html=True,
            )
        elif room_num == 5:
            replay_status.info(
                f"Step {step_info.timestep}/{total_steps} • Action: {step_info.action.name} • "
                f"Reward: {step_info.reward:.2f} • Cumulative: {step_info.cumulative_reward:.2f} • "
                f"Events: {', '.join(step_info.events)}"
            )
            replay_frame.markdown(
                render_room5_html(environment, step_info.after_snapshot),
                unsafe_allow_html=True,
            )

    if st.session_state[playing_key]:
        if room_num == 4:
            base_delay = 0.02
        elif room_num == 5:
            base_delay = 0.05
        else:
            base_delay = 0.2

        delay = base_delay / float(playback_speed)
        start_step = st.session_state[step_key]
        for step_i in range(start_step, total_steps):
            if not st.session_state[playing_key]:
                break
            st.session_state[step_key] = step_i
            render_single_step(step_i)
            if step_i < total_steps - 1:
                time.sleep(delay)
        st.session_state[playing_key] = False
        replay_status.success(
            "Replay complete." if getattr(selected_ep, "success", False) else "Replay complete — episode ended without reaching goal."
        )
    else:
        render_single_step(st.session_state[step_key])



def render_training_page(room_num: int, requests: dict[str, bool]) -> None:

    p = room_prefix(room_num)
    room1_algo = st.session_state.get("room1_algorithm", "Policy Iteration")
    algo_names = {1: room1_algo, 2: "SARSA", 3: "Q-Learning"}
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
            if room1_algo == "Value Iteration":
                alg_config = ValueIterationConfig(
                    gamma=controls["gamma"],
                    theta=controls["theta"],
                    max_iterations=controls["max_iterations"],
                    seed=controls["seed"],
                )

                def callback_vi(metric, _values, _policy) -> None:
                    live_rows.append(asdict(metric))
                    if metric.iteration % controls["live_update_every"] != 0:
                        return
                    df = pd.DataFrame(live_rows)
                    status_slot.info(f"Value Iteration sweep {metric.iteration} • delta={metric.delta:.3e}")
                    chart_slot_1.line_chart(df.set_index("global_step")[["delta"]], x_label="Sweep", y_label="Delta")
                    chart_slot_2.line_chart(df.set_index("global_step")[["mean_value"]], x_label="Sweep", y_label="Mean V(s)")

                with st.spinner("Computing Value Iteration..."):
                    res = run_value_iteration(env, alg_config, callback=callback_vi)
            else:
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
        if hasattr(res, "iterations"):
            m2.metric("Value Iterations (Sweeps)", res.iterations)
            m3.metric("Mean V(s)", f"{np.mean(list(res.values.values())):.3f}")
        else:
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
        if room_num == 1:
            y_col2 = "policy_changes"
            x_col2 = "iteration" if hasattr(res, "iterations") else "policy_iteration"
            x_label2 = "Value iteration (Sweep)" if hasattr(res, "iterations") else "Policy iteration"
        else:
            y_col2 = "epsilon"
            x_col2 = "episode"
            x_label2 = "Training episode"
        st.line_chart(
            frame.set_index(x_col2)[[y_col2]],
            x_label=x_label2,
            y_label="Changed actions" if room_num == 1 else "Exploration epsilon",
        )

    if hasattr(res, "training_episodes") and res.training_episodes:
        render_episode_replay_visualizer(
            env,
            res.training_episodes,
            f"{p}_tr_replay",
            room_num,
            policy=res.policy,
            values=res.values,
            title="Training Episodes Replay",
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

    render_episode_replay_visualizer(
        env,
        test_results,
        f"{p}_te_replay",
        room_num,
        policy=res.policy,
        values=res.values,
        title="Test Episodes Replay & Animation",
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
    defaults_slippery = {1: DEFAULT_SLIPPERY, 2: DEFAULT_ROOM2_SLIPPERY, 3: {}}
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
            help="Restores the room's original walls, icy cells, start, goal, and removes custom cell rewards.",
        ):
            st.session_state[f"{p}_walls"] = set(defaults_walls[room_num])
            st.session_state[f"{p}_slippery"] = dict(defaults_slippery.get(room_num, {}))
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
            selected_algo = st.sidebar.radio(
                "Dynamic Programming Algorithm",
                options=["Policy Iteration", "Value Iteration"],
                index=0 if st.session_state.get("room1_algorithm", "Policy Iteration") == "Policy Iteration" else 1,
                key=f"{p}_algo_selection",
                help="Choose between Policy Iteration and Value Iteration for Room 1.",
            )
            if selected_algo != st.session_state.get("room1_algorithm"):
                st.session_state["room1_algorithm"] = selected_algo
                invalidate_room_model(1)
                st.rerun()

            gamma = st.sidebar.slider(
                "Gamma", 0.0, 0.999, float(controls.get("gamma", 0.95)), 0.001,
                key=f"{p}_gamma",
                help="Discount factor: higher values make future rewards more important.",
            )
            theta = st.sidebar.number_input(
                "Theta", 1e-12, 1.0, float(controls.get("theta", 1e-6)), format="%.8f",
                key=f"{p}_theta",
                help="Value update convergence threshold.",
            )
            if selected_algo == "Value Iteration":
                max_vi = st.sidebar.number_input(
                    "Max value iterations (sweeps)", 1, 100000, int(controls.get("max_iterations", 1000)),
                    key=f"{p}_max_vi",
                    help="Maximum Bellman optimality sweeps allowed for Value Iteration.",
                )
                seed = st.sidebar.number_input(
                    "Seed", 0, value=int(controls.get("seed", 42)), key=f"{p}_seed",
                    help="Controls reproducible policy tie-breaking.",
                )
                live_update = st.sidebar.number_input(
                    "Update charts every N sweeps", 1, 1000, int(controls.get("live_update_every", 5)),
                    key=f"{p}_live_update",
                    help="Lower values refresh live graphs more often but add UI overhead.",
                )
                st.session_state[f"{p}_training_controls"] = {
                    "algorithm": "Value Iteration",
                    "gamma": float(gamma), "theta": float(theta),
                    "max_iterations": int(max_vi),
                    "seed": int(seed), "live_update_every": int(live_update),
                }
            else:
                max_pi = st.sidebar.number_input(
                    "Max policy iterations", 1, 1000, int(controls.get("max_policy_iterations", 100)),
                    key=f"{p}_max_pi",
                    help="Safety limit on complete policy evaluation-and-improvement cycles.",
                )
                max_sweeps = st.sidebar.number_input(
                    "Max evaluation sweeps", 1, 100000, int(controls.get("max_evaluation_sweeps", 10000)),
                    key=f"{p}_max_sweeps",
                    help="Maximum Bellman sweeps allowed during each policy evaluation phase.",
                )
                seed = st.sidebar.number_input(
                    "Seed", 0, value=int(controls.get("seed", 42)), key=f"{p}_seed",
                    help="Controls reproducible policy initialization and tie-breaking.",
                )
                live_update = st.sidebar.number_input(
                    "Update charts every N sweeps", 1, 1000, int(controls.get("live_update_every", 5)),
                    key=f"{p}_live_update",
                    help="Lower values refresh live graphs more often but add UI overhead.",
                )
                st.session_state[f"{p}_training_controls"] = {
                    "algorithm": "Policy Iteration",
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
                    if isinstance(config_l, ValueIterationConfig):
                        st.session_state["room1_algorithm"] = "Value Iteration"
                    else:
                        st.session_state["room1_algorithm"] = "Policy Iteration"
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


# =====================================================================
# ROOM 4 (PPO - FLAPPY BIRD) IMPLEMENTATION
# =====================================================================

ROOM4_CONTROL_HELP = {
    "section": "Choose which Room 4 workspace to display: configure the environment, train PPO, test a trained network, or manage saved model files.",
    "pipe_count": "Sets the number of obstacles. Whenever this value changes, all pipes are redistributed at equal horizontal distances between 2m and 8m so the layout remains valid, including five pipes.",
    "pipe_x": "The horizontal center of this pipe in meters. Moving it changes where the agent must pass the obstacle; avoid overlapping another pipe or the goal zone.",
    "pipe_width": "The horizontal thickness of the pipe. A wider pipe occupies more travel distance and makes collision avoidance harder.",
    "gap_start": "The height, in meters, where the safe opening begins. It controls the lower edge of the gap and therefore the vertical route the agent must learn.",
    "gap_size": "The vertical size of the safe opening. Smaller gaps make the task harder; the gap must remain fully inside the 10m room.",
    "step_reward": "Reward added on every timestep. A small negative value encourages shorter solutions; a stronger penalty may make the agent overly cautious about exploring.",
    "progress_reward": "Reward multiplier for positive horizontal progress. The reward is multiplied by the distance moved to the right during that timestep.",
    "backward_reward": "Reward added whenever the agent moves left, including diagonal left actions. Use a negative value to penalize backward movement, or zero to ignore it.",
    "pipe_reward": "One-time reward received when the agent crosses a pipe's horizontal center from left to right. Larger values emphasize obstacle completion.",
    "goal_reward": "Terminal reward received for reaching the goal zone. It should normally be large enough to outweigh accumulated step penalties.",
    "collision_reward": "Terminal penalty applied after hitting a wall, floor, ceiling, or pipe. More negative values teach stronger collision avoidance.",
    "alpha": "The optimizer learning rate for actor and critic networks. Higher values change network weights faster but can make learning unstable; lower values are steadier but slower.",
    "gamma": "The discount factor for future rewards. Values near 1 make PPO plan farther ahead; lower values emphasize immediate rewards.",
    "gae_lambda": "Generalized Advantage Estimation (GAE) smoothing parameter lambda. Balances bias vs variance in advantage estimations.",
    "clip_epsilon": "PPO policy surrogate clipping threshold epsilon. Limits policy update ratio to prevent destructively large policy updates.",
    "entropy_coefficient": "Coefficient for entropy bonus in PPO loss. Higher values encourage policy exploration by penalizing overly deterministic policies.",
    "value_coefficient": "Multiplier for the mean-squared-error value loss relative to policy loss in the combined PPO objective.",
    "update_epochs": "Number of optimization epochs over collected trajectory mini-batches in each PPO iteration update.",
    "mini_batch_size": "Batch size used for gradient steps during PPO update epochs.",
    "episodes": "Number of complete training attempts. More episodes provide more experience but increase training time.",
    "max_timesteps": "Maximum actions permitted in one episode. The episode also stops earlier if the agent reaches the goal or collides.",
    "hidden_layers": "Number of fully connected hidden layers in the shared network. More layers can represent more complex policies but require more data and computation.",
    "hidden_units": "Number of neurons in each hidden layer. More neurons increase model capacity as well as training cost and overfitting risk.",
    "activation": "Non-linear function between hidden layers. Tanh or ReLU are reliable defaults.",
    "seed": "Controls random network initialization and trajectory sampling. Reusing the same seed helps reproduce an experiment.",
    "live_update": "Refresh live charts after this many episodes. Smaller values show finer progress but add UI overhead during training.",
    "train": "Start a new PPO training run with the current environment, rewards, network architecture, and hyperparameters.",
    "reset": "Remove the trained Room 4 model and its stored test results from this browser session. Environment settings are kept.",
    "test_episodes": "Number of evaluation episodes run with deterministic greedy policy actions.",
    "test_timesteps": "Maximum actions allowed in each evaluation episode before it is marked unfinished.",
    "test_seed": "Seed reserved for reproducible evaluation. The current Room 4 environment is deterministic, but the setting keeps the test configuration explicit.",
    "run_test": "Evaluate the trained network without exploratory sampling and record metrics, action choices, and replay trajectories.",
    "download": "Download the trained network weights, environment, hyperparameters, metrics, duration, and action counts as a JSON artifact.",
    "upload": "Select a Room 4 PPO JSON artifact previously downloaded from this dashboard.",
    "load": "Validate the selected artifact and restore its environment, network weights, configuration, and training results.",
}
ROOM4_REPLAY_BASE_STEP_SECONDS = 0.02


def _sync_room4_pipe_widget_state(
    pipes: list[PipeObstacle], *, overwrite: bool = False
) -> None:
    for idx, pipe in enumerate(pipes):
        values = {
            f"p_{idx}_x": float(pipe.x),
            f"p_{idx}_w": float(pipe.width),
            f"p_{idx}_gs": float(pipe.gap_start),
            f"p_{idx}_gz": float(pipe.gap_size),
        }
        for key, value in values.items():
            if overwrite or key not in st.session_state:
                st.session_state[key] = value


def _redistribute_room4_pipes() -> None:
    pipe_count = int(st.session_state.room4_pipe_count_v2)
    pipes = distribute_pipes_evenly(list(st.session_state.room4_pipes), pipe_count)
    st.session_state.room4_pipes = pipes
    _sync_room4_pipe_widget_state(pipes, overwrite=True)
    for idx in range(pipe_count, 5):
        for suffix in ("x", "w", "gs", "gz"):
            st.session_state.pop(f"p_{idx}_{suffix}", None)
    st.session_state.room4_result = None
    st.session_state.room4_result_environment = None
    st.session_state.room4_algorithm_config = None
    st.session_state.room4_test_results = None


def _room4_action_dataframe(action_counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Action": [action.name for action in Action4],
            "Selections": [int(action_counts.get(action.name, 0)) for action in Action4],
        }
    )


def _format_training_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f} seconds"
    minutes, remaining_seconds = divmod(seconds, 60.0)
    return f"{int(minutes)}m {remaining_seconds:.1f}s"

def build_room4_environment() -> Room4Environment:
    config = Room4Config(
        pipes=list(st.session_state.room4_pipes),
        rewards=dict(st.session_state.room4_reward_values),
    )
    return Room4Environment(config)


def render_room4_controls() -> tuple[str, dict[str, bool], bool]:
    st.sidebar.title("Room 4 Controls (PPO)")
    section = st.sidebar.radio(
        "Control section",
        options=["Environment", "Training", "Testing", "Models"],
        key="room4_control_section",
        help=ROOM4_CONTROL_HELP["section"],
    )

    requests: dict[str, bool] = {"train": False, "reset": False}
    run_test = False

    if section == "Environment":
        st.sidebar.subheader("Flappy Bird pipe obstacles")
        current_pipes = list(st.session_state.room4_pipes)
        if int(st.session_state.get("room4_pipe_count_v2", len(current_pipes))) != len(
            current_pipes
        ):
            st.session_state.room4_pipe_count_v2 = len(current_pipes)
        st.sidebar.slider(
            "Number of pipes",
            1,
            5,
            value=len(current_pipes),
            key="room4_pipe_count_v2",
            on_change=_redistribute_room4_pipes,
            help=ROOM4_CONTROL_HELP["pipe_count"],
        )

        current_pipes = list(st.session_state.room4_pipes)
        _sync_room4_pipe_widget_state(current_pipes)

        updated_pipes = []
        for idx, pipe in enumerate(current_pipes):
            with st.sidebar.expander(f"Pipe {idx + 1} configuration", expanded=False):
                px = st.number_input(
                    f"Pipe {idx + 1} X position (m)",
                    0.5,
                    9.0,
                    step=0.1,
                    key=f"p_{idx}_x",
                    help=ROOM4_CONTROL_HELP["pipe_x"],
                )
                pw = st.number_input(
                    f"Pipe {idx + 1} width (m)",
                    0.2,
                    2.0,
                    step=0.1,
                    key=f"p_{idx}_w",
                    help=ROOM4_CONTROL_HELP["pipe_width"],
                )
                g_start = st.number_input(
                    f"Pipe {idx + 1} gap start Y (m)",
                    0.5,
                    8.0,
                    step=0.1,
                    key=f"p_{idx}_gs",
                    help=ROOM4_CONTROL_HELP["gap_start"],
                )
                g_size = st.number_input(
                    f"Pipe {idx + 1} gap size (m)",
                    1.0,
                    6.0,
                    step=0.1,
                    key=f"p_{idx}_gz",
                    help=ROOM4_CONTROL_HELP["gap_size"],
                )
                updated_pipes.append(PipeObstacle(x=px, width=pw, gap_start=g_start, gap_size=g_size))

        if updated_pipes != st.session_state.room4_pipes:
            st.session_state.room4_pipes = updated_pipes
            st.session_state.room4_result = None

        st.sidebar.subheader("Reward structure")
        rewards = dict(st.session_state.room4_reward_values)
        r_step = st.sidebar.slider("Step penalty", -1.0, 0.0, float(rewards.get("step", -0.05)), 0.01, key="r4_r_step", help=ROOM4_CONTROL_HELP["step_reward"])
        r_prog = st.sidebar.slider("Progress reward", 0.0, 2.0, float(rewards.get("progress", 0.5)), 0.1, key="r4_r_prog", help=ROOM4_CONTROL_HELP["progress_reward"])
        r_back = st.sidebar.slider("Backward movement reward", -10.0, 0.0, float(rewards.get("backward", 0.0)), 0.1, key="r4_r_back", help=ROOM4_CONTROL_HELP["backward_reward"])
        r_pipe = st.sidebar.slider("Pipe passed reward", 0.0, 20.0, float(rewards.get("pipe_passed", 5.0)), 0.5, key="r4_r_pipe", help=ROOM4_CONTROL_HELP["pipe_reward"])
        r_goal = st.sidebar.slider("Goal reward", 5.0, 50.0, float(rewards.get("goal_reached", 20.0)), 1.0, key="r4_r_goal", help=ROOM4_CONTROL_HELP["goal_reward"])
        r_coll = st.sidebar.slider("Collision penalty", -50.0, -1.0, float(rewards.get("collision", -20.0)), 1.0, key="r4_r_coll", help=ROOM4_CONTROL_HELP["collision_reward"])

        st.session_state.room4_reward_values = {
            "step": float(r_step),
            "progress": float(r_prog),
            "backward": float(r_back),
            "pipe_passed": float(r_pipe),
            "goal_reached": float(r_goal),
            "collision": float(r_coll),
        }

    elif section == "Training":
        controls = st.session_state.room4_training_controls
        st.sidebar.subheader("PPO hyperparameters")
        alpha = st.sidebar.number_input("Learning rate (α)", 0.00001, 0.1, float(controls.get("alpha", 0.0003)), format="%.5f", key="r4_alpha", help=ROOM4_CONTROL_HELP["alpha"])
        gamma = st.sidebar.number_input("Discount factor (γ)", 0.0, 0.999, float(controls.get("gamma", 0.99)), format="%.3f", key="r4_gamma", help=ROOM4_CONTROL_HELP["gamma"])
        gae_lambda = st.sidebar.number_input("GAE lambda (λ)", 0.0, 1.0, float(controls.get("gae_lambda", 0.95)), format="%.3f", key="r4_gae_lambda", help=ROOM4_CONTROL_HELP["gae_lambda"])
        clip_eps = st.sidebar.number_input("Clipping epsilon (ε)", 0.01, 0.5, float(controls.get("clip_epsilon", 0.2)), format="%.2f", key="r4_clip_eps", help=ROOM4_CONTROL_HELP["clip_epsilon"])
        ent_coef = st.sidebar.number_input("Entropy coefficient", 0.0, 0.5, float(controls.get("entropy_coefficient", 0.01)), format="%.4f", key="r4_ent_coef", help=ROOM4_CONTROL_HELP["entropy_coefficient"])
        val_coef = st.sidebar.number_input("Value loss coefficient", 0.0, 2.0, float(controls.get("value_coefficient", 0.5)), format="%.2f", key="r4_val_coef", help=ROOM4_CONTROL_HELP["value_coefficient"])
        update_epochs = st.sidebar.number_input("PPO update epochs", 1, 20, int(controls.get("update_epochs", 4)), key="r4_update_epochs", help=ROOM4_CONTROL_HELP["update_epochs"])
        mini_batch = st.sidebar.number_input("Mini-batch size", 8, 512, int(controls.get("mini_batch_size", 64)), key="r4_mini_batch", help=ROOM4_CONTROL_HELP["mini_batch_size"])
        episodes = st.sidebar.number_input("Episodes", 1, 5000, int(controls.get("episodes", 300)), key="r4_episodes", help=ROOM4_CONTROL_HELP["episodes"])
        max_steps = st.sidebar.number_input("Maximum timesteps", 50, 2000, int(controls.get("max_timesteps", 500)), key="r4_max_steps", help=ROOM4_CONTROL_HELP["max_timesteps"])

        st.sidebar.subheader("Neural network architecture")
        h_layers = st.sidebar.slider("Hidden layer count", 1, 4, int(controls.get("hidden_layers", 2)), key="r4_h_layers", help=ROOM4_CONTROL_HELP["hidden_layers"])
        h_units = st.sidebar.select_slider("Neurons per hidden layer", options=[16, 32, 64, 128, 256], value=int(controls.get("hidden_units", 64)), key="r4_h_units", help=ROOM4_CONTROL_HELP["hidden_units"])
        activation_opts = ["Tanh", "ReLU", "LeakyReLU", "ELU", "SiLU"]
        current_act = controls.get("activation_fn", "Tanh")
        act_idx = activation_opts.index(current_act) if current_act in activation_opts else 0
        act_fn = st.sidebar.selectbox("Activation function", options=activation_opts, index=act_idx, key="r4_activation_fn", help=ROOM4_CONTROL_HELP["activation"])

        seed = st.sidebar.number_input("Random seed", 0, value=int(controls.get("seed", 42)), key="r4_seed", help=ROOM4_CONTROL_HELP["seed"])
        live_update = st.sidebar.number_input("Update charts every N episodes", 1, 100, int(controls.get("live_update_every", 10)), key="r4_live_update", help=ROOM4_CONTROL_HELP["live_update"])

        st.session_state.room4_training_controls = {
            "alpha": float(alpha),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "clip_epsilon": float(clip_eps),
            "entropy_coefficient": float(ent_coef),
            "value_coefficient": float(val_coef),
            "update_epochs": int(update_epochs),
            "mini_batch_size": int(mini_batch),
            "episodes": int(episodes),
            "max_timesteps": int(max_steps),
            "hidden_layers": int(h_layers),
            "hidden_units": int(h_units),
            "activation_fn": str(act_fn),
            "seed": int(seed),
            "live_update_every": int(live_update),
        }

        requests["train"] = st.sidebar.button("Train PPO agent", icon=":material/play_arrow:", type="primary", width="stretch", key="r4_train_btn", help=ROOM4_CONTROL_HELP["train"])
        requests["reset"] = st.sidebar.button("Reset trained model", icon=":material/restart_alt:", width="stretch", key="r4_reset_btn", help=ROOM4_CONTROL_HELP["reset"])

    elif section == "Testing":
        controls = st.session_state.room4_test_controls
        st.sidebar.subheader("Test configuration")
        episodes = st.sidebar.number_input("Test episodes", 1, 1000, int(controls["episodes"]), key="r4_test_episodes", help=ROOM4_CONTROL_HELP["test_episodes"])
        max_steps = st.sidebar.number_input("Maximum timesteps per episode", 10, 5000, int(controls["max_timesteps"]), key="r4_test_max_steps", help=ROOM4_CONTROL_HELP["test_timesteps"])
        seed = st.sidebar.number_input("Test seed", 0, value=int(controls["seed"]), key="r4_test_seed", help=ROOM4_CONTROL_HELP["test_seed"])

        st.session_state.room4_test_controls = {"episodes": int(episodes), "max_timesteps": int(max_steps), "seed": int(seed)}
        run_test = st.sidebar.button(
            "Run test", icon=":material/science:", type="primary", width="stretch",
            disabled=st.session_state.room4_result is None, key="r4_run_test_btn",
            help=ROOM4_CONTROL_HELP["run_test"],
        )

    else:  # Models
        st.sidebar.subheader("Model artifact")
        res = st.session_state.room4_result
        env = st.session_state.room4_result_environment
        config = st.session_state.room4_algorithm_config
        if res and env and config:
            art = export_room4_artifact(env, config, res)
            st.sidebar.download_button(
                "Download Room 4 model (JSON)",
                data=art,
                file_name="room4_ppo_model.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch",
                help=ROOM4_CONTROL_HELP["download"],
            )

        uploaded = st.sidebar.file_uploader("Upload model JSON", type=["json"], key="r4_upload", help=ROOM4_CONTROL_HELP["upload"])
        if uploaded is not None and st.sidebar.button("Load model", icon=":material/upload_file:", width="stretch", key="r4_load_btn", help=ROOM4_CONTROL_HELP["load"]):
            try:
                env_l, config_l, res_l = import_room4_artifact(uploaded.getvalue().decode("utf-8"))
            except Exception as exc:
                st.sidebar.error(f"Invalid artifact: {exc}")
            else:
                st.session_state.room4_result_environment = env_l
                st.session_state.room4_algorithm_config = config_l
                st.session_state.room4_result = res_l
                st.session_state.room4_pipes = list(env_l.config.pipes)
                st.session_state.room4_pipe_count_v2 = len(env_l.config.pipes)
                _sync_room4_pipe_widget_state(list(env_l.config.pipes), overwrite=True)
                st.session_state.room4_reward_values = dict(env_l.config.rewards)
                st.sidebar.success("Room 4 PPO model loaded successfully!")
                st.rerun()

    return section, requests, run_test


def render_room4() -> None:
    section, requests, run_test = render_room4_controls()

    st.markdown('<div class="room-header"><h2>🐤 Room 4: Continuous Flappy Bird (PPO Algorithm)</h2></div>', unsafe_allow_html=True)

    if section == "Environment":
        env = build_room4_environment()
        st.subheader("10m × 10m continuous Flappy Bird room layout")
        st.markdown(render_room4_html(env), unsafe_allow_html=True)

        st.subheader("Custom pipe obstacles overview")
        pipe_df = pd.DataFrame([
            {
                "Pipe": idx + 1,
                "X Position (m)": pipe.x,
                "Width (m)": pipe.width,
                "Gap Start Y (m)": pipe.gap_start,
                "Gap End Y (m)": pipe.gap_end,
                "Gap Size (m)": pipe.gap_size,
            }
            for idx, pipe in enumerate(env.config.pipes)
        ])
        st.dataframe(pipe_df, width="stretch")

    elif section == "Training":
        if requests["reset"]:
            st.session_state.room4_result = None
            st.session_state.room4_result_environment = None
            st.session_state.room4_algorithm_config = None
            st.session_state.room4_test_results = None
            st.success("Trained Room 4 model reset.")
            st.rerun()

        if requests["train"]:
            env = build_room4_environment()
            ctrls = st.session_state.room4_training_controls

            hidden_dims = tuple([ctrls["hidden_units"]] * ctrls["hidden_layers"])
            config = PPOConfig(
                alpha=ctrls["alpha"],
                gamma=ctrls["gamma"],
                gae_lambda=ctrls["gae_lambda"],
                clip_epsilon=ctrls["clip_epsilon"],
                entropy_coefficient=ctrls["entropy_coefficient"],
                value_coefficient=ctrls["value_coefficient"],
                update_epochs=ctrls["update_epochs"],
                mini_batch_size=ctrls["mini_batch_size"],
                episodes=ctrls["episodes"],
                max_timesteps=ctrls["max_timesteps"],
                hidden_dims=hidden_dims,
                activation_fn=ctrls.get("activation_fn", "Tanh"),
                seed=ctrls["seed"],
            )

            status_placeholder = st.empty()
            st.subheader("Live training metrics")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Total reward per episode**")
                slot_reward = st.empty()
                st.caption("**Entropy**")
                slot_entropy = st.empty()
            with c2:
                st.caption("**Policy loss**")
                slot_ploss = st.empty()
                st.caption("**Value loss**")
                slot_vloss = st.empty()

            live_update_freq = ctrls["live_update_every"]
            live_rows: list[dict] = []

            def live_callback(metric, policy_net):
                live_rows.append(asdict(metric))
                if metric.episode % live_update_freq == 0 or metric.episode == config.episodes:
                    status_placeholder.info(
                        f"Training episode {metric.episode}/{config.episodes} • "
                        f"Reward: {metric.total_reward:.2f} • Policy Loss: {metric.policy_loss:.4f} • Value Loss: {metric.value_loss:.4f}"
                    )
                    df_live = pd.DataFrame(live_rows)
                    slot_reward.line_chart(df_live.set_index("episode")[["total_reward"]], x_label="Episode", y_label="Total reward")
                    slot_ploss.line_chart(df_live.set_index("episode")[["policy_loss"]], x_label="Episode", y_label="Policy loss")
                    slot_vloss.line_chart(df_live.set_index("episode")[["value_loss"]], x_label="Episode", y_label="Value loss")
                    slot_entropy.line_chart(df_live.set_index("episode")[["entropy"]], x_label="Episode", y_label="Entropy")

            with st.spinner("Training PPO agent..."):
                result = run_ppo(env, config, callback=live_callback)

            st.session_state.room4_result = result
            st.session_state.room4_result_environment = env
            st.session_state.room4_algorithm_config = config
            st.session_state.room4_test_results = None
            duration_label = _format_training_duration(result.training_duration_seconds)
            status_placeholder.success(f"PPO training complete in {duration_label}.")

            summary_cols = st.columns(3)
            summary_cols[0].metric("Training duration", duration_label)
            summary_cols[1].metric("Episodes completed", result.episodes_run)
            summary_cols[2].metric("Actions selected", sum(result.action_counts.values()))
            st.subheader("Training action distribution")
            st.bar_chart(
                _room4_action_dataframe(result.action_counts),
                x="Action",
                y="Selections",
                x_label="Action",
                y_label="Number of selections",
            )

        res = st.session_state.room4_result
        if res and not requests["train"]:
            df = pd.DataFrame([asdict(m) for m in res.metrics])
            summary_cols = st.columns(3)
            summary_cols[0].metric(
                "Training duration",
                _format_training_duration(
                    float(getattr(res, "training_duration_seconds", 0.0))
                ),
            )
            summary_cols[1].metric("Episodes completed", res.episodes_run)
            summary_cols[2].metric(
                "Actions selected", sum(getattr(res, "action_counts", {}).values())
            )

            st.subheader("Training metrics")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Total reward per episode**")
                st.line_chart(df.set_index("episode")[["total_reward"]], x_label="Episode", y_label="Total reward")
                st.caption("**Entropy**")
                if "entropy" in df.columns:
                    st.line_chart(df.set_index("episode")[["entropy"]], x_label="Episode", y_label="Entropy")
            with c2:
                st.caption("**Policy loss**")
                if "policy_loss" in df.columns:
                    st.line_chart(df.set_index("episode")[["policy_loss"]], x_label="Episode", y_label="Policy loss")
                st.caption("**Value loss**")
                if "value_loss" in df.columns:
                    st.line_chart(df.set_index("episode")[["value_loss"]], x_label="Episode", y_label="Value loss")

            st.subheader("Training action distribution")
            st.bar_chart(
                _room4_action_dataframe(getattr(res, "action_counts", {})),
                x="Action",
                y="Selections",
                x_label="Action",
                y_label="Number of selections",
            )

            st.write(f"**Episodes run:** {res.episodes_run} | **Goal reached in late training:** {'Yes ✅' if res.converged else 'No ❌'}")
            if hasattr(res, "training_episodes") and res.training_episodes:
                env_cur = st.session_state.room4_result_environment or build_room4_environment()
                render_episode_replay_visualizer(
                    env_cur,
                    res.training_episodes,
                    "room4_tr_replay",
                    4,
                    title="Training Episodes Replay",
                )
        elif not requests["train"]:
            st.info("Click '▶ Train PPO Agent' in the left sidebar to start training.")

    elif section == "Testing":
        if run_test:
            res = st.session_state.room4_result
            env = st.session_state.room4_result_environment
            if res and env:
                t_ctrls = st.session_state.room4_test_controls
                test_results = evaluate_room4_ppo(
                    environment=env,
                    policy_net=res.policy_net,
                    episodes=t_ctrls["episodes"],
                    max_timesteps=t_ctrls["max_timesteps"],
                    seed=t_ctrls["seed"],
                )
                st.session_state.room4_test_results = test_results

        test_res = st.session_state.room4_test_results
        env = st.session_state.room4_result_environment
        if test_res and env:
            st.subheader("Test execution summary")
            df_test = pd.DataFrame([
                {
                    "Episode": ep.episode,
                    "Success": ep.success,
                    "Timesteps": ep.timesteps,
                    "Total Reward": ep.total_reward,
                    "Pipes Passed": ep.pipes_passed,
                }
                for ep in test_res
            ])
            success_rate = (df_test["Success"].sum() / len(df_test)) * 100.0
            st.metric("Test success rate", f"{success_rate:.1f}%")
            st.dataframe(df_test, width="stretch")

            test_action_counts = {action.name: 0 for action in Action4}
            for episode in test_res:
                for step in episode.trajectory:
                    test_action_counts[step.action.name] += 1
            st.subheader("Test action distribution")
            st.bar_chart(
                _room4_action_dataframe(test_action_counts),
                x="Action",
                y="Selections",
                x_label="Action",
                y_label="Number of selections",
            )

            render_episode_replay_visualizer(
                env,
                test_res,
                "room4_te_replay",
                4,
                title="Test Episodes Replay & Animation",
            )
        else:
            st.info("Run a test from the left sidebar to view test metrics and replay trajectories.")


    else:  # Models
        res = st.session_state.room4_result
        config = st.session_state.room4_algorithm_config
        st.subheader("Model & Network Information")
        if res and config:
            st.json({
                "Algorithm": "PPO (Proximal Policy Optimization)",
                "Input State Dimension": 4,
                "Output Action Dimension": 9,
                "Hidden Architecture": list(config.hidden_dims),
                "Activation Function": getattr(config, "activation_fn", "ReLU"),
                "Learning Rate": config.alpha,
                "Discount Factor": config.gamma,
                "Episodes Trained": res.episodes_run,
                "Training Duration Seconds": float(
                    getattr(res, "training_duration_seconds", 0.0)
                ),
                "Action Counts": getattr(res, "action_counts", {}),
                "Converged": res.converged,
            })

        else:
            st.info("No trained model currently in memory.")


# =====================================================================
# ROOM 5 (PPO - ONE-WAY ROAD) IMPLEMENTATION
# =====================================================================

ROOM5_HELP = {
    "section": "Choose whether to configure the road, train PPO, evaluate a trained policy, or manage saved Room 5 models.",
    "lanes": "Number of one-way traffic lanes. The agent can move left, keep its lane, or move right. More lanes provide more escape routes but enlarge the decision space.",
    "vision": "Maximum distance ahead, in meters, included in the agent's observation. A longer range gives earlier warning but compresses nearby distance differences.",
    "road_length": "Forward distance required to complete an episode successfully. Longer roads require the policy to avoid traffic for more timesteps.",
    "traffic_count": "Number of slower same-direction cars circulating ahead. More cars increase traffic density and the frequency of avoidance decisions.",
    "ego_speed": "Constant speed of the agent car. It must remain faster than traffic so that other cars approach in the agent-relative view and can be overtaken.",
    "traffic_min": "Minimum speed assigned to a traffic car. Slower cars approach the agent more quickly and are harder to avoid.",
    "traffic_max": "Maximum traffic-car speed. It stays below the agent speed to preserve same-direction overtaking behavior.",
    "env_seed": "Controls the initial traffic lanes, distances, and speeds so the same road setup can be reproduced.",
    "step": "Reward on every timestep. A small negative value discourages unnecessarily long episodes.",
    "progress": "Reward multiplier per meter of forward travel. It provides dense feedback even before a car is overtaken.",
    "overtake": "Reward granted each time the agent safely passes a traffic car.",
    "lane_change": "Reward applied to a valid lane change. A small penalty reduces needless weaving while still allowing evasive maneuvers.",
    "invalid_change": "Penalty for requesting a lane beyond the left or right road boundary.",
    "collision": "Terminal penalty for occupying the same lane and longitudinal space as another car.",
    "goal": "Terminal reward for reaching the configured road length without a collision.",
    "alpha": "PPO optimizer learning rate. Larger values learn faster but may destabilize both policy and value estimates.",
    "gamma": "Discount factor for future rewards. Values near 1 make safe long-term driving more important.",
    "gae": "Generalized Advantage Estimation lambda. Higher values reduce bias but increase variance in policy updates.",
    "clip": "Limits how far the policy may change during one PPO update. Smaller values are conservative; larger values allow faster but riskier changes.",
    "entropy": "Strength of the exploration bonus. Higher values keep action probabilities more diverse for longer.",
    "value_coef": "Weight of the critic's value-prediction loss in the combined PPO objective.",
    "epochs": "Number of optimization passes over each episode rollout. More passes reuse data more heavily but can overfit a rollout.",
    "batch": "Maximum rollout samples used in one PPO gradient step. Larger batches are smoother and use more memory.",
    "episodes": "Number of complete road attempts used for training.",
    "timesteps": "Maximum decisions per episode. Collision or road completion can end the episode earlier.",
    "layers": "Number of hidden layers shared by the actor and critic before their separate output heads.",
    "units": "Number of neurons in each hidden layer. More neurons increase capacity and training cost.",
    "activation": "Non-linear function used between hidden layers. Tanh is a common stable choice for PPO.",
    "train_seed": "Controls neural-network initialization and sampled PPO actions for reproducible experiments.",
    "live": "Refresh live graphs every N episodes. Smaller values provide finer feedback but add dashboard overhead.",
    "test_episodes": "Number of greedy evaluation episodes with no sampled exploration.",
    "test_steps": "Maximum decisions allowed in each evaluation episode.",
    "test_seed": "Controls evaluation traffic generation so results can be reproduced.",
}


def _invalidate_room5_model() -> None:
    st.session_state.room5_result = None
    st.session_state.room5_result_environment = None
    st.session_state.room5_algorithm_config = None
    st.session_state.room5_test_results = None


def build_room5_environment() -> Room5Environment:
    controls = st.session_state.room5_environment_controls
    return Room5Environment(
        Room5Config(
            lane_count=int(controls["lane_count"]),
            vision_distance=float(controls["vision_distance"]),
            road_length=float(controls["road_length"]),
            ego_speed=float(controls["ego_speed"]),
            traffic_speed_min=float(controls["traffic_speed_min"]),
            traffic_speed_max=float(controls["traffic_speed_max"]),
            traffic_count=int(controls["traffic_count"]),
            rewards=dict(st.session_state.room5_reward_values),
            seed=int(controls["seed"]),
        )
    )


def _room5_action_dataframe(action_counts: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Action": [action.name for action in Action5],
            "Selections": [int(action_counts.get(action.name, 0)) for action in Action5],
        }
    )


def render_room5_controls() -> tuple[str, dict[str, bool], bool]:
    st.sidebar.title("Room 5 Controls (PPO)")
    section = st.sidebar.radio(
        "Control section",
        ["Environment", "Training", "Testing", "Models"],
        key="room5_control_section",
        help=ROOM5_HELP["section"],
    )
    requests = {"train": False, "reset": False}
    run_test = False

    if section == "Environment":
        controls = dict(st.session_state.room5_environment_controls)
        st.sidebar.subheader("One-way road")
        lane_count = st.sidebar.slider("Number of lanes", 2, 6, int(controls["lane_count"]), key="r5_lanes", help=ROOM5_HELP["lanes"])
        vision = st.sidebar.slider("Forward vision distance (m)", 40.0, 200.0, float(controls["vision_distance"]), 5.0, key="r5_vision", help=ROOM5_HELP["vision"])
        road_length = st.sidebar.slider("Road completion distance (m)", 200.0, 2000.0, float(controls["road_length"]), 50.0, key="r5_road_length", help=ROOM5_HELP["road_length"])
        traffic_count = st.sidebar.slider("Traffic car count", 4, 30, int(controls["traffic_count"]), key="r5_traffic_count", help=ROOM5_HELP["traffic_count"])
        ego_speed = st.sidebar.slider("Agent speed (m/s)", 25.0, 40.0, float(controls["ego_speed"]), 1.0, key="r5_ego_speed", help=ROOM5_HELP["ego_speed"])
        traffic_min = st.sidebar.slider("Minimum traffic speed (m/s)", 5.0, 20.0, float(controls["traffic_speed_min"]), 1.0, key="r5_traffic_min", help=ROOM5_HELP["traffic_min"])
        traffic_max = st.sidebar.slider("Maximum traffic speed (m/s)", 21.0, 24.0, float(controls["traffic_speed_max"]), 1.0, key="r5_traffic_max", help=ROOM5_HELP["traffic_max"])
        env_seed = st.sidebar.number_input("Environment seed", 0, value=int(controls["seed"]), key="r5_env_seed", help=ROOM5_HELP["env_seed"])

        next_controls = {
            "lane_count": int(lane_count),
            "vision_distance": float(vision),
            "road_length": float(road_length),
            "ego_speed": float(ego_speed),
            "traffic_speed_min": float(traffic_min),
            "traffic_speed_max": float(traffic_max),
            "traffic_count": int(traffic_count),
            "seed": int(env_seed),
        }

        st.sidebar.subheader("Reward structure")
        rewards = dict(st.session_state.room5_reward_values)
        next_rewards = {
            "step": float(st.sidebar.slider("Step reward", -1.0, 0.0, float(rewards["step"]), 0.01, key="r5_step_reward", help=ROOM5_HELP["step"])),
            "forward_progress": float(st.sidebar.slider("Forward progress reward per meter", 0.0, 1.0, float(rewards["forward_progress"]), 0.01, key="r5_progress_reward", help=ROOM5_HELP["progress"])),
            "overtake": float(st.sidebar.slider("Overtake reward", 0.0, 30.0, float(rewards["overtake"]), 0.5, key="r5_overtake_reward", help=ROOM5_HELP["overtake"])),
            "lane_change": float(st.sidebar.slider("Lane-change reward", -2.0, 0.0, float(rewards["lane_change"]), 0.05, key="r5_lane_change_reward", help=ROOM5_HELP["lane_change"])),
            "invalid_lane_change": float(st.sidebar.slider("Invalid lane-change penalty", -10.0, 0.0, float(rewards["invalid_lane_change"]), 0.5, key="r5_invalid_change_reward", help=ROOM5_HELP["invalid_change"])),
            "collision": float(st.sidebar.slider("Collision penalty", -100.0, -1.0, float(rewards["collision"]), 1.0, key="r5_collision_reward", help=ROOM5_HELP["collision"])),
            "goal_reached": float(st.sidebar.slider("Road completion reward", 5.0, 100.0, float(rewards["goal_reached"]), 1.0, key="r5_goal_reward", help=ROOM5_HELP["goal"])),
        }
        if next_controls != controls or next_rewards != rewards:
            st.session_state.room5_environment_controls = next_controls
            st.session_state.room5_reward_values = next_rewards
            _invalidate_room5_model()

    elif section == "Training":
        controls = st.session_state.room5_training_controls
        st.sidebar.subheader("PPO hyperparameters")
        alpha = st.sidebar.number_input("Learning rate", 0.00001, 0.01, float(controls["alpha"]), format="%.5f", key="r5_alpha", help=ROOM5_HELP["alpha"])
        gamma = st.sidebar.number_input("Discount factor", 0.0, 1.0, float(controls["gamma"]), format="%.3f", key="r5_gamma", help=ROOM5_HELP["gamma"])
        gae_lambda = st.sidebar.number_input("GAE lambda", 0.0, 1.0, float(controls["gae_lambda"]), format="%.3f", key="r5_gae", help=ROOM5_HELP["gae"])
        clip_epsilon = st.sidebar.number_input("PPO clip epsilon", 0.01, 0.5, float(controls["clip_epsilon"]), format="%.3f", key="r5_clip", help=ROOM5_HELP["clip"])
        entropy_coefficient = st.sidebar.number_input("Entropy coefficient", 0.0, 0.2, float(controls["entropy_coefficient"]), format="%.3f", key="r5_entropy_coef", help=ROOM5_HELP["entropy"])
        value_coefficient = st.sidebar.number_input("Value-loss coefficient", 0.0, 2.0, float(controls["value_coefficient"]), format="%.2f", key="r5_value_coef", help=ROOM5_HELP["value_coef"])
        update_epochs = st.sidebar.number_input("Update epochs per rollout", 1, 20, int(controls["update_epochs"]), key="r5_update_epochs", help=ROOM5_HELP["epochs"])
        mini_batch_size = st.sidebar.select_slider("Mini-batch size", [16, 32, 64, 128, 256], value=int(controls["mini_batch_size"]), key="r5_batch", help=ROOM5_HELP["batch"])
        episodes = st.sidebar.number_input("Episodes", 1, 5000, int(controls["episodes"]), key="r5_episodes", help=ROOM5_HELP["episodes"])
        max_timesteps = st.sidebar.number_input("Maximum timesteps", 10, 2000, int(controls["max_timesteps"]), key="r5_max_steps", help=ROOM5_HELP["timesteps"])

        st.sidebar.subheader("Actor-critic network")
        hidden_layers = st.sidebar.slider("Hidden layer count", 1, 4, int(controls["hidden_layers"]), key="r5_hidden_layers", help=ROOM5_HELP["layers"])
        hidden_units = st.sidebar.select_slider("Neurons per hidden layer", [32, 64, 128, 256], value=int(controls["hidden_units"]), key="r5_hidden_units", help=ROOM5_HELP["units"])
        activations = ["Tanh", "ReLU", "LeakyReLU", "ELU", "SiLU"]
        activation = st.sidebar.selectbox("Activation function", activations, index=activations.index(controls["activation_fn"]), key="r5_activation", help=ROOM5_HELP["activation"])
        seed = st.sidebar.number_input("Training seed", 0, value=int(controls["seed"]), key="r5_train_seed", help=ROOM5_HELP["train_seed"])
        live_update = st.sidebar.number_input("Update charts every N episodes", 1, 100, int(controls["live_update_every"]), key="r5_live_update", help=ROOM5_HELP["live"])
        st.session_state.room5_training_controls = {
            "alpha": float(alpha), "gamma": float(gamma), "gae_lambda": float(gae_lambda),
            "clip_epsilon": float(clip_epsilon), "entropy_coefficient": float(entropy_coefficient),
            "value_coefficient": float(value_coefficient), "update_epochs": int(update_epochs),
            "mini_batch_size": int(mini_batch_size), "episodes": int(episodes),
            "max_timesteps": int(max_timesteps), "hidden_layers": int(hidden_layers),
            "hidden_units": int(hidden_units), "activation_fn": str(activation),
            "seed": int(seed), "live_update_every": int(live_update),
        }
        requests["train"] = st.sidebar.button("Train PPO agent", icon=":material/play_arrow:", type="primary", width="stretch", key="r5_train", help="Start PPO training with the current road, rewards, and hyperparameters.")
        requests["reset"] = st.sidebar.button("Reset trained model", icon=":material/restart_alt:", width="stretch", key="r5_reset", help="Remove the Room 5 model and test results from this browser session.")

    elif section == "Testing":
        controls = st.session_state.room5_test_controls
        st.sidebar.subheader("Test configuration")
        episodes = st.sidebar.number_input("Test episodes", 1, 1000, int(controls["episodes"]), key="r5_test_episodes", help=ROOM5_HELP["test_episodes"])
        max_timesteps = st.sidebar.number_input("Maximum timesteps per episode", 10, 5000, int(controls["max_timesteps"]), key="r5_test_steps", help=ROOM5_HELP["test_steps"])
        seed = st.sidebar.number_input("Test seed", 0, value=int(controls["seed"]), key="r5_test_seed", help=ROOM5_HELP["test_seed"])
        st.session_state.room5_test_controls = {"episodes": int(episodes), "max_timesteps": int(max_timesteps), "seed": int(seed)}
        run_test = st.sidebar.button("Run test", icon=":material/science:", type="primary", width="stretch", disabled=st.session_state.room5_result is None, key="r5_run_test", help="Evaluate the trained PPO policy greedily and record metrics and replay trajectories.")

    else:
        result = st.session_state.room5_result
        environment = st.session_state.room5_result_environment
        algorithm_config = st.session_state.room5_algorithm_config
        st.sidebar.subheader("Model artifact")
        if result and environment and algorithm_config:
            st.sidebar.download_button("Download Room 5 model (JSON)", export_room5_artifact(environment, algorithm_config, result), "room5_ppo_model.json", "application/json", icon=":material/download:", width="stretch", help="Download the PPO weights, environment, hyperparameters, and training metrics.")
        uploaded = st.sidebar.file_uploader("Upload Room 5 model JSON", type=["json"], key="r5_upload", help="Select a Room 5 PPO artifact exported by this dashboard.")
        if uploaded is not None and st.sidebar.button("Load model", icon=":material/upload_file:", width="stretch", key="r5_load", help="Validate and restore the uploaded Room 5 model."):
            try:
                environment, algorithm_config, result = import_room5_artifact(uploaded.getvalue().decode("utf-8"))
            except Exception as exc:
                st.sidebar.error(f"Invalid artifact: {exc}")
            else:
                st.session_state.room5_result = result
                st.session_state.room5_result_environment = environment
                st.session_state.room5_algorithm_config = algorithm_config
                st.session_state.room5_test_results = None
                config = environment.config
                st.session_state.room5_environment_controls = {
                    "lane_count": config.lane_count, "vision_distance": config.vision_distance,
                    "road_length": config.road_length, "ego_speed": config.ego_speed,
                    "traffic_speed_min": config.traffic_speed_min, "traffic_speed_max": config.traffic_speed_max,
                    "traffic_count": config.traffic_count, "seed": config.seed,
                }
                st.session_state.room5_reward_values = dict(config.rewards)
                st.sidebar.success("Room 5 PPO model loaded successfully.")
                st.rerun()

    return section, requests, run_test


def _render_room5_training_summary(result: Any) -> None:
    metrics = pd.DataFrame([asdict(metric) for metric in result.metrics])
    kpis = st.columns(4)
    kpis[0].metric("Training duration", _format_training_duration(float(result.training_duration_seconds)))
    kpis[1].metric("Episodes", result.episodes_run)
    kpis[2].metric("Total overtakes", int(metrics["overtakes"].sum()))
    kpis[3].metric("Late success", "Yes" if result.converged else "No")

    st.subheader("Training metrics")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.caption("**Total reward per episode**")
        st.line_chart(metrics, x="episode", y="total_reward", x_label="Episode", y_label="Total reward")
        st.caption("**Overtakes per episode**")
        st.line_chart(metrics, x="episode", y="overtakes", x_label="Episode", y_label="Overtakes")
    with chart_columns[1]:
        st.caption("**PPO policy and value loss**")
        st.line_chart(metrics, x="episode", y=["policy_loss", "value_loss"], x_label="Episode", y_label="Loss")
        st.caption("**Policy entropy**")
        st.line_chart(metrics, x="episode", y="entropy", x_label="Episode", y_label="Entropy")

    st.subheader("Training action distribution")
    st.bar_chart(_room5_action_dataframe(result.action_counts), x="Action", y="Selections", x_label="Action", y_label="Number of selections")

    if hasattr(result, "training_episodes") and result.training_episodes:
        environment = st.session_state.room5_result_environment or build_room5_environment()
        render_episode_replay_visualizer(
            environment,
            result.training_episodes,
            "room5_tr_replay",
            5,
            title="Training Episodes Replay",
        )



def render_room5() -> None:
    section, requests, run_test = render_room5_controls()
    st.markdown('<div class="room-header"><h2>🚘 Room 5: One-way traffic avoidance (PPO)</h2></div>', unsafe_allow_html=True)

    if section == "Environment":
        environment = build_room5_environment()
        st.subheader("Configurable one-way road")
        st.markdown(render_room5_html(environment), unsafe_allow_html=True)
        controls = environment.config
        with st.container(horizontal=True):
            st.metric("Lanes", controls.lane_count, border=True)
            st.metric("Vision", f"{controls.vision_distance:.0f}m", border=True)
            st.metric("Traffic cars", controls.traffic_count, border=True)
            st.metric("Goal distance", f"{controls.road_length:.0f}m", border=True)
        st.caption(
            "All traffic travels in the same direction. Because the agent is faster, "
            "slower vehicles move toward it in the agent-relative view and must be avoided or overtaken."
        )

    elif section == "Training":
        if requests["reset"]:
            _invalidate_room5_model()
            st.success("Room 5 trained model reset.")
            st.rerun()

        if requests["train"]:
            environment = build_room5_environment()
            controls = st.session_state.room5_training_controls
            config = PPOConfig(
                alpha=controls["alpha"], gamma=controls["gamma"],
                gae_lambda=controls["gae_lambda"], clip_epsilon=controls["clip_epsilon"],
                entropy_coefficient=controls["entropy_coefficient"],
                value_coefficient=controls["value_coefficient"], update_epochs=controls["update_epochs"],
                mini_batch_size=controls["mini_batch_size"], episodes=controls["episodes"],
                max_timesteps=controls["max_timesteps"],
                hidden_dims=tuple([controls["hidden_units"]] * controls["hidden_layers"]),
                activation_fn=controls["activation_fn"], seed=controls["seed"],
            )
            status = st.empty()
            st.subheader("Live training metrics")
            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.caption("**Total reward per episode**")
                reward_slot = st.empty()
                st.caption("**Overtakes per episode**")
                overtake_slot = st.empty()
            with chart_columns[1]:
                st.caption("**Policy and value loss**")
                loss_slot = st.empty()
                st.caption("**Policy entropy**")
                entropy_slot = st.empty()
            live_rows: list[dict[str, Any]] = []

            def room5_live_callback(metric: Any, policy_net: Any) -> None:
                live_rows.append(asdict(metric))
                if metric.episode % controls["live_update_every"] == 0 or metric.episode == config.episodes:
                    status.info(
                        f"Training episode {metric.episode}/{config.episodes} • "
                        f"Reward: {metric.total_reward:.2f} • Overtakes: {metric.overtakes} • "
                        f"Entropy: {metric.entropy:.3f}"
                    )
                    frame = pd.DataFrame(live_rows)
                    reward_slot.line_chart(frame, x="episode", y="total_reward", x_label="Episode", y_label="Total reward")
                    overtake_slot.line_chart(frame, x="episode", y="overtakes", x_label="Episode", y_label="Overtakes")
                    loss_slot.line_chart(frame, x="episode", y=["policy_loss", "value_loss"], x_label="Episode", y_label="Loss")
                    entropy_slot.line_chart(frame, x="episode", y="entropy", x_label="Episode", y_label="Entropy")

            with st.spinner("Training PPO agent..."):
                result = run_ppo(environment, config, callback=room5_live_callback)
            st.session_state.room5_result = result
            st.session_state.room5_result_environment = environment
            st.session_state.room5_algorithm_config = config
            st.session_state.room5_test_results = None
            status.success(f"PPO training complete in {_format_training_duration(result.training_duration_seconds)}.")
            st.subheader("Training action distribution")
            st.bar_chart(_room5_action_dataframe(result.action_counts), x="Action", y="Selections", x_label="Action", y_label="Number of selections")

        result = st.session_state.room5_result
        if result is not None and not requests["train"]:
            _render_room5_training_summary(result)
        elif result is None and not requests["train"]:
            st.info("Use the left sidebar to configure and train the Room 5 PPO agent.")

    elif section == "Testing":
        if run_test:
            result = st.session_state.room5_result
            environment = st.session_state.room5_result_environment
            if result and environment:
                controls = st.session_state.room5_test_controls
                with st.spinner("Evaluating PPO policy..."):
                    st.session_state.room5_test_results = evaluate_room5_ppo(
                        environment, result.policy_net,
                        episodes=controls["episodes"], max_timesteps=controls["max_timesteps"], seed=controls["seed"],
                    )

        results = st.session_state.room5_test_results
        environment = st.session_state.room5_result_environment
        if results and environment:
            table = pd.DataFrame(
                [{"Episode": item.episode, "Success": item.success, "Collision": item.collision,
                  "Timesteps": item.timesteps, "Total reward": item.total_reward, "Overtakes": item.overtakes}
                 for item in results]
            )
            with st.container(horizontal=True):
                st.metric("Success rate", f"{100.0 * table['Success'].mean():.1f}%", border=True)
                st.metric("Collision rate", f"{100.0 * table['Collision'].mean():.1f}%", border=True)
                st.metric("Mean overtakes", f"{table['Overtakes'].mean():.2f}", border=True)
                st.metric("Mean reward", f"{table['Total reward'].mean():.2f}", border=True)
            charts = st.columns(2)
            with charts[0]:
                st.caption("**Test reward by episode**")
                st.line_chart(table, x="Episode", y="Total reward", x_label="Episode", y_label="Total reward")
            action_counts = {action.name: 0 for action in Action5}
            for episode in results:
                for step in episode.trajectory:
                    action_counts[step.action.name] += 1
            with charts[1]:
                st.caption("**Test action distribution**")
                st.bar_chart(_room5_action_dataframe(action_counts), x="Action", y="Selections", x_label="Action", y_label="Number of selections")
            st.dataframe(table, width="stretch")

            render_episode_replay_visualizer(
                environment,
                results,
                "room5_te_replay",
                5,
                title="Test Episodes Replay & Animation",
            )
        else:
            st.info("Train a Room 5 model and run a test to view evaluation metrics and replay trajectories.")


    else:
        result = st.session_state.room5_result
        config = st.session_state.room5_algorithm_config
        st.subheader("Model and network information")
        if result and config:
            st.json({
                "Algorithm": "PPO (Proximal Policy Optimization)",
                "Observation dimension": 19,
                "Actions": [action.name for action in Action5],
                "Hidden architecture": list(config.hidden_dims),
                "Activation function": config.activation_fn,
                "Learning rate": config.alpha,
                "Discount factor": config.gamma,
                "GAE lambda": config.gae_lambda,
                "Clip epsilon": config.clip_epsilon,
                "Episodes trained": result.episodes_run,
                "Training duration seconds": result.training_duration_seconds,
                "Action counts": result.action_counts,
                "Converged": result.converged,
            })
        else:
            st.info("No trained Room 5 model is currently in memory.")


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


# Execution Entry Point
initialize_state()
active_room = render_room_navigation()
if active_room == "Room 1":
    render_room(1)
elif active_room == "Room 2":
    render_room(2)
elif active_room == "Room 3":
    render_room(3)
elif active_room == "Room 4":
    render_room4()
elif active_room == "Room 5":
    render_room5()
else:
    render_future_room(active_room)

