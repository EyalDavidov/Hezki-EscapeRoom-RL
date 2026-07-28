"""Streamlit dashboard for the Escape Room RL project."""

from __future__ import annotations

import sys
from dataclasses import asdict
from math import isclose
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
    generate_random_slippery_cells,
    slippery_candidates,
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
        "room1_random_controls": {"count": 8, "seed": 42},
        # Room 2 (SARSA)
        "room2_walls": set(DEFAULT_ROOM2_WALLS),
        "room2_slippery": {},
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
        "room2_random_controls": {"count": 6, "seed": 42},
        # Room 3 (Q-Learning)
        "room3_walls": set(DEFAULT_ROOM3_WALLS),
        "room3_slippery": {},
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
        "room3_random_controls": {"count": 6, "seed": 42},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
        for column, room in zip(columns, rooms, strict=True):
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
    if state in ((0, 0), (9, 9)):
        return
    previous_type = current_cell_type(room_num, state)
    if selected_type == previous_type:
        return

    st.session_state[f"{p}_walls"].discard(state)
    st.session_state[f"{p}_slippery"].pop(state, None)
    st.session_state[f"{p}_probability_errors"].discard(state)
    if selected_type == "Wall":
        st.session_state[f"{p}_walls"].add(state)
    elif selected_type == "Icy":
        st.session_state[f"{p}_slippery"][state] = SlipperyCell()
    invalidate_room_model(room_num)


def build_environment(room_num: int) -> Any:
    p = room_prefix(room_num)
    walls = frozenset(st.session_state[f"{p}_walls"])
    slippery = dict(st.session_state[f"{p}_slippery"])
    rewards = dict(st.session_state[f"{p}_reward_values"])
    if room_num == 1:
        return Room1Environment(Room1Config(walls=walls, slippery=slippery, rewards=rewards))
    elif room_num == 2:
        return Room2Environment(Room2Config(walls=walls, slippery=slippery, rewards=rewards))
    else:
        return Room3Environment(Room3Config(walls=walls, slippery=slippery, rewards=rewards))


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
    """Render the interactive 10x10 map with cell popovers to edit cell type and probabilities."""
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

    cell_styles: list[str] = []
    for y in reversed(range(10)):
        for x in reversed(range(10)):
            state = (x, y)
            cell_type = current_cell_type(room_num, state)
            icon = ""
            background = "#f7f9fb"
            coordinate_color = "#263238"
            outline = "none"
            if state == (0, 0):
                icon = "🐕"
                background = "#fff3cd"
                outline = "inset 0 0 0 3px #43a047"
            elif state == (9, 9):
                icon = "🚪"
                outline = "inset 0 0 0 3px #f9a825"
            elif cell_type == "Wall":
                icon = "🧱"
                background = "#455a64"
                coordinate_color = "white"
            elif cell_type == "Icy":
                icon = "❄️"
                background = "#dff6ff"

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
                protected_label = None
                if state == (0, 0):
                    cell_label = f"🐕 {x},{y}"
                    protected_label = "This is the protected agent start cell."
                elif state == (9, 9):
                    cell_label = f"🚪 {x},{y}"
                    protected_label = "This is the protected goal cell."
                else:
                    cell_label = f"{icon_by_type[current_cell_type(room_num, state)]} {x},{y}"

                with column:
                    with st.container(key=f"{p}_cell_{x}_{y}"):
                        with st.popover(cell_label, use_container_width=True):
                            st.markdown(f"**Cell {state_label(state)}**")
                            if protected_label is not None:
                                st.info(protected_label)
                                st.radio(
                                    "Cell type",
                                    options=["Normal"],
                                    disabled=True,
                                    key=f"protected_cell_{p}_{nonce}_{x}_{y}",
                                )
                                continue

                            type_key = f"cell_type_{p}_{nonce}_{x}_{y}"
                            cell_type = current_cell_type(room_num, state)
                            selected_type = st.radio(
                                "Cell type",
                                options=["Normal", "Icy", "Wall"],
                                index=["Normal", "Icy", "Wall"].index(cell_type),
                                horizontal=True,
                                key=type_key,
                                on_change=apply_cell_type,
                                args=(state, type_key, room_num),
                            )
                            if selected_type != "Icy":
                                continue

                            current = st.session_state[f"{p}_slippery"][state]
                            st.caption(
                                "Set the complete outcome distribution. The five values must total 100%."
                            )
                            percentages: dict[str, float] = {}
                            for outcome in SLIP_OUTCOMES:
                                percentages[outcome] = st.number_input(
                                    f"{outcome_labels[outcome]} (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=float(getattr(current, outcome) * 100.0),
                                    step=1.0,
                                    format="%.1f",
                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_{outcome}",
                                )
                            total = sum(percentages.values())
                            st.metric("Probability total", f"{total:.1f}%")
                            if not isclose(total, 100.0, abs_tol=1e-9):
                                st.session_state[f"{p}_probability_errors"].add(state)
                                st.error("The probabilities must total exactly 100%.")
                                continue

                            st.session_state[f"{p}_probability_errors"].discard(state)
                            updated = SlipperyCell(
                                **{
                                    outcome: percentage / 100.0
                                    for outcome, percentage in percentages.items()
                                }
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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Grid cells", 100)
    m2.metric("Walkable states", 100 - len(walls))
    m3.metric("Walls", len(walls))
    m4.metric("Icy cells", len(slippery))
    st.markdown(
        '<div class="legend-row">'
        '<span class="legend-item">🐕 Agent / start</span>'
        '<span class="legend-item">🚪 Goal</span>'
        '<span class="legend-item">🧱 Wall</span>'
        '<span class="legend-item">❄️ Icy cell</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Select any cell in the grid below to open its popover editor. Start (0,0) and goal (9,9) are protected.")
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
        chart_slot_1 = st.empty()
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
        st.line_chart(frame.set_index(x_col)[[y_col]])
    with c2:
        st.subheader("Policy Changes / Epsilon Decay")
        y_col2 = "policy_changes" if room_num == 1 else "epsilon"
        x_col2 = "policy_iteration" if room_num == 1 else "episode"
        st.line_chart(frame.set_index(x_col2)[[y_col2]])

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
        st.bar_chart(frame.set_index("episode")[["timesteps"]])
    with c2:
        st.subheader("Reward by episode")
        st.line_chart(frame.set_index("episode")[["total_reward"]])

    st.subheader("Detailed Episode Results")
    st.dataframe(frame, hide_index=True, use_container_width=True)

    st.subheader("Episode Replay & Animation")
    selected_number = st.selectbox(
        "Select episode to replay",
        options=[episode.episode for episode in test_results],
        key=f"{p}_replay_ep_select",
    )
    selected_episode = next(
        episode for episode in test_results if episode.episode == selected_number
    )
    replay_step = st.slider(
        "Replay timestep",
        min_value=0,
        max_value=len(selected_episode.trajectory),
        value=0,
        key=f"{p}_replay_step_slider",
    )
    if replay_step == 0:
        agent_state = env.start
        st.write("Initial state `(0, 0)`")
    else:
        step = selected_episode.trajectory[replay_step - 1]
        agent_state = step.next_state
        st.write(
            f"Action: `{step.action.value}` • outcome: `{step.outcome}` • "
            f"reward: {step.reward:.3f} • cumulative: {step.cumulative_reward:.3f}"
        )

    st.markdown(
        render_grid_html(
            env,
            agent_state=agent_state,
            policy=res.policy,
            values=res.values,
        ),
        unsafe_allow_html=True,
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
    )
    st.sidebar.divider()

    # Environment controls in sidebar
    base_configs = {1: Room1Config, 2: Room2Config, 3: Room3Config}
    defaults_walls = {1: DEFAULT_WALLS, 2: DEFAULT_ROOM2_WALLS, 3: DEFAULT_ROOM3_WALLS}
    base_config = base_configs[room_num](walls=frozenset(st.session_state[f"{p}_walls"]))
    candidates = slippery_candidates(base_config)
    random_controls = st.session_state[f"{p}_random_controls"]

    prob_err = room_configuration_error(room_num)
    requests = {"train": False, "reset": False}
    run_test = False

    if section == "Environment":
        st.sidebar.subheader("Grid layout")
        if st.sidebar.button("Reset grid to default", key=f"{p}_reset_grid", use_container_width=True):
            st.session_state[f"{p}_walls"] = set(defaults_walls[room_num])
            st.session_state[f"{p}_slippery"] = {}
            st.session_state[f"{p}_probability_errors"] = set()
            st.session_state[f"{p}_editor_nonce"] += 1
            invalidate_room_model(room_num)
            st.rerun()

        st.sidebar.subheader("Random ice generator")
        random_count = st.sidebar.number_input(
            "Number of icy cells", 0, len(candidates), int(random_controls["count"]), key=f"{p}_random_count"
        )
        random_seed = st.sidebar.number_input(
            "Generator seed", 0, value=int(random_controls["seed"]), key=f"{p}_random_seed"
        )
        st.session_state[f"{p}_random_controls"] = {"count": int(random_count), "seed": int(random_seed)}
        if st.sidebar.button("🎲 Generate icy cells", key=f"{p}_gen_ice", use_container_width=True):
            st.session_state[f"{p}_slippery"] = generate_random_slippery_cells(
                base_config, int(random_count), int(random_seed)
            )
            st.session_state[f"{p}_editor_nonce"] += 1
            invalidate_room_model(room_num)
            st.rerun()

        st.sidebar.subheader("Reward configuration")
        for event, description in SUPPORTED_REWARD_EVENTS.items():
            enabled = st.sidebar.checkbox(
                description,
                value=event in st.session_state[f"{p}_reward_enabled"],
                key=f"{p}_reward_enabled_{event}",
            )
            value = st.sidebar.number_input(
                f"{description} value",
                value=float(st.session_state[f"{p}_reward_values"].get(event, 0.0)),
                step=0.1,
                format="%.3f",
                disabled=not enabled,
                key=f"{p}_reward_value_{event}",
            )
            if enabled:
                st.session_state[f"{p}_reward_enabled"].add(event)
                st.session_state[f"{p}_reward_values"][event] = float(value)
            else:
                st.session_state[f"{p}_reward_enabled"].discard(event)
                st.session_state[f"{p}_reward_values"][event] = 0.0

    elif section == "Training":
        controls = st.session_state[f"{p}_training_controls"]
        st.sidebar.subheader(algo_names[room_num])
        if room_num == 1:
            gamma = st.sidebar.slider("Gamma", 0.0, 0.999, float(controls["gamma"]), 0.001, key=f"{p}_gamma")
            theta = st.sidebar.number_input("Theta", 1e-12, 1.0, float(controls["theta"]), format="%.8f", key=f"{p}_theta")
            max_pi = st.sidebar.number_input("Max policy iterations", 1, 1000, int(controls["max_policy_iterations"]), key=f"{p}_max_pi")
            max_sweeps = st.sidebar.number_input("Max evaluation sweeps", 1, 100000, int(controls["max_evaluation_sweeps"]), key=f"{p}_max_sweeps")
            seed = st.sidebar.number_input("Seed", 0, value=int(controls["seed"]), key=f"{p}_seed")
            live_update = st.sidebar.number_input("Update charts every N sweeps", 1, 1000, int(controls["live_update_every"]), key=f"{p}_live_update")
            st.session_state[f"{p}_training_controls"] = {
                "gamma": float(gamma), "theta": float(theta),
                "max_policy_iterations": int(max_pi), "max_evaluation_sweeps": int(max_sweeps),
                "seed": int(seed), "live_update_every": int(live_update),
            }
        else:
            alpha = st.sidebar.slider("Alpha (learning rate)", 0.01, 1.0, float(controls["alpha"]), 0.01, key=f"{p}_alpha")
            gamma = st.sidebar.slider("Gamma (discount)", 0.0, 0.999, float(controls["gamma"]), 0.001, key=f"{p}_gamma")
            eps_start = st.sidebar.slider("Epsilon start", 0.05, 1.0, float(controls["epsilon_start"]), 0.05, key=f"{p}_eps_start")
            eps_min = st.sidebar.slider("Epsilon min", 0.01, 0.5, float(controls["epsilon_min"]), 0.01, key=f"{p}_eps_min")
            eps_decay = st.sidebar.number_input("Epsilon decay rate", 0.8, 1.0, float(controls["epsilon_decay"]), format="%.4f", key=f"{p}_eps_decay")
            episodes = st.sidebar.number_input("Training episodes", 10, 10000, int(controls["episodes"]), key=f"{p}_episodes")
            max_steps = st.sidebar.number_input("Max timesteps per episode", 10, 5000, int(controls["max_timesteps"]), key=f"{p}_max_steps")
            seed = st.sidebar.number_input("Seed", 0, value=int(controls["seed"]), key=f"{p}_seed")
            live_update = st.sidebar.number_input("Update charts every N episodes", 1, 1000, int(controls["live_update_every"]), key=f"{p}_live_update")
            st.session_state[f"{p}_training_controls"] = {
                "alpha": float(alpha), "gamma": float(gamma),
                "epsilon_start": float(eps_start), "epsilon_min": float(eps_min),
                "epsilon_decay": float(eps_decay), "episodes": int(episodes),
                "max_timesteps": int(max_steps), "seed": int(seed),
                "live_update_every": int(live_update),
            }

        if prob_err is not None:
            st.sidebar.error(f"Fix grid error: {prob_err}")
        requests["train"] = st.sidebar.button("▶ Train / compute policy", type="primary", use_container_width=True, disabled=prob_err is not None, key=f"{p}_train_btn")
        requests["reset"] = st.sidebar.button("Reset trained model", use_container_width=True, key=f"{p}_reset_btn")

    elif section == "Testing":
        controls = st.session_state[f"{p}_test_controls"]
        st.sidebar.subheader("Test configuration")
        episodes = st.sidebar.number_input("Test episodes", 1, 10000, int(controls["episodes"]), key=f"{p}_test_episodes")
        max_steps = st.sidebar.number_input("Max timesteps per episode", 1, 50000, int(controls["max_timesteps"]), key=f"{p}_test_max_steps")
        seed = st.sidebar.number_input("Test seed", 0, value=int(controls["seed"]), key=f"{p}_test_seed")
        st.session_state[f"{p}_test_controls"] = {"episodes": int(episodes), "max_timesteps": int(max_steps), "seed": int(seed)}
        run_test = st.sidebar.button("🧪 Run test", type="primary", use_container_width=True, disabled=st.session_state[f"{p}_result"] is None, key=f"{p}_run_test_btn")

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
            st.sidebar.download_button(f"⬇ Download Room {room_num} model", data=art, file_name=f"room{room_num}_model.json", mime="application/json", use_container_width=True)

        uploaded = st.sidebar.file_uploader("Upload model JSON", type=["json"], key=f"{p}_upload")
        if uploaded is not None and st.sidebar.button("Load model", use_container_width=True, key=f"{p}_load_btn"):
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
                st.session_state[f"{p}_walls"] = set(env_l.config.walls)
                st.session_state[f"{p}_slippery"] = dict(env_l.config.slippery)
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
