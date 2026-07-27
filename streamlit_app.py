"""Streamlit dashboard for the Escape Room RL project."""

from __future__ import annotations

import sys
from dataclasses import asdict
from math import isclose
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from escape_room_rl.artifacts import (  # noqa: E402
    export_room1_artifact,
    import_room1_artifact,
)
from escape_room_rl.evaluation import evaluate_policy  # noqa: E402
from escape_room_rl.policy_iteration import (  # noqa: E402
    PolicyIterationConfig,
    run_policy_iteration,
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
      [data-testid="stSidebar"] { min-width: 350px; max-width: 350px; }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 { letter-spacing: -0.02em; }
      .room-header {
        padding: 0.4rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
      }
      .legend-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
      .legend-item { padding: 0.25rem 0.65rem; border-radius: 999px; background: rgba(128,128,128,0.12); }
      .st-key-room1_grid_editor {
        width: 100%;
        max-width: 720px;
        border: 3px solid #263238;
        border-radius: 8px;
        overflow: hidden;
        background: #f7f9fb;
      }
      .st-key-room1_grid_editor [data-testid="stVerticalBlock"] { gap: 0; }
      .st-key-room1_grid_editor [data-testid="stHorizontalBlock"] { gap: 0; }
      .st-key-room1_grid_editor [data-testid="stPopover"] button {
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
      .st-key-room1_grid_editor [data-testid="stPopover"] button:hover {
        filter: brightness(0.94);
        z-index: 1;
        box-shadow: inset 0 0 0 2px #ff4b4b;
      }
      .st-key-room1_grid_editor [data-testid="stPopover"] button p { font-size: 0; }
      .st-key-room1_grid_editor [data-testid="stPopover"] button::before {
        font-size: clamp(18px, 2.2vw, 30px);
        line-height: 1;
      }
      .st-key-room1_grid_editor [data-testid="stPopover"] button::after {
        position: absolute;
        right: 3px;
        bottom: 1px;
        font-size: 9px;
        line-height: 1;
        color: #263238;
        opacity: 0.75;
      }
      .st-key-room1_grid_editor [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
        display: none;
      }
      .st-key-room1_grid_editor [data-testid="column"] { min-width: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
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


def build_room1_environment() -> Room1Environment:
    return Room1Environment(
        Room1Config(
            walls=frozenset(st.session_state.room1_walls),
            slippery=dict(st.session_state.room1_slippery),
            rewards=dict(st.session_state.room1_reward_values),
        )
    )


def invalidate_room1_model() -> None:
    """Discard results that belong to an older grid configuration."""

    st.session_state.room1_result = None
    st.session_state.room1_result_environment = None
    st.session_state.room1_algorithm_config = None
    st.session_state.room1_test_results = None


def current_cell_type(state: tuple[int, int]) -> str:
    if state in st.session_state.room1_walls:
        return "Wall"
    if state in st.session_state.room1_slippery:
        return "Icy"
    return "Normal"


def apply_cell_type(state: tuple[int, int], widget_key: str) -> None:
    """Apply a grid-cell type selected inside a cell popover."""

    selected_type = st.session_state[widget_key]
    if state in ((0, 0), (9, 9)):
        return
    previous_type = current_cell_type(state)
    if selected_type == previous_type:
        return

    st.session_state.room1_walls.discard(state)
    st.session_state.room1_slippery.pop(state, None)
    st.session_state.room1_probability_errors.discard(state)
    if selected_type == "Wall":
        st.session_state.room1_walls.add(state)
    elif selected_type == "Icy":
        st.session_state.room1_slippery[state] = SlipperyCell()
    invalidate_room1_model()


def room1_configuration_error() -> str | None:
    if st.session_state.room1_probability_errors:
        cells = ", ".join(
            state_label(state)
            for state in sorted(st.session_state.room1_probability_errors)
        )
        return f"Icy-cell probabilities must total 100%: {cells}."
    try:
        build_room1_environment()
    except ValueError as exc:
        return str(exc)
    return None


def render_room_navigation() -> str:
    st.markdown("### Room navigation")
    selected = st.radio(
        "Room navigation",
        options=["Room 1", "Room 2", "Room 3", "Room 4"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_room",
    )
    st.divider()
    return selected


def render_room1_environment_controls() -> str | None:
    base_config = Room1Config(walls=frozenset(st.session_state.room1_walls))
    candidates = slippery_candidates(base_config)
    random_controls = st.session_state.room1_random_controls

    st.subheader("Grid layout")
    st.caption("Click any grid cell in the main panel to edit its type.")
    if st.button("Reset grid to default", use_container_width=True):
        st.session_state.room1_walls = set(DEFAULT_WALLS)
        st.session_state.room1_slippery = {}
        st.session_state.room1_probability_errors = set()
        st.session_state.room1_editor_nonce += 1
        invalidate_room1_model()
        st.rerun()

    st.subheader("Random ice generator")
    random_count = st.number_input(
        "Number of icy cells",
        min_value=0,
        max_value=len(candidates),
        value=int(random_controls["count"]),
        step=1,
    )
    random_seed = st.number_input(
        "Generator seed",
        min_value=0,
        value=int(random_controls["seed"]),
        step=1,
    )
    st.session_state.room1_random_controls = {
        "count": int(random_count),
        "seed": int(random_seed),
    }
    if st.button("🎲 Generate icy cells", use_container_width=True):
        st.session_state.room1_slippery = generate_random_slippery_cells(
            base_config, int(random_count), int(random_seed)
        )
        st.session_state.room1_probability_errors = set()
        st.session_state.room1_editor_nonce += 1
        invalidate_room1_model()
        st.rerun()

    st.subheader("Reward configuration")
    st.caption("Enable any supported event and assign its reward or penalty.")
    for event, description in SUPPORTED_REWARD_EVENTS.items():
        enabled = st.checkbox(
            description,
            value=event in st.session_state.room1_reward_enabled,
            key=f"reward_enabled_{event}",
        )
        value = st.number_input(
            f"{description} value",
            value=float(st.session_state.room1_reward_values.get(event, 0.0)),
            step=0.1,
            format="%.3f",
            disabled=not enabled,
            key=f"reward_value_{event}",
        )
        if enabled:
            st.session_state.room1_reward_enabled.add(event)
            st.session_state.room1_reward_values[event] = float(value)
        else:
            st.session_state.room1_reward_enabled.discard(event)
            st.session_state.room1_reward_values[event] = 0.0
    return room1_configuration_error()


def render_room1_training_controls(probability_error: str | None) -> dict[str, bool]:
    controls = st.session_state.room1_training_controls
    st.subheader("Policy Iteration")
    gamma = st.slider(
        "Gamma",
        min_value=0.0,
        max_value=0.999,
        value=float(controls["gamma"]),
        step=0.001,
    )
    theta = st.number_input(
        "Theta (convergence threshold)",
        min_value=1e-12,
        max_value=1.0,
        value=float(controls["theta"]),
        format="%.8f",
    )
    max_policy_iterations = st.number_input(
        "Maximum policy iterations",
        min_value=1,
        max_value=1_000,
        value=int(controls["max_policy_iterations"]),
    )
    max_evaluation_sweeps = st.number_input(
        "Maximum evaluation sweeps",
        min_value=1,
        max_value=100_000,
        value=int(controls["max_evaluation_sweeps"]),
    )
    seed = st.number_input(
        "Initialization and tie-breaking seed",
        min_value=0,
        value=int(controls["seed"]),
        step=1,
    )
    live_update_every = st.number_input(
        "Update charts every N sweeps",
        min_value=1,
        max_value=1_000,
        value=int(controls["live_update_every"]),
    )
    st.session_state.room1_training_controls = {
        "gamma": float(gamma),
        "theta": float(theta),
        "max_policy_iterations": int(max_policy_iterations),
        "max_evaluation_sweeps": int(max_evaluation_sweeps),
        "seed": int(seed),
        "live_update_every": int(live_update_every),
    }
    if probability_error is not None:
        st.error(f"Fix the grid before training: {probability_error}")
    train_requested = st.button(
        "▶ Train / compute policy",
        type="primary",
        use_container_width=True,
        disabled=probability_error is not None,
    )
    reset_requested = st.button("Reset trained model", use_container_width=True)
    st.caption("Use Streamlit's Stop button in the upper-right corner to interrupt a run.")
    return {"train": train_requested, "reset": reset_requested}


def render_room1_test_controls() -> bool:
    controls = st.session_state.room1_test_controls
    st.subheader("Test configuration")
    episodes = st.number_input(
        "Number of test episodes",
        min_value=1,
        max_value=10_000,
        value=int(controls["episodes"]),
    )
    max_timesteps = st.number_input(
        "Maximum timesteps per episode",
        min_value=1,
        max_value=100_000,
        value=int(controls["max_timesteps"]),
    )
    seed = st.number_input(
        "Test seed", min_value=0, value=int(controls["seed"]), step=1
    )
    st.session_state.room1_test_controls = {
        "episodes": int(episodes),
        "max_timesteps": int(max_timesteps),
        "seed": int(seed),
    }
    return st.button(
        "🧪 Run test",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.room1_result is None,
    )


def render_room1_model_controls() -> None:
    result = st.session_state.room1_result
    environment = st.session_state.room1_result_environment
    algorithm_config = st.session_state.room1_algorithm_config
    st.subheader("Model artifact")
    if result is not None and environment is not None and algorithm_config is not None:
        artifact = export_room1_artifact(environment, algorithm_config, result)
        st.download_button(
            "⬇ Download trained model",
            data=artifact,
            file_name="room1_policy_iteration_model.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Train or load a model before downloading it.")

    uploaded = st.file_uploader("Upload trained model", type=["json"])
    if uploaded is not None and st.button("Load model", use_container_width=True):
        try:
            environment, algorithm_config, result = import_room1_artifact(
                uploaded.getvalue().decode("utf-8")
            )
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Invalid Room 1 model: {exc}")
        else:
            st.session_state.room1_result_environment = environment
            st.session_state.room1_algorithm_config = algorithm_config
            st.session_state.room1_result = result
            st.session_state.room1_test_results = None
            st.session_state.room1_walls = set(environment.config.walls)
            st.session_state.room1_slippery = dict(environment.config.slippery)
            st.session_state.room1_probability_errors = set()
            st.session_state.room1_reward_values = dict(environment.config.rewards)
            st.session_state.room1_reward_enabled = {
                event
                for event, value in environment.config.rewards.items()
                if value != 0.0
            }
            st.session_state.room1_editor_nonce += 1
            st.success("Model loaded successfully.")
            st.rerun()


def render_grid_editor() -> None:
    """Render the 10x10 map as per-cell popover editors."""

    outcome_labels = {
        "reach": "Reach the icy cell (no slide)",
        "up": "Slide up",
        "down": "Slide down",
        "right": "Slide right",
        "left": "Slide left",
    }
    icon_by_type = {"Normal": "⬜", "Icy": "❄️", "Wall": "🧱"}
    nonce = st.session_state.room1_editor_nonce

    cell_styles: list[str] = []
    for y in reversed(range(10)):
        for x in reversed(range(10)):
            state = (x, y)
            cell_type = current_cell_type(state)
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

            selector = (
                f'.st-key-room1_cell_{x}_{y} '
                '[data-testid="stPopover"] > button'
            )
            cell_styles.extend(
                [
                    f'{selector} {{ background: {background} !important; '
                    f'box-shadow: {outline}; }}',
                    f'{selector}::before {{ content: "{icon}"; }}',
                    f'{selector}::after {{ content: "{x},{y}"; '
                    f'color: {coordinate_color}; }}',
                ]
            )
    st.markdown(f"<style>{''.join(cell_styles)}</style>", unsafe_allow_html=True)

    with st.container(key="room1_grid_editor"):
        for y in reversed(range(10)):
            columns = st.columns(10, gap="small")
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
                    cell_label = f"{icon_by_type[current_cell_type(state)]} {x},{y}"

                with column:
                    with st.container(key=f"room1_cell_{x}_{y}"):
                      with st.popover(cell_label, use_container_width=True):
                        st.markdown(f"**Cell {state_label(state)}**")
                        if protected_label is not None:
                            st.info(protected_label)
                            st.radio(
                                "Cell type",
                                options=["Normal"],
                                disabled=True,
                                key=f"protected_cell_{nonce}_{x}_{y}",
                            )
                            continue

                        type_key = f"cell_type_{nonce}_{x}_{y}"
                        cell_type = current_cell_type(state)
                        selected_type = st.radio(
                            "Cell type",
                            options=["Normal", "Icy", "Wall"],
                            index=["Normal", "Icy", "Wall"].index(cell_type),
                            horizontal=True,
                            key=type_key,
                            on_change=apply_cell_type,
                            args=(state, type_key),
                        )
                        if selected_type != "Icy":
                            continue

                        current = st.session_state.room1_slippery[state]
                        st.caption(
                            "Set the complete outcome distribution. The five values "
                            "must total 100%."
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
                                key=f"cell_probability_{nonce}_{x}_{y}_{outcome}",
                            )
                        total = sum(percentages.values())
                        st.metric("Probability total", f"{total:.1f}%")
                        if not isclose(total, 100.0, abs_tol=1e-9):
                            st.session_state.room1_probability_errors.add(state)
                            st.error("The probabilities must total exactly 100%.")
                            continue

                        st.session_state.room1_probability_errors.discard(state)
                        updated = SlipperyCell(
                            **{
                                outcome: percentage / 100.0
                                for outcome, percentage in percentages.items()
                            }
                        )
                        if updated != current:
                            st.session_state.room1_slippery[state] = updated
                            invalidate_room1_model()


def render_environment_page() -> None:
    st.markdown(
        '<div class="room-header"><h1>Room 1 — Environment</h1>'
        '<p>Click a grid cell to configure it as normal, icy, or a wall.</p></div>',
        unsafe_allow_html=True,
    )
    walls = st.session_state.room1_walls
    slippery = st.session_state.room1_slippery
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
    st.caption("Select any cell to open its editor. Start and goal are protected.")
    render_grid_editor()

    configuration_error = room1_configuration_error()
    if configuration_error is not None:
        st.error(f"Grid configuration is not trainable: {configuration_error}")
    else:
        st.success("The grid configuration is valid and ready for training.")

    if slippery:
        rows = []
        for (x, y), probabilities in sorted(slippery.items()):
            rows.append({"x": x, "y": y, **probabilities.as_dict()})
        st.subheader("Icy-cell transition probabilities")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_training_page(requests: dict[str, bool]) -> None:
    st.markdown(
        '<div class="room-header"><h1>Room 1 — Training</h1>'
        '<p>Policy Iteration for a known stochastic environment model.</p></div>',
        unsafe_allow_html=True,
    )
    if requests["reset"]:
        st.session_state.room1_result = None
        st.session_state.room1_result_environment = None
        st.session_state.room1_algorithm_config = None
        st.session_state.room1_test_results = None
        st.rerun()

    if requests["train"]:
        environment = build_room1_environment()
        controls = st.session_state.room1_training_controls
        algorithm_config = PolicyIterationConfig(
            gamma=controls["gamma"],
            theta=controls["theta"],
            max_policy_iterations=controls["max_policy_iterations"],
            max_evaluation_sweeps=controls["max_evaluation_sweeps"],
            seed=controls["seed"],
        )
        status_slot = st.empty()
        delta_slot = st.empty()
        value_slot = st.empty()
        live_rows: list[dict] = []

        def update_live_graphs(metric, _values, _policy) -> None:
            live_rows.append(asdict(metric))
            should_render = (
                metric.phase == "improvement"
                or metric.global_step % controls["live_update_every"] == 0
            )
            if not should_render:
                return
            frame = pd.DataFrame(live_rows)
            status_slot.info(
                f"Policy iteration {metric.policy_iteration} • {metric.phase} • "
                f"delta={metric.delta:.3e} • policy changes={metric.policy_changes}"
            )
            evaluation = frame[frame["phase"] == "evaluation"]
            if not evaluation.empty:
                delta_slot.line_chart(
                    evaluation.set_index("global_step")[["delta"]],
                    x_label="Sweep",
                    y_label="Maximum delta",
                )
                value_slot.line_chart(
                    evaluation.set_index("global_step")[["mean_value"]],
                    x_label="Sweep",
                    y_label="Mean V(s)",
                )

        with st.spinner("Evaluating and improving the policy..."):
            result = run_policy_iteration(
                environment, algorithm_config, callback=update_live_graphs
            )
        st.session_state.room1_result = result
        st.session_state.room1_result_environment = environment
        st.session_state.room1_algorithm_config = algorithm_config
        st.session_state.room1_test_results = None
        if result.converged:
            status_slot.success(
                f"Converged after {result.policy_iterations} policy iterations and "
                f"{result.evaluation_sweeps} evaluation sweeps."
            )
        else:
            status_slot.warning("Stopped at the configured iteration limit.")

    result = st.session_state.room1_result
    environment = st.session_state.room1_result_environment
    if result is None or environment is None:
        st.info("Configure the training parameters in the left bar, then start training.")
        configuration_error = room1_configuration_error()
        if configuration_error is not None:
            st.error(f"The current grid cannot be trained: {configuration_error}")
            return
        current_environment = build_room1_environment()
        st.markdown(
            render_grid_html(current_environment, agent_state=current_environment.start),
            unsafe_allow_html=True,
        )
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Converged", "Yes" if result.converged else "No")
    m2.metric("Policy iterations", result.policy_iterations)
    m3.metric("Evaluation sweeps", result.evaluation_sweeps)
    st.markdown(
        render_grid_html(environment, policy=result.policy, values=result.values),
        unsafe_allow_html=True,
    )
    frame = metrics_dataframe(result)
    evaluation = frame[frame["phase"] == "evaluation"]
    improvement = frame[frame["phase"] == "improvement"]
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("Value convergence")
        st.line_chart(evaluation.set_index("global_step")[["delta"]])
    with chart2:
        st.subheader("Policy changes")
        st.bar_chart(improvement.set_index("policy_iteration")[["policy_changes"]])


def render_test_page(run_requested: bool) -> None:
    st.markdown(
        '<div class="room-header"><h1>Room 1 — Testing</h1>'
        '<p>Run the trained policy without updating its values or actions.</p></div>',
        unsafe_allow_html=True,
    )
    result = st.session_state.room1_result
    environment = st.session_state.room1_result_environment
    if result is None or environment is None:
        st.info("Train or load a model before running a test.")
        return
    if run_requested:
        controls = st.session_state.room1_test_controls
        with st.spinner("Running test episodes..."):
            st.session_state.room1_test_results = evaluate_policy(
                environment,
                result.policy,
                episodes=controls["episodes"],
                max_timesteps=controls["max_timesteps"],
                seed=controls["seed"],
            )
    test_results = st.session_state.room1_test_results
    if not test_results:
        st.info("Set the test parameters in the left bar and click Run test.")
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
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("Timesteps by episode")
        st.bar_chart(frame.set_index("episode")[["timesteps"]])
    with chart2:
        st.subheader("Reward by episode")
        st.line_chart(frame.set_index("episode")[["total_reward"]])
    st.dataframe(frame, hide_index=True, use_container_width=True)

    st.subheader("Episode replay")
    selected_number = st.selectbox(
        "Episode", options=[episode.episode for episode in test_results]
    )
    selected_episode = next(
        episode for episode in test_results if episode.episode == selected_number
    )
    replay_step = st.slider(
        "Replay timestep",
        min_value=0,
        max_value=len(selected_episode.trajectory),
        value=0,
    )
    if replay_step == 0:
        agent_state = environment.start
        st.write("Initial state")
    else:
        step = selected_episode.trajectory[replay_step - 1]
        agent_state = step.next_state
        st.write(
            f"Action: `{step.action.value}` • outcome: `{step.outcome}` • "
            f"reward: {step.reward:.3f} • cumulative: {step.cumulative_reward:.3f}"
        )
    st.markdown(
        render_grid_html(
            environment,
            agent_state=agent_state,
            policy=result.policy,
            values=result.values,
        ),
        unsafe_allow_html=True,
    )


def render_models_page() -> None:
    st.markdown(
        '<div class="room-header"><h1>Room 1 — Models</h1>'
        '<p>Download or upload a portable JSON model artifact from the left bar.</p></div>',
        unsafe_allow_html=True,
    )
    result = st.session_state.room1_result
    environment = st.session_state.room1_result_environment
    algorithm_config = st.session_state.room1_algorithm_config
    if result is None or environment is None or algorithm_config is None:
        st.info("No trained model is loaded.")
        return
    st.success("A Room 1 Policy Iteration model is loaded and ready.")
    summary = {
        "algorithm": "Policy Iteration",
        "converged": result.converged,
        "policy_iterations": result.policy_iterations,
        "evaluation_sweeps": result.evaluation_sweeps,
        "icy_cells": len(environment.config.slippery),
        **asdict(algorithm_config),
    }
    st.json(summary)


def render_room1() -> None:
    with st.sidebar:
        st.title("Room 1 Controls")
        st.caption("Known model • Policy Iteration")
        section = st.radio(
            "Control section",
            options=["Environment", "Training", "Testing", "Models"],
            key="room1_control_section",
        )
        st.divider()
        probability_error = room1_configuration_error()
        requests = {"train": False, "reset": False}
        run_test = False
        if section == "Environment":
            probability_error = render_room1_environment_controls()
        elif section == "Training":
            requests = render_room1_training_controls(probability_error)
        elif section == "Testing":
            run_test = render_room1_test_controls()
        else:
            render_room1_model_controls()

    if section == "Environment":
        render_environment_page()
    elif section == "Training":
        render_training_page(requests)
    elif section == "Testing":
        render_test_page(run_test)
    else:
        render_models_page()


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
        "This room will be connected when its environment and algorithm are implemented."
    )


initialize_state()
active_room = render_room_navigation()
if active_room == "Room 1":
    render_room1()
else:
    render_future_room(active_room)
