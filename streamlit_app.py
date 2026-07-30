"""Streamlit dashboard for the Escape Room RL project."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import altair as alt
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
    OBSERVATION_SIZE,
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
        top: 0 !important;
        height: var(--app-nav-height) !important;
        background: transparent !important;
        box-shadow: none !important;
        pointer-events: none !important;
        z-index: 1000002 !important;
        overflow: visible !important;
      }
      header[data-testid="stHeader"] [data-testid="stToolbar"] {
        position: absolute !important;
        top: 50% !important;
        right: 0.75rem !important;
        height: auto !important;
        padding: 0 !important;
        transform: translateY(-50%) !important;
        background: transparent !important;
        pointer-events: none !important;
      }
      header[data-testid="stHeader"] [data-testid="stAppDeployButton"],
      header[data-testid="stHeader"] [data-testid="stMainMenu"],
      header[data-testid="stHeader"] [data-testid="stMainMenuButton"],
      header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
        align-items: center !important;
        pointer-events: auto !important;
      }
      header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
        position: fixed !important;
        top: 0.7rem !important;
        left: 0.65rem !important;
        z-index: 1000004 !important;
        margin: 0 !important;
      }
      body:has([data-testid="stExpandSidebarButton"]) .st-key-nav_brand {
        margin-left: 2.25rem !important;
      }
      header[data-testid="stHeader"] button {
        color: #e2e8f0 !important;
      }
      header[data-testid="stHeader"] button:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.12) !important;
      }
      [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        padding-top: calc(var(--app-nav-height) + 0.85rem) !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
      }
      [data-testid="stSidebar"] {
        min-width: 340px;
        max-width: 340px;
        top: var(--app-nav-height) !important;
        height: calc(100vh - var(--app-nav-height)) !important;
      }
      [data-testid="stSidebar"] [data-testid="stSidebarHeader"],
      [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display: none !important;
      }
      [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
      }
      [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 0.7rem !important;
      }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 { letter-spacing: -0.02em; }

      /* Keep compact numeric controls on one row inside the fixed sidebar. */
      [data-testid="stSidebar"] [data-testid="stNumberInput"] {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) minmax(7.4rem, 8.25rem) !important;
        align-items: center !important;
        column-gap: 0.65rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stNumberInput"] > label {
        min-width: 0 !important;
        margin-bottom: 0 !important;
      }
      [data-testid="stSidebar"] [data-testid="stNumberInput"] > label p {
        line-height: 1.2 !important;
      }
      [data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
        min-width: 0 !important;
      }

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
        position: relative !important;
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
        display: flex !important;
        flex: 0 0 auto !important;
        align-items: center !important;
        height: 100% !important;
        min-width: 250px !important;
      }
      .st-key-nav_brand > [data-testid="stVerticalBlock"] {
        display: flex !important;
        justify-content: center !important;
        height: 100% !important;
      }
      .st-key-nav_brand [data-testid="stMarkdownContainer"],
      .st-key-nav_brand [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
      }
      .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        height: 100%;
        min-height: 0;
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
      .st-key-room_nav_buttons {
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: max-content !important;
        transform: translate(-50%, -50%) !important;
        z-index: 1 !important;
        pointer-events: auto !important;
      }
      .st-key-room_nav_buttons > [data-testid="stHorizontalBlock"] {
        width: max-content !important;
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
      @media (max-width: 1100px) {
        .st-key-nav_brand {
          min-width: 2.15rem !important;
        }
        .nav-project-name {
          display: none;
        }
      }
      @media (max-width: 760px) {
        .st-key-main_top_nav {
          padding-left: 0.65rem !important;
          padding-right: 0.65rem !important;
        }
        .st-key-main_top_nav button {
          min-width: 3.75rem !important;
          padding-left: 0.55rem !important;
          padding-right: 0.55rem !important;
          font-size: 0.78rem !important;
        }
      }
      .room-header {
        padding: 0.4rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
      }
      .room-page-hero {
        position: relative;
        overflow: hidden;
        padding: 1.15rem 1.35rem;
        margin: 0 0 0.85rem;
        border: 1px solid rgba(59, 130, 246, 0.28);
        border-radius: 1rem;
        background:
          radial-gradient(circle at 92% 15%, rgba(20, 184, 166, 0.22), transparent 28%),
          linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(15, 23, 42, 0.08));
      }
      .room-page-hero h1 { margin: 0; font-size: clamp(1.65rem, 2.7vw, 2.45rem); }
      .room-page-hero p { margin: 0.45rem 0 0; max-width: 72rem; opacity: 0.82; }
      .room-page-eyebrow {
        margin-bottom: 0.35rem;
        color: #3b82f6;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .room-section-anchor { scroll-margin-top: calc(var(--app-nav-height) + 1.2rem); }
      .room-section-title { margin-bottom: 0.85rem; }
      .room-section-title h2 { margin: 0; font-size: 1.32rem; }
      .room-section-title p { margin: 0.28rem 0 0; opacity: 0.72; font-size: 0.9rem; }
      [data-testid="stSidebar"] div:has(> .st-key-control_section_switcher) {
        position: sticky !important;
        top: 0 !important;
        z-index: 1200 !important;
        padding: 0.22rem 0 0.3rem !important;
        margin: 0 !important;
        background: var(--secondary-background-color, #f0f2f6) !important;
      }
      [data-testid="stSidebar"] .st-key-control_section_switcher {
        padding: 0.25rem !important;
        margin: 0 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.25rem !important;
        background: transparent !important;
        box-shadow: none !important;
      }
      .st-key-control_section_switcher [data-testid="stTabs"] {
        width: 100% !important;
      }
      .st-key-control_section_switcher [data-baseweb="tab-list"] {
        gap: 0.18rem !important;
        background: transparent !important;
      }
      .st-key-control_section_switcher button[data-baseweb="tab"] {
        flex: 1 1 0 !important;
        min-height: 1.7rem !important;
        padding: 0.08rem 0.2rem !important;
        border-radius: 0.25rem !important;
        border: 1px solid #d1d5db !important;
        background: transparent !important;
        color: #1f2937 !important;
        font-size: 0.68rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transform: none !important;
      }
      .st-key-control_section_switcher button[data-baseweb="tab"]:hover {
        border-color: #9ca3af !important;
        background: #f3f4f6 !important;
        color: #111827 !important;
      }
      .st-key-control_section_switcher button[data-baseweb="tab"][aria-selected="true"] {
        border-color: #111827 !important;
        background: transparent !important;
        color: #111827 !important;
        font-weight: 700 !important;
      }
      .st-key-control_section_switcher [data-baseweb="tab-highlight"] {
        background-color: #111827 !important;
        height: 2px !important;
      }
      [class*="st-key-reward_row_"] {
        margin: 0 0 0.12rem !important;
      }
      [class*="st-key-reward_row_"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.32rem !important;
      }
      [class*="st-key-reward_row_"] [data-testid="stCheckbox"] {
        width: 1.2rem !important;
        min-width: 1.2rem !important;
      }
      [class*="st-key-reward_row_"] [data-testid="stCheckbox"] label {
        padding: 0 !important;
      }
      [class*="st-key-replay_panel_"] {
        padding: 0.55rem !important;
        border-radius: 0.4rem !important;
      }
      [class*="st-key-replay_panel_"] [data-testid="stSelectbox"] label {
        font-size: 0.72rem !important;
      }
      [class*="st-key-replay_controls_"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.25rem !important;
        flex-wrap: nowrap !important;
      }
      [class*="st-key-replay_controls_"] button {
        width: 2rem !important;
        min-width: 2rem !important;
        min-height: 2rem !important;
        padding: 0.1rem !important;
        border-radius: 0.3rem !important;
        font-size: 0.78rem !important;
      }
      [class*="st-key-replay_controls_"] [data-testid="stSelectbox"] {
        width: 4.4rem !important;
        min-width: 4.4rem !important;
      }
      [class*="st-key-replay_controls_"] [data-baseweb="select"] > div {
        min-height: 2rem !important;
        height: 2rem !important;
        font-size: 0.75rem !important;
      }
      .st-key-page_environment_section,
      .st-key-page_training_section,
      .st-key-page_testing_section,
      .st-key-page_models_section {
        padding: 1rem 1.1rem 1.1rem !important;
        margin-bottom: 1.15rem !important;
        border-radius: 1rem !important;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      }
      .st-key-page_environment_section { background: linear-gradient(135deg, rgba(14, 165, 233, 0.075), rgba(20, 184, 166, 0.045)); }
      .st-key-page_training_section { background: linear-gradient(135deg, rgba(139, 92, 246, 0.075), rgba(59, 130, 246, 0.04)); }
      .st-key-page_testing_section { background: linear-gradient(135deg, rgba(245, 158, 11, 0.075), rgba(249, 115, 22, 0.04)); }
      .st-key-page_models_section { background: linear-gradient(135deg, rgba(100, 116, 139, 0.085), rgba(15, 23, 42, 0.025)); }
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
      [class*="st-key-ice_odds_box_"] {
        background: #ffffff !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
      }
      [class*="st-key-ice_odds_box_"] p, 
      [class*="st-key-ice_odds_box_"] label, 
      [class*="st-key-ice_odds_box_"] div {
        color: #111827 !important;
      }
      [class*="st-key-ice_cross_"] {
        position: relative;
        padding: 0.35rem;
        border-radius: 0.5rem;
        background: #f8fafc;
      }
      [class*="st-key-ice_cross_"]::before,
      [class*="st-key-ice_cross_"]::after {
        content: "";
        position: absolute;
        z-index: 0;
        border-radius: 999px;
        background: #cbd5e1;
      }
      [class*="st-key-ice_cross_"]::before {
        left: 16%;
        right: 16%;
        top: 50%;
        height: 2px;
      }
      [class*="st-key-ice_cross_"]::after {
        top: 16%;
        bottom: 16%;
        left: 50%;
        width: 2px;
      }
      [class*="st-key-ice_cross_"] [data-testid="stNumberInput"] {
        position: relative;
        z-index: 1;
        padding: 0.2rem;
        border-radius: 0.4rem;
        background: var(--secondary-background-color, #f0f2f6);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_reward_controls(widget_values: dict[str, float]) -> None:
    """Color reward controls by sign without changing their numeric behavior."""
    rules: list[str] = []
    tones = {
        "positive": ("#15803d", "rgba(21, 128, 61, 0.42)", "rgba(21, 128, 61, 0.08)"),
        "negative": ("#dc2626", "rgba(220, 38, 38, 0.42)", "rgba(220, 38, 38, 0.08)"),
        "neutral": ("#64748b", "rgba(100, 116, 139, 0.32)", "transparent"),
    }
    for key, raw_value in widget_values.items():
        value = float(raw_value)
        sign = "positive" if value > 0 else "negative" if value < 0 else "neutral"
        color, border, background = tones[sign]
        selector = f".st-key-{key}"
        rules.extend(
            [
                f"{selector} input {{ color: {color} !important; -webkit-text-fill-color: {color} !important; font-weight: 800 !important; }}",
                f"{selector} [data-testid='stNumberInputContainer'] {{ border-color: {border} !important; background: {background} !important; }}",
                f"{selector} [data-testid='stNumberInputContainer'] button {{ color: {color} !important; }}",
                f"{selector} [data-testid='stThumbValue'] {{ color: {color} !important; font-weight: 800 !important; }}",
            ]
        )
    if rules:
        st.sidebar.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def render_reward_control(
    *,
    enabled_state_key: str,
    event: str,
    label: str,
    value: float,
    widget_key: str,
    step: float,
    number_format: str,
    help_text: str,
) -> tuple[float, float]:
    """Render one compact reward toggle and numeric value on the same row."""
    enabled_events = set(st.session_state[enabled_state_key])
    with st.container(key=f"reward_row_{widget_key}"):
        toggle_column, value_column = st.columns(
            [0.08, 0.92],
            gap="small",
            vertical_alignment="center",
        )
        with toggle_column:
            enabled = st.checkbox(
                f"Enable {label}",
                value=event in enabled_events,
                key=f"{widget_key}_enabled",
                help=f"Enable or disable {label.lower()}.",
                label_visibility="collapsed",
                width="content",
            )
        with value_column:
            configured_value = float(
                st.number_input(
                    label,
                    value=float(value),
                    step=step,
                    format=number_format,
                    disabled=not enabled,
                    key=widget_key,
                    help=help_text,
                )
            )

    if enabled:
        enabled_events.add(event)
    else:
        enabled_events.discard(event)
    st.session_state[enabled_state_key] = enabled_events
    return (configured_value if enabled else 0.0), configured_value


def render_room_page_header(
    room_num: int,
    title: str,
    algorithm: str,
    description: str,
) -> None:
    page_id = f"room-{room_num}-overview"
    st.markdown(
        f'<div id="{page_id}" class="room-page-hero room-section-anchor">'
        f'<div class="room-page-eyebrow">Room {room_num} · {algorithm}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_control_section_tabs(
    _room_num: int,
    state_key: str,
    _help_text: str,
) -> tuple[Any, Any, Any, Any]:
    with st.sidebar.container(key="control_section_switcher"):
        tabs = st.tabs(
            ["Env", "Training", "Testing", "Model"],
            key=f"{state_key}_tabs_v2",
            on_change="ignore",
        )
    return tuple(tabs)


@contextmanager
def room_page_section(
    room_num: int,
    section: str,
    title: str,
    description: str,
):
    st.markdown(
        f'<div id="room-{room_num}-{section}" class="room-section-anchor"></div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True, key=f"page_{section}_section"):
        st.markdown(
            f'<div class="room-section-title"><h2>{title}</h2><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
        yield


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
        "room4_reward_enabled": set(DEFAULT_ROOM4_REWARDS),
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
            "episodes": 400,
            "max_timesteps": 700,
            "hidden_layers": 2,
            "hidden_units": 64,
            "activation_fn": "ReLU",
            "seed": 42,
            "live_update_every": 10,
        },
        "room4_test_controls": {
            "episodes": 50,
            "max_timesteps": 700,
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
        "room5_reward_enabled": set(DEFAULT_ROOM5_REWARDS),
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


def _chart_frame(data: Any, x: str) -> pd.DataFrame:
    """Return chart data with the requested X field as an ordinary column."""
    frame = pd.DataFrame(data).copy()
    if x not in frame.columns:
        frame = frame.reset_index()
    if x not in frame.columns:
        raise ValueError(f"Chart X field {x!r} is missing from the supplied data.")
    return frame


def _altair_field_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"
    if pd.api.types.is_numeric_dtype(series):
        return "quantitative"
    return "nominal"


def render_locked_line_chart(
    data: Any,
    *,
    x: str,
    y: str | list[str],
    x_label: str,
    y_label: str,
    target: Any = st,
) -> None:
    """Render a tooltip-enabled line chart with no pan or zoom bindings."""
    frame = _chart_frame(data, x)
    y_fields = [y] if isinstance(y, str) else list(y)
    x_encoding = alt.X(field=x, type=_altair_field_type(frame[x]), title=x_label)

    if len(y_fields) == 1:
        y_field = y_fields[0]
        chart = alt.Chart(frame).mark_line(point=False, strokeWidth=2.2).encode(
            x=x_encoding,
            y=alt.Y(
                field=y_field,
                type="quantitative",
                title=y_label,
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip(field=x, type=_altair_field_type(frame[x]), title=x_label),
                alt.Tooltip(field=y_field, type="quantitative", title=y_label, format=".4g"),
            ],
        )
    else:
        tidy = frame.melt(
            id_vars=[x], value_vars=y_fields, var_name="Series", value_name="Value"
        )
        chart = alt.Chart(tidy).mark_line(point=False, strokeWidth=2.2).encode(
            x=alt.X(field=x, type=_altair_field_type(tidy[x]), title=x_label),
            y=alt.Y(
                field="Value",
                type="quantitative",
                title=y_label,
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(field="Series", type="nominal", title="Metric"),
            tooltip=[
                alt.Tooltip(field=x, type=_altair_field_type(tidy[x]), title=x_label),
                alt.Tooltip(field="Series", type="nominal", title="Metric"),
                alt.Tooltip(field="Value", type="quantitative", title=y_label, format=".4g"),
            ],
        )

    # Deliberately omit interactive scale bindings: axes stay fixed while hover
    # tooltips remain available.
    target.altair_chart(chart.properties(height=300), width="stretch")


def render_locked_bar_chart(
    data: Any,
    *,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    target: Any = st,
) -> None:
    """Render a fixed-axis bar chart with no pan or zoom bindings."""
    frame = _chart_frame(data, x)
    chart = alt.Chart(frame).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(field=x, type=_altair_field_type(frame[x]), title=x_label),
        y=alt.Y(field=y, type="quantitative", title=y_label),
        tooltip=[
            alt.Tooltip(field=x, type=_altair_field_type(frame[x]), title=x_label),
            alt.Tooltip(field=y, type="quantitative", title=y_label, format=".4g"),
        ],
    )
    target.altair_chart(chart.properties(height=300), width="stretch")


def render_algorithm_overview(room_num: int) -> None:
    """Explain the RL contract before the room's environment controls and visual."""
    room1_algorithm = st.session_state.get("room1_algorithm", "Policy Iteration")
    overviews = {
        1: {
            "algorithm": room1_algorithm,
            "summary": "A dynamic-programming method that uses the complete transition and reward model to compute a policy for every walkable cell.",
            "input": "Current grid cell, legal transitions, icy-cell probabilities, terminal states, and configured rewards.",
            "output": "A state-value V(s) and one selected action for every walkable grid state.",
            "actions": "UP, DOWN, LEFT, RIGHT. Illegal boundary and wall moves are excluded.",
        },
        2: {
            "algorithm": "SARSA",
            "summary": "An on-policy temporal-difference algorithm that learns from the action actually selected in the next state.",
            "input": "Current grid cell, selected action, received reward, next cell, and the next epsilon-greedy action.",
            "output": "A learned Q(s, a) table and its derived epsilon-greedy navigation policy.",
            "actions": "UP, DOWN, LEFT, RIGHT. Icy cells can make the executed movement stochastic.",
        },
        3: {
            "algorithm": "Q-Learning",
            "summary": "An off-policy temporal-difference algorithm that learns toward the best estimated next action while still exploring.",
            "input": "Current grid cell, selected action, reward, next cell, and the maximum Q-value available there.",
            "output": "A learned Q(s, a) table and a greedy four-direction navigation policy.",
            "actions": "UP, DOWN, LEFT, RIGHT. Illegal boundary and wall moves are excluded.",
        },
        4: {
            "algorithm": "PPO actor-critic",
            "summary": "A neural policy-gradient algorithm that learns continuous-space obstacle avoidance while clipping overly large policy updates.",
            "input": "Four continuous values: X position, Y position, horizontal velocity, and vertical velocity.",
            "output": "Probabilities for nine actions plus a critic estimate of the current state's expected return.",
            "actions": "Each action selects discrete horizontal and vertical velocities from {-1, 0, +1} m/s: eight directions (including diagonals) and HOVER (0, 0).",
        },
        5: {
            "algorithm": "PPO actor-critic",
            "summary": "A neural policy-gradient algorithm that learns lane selection and overtaking from a fixed-size road observation.",
            "input": "13 values: Agent lane (one-hot), Progress, and Clearance/Speed for left, current & right lanes.",
            "output": "Probabilities for three driving actions plus a critic estimate of the current state's expected return.",
            "actions": "LEFT lane change, KEEP_LANE, and RIGHT lane change.",
        },
    }
    overview = overviews[room_num]
    with st.container(border=True):
        st.markdown(f"#### How {overview['algorithm']} interacts with this room")
        st.caption(overview["summary"])
        input_col, output_col, actions_col = st.columns(3)
        input_col.markdown(f"**Input / observation**\n\n{overview['input']}")
        output_col.markdown(f"**Model output**\n\n{overview['output']}")
        actions_col.markdown(f"**Available actions**\n\n{overview['actions']}")


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
        with st.container(
            key="room_nav_buttons",
            horizontal=True,
            vertical_alignment="center",
            gap="small",
        ):
            for room in rooms:
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


def apply_cell_editor_submission(
    room_num: int,
    state: tuple[int, int],
    *,
    selected_type: str,
    set_as_start: bool,
    set_as_goal: bool,
    termination_enabled: bool,
    reward_enabled: bool,
    reward_value: float,
    percentages: dict[str, int] | None,
) -> str | None:
    """Validate and atomically apply one submitted grid-cell draft."""
    p = room_prefix(room_num)
    start = st.session_state[f"{p}_start"]
    goal = st.session_state[f"{p}_goal"]

    if set_as_start and set_as_goal:
        return "A cell cannot become both the start and the goal."
    if set_as_start and state == goal:
        return "Move the goal first before making this cell the start."
    if set_as_goal and state == start:
        return "Move the start first before making this cell the goal."
    if selected_type not in {"Normal", "Icy", "Wall"}:
        return "Unknown cell type."

    if selected_type == "Icy":
        percentages = percentages or {
            outcome: int(round(getattr(SlipperyCell(), outcome) * 100))
            for outcome in SLIP_OUTCOMES
        }
        if sum(percentages.values()) != 100:
            return "Icy-cell probabilities must total exactly 100%."

    walls = set(st.session_state[f"{p}_walls"])
    slippery = dict(st.session_state[f"{p}_slippery"])
    terminals = set(st.session_state[f"{p}_terminal_states"])
    cell_rewards = dict(st.session_state[f"{p}_cell_rewards"])
    probability_errors = set(st.session_state[f"{p}_probability_errors"])

    if set_as_start:
        start = state
        selected_type = "Normal"
        terminals.discard(state)
    elif set_as_goal:
        terminals.discard(goal)
        goal = state
        terminals.add(state)
        selected_type = "Normal"

    if state in {start, goal}:
        selected_type = "Normal"

    walls.discard(state)
    slippery.pop(state, None)
    probability_errors.discard(state)

    if selected_type == "Wall":
        walls.add(state)
        terminals.discard(state)
        cell_rewards.pop(state, None)
    else:
        if state == goal:
            terminals.add(state)
        elif state == start:
            terminals.discard(state)
        elif termination_enabled:
            terminals.add(state)
        else:
            terminals.discard(state)

        if reward_enabled:
            cell_rewards[state] = float(reward_value)
        else:
            cell_rewards.pop(state, None)

        if selected_type == "Icy":
            assert percentages is not None
            slippery[state] = SlipperyCell(
                **{
                    outcome: float(percentages[outcome]) / 100.0
                    for outcome in SLIP_OUTCOMES
                }
            )

    st.session_state[f"{p}_start"] = start
    st.session_state[f"{p}_goal"] = goal
    st.session_state[f"{p}_walls"] = walls
    st.session_state[f"{p}_slippery"] = slippery
    st.session_state[f"{p}_terminal_states"] = terminals
    st.session_state[f"{p}_cell_rewards"] = cell_rewards
    st.session_state[f"{p}_probability_errors"] = probability_errors
    st.session_state[f"{p}_editor_nonce"] += 1
    invalidate_room_model(room_num)
    return None


@st.fragment
def render_grid_editor(room_num: int) -> None:
    """Render a full per-cell editor for layout, roles, rewards and ice."""
    p = room_prefix(room_num)
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
                        with st.popover(cell_label, width="stretch"):
                            st.markdown(f"**Cell {state_label(state)}**")
                            cell_type = current_cell_type(room_num, state)
                            form_key = f"cell_form_{p}_{nonce}_{x}_{y}"
                            with st.form(form_key, border=False):
                                selected_type = st.radio(
                                    "Cell type",
                                    options=["Normal", "Icy", "Wall"],
                                    index=["Normal", "Icy", "Wall"].index(cell_type),
                                    horizontal=True,
                                    key=f"cell_type_{p}_{nonce}_{x}_{y}",
                                    disabled=state == start or state in terminals,
                                    help=(
                                        "Choose the cell type. Nothing changes in the room until "
                                        "you submit the complete cell form."
                                    ),
                                )

                                st.caption("Cell roles")
                                start_col, goal_col = st.columns(2)
                                with start_col:
                                    set_as_start = st.checkbox(
                                        "Set as start",
                                        value=False,
                                        key=f"set_start_{p}_{nonce}_{x}_{y}",
                                        disabled=state in {start, goal},
                                        help="Move the dog's start position here when the form is submitted.",
                                    )
                                with goal_col:
                                    set_as_goal = st.checkbox(
                                        "Set as goal",
                                        value=False,
                                        key=f"set_goal_{p}_{nonce}_{x}_{y}",
                                        disabled=state in {start, goal},
                                        help="Move the goal here when the form is submitted.",
                                    )

                                termination_enabled = st.checkbox(
                                    "Termination state",
                                    value=state in terminals,
                                    key=f"termination_{p}_{nonce}_{x}_{y}",
                                    disabled=state in {start, goal},
                                    help="End the episode on entry after this form is submitted.",
                                )

                                reward_toggle = st.checkbox(
                                    "Custom reward on entry",
                                    value=state in cell_rewards,
                                    key=f"cell_reward_enabled_{p}_{nonce}_{x}_{y}",
                                    help=(
                                        "Add a cell-specific reward or penalty on entry. Wall cells "
                                        "remove custom rewards when submitted."
                                    ),
                                )
                                reward_value = st.number_input(
                                    "Cell reward value",
                                    value=float(cell_rewards.get(state, 0.0)),
                                    step=0.1,
                                    format="%.3f",
                                    key=f"cell_reward_value_{p}_{nonce}_{x}_{y}",
                                    help="Positive values attract the agent; negative values discourage entry.",
                                )

                                percentages: dict[str, int] | None = None
                                if selected_type == "Icy":
                                    current = st.session_state[f"{p}_slippery"].get(
                                        state,
                                        SlipperyCell(),
                                    )
                                    with st.container(border=True, key=f"ice_odds_box_{p}_{nonce}_{x}_{y}"):
                                        st.caption(
                                            "Ice outcome distribution — all five percentages must total 100%."
                                        )
                                        percentages = {}
                                        with st.container(key=f"ice_cross_{p}_{nonce}_{x}_{y}"):
                                            _top_left, top_middle, _top_right = st.columns([1, 1.25, 1])
                                            with top_middle:
                                                percentages["up"] = int(st.slider(
                                                    "↑ Slide up (%)",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=int(round(current.up * 100)),
                                                    step=1,
                                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_up",
                                                ))
    
                                            middle_left, middle_center, middle_right = st.columns([1, 1.25, 1])
                                            with middle_left:
                                                percentages["left"] = int(st.slider(
                                                    "← Slide left (%)",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=int(round(current.left * 100)),
                                                    step=1,
                                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_left",
                                                ))
                                            with middle_center:
                                                percentages["reach"] = int(st.slider(
                                                    "Reach (%)",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=int(round(current.reach * 100)),
                                                    step=1,
                                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_reach",
                                                ))
                                            with middle_right:
                                                percentages["right"] = int(st.slider(
                                                    "Slide right (%) →",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=int(round(current.right * 100)),
                                                    step=1,
                                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_right",
                                                ))
    
                                            _bottom_left, bottom_middle, _bottom_right = st.columns([1, 1.25, 1])
                                            with bottom_middle:
                                                percentages["down"] = int(st.slider(
                                                    "↓ Slide down (%)",
                                                    min_value=0,
                                                    max_value=100,
                                                    value=int(round(current.down * 100)),
                                                    step=1,
                                                    key=f"cell_prob_{p}_{nonce}_{x}_{y}_down",
                                                ))
    
                                        total = sum(percentages.values())
                                        st.caption(
                                            f"Probability total: **{total}%**"
                                            + (" ✓" if total == 100 else " — must equal 100%")
                                        )

                                submitted = st.form_submit_button(
                                    "Submit cell changes",
                                    icon=":material/check:",
                                    type="primary",
                                    width="stretch",
                                    help="Apply every field in this editor as one atomic update.",
                                )

                            if submitted:
                                error = apply_cell_editor_submission(
                                    room_num,
                                    state,
                                    selected_type=selected_type,
                                    set_as_start=bool(set_as_start),
                                    set_as_goal=bool(set_as_goal),
                                    termination_enabled=bool(termination_enabled),
                                    reward_enabled=bool(reward_toggle),
                                    reward_value=float(reward_value),
                                    percentages=percentages,
                                )
                                if error:
                                    st.error(error)
                                else:
                                    st.toast(f"Cell {state_label(state)} updated.")
                                    st.rerun()


def render_environment_page(room_num: int, *, show_header: bool = True) -> None:
    p = room_prefix(room_num)
    room1_algo = st.session_state.get("room1_algorithm", "Policy Iteration")
    algo_names = {1: room1_algo, 2: "SARSA", 3: "Q-Learning"}
    if show_header:
        st.markdown(
            f'<div class="room-header"><h1>Room {room_num} — Environment ({algo_names[room_num]})</h1>'
            '<p>Click any grid cell below to configure it as normal, icy, or a wall.</p></div>',
            unsafe_allow_html=True,
        )
    render_algorithm_overview(room_num)
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
@st.fragment
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
        status_str = "Success" if success_val else "Failed"
        extra = ""
        if hasattr(ep, "pipes_passed"):
            extra = f", Pipes: {ep.pipes_passed}"
        elif hasattr(ep, "overtakes"):
            extra = f", Overtakes: {ep.overtakes}"
        elif hasattr(ep, "slipped_count") and ep.slipped_count > 0:
            extra = f", Slips: {ep.slipped_count}"
        return f"Episode {getattr(ep, 'episode', index + 1)} • {status_str} • R {reward_val:.2f}{extra}"

    ep_select_key = f"{key_prefix}_select"

    visual_column, control_column = st.columns(
        [4.5, 1.5],
        gap="medium",
        vertical_alignment="top",
    )

    with control_column:
        with st.container(border=True, key=f"replay_panel_{key_prefix}"):
            selected_idx = st.selectbox(
                "Episode",
                options=list(range(len(episodes))),
                format_func=format_ep_option,
                key=ep_select_key,
                help="Choose which recorded episode to replay.",
            )
            selected_ep = episodes[selected_idx]
            trajectory = getattr(selected_ep, "trajectory", [])
            
            # Clean up deprecated state keys if they exist from before
            for old_key in [f"{key_prefix}_is_playing", f"{key_prefix}_step", f"{key_prefix}_speed"]:
                if old_key in st.session_state:
                    del st.session_state[old_key]

    with visual_column:
        if not trajectory:
            st.info("This episode has an empty trajectory.")
            return

        total_steps = len(trajectory)
        frames = []
        captions = []

        for step_idx, step_info in enumerate(trajectory):
            if room_num in (1, 2, 3):
                action_name = getattr(step_info.action, "value", str(step_info.action))
                caption = f"Timestep {step_idx + 1}/{total_steps} • Action `{action_name}` • Outcome `{getattr(step_info, 'outcome', 'normal')}` • Reward {step_info.reward:.3f} • Cumulative {step_info.cumulative_reward:.3f}"
                svg = render_grid_html(environment, agent_state=step_info.next_state, policy=policy, values=values)
            elif room_num == 4:
                st_val = step_info.state
                caption = f"Step {step_info.timestep}/{total_steps} • State: (x={st_val[0]:.2f}, y={st_val[1]:.2f}, Vx={st_val[2]:.1f}, Vy={st_val[3]:.1f}) • Action: {step_info.action.name} • Reward: {step_info.reward:.2f} • Cum.: {step_info.cumulative_reward:.2f}"
                trajectory_states = [s.state for s in trajectory[: step_idx + 1]] + [step_info.next_state]
                svg = render_room4_html(environment, agent_state=step_info.next_state, trajectory=trajectory_states)
            elif room_num == 5:
                caption = f"Step {step_info.timestep}/{total_steps} • Action: {step_info.action.name} • Reward: {step_info.reward:.2f} • Cum.: {step_info.cumulative_reward:.2f} • Events: {', '.join(step_info.events)}"
                svg = render_room5_html(environment, step_info.after_snapshot)
            else:
                svg, caption = "", ""

            frames.append(svg)
            captions.append(caption)

        if room_num == 4:
            base_delay = 20
        elif room_num == 5:
            base_delay = 50
        else:
            base_delay = 200

        import json
        frames_json = json.dumps(frames)
        captions_json = json.dumps(captions)
        
        html_code = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background: transparent; }
        .caption { color: #64748b; font-size: 0.85rem; margin-bottom: 8px; font-weight: 500; height: 1.2rem; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; text-align: center; }
        .controls { display: flex; align-items: center; gap: 8px; margin-top: 12px; padding: 12px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; }
        button { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-weight: 600; color: #334155; transition: 0.1s; display: flex; align-items: center; justify-content: center; user-select: none; }
        button:hover { background: #f1f5f9; border-color: #94a3b8; }
        button:active { background: #e2e8f0; }
        button.primary { background: #3b82f6; color: white; border-color: #2563eb; }
        button.primary:hover { background: #2563eb; }
        button.primary:active { background: #1d4ed8; }
        input[type=range] { flex-grow: 1; accent-color: #3b82f6; cursor: pointer; height: 6px; }
        .speed-select { padding: 6px 8px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 0.875rem; color: #334155; background: #fff; cursor: pointer; outline: none; }
        .speed-select:focus { border-color: #3b82f6; }
        .svg-container { display: flex; justify-content: center; align-items: center; min-height: 300px; }
        </style>
        </head>
        <body>
        <div class="caption" id="caption-el"></div>
        <div class="svg-container" id="svg-el"></div>
        <div class="controls">
            <button id="btn-prev" title="Previous Frame (Left Arrow)">◀</button>
            <button id="btn-play" class="primary" title="Play/Pause (Spacebar)" style="width: 75px;">Play</button>
            <button id="btn-next" title="Next Frame (Right Arrow)">▶</button>
            <input type="range" id="slider" min="0" max="100" value="0" title="Scrub Timeline">
            <select id="speed-select" class="speed-select" title="Playback Speed">
                <option value="0.5">0.5×</option>
                <option value="1.0" selected>1.0×</option>
                <option value="2.0">2.0×</option>
                <option value="4.0">4.0×</option>
            </select>
        </div>
        <script>
            const frames = __FRAMES__;
            const captions = __CAPTIONS__;
            const baseDelay = __BASE_DELAY__;
            const numFrames = frames.length;

            const svgEl = document.getElementById('svg-el');
            const captionEl = document.getElementById('caption-el');
            const slider = document.getElementById('slider');
            const btnPlay = document.getElementById('btn-play');
            const btnPrev = document.getElementById('btn-prev');
            const btnNext = document.getElementById('btn-next');
            const speedSelect = document.getElementById('speed-select');

            slider.max = numFrames - 1;
            let currentFrame = 0;
            let playing = false;
            let lastTime = 0;
            let animFrame = null;

            function render(index) {
                if (index < 0) index = 0;
                if (index >= numFrames) index = numFrames - 1;
                currentFrame = index;
                slider.value = index;
                svgEl.innerHTML = frames[index];
                captionEl.innerText = captions[index];
            }

            function togglePlay() {
                playing = !playing;
                btnPlay.innerText = playing ? "Pause" : "Play";
                btnPlay.className = playing ? "" : "primary";
                if (playing) {
                    if (currentFrame >= numFrames - 1) {
                        currentFrame = 0;
                    }
                    lastTime = performance.now();
                    animFrame = requestAnimationFrame(loop);
                } else {
                    cancelAnimationFrame(animFrame);
                }
            }

            function loop(time) {
                if (!playing) return;
                const speed = parseFloat(speedSelect.value);
                const delay = baseDelay / speed;
                
                if (time - lastTime >= delay) {
                    currentFrame++;
                    if (currentFrame >= numFrames) {
                        currentFrame = numFrames - 1;
                        render(currentFrame);
                        togglePlay();
                        return;
                    }
                    render(currentFrame);
                    lastTime = time;
                }
                animFrame = requestAnimationFrame(loop);
            }

            btnPlay.addEventListener('click', togglePlay);
            btnPrev.addEventListener('click', () => { if (playing) togglePlay(); render(currentFrame - 1); });
            btnNext.addEventListener('click', () => { if (playing) togglePlay(); render(currentFrame + 1); });
            slider.addEventListener('input', (e) => {
                if (playing) togglePlay();
                render(parseInt(e.target.value));
            });

            // Keyboard shortcuts (only if clicking inside the iframe)
            document.addEventListener('keydown', (e) => {
                if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
                else if (e.code === 'ArrowLeft') { e.preventDefault(); if (playing) togglePlay(); render(currentFrame - 1); }
                else if (e.code === 'ArrowRight') { e.preventDefault(); if (playing) togglePlay(); render(currentFrame + 1); }
            });

            // Render initial frame
            render(0);
        </script>
        </body>
        </html>
        """.replace("__FRAMES__", frames_json).replace("__CAPTIONS__", captions_json).replace("__BASE_DELAY__", str(base_delay))
        
        st.components.v1.html(html_code, height=750, scrolling=False)



def render_training_page(
    room_num: int,
    requests: dict[str, bool],
    *,
    show_header: bool = True,
) -> None:

    p = room_prefix(room_num)
    room1_algo = st.session_state.get("room1_algorithm", "Policy Iteration")
    algo_names = {1: room1_algo, 2: "SARSA", 3: "Q-Learning"}
    if show_header:
        st.markdown(
            f'<div class="room-header"><h1>Room {room_num} — Training ({algo_names[room_num]})</h1></div>',
            unsafe_allow_html=True,
        )
    if requests["reset"]:
        invalidate_room_model(room_num)

    has_training_output = (
        requests["train"] or st.session_state[f"{p}_result"] is not None
    )
    status_slot = st.empty() if requests["train"] else None
    chart_slot_1 = None
    chart_slot_2 = None
    if has_training_output:
        st.subheader("Training metrics")
        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            st.caption(
                "**Bellman convergence delta**"
                if room_num == 1
                else "**Total reward per episode**"
            )
            chart_slot_1 = st.empty()
        with chart_col_2:
            st.caption(
                "**Mean state value**"
                if room_num == 1
                else "**Exploration epsilon**"
            )
            chart_slot_2 = st.empty()

    if requests["train"]:
        # Start with an empty training state so no metrics, charts, or replay from
        # the previous model remain visible while the new run is being produced.
        invalidate_room_model(room_num)
        st.session_state.pop(f"{p}_training_notice", None)
        for replay_key in (
            f"{p}_tr_replay_select",
            f"{p}_tr_replay_step",
            f"{p}_tr_replay_is_playing",
            f"{p}_tr_replay_speed",
        ):
            st.session_state.pop(replay_key, None)
        env = build_environment(room_num)
        controls = st.session_state[f"{p}_training_controls"]
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
                    render_locked_line_chart(df, x="global_step", y="delta", x_label="Sweep", y_label="Delta", target=chart_slot_1)
                    render_locked_line_chart(df, x="global_step", y="mean_value", x_label="Sweep", y_label="Mean V(s)", target=chart_slot_2)

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
                        render_locked_line_chart(eval_df, x="global_step", y="delta", x_label="Sweep", y_label="Delta", target=chart_slot_1)
                        render_locked_line_chart(eval_df, x="global_step", y="mean_value", x_label="Sweep", y_label="Mean V(s)", target=chart_slot_2)

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
                render_locked_line_chart(df, x="episode", y="total_reward", x_label="Episode", y_label="Total Reward", target=chart_slot_1)
                render_locked_line_chart(df, x="episode", y="epsilon", x_label="Episode", y_label="Epsilon", target=chart_slot_2)

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
                render_locked_line_chart(df, x="episode", y="total_reward", x_label="Episode", y_label="Total Reward", target=chart_slot_1)
                render_locked_line_chart(df, x="episode", y="epsilon", x_label="Episode", y_label="Epsilon", target=chart_slot_2)

            with st.spinner("Training Q-Learning agent..."):
                res = run_q_learning(env, alg_config, callback=callback_ql)

        st.session_state[f"{p}_result"] = res
        st.session_state[f"{p}_result_environment"] = env
        st.session_state[f"{p}_algorithm_config"] = alg_config
        st.session_state[f"{p}_test_results"] = None
        st.session_state[f"{p}_training_notice"] = "Training completed successfully."
        for replay_key in (
            f"{p}_tr_replay_select",
            f"{p}_tr_replay_step",
            f"{p}_tr_replay_is_playing",
        ):
            st.session_state.pop(replay_key, None)

    training_notice = st.session_state.pop(f"{p}_training_notice", None)
    if training_notice:
        st.success(training_notice)

    res = st.session_state[f"{p}_result"]
    env = st.session_state[f"{p}_result_environment"]
    if res is None or env is None:
        st.info("Configure training parameters in the left bar and click ▶ Train.")
        return

    frame = metrics_dataframe(res)
    if room_num == 1:
        training_frame = (
            frame[frame["phase"] == "evaluation"]
            if "phase" in frame.columns
            else frame
        )
        render_locked_line_chart(
            training_frame,
            x="global_step",
            y="delta",
            x_label="Bellman sweep",
            y_label="Maximum value delta",
            target=chart_slot_1,
        )
        render_locked_line_chart(
            training_frame,
            x="global_step",
            y="mean_value",
            x_label="Bellman sweep",
            y_label="Mean V(s)",
            target=chart_slot_2,
        )
    else:
        render_locked_line_chart(
            frame,
            x="episode",
            y="total_reward",
            x_label="Training episode",
            y_label="Episode total reward",
            target=chart_slot_1,
        )
        render_locked_line_chart(
            frame,
            x="episode",
            y="epsilon",
            x_label="Training episode",
            y_label="Exploration epsilon",
            target=chart_slot_2,
        )

    m1, m2, m3 = st.columns(3)
    if room_num == 1:
        m1.metric("Converged", "Yes" if res.converged else "No")
        if hasattr(res, "iterations"):
            m2.metric("Value Iterations (Sweeps)", res.iterations)
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
    else:
        st.markdown(render_grid_html(env, policy=res.policy, values=res.values), unsafe_allow_html=True)


def render_test_page(
    room_num: int,
    run_requested: bool,
    *,
    show_header: bool = True,
) -> None:
    p = room_prefix(room_num)
    if show_header:
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
        render_locked_line_chart(
            frame,
            x="episode",
            y="timesteps",
            x_label="Test episode",
            y_label="Timesteps used",
        )
    with c2:
        st.subheader("Reward by episode")
        render_locked_line_chart(
            frame,
            x="episode",
            y="total_reward",
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



def render_models_page(room_num: int, *, show_header: bool = True) -> None:
    p = room_prefix(room_num)
    algo_names = {
        1: st.session_state.get("room1_algorithm", "Policy Iteration"),
        2: "SARSA",
        3: "Q-Learning",
    }
    if show_header:
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

def request_full_app_action(action_key: str, action: str) -> None:
    """Queue one expensive action, then leave the lightweight sidebar fragment."""
    st.session_state[action_key] = action
    st.rerun()


@st.fragment
def render_room_controls(room_num: int) -> tuple[str, dict[str, bool], bool]:
    p = room_prefix(room_num)
    action_key = f"{p}_pending_control_action"
    pending_action = st.session_state.pop(action_key, None)
    algo_names = {1: "Policy Iteration", 2: "SARSA", 3: "Q-Learning"}
    st.sidebar.title(f"Room {room_num} Controls")
    st.sidebar.caption(f"Model • {algo_names[room_num]}")
    environment_tab, training_tab, testing_tab, models_tab = render_control_section_tabs(
        room_num,
        f"{p}_control_section",
        "Choose which controls are visible in the sidebar.",
    )

    # Environment controls in sidebar
    defaults_walls = {1: DEFAULT_WALLS, 2: DEFAULT_ROOM2_WALLS, 3: DEFAULT_ROOM3_WALLS}
    defaults_slippery = {1: DEFAULT_SLIPPERY, 2: DEFAULT_ROOM2_SLIPPERY, 3: {}}
    random_controls = st.session_state[f"{p}_random_controls"]

    prob_err = room_configuration_error(room_num)
    requests = {
        "train": pending_action == "train",
        "reset": pending_action == "reset",
    }
    run_test = pending_action == "test"

    with environment_tab:
        st.subheader("Grid layout")
        if st.button(
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

        st.subheader("Random full-grid generator")
        random_walls = st.number_input(
            "Number of walls",
            0,
            80,
            int(random_controls["wall_count"]),
            key=f"{p}_random_walls",
            help="Number of impassable wall cells placed by the generator.",
        )
        random_ice = st.number_input(
            "Number of icy cells",
            0,
            98,
            int(random_controls["icy_count"]),
            key=f"{p}_random_ice",
            help="Number of stochastic icy cells, each with an integer probability distribution.",
        )
        random_seed = st.number_input(
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
            st.error("Walls + icy cells must leave two cells for start and goal.")
        if st.button(
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

        st.subheader("Reward configuration")
        reward_widget_values: dict[str, float] = {}
        for event, description in SUPPORTED_REWARD_EVENTS.items():
            previous_enabled = event in st.session_state[f"{p}_reward_enabled"]
            previous_value = st.session_state[f"{p}_reward_values"].get(event, 0.0)
            actual_value, configured_value = render_reward_control(
                enabled_state_key=f"{p}_reward_enabled",
                event=event,
                label=description,
                value=float(previous_value),
                widget_key=f"{p}_reward_value_{event}",
                step=0.1,
                number_format="%.3f",
                help_text="The reward added whenever this event occurs. Positive attracts; negative penalizes.",
            )
            reward_widget_values[f"{p}_reward_value_{event}"] = configured_value
            st.session_state[f"{p}_reward_values"][event] = actual_value
            currently_enabled = event in st.session_state[f"{p}_reward_enabled"]
            if previous_enabled != currently_enabled or previous_value != actual_value:
                invalidate_room_model(room_num)
        style_reward_controls(reward_widget_values)
        
        if st.button(
            "Apply changes to room preview",
            icon=":material/refresh:",
            width="stretch",
            key=f"{p}_refresh_environment_preview",
            help="Refresh the main room visualization once after you finish editing several environment or reward values.",
        ):
            st.rerun()
    with training_tab:
        controls = st.session_state[f"{p}_training_controls"]
        st.subheader(algo_names[room_num])
        if room_num == 1:
            selected_algo = st.radio(
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

            gamma = st.slider(
                "Gamma", 0.0, 0.999, float(controls.get("gamma", 0.95)), 0.001,
                key=f"{p}_gamma",
                help="Discount factor: higher values make future rewards more important.",
            )
            theta = st.number_input(
                "Theta", 1e-12, 1.0, float(controls.get("theta", 1e-6)), format="%.8f",
                key=f"{p}_theta",
                help="Value update convergence threshold.",
            )
            if selected_algo == "Value Iteration":
                max_vi = st.number_input(
                    "Max value iterations (sweeps)", 1, 100000, int(controls.get("max_iterations", 1000)),
                    key=f"{p}_max_vi",
                    help="Maximum Bellman optimality sweeps allowed for Value Iteration.",
                )
                seed = st.number_input(
                    "Seed", 0, value=int(controls.get("seed", 42)), key=f"{p}_seed",
                    help="Controls reproducible policy tie-breaking.",
                )
                live_update = st.number_input(
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
                max_pi = st.number_input(
                    "Max policy iterations", 1, 1000, int(controls.get("max_policy_iterations", 100)),
                    key=f"{p}_max_pi",
                    help="Safety limit on complete policy evaluation-and-improvement cycles.",
                )
                max_sweeps = st.number_input(
                    "Max evaluation sweeps", 1, 100000, int(controls.get("max_evaluation_sweeps", 10000)),
                    key=f"{p}_max_sweeps",
                    help="Maximum Bellman sweeps allowed during each policy evaluation phase.",
                )
                seed = st.number_input(
                    "Seed", 0, value=int(controls.get("seed", 42)), key=f"{p}_seed",
                    help="Controls reproducible policy initialization and tie-breaking.",
                )
                live_update = st.number_input(
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
            alpha = st.slider(
                "Alpha (learning rate)", 0.01, 1.0, float(controls["alpha"]), 0.01,
                key=f"{p}_alpha", help="Controls how strongly each new experience changes the Q-value.",
            )
            gamma = st.slider(
                "Gamma (discount)", 0.0, 0.999, float(controls["gamma"]), 0.001,
                key=f"{p}_gamma", help="Higher values give more weight to rewards farther in the future.",
            )
            eps_start = st.slider(
                "Epsilon start", 0.05, 1.0, float(controls["epsilon_start"]), 0.05,
                key=f"{p}_eps_start", help="Initial probability of choosing a random exploratory action.",
            )
            eps_min = st.slider(
                "Epsilon min", 0.01, 0.5, float(controls["epsilon_min"]), 0.01,
                key=f"{p}_eps_min", help="Minimum exploration probability retained late in training.",
            )
            eps_decay = st.number_input(
                "Epsilon decay rate", 0.8, 1.0, float(controls["epsilon_decay"]),
                format="%.4f", key=f"{p}_eps_decay",
                help="Multiplier applied to epsilon after each episode; closer to 1 decays more slowly.",
            )
            episodes = st.number_input(
                "Training episodes", 10, 10000, int(controls["episodes"]), key=f"{p}_episodes",
                help="Number of complete learning episodes to run.",
            )
            max_steps = st.number_input(
                "Max timesteps per episode", 10, 5000, int(controls["max_timesteps"]),
                key=f"{p}_max_steps", help="Stops an episode that has not reached a termination state.",
            )
            seed = st.number_input(
                "Seed", 0, value=int(controls["seed"]), key=f"{p}_seed",
                help="Controls reproducible exploration and stochastic transitions.",
            )
            live_update = st.number_input(
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
            st.error(f"Fix grid error: {prob_err}")
        if st.button(
            "▶ Train / compute policy", type="primary", use_container_width=True,
            disabled=prob_err is not None, key=f"{p}_train_btn",
            help="Starts training with the current grid, rewards, and hyperparameters.",
        ):
            request_full_app_action(action_key, "train")
        if st.button(
            "Reset trained model", use_container_width=True, key=f"{p}_reset_btn",
            help="Clears the trained policy and test results without changing the grid.",
        ):
            request_full_app_action(action_key, "reset")

    with testing_tab:
        controls = st.session_state[f"{p}_test_controls"]
        st.subheader("Test configuration")
        episodes = st.number_input(
            "Test episodes", 1, 10000, int(controls["episodes"]), key=f"{p}_test_episodes",
            help="Number of evaluation episodes run without learning.",
        )
        max_steps = st.number_input(
            "Max timesteps per episode", 1, 50000, int(controls["max_timesteps"]),
            key=f"{p}_test_max_steps", help="Maximum length of each evaluation episode.",
        )
        seed = st.number_input(
            "Test seed", 0, value=int(controls["seed"]), key=f"{p}_test_seed",
            help="Reproduces the same stochastic test outcomes when settings are unchanged.",
        )
        st.session_state[f"{p}_test_controls"] = {"episodes": int(episodes), "max_timesteps": int(max_steps), "seed": int(seed)}
        if st.button(
            "🧪 Run test", type="primary", use_container_width=True,
            disabled=st.session_state[f"{p}_result"] is None, key=f"{p}_run_test_btn",
            help="Evaluates the current trained model without updating it.",
        ):
            request_full_app_action(action_key, "test")

    with models_tab:
        st.subheader("Model artifact")
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
            st.download_button(
                f"⬇ Download Room {room_num} model",
                data=art,
                file_name=f"room{room_num}_model.json",
                mime="application/json",
                use_container_width=True,
                on_click="ignore",
                help="Downloads the trained policy, environment layout, rewards, and hyperparameters as JSON.",
            )

        uploaded = st.file_uploader(
            "Upload model JSON", type=["json"], key=f"{p}_upload",
            help="Select a previously exported model artifact for this room and algorithm.",
        )
        if uploaded is not None and st.button(
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
                st.error(f"Invalid artifact: {exc}")
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
                st.success("Model loaded successfully!")
                st.rerun()

    return "All", requests, run_test


def render_room(room_num: int) -> None:
    _section, requests, run_test = render_room_controls(room_num)
    room1_algo = st.session_state.get("room1_algorithm", "Policy Iteration")
    algorithms = {1: room1_algo, 2: "SARSA", 3: "Q-Learning"}
    descriptions = {
        1: "Design the grid, compute an exact dynamic-programming policy, inspect convergence, and validate the solution from one continuous workspace.",
        2: "Configure the icy grid and follow the on-policy SARSA workflow from exploration through evaluation and replay.",
        3: "Build the room, train an off-policy Q-Learning agent, and compare its learned behavior with deterministic tests.",
    }
    render_room_page_header(
        room_num,
        "Grid escape workspace",
        algorithms[room_num],
        descriptions[room_num],
    )
    with room_page_section(
        room_num,
        "environment",
        "Environment and algorithm",
        "Edit the complete 10×10 map and review exactly what the agent observes and can do.",
    ):
        render_environment_page(room_num, show_header=False)
    with room_page_section(
        room_num,
        "training",
        "Training and learned policy",
        "Live progress, final policy, convergence charts, and training-episode replay stay together.",
    ):
        training_section_slot = st.empty()
        with training_section_slot.container():
            render_training_page(room_num, requests, show_header=False)
    with room_page_section(
        room_num,
        "testing",
        "Deterministic testing",
        "Evaluate the stored policy without learning and inspect episode-level results and replay.",
    ):
        render_test_page(room_num, run_test, show_header=False)
    with room_page_section(
        room_num,
        "models",
        "Model information",
        "Inspect the model currently held in memory; import and export actions remain in the sidebar.",
    ):
        render_models_page(room_num, show_header=False)


# =====================================================================
# ROOM 4 (PPO - FLAPPY BIRD) IMPLEMENTATION
# =====================================================================

ROOM4_CONTROL_HELP = {
    "section": "Choose which Room 4 control group appears in the sidebar. The full environment, training, testing, and model dashboard remains visible on the page.",
    "pipe_count": "Sets any positive number of obstacles. Whenever this value changes, all pipes are redistributed at equal horizontal distances between 2m and 8m.",
    "pipe_x": "The horizontal center of this pipe in meters. Moving it changes where the agent must pass the obstacle; avoid overlapping another pipe or the goal zone.",
    "pipe_width": "The horizontal thickness of the pipe. A wider pipe occupies more travel distance and makes collision avoidance harder.",
    "gap_start": "The height, in meters, where the safe opening begins. It controls the lower edge of the gap and therefore the vertical route the agent must learn.",
    "gap_size": "The vertical size of the safe opening. Smaller gaps make the task harder; the gap must remain fully inside the 10m room.",
    "step_reward": "Reward added on every timestep. A small negative value encourages shorter solutions; a stronger penalty may make the agent overly cautious about exploring.",
    "progress_reward": "Reward multiplier for positive horizontal progress. The reward is multiplied by the distance moved to the right during that timestep.",
    "backward_reward": "Reward added whenever the agent moves left, including diagonal left actions. Use a negative value to penalize backward movement, or zero to ignore it.",
    "hover_reward": "Penalty applied specifically when PPO selects HOVER. It is added on top of the ordinary step reward and the non-right-action penalty.",
    "non_right_reward": "Penalty applied whenever an action has no rightward horizontal component: left, vertical, or HOVER. Diagonal right actions still count as rightward progress.",
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
    previous_count = len(st.session_state.room4_pipes)
    st.session_state.room4_pipes = pipes
    _sync_room4_pipe_widget_state(pipes, overwrite=True)
    for idx in range(pipe_count, previous_count):
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
    action_key = "room4_pending_control_action"
    pending_action = st.session_state.pop(action_key, None)
    st.sidebar.title("Room 4 Controls (PPO)")
    environment_tab, training_tab, testing_tab, models_tab = render_control_section_tabs(
        4,
        "room4_control_section",
        ROOM4_CONTROL_HELP["section"],
    )

    requests: dict[str, bool] = {
        "train": pending_action == "train",
        "reset": pending_action == "reset",
    }
    run_test = pending_action == "test"

    with environment_tab:
        st.subheader("Flappy Bird pipe obstacles")
        current_pipes = list(st.session_state.room4_pipes)
        if int(st.session_state.get("room4_pipe_count_v2", len(current_pipes))) != len(
            current_pipes
        ):
            st.session_state.room4_pipe_count_v2 = len(current_pipes)
        st.number_input(
            "Number of pipes",
            min_value=1,
            value=len(current_pipes),
            step=1,
            key="room4_pipe_count_v2",
            on_change=_redistribute_room4_pipes,
            help=ROOM4_CONTROL_HELP["pipe_count"],
        )

        current_pipes = list(st.session_state.room4_pipes)
        _sync_room4_pipe_widget_state(current_pipes)

        updated_pipes = []
        pipe_configuration_valid = True
        for idx, pipe in enumerate(current_pipes):
            with st.expander(f"Pipe {idx + 1} configuration", expanded=False):
                px = st.number_input(
                    f"Pipe {idx + 1} X position (m)",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    key=f"p_{idx}_x",
                    help=ROOM4_CONTROL_HELP["pipe_x"],
                )
                pw = st.number_input(
                    f"Pipe {idx + 1} width (m)",
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    key=f"p_{idx}_w",
                    help=ROOM4_CONTROL_HELP["pipe_width"],
                )
                g_start = st.number_input(
                    f"Pipe {idx + 1} gap start Y (m)",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    key=f"p_{idx}_gs",
                    help=ROOM4_CONTROL_HELP["gap_start"],
                )
                g_size = st.number_input(
                    f"Pipe {idx + 1} gap size (m)",
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    key=f"p_{idx}_gz",
                    help=ROOM4_CONTROL_HELP["gap_size"],
                )
                candidate_pipe = PipeObstacle(x=px, width=pw, gap_start=g_start, gap_size=g_size)
                if candidate_pipe.x_min < 0.0 or candidate_pipe.x_max > 10.0:
                    st.error("The complete pipe width must remain inside the 10 m room.")
                    pipe_configuration_valid = False
                if candidate_pipe.gap_end > 10.0:
                    st.error("Gap start + gap size must not exceed the 10 m room height.")
                    pipe_configuration_valid = False
                updated_pipes.append(candidate_pipe)

        if pipe_configuration_valid and updated_pipes != st.session_state.room4_pipes:
            st.session_state.room4_pipes = updated_pipes
            st.session_state.room4_result = None
            st.session_state.room4_result_environment = None
            st.session_state.room4_algorithm_config = None
            st.session_state.room4_test_results = None

        st.subheader("Reward structure")
        rewards = dict(st.session_state.room4_reward_values)
        previous_enabled = set(st.session_state.room4_reward_enabled)
        reward_specs = (
            ("step", "Step reward", -0.05, 0.01, "r4_r_step", ROOM4_CONTROL_HELP["step_reward"]),
            ("progress", "Progress reward", 0.5, 0.1, "r4_r_prog", ROOM4_CONTROL_HELP["progress_reward"]),
            ("backward", "Backward movement reward", 0.0, 0.1, "r4_r_back", ROOM4_CONTROL_HELP["backward_reward"]),
            ("hover", "Hover reward", -0.1, 0.1, "r4_r_hover", ROOM4_CONTROL_HELP["hover_reward"]),
            ("non_right", "Non-right action reward", -0.1, 0.1, "r4_r_non_right", ROOM4_CONTROL_HELP["non_right_reward"]),
            ("pipe_passed", "Pipe passed reward", 5.0, 0.5, "r4_r_pipe", ROOM4_CONTROL_HELP["pipe_reward"]),
            ("goal_reached", "Goal reward", 20.0, 1.0, "r4_r_goal", ROOM4_CONTROL_HELP["goal_reward"]),
            ("collision", "Collision reward", -20.0, 1.0, "r4_r_coll", ROOM4_CONTROL_HELP["collision_reward"]),
        )
        next_rewards: dict[str, float] = {}
        reward_widget_values: dict[str, float] = {}
        for event, label, default, step, widget_key, help_text in reward_specs:
            actual_value, configured_value = render_reward_control(
                enabled_state_key="room4_reward_enabled",
                event=event,
                label=label,
                value=float(rewards.get(event, default)),
                widget_key=widget_key,
                step=step,
                number_format="%.2f",
                help_text=help_text,
            )
            next_rewards[event] = actual_value
            reward_widget_values[widget_key] = configured_value
        style_reward_controls(reward_widget_values)

        if next_rewards != rewards or st.session_state.room4_reward_enabled != previous_enabled:
            st.session_state.room4_reward_values = next_rewards
            st.session_state.room4_result = None
            st.session_state.room4_result_environment = None
            st.session_state.room4_algorithm_config = None
            st.session_state.room4_test_results = None

    with training_tab:
        controls = st.session_state.room4_training_controls
        st.subheader("PPO hyperparameters")
        alpha = st.number_input("Learning rate (α)", 0.00001, 0.1, float(controls.get("alpha", 0.0003)), format="%.5f", key="r4_alpha", help=ROOM4_CONTROL_HELP["alpha"])
        gamma = st.number_input("Discount factor (γ)", 0.0, 0.999, float(controls.get("gamma", 0.99)), format="%.3f", key="r4_gamma", help=ROOM4_CONTROL_HELP["gamma"])
        gae_lambda = st.number_input("GAE lambda (λ)", 0.0, 1.0, float(controls.get("gae_lambda", 0.95)), format="%.3f", key="r4_gae_lambda", help=ROOM4_CONTROL_HELP["gae_lambda"])
        clip_eps = st.number_input("Clipping epsilon (ε)", 0.01, 0.5, float(controls.get("clip_epsilon", 0.2)), format="%.2f", key="r4_clip_eps", help=ROOM4_CONTROL_HELP["clip_epsilon"])
        ent_coef = st.number_input("Entropy coefficient", 0.0, 0.5, float(controls.get("entropy_coefficient", 0.01)), format="%.4f", key="r4_ent_coef", help=ROOM4_CONTROL_HELP["entropy_coefficient"])
        val_coef = st.number_input("Value loss coefficient", 0.0, 2.0, float(controls.get("value_coefficient", 0.5)), format="%.2f", key="r4_val_coef", help=ROOM4_CONTROL_HELP["value_coefficient"])
        update_epochs = st.number_input("PPO update epochs", 1, 20, int(controls.get("update_epochs", 4)), key="r4_update_epochs", help=ROOM4_CONTROL_HELP["update_epochs"])
        mini_batch = st.number_input("Mini-batch size", 8, 512, int(controls.get("mini_batch_size", 64)), key="r4_mini_batch", help=ROOM4_CONTROL_HELP["mini_batch_size"])
        episodes = st.number_input("Episodes", 1, 5000, int(controls.get("episodes", 300)), key="r4_episodes", help=ROOM4_CONTROL_HELP["episodes"])
        max_steps = st.number_input("Maximum timesteps", 50, 2000, int(controls.get("max_timesteps", 500)), key="r4_max_steps", help=ROOM4_CONTROL_HELP["max_timesteps"])

        st.subheader("Neural network architecture")
        h_layers = st.slider("Hidden layer count", 1, 4, int(controls.get("hidden_layers", 2)), key="r4_h_layers", help=ROOM4_CONTROL_HELP["hidden_layers"])
        h_units = st.select_slider("Neurons per hidden layer", options=[16, 32, 64, 128, 256], value=int(controls.get("hidden_units", 64)), key="r4_h_units", help=ROOM4_CONTROL_HELP["hidden_units"])
        activation_opts = ["Tanh", "ReLU", "LeakyReLU", "ELU", "SiLU"]
        current_act = controls.get("activation_fn", "Tanh")
        act_idx = activation_opts.index(current_act) if current_act in activation_opts else 0
        act_fn = st.selectbox("Activation function", options=activation_opts, index=act_idx, key="r4_activation_fn", help=ROOM4_CONTROL_HELP["activation"])

        seed = st.number_input("Random seed", 0, value=int(controls.get("seed", 42)), key="r4_seed", help=ROOM4_CONTROL_HELP["seed"])
        live_update = st.number_input("Update charts every N episodes", 1, 100, int(controls.get("live_update_every", 10)), key="r4_live_update", help=ROOM4_CONTROL_HELP["live_update"])

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

        if st.button("Train PPO agent", icon=":material/play_arrow:", type="primary", width="stretch", key="r4_train_btn", help=ROOM4_CONTROL_HELP["train"]):
            request_full_app_action(action_key, "train")
        if st.button("Reset trained model", icon=":material/restart_alt:", width="stretch", key="r4_reset_btn", help=ROOM4_CONTROL_HELP["reset"]):
            request_full_app_action(action_key, "reset")

    with testing_tab:
        controls = st.session_state.room4_test_controls
        st.subheader("Test configuration")
        episodes = st.number_input("Test episodes", 1, 1000, int(controls["episodes"]), key="r4_test_episodes", help=ROOM4_CONTROL_HELP["test_episodes"])
        max_steps = st.number_input("Maximum timesteps per episode", 10, 5000, int(controls["max_timesteps"]), key="r4_test_max_steps", help=ROOM4_CONTROL_HELP["test_timesteps"])
        seed = st.number_input("Test seed", 0, value=int(controls["seed"]), key="r4_test_seed", help=ROOM4_CONTROL_HELP["test_seed"])

        st.session_state.room4_test_controls = {"episodes": int(episodes), "max_timesteps": int(max_steps), "seed": int(seed)}
        if st.button(
            "Run test", icon=":material/science:", type="primary", width="stretch",
            disabled=st.session_state.room4_result is None, key="r4_run_test_btn",
            help=ROOM4_CONTROL_HELP["run_test"],
        ):
            request_full_app_action(action_key, "test")

    with models_tab:
        st.subheader("Model artifact")
        res = st.session_state.room4_result
        env = st.session_state.room4_result_environment
        config = st.session_state.room4_algorithm_config
        if res and env and config:
            art = export_room4_artifact(env, config, res)
            st.download_button(
                "Download Room 4 model (JSON)",
                data=art,
                file_name="room4_ppo_model.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch",
                on_click="ignore",
                help=ROOM4_CONTROL_HELP["download"],
            )

        uploaded = st.file_uploader("Upload model JSON", type=["json"], key="r4_upload", help=ROOM4_CONTROL_HELP["upload"])
        
        def r4_load_model_callback():
            if st.session_state.r4_upload is not None:
                try:
                    env_l, config_l, res_l = import_room4_artifact(st.session_state.r4_upload.getvalue().decode("utf-8"))
                except Exception as exc:
                    st.session_state.r4_load_error = str(exc)
                else:
                    st.session_state.room4_result_environment = env_l
                    st.session_state.room4_algorithm_config = config_l
                    st.session_state.room4_result = res_l
                    st.session_state.room4_pipes = list(env_l.config.pipes)
                    st.session_state.room4_pipe_count_v2 = len(env_l.config.pipes)
                    _sync_room4_pipe_widget_state(list(env_l.config.pipes), overwrite=True)
                    st.session_state.room4_reward_values = dict(env_l.config.rewards)
                    st.session_state.room4_reward_enabled = {
                        event for event, value in env_l.config.rewards.items() if float(value) != 0.0
                    }
                    st.session_state.r4_load_success = True

        if uploaded is not None:
            st.button("Load model", icon=":material/upload_file:", width="stretch", key="r4_load_btn", help=ROOM4_CONTROL_HELP["load"], on_click=r4_load_model_callback)
            
        if st.session_state.pop("r4_load_error", None):
            st.error(f"Invalid artifact: {st.session_state.r4_load_error}")
        if st.session_state.pop("r4_load_success", None):
            st.success("Room 4 PPO model loaded successfully!")

    return "All", requests, run_test


def _render_room4_training_summary(result: Any, *, show_charts: bool = True) -> None:
    df = pd.DataFrame([asdict(m) for m in result.metrics])
    summary_cols = st.columns(3)
    summary_cols[0].metric(
        "Training duration",
        _format_training_duration(float(getattr(result, "training_duration_seconds", 0.0)))
    )
    summary_cols[1].metric("Episodes completed", result.episodes_run)
    summary_cols[2].metric("Actions selected", sum(getattr(result, "action_counts", {}).values()))

    if show_charts:
        st.subheader("Training metrics")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Total reward per episode**")
            render_locked_line_chart(df, x="episode", y="total_reward", x_label="Episode", y_label="Total reward")
            st.caption("**Entropy**")
            if "entropy" in df.columns:
                render_locked_line_chart(df, x="episode", y="entropy", x_label="Episode", y_label="Entropy")
        with c2:
            st.caption("**Policy loss**")
            if "policy_loss" in df.columns:
                render_locked_line_chart(df, x="episode", y="policy_loss", x_label="Episode", y_label="Policy loss")
            st.caption("**Value loss**")
            if "value_loss" in df.columns:
                render_locked_line_chart(df, x="episode", y="value_loss", x_label="Episode", y_label="Value loss")

    st.subheader("Training action distribution")
    render_locked_bar_chart(
        _room4_action_dataframe(getattr(result, "action_counts", {})),
        x="Action",
        y="Selections",
        x_label="Action",
        y_label="Number of selections",
    )

    st.write(f"**Episodes run:** {result.episodes_run} | **Goal reached in late training:** {'Yes ✅' if result.converged else 'No ❌'}")
    if hasattr(result, "training_episodes") and result.training_episodes:
        env_cur = st.session_state.room4_result_environment or build_room4_environment()
        render_episode_replay_visualizer(
            env_cur,
            result.training_episodes,
            "room4_tr_replay",
            4,
            title="Training Episodes Replay",
        )

def _render_room4_section(
    section: str,
    requests: dict[str, bool],
    run_test: bool,
) -> None:
    if section == "Environment":
        env = build_room4_environment()
        render_algorithm_overview(4)
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

        training_notice = st.session_state.pop("room4_training_notice", None)
        if training_notice:
            st.success(training_notice)

        if requests["train"]:
            st.session_state.room4_result = None
            st.session_state.room4_result_environment = None
            st.session_state.room4_algorithm_config = None
            st.session_state.room4_test_results = None
            st.session_state.pop("room4_training_notice", None)
            for replay_key in (
                "room4_tr_replay_select",
                "room4_tr_replay_step",
                "room4_tr_replay_is_playing",
                "room4_tr_replay_speed",
            ):
                st.session_state.pop(replay_key, None)
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
                    render_locked_line_chart(df_live, x="episode", y="total_reward", x_label="Episode", y_label="Total reward", target=slot_reward)
                    render_locked_line_chart(df_live, x="episode", y="policy_loss", x_label="Episode", y_label="Policy loss", target=slot_ploss)
                    render_locked_line_chart(df_live, x="episode", y="value_loss", x_label="Episode", y_label="Value loss", target=slot_vloss)
                    render_locked_line_chart(df_live, x="episode", y="entropy", x_label="Episode", y_label="Entropy", target=slot_entropy)

            with st.spinner("Training PPO agent..."):
                result = run_ppo(env, config, callback=live_callback)

            st.session_state.room4_result = result
            st.session_state.room4_result_environment = env
            st.session_state.room4_algorithm_config = config
            st.session_state.room4_test_results = None
            duration_label = _format_training_duration(result.training_duration_seconds)
            st.success(f"PPO training complete in {duration_label}.")
            for replay_key in (
                "room4_tr_replay_select",
                "room4_tr_replay_step",
                "room4_tr_replay_is_playing",
            ):
                st.session_state.pop(replay_key, None)

        res = st.session_state.room4_result
        if res:
            _render_room4_training_summary(res, show_charts=not requests["train"])
        else:
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
            render_locked_bar_chart(
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


def render_room4() -> None:
    _section, requests, run_test = render_room4_controls()
    render_room_page_header(
        4,
        "Continuous Flappy Bird workspace",
        "PPO",
        "Design continuous pipe obstacles, train the actor-critic policy, inspect its action distribution, and replay complete flights from one page.",
    )
    with room_page_section(
        4,
        "environment",
        "Environment and observation space",
        "Review the continuous 10×10 room, the discrete velocity actions, and every configured pipe.",
    ):
        _render_room4_section("Environment", {"train": False, "reset": False}, False)
    with room_page_section(
        4,
        "training",
        "PPO training dashboard",
        "Training KPIs, locked charts, action distribution, and episode replay remain visible after training.",
    ):
        training_section_slot = st.empty()
        with training_section_slot.container():
            _render_room4_section("Training", requests, False)
    with room_page_section(
        4,
        "testing",
        "Policy testing and replay",
        "Run deterministic evaluation from the sidebar and inspect every recorded flight here.",
    ):
        _render_room4_section("Testing", {"train": False, "reset": False}, run_test)
    with room_page_section(
        4,
        "models",
        "Model and network information",
        "Inspect the active PPO network; download and upload actions stay in the sidebar.",
    ):
        _render_room4_section("Models", {"train": False, "reset": False}, False)


# =====================================================================
# ROOM 5 (PPO - ONE-WAY ROAD) IMPLEMENTATION
# =====================================================================

ROOM5_HELP = {
    "section": "Choose which Room 5 control group appears in the sidebar. The complete road, training, testing, and model dashboard remains visible on the page.",
    "lanes": "Number of one-way traffic lanes. The agent can move left, keep its lane, or move right. More lanes provide more escape routes but enlarge the decision space.",
    "vision": "Maximum edge-to-edge clearance ahead, in meters, observed in the agent's current lane only. Distance is measured from the agent's front edge to the rear edge of the nearest traffic car. Cars in adjacent lanes are hidden until the agent changes into that lane.",
    "road_length": "Forward distance required to complete an episode successfully. Longer roads require the policy to avoid traffic for more timesteps.",
    "traffic_count": "Target number of slower same-direction cars. Every episode starts empty, then cars arrive progressively until this traffic level is reached.",
    "ego_speed": "Constant speed of the agent car. It must remain faster than traffic so that other cars approach in the agent-relative view and can be overtaken.",
    "traffic_min": "Minimum speed assigned to a traffic car. Slower cars approach the agent more quickly and are harder to avoid.",
    "traffic_max": "Maximum traffic-car speed. It stays below the agent speed to preserve same-direction overtaking behavior.",
    "env_seed": "Controls arriving traffic lanes, distances, and speeds so the same traffic sequence can be reproduced.",
    "step": "Reward on every timestep. A small negative value discourages unnecessarily long episodes.",
    "progress": "Reward multiplier per meter of forward travel. It provides dense feedback even before a car is overtaken.",
    "overtake": "Reward granted each time the agent safely passes a traffic car.",
    "lane_change": "Reward applied to a valid lane change. A small penalty reduces needless weaving while still allowing evasive maneuvers.",
    "safer_change": "Reward for a valid lane change that increases the observable distance to the nearest car ahead. An empty destination lane counts as clearance up to the vision limit.",
    "riskier_change": "Penalty for a valid lane change that decreases the observable distance to the nearest car ahead. This discourages moving into tighter traffic.",
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
            road_length=float(controls.get("road_length", 50.0)),
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
    action_key = "room5_pending_control_action"
    pending_action = st.session_state.pop(action_key, None)
    st.sidebar.title("Room 5 Controls (PPO)")
    environment_tab, training_tab, testing_tab, models_tab = render_control_section_tabs(
        5,
        "room5_control_section",
        ROOM5_HELP["section"],
    )
    requests = {
        "train": pending_action == "train",
        "reset": pending_action == "reset",
    }
    run_test = pending_action == "test"

    with environment_tab:
        controls = dict(st.session_state.room5_environment_controls)
        st.subheader("One-way road")
        lane_count = st.number_input("Number of lanes", min_value=2, max_value=6, value=int(controls["lane_count"]), step=1, key="r5_lanes_input", help=ROOM5_HELP["lanes"])
        road_length = st.number_input("Road length (m)", min_value=10.0, value=float(controls.get("road_length", 50.0)), step=10.0, format="%.1f", key="r5_length_input", help=ROOM5_HELP.get("road_length", "The total length of the one-way road track."))
        vision = st.number_input("Forward vision distance (m)", min_value=5.0, max_value=50.0, value=max(5.0, min(50.0, float(controls["vision_distance"]))), step=5.0, format="%.1f", key="r5_vision_input", help=ROOM5_HELP["vision"])
        traffic_count = st.number_input("Traffic car count", min_value=0, value=int(controls["traffic_count"]), step=1, key="r5_traffic_count_input", help=ROOM5_HELP["traffic_count"])
        ego_speed = st.number_input("Agent speed (m/s)", min_value=0.1, value=float(controls["ego_speed"]), step=1.0, format="%.1f", key="r5_ego_speed_input", help=ROOM5_HELP["ego_speed"])
        traffic_min = st.number_input("Minimum traffic speed (m/s)", min_value=0.0, value=float(controls["traffic_speed_min"]), step=1.0, format="%.1f", key="r5_traffic_min_input", help=ROOM5_HELP["traffic_min"])
        traffic_max = st.number_input("Maximum traffic speed (m/s)", min_value=0.0, value=float(controls["traffic_speed_max"]), step=1.0, format="%.1f", key="r5_traffic_max_input", help=ROOM5_HELP["traffic_max"])
        env_seed = st.number_input("Environment seed", min_value=0, value=int(controls["seed"]), key="r5_env_seed", help=ROOM5_HELP["env_seed"])

        requested_controls = {
            "lane_count": int(lane_count),
            "vision_distance": float(vision),
            "road_length": float(road_length),
            "ego_speed": float(ego_speed),
            "traffic_speed_min": float(traffic_min),
            "traffic_speed_max": float(traffic_max),
            "traffic_count": int(traffic_count),
            "seed": int(env_seed),
        }
        environment_errors: list[str] = []
        if float(traffic_min) > float(traffic_max):
            environment_errors.append("Minimum traffic speed cannot exceed maximum traffic speed.")
        if float(traffic_max) >= float(ego_speed):
            environment_errors.append("Maximum traffic speed must remain below the agent speed.")
        for message in environment_errors:
            st.error(message)

        st.subheader("Reward structure")
        rewards = dict(st.session_state.room5_reward_values)
        previous_reward_enabled = set(st.session_state.room5_reward_enabled)
        reward_specs = (
            ("step", "Step reward", 0.01, "%.2f", "r5_step_reward_input", ROOM5_HELP["step"]),
            ("forward_progress", "Forward progress reward per meter", 0.01, "%.2f", "r5_progress_reward_input", ROOM5_HELP["progress"]),
            ("overtake", "Overtake reward", 0.5, "%.1f", "r5_overtake_reward_input", ROOM5_HELP["overtake"]),
            ("lane_change", "Lane-change reward", 0.05, "%.2f", "r5_lane_change_reward_input", ROOM5_HELP["lane_change"]),
            ("safer_lane_change", "Safer lane-change reward", 0.1, "%.1f", "r5_safer_change_reward_input", ROOM5_HELP["safer_change"]),
            ("riskier_lane_change", "Riskier lane-change reward", 0.1, "%.1f", "r5_riskier_change_reward_input", ROOM5_HELP["riskier_change"]),
            ("invalid_lane_change", "Invalid lane-change reward", 0.5, "%.1f", "r5_invalid_change_reward_input", ROOM5_HELP["invalid_change"]),
            ("collision", "Collision reward", 1.0, "%.1f", "r5_collision_reward_input", ROOM5_HELP["collision"]),
            ("goal_reached", "Road completion reward", 1.0, "%.1f", "r5_goal_reward_input", ROOM5_HELP["goal"]),
        )
        next_rewards: dict[str, float] = {}
        reward_widget_values: dict[str, float] = {}
        for event, label, step, number_format, widget_key, help_text in reward_specs:
            actual_value, configured_value = render_reward_control(
                enabled_state_key="room5_reward_enabled",
                event=event,
                label=label,
                value=float(rewards[event]),
                widget_key=widget_key,
                step=step,
                number_format=number_format,
                help_text=help_text,
            )
            next_rewards[event] = actual_value
            reward_widget_values[widget_key] = configured_value
        style_reward_controls(reward_widget_values)
        environment_changed = not environment_errors and requested_controls != controls
        rewards_changed = (
            next_rewards != rewards
            or st.session_state.room5_reward_enabled != previous_reward_enabled
        )
        if environment_changed:
            st.session_state.room5_environment_controls = requested_controls
        if rewards_changed:
            st.session_state.room5_reward_values = next_rewards
        if environment_changed or rewards_changed:
            _invalidate_room5_model()

    with training_tab:
        controls = st.session_state.room5_training_controls
        st.subheader("PPO hyperparameters")
        alpha = st.number_input("Learning rate", 0.00001, 0.01, float(controls["alpha"]), step=0.00001, format="%.5f", key="r5_alpha", help=ROOM5_HELP["alpha"])
        gamma = st.number_input("Discount factor", 0.0, 1.0, float(controls["gamma"]), format="%.3f", key="r5_gamma", help=ROOM5_HELP["gamma"])
        gae_lambda = st.number_input("GAE lambda", 0.0, 1.0, float(controls["gae_lambda"]), format="%.3f", key="r5_gae", help=ROOM5_HELP["gae"])
        clip_epsilon = st.number_input("PPO clip epsilon", 0.01, 0.5, float(controls["clip_epsilon"]), format="%.3f", key="r5_clip", help=ROOM5_HELP["clip"])
        entropy_coefficient = st.number_input("Entropy coefficient", 0.0, 0.2, float(controls["entropy_coefficient"]), format="%.3f", key="r5_entropy_coef", help=ROOM5_HELP["entropy"])
        value_coefficient = st.number_input("Value-loss coefficient", 0.0, 2.0, float(controls["value_coefficient"]), format="%.2f", key="r5_value_coef", help=ROOM5_HELP["value_coef"])
        update_epochs = st.number_input("Update epochs per rollout", 1, 20, int(controls["update_epochs"]), key="r5_update_epochs", help=ROOM5_HELP["epochs"])
        mini_batch_size = st.select_slider("Mini-batch size", [16, 32, 64, 128, 256], value=int(controls["mini_batch_size"]), key="r5_batch", help=ROOM5_HELP["batch"])
        episodes = st.number_input("Episodes", 1, 5000, int(controls["episodes"]), key="r5_episodes", help=ROOM5_HELP["episodes"])
        max_timesteps = st.number_input("Maximum timesteps", 10, 2000, int(controls["max_timesteps"]), key="r5_max_steps", help=ROOM5_HELP["timesteps"])

        st.subheader("Actor-critic network")
        hidden_layers = st.slider("Hidden layer count", 1, 4, int(controls["hidden_layers"]), key="r5_hidden_layers", help=ROOM5_HELP["layers"])
        hidden_units = st.select_slider("Neurons per hidden layer", [32, 64, 128, 256], value=int(controls["hidden_units"]), key="r5_hidden_units", help=ROOM5_HELP["units"])
        activations = ["Tanh", "ReLU", "LeakyReLU", "ELU", "SiLU"]
        activation = st.selectbox("Activation function", activations, index=activations.index(controls["activation_fn"]), key="r5_activation", help=ROOM5_HELP["activation"])
        seed = st.number_input("Training seed", 0, value=int(controls["seed"]), key="r5_train_seed", help=ROOM5_HELP["train_seed"])
        live_update = st.number_input("Update charts every N episodes", 1, 100, int(controls["live_update_every"]), key="r5_live_update", help=ROOM5_HELP["live"])
        st.session_state.room5_training_controls = {
            "alpha": float(alpha), "gamma": float(gamma), "gae_lambda": float(gae_lambda),
            "clip_epsilon": float(clip_epsilon), "entropy_coefficient": float(entropy_coefficient),
            "value_coefficient": float(value_coefficient), "update_epochs": int(update_epochs),
            "mini_batch_size": int(mini_batch_size), "episodes": int(episodes),
            "max_timesteps": int(max_timesteps), "hidden_layers": int(hidden_layers),
            "hidden_units": int(hidden_units), "activation_fn": str(activation),
            "seed": int(seed), "live_update_every": int(live_update),
        }
        if st.button("Train PPO agent", icon=":material/play_arrow:", type="primary", width="stretch", key="r5_train", help="Start PPO training with the current road, rewards, and hyperparameters."):
            request_full_app_action(action_key, "train")
        if st.button("Reset trained model", icon=":material/restart_alt:", width="stretch", key="r5_reset", help="Remove the Room 5 model and test results from this browser session."):
            request_full_app_action(action_key, "reset")

    with testing_tab:
        controls = st.session_state.room5_test_controls
        st.subheader("Test configuration")
        episodes = st.number_input("Test episodes", 1, 1000, int(controls["episodes"]), key="r5_test_episodes", help=ROOM5_HELP["test_episodes"])
        max_timesteps = st.number_input("Maximum timesteps per episode", 10, 5000, int(controls["max_timesteps"]), key="r5_test_steps", help=ROOM5_HELP["test_steps"])
        seed = st.number_input("Test seed", 0, value=int(controls["seed"]), key="r5_test_seed", help=ROOM5_HELP["test_seed"])
        st.session_state.room5_test_controls = {"episodes": int(episodes), "max_timesteps": int(max_timesteps), "seed": int(seed)}
        if st.button("Run test", icon=":material/science:", type="primary", width="stretch", disabled=st.session_state.room5_result is None, key="r5_run_test", help="Evaluate the trained PPO policy greedily and record metrics and replay trajectories."):
            request_full_app_action(action_key, "test")

    with models_tab:
        result = st.session_state.room5_result
        environment = st.session_state.room5_result_environment
        algorithm_config = st.session_state.room5_algorithm_config
        st.subheader("Model artifact")
        if result and environment and algorithm_config:
            st.download_button("Download Room 5 model (JSON)", export_room5_artifact(environment, algorithm_config, result), "room5_ppo_model.json", "application/json", icon=":material/download:", width="stretch", on_click="ignore", help="Download the PPO weights, environment, hyperparameters, and training metrics.")
        uploaded = st.file_uploader("Upload Room 5 model JSON", type=["json"], key="r5_upload", help="Select a Room 5 PPO artifact exported by this dashboard.")
        if uploaded is not None and st.button("Load model", icon=":material/upload_file:", width="stretch", key="r5_load", help="Validate and restore the uploaded Room 5 model."):
            try:
                environment, algorithm_config, result = import_room5_artifact(uploaded.getvalue().decode("utf-8"))
            except Exception as exc:
                st.error(f"Invalid artifact: {exc}")
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
                st.session_state.room5_reward_enabled = {
                    event for event, value in config.rewards.items() if float(value) != 0.0
                }
                st.success("Room 5 PPO model loaded successfully.")
                st.rerun()

    return "All", requests, run_test


def _render_room5_training_summary(
    result: Any,
    *,
    show_charts: bool = True,
) -> None:
    metrics = pd.DataFrame([asdict(metric) for metric in result.metrics])
    kpis = st.columns(4)
    kpis[0].metric("Training duration", _format_training_duration(float(result.training_duration_seconds)))
    kpis[1].metric("Episodes", result.episodes_run)
    kpis[2].metric("Total overtakes", int(metrics["overtakes"].sum()))
    kpis[3].metric("Late success", "Yes" if result.converged else "No")

    if show_charts:
        st.subheader("Training metrics")
        reward_column, overtake_column, entropy_column = st.columns(3)
        with reward_column:
            st.caption("**Total reward per episode**")
            render_locked_line_chart(metrics, x="episode", y="total_reward", x_label="Episode", y_label="Total reward")
        with overtake_column:
            st.caption("**Overtakes per episode**")
            render_locked_line_chart(metrics, x="episode", y="overtakes", x_label="Episode", y_label="Overtakes")
        with entropy_column:
            st.caption("**Policy entropy**")
            render_locked_line_chart(metrics, x="episode", y="entropy", x_label="Episode", y_label="Entropy")

    st.subheader("Training action distribution")
    distribution_column, _distribution_spacer = st.columns([1, 2])
    with distribution_column:
        render_locked_bar_chart(_room5_action_dataframe(result.action_counts), x="Action", y="Selections", x_label="Action", y_label="Number of selections")

    if hasattr(result, "training_episodes") and result.training_episodes:
        environment = st.session_state.room5_result_environment or build_room5_environment()
        render_episode_replay_visualizer(
            environment,
            result.training_episodes,
            "room5_tr_replay",
            5,
            title="Training Episodes Replay",
        )



def _render_room5_section(
    section: str,
    requests: dict[str, bool],
    run_test: bool,
) -> None:
    if section == "Environment":
        environment = build_room5_environment()
        render_algorithm_overview(5)
        st.subheader("Configurable one-way road")
        st.markdown(render_room5_html(environment), unsafe_allow_html=True)
        controls = environment.config
        with st.container(horizontal=True):
            st.metric("Lanes", controls.lane_count, border=True)
            st.metric("Vision", f"{controls.vision_distance:.0f}m", border=True)
            st.metric("Target traffic cars", controls.traffic_count, border=True)
            st.metric("Goal distance", f"{controls.road_length:.0f}m", border=True)
        st.caption(
            "Each episode starts with an empty road, then slower same-direction cars arrive "
            "progressively from beyond the vision line. Because the agent is faster, they move "
            "toward it in the agent-relative view and must be avoided or overtaken."
        )

    elif section == "Training":
        if requests["reset"]:
            _invalidate_room5_model()
            st.success("Room 5 trained model reset.")

        training_notice = st.session_state.pop("room5_training_notice", None)
        if training_notice:
            st.success(training_notice)

        if requests["train"]:
            _invalidate_room5_model()
            st.session_state.pop("room5_training_notice", None)
            for replay_key in (
                "room5_tr_replay_select",
                "room5_tr_replay_step",
                "room5_tr_replay_is_playing",
                "room5_tr_replay_speed",
            ):
                st.session_state.pop(replay_key, None)
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
            reward_column, overtake_column, entropy_column = st.columns(3)
            with reward_column:
                st.caption("**Total reward per episode**")
                reward_slot = st.empty()
            with overtake_column:
                st.caption("**Overtakes per episode**")
                overtake_slot = st.empty()
            with entropy_column:
                st.caption("**Policy entropy**")
                entropy_slot = st.empty()
            live_rows: list[dict[str, Any]] = []

            def room5_live_callback(metric: Any, policy_net: Any) -> None:
                live_rows.append(asdict(metric))
                if metric.episode % controls["live_update_every"] == 0 or metric.episode == config.episodes:
                    status.info(
                        f"Training episode {metric.episode}/{config.episodes} • "
                        f"Reward: {metric.total_reward:.2f} • Overtakes: {metric.overtakes} • "
                        f"Policy loss: {metric.policy_loss:.3e} • Entropy: {metric.entropy:.3f}"
                    )
                    frame = pd.DataFrame(live_rows)
                    render_locked_line_chart(frame, x="episode", y="total_reward", x_label="Episode", y_label="Total reward", target=reward_slot)
                    render_locked_line_chart(frame, x="episode", y="overtakes", x_label="Episode", y_label="Overtakes", target=overtake_slot)
                    render_locked_line_chart(frame, x="episode", y="entropy", x_label="Episode", y_label="Entropy", target=entropy_slot)

            with st.spinner("Training PPO agent..."):
                result = run_ppo(environment, config, callback=room5_live_callback)
            st.session_state.room5_result = result
            st.session_state.room5_result_environment = environment
            st.session_state.room5_algorithm_config = config
            st.session_state.room5_test_results = None
            duration = _format_training_duration(result.training_duration_seconds)
            st.success(f"PPO training complete in {duration}.")
            for replay_key in (
                "room5_tr_replay_select",
                "room5_tr_replay_step",
                "room5_tr_replay_is_playing",
            ):
                st.session_state.pop(replay_key, None)

        result = st.session_state.room5_result
        if result is not None:
            _render_room5_training_summary(
                result,
                show_charts=not requests["train"],
            )
        else:
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
                render_locked_line_chart(table, x="Episode", y="Total reward", x_label="Episode", y_label="Total reward")
            action_counts = {action.name: 0 for action in Action5}
            for episode in results:
                for step in episode.trajectory:
                    action_counts[step.action.name] += 1
            with charts[1]:
                st.caption("**Test action distribution**")
                render_locked_bar_chart(_room5_action_dataframe(action_counts), x="Action", y="Selections", x_label="Action", y_label="Number of selections")
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
                "Observation dimension": OBSERVATION_SIZE,
                "Observation schema": "Lane (one-hot), progress, and clearance/speed for left, current, and right lanes",
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


def render_room5() -> None:
    _section, requests, run_test = render_room5_controls()
    render_room_page_header(
        5,
        "One-way traffic avoidance workspace",
        "PPO",
        "Configure the road and current-lane sensor, train a driving policy, measure safety, and replay complete episodes from one continuous dashboard.",
    )
    with room_page_section(
        5,
        "environment",
        "Road environment and agent vision",
        "See the full road while the agent observes only the nearest vehicle in its own forward lane.",
    ):
        _render_room5_section("Environment", {"train": False, "reset": False}, False)
    with room_page_section(
        5,
        "training",
        "PPO training dashboard",
        "Review duration, rewards, overtakes, losses, entropy, action choices, and training replay together.",
    ):
        training_section_slot = st.empty()
        with training_section_slot.container():
            _render_room5_section("Training", requests, False)
    with room_page_section(
        5,
        "testing",
        "Safety testing and episode replay",
        "Run deterministic road tests from the sidebar and compare success, collision, reward, and behavior.",
    ):
        _render_room5_section("Testing", {"train": False, "reset": False}, run_test)
    with room_page_section(
        5,
        "models",
        "Model and network information",
        "Inspect the active observation schema and PPO network; model file actions stay in the sidebar.",
    ):
        _render_room5_section("Models", {"train": False, "reset": False}, False)


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

