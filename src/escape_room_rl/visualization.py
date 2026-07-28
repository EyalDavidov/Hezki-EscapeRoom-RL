"""HTML rendering helpers for the Streamlit client."""

from __future__ import annotations

from html import escape

from typing import Any
from .display_formatting import format_reward_label, reward_sign_class
from .room1 import Action, State

ACTION_ARROWS = {
    Action.UP: "↑",
    Action.DOWN: "↓",
    Action.LEFT: "←",
    Action.RIGHT: "→",
}


def render_grid_html(
    environment: Any,
    agent_state: State | None = None,
    policy: dict[State, Action] | None = None,
    values: dict[State, float] | None = None,
) -> str:
    cells: list[str] = []
    value_numbers = list(values.values()) if values else []
    low = min(value_numbers) if value_numbers else 0.0
    high = max(value_numbers) if value_numbers else 1.0
    span = max(high - low, 1e-12)

    # High x is displayed on the left; (0, 0) is therefore bottom-right.
    for y in reversed(range(environment.config.height)):
        for x in reversed(range(environment.config.width)):
            state = (x, y)
            classes = ["room-cell"]
            content = ""
            reward_label = ""
            title_parts = [f"({x}, {y})"]
            style = ""
            if state in environment.config.walls:
                classes.append("wall")
                content = "🧱"
            else:
                if values and state in values:
                    normalized = (values[state] - low) / span
                    alpha = 0.12 + normalized * 0.45
                    style = f"background-color: rgba(70, 130, 180, {alpha:.3f});"
                    title_parts.append(f"V={values[state]:.4f}")
                if state in environment.config.slippery:
                    classes.append("slippery")
                    content = "❄️"
                if state in environment.config.cell_rewards:
                    classes.append("custom-reward")
                    cell_reward = environment.config.cell_rewards[state]
                    title_parts.append(
                        f"cell reward={cell_reward:.3f}"
                    )
                    reward_label = (
                        f'<span class="cell-reward {reward_sign_class(cell_reward)}">'
                        f'{escape(format_reward_label(cell_reward))}</span>'
                    )
                if state in environment.config.terminal_states and state != environment.goal:
                    classes.append("termination")
                    content = "🛑"
                if state == environment.start:
                    classes.append("start")
                    content = "🏁"
                if state == environment.goal:
                    classes.append("goal")
                    content = "🚪"
                if policy and state in policy and state != agent_state:
                    content = ACTION_ARROWS[policy[state]]
                if state == agent_state:
                    classes.append("agent")
                    content = "🐕"
            label = f'<span class="cell-main">{escape(content)}</span>'
            label += reward_label
            label += f'<span class="cell-coordinate">{x},{y}</span>'
            cells.append(
                f'<div class="{" ".join(classes)}" style="{style}" '
                f'title="{escape(" | ".join(title_parts))}">{label}</div>'
            )

    return f"""
    <style>
      .room-grid {{
        direction: ltr;
        display: grid;
        grid-template-columns: repeat(10, minmax(42px, 1fr));
        max-width: 720px;
        border: 3px solid #263238;
        border-radius: 8px;
        overflow: hidden;
        background: #f7f9fb;
      }}
      .room-cell {{
        position: relative;
        aspect-ratio: 1;
        border: 1px solid #aeb8c2;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: clamp(18px, 2.2vw, 30px);
        box-sizing: border-box;
      }}
      .room-cell.wall {{ background: #455a64 !important; }}
      .room-cell.slippery {{ background: #dff6ff !important; }}
      .room-cell.start {{ outline: 3px solid #43a047; outline-offset: -3px; }}
      .room-cell.goal {{ outline: 3px solid #f9a825; outline-offset: -3px; }}
      .room-cell.termination {{ outline: 3px solid #e11d48; outline-offset: -3px; }}
      .room-cell.agent {{ background: #fff3cd !important; }}
      .cell-main {{ line-height: 1; }}
      .cell-reward {{
        position: absolute;
        top: 4px;
        left: 4px;
        padding: 2px 4px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.88);
        font-size: clamp(10px, 1.2vw, 15px);
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.02em;
      }}
      .cell-main:empty + .cell-reward {{
        position: static;
        padding: 0;
        background: transparent;
        font-size: clamp(14px, 1.7vw, 21px);
      }}
      .reward-positive {{ color: #15803d; }}
      .reward-negative {{ color: #dc2626; }}
      .reward-neutral {{ color: #475569; }}
      .cell-coordinate {{
        position: absolute;
        bottom: 1px;
        right: 3px;
        font-size: 9px;
        color: #263238;
        opacity: 0.75;
      }}
      .wall .cell-coordinate {{ color: white; }}
    </style>
    <div class="room-grid">{''.join(cells)}</div>
    """
