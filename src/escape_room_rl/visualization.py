"""HTML rendering helpers for the Streamlit client."""

from __future__ import annotations

from html import escape
from typing import Any
from .display_formatting import format_reward_label, reward_sign_class
from .room1 import Action, State
from .room4 import State4

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


def render_room4_html(
    environment: Any,
    agent_state: State4 | None = None,
    trajectory: list[State4] | None = None,
) -> str:
    """Render a styled Flappy Bird 2D continuous 10x10m SVG visualization."""
    svg_width = 720
    svg_height = 540
    scale_x = svg_width / environment.config.width
    scale_y = svg_height / environment.config.height

    def to_svg_x(x: float) -> float:
        return x * scale_x

    def to_svg_y(y: float) -> float:
        # Invert Y so 0 is at the bottom and 10 is at the top
        return (environment.config.height - y) * scale_y

    elements: list[str] = []

    # 1. Background sky & grass floor
    elements.append(
        f'<rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="#70c5ce" />'
    )
    elements.append(
        f'<rect x="0" y="{svg_height - 20}" width="{svg_width}" height="20" fill="#ded895" stroke="#73be2e" stroke-width="4" />'
    )

    # 2. Goal Zone (x >= goal_x)
    goal_svg_x = to_svg_x(environment.config.goal_x)
    elements.append(
        f'<rect x="{goal_svg_x}" y="0" width="{svg_width - goal_svg_x}" height="{svg_height}" fill="rgba(245, 158, 11, 0.3)" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6,4" />'
    )
    elements.append(
        f'<text x="{goal_svg_x + 10}" y="30" fill="#92400e" font-size="18" font-weight="bold">🚪 GOAL</text>'
    )

    # 3. Flappy Bird Green Pipes
    for pipe in environment.config.pipes:
        px_min = to_svg_x(pipe.x_min)
        pipe_w = pipe.width * scale_x
        top_h = to_svg_y(pipe.gap_end)
        bot_y = to_svg_y(pipe.gap_start)
        bot_h = svg_height - bot_y

        # Top Pipe Body
        elements.append(
            f'<rect x="{px_min}" y="0" width="{pipe_w}" height="{top_h}" fill="#73bf2e" stroke="#538021" stroke-width="3" rx="4" />'
        )
        # Top Pipe Cap
        elements.append(
            f'<rect x="{px_min - 4}" y="{top_h - 18}" width="{pipe_w + 8}" height="18" fill="#73bf2e" stroke="#538021" stroke-width="3" rx="3" />'
        )

        # Bottom Pipe Body
        elements.append(
            f'<rect x="{px_min}" y="{bot_y}" width="{pipe_w}" height="{bot_h}" fill="#73bf2e" stroke="#538021" stroke-width="3" rx="4" />'
        )
        # Bottom Pipe Cap
        elements.append(
            f'<rect x="{px_min - 4}" y="{bot_y}" width="{pipe_w + 8}" height="18" fill="#73bf2e" stroke="#538021" stroke-width="3" rx="3" />'
        )

    # 4. Trajectory Path
    if trajectory and len(trajectory) > 1:
        points = " ".join([f"{to_svg_x(st[0]):.1f},{to_svg_y(st[1]):.1f}" for st in trajectory])
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="#ef4444" stroke-width="3" stroke-dasharray="4,2" />'
        )

    # 5. Agent (Flappy Bird)
    curr_state = agent_state or environment.config.start
    bx = to_svg_x(curr_state[0])
    by = to_svg_y(curr_state[1])
    bird_r = environment.config.bird_radius * scale_x

    # Bird Outer Circle
    elements.append(
        f'<circle cx="{bx}" cy="{by}" r="{bird_r + 4}" fill="#facc15" stroke="#ca8a04" stroke-width="3" />'
    )
    # Bird Eye & Beak
    elements.append(
        f'<circle cx="{bx + bird_r * 0.4}" cy="{by - bird_r * 0.3}" r="4" fill="white" stroke="black" stroke-width="1" />'
    )
    elements.append(
        f'<circle cx="{bx + bird_r * 0.5}" cy="{by - bird_r * 0.3}" r="1.5" fill="black" />'
    )
    elements.append(
        f'<polygon points="{bx + bird_r * 0.6},{by} {bx + bird_r * 1.2},{by + 3} {bx + bird_r * 0.6},{by + 6}" fill="#f97316" />'
    )

    # 6. Start Marker
    sx = to_svg_x(environment.config.start[0])
    sy = to_svg_y(environment.config.start[1])
    elements.append(
        f'<circle cx="{sx}" cy="{sy}" r="8" fill="#22c55e" stroke="white" stroke-width="2" />'
    )
    elements.append(
        f'<text x="{sx - 6}" y="{sy + 4}" fill="white" font-size="10" font-weight="bold">S</text>'
    )

    return f"""
    <div style="display: flex; justify-content: center; width: 100%;">
      <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" style="border: 3px solid #263238; border-radius: 10px; background: #70c5ce; box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
        {''.join(elements)}
      </svg>
    </div>
    """
