"""HTML rendering helpers for the Streamlit client."""

from __future__ import annotations

from html import escape
from typing import Any
from .display_formatting import format_reward_label, reward_sign_class
from .room1 import Action, State
from .room4 import State4
from .room5 import RoadSnapshot

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


def render_room5_html(
    environment: Any,
    snapshot: RoadSnapshot | None = None,
) -> str:
    """Render Room 5 as a top-down one-way road with moving traffic."""
    snapshot = snapshot or environment.snapshot()
    svg_width = 720
    svg_height = 600
    lane_width = 78
    road_width = environment.config.lane_count * lane_width
    road_left = (svg_width - road_width) / 2
    road_right = road_left + road_width
    ego_y = 525.0
    horizon_y = 55.0
    visible_height = ego_y - horizon_y

    def lane_center(lane: int) -> float:
        return road_left + (lane + 0.5) * lane_width

    def distance_y(distance: float) -> float:
        ratio = max(-0.04, min(1.0, distance / environment.config.vision_distance))
        return ego_y - ratio * visible_height

    def car_svg(x: float, y: float, color: str, *, ego: bool = False) -> str:
        width = 34 if ego else 31
        height = 58 if ego else 52
        label = "AGENT" if ego else ""
        return (
            f'<g transform="translate({x - width / 2:.1f},{y - height / 2:.1f})">'
            f'<rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="{color}" stroke="#111827" stroke-width="2" />'
            f'<rect x="5" y="9" width="{width - 10}" height="13" rx="3" fill="#dbeafe" opacity="0.9" />'
            f'<rect x="5" y="{height - 22}" width="{width - 10}" height="12" rx="3" fill="#bfdbfe" opacity="0.75" />'
            f'<rect x="-3" y="10" width="4" height="12" rx="2" fill="#111827" />'
            f'<rect x="{width - 1}" y="10" width="4" height="12" rx="2" fill="#111827" />'
            f'<rect x="-3" y="{height - 22}" width="4" height="12" rx="2" fill="#111827" />'
            f'<rect x="{width - 1}" y="{height - 22}" width="4" height="12" rx="2" fill="#111827" />'
            + (
                f'<text x="{width / 2}" y="{height + 15}" text-anchor="middle" fill="#dbeafe" font-size="11" font-weight="700">{label}</text>'
                if ego
                else ""
            )
            + "</g>"
        )

    elements: list[str] = [
        f'<rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="#166534" />',
        f'<rect x="{road_left}" y="0" width="{road_width}" height="{svg_height}" fill="#303640" stroke="#f8fafc" stroke-width="5" />',
        f'<rect x="{road_left + 8}" y="0" width="4" height="{svg_height}" fill="#facc15" />',
        f'<rect x="{road_right - 12}" y="0" width="4" height="{svg_height}" fill="#facc15" />',
    ]

    for lane in range(1, environment.config.lane_count):
        x = road_left + lane * lane_width
        elements.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{svg_height}" stroke="#f8fafc" stroke-width="3" stroke-dasharray="18,18" opacity="0.8" />'
        )

    ego_x = lane_center(snapshot.ego_lane)
    view_left = road_left + snapshot.ego_lane * lane_width + 6
    view_right = view_left + lane_width - 12
    view_near_y = ego_y - 34
    elements.extend(
        [
            f'<polygon points="{ego_x - 14:.1f},{view_near_y:.1f} {ego_x + 14:.1f},{view_near_y:.1f} {view_right:.1f},{horizon_y:.1f} {view_left:.1f},{horizon_y:.1f}" fill="#38bdf8" opacity="0.18" stroke="#7dd3fc" stroke-width="2" stroke-dasharray="7,5" />',
            f'<line x1="{view_left:.1f}" y1="{horizon_y:.1f}" x2="{view_right:.1f}" y2="{horizon_y:.1f}" stroke="#38bdf8" stroke-width="4" />',
            f'<text x="{ego_x:.1f}" y="{horizon_y - 12:.1f}" text-anchor="middle" fill="#e0f2fe" font-size="12" font-weight="800">CURRENT-LANE VIEW • {environment.config.vision_distance:.0f}m</text>',
        ]
    )

    elements.extend(
        [
            f'<text x="18" y="32" fill="#dcfce7" font-size="16" font-weight="800">ONE-WAY ROAD</text>',
            f'<text x="18" y="55" fill="#bbf7d0" font-size="13">Progress: {snapshot.progress:.1f}/{environment.config.road_length:.0f}m</text>',
            f'<path d="M {svg_width - 46} 74 L {svg_width - 46} 24 M {svg_width - 58} 38 L {svg_width - 46} 24 L {svg_width - 34} 38" fill="none" stroke="#f8fafc" stroke-width="4" />',
        ]
    )

    traffic_colors = ("#ef4444", "#f97316", "#a855f7", "#eab308")
    visible_traffic = [
        car
        for car in snapshot.traffic
        if -environment.config.car_length
        <= car.distance
        <= environment.config.vision_distance + environment.config.car_length
    ]
    closest_ahead = min(
        (
            car
            for car in visible_traffic
            if environment.forward_clearance(car) >= 0.0
            and car.lane == snapshot.ego_lane
        ),
        key=environment.forward_clearance,
        default=None,
    )
    for car in sorted(visible_traffic, key=lambda item: item.distance, reverse=True):
        car_x = lane_center(car.lane)
        car_y = distance_y(car.distance)
        elements.append(
            car_svg(
                car_x,
                car_y,
                traffic_colors[car.car_id % len(traffic_colors)],
            )
        )
        if closest_ahead is not None and car.car_id == closest_ahead.car_id:
            label_x = car_x + (42 if car_x + 78 < road_right else -42)
            label_y = max(horizon_y + 18, car_y - 31)
            distance_label = f"{environment.forward_clearance(car):.1f} m"
            elements.extend(
                [
                    f'<rect x="{label_x - 30:.1f}" y="{label_y - 15:.1f}" width="60" height="23" rx="7" fill="#0f172a" stroke="#38bdf8" stroke-width="1.5" opacity="0.94" />',
                    f'<text x="{label_x:.1f}" y="{label_y + 1:.1f}" text-anchor="middle" fill="#e0f2fe" font-size="12" font-weight="800" aria-label="Nearest car distance {distance_label}">{distance_label}</text>',
                ]
            )

    elements.append(car_svg(lane_center(snapshot.ego_lane), ego_y, "#2563eb", ego=True))

    return f"""
    <div style="display:flex;justify-content:center;width:100%;">
      <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" role="img" aria-label="Room 5 one-way road with {environment.config.lane_count} lanes" style="border:3px solid #111827;border-radius:12px;background:#166534;box-shadow:0 6px 22px rgba(0,0,0,.3);">
        {''.join(elements)}
      </svg>
    </div>
    """
