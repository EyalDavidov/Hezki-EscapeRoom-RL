"""Shared formatting helpers for values displayed in the client grid."""

from __future__ import annotations


def format_reward_label(value: float) -> str:
    """Format a cell reward as a compact signed value for the grid."""
    if value == 0:
        return "0"
    magnitude = f"{abs(value):.3f}".rstrip("0").rstrip(".")
    if magnitude == "0":
        magnitude = f"{abs(value):.3g}"
    return f"{'+' if value > 0 else '-'}{magnitude}"


def reward_sign_class(value: float) -> str:
    if value > 0:
        return "reward-positive"
    if value < 0:
        return "reward-negative"
    return "reward-neutral"
