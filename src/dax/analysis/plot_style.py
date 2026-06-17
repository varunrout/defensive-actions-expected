"""Presentation styling utilities for DAx analysis charts."""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

import matplotlib.pyplot as plt

DISPLAY_LABELS = {
    "phase_settled_low_block_proxy_share": "Settled low block",
    "phase_transition_defence_share": "Transition defence",
    "phase_transition_defence_count": "Transition defence actions",
    "action_family_intervention_share": "Interventions",
    "action_family_pressure_share": "Pressures",
    "action_family_recovery_share": "Recoveries",
    "action_family_contest_share": "Contests",
    "mean_local_numerical_balance_10m": "Local numerical balance, 10m",
    "mean_local_numerical_balance_5m": "Local numerical balance, 5m",
    "box_defence_share": "Defensive-box exposure",
    "reliable_visibility_share": "Reliable 360 visibility",
    "possession_win_rate": "Possession-win rate",
    "opponent_possession_end_rate": "Opponent possessions ended",
    "retained_control_rate": "Retained control",
    "actions_under_opponent_possession_rate": "Opponent-possession actions",
    "future_shot_rate": "Future-shot rate",
    "future_xg_mean": "Mean future xG",
    "total_actions": "Actions",
    "actions_per_match": "Actions per match",
    "mean_action_x": "Mean action height",
    "mean_action_y": "Mean action width",
}
CLUSTER_COLOURS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd", 5: "#8c564b"}


@dataclass(frozen=True)
class AnalysisTheme:
    figure_size: tuple[float, float] = (8.5, 5.0)
    wide_figure_size: tuple[float, float] = (11.0, 6.0)
    title_size: int = 14
    subtitle_size: int = 10
    label_size: int = 10
    tick_size: int = 9
    note_size: int = 8
    dpi: int = 180
    background: str = "#fbfbf8"
    axes_background: str = "#ffffff"
    grid_colour: str = "#e6e6e6"
    text_colour: str = "#222222"
    bar_colour: str = "#376795"


DEFAULT_THEME = AnalysisTheme()


def display_label(name: object, *, width: int = 24) -> str:
    """Return a human-readable chart label for an internal feature name."""
    text = DISPLAY_LABELS.get(str(name), str(name).replace("_share", "").replace("_", " ").title())
    return "\n".join(textwrap.wrap(text, width=width)) if len(text) > width else text


def apply_theme(ax: plt.Axes, *, title: str, xlabel: str | None = None, ylabel: str | None = None, subtitle: str | None = None) -> None:
    """Apply DAx analysis styling to one axes without changing global Matplotlib state."""
    theme = DEFAULT_THEME
    ax.figure.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.axes_background)
    ax.set_title(title, fontsize=theme.title_size, color=theme.text_colour, loc="left", pad=14, fontweight="bold")
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=theme.subtitle_size, color="#555555", va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=theme.label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=theme.label_size)
    ax.tick_params(labelsize=theme.tick_size)
    ax.grid(True, axis="y", color=theme.grid_colour, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def add_note(ax: plt.Axes, note: str | None) -> None:
    """Add a source/denominator note beneath a chart."""
    if note:
        ax.text(0, -0.18, note, transform=ax.transAxes, fontsize=DEFAULT_THEME.note_size, color="#555555", va="top")
