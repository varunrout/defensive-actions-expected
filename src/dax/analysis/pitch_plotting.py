"""mplsoccer-based pitch visualisations for analysis outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mplsoccer import Pitch

from .plot_style import DEFAULT_THEME, add_note


def _pitch() -> Pitch:
    return Pitch(pitch_type="statsbomb", pitch_color=DEFAULT_THEME.background, line_color="#555555", linewidth=1)


def _save(fig: plt.Figure, output_path: str | Path, dpi: int = 180) -> plt.Figure:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_pitch_scatter(df: pd.DataFrame, output_path: str | Path, *, title: str, subtitle: str = "StatsBomb coordinates; attacking direction left to right", x_col: str = "action_x", y_col: str = "action_y", dpi: int = 180) -> plt.Figure:
    """Plot action locations on a StatsBomb pitch without mutating input data."""
    data = df.copy(deep=True)
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=(10, 6))
    if {x_col, y_col}.issubset(data.columns) and not data.empty:
        pitch.scatter(data[x_col], data[y_col], ax=ax, s=18, alpha=0.55, c="#376795", edgecolors="none")
        ax.scatter(data[x_col].mean(), data[y_col].mean(), s=80, c="#d62728", marker="x", label="Mean location")
        ax.legend(loc="upper right", frameon=False)
    ax.set_title(title, loc="left", fontsize=DEFAULT_THEME.title_size, fontweight="bold")
    ax.text(0, 82, subtitle, fontsize=DEFAULT_THEME.subtitle_size, color="#555555")
    add_note(ax, f"n={len(data):,}. Defensive goal x=0; attacking goal x=120.")
    return _save(fig, output_path, dpi)


def plot_pitch_density(df: pd.DataFrame, output_path: str | Path, *, title: str, bins: tuple[int, int] = (12, 8), x_col: str = "action_x", y_col: str = "action_y", dpi: int = 180) -> plt.Figure:
    """Plot binned action density on an mplsoccer StatsBomb pitch."""
    data = df.copy(deep=True)
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=(10, 6))
    if {x_col, y_col}.issubset(data.columns) and not data.empty:
        stat = pitch.bin_statistic(data[x_col], data[y_col], statistic="count", bins=bins)
        pcm = pitch.heatmap(stat, ax=ax, cmap="Blues", edgecolors="#f7f7f7")
        fig.colorbar(pcm, ax=ax, label="Actions")
    ax.set_title(title, loc="left", fontsize=DEFAULT_THEME.title_size, fontweight="bold")
    add_note(ax, f"Denominator: {len(data):,} actions. Bins: {bins[0]}x{bins[1]}; not tactical ground truth.")
    return _save(fig, output_path, dpi)


def plot_pitch_rate_map(df: pd.DataFrame, output_path: str | Path, *, value_col: str, title: str, bins: tuple[int, int] = (12, 8), min_bin_actions: int = 20, x_col: str = "action_x", y_col: str = "action_y", dpi: int = 180) -> plt.Figure:
    """Plot a binned rate/mean map with low-sample bins masked."""
    data = df.copy(deep=True)
    pitch = _pitch()
    fig, ax = pitch.draw(figsize=(10, 6))
    if {x_col, y_col, value_col}.issubset(data.columns) and not data.empty:
        counts = pitch.bin_statistic(data[x_col], data[y_col], statistic="count", bins=bins)
        means = pitch.bin_statistic(data[x_col], data[y_col], values=data[value_col], statistic="mean", bins=bins)
        means["statistic"] = np.where(counts["statistic"] >= min_bin_actions, means["statistic"], np.nan)
        pcm = pitch.heatmap(means, ax=ax, cmap="RdYlBu_r", edgecolors="#f7f7f7")
        fig.colorbar(pcm, ax=ax, label=value_col.replace("_", " "))
    ax.set_title(title, loc="left", fontsize=DEFAULT_THEME.title_size, fontweight="bold")
    add_note(ax, f"Cells with fewer than {min_bin_actions} actions are masked.")
    return _save(fig, output_path, dpi)


def plot_player_spatial_profile(df: pd.DataFrame, output_path: str | Path, *, title: str, dpi: int = 180) -> plt.Figure:
    return plot_pitch_scatter(df, output_path, title=title, dpi=dpi)


def plot_cluster_spatial_profile(df: pd.DataFrame, output_path: str | Path, *, title: str, dpi: int = 180) -> plt.Figure:
    return plot_pitch_density(df, output_path, title=title, dpi=dpi)
