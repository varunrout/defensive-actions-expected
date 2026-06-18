"""mplsoccer-based pitch visualisations for analysis outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import textwrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mplsoccer import Pitch, VerticalPitch

from .plot_style import CLUSTER_COLOURS, DEFAULT_THEME

DEFAULT_PITCH_CONFIG: dict[str, Any] = {
    "pitch_type": "statsbomb",
    "orientation": "horizontal",
    "density_bins": [12, 8],
    "coarse_bins": [6, 4],
    "positional_bins": True,
    "kde_levels": 20,
    "pitch_color": DEFAULT_THEME.background,
    "line_color": "#555555",
    "density_cmap": "Blues",
    "rate_cmap": "RdYlBu_r",
}


def pitch_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return merged pitch-visualisation configuration."""
    merged = DEFAULT_PITCH_CONFIG.copy()
    if config:
        merged.update(config.get("pitch_visualisation", config))
    return merged


def _pitch(config: dict[str, Any]) -> Pitch | VerticalPitch:
    klass = VerticalPitch if config.get("orientation") == "vertical" else Pitch
    return klass(
        pitch_type=config.get("pitch_type", "statsbomb"),
        pitch_color=config.get("pitch_color", DEFAULT_THEME.background),
        line_color=config.get("line_color", "#555555"),
        linewidth=1,
    )


def _draw(config: dict[str, Any]) -> tuple[plt.Figure, plt.Axes, Pitch | VerticalPitch]:
    pitch = _pitch(config)
    fig, ax = pitch.draw(figsize=(10.8, 6.8), constrained_layout=False)
    fig.patch.set_facecolor(config.get("pitch_color", DEFAULT_THEME.background))
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.05, right=0.90)
    return fig, ax, pitch


def _save(fig: plt.Figure, output_path: str | Path, dpi: int) -> plt.Figure:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def _compose_pitch_chart(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str | None, footer: str | None) -> None:
    """Apply a non-overlapping title/subtitle/footer layout to a pitch chart."""
    ax.set_title("")
    fig.text(0.06, 0.965, "\n".join(textwrap.wrap(title, width=76)), fontsize=DEFAULT_THEME.title_size, fontweight="bold", color=DEFAULT_THEME.text_colour, ha="left", va="top")
    if subtitle:
        fig.text(0.06, 0.895, "\n".join(textwrap.wrap(subtitle, width=94)), fontsize=DEFAULT_THEME.subtitle_size, color="#555555", ha="left", va="top")
    if footer:
        fig.text(0.06, 0.045, "\n".join(textwrap.wrap(footer, width=120)), fontsize=DEFAULT_THEME.note_size, color="#555555", ha="left", va="bottom")


def _add_direction_note(ax: plt.Axes) -> None:
    ax.text(0.98, -0.06, "Attacking direction →", transform=ax.transAxes, ha="right", va="top", color="#555555", fontsize=8)


def plot_pitch_scatter(df: pd.DataFrame, output_path: str | Path, *, title: str, subtitle: str = "StatsBomb coordinates; attacking direction left to right", x_col: str = "action_x", y_col: str = "action_y", config: dict[str, Any] | None = None, dpi: int | None = None, cluster_col: str | None = None) -> plt.Figure:
    """Plot action locations on a StatsBomb pitch without mutating input data."""
    cfg = pitch_config(config)
    data = df.copy(deep=True)
    fig, ax, pitch = _draw(cfg)
    if {x_col, y_col}.issubset(data.columns) and not data.empty:
        if cluster_col and cluster_col in data.columns:
            for cluster, part in data.groupby(cluster_col):
                pitch.scatter(part[x_col], part[y_col], ax=ax, s=18, alpha=0.55, c=CLUSTER_COLOURS.get(int(cluster), "#777777"), edgecolors="none", label=f"Cluster {cluster}")
            ax.legend(loc="upper right", frameon=False)
        else:
            pitch.scatter(data[x_col], data[y_col], ax=ax, s=18, alpha=0.55, c="#376795", edgecolors="none")
        ax.scatter(data[x_col].mean(), data[y_col].mean(), s=80, c="#d62728", marker="x", label="Mean location")
    _add_direction_note(ax)
    _compose_pitch_chart(fig, ax, title, subtitle, f"n={len(data):,}. Defensive goal x=0; attacking goal x=120.")
    return _save(fig, output_path, dpi or int(cfg.get("chart_dpi", DEFAULT_THEME.dpi)))


def plot_pitch_density(df: pd.DataFrame, output_path: str | Path, *, title: str, bins: tuple[int, int] | None = None, x_col: str = "action_x", y_col: str = "action_y", config: dict[str, Any] | None = None, dpi: int | None = None, subtitle: str | None = None) -> plt.Figure:
    """Plot binned action density on an mplsoccer StatsBomb pitch."""
    cfg = pitch_config(config)
    data = df.copy(deep=True)
    use_bins = bins or tuple(cfg.get("density_bins", [12, 8]))
    fig, ax, pitch = _draw(cfg)
    if {x_col, y_col}.issubset(data.columns) and not data.empty:
        stat = pitch.bin_statistic(data[x_col], data[y_col], statistic="count", bins=use_bins)
        pcm = pitch.heatmap(stat, ax=ax, cmap=cfg.get("density_cmap", "Blues"), edgecolors="#f7f7f7")
        fig.colorbar(pcm, ax=ax, label="Actions", shrink=0.72, pad=0.02)
    _add_direction_note(ax)
    _compose_pitch_chart(fig, ax, title, subtitle or "Style distribution, not a performance map", f"Denominator: {len(data):,} actions. Bins: {use_bins[0]}x{use_bins[1]}; not tactical ground truth.")
    return _save(fig, output_path, dpi or int(cfg.get("chart_dpi", DEFAULT_THEME.dpi)))


def plot_pitch_rate_map(df: pd.DataFrame, output_path: str | Path, *, value_col: str, title: str, bins: tuple[int, int] | None = None, min_bin_actions: int = 20, x_col: str = "action_x", y_col: str = "action_y", config: dict[str, Any] | None = None, dpi: int | None = None) -> plt.Figure:
    """Plot a binned rate/mean map with low-sample bins masked."""
    cfg = pitch_config(config)
    data = df.copy(deep=True)
    use_bins = bins or tuple(cfg.get("density_bins", [12, 8]))
    fig, ax, pitch = _draw(cfg)
    if {x_col, y_col, value_col}.issubset(data.columns) and not data.empty:
        counts = pitch.bin_statistic(data[x_col], data[y_col], statistic="count", bins=use_bins)
        means = pitch.bin_statistic(data[x_col], data[y_col], values=data[value_col], statistic="mean", bins=use_bins)
        means["statistic"] = np.where(counts["statistic"] >= min_bin_actions, means["statistic"], np.nan)
        pcm = pitch.heatmap(means, ax=ax, cmap=cfg.get("rate_cmap", "RdYlBu_r"), edgecolors="#f7f7f7")
        fig.colorbar(pcm, ax=ax, label=value_col.replace("_", " "), shrink=0.72, pad=0.02)
    _add_direction_note(ax)
    _compose_pitch_chart(fig, ax, title, "Low-sample bins are masked", f"Cells with fewer than {min_bin_actions} actions are masked. Denominator: actions in each bin.")
    return _save(fig, output_path, dpi or int(cfg.get("chart_dpi", DEFAULT_THEME.dpi)))


def plot_cluster_population_difference(cluster_df: pd.DataFrame, population_df: pd.DataFrame, output_path: str | Path, *, title: str, config: dict[str, Any] | None = None) -> plt.Figure:
    """Plot cluster density minus population density as a relative style profile."""
    cfg = pitch_config(config)
    bins = tuple(cfg.get("density_bins", [12, 8]))
    fig, ax, pitch = _draw(cfg)
    if not cluster_df.empty and not population_df.empty:
        c = pitch.bin_statistic(cluster_df["action_x"], cluster_df["action_y"], statistic="count", bins=bins)
        p = pitch.bin_statistic(population_df["action_x"], population_df["action_y"], statistic="count", bins=bins)
        c_stat = c["statistic"] / max(1, np.nansum(c["statistic"]))
        p_stat = p["statistic"] / max(1, np.nansum(p["statistic"]))
        c["statistic"] = c_stat - p_stat
        vmax = float(np.nanmax(np.abs(c["statistic"]))) if np.isfinite(c["statistic"]).any() else None
        pcm = pitch.heatmap(c, ax=ax, cmap="RdBu_r", edgecolors="#f7f7f7", vmin=-vmax if vmax else None, vmax=vmax if vmax else None)
        fig.colorbar(pcm, ax=ax, label="Cluster density minus population density", shrink=0.72, pad=0.02)
    _add_direction_note(ax)
    _compose_pitch_chart(fig, ax, title, "Relative spatial style profile, not a performance map", "Positive cells indicate locations where the cluster acts more often than the population share; this is not a performance map.")
    return _save(fig, output_path, int(cfg.get("chart_dpi", DEFAULT_THEME.dpi)))


def plot_player_spatial_profile(df: pd.DataFrame, output_path: str | Path, *, title: str, config: dict[str, Any] | None = None, dpi: int | None = None) -> plt.Figure:
    return plot_pitch_scatter(df, output_path, title=title, config=config, dpi=dpi)


def plot_cluster_spatial_profile(df: pd.DataFrame, output_path: str | Path, *, title: str, config: dict[str, Any] | None = None, dpi: int | None = None) -> plt.Figure:
    return plot_pitch_density(df, output_path, title=title, config=config, dpi=dpi)
