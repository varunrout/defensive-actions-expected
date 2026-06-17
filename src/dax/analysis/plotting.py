"""Reusable Matplotlib chart functions for pre-modelling analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_style import CLUSTER_COLOURS, DEFAULT_THEME, add_note, apply_theme, display_label


def _save(fig: plt.Figure, output_path: str | Path, *, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if save_svg:
        fig.savefig(path.with_suffix(".svg"), dpi=dpi)
    plt.close(fig)
    return fig


def _prepared_categories(df: pd.DataFrame, x: str, y: str, *, top_n: int = 25, keep: list[str] | None = None) -> pd.DataFrame:
    columns = [x, y] + [column for column in (keep or []) if column in df.columns and column not in {x, y}]
    if df.empty or not {x, y}.issubset(df.columns):
        return pd.DataFrame(columns=columns)
    values = df[columns].copy().head(top_n)
    values[x] = values[x].map(display_label)
    return values


def _cluster_colours(values: pd.Series | None) -> list[str] | str:
    if values is None:
        return DEFAULT_THEME.bar_colour
    colours: list[str] = []
    for value in values:
        try:
            colours.append(CLUSTER_COLOURS.get(int(value), DEFAULT_THEME.bar_colour))
        except (TypeError, ValueError):
            colours.append(DEFAULT_THEME.bar_colour)
    return colours


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: str | Path,
    title: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    note: str | None = None,
    dpi: int = 150,
    save_svg: bool = False,
    top_n: int = 25,
    force_vertical: bool = False,
    color: str | None = None,
) -> plt.Figure:
    """Save an intelligently oriented bar chart without default 45-degree tick rotation."""
    values = _prepared_categories(df, x, y, top_n=top_n, keep=[color] if color else None)
    bar_colours = _cluster_colours(values[color]) if color and color in values.columns else DEFAULT_THEME.bar_colour
    max_label_length = int(values[x].astype(str).map(len).max()) if not values.empty else 0
    horizontal = (not force_vertical) and (len(values) > 8 or max_label_length > 18)
    figsize = (DEFAULT_THEME.wide_figure_size[0], max(4.5, len(values) * 0.35)) if horizontal else DEFAULT_THEME.figure_size
    fig, ax = plt.subplots(figsize=figsize)
    if not values.empty:
        if horizontal:
            ax.barh(values[x], values[y], color=bar_colours)
            ax.invert_yaxis()
        else:
            ax.bar(values[x], values[y], color=bar_colours)
            rotation = 90 if len(values) > 14 else 0
            ax.tick_params(axis="x", rotation=rotation)
    apply_theme(ax, title=title, xlabel=xlabel or display_label(x), ylabel=ylabel or display_label(y))
    add_note(ax, note)
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def histogram(df: pd.DataFrame, column: str, output_path: str | Path, title: str, *, bins: int = 20, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=DEFAULT_THEME.figure_size)
    if column in df.columns:
        ax.hist(pd.to_numeric(df[column], errors="coerce").dropna(), bins=bins, color=DEFAULT_THEME.bar_colour)
    apply_theme(ax, title=title, xlabel=display_label(column), ylabel="Rows")
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def scatter_chart(df: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str, *, color: str | None = None, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=DEFAULT_THEME.figure_size)
    if not df.empty and {x, y}.issubset(df.columns):
        if color and color in df.columns:
            colours = df[color].map(lambda value: CLUSTER_COLOURS.get(int(value), "#777777") if pd.notna(value) else "#777777")
        else:
            colours = None
        ax.scatter(df[x], df[y], c=colours, alpha=0.72)
    apply_theme(ax, title=title, xlabel=display_label(x), ylabel=display_label(y))
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def labelled_heatmap(matrix: pd.DataFrame, output_path: str | Path, title: str, *, xlabel: str = "Columns", ylabel: str = "Rows", dpi: int = 150, save_svg: bool = False, center_zero: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(7, min(16, len(matrix.columns) * 0.45)), max(5, len(matrix) * 0.3)))
    values = matrix.fillna(0).to_numpy() if not matrix.empty else np.zeros((1, 1))
    vlim = np.nanmax(np.abs(values)) if center_zero and values.size else None
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r" if center_zero else "viridis", vmin=-vlim if vlim else None, vmax=vlim if vlim else None)
    apply_theme(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    if not matrix.empty:
        ax.set_xticks(range(len(matrix.columns)), labels=[display_label(column, width=18) for column in matrix.columns], rotation=90)
        ax.set_yticks(range(len(matrix.index)), labels=[display_label(index, width=22) for index in matrix.index])
    fig.colorbar(image, ax=ax, label="Standardised value" if center_zero else "Value")
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def pitch_grid_heatmap(*args, **kwargs):
    """Technical diagnostic fallback; football outputs should use pitch_plotting."""
    return labelled_heatmap(pd.DataFrame(), args[2] if len(args) > 2 else kwargs["output_path"], args[3] if len(args) > 3 else kwargs.get("title", "Pitch grid"))


def target_rate_by_bins(df: pd.DataFrame, feature: str, target: str, output_path: str | Path, *, bins: int = 10, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    data = df[[feature, target]].copy() if {feature, target}.issubset(df.columns) else pd.DataFrame(columns=[feature, target])
    if not data.empty:
        data["bin"] = pd.qcut(pd.to_numeric(data[feature], errors="coerce"), q=min(bins, data[feature].nunique()), duplicates="drop")
        plot = data.groupby("bin", observed=False)[target].mean().reset_index()
        plot["bin"] = plot["bin"].astype(str)
    else:
        plot = pd.DataFrame({"bin": [], target: []})
    return bar_chart(plot, "bin", target, output_path, f"{display_label(target)} by {display_label(feature)} bins", xlabel=display_label(feature), ylabel=f"Mean {display_label(target)}", dpi=dpi, save_svg=save_svg)


def save_bar(*args, **kwargs):
    return bar_chart(*args, **kwargs)


def save_heatmap(*args, **kwargs):
    return labelled_heatmap(*args, **kwargs)


def save_scatter(*args, **kwargs):
    return scatter_chart(*args, **kwargs)
