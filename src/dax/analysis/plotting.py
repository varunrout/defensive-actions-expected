"""Reusable Matplotlib chart functions for pre-modelling analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, output_path: str | Path, *, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    if save_svg:
        fig.savefig(path.with_suffix(".svg"), dpi=dpi)
    plt.close(fig)
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str, *, xlabel: str | None = None, ylabel: str | None = None, note: str | None = None, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    """Save a labelled bar chart that handles empty data."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not df.empty and {x, y}.issubset(df.columns):
        values = df[[x, y]].head(40)
        ax.bar(values[x].astype(str), values[y])
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if note:
        ax.text(0.01, -0.28, note, transform=ax.transAxes, fontsize=8)
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def histogram(df: pd.DataFrame, column: str, output_path: str | Path, title: str, *, bins: int = 20, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    """Save a numeric histogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    if column in df.columns:
        ax.hist(pd.to_numeric(df[column], errors="coerce").dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Rows")
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def scatter_chart(df: pd.DataFrame, x: str, y: str, output_path: str | Path, title: str, *, color: str | None = None, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    """Save a labelled scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    if not df.empty and {x, y}.issubset(df.columns):
        colors = df[color] if color and color in df.columns else None
        ax.scatter(df[x], df[y], c=colors)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def labelled_heatmap(matrix: pd.DataFrame, output_path: str | Path, title: str, *, xlabel: str = "Columns", ylabel: str = "Rows", dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    """Save a heatmap with row and column labels."""
    fig, ax = plt.subplots(figsize=(max(7, len(matrix.columns) * 0.35), max(5, len(matrix) * 0.25)))
    values = matrix.fillna(0).to_numpy() if not matrix.empty else np.zeros((1, 1))
    image = ax.imshow(values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not matrix.empty:
        ax.set_xticks(range(len(matrix.columns)), labels=[str(column) for column in matrix.columns], rotation=90)
        ax.set_yticks(range(len(matrix.index)), labels=[str(index) for index in matrix.index])
    fig.colorbar(image, ax=ax)
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def pitch_grid_heatmap(zone_table: pd.DataFrame, value_column: str, output_path: str | Path, title: str, *, bins_x: int, bins_y: int, denominator_note: str | None = None, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    """Save a pitch-zone grid heatmap for density or rate summaries."""
    grid = np.full((bins_y, bins_x), np.nan)
    if not zone_table.empty and {"x_bin", "y_bin", value_column}.issubset(zone_table.columns):
        for row in zone_table.itertuples(index=False):
            x_bin = getattr(row, "x_bin")
            y_bin = getattr(row, "y_bin")
            if pd.notna(x_bin) and pd.notna(y_bin):
                grid[int(y_bin), int(x_bin)] = getattr(row, value_column)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(grid, origin="lower", extent=(0, 120, 0, 80), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Pitch length (0=defending goal, 120=attacking goal)")
    ax.set_ylabel("Pitch width")
    if denominator_note:
        ax.text(0.01, -0.12, denominator_note, transform=ax.transAxes, fontsize=8)
    fig.colorbar(image, ax=ax)
    return _save(fig, output_path, dpi=dpi, save_svg=save_svg)


def target_rate_by_bins(df: pd.DataFrame, feature: str, target: str, output_path: str | Path, *, bins: int = 10, dpi: int = 150, save_svg: bool = False) -> plt.Figure:
    """Save target mean by numeric feature quantile bins."""
    data = df[[feature, target]].copy() if {feature, target}.issubset(df.columns) else pd.DataFrame(columns=[feature, target])
    if not data.empty:
        data["bin"] = pd.qcut(pd.to_numeric(data[feature], errors="coerce"), q=min(bins, data[feature].nunique()), duplicates="drop")
        plot = data.groupby("bin", observed=False)[target].mean().reset_index()
        plot["bin"] = plot["bin"].astype(str)
    else:
        plot = pd.DataFrame({"bin": [], target: []})
    return bar_chart(plot, "bin", target, output_path, f"{target} by {feature} bins", xlabel=feature, ylabel=f"Mean {target}", dpi=dpi, save_svg=save_svg)


def save_bar(*args, **kwargs):
    """Backward-compatible wrapper for older tests."""
    return bar_chart(*args, **kwargs)


def save_heatmap(*args, **kwargs):
    """Backward-compatible wrapper for older callers."""
    return labelled_heatmap(*args, **kwargs)


def save_scatter(*args, **kwargs):
    """Backward-compatible wrapper for older callers."""
    return scatter_chart(*args, **kwargs)
