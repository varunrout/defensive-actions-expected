"""Common plotting helpers for player-feature analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from mplsoccer import Pitch
except Exception:  # pragma: no cover - optional graceful fallback
    Pitch = None

sns.set_theme(style="whitegrid")


def save_barh(df: pd.DataFrame, y: str, x: str, title: str, out_path: Path, top_n: int = 20) -> None:
    """Save a horizontal bar chart for top rows."""
    if df.empty or y not in df.columns or x not in df.columns:
        return
    plot_df = df.head(top_n).sort_values(x, ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[y], plot_df[x])
    plt.title(title)
    plt.xlabel(x)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, title: str, out_path: Path) -> None:
    """Save a heatmap from long-form data."""
    if df.empty or any(col not in df.columns for col in [index, columns, values]):
        return
    pivot = df.pivot(index=index, columns=columns, values=values)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_hist(series: pd.Series, title: str, out_path: Path) -> None:
    """Save a histogram for a numeric feature."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(s, bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_rate_ci_bar(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    out_path: Path,
    top_n: int = 12,
) -> None:
    """Plot shot rate bars with Wilson confidence interval whiskers."""
    required = {category_col, "shot_rate", "ci_low", "ci_high", "size"}
    if df.empty or not required.issubset(df.columns):
        return

    plot_df = df.sort_values("shot_rate", ascending=False).head(top_n).copy()
    for col in ["shot_rate", "ci_low", "ci_high", "size"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["shot_rate", "ci_low", "ci_high", "size"])
    if plot_df.empty:
        return

    # Guard against misordered or inconsistent bounds so matplotlib xerr stays non-negative.
    ci_min = np.minimum(plot_df["ci_low"].values, plot_df["ci_high"].values)
    ci_max = np.maximum(plot_df["ci_low"].values, plot_df["ci_high"].values)
    rate = plot_df["shot_rate"].values

    lower = np.minimum(rate, ci_min)
    upper = np.maximum(rate, ci_max)

    plot_df = plot_df.sort_values("shot_rate", ascending=True)
    err_low = np.clip(plot_df["shot_rate"].values - lower, a_min=0.0, a_max=None)
    err_high = np.clip(upper - plot_df["shot_rate"].values, a_min=0.0, a_max=None)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[category_col].astype(str), plot_df["shot_rate"], color="#4C78A8")
    plt.errorbar(
        plot_df["shot_rate"],
        plot_df[category_col].astype(str),
        xerr=np.vstack([err_low, err_high]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1,
    )
    for idx, row in plot_df.iterrows():
        plt.text(float(row["shot_rate"]) + 0.002, row[category_col], f"n={int(row['size'])}", va="center", fontsize=8)
    plt.title(title)
    plt.xlabel("Shot-in-10s rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_stability_errorbar(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 12) -> None:
    """Plot correlation stability with p10-p90 intervals."""
    required = {"feature", "corr_mean", "corr_p10", "corr_p90"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.sort_values("corr_mean", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    for col in ["corr_mean", "corr_p10", "corr_p90"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=["corr_mean", "corr_p10", "corr_p90"])
    if plot_df.empty:
        return

    # Keep whiskers valid even if percentile columns are not ordered in input.
    corr_min = np.minimum(plot_df["corr_p10"].values, plot_df["corr_p90"].values)
    corr_max = np.maximum(plot_df["corr_p10"].values, plot_df["corr_p90"].values)

    plot_df = plot_df.sort_values("corr_mean", ascending=True)
    low = np.clip(plot_df["corr_mean"].values - corr_min, a_min=0.0, a_max=None)
    high = np.clip(corr_max - plot_df["corr_mean"].values, a_min=0.0, a_max=None)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x=plot_df["corr_mean"],
        y=plot_df["feature"],
        xerr=np.vstack([low, high]),
        fmt="o",
        capsize=3,
    )
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Correlation with target (mean, p10-p90)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_stratified_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of stratified correlations by feature and group dimension."""
    required = {"group_col", "feature", "corr"}
    if df.empty or not required.issubset(df.columns):
        return
    agg = df.groupby(["group_col", "feature"], observed=True)["corr"].mean().reset_index()
    pivot = agg.pivot(index="feature", columns="group_col", values="corr")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Stratified feature-target correlation map")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_negative_control_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Compare observed feature signal vs negative controls."""
    required = {"observed_abs_corr", "perm_p95_abs_corr", "noise_abs_corr"}
    if df.empty or not required.issubset(df.columns):
        return
    vals = [
        float(df.iloc[0]["observed_abs_corr"]),
        float(df.iloc[0]["perm_p95_abs_corr"]),
        float(df.iloc[0]["noise_abs_corr"]),
    ]
    labels = ["Observed", "Permutation p95", "Random noise"]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals, color=["#4C78A8", "#F58518", "#E45756"])
    plt.title("Negative-control sanity check")
    plt.ylabel("Absolute correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_pitch_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    statistic: str = "count",
    normalize: bool = False,
    value_col: str | None = None,
) -> None:
    """Save a pitch heatmap using action coordinates."""
    if Pitch is None:
        return
    required = {"action_x", "action_y"}
    if df.empty or not required.issubset(df.columns):
        return
    work = df[["action_x", "action_y"] + ([value_col] if value_col else [])].copy()
    work["action_x"] = pd.to_numeric(work["action_x"], errors="coerce")
    work["action_y"] = pd.to_numeric(work["action_y"], errors="coerce")
    work = work.dropna(subset=["action_x", "action_y"])
    if work.empty:
        return

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 7))

    values = None
    if value_col is not None and value_col in work.columns:
        values = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)

    binned = pitch.bin_statistic(
        work["action_x"],
        work["action_y"],
        values=values,
        statistic=statistic,
        bins=(12, 8),
        normalize=normalize,
    )
    pitch.heatmap(binned, ax=ax, cmap="magma", edgecolors="#333333")
    pitch.label_heatmap(binned, color="white", fontsize=8, ax=ax, str_format="{:.2f}")
    ax.annotate(
        "Attacking direction ->",
        xy=(104, 83),
        xytext=(16, 83),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#222222"),
        ha="left",
        va="center",
        fontsize=9,
        color="#222222",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


