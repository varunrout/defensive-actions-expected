"""Plots for player-feature analysis module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_missingness(missing_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    plot_df = missing_df.head(top_n).sort_values("missing_pct", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["missing_pct"])
    plt.title("Player feature missingness (%)")
    plt.xlabel("Missing %")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_phase_position_heatmap(phase_position: pd.DataFrame, out_path: Path) -> None:
    if phase_position.empty:
        return
    pivot = phase_position.pivot(index="phase_label", columns="position_group", values="shot_rate")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Shot-in-10s rate by phase and position group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_numeric_signal(signal_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    if signal_df.empty:
        return
    plot_df = signal_df.head(top_n).copy().sort_values("corr_with_target")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["corr_with_target"])
    plt.title("Top numeric feature correlations with target_future_shot_10s")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_player_volume(player_rel: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    if player_rel.empty:
        return
    plot_df = player_rel.head(top_n).copy().sort_values("actions", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(plot_df["player"], plot_df["actions"])
    plt.title("Top players by defensive action volume")
    plt.xlabel("Action count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_stability(stability_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    if stability_df.empty:
        return
    plot_df = stability_df.head(top_n).copy().sort_values("corr_mean")
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x=plot_df["corr_mean"],
        y=plot_df["feature"],
        xerr=plot_df["corr_std"],
        fmt="o",
        capsize=3,
    )
    plt.axvline(0.0, color="black", linewidth=1.0)
    plt.title("Bootstrap stability of feature-target correlations")
    plt.xlabel("Mean correlation (+/- 1 std)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

