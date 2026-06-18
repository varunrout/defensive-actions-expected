#!/usr/bin/env python
"""Build player-team defensive summaries from canonical action features."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.analysis.config import load_analysis_config
from dax.analysis.player_aggregation import build_player_summary
from dax.analysis.plotting import bar_chart, scatter_chart
from dax.analysis.schemas import validate_player_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player defensive summary table.")
    parser.add_argument("--input", default="data/features/player_defensive_actions.parquet")
    parser.add_argument("--output", default="data/features/player_defensive_summary.parquet")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--min-actions", type=int, default=None, help="Override configured minimum player actions.")
    parser.add_argument("--charts-dir", default="outputs/analysis/players")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_analysis_config(args.config)
    minimum_actions = args.min_actions if args.min_actions is not None else int(config["minimum_player_actions"])
    df = pd.read_parquet(args.input)
    validate_player_actions(df)
    summary = build_player_summary(df, minimum_actions, grid_dimensions=tuple(config["pitch_grid_dimensions"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output, index=False)
    summary.to_csv(output.with_suffix(".csv"), index=False)

    charts_dir = Path(args.charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["chart_dpi"])
    bar_chart(summary.sort_values("total_actions", ascending=False).head(30), "player_name", "total_actions", charts_dir / "player_sample_size_ranking.png", "Top players by defensive-action sample", dpi=dpi)
    action_cols = [c for c in summary.columns if c.startswith("action_family_") and c.endswith("_share")]
    if action_cols:
        profile = summary[["player_name", *action_cols]].head(20).melt("player_name", var_name="action_family", value_name="share")
        bar_chart(profile, "action_family", "share", charts_dir / "player_action_family_profile.png", "Player action-family profile (first 20 players)", dpi=dpi)
    phase_cols = [c for c in summary.columns if c.startswith("phase_") and c.endswith("_share")]
    if phase_cols:
        profile = summary[["player_name", *phase_cols]].head(20).melt("player_name", var_name="phase", value_name="share")
        bar_chart(profile, "phase", "share", charts_dir / "player_phase_profile.png", "Player phase profile (first 20 players)", dpi=dpi)
    scatter_chart(summary, "actions_per_match", "future_shot_rate", charts_dir / "activity_vs_outcome_scatter.png", "Activity versus future-shot outcome", dpi=dpi)
    if "mean_local_numerical_balance_10m" in summary.columns:
        scatter_chart(summary, "mean_local_numerical_balance_10m", "future_shot_rate", charts_dir / "difficulty_vs_outcome_scatter.png", "Difficulty versus future-shot outcome", dpi=dpi)
    filtered = summary[~summary["minimum_sample_flag"]] if "minimum_sample_flag" in summary.columns else summary
    bar_chart(filtered.sort_values("possession_win_rate", ascending=False).head(30), "player_name", "possession_win_rate", charts_dir / "possession_win_rate_min_sample.png", "Possession-win rate (minimum-sample filtered)", dpi=dpi)
    scatter_chart(summary, "total_actions", "future_xg_mean", charts_dir / "future_xg_vs_action_volume.png", "Future xG versus action volume", dpi=dpi)
    if "reliable_visibility_share" in summary.columns:
        bar_chart(summary.sort_values("reliable_visibility_share", ascending=False).head(30), "player_name", "reliable_visibility_share", charts_dir / "visibility_reliability_chart.png", "Visibility reliability by player", dpi=dpi)
    print(f"Built player summary: {len(summary):,} player-team rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
