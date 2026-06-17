#!/usr/bin/env python
"""Run player defensive-action feature diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.analysis.config import load_analysis_config
from dax.analysis.data_quality import missingness_summary, write_tables
from dax.analysis.feature_diagnostics import diagnostics_tables
from dax.analysis.phase_analysis import phase_tables
from dax.analysis.plotting import bar_chart, histogram, labelled_heatmap, pitch_grid_heatmap
from dax.analysis.pitch_plotting import plot_pitch_density, plot_pitch_rate_map, plot_pitch_scatter
from dax.analysis.schemas import validate_player_actions
from dax.analysis.spatial_analysis import player_spatial_profiles, zone_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse canonical player defensive-action features.")
    parser.add_argument("--input", default="data/features/player_defensive_actions.parquet")
    parser.add_argument("--output-dir", default="outputs/analysis/features")
    parser.add_argument("--config", default="configs/analysis.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_analysis_config(args.config)
    output_dir = Path(args.output_dir)
    df = pd.read_parquet(args.input)
    validate_player_actions(df)
    bins_x, bins_y = config["pitch_grid_dimensions"]
    tables = diagnostics_tables(df, min_category_count=int(config["minimum_category_sample_size"]))
    tables["missingness"] = missingness_summary(df)
    tables["event_type_distribution"] = df["event_type"].value_counts(dropna=False).rename_axis("event_type").reset_index(name="rows")
    tables["zone_summary"] = zone_summary(df, bins_x=bins_x, bins_y=bins_y)
    tables["player_spatial_profiles"] = player_spatial_profiles(df)
    tables.update(phase_tables(df, min_count=int(config["minimum_category_sample_size"])))
    write_tables(tables, output_dir)

    dpi = int(config["chart_dpi"])
    bar_chart(df["action_family"].value_counts().rename_axis("action_family").reset_index(name="rows"), "action_family", "rows", output_dir / "action_family_distribution.png", "Action-family distribution", dpi=dpi)
    bar_chart(tables["event_type_distribution"], "event_type", "rows", output_dir / "player_event_type_distribution.png", "Player action event-type distribution", dpi=dpi)
    bar_chart(tables["missingness_by_feature_family"], "group", "mean_missing_rate", output_dir / "missingness_by_feature_family.png", "Missingness by feature family", dpi=dpi)
    labelled_heatmap(tables["correlations"], output_dir / "correlation_heatmap.png", "Labelled feature correlation heatmap", dpi=dpi)
    for column in ["visible_attacker_count", "visible_defender_count", "local_numerical_balance_5m", "local_numerical_balance_10m"]:
        if column in df.columns:
            histogram(df, column, output_dir / f"{column}_distribution.png", f"{column} distribution", bins=int(config["feature_bins"]), dpi=dpi)
    spatial_dir = output_dir.parent / "spatial" if output_dir.name == "features" else output_dir
    plot_pitch_scatter(df, spatial_dir / "all_actions_scatter.png", title="All defensive-action locations", dpi=dpi)
    plot_pitch_density(df, spatial_dir / "all_actions_density.png", title="All defensive-action density", bins=tuple(config.get("pitch_visualisation", {}).get("density_bins", [12, 8])), dpi=dpi)
    if "action_won_possession" in df.columns:
        plot_pitch_rate_map(df, spatial_dir / "possession_win_rate_map.png", value_col="action_won_possession", title="Possession-win rate by pitch location", min_bin_actions=int(config.get("minimum_spatial_bin_actions", 20)), dpi=dpi)
    plot_pitch_rate_map(df, spatial_dir / "future_shot_rate_map.png", value_col="target_future_shot_10s", title="Future-shot rate by pitch location", min_bin_actions=int(config.get("minimum_spatial_bin_actions", 20)), dpi=dpi)
    plot_pitch_rate_map(df, spatial_dir / "future_xg_map.png", value_col="target_future_xg_10s", title="Future-xG mean by pitch location", min_bin_actions=int(config.get("minimum_spatial_bin_actions", 20)), dpi=dpi)
    if "is_defensive_box_action" in df.columns:
        plot_pitch_density(df[df["is_defensive_box_action"]], spatial_dir / "defensive_box_actions.png", title="Defensive-box action density", dpi=dpi)
    pitch_grid_heatmap(tables["zone_summary"], "rows", output_dir / "total_action_pitch_heatmap.png", "Technical diagnostic: total action pitch grid", bins_x=bins_x, bins_y=bins_y, denominator_note="Denominator: player defensive-action rows", dpi=dpi)
    for family, family_df in df.groupby("action_family"):
        family_zone = zone_summary(family_df, bins_x=bins_x, bins_y=bins_y)
        plot_pitch_density(family_df, spatial_dir / f"{family}_density.png", title=f"Action-family density: {family}", dpi=dpi)
        pitch_grid_heatmap(family_zone, "rows", output_dir / f"action_family_pitch_map_{family}.png", f"Technical diagnostic: action-family grid {family}", bins_x=bins_x, bins_y=bins_y, denominator_note="Denominator: action-family rows", dpi=dpi)
    for phase, phase_df in df.groupby("phase_label"):
        safe_phase = str(phase).replace("/", "_").replace(" ", "_")
        phase_zone = zone_summary(phase_df, bins_x=bins_x, bins_y=bins_y)
        plot_pitch_density(phase_df, spatial_dir / f"phase_{safe_phase}_density.png", title=f"Phase density: {phase}", dpi=dpi)
        pitch_grid_heatmap(phase_zone, "rows", output_dir / f"phase_pitch_map_{safe_phase}.png", f"Technical diagnostic: phase grid {phase}", bins_x=bins_x, bins_y=bins_y, denominator_note="Denominator: phase rows", dpi=dpi)
    print(f"Analysed player defensive-action features: {len(df):,} rows -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
