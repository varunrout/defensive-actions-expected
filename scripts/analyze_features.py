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
    tables["zone_summary"] = zone_summary(df, bins_x=bins_x, bins_y=bins_y)
    tables["player_spatial_profiles"] = player_spatial_profiles(df)
    tables.update(phase_tables(df, min_count=int(config["minimum_category_sample_size"])))
    write_tables(tables, output_dir)

    dpi = int(config["chart_dpi"])
    bar_chart(df["action_family"].value_counts().rename_axis("action_family").reset_index(name="rows"), "action_family", "rows", output_dir / "action_family_distribution.png", "Action-family distribution", dpi=dpi)
    bar_chart(tables["missingness_by_feature_family"], "group", "mean_missing_rate", output_dir / "missingness_by_feature_family.png", "Missingness by feature family", dpi=dpi)
    labelled_heatmap(tables["correlations"], output_dir / "correlation_heatmap.png", "Labelled feature correlation heatmap", dpi=dpi)
    for column in ["visible_attacker_count", "visible_defender_count", "local_numerical_balance_5m", "local_numerical_balance_10m"]:
        if column in df.columns:
            histogram(df, column, output_dir / f"{column}_distribution.png", f"{column} distribution", bins=int(config["feature_bins"]), dpi=dpi)
    pitch_grid_heatmap(tables["zone_summary"], "rows", output_dir / "total_action_pitch_heatmap.png", "Total action density by pitch zone", bins_x=bins_x, bins_y=bins_y, denominator_note="Denominator: player defensive-action rows", dpi=dpi)
    print(f"Analysed player defensive-action features: {len(df):,} rows -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
