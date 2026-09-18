#!/usr/bin/env python
"""Run processed event-data pre-modelling analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.analysis.config import load_analysis_config
from dax.analysis.data_quality import processed_event_tables, write_tables
from dax.analysis.plotting import bar_chart, histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse processed events with corrected targets.")
    parser.add_argument("--input", default="data/processed/events_with_targets.parquet")
    parser.add_argument("--output-dir", default="outputs/analysis/data_quality")
    parser.add_argument("--config", default="configs/analysis.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_analysis_config(args.config)
    output_dir = Path(args.output_dir)
    df = pd.read_parquet(args.input)
    tables = processed_event_tables(df)
    write_tables(tables, output_dir)

    dpi = int(config["chart_dpi"])
    if "competition" in df.columns:
        competition_counts = df["competition"].value_counts(dropna=False).rename_axis("competition").reset_index(name="rows")
        bar_chart(competition_counts, "competition", "rows", output_dir / "events_by_competition.png", "Events by competition", dpi=dpi)
    bar_chart(tables["event_counts_by_type"], "type", "rows", output_dir / "events_by_type.png", "Events by type", dpi=dpi)
    bar_chart(tables["rows_per_match"], "match_id", "rows", output_dir / "rows_per_match.png", "Rows per match", dpi=dpi)
    bar_chart(tables["missingness"], "column", "missing_rate", output_dir / "missingness.png", "Missingness by column", ylabel="Missing rate", dpi=dpi)
    bar_chart(tables["phase_distribution"], "phase_label", "rows", output_dir / "phase_distribution.png", "Rule-based phase distribution", dpi=dpi)
    histogram(df, "target_future_xg_10s", output_dir / "target_future_xg_distribution.png", "Future xG target distribution", bins=int(config["feature_bins"]), dpi=dpi)
    print(f"Analysed processed events: {len(df):,} rows -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
