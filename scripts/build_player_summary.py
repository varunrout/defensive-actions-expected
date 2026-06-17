#!/usr/bin/env python
"""Build player-team defensive summaries from canonical action features."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.analysis.config import load_analysis_config
from dax.analysis.player_aggregation import build_player_summary
from dax.analysis.schemas import validate_player_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player defensive summary table.")
    parser.add_argument("--input", default="data/features/player_defensive_actions.parquet")
    parser.add_argument("--output", default="data/features/player_defensive_summary.parquet")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--min-actions", type=int, default=None, help="Override configured minimum player actions.")
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
    print(f"Built player summary: {len(summary):,} player-team rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
