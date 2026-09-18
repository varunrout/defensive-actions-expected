#!/usr/bin/env python
"""Build provisional descriptive defensive signal components."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dax.analysis.config import load_analysis_config
from dax.analysis.signal_design import build_descriptive_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build provisional descriptive defensive signals.")
    parser.add_argument("--input", default="data/features/player_defensive_summary.parquet")
    parser.add_argument("--clusters", default="outputs/analysis/clustering/player_clusters.parquet")
    parser.add_argument("--output", default="data/features/player_defensive_signals_descriptive.parquet")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--min-actions", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_analysis_config(args.config)
    minimum_actions = args.min_actions if args.min_actions is not None else int(config["minimum_player_actions"])
    summary = pd.read_parquet(args.input)
    clusters = pd.read_parquet(args.clusters) if Path(args.clusters).exists() else None
    signals = build_descriptive_signals(summary, clusters, minimum_actions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output, index=False)
    signals.to_csv(output.with_suffix(".csv"), index=False)
    print(f"Built descriptive signals: {len(signals):,} player-team rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
