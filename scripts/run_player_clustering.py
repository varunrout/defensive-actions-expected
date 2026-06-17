#!/usr/bin/env python
"""Run defensive-style clustering on player summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dax.analysis.clustering import prepare_clustering_matrix, run_clustering, write_clustering_outputs
from dax.analysis.config import load_analysis_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run player defensive-style clustering.")
    parser.add_argument("--input", default="data/features/player_defensive_summary.parquet")
    parser.add_argument("--output-dir", default="outputs/analysis/clustering")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--matrix-output", default="data/features/player_clustering_matrix.parquet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_analysis_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_parquet(args.input)
    matrix, audit, metadata = prepare_clustering_matrix(summary, config)
    matrix_output = Path(args.matrix_output)
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(matrix_output, index=False)
    audit.to_csv(output_dir / "clustering_unscaled_audit.csv", index=False)
    tables = run_clustering(matrix, config)
    write_clustering_outputs(tables, output_dir, metadata)
    (output_dir / "selected_features.json").write_text(json.dumps(metadata["selected_features"], indent=2), encoding="utf-8")
    print(f"Ran clustering for {len(matrix):,} eligible players -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
