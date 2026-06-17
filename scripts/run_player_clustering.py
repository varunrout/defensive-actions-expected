#!/usr/bin/env python
"""Run defensive-style clustering on player summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dax.analysis.clustering import (
    feature_group_sensitivity,
    prepare_clustering_matrix,
    run_clustering,
    threshold_sensitivity,
    write_clustering_outputs,
)
from dax.analysis.config import load_analysis_config
from dax.analysis.plotting import bar_chart, labelled_heatmap, scatter_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run player defensive-style clustering.")
    parser.add_argument("--input", default="data/features/player_defensive_summary.parquet")
    parser.add_argument("--output-dir", default="outputs/analysis/clustering")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--matrix-output", default="data/features/player_clustering_matrix.parquet")
    return parser.parse_args()


def _write_table(table: pd.DataFrame, path_base: Path) -> None:
    table.to_csv(path_base.with_suffix(".csv"), index=False)
    table.to_parquet(path_base.with_suffix(".parquet"), index=False)


def _generate_clustering_charts(tables: dict[str, pd.DataFrame], output_dir: Path, dpi: int) -> None:
    player_clusters = tables["player_clusters"]
    cluster_sizes = player_clusters["cluster"].value_counts().rename_axis("cluster").reset_index(name="players")
    bar_chart(cluster_sizes, "cluster", "players", output_dir / "cluster_size_chart.png", "Cluster sizes", dpi=dpi)

    centroids = tables["cluster_centroids"].set_index("cluster") if not tables["cluster_centroids"].empty else pd.DataFrame()
    labelled_heatmap(centroids, output_dir / "cluster_centroid_heatmap.png", "Cluster centroid heatmap", dpi=dpi)

    pca_scores = tables.get("pca_scores", pd.DataFrame())
    if {"pc1", "pc2"}.issubset(pca_scores.columns):
        scatter_chart(pca_scores, "pc1", "pc2", output_dir / "pca_cluster_scatter.png", "PCA cluster scatter", color="cluster", dpi=dpi)

    loadings = tables.get("pca_loadings", pd.DataFrame())
    if not loadings.empty and "0" in [str(c) for c in loadings.columns]:
        first_component = loadings.columns[1]
        plot = loadings[["feature", first_component]].copy().sort_values(first_component, key=lambda s: s.abs(), ascending=False).head(20)
        bar_chart(plot, "feature", first_component, output_dir / "pca_loading_chart.png", "Top PCA loadings", dpi=dpi)

    stability = tables["cluster_stability"]
    stability_plot = stability.assign(solution=stability["method"] + "_k" + stability["clusters"].astype(str))
    bar_chart(stability_plot, "solution", "subsample_ari_stability", output_dir / "cluster_stability_chart.png", "Cluster subsample stability", dpi=dpi)

    profiles = tables["cluster_profiles"]
    bar_chart(profiles, "cluster", "mean_total_actions", output_dir / "cluster_profile_chart.png", "Cluster profile: mean actions", dpi=dpi)

    if not player_clusters.empty:
        sample = player_clusters.head(20).copy()
        sample["player_label"] = sample["player_name"].astype(str) + " (" + sample["cluster"].astype(str) + ")"
        bar_chart(sample, "player_label", "total_actions", output_dir / "selected_player_cluster_comparison.png", "Selected player cluster comparison", dpi=dpi)


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
    threshold_table = threshold_sensitivity(summary, config, tables["player_clusters"])
    feature_group_table = feature_group_sensitivity(summary, config, tables["player_clusters"])
    tables["cluster_threshold_sensitivity"] = threshold_table
    tables["cluster_feature_group_sensitivity"] = feature_group_table
    write_clustering_outputs(tables, output_dir, metadata)
    (output_dir / "selected_features.json").write_text(json.dumps(metadata["selected_features"], indent=2), encoding="utf-8")
    _generate_clustering_charts(tables, output_dir, int(config["chart_dpi"]))
    print(f"Ran clustering for {len(matrix):,} eligible players -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
