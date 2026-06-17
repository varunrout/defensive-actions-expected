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
from dax.analysis.pitch_plotting import plot_cluster_population_difference, plot_pitch_density, plot_pitch_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run player defensive-style clustering.")
    parser.add_argument("--input", default="data/features/player_defensive_summary.parquet")
    parser.add_argument("--output-dir", default="outputs/analysis/clustering")
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--matrix-output", default="data/features/player_clustering_matrix.parquet")
    parser.add_argument("--actions-input", default="data/features/player_defensive_actions.parquet")
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



def _write_k2_k3_comparison(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    evaluation = tables["cluster_evaluation"]
    comparison = evaluation[(evaluation["method"] == "kmeans") & (evaluation["clusters"].isin([2, 3]))].copy()
    comparison.to_csv(output_dir / "k2_k3_comparison.csv", index=False)
    comparison.to_parquet(output_dir / "k2_k3_comparison.parquet", index=False)
    lines = [
        "# K=2 versus K=3 candidate interpretation",
        "",
        "These are candidate analytical views, not proof of a natural number of defensive styles.",
        "Stability should be interpreted alongside separation, size balance and football plausibility.",
        "",
    ]
    for row in comparison.to_dict("records"):
        lines.append(f"- K-means k={int(row['clusters'])}: silhouette={row['silhouette']:.6f}, stability={row['subsample_ari_stability']:.6f}, size balance={row['size_balance']:.6f}.")
    (output_dir / "k2_k3_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_position_aware_clustering(summary: pd.DataFrame, config: dict, output_dir: Path) -> None:
    if "position_group" not in summary.columns:
        return
    position_dir = output_dir / "by_position"
    position_dir.mkdir(parents=True, exist_ok=True)
    buckets = {
        "defenders": ["centre_back", "fullback_wingback", "goalkeeper"],
        "midfielders": ["defensive_midfielder", "midfielder"],
        "attackers": ["winger", "forward"],
    }
    for name, groups in buckets.items():
        subset = summary[summary["position_group"].isin(groups)].copy()
        group_dir = position_dir / name
        group_dir.mkdir(parents=True, exist_ok=True)
        if len(subset) < max(6, int(config["minimum_player_actions"]) // 2):
            pd.DataFrame([{"position_segment": name, "eligible_players": len(subset), "status": "too_few_players"}]).to_csv(group_dir / "cluster_evaluation.csv", index=False)
            continue
        variant = config.copy()
        variant["minimum_player_actions"] = min(int(config["minimum_player_actions"]), int(subset["total_actions"].max()))
        variant["cluster_count_candidates"] = [k for k in config["cluster_count_candidates"] if k < len(subset)] or [2]
        try:
            matrix, _, metadata = prepare_clustering_matrix(subset, variant)
            pos_tables = run_clustering(matrix, variant)
            write_clustering_outputs(pos_tables, group_dir, metadata)
            _generate_clustering_charts(pos_tables, group_dir, int(config["chart_dpi"]))
        except ValueError as exc:
            pd.DataFrame([{"position_segment": name, "eligible_players": len(subset), "status": "skipped", "reason": str(exc)}]).to_csv(group_dir / "cluster_evaluation.csv", index=False)


def _generate_cluster_spatial_outputs(actions_path: str, tables: dict[str, pd.DataFrame], output_dir: Path, config: dict) -> None:
    path = Path(actions_path)
    if not path.exists():
        return
    actions = pd.read_parquet(path)
    assignments = tables["player_clusters"]
    keys = [column for column in ["player_id", "team"] if column in actions.columns and column in assignments.columns]
    if not keys:
        return
    joined = actions.merge(assignments[keys + ["cluster"]], on=keys, how="inner")
    if joined.empty:
        return
    for cluster, cluster_actions in joined.groupby("cluster"):
        plot_pitch_density(cluster_actions, output_dir / f"cluster_{cluster}_spatial_profile.png", title=f"Cluster {cluster} spatial style profile", config=config)
        plot_cluster_population_difference(cluster_actions, joined, output_dir / f"cluster_{cluster}_spatial_difference.png", title=f"Cluster {cluster} density minus population", config=config)
        for family, family_actions in cluster_actions.groupby("action_family"):
            if len(family_actions) >= int(config.get("minimum_spatial_bin_actions", 20)):
                safe_family = str(family).replace(" ", "_").replace("/", "_")
                plot_pitch_density(family_actions, output_dir / f"cluster_{cluster}_{safe_family}_spatial_profile.png", title=f"Cluster {cluster} {family} spatial profile", config=config)
    # Representative player maps use highest-action player per cluster as a robust fallback.
    for cluster, cluster_players in assignments.sort_values("total_actions", ascending=False).groupby("cluster"):
        representative = cluster_players.iloc[0]
        mask = (joined["cluster"] == cluster) & (joined["player_id"] == representative["player_id"])
        if "team" in representative:
            mask &= joined["team"].eq(representative["team"])
        plot_pitch_scatter(joined[mask], output_dir / f"cluster_{cluster}_representative_player.png", title=f"Representative player, cluster {cluster}: {representative.get('player_name', representative['player_id'])}", config=config)

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
    _write_k2_k3_comparison(tables, output_dir)
    _run_position_aware_clustering(summary, config, output_dir)
    _generate_cluster_spatial_outputs(args.actions_input, tables, output_dir, config)
    _generate_clustering_charts(tables, output_dir, int(config["chart_dpi"]))
    print(f"Ran clustering for {len(matrix):,} eligible players -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
