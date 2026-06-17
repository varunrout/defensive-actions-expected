#!/usr/bin/env python
"""Run defensive-style clustering on player summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dax.analysis.clustering import (
    expanded_cluster_profiles,
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
    bar_chart(cluster_sizes, "cluster", "players", output_dir / "cluster_size_chart.png", "Cluster sizes", dpi=dpi, color="cluster")

    centroids = tables["cluster_centroids"].set_index("cluster") if not tables["cluster_centroids"].empty else pd.DataFrame()
    labelled_heatmap(centroids, output_dir / "cluster_centroid_heatmap.png", "Cluster centroid heatmap", dpi=dpi)

    pca_scores = tables.get("pca_scores", pd.DataFrame())
    if {"pc1", "pc2"}.issubset(pca_scores.columns):
        scatter_chart(pca_scores, "pc1", "pc2", output_dir / "pca_cluster_scatter.png", "PCA cluster scatter", color="cluster", dpi=dpi)

    loadings = tables.get("pca_loadings", pd.DataFrame())
    if not loadings.empty and "0" in [str(c) for c in loadings.columns]:
        first_component = loadings.columns[1]
        plot = loadings[["feature", first_component]].copy().sort_values(first_component, key=lambda series: series.abs(), ascending=False).head(20)
        bar_chart(plot, "feature", first_component, output_dir / "pca_loading_chart.png", "Top PCA loadings", dpi=dpi)

    stability = tables["cluster_stability"]
    stability_plot = stability.assign(solution=stability["method"] + "_k" + stability["clusters"].astype(str))
    bar_chart(stability_plot, "solution", "subsample_ari_stability", output_dir / "cluster_stability_chart.png", "Cluster subsample stability", dpi=dpi)

    distinguishing = tables.get("cluster_distinguishing_features", pd.DataFrame())
    if not distinguishing.empty and "standardised_difference" in distinguishing.columns:
        profile = distinguishing.copy()
        profile["cluster_feature"] = "C" + profile["cluster"].astype(str) + ": " + profile["feature"].astype(str)
        profile = profile.sort_values("standardised_difference", key=lambda series: series.abs(), ascending=False).head(24)
        bar_chart(
            profile,
            "cluster_feature",
            "standardised_difference",
            output_dir / "cluster_profile_chart.png",
            "Cluster profile: distinguishing features",
            ylabel="Standardised difference from population",
            dpi=dpi,
            color="cluster",
        )
        positive = profile[profile["standardised_difference"] > 0].copy()
        negative = profile[profile["standardised_difference"] < 0].copy()
        if not positive.empty:
            bar_chart(positive, "cluster_feature", "standardised_difference", output_dir / "cluster_profile_positive_features.png", "Top positive cluster features", dpi=dpi, color="cluster")
        if not negative.empty:
            negative["absolute_difference"] = negative["standardised_difference"].abs()
            bar_chart(negative, "cluster_feature", "absolute_difference", output_dir / "cluster_profile_negative_features.png", "Top negative cluster features", dpi=dpi, color="cluster")

    representatives = tables.get("representative_players", pd.DataFrame())
    if not representatives.empty:
        sample = representatives.copy()
        sample["player_label"] = (
            sample.get("player_name", sample["player_id"]).astype(str)
            + " | "
            + sample.get("team", pd.Series("", index=sample.index)).astype(str)
            + " | C"
            + sample["cluster"].astype(str)
            + " | actions "
            + sample.get("total_actions", pd.Series(0, index=sample.index)).astype(str)
        )
        bar_chart(
            sample.sort_values(["cluster", "centroid_distance"]),
            "player_label",
            "centroid_distance",
            output_dir / "selected_player_cluster_comparison.png",
            "Centroid-selected representative players",
            ylabel="Distance to cluster centroid",
            dpi=dpi,
            color="cluster",
        )


def _write_expanded_profile_tables(matrix: pd.DataFrame, tables: dict[str, pd.DataFrame], summary: pd.DataFrame, output_dir: Path) -> None:
    expanded = expanded_cluster_profiles(matrix, tables["player_clusters"], summary)
    for name, table in expanded.items():
        if not table.empty:
            table.to_csv(output_dir / f"{name}.csv", index=False)
            table.to_parquet(output_dir / f"{name}.parquet", index=False)
            tables[name] = table


def _write_fixed_kmeans_view(matrix: pd.DataFrame, summary: pd.DataFrame, k: int, config: dict, output_dir: Path) -> None:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    feature_columns = [c for c in matrix.columns if c not in {"player_id", "player_name", "team", "total_actions", "matches"}]
    if len(matrix) <= k or not feature_columns:
        return
    view_dir = output_dir / f"k{k}"
    view_dir.mkdir(parents=True, exist_ok=True)
    labels = KMeans(n_clusters=k, random_state=int(config["random_seed"]), n_init=20).fit_predict(matrix[feature_columns])
    assignments = matrix[[c for c in ["player_id", "player_name", "team", "total_actions", "matches"] if c in matrix]].copy()
    assignments["cluster"] = labels
    assignments.to_csv(view_dir / "assignments.csv", index=False)
    assignments.to_parquet(view_dir / "assignments.parquet", index=False)
    sizes = assignments["cluster"].value_counts().rename_axis("cluster").reset_index(name="players")
    sizes.to_csv(view_dir / "cluster_sizes.csv", index=False)
    sizes.to_parquet(view_dir / "cluster_sizes.parquet", index=False)
    bar_chart(sizes, "cluster", "players", view_dir / "cluster_size_chart.png", f"K-means k={k} cluster sizes", color="cluster", dpi=int(config["chart_dpi"]))
    centroids = matrix[feature_columns].join(assignments["cluster"]).groupby("cluster").mean().reset_index()
    centroids.to_csv(view_dir / "centroids.csv", index=False)
    centroids.to_parquet(view_dir / "centroids.parquet", index=False)
    reps = expanded_cluster_profiles(matrix, assignments, summary)
    for name, table in reps.items():
        if not table.empty:
            table.to_csv(view_dir / f"{name}.csv", index=False)
            table.to_parquet(view_dir / f"{name}.parquet", index=False)
    distinguishing = reps.get("cluster_distinguishing_features", pd.DataFrame())
    if not distinguishing.empty:
        profile = distinguishing.assign(cluster_feature="C" + distinguishing["cluster"].astype(str) + ": " + distinguishing["feature"].astype(str))
        profile = profile.sort_values("standardised_difference", key=lambda series: series.abs(), ascending=False).head(20)
        bar_chart(profile, "cluster_feature", "standardised_difference", view_dir / "cluster_profile_chart.png", f"K-means k={k} distinguishing features", color="cluster", dpi=int(config["chart_dpi"]))
    # representatives by centroid distance
    from dax.analysis.clustering import representative_players_from_centroids
    fixed_representatives = representative_players_from_centroids(matrix, assignments)
    fixed_representatives.to_csv(view_dir / "representative_players.csv", index=False)
    fixed_representatives.to_parquet(view_dir / "representative_players.parquet", index=False)
    if len(feature_columns) >= 2:
        pca = PCA(n_components=2, random_state=int(config["random_seed"])).fit_transform(matrix[feature_columns])
        pca_df = assignments.assign(pc1=pca[:, 0], pc2=pca[:, 1])
        scatter_chart(pca_df, "pc1", "pc2", view_dir / "pca_scatter.png", f"K-means k={k} PCA view", color="cluster", dpi=int(config["chart_dpi"]))

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
    minimum_actions = int(config["minimum_player_actions"])
    configured_candidates = [int(k) for k in config.get("cluster_count_candidates", [])]
    for name, groups in buckets.items():
        subset = summary[summary["position_group"].isin(groups)].copy()
        group_dir = position_dir / name
        group_dir.mkdir(parents=True, exist_ok=True)
        total_players = int(len(subset))
        eligible = subset[subset["total_actions"] >= minimum_actions].copy() if "total_actions" in subset else subset.iloc[0:0].copy()
        eligible_players = int(len(eligible))
        excluded_players = total_players - eligible_players
        valid_candidates = [k for k in configured_candidates if eligible_players >= max(10, 4 * k) and eligible_players > k]
        status_payload = {
            "position_segment": name,
            "total_players": total_players,
            "eligible_players": eligible_players,
            "excluded_players": excluded_players,
            "minimum_player_actions": minimum_actions,
        }
        if not valid_candidates:
            reason = "insufficient eligible players after applying minimum_player_actions"
            pd.DataFrame([{**status_payload, "status": "skipped", "reason": reason}]).to_csv(group_dir / "cluster_evaluation.csv", index=False)
            continue
        variant = config.copy()
        variant["minimum_player_actions"] = minimum_actions
        variant["cluster_count_candidates"] = valid_candidates
        try:
            matrix, _, metadata = prepare_clustering_matrix(eligible, variant)
            pos_tables = run_clustering(matrix, variant)
            for table in pos_tables.values():
                if isinstance(table, pd.DataFrame):
                    for key, value in status_payload.items():
                        table[key] = value
            write_clustering_outputs(pos_tables, group_dir, metadata)
            _generate_clustering_charts(pos_tables, group_dir, int(config["chart_dpi"]))
        except ValueError as exc:
            pd.DataFrame([{**status_payload, "status": "skipped", "reason": str(exc)}]).to_csv(group_dir / "cluster_evaluation.csv", index=False)


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
    reps = tables.get("representative_players", pd.DataFrame())
    for _, representative in reps.iterrows():
        cluster = representative["cluster"]
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
    _write_expanded_profile_tables(matrix, tables, summary, output_dir)
    threshold_table = threshold_sensitivity(summary, config, tables["player_clusters"])
    feature_group_table = feature_group_sensitivity(summary, config, tables["player_clusters"])
    tables["cluster_threshold_sensitivity"] = threshold_table
    tables["cluster_feature_group_sensitivity"] = feature_group_table
    write_clustering_outputs(tables, output_dir, metadata)
    (output_dir / "selected_features.json").write_text(json.dumps(metadata["selected_features"], indent=2), encoding="utf-8")
    _write_fixed_kmeans_view(matrix, summary, 2, config, output_dir)
    _write_fixed_kmeans_view(matrix, summary, 3, config, output_dir)
    _write_k2_k3_comparison(tables, output_dir)
    _run_position_aware_clustering(summary, config, output_dir)
    _generate_cluster_spatial_outputs(args.actions_input, tables, output_dir, config)
    _generate_clustering_charts(tables, output_dir, int(config["chart_dpi"]))
    print(f"Ran clustering for {len(matrix):,} eligible players -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
