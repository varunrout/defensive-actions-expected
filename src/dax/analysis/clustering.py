"""Defensive-style clustering preprocessing, algorithms, and evaluation."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

IDENTIFIER_COLUMNS = {"player_id", "player_name", "team"}
EXCLUDED_PATTERNS = (
    "matches",
    "total_actions",
    "future_*",
    "target_*",
    "*_denominator",
    "*_count",
    "minimum_sample_flag",
    "reliable_sample",
)


def _matches_any(column: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(column, pattern) for pattern in patterns)


def select_clustering_features(summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    """Select configured clustering features by feature group and glob patterns."""
    groups = config["clustering_feature_groups"]
    primary_groups = config["primary_clustering_feature_groups"]
    numeric_columns = set(summary.select_dtypes(include="number").columns)
    spatial_mode = config.get("clustering_spatial_feature_mode", "reduced")
    selected: dict[str, list[str]] = {}
    for group in primary_groups:
        columns: list[str] = []
        patterns = list(groups[group])
        if group == "spatial_style" and spatial_mode == "full_grid" and "zone_*_share" not in patterns:
            patterns.append("zone_*_share")
        if group == "spatial_style" and spatial_mode == "reduced":
            patterns = [pattern for pattern in patterns if pattern != "zone_*_share"]
        for pattern in patterns:
            matches = [column for column in numeric_columns if fnmatch.fnmatch(column, pattern)]
            columns.extend(matches)
        filtered = sorted(
            column
            for column in set(columns)
            if column not in IDENTIFIER_COLUMNS and not _matches_any(column, EXCLUDED_PATTERNS)
        )
        selected[group] = filtered
    return selected


def prepare_clustering_matrix(summary: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create scaled and unscaled player matrices using explicit configured feature groups."""
    minimum_actions = int(config["minimum_player_actions"])
    eligible = summary[summary["total_actions"] >= minimum_actions].copy()
    id_columns = [column for column in ["player_id", "player_name", "team", "total_actions", "matches"] if column in eligible]
    selected_by_group = select_clustering_features(eligible, config)
    selected_features = sorted({column for columns in selected_by_group.values() for column in columns if column in eligible.columns})

    if not selected_features:
        raise ValueError("No clustering features selected. Check configs/analysis.yaml feature groups and player summary columns.")

    raw_features = eligible[selected_features].copy()
    missing_rates = raw_features.isna().mean()
    threshold = float(config["missing_value_threshold"])
    high_missing = sorted(missing_rates[missing_rates > threshold].index.tolist())
    retained_features = [column for column in selected_features if column not in high_missing]
    if not retained_features:
        raise ValueError("All selected clustering features exceed the configured missingness threshold.")

    raw_features = raw_features[retained_features]
    unique_counts = raw_features.nunique(dropna=True)
    constant_features = sorted(unique_counts[unique_counts <= 1].index.tolist())
    raw_features = raw_features.drop(columns=constant_features)
    if raw_features.empty:
        raise ValueError("All selected clustering features are constant after filtering.")

    imputer = SimpleImputer(strategy="median")
    imputed = pd.DataFrame(imputer.fit_transform(raw_features), columns=raw_features.columns, index=raw_features.index)
    scaler_name = config.get("scaling_method", "standard")
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(imputed), columns=imputed.columns, index=imputed.index)

    ids = eligible[id_columns].reset_index(drop=True)
    scaled_matrix = ids.join(scaled.reset_index(drop=True))
    unscaled_audit = ids.join(imputed.reset_index(drop=True))
    metadata = {
        "minimum_actions": minimum_actions,
        "selected_feature_groups": selected_by_group,
        "selected_features": list(scaled.columns),
        "excluded_patterns": list(EXCLUDED_PATTERNS),
        "high_missing_features_removed": high_missing,
        "constant_features_removed": constant_features,
        "missing_value_policy": "median_imputation",
        "scaling": scaler_name,
    }
    return scaled_matrix, unscaled_audit, metadata


def _cluster_scores(features: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    cluster_count = len(set(labels))
    if len(features) <= cluster_count or cluster_count < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": float(silhouette_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
    }


def _fit_labels(features: pd.DataFrame, method: str, k: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        return model.fit_predict(features), None
    if method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k)
        return model.fit_predict(features), None
    if method == "gmm":
        model = GaussianMixture(n_components=k, random_state=seed, n_init=5)
        labels = model.fit(features).predict(features)
        probabilities = model.predict_proba(features).max(axis=1)
        return labels, probabilities
    raise ValueError(f"Unsupported clustering method: {method}")


def _subsample_stability(features: pd.DataFrame, method: str, k: int, seed: int, repeats: int = 10) -> float:
    """Estimate label stability with repeated 80% subsamples and adjusted Rand index."""
    if len(features) < max(8, k * 3):
        return np.nan
    rng = np.random.default_rng(seed)
    reference_labels, _ = _fit_labels(features, method, k, seed)
    scores: list[float] = []
    sample_size = max(k + 1, int(len(features) * 0.8))
    for repeat in range(repeats):
        sample_index = np.sort(rng.choice(features.index.to_numpy(), size=sample_size, replace=False))
        sample = features.loc[sample_index]
        labels, _ = _fit_labels(sample, method, k, seed + repeat + 1)
        reference = pd.Series(reference_labels, index=features.index).loc[sample_index].to_numpy()
        scores.append(float(adjusted_rand_score(reference, labels)))
    return float(np.mean(scores)) if scores else np.nan


def evaluate_cluster_solutions(matrix: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Evaluate K-means, hierarchical, and GMM solutions across configured k values."""
    id_columns = [column for column in ["player_id", "player_name", "team", "total_actions", "matches"] if column in matrix]
    features = matrix.drop(columns=id_columns).fillna(0)
    rows: list[dict[str, Any]] = []
    max_k = min(max(config["cluster_count_candidates"]), len(features) - 1)
    for k in [candidate for candidate in config["cluster_count_candidates"] if 2 <= candidate <= max_k]:
        for method in ("kmeans", "hierarchical", "gmm"):
            labels, _ = _fit_labels(features, method, k, int(config["random_seed"]))
            sizes = pd.Series(labels).value_counts()
            rows.append(
                {
                    "method": method,
                    "clusters": k,
                    **_cluster_scores(features, labels),
                    "min_cluster_size": int(sizes.min()),
                    "max_cluster_size": int(sizes.max()),
                    "size_balance": float(sizes.min() / sizes.max()),
                    "subsample_ari_stability": _subsample_stability(features, method, k, int(config["random_seed"])),
                }
            )
    evaluation = pd.DataFrame(rows)
    if evaluation.empty:
        return evaluation
    evaluation["selection_score"] = (
        evaluation["silhouette"].fillna(0).rank(pct=True)
        + evaluation["calinski_harabasz"].fillna(0).rank(pct=True)
        + (1 / evaluation["davies_bouldin"].replace(0, np.nan)).fillna(0).rank(pct=True)
        + evaluation["size_balance"].fillna(0).rank(pct=True)
        + evaluation["subsample_ari_stability"].fillna(0).rank(pct=True)
    ) / 5
    return evaluation.sort_values("selection_score", ascending=False).reset_index(drop=True)


def run_clustering(matrix: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run evaluated clustering and return selected assignments plus diagnostics."""
    id_columns = [column for column in ["player_id", "player_name", "team", "total_actions", "matches"] if column in matrix]
    features = matrix.drop(columns=id_columns).fillna(0)
    evaluation = evaluate_cluster_solutions(matrix, config)
    if evaluation.empty:
        raise ValueError("No valid clustering solutions could be evaluated; lower minimum actions or k candidates.")

    selected = evaluation.iloc[0].to_dict()
    labels, probabilities = _fit_labels(features, str(selected["method"]), int(selected["clusters"]), int(config["random_seed"]))
    assignments = matrix[id_columns].copy()
    assignments["cluster"] = labels
    assignments["selected_method"] = selected["method"]
    assignments["selected_k"] = int(selected["clusters"])
    if probabilities is not None:
        assignments["membership_probability"] = probabilities
        assignments["low_confidence_assignment"] = assignments["membership_probability"] < 0.60
    else:
        assignments["membership_probability"] = np.nan
        assignments["low_confidence_assignment"] = False

    profiles = assignments.groupby("cluster", dropna=False).agg(players=("cluster", "size"), mean_total_actions=("total_actions", "mean")).reset_index()
    centroids = features.join(assignments["cluster"]).groupby("cluster").mean(numeric_only=True).reset_index()
    representatives = representative_players_from_centroids(matrix, assignments)

    pca_scores = pd.DataFrame()
    loadings = pd.DataFrame()
    explained = pd.DataFrame()
    if len(features.columns) and len(features) >= 2:
        pca = PCA(n_components=min(2, len(features.columns), len(features)), random_state=int(config["random_seed"])).fit(features)
        transformed = pca.transform(features)
        pca_scores = assignments[id_columns + ["cluster"]].join(
            pd.DataFrame(transformed, columns=["pc1", "pc2"][: pca.n_components_], index=assignments.index)
        )
        loadings = pd.DataFrame(pca.components_.T, index=features.columns).reset_index(names="feature")
        explained = pd.DataFrame(
            {"component": [f"pc{i + 1}" for i in range(pca.n_components_)], "explained_variance_ratio": pca.explained_variance_ratio_}
        )

    selection = pd.DataFrame(
        [
            {
                "selected_method": selected["method"],
                "selected_k": int(selected["clusters"]),
                "selection_rule": "highest average percentile rank across silhouette, Calinski-Harabasz, inverse Davies-Bouldin, size balance, and subsample ARI stability",
                "selection_score": selected["selection_score"],
            }
        ]
    )

    return {
        "cluster_evaluation": evaluation,
        "cluster_selection": selection,
        "player_clusters": assignments,
        "representative_players": representatives,
        "cluster_profiles": profiles,
        "cluster_centroids": centroids,
        "cluster_stability": evaluation[["method", "clusters", "subsample_ari_stability"]].copy(),
        "pca_scores": pca_scores,
        "pca_loadings": loadings,
        "pca_explained_variance": explained,
    }


def _selected_row_from_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    selection = tables["cluster_selection"].iloc[0]
    evaluation = tables["cluster_evaluation"]
    selected_eval = evaluation[
        (evaluation["method"] == selection["selected_method"]) & (evaluation["clusters"] == selection["selected_k"])
    ].iloc[0]
    return {
        "selected_method": selection["selected_method"],
        "selected_k": int(selection["selected_k"]),
        "silhouette": selected_eval["silhouette"],
        "calinski_harabasz": selected_eval["calinski_harabasz"],
        "davies_bouldin": selected_eval["davies_bouldin"],
        "stability": selected_eval["subsample_ari_stability"],
        "size_balance": selected_eval["size_balance"],
    }


def _assignment_agreement(base_assignments: pd.DataFrame, comparison_assignments: pd.DataFrame) -> float:
    keys = [column for column in ["player_id", "team"] if column in base_assignments.columns and column in comparison_assignments.columns]
    if not keys:
        return np.nan
    merged = base_assignments[keys + ["cluster"]].merge(
        comparison_assignments[keys + ["cluster"]], on=keys, suffixes=("_base", "_comparison")
    )
    if len(merged) < 2:
        return np.nan
    return float(adjusted_rand_score(merged["cluster_base"], merged["cluster_comparison"]))


def threshold_sensitivity(summary: pd.DataFrame, config: dict[str, Any], base_assignments: pd.DataFrame) -> pd.DataFrame:
    """Evaluate selected clustering solutions across minimum-action thresholds."""
    rows: list[dict[str, Any]] = []
    for threshold in config.get("minimum_action_threshold_sensitivity", []):
        variant = config.copy()
        variant["minimum_player_actions"] = int(threshold)
        try:
            matrix, _, _ = prepare_clustering_matrix(summary, variant)
            tables = run_clustering(matrix, variant)
            selected = _selected_row_from_tables(tables)
            rows.append(
                {
                    "minimum_actions": int(threshold),
                    "eligible_player_count": len(matrix),
                    **selected,
                    "assignment_agreement": _assignment_agreement(base_assignments, tables["player_clusters"]),
                }
            )
        except ValueError as exc:
            rows.append({"minimum_actions": int(threshold), "eligible_player_count": 0, "error": str(exc)})
    return pd.DataFrame(rows)


def feature_group_sensitivity(summary: pd.DataFrame, config: dict[str, Any], base_assignments: pd.DataFrame) -> pd.DataFrame:
    """Evaluate selected clustering solutions across configured feature-group subsets."""
    rows: list[dict[str, Any]] = []
    for sensitivity_name, groups in config.get("feature_group_sensitivity_sets", {}).items():
        variant = config.copy()
        variant["primary_clustering_feature_groups"] = list(groups)
        try:
            matrix, _, metadata = prepare_clustering_matrix(summary, variant)
            tables = run_clustering(matrix, variant)
            selected = _selected_row_from_tables(tables)
            rows.append(
                {
                    "feature_group_set": sensitivity_name,
                    "feature_groups": ",".join(groups),
                    "eligible_player_count": len(matrix),
                    "selected_feature_count": len(metadata["selected_features"]),
                    **selected,
                    "assignment_agreement": _assignment_agreement(base_assignments, tables["player_clusters"]),
                }
            )
        except ValueError as exc:
            rows.append({"feature_group_set": sensitivity_name, "feature_groups": ",".join(groups), "eligible_player_count": 0, "error": str(exc)})
    return pd.DataFrame(rows)


def clustering_feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Return scaled feature columns from a clustering matrix."""
    id_columns = [column for column in ["player_id", "player_name", "team", "total_actions", "matches"] if column in matrix]
    return [column for column in matrix.columns if column not in id_columns]


def representative_players_from_centroids(matrix: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Select the nearest player to each assigned cluster centroid in scaled feature space."""
    feature_columns = clustering_feature_columns(matrix)
    keys = [column for column in ["player_id", "team"] if column in matrix.columns and column in assignments.columns]
    data = matrix.merge(assignments[keys + ["cluster"]], on=keys, how="inner") if keys else matrix.join(assignments[["cluster"]])
    if not feature_columns or data.empty:
        fallback = assignments.sort_values("total_actions", ascending=False).groupby("cluster", as_index=False).head(1).copy()
        fallback["centroid_distance"] = np.nan
        fallback["centroid_rank_within_cluster"] = 1
        fallback["representative_selection_method"] = "highest_action_fallback"
        return fallback
    centroids = data.groupby("cluster")[feature_columns].mean()
    rows = []
    for cluster, part in data.groupby("cluster"):
        centroid = centroids.loc[cluster]
        distances = np.sqrt(((part[feature_columns] - centroid) ** 2).sum(axis=1))
        ranked = part.assign(centroid_distance=distances).sort_values("centroid_distance").reset_index(drop=True)
        representative = ranked.iloc[0].to_dict()
        representative["centroid_rank_within_cluster"] = 1
        representative["representative_selection_method"] = "nearest_centroid"
        rows.append(representative)
    columns = [column for column in ["player_id", "player_name", "team", "total_actions", "matches", "cluster", "centroid_distance", "centroid_rank_within_cluster", "representative_selection_method"] if rows and column in rows[0]]
    return pd.DataFrame(rows)[columns]


def expanded_cluster_profiles(matrix: pd.DataFrame, assignments: pd.DataFrame, summary: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Build expanded cluster profiles, distinguishing features, outcomes, and composition."""
    feature_columns = clustering_feature_columns(matrix)
    keys = [column for column in ["player_id", "team"] if column in matrix.columns and column in assignments.columns]
    data = matrix.merge(assignments[keys + ["cluster"]], on=keys, how="inner") if keys else matrix.join(assignments[["cluster"]])
    means = data.groupby("cluster")[feature_columns].mean() if feature_columns else pd.DataFrame()
    population_mean = data[feature_columns].mean() if feature_columns else pd.Series(dtype="float64")
    population_std = data[feature_columns].std(ddof=0).replace(0, np.nan) if feature_columns else pd.Series(dtype="float64")
    long_rows = []
    for cluster, row in means.iterrows():
        for feature in feature_columns:
            diff = row[feature] - population_mean[feature]
            long_rows.append({"cluster": cluster, "feature": feature, "mean": row[feature], "population_mean": population_mean[feature], "difference_from_population": diff, "standardised_difference": diff / population_std[feature] if pd.notna(population_std[feature]) else np.nan})
    long = pd.DataFrame(long_rows)
    top = (
        long.assign(abs_standardised_difference=long["standardised_difference"].abs())
        .sort_values(["cluster", "abs_standardised_difference"], ascending=[True, False])
        .groupby("cluster")
        .head(10)
        if not long.empty else pd.DataFrame()
    )
    profiles = means.reset_index() if not means.empty else pd.DataFrame(columns=["cluster"])
    counts = assignments.groupby("cluster").size().reset_index(name="players")
    profiles = counts.merge(profiles, on="cluster", how="left")
    outcome = pd.DataFrame()
    position = pd.DataFrame()
    if summary is not None and keys:
        joined = summary.merge(assignments[keys + ["cluster"]], on=keys, how="inner")
        outcome_cols = [c for c in ["future_shot_rate", "future_xg_mean", "possession_win_rate", "box_defence_share"] if c in joined]
        if outcome_cols:
            outcome = joined.groupby("cluster")[outcome_cols].mean().reset_index()
        if "position_group" in joined:
            position = pd.crosstab(joined["cluster"], joined["position_group"], normalize="index").reset_index()
    return {"cluster_profiles": profiles, "cluster_feature_profiles": long, "cluster_distinguishing_features": top, "cluster_outcome_summaries": outcome, "cluster_position_composition": position}


def write_clustering_outputs(tables: dict[str, pd.DataFrame], outdir: str | Path, metadata: dict[str, Any]) -> None:
    """Write clustering tables and preprocessing metadata."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
        table.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
