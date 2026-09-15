"""Passive-defence archetype clustering (Phase 8).

Clusters passive_defense.parquet rows -- one row per defender-slot per
on-ball attacking event, no player identity, ever, by design -- separately
within each defender_functional_role bucket. This is deliberately NOT an
adaptation of dax.analysis.clustering, which is built around per-player
summary features (player_id, player_name, action_family_*_share,
total_actions) for the active-defence leg. Passive defence has no player
identity to key on, so that module's ID-column/feature-group config
plumbing doesn't apply here -- only its evaluation shape
(_cluster_scores, the percentile-rank multi-metric selection) is reused.

Clustering asks what varies WITHIN a role, not the role itself, so
defender_x/defender_y are deliberately excluded from the feature set --
those already define bucket membership via _functional_roles' tercile
split. unclassified is excluded from this analysis entirely: it's the
n < 2 structural case (nothing to be "relative to"), not a behavioural
population to find archetypes within.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.dax.analysis.clustering import _cluster_scores

BUCKETS = ["last_line", "wide_cover", "central_screen", "advanced_wide", "mid_block"]

FEATURE_COLUMNS = [
    "marking_tightness",
    "engagement_distance_to_carrier",
    "zone_defensive_value",
    "angle_to_attacking_goal",
    "attacking_goal_centrality",
    "overload_score",
    "lane_screening_score_option_1",
    "lane_screening_score_option_2",
    "lane_screening_score_option_3",
    "top_option_1_threat_score",
    "top_option_2_threat_score",
    "top_option_3_threat_score",
    "is_goal_side_of_nearest_attacker",
    "is_wide_lane",
    "has_option_2",
    "has_option_3",
]
BOOLEAN_FEATURE_COLUMNS = {"is_goal_side_of_nearest_attacker", "is_wide_lane", "has_option_2", "has_option_3"}

K_CANDIDATES = (2, 3, 4, 5)
DEFAULT_SAMPLE_N = 30_000
DEFAULT_SEED = 42
FINAL_N_INIT = 20
SWEEP_N_INIT = 10

TARGET_SHOT_COL = "target_future_shot_10s"
TARGET_XG_COL = "target_future_xg_10s"


def _prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cast booleans to 0/1, median-impute, StandardScaler. Returns a
    DataFrame of scaled features indexed the same as the input rows."""
    raw = df[FEATURE_COLUMNS].copy()
    for col in BOOLEAN_FEATURE_COLUMNS:
        raw[col] = raw[col].astype(float)

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    return pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=df.index)


def evaluate_k_sweep(features: pd.DataFrame, k_candidates: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Sweep k with KMeans, score with the same percentile-rank multi-metric
    approach as dax.analysis.clustering.evaluate_cluster_solutions, minus
    the subsample-stability term (too expensive to repeat per bucket at
    this sample size x k-sweep x 5-bucket scale)."""
    rows: list[dict[str, Any]] = []
    for k in k_candidates:
        if k >= len(features):
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=SWEEP_N_INIT)
        labels = model.fit_predict(features)
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "k": k,
            **_cluster_scores(features, labels),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "size_balance": float(sizes.min() / sizes.max()),
        })

    evaluation = pd.DataFrame(rows)
    if evaluation.empty:
        return evaluation

    evaluation["selection_score"] = (
        evaluation["silhouette"].fillna(0).rank(pct=True)
        + evaluation["calinski_harabasz"].fillna(0).rank(pct=True)
        + (1 / evaluation["davies_bouldin"].replace(0, np.nan)).fillna(0).rank(pct=True)
        + evaluation["size_balance"].fillna(0).rank(pct=True)
    ) / 4
    return evaluation.sort_values("selection_score", ascending=False).reset_index(drop=True)


def _name_cluster(diffs: dict[str, float], top_n: int = 2) -> str:
    """Descriptively name a cluster from its top-|standardised-difference|
    features -- never "Cluster 0"/"Cluster 1", since KMeans label indices
    are arbitrary per run."""
    ranked = sorted(diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    parts = []
    for feature, diff in ranked:
        direction = "high" if diff > 0 else "low"
        label = feature.replace("_", " ").replace("score", "").replace("option", "opt").strip()
        parts.append(f"{direction} {label}")
    return ", ".join(parts)


def cluster_bucket(df: pd.DataFrame, bucket: str, sample_n: int, seed: int) -> dict[str, Any]:
    bucket_df = df[df["defender_functional_role"] == bucket]
    n_total = len(bucket_df)
    sampled = bucket_df.sample(n=min(sample_n, n_total), random_state=seed) if n_total > sample_n else bucket_df

    features = _prepare_feature_matrix(sampled)
    evaluation = evaluate_k_sweep(features, K_CANDIDATES, seed)
    if evaluation.empty:
        raise ValueError(f"No valid clustering solution for bucket {bucket!r} (n={n_total}).")

    selected_k = int(evaluation.iloc[0]["k"])
    final_model = KMeans(n_clusters=selected_k, random_state=seed, n_init=FINAL_N_INIT)
    labels = final_model.fit_predict(features)

    assigned = sampled.copy()
    assigned["cluster"] = labels
    features_with_cluster = features.copy()
    features_with_cluster["cluster"] = labels

    bucket_mean_shot_rate = float(sampled[TARGET_SHOT_COL].mean() * 100)
    bucket_mean_xg = float(sampled[TARGET_XG_COL].mean())

    clusters = []
    for cluster_id in sorted(assigned["cluster"].unique()):
        c_rows = assigned[assigned["cluster"] == cluster_id]
        c_features_scaled = features_with_cluster[features_with_cluster["cluster"] == cluster_id][FEATURE_COLUMNS]
        # Features are already StandardScaler'd against the full bucket
        # sample (population mean 0, std 1), so the cluster's mean in
        # scaled space IS the standardised difference from the bucket
        # population mean.
        standardised_diffs = c_features_scaled.mean().to_dict()

        shot_rate = float(c_rows[TARGET_SHOT_COL].mean() * 100)
        xg_mean = float(c_rows[TARGET_XG_COL].mean())

        clusters.append({
            "cluster_id": int(cluster_id),
            "name": _name_cluster(standardised_diffs),
            "n": int(len(c_rows)),
            "share_of_bucket_sample": round(len(c_rows) / len(sampled) * 100, 2),
            "shot_rate_pct": round(shot_rate, 3),
            "xg_mean": round(xg_mean, 5),
            "shot_rate_delta_vs_bucket_pp": round(shot_rate - bucket_mean_shot_rate, 3),
            "standardised_feature_diffs": {k: round(float(v), 4) for k, v in standardised_diffs.items()},
            "top_distinguishing_features": [
                {"feature": f, "standardised_diff": round(float(d), 4)}
                for f, d in sorted(standardised_diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            ],
        })

    return {
        "bucket": bucket,
        "n_rows_total": n_total,
        "n_rows_sampled": len(sampled),
        "sampled": n_total > sample_n,
        "sample_n_limit": sample_n,
        "bucket_mean_shot_rate_pct": round(bucket_mean_shot_rate, 3),
        "bucket_mean_xg": round(bucket_mean_xg, 5),
        "k_evaluation": evaluation.to_dict(orient="records"),
        "selected_k": selected_k,
        "clusters": clusters,
    }


def run_all_buckets(df: pd.DataFrame, sample_n: int = DEFAULT_SAMPLE_N, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    results = {}
    for bucket in BUCKETS:
        results[bucket] = cluster_bucket(df, bucket, sample_n, seed)
    return results
