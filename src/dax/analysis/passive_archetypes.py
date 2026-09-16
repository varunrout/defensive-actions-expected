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

import argparse
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from dax.analysis.clustering import _cluster_scores

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARQUET_PATH = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
DEFAULT_MODEL_DIR = REPO_ROOT / "outputs" / "models" / "passive_archetypes"

CLUSTER_ID_COLUMN = "defender_archetype_cluster_id"
CLUSTER_NAME_COLUMN = "defender_archetype_name"

# How much a full-population stat is allowed to differ from the same stat
# computed on the ≤30,000-row sample before it's called out as meaningful
# drift (i.e. the sample may not have been representative of the bucket).
DRIFT_SHOT_RATE_PP_THRESHOLD = 1.0
DRIFT_SHARE_PP_THRESHOLD = 5.0

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


def _cast_boolean_features(df: pd.DataFrame) -> pd.DataFrame:
    raw = df[FEATURE_COLUMNS].copy()
    for col in BOOLEAN_FEATURE_COLUMNS:
        raw[col] = raw[col].astype(float)
    return raw


def _prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer, StandardScaler]:
    """Cast booleans to 0/1, median-impute, StandardScaler -- fitting both
    the imputer and scaler on `df`. Returns the scaled feature DataFrame
    (indexed the same as the input rows) plus the two fitted transformers,
    so callers can persist and reuse them (see `_transform_feature_matrix`)
    instead of only ever fitting fresh on a sample."""
    raw = _cast_boolean_features(df)

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)

    return pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=df.index), imputer, scaler


def _transform_feature_matrix(df: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler) -> pd.DataFrame:
    """Apply already-fitted imputer/scaler to `df` (transform only, no
    refitting) -- used to score the full bucket population against the
    pipeline that was fit on the bucket's sample."""
    raw = _cast_boolean_features(df)
    imputed = imputer.transform(raw)
    scaled = scaler.transform(imputed)
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

    features, imputer, scaler = _prepare_feature_matrix(sampled)
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
    # Single source of truth for cluster_id -> name: built in the same loop
    # that computes each cluster's report row, so the mapping persisted
    # alongside the fitted pipeline can never drift from the name shown in
    # the report.
    cluster_id_to_name: dict[int, str] = {}
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
        name = _name_cluster(standardised_diffs)
        cluster_id_to_name[int(cluster_id)] = name

        clusters.append({
            "cluster_id": int(cluster_id),
            "name": name,
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
        "cluster_id_to_name": cluster_id_to_name,
        # Not JSON-serialisable -- callers writing PASSIVE_ARCHETYPES.json
        # must pop this key first. Kept here (rather than a sibling
        # function) so the fitted pipeline can never fall out of sync with
        # the report/name computed above -- same fit, same call.
        "pipeline": {"imputer": imputer, "scaler": scaler, "kmeans": final_model},
    }


def run_all_buckets(df: pd.DataFrame, sample_n: int = DEFAULT_SAMPLE_N, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    results = {}
    for bucket in BUCKETS:
        results[bucket] = cluster_bucket(df, bucket, sample_n, seed)
    return results


def persist_pipelines(results: dict[str, dict[str, Any]], output_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Path]:
    """Dump each bucket's fitted imputer/scaler/kmeans + cluster_id_to_name
    map to `output_dir/<bucket>.joblib`, following this repo's
    outputs/models/<name>/... convention for persisted sklearn pipelines
    (see dax.models.training.ensure_output_dirs)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for bucket, result in results.items():
        bundle = {
            "bucket": bucket,
            "imputer": result["pipeline"]["imputer"],
            "scaler": result["pipeline"]["scaler"],
            "kmeans": result["pipeline"]["kmeans"],
            "cluster_id_to_name": result["cluster_id_to_name"],
            "feature_columns": FEATURE_COLUMNS,
            "boolean_feature_columns": sorted(BOOLEAN_FEATURE_COLUMNS),
            "selected_k": result["selected_k"],
            "sample_n_limit": result["sample_n_limit"],
        }
        path = output_dir / f"{bucket}.joblib"
        joblib.dump(bundle, path)
        paths[bucket] = path
    return paths


def load_pipeline(bucket: str, model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Reload a bundle previously written by `persist_pipelines`."""
    return joblib.load(model_dir / f"{bucket}.joblib")


def assign_cluster_labels(
    df: pd.DataFrame,
    bucket: str,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    kmeans: KMeans,
    cluster_id_to_name: dict[int, str],
) -> tuple[pd.Series, pd.Series]:
    """Score EVERY row of `df` in `bucket` (not just the fitting sample)
    through the already-fitted pipeline. Returns (cluster_id, cluster_name)
    Series aligned to the bucket's rows in `df` -- empty Series if the
    bucket has no rows."""
    bucket_df = df[df["defender_functional_role"] == bucket]
    if bucket_df.empty:
        return (
            pd.Series(dtype="Int64", name=CLUSTER_ID_COLUMN),
            pd.Series(dtype="object", name=CLUSTER_NAME_COLUMN),
        )
    features = _transform_feature_matrix(bucket_df, imputer, scaler)
    labels = kmeans.predict(features)
    cluster_id_series = pd.Series(labels, index=bucket_df.index, dtype="Int64", name=CLUSTER_ID_COLUMN)
    name_series = cluster_id_series.map(cluster_id_to_name).astype("object").rename(CLUSTER_NAME_COLUMN)
    return cluster_id_series, name_series


def assign_all_labels(df: pd.DataFrame, results: dict[str, dict[str, Any]]) -> tuple[pd.Series, pd.Series]:
    """Build the full-length defender_archetype_cluster_id / _name columns
    for `df`: every row whose bucket has a fitted pipeline in `results` gets
    a real label; every other row (unclassified, or any role absent from
    `results`) is left null -- it was never clustered and should not
    silently get a label. Iterates `results`' own keys (rather than the
    module-level BUCKETS list) so callers may pass a partial results dict."""
    cluster_id_col = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index, name=CLUSTER_ID_COLUMN)
    name_col = pd.Series([None] * len(df), index=df.index, dtype="object", name=CLUSTER_NAME_COLUMN)
    for bucket, result in results.items():
        ids, names = assign_cluster_labels(
            df,
            bucket,
            result["pipeline"]["imputer"],
            result["pipeline"]["scaler"],
            result["pipeline"]["kmeans"],
            result["cluster_id_to_name"],
        )
        cluster_id_col.loc[ids.index] = ids
        name_col.loc[names.index] = names
    return cluster_id_col, name_col


def full_population_cluster_stats(df_labeled: pd.DataFrame, bucket: str, sample_clusters: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute share/shot-rate/xG for `bucket` over every labeled row
    (not just the sample used to fit), and flag any cluster where the full
    population meaningfully disagrees with the sample -- a real disagreement
    would mean the ≤30,000-row sample wasn't representative, which is worth
    surfacing rather than burying."""
    bucket_df = df_labeled[df_labeled["defender_functional_role"] == bucket]
    n_total = len(bucket_df)
    bucket_mean_shot_rate_full = float(bucket_df[TARGET_SHOT_COL].mean() * 100) if n_total else float("nan")
    bucket_mean_xg_full = float(bucket_df[TARGET_XG_COL].mean()) if n_total else float("nan")

    clusters_full = []
    for sample_cluster in sample_clusters:
        cluster_id = sample_cluster["cluster_id"]
        c_rows = bucket_df[bucket_df[CLUSTER_ID_COLUMN] == cluster_id]
        n_full = int(len(c_rows))
        share_full = round(n_full / n_total * 100, 2) if n_total else 0.0
        shot_rate_full = round(float(c_rows[TARGET_SHOT_COL].mean() * 100), 3) if n_full else float("nan")
        xg_mean_full = round(float(c_rows[TARGET_XG_COL].mean()), 5) if n_full else float("nan")

        shot_rate_sample = sample_cluster["shot_rate_pct"]
        share_sample = sample_cluster["share_of_bucket_sample"]
        shot_rate_delta = round(shot_rate_full - shot_rate_sample, 3) if n_full else None
        share_delta = round(share_full - share_sample, 2)
        drift_flag = bool(
            n_full
            and (
                abs(shot_rate_delta) >= DRIFT_SHOT_RATE_PP_THRESHOLD
                or abs(share_delta) >= DRIFT_SHARE_PP_THRESHOLD
            )
        )

        clusters_full.append({
            "cluster_id": cluster_id,
            "name": sample_cluster["name"],
            "n_full": n_full,
            "share_of_bucket_full_pct": share_full,
            "shot_rate_pct_full": shot_rate_full,
            "xg_mean_full": xg_mean_full,
            "shot_rate_delta_vs_bucket_full_pp": round(shot_rate_full - bucket_mean_shot_rate_full, 3) if n_full else None,
            "sample_vs_full_shot_rate_delta_pp": shot_rate_delta,
            "sample_vs_full_share_delta_pp": share_delta,
            "drift_flag": drift_flag,
        })

    return {
        "n_rows_total_labeled": n_total,
        "bucket_mean_shot_rate_pct_full": round(bucket_mean_shot_rate_full, 3) if n_total else None,
        "bucket_mean_xg_full": round(bucket_mean_xg_full, 5) if n_total else None,
        "clusters_full": clusters_full,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit passive-defense archetype clustering pipelines, persist them, and write "
        "defender_archetype_cluster_id/name labels back across the full passive_defense.parquet population."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_PARQUET_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_PARQUET_PATH))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="Fit and validate without writing any files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_dir = Path(args.model_dir)

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    results = run_all_buckets(df, sample_n=args.sample_n, seed=args.seed)
    cluster_id_col, name_col = assign_all_labels(df, results)

    n_labeled = int(cluster_id_col.notna().sum())
    n_unclassified = len(df) - n_labeled
    print(f"Labeled {n_labeled:,} rows across {len(BUCKETS)} buckets; {n_unclassified:,} rows left null (unclassified/other).")

    if args.dry_run:
        print("[dry-run] fitted pipelines and computed labels, not writing any files")
        return 0

    paths = persist_pipelines(results, model_dir)
    for bucket, path in paths.items():
        print(f"Saved pipeline: {path}")

    if output_path == input_path and output_path.exists():
        backup_path = output_path.with_name(output_path.stem + ".pre_archetype_labels_backup" + output_path.suffix)
        shutil.copy2(output_path, backup_path)
        print(f"Backed up pre-change file to {backup_path}")

    df_out = df.copy()
    df_out[CLUSTER_ID_COLUMN] = cluster_id_col
    df_out[CLUSTER_NAME_COLUMN] = name_col
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"Saved {output_path} ({len(df_out.columns)} columns, {len(df_out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
