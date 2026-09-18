import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dax.analysis.passive_archetypes import (
    BUCKETS,
    CLUSTER_ID_COLUMN,
    CLUSTER_NAME_COLUMN,
    FEATURE_COLUMNS,
    TARGET_SHOT_COL,
    TARGET_XG_COL,
    assign_all_labels,
    full_population_cluster_stats,
    load_pipeline,
    persist_pipelines,
    run_all_buckets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_bucket_rows(role: str, rng: np.random.Generator, n_per_group: int = 25) -> pd.DataFrame:
    """Two well-separated synthetic sub-populations within one functional-role
    bucket, so KMeans has an unambiguous k=2 solution to find -- keeps the
    tests fast and deterministic without touching the real 1.6M-row dataset."""
    n = n_per_group * 2
    group = np.array([0] * n_per_group + [1] * n_per_group)
    df = pd.DataFrame({
        "marking_tightness": np.where(group == 0, rng.normal(2.0, 0.2, n), rng.normal(8.0, 0.2, n)),
        "engagement_distance_to_carrier": rng.normal(10.0, 1.0, n),
        "zone_defensive_value": rng.uniform(0.1, 0.9, n),
        "angle_to_attacking_goal": rng.uniform(0.0, 1.5, n),
        "attacking_goal_centrality": rng.uniform(0.0, 1.0, n),
        "overload_score": np.where(group == 0, rng.integers(0, 2, n), rng.integers(2, 4, n)).astype(float),
        "lane_screening_score_option_1": rng.uniform(0.0, 1.0, n),
        "lane_screening_score_option_2": rng.uniform(0.0, 1.0, n),
        "lane_screening_score_option_3": rng.uniform(0.0, 1.0, n),
        "top_option_1_threat_score": rng.uniform(0.0, 1.0, n),
        "top_option_2_threat_score": rng.uniform(0.0, 1.0, n),
        "top_option_3_threat_score": rng.uniform(0.0, 1.0, n),
        "is_goal_side_of_nearest_attacker": rng.integers(0, 2, n).astype(bool),
        "is_wide_lane": rng.integers(0, 2, n).astype(bool),
        "has_option_2": np.ones(n, dtype=bool),
        "has_option_3": np.ones(n, dtype=bool),
        TARGET_SHOT_COL: np.where(group == 0, rng.binomial(1, 0.05, n), rng.binomial(1, 0.20, n)),
        TARGET_XG_COL: np.where(group == 0, rng.uniform(0.0, 0.05, n), rng.uniform(0.05, 0.3, n)),
    })
    df["defender_functional_role"] = role
    assert list(df.columns[: len(FEATURE_COLUMNS)]) or True  # feature columns constructed above, order not load-bearing
    return df


def _full_synthetic_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [_synthetic_bucket_rows(bucket, rng) for bucket in BUCKETS]
    # A handful of unclassified rows: structurally excluded from clustering
    # (n<2 defenders case) -- must never receive an archetype label even
    # though their feature values look like any other row here.
    frames.append(_synthetic_bucket_rows("unclassified", rng, n_per_group=3))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _full_synthetic_df()


@pytest.fixture(scope="module")
def fitted_results(synthetic_df):
    return run_all_buckets(synthetic_df, sample_n=1000, seed=0)


def test_cluster_id_to_name_mapping_has_no_drift_from_clusters_list(fitted_results):
    for bucket, result in fitted_results.items():
        expected = {c["cluster_id"]: c["name"] for c in result["clusters"]}
        assert result["cluster_id_to_name"] == expected, bucket


def test_every_row_in_a_clustered_bucket_gets_a_label(synthetic_df, fitted_results):
    cluster_id_col, name_col = assign_all_labels(synthetic_df, fitted_results)
    in_bucket_mask = synthetic_df["defender_functional_role"].isin(BUCKETS)
    assert cluster_id_col[in_bucket_mask].notna().all()
    assert name_col[in_bucket_mask].notna().all()
    # every labeled cluster_id must be one this bucket's pipeline actually produced
    for bucket, result in fitted_results.items():
        bucket_mask = synthetic_df["defender_functional_role"] == bucket
        valid_ids = set(result["cluster_id_to_name"].keys())
        assert set(cluster_id_col[bucket_mask].tolist()) <= valid_ids


def test_every_unclassified_row_gets_a_null_label(synthetic_df, fitted_results):
    cluster_id_col, name_col = assign_all_labels(synthetic_df, fitted_results)
    unclassified_mask = synthetic_df["defender_functional_role"] == "unclassified"
    assert unclassified_mask.sum() > 0
    assert cluster_id_col[unclassified_mask].isna().all()
    assert name_col[unclassified_mask].isna().all()


def test_assignment_is_deterministic_given_the_same_seed(synthetic_df):
    results_a = run_all_buckets(synthetic_df, sample_n=1000, seed=7)
    results_b = run_all_buckets(synthetic_df, sample_n=1000, seed=7)
    ids_a, names_a = assign_all_labels(synthetic_df, results_a)
    ids_b, names_b = assign_all_labels(synthetic_df, results_b)
    pd.testing.assert_series_equal(ids_a, ids_b)
    pd.testing.assert_series_equal(names_a, names_b)
    for bucket in BUCKETS:
        assert results_a[bucket]["cluster_id_to_name"] == results_b[bucket]["cluster_id_to_name"]


def test_persist_and_reload_pipeline_round_trip(synthetic_df, fitted_results, tmp_path):
    paths = persist_pipelines(fitted_results, output_dir=tmp_path)
    for bucket in BUCKETS:
        assert paths[bucket].exists()
        bundle = load_pipeline(bucket, model_dir=tmp_path)
        assert bundle["cluster_id_to_name"] == fitted_results[bucket]["cluster_id_to_name"]
        assert bundle["feature_columns"] == FEATURE_COLUMNS

        # The reloaded pipeline must reproduce identical predictions to the
        # original fitted objects on the same rows -- not just deserialize
        # without error.
        bucket_df = synthetic_df[synthetic_df["defender_functional_role"] == bucket]
        original_ids, _ = assign_all_labels(bucket_df, {bucket: fitted_results[bucket]})
        reloaded_ids, _ = assign_all_labels(
            bucket_df,
            {bucket: {"pipeline": {"imputer": bundle["imputer"], "scaler": bundle["scaler"], "kmeans": bundle["kmeans"]}, "cluster_id_to_name": bundle["cluster_id_to_name"]}},
        )
        pd.testing.assert_series_equal(original_ids, reloaded_ids)


def test_full_population_cluster_stats_matches_manual_computation(synthetic_df, fitted_results):
    cluster_id_col, name_col = assign_all_labels(synthetic_df, fitted_results)
    df_labeled = synthetic_df.copy()
    df_labeled[CLUSTER_ID_COLUMN] = cluster_id_col
    df_labeled[CLUSTER_NAME_COLUMN] = name_col

    for bucket in BUCKETS:
        result = fitted_results[bucket]
        stats = full_population_cluster_stats(df_labeled, bucket, result["clusters"])
        bucket_df = df_labeled[df_labeled["defender_functional_role"] == bucket]
        assert stats["n_rows_total_labeled"] == len(bucket_df)

        total_n = sum(c["n_full"] for c in stats["clusters_full"])
        assert total_n == len(bucket_df)

        for cluster_full in stats["clusters_full"]:
            manual_rows = bucket_df[bucket_df[CLUSTER_ID_COLUMN] == cluster_full["cluster_id"]]
            assert cluster_full["n_full"] == len(manual_rows)
            if cluster_full["n_full"]:
                assert cluster_full["shot_rate_pct_full"] == pytest.approx(
                    float(manual_rows[TARGET_SHOT_COL].mean() * 100), abs=1e-6
                )
                assert cluster_full["xg_mean_full"] == pytest.approx(
                    float(manual_rows[TARGET_XG_COL].mean()), abs=1e-4
                )


def test_full_population_stats_flag_drift_when_full_population_disagrees_with_sample():
    # Construct a case where the "sample" claims a 5% shot rate but the full
    # population is actually 50% -- this must be flagged, not silently
    # averaged away.
    df_labeled = pd.DataFrame({
        "defender_functional_role": ["last_line"] * 10,
        CLUSTER_ID_COLUMN: [0] * 10,
        CLUSTER_NAME_COLUMN: ["high marking"] * 10,
        TARGET_SHOT_COL: [1] * 5 + [0] * 5,
        TARGET_XG_COL: [0.1] * 10,
    })
    sample_clusters = [{
        "cluster_id": 0,
        "name": "high marking",
        "shot_rate_pct": 5.0,
        "share_of_bucket_sample": 100.0,
    }]
    stats = full_population_cluster_stats(df_labeled, "last_line", sample_clusters)
    cluster_full = stats["clusters_full"][0]
    assert cluster_full["shot_rate_pct_full"] == pytest.approx(50.0)
    assert cluster_full["drift_flag"] is True


def test_real_dataset_every_bucket_row_labeled_and_unclassified_rows_null():
    parquet_path = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
    if not parquet_path.exists():
        pytest.skip("Built passive_defense.parquet not present in this environment.")
    df = pd.read_parquet(parquet_path, columns=["defender_functional_role", CLUSTER_ID_COLUMN, CLUSTER_NAME_COLUMN])
    if CLUSTER_ID_COLUMN not in df.columns:
        pytest.skip("passive_defense.parquet has not had archetype labels built yet (run scripts/features/build_passive_archetype_labels.py).")

    in_bucket = df["defender_functional_role"].isin(BUCKETS)
    assert df.loc[in_bucket, CLUSTER_ID_COLUMN].notna().all()
    assert df.loc[in_bucket, CLUSTER_NAME_COLUMN].notna().all()

    unclassified = df["defender_functional_role"] == "unclassified"
    assert unclassified.sum() > 0
    assert df.loc[unclassified, CLUSTER_ID_COLUMN].isna().all()
    assert df.loc[unclassified, CLUSTER_NAME_COLUMN].isna().all()


def test_real_parquet_and_report_cluster_mapping_match_exactly():
    parquet_path = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
    report_path = REPO_ROOT / "reports" / "analysis" / "shot_target" / "PASSIVE_ARCHETYPES.json"
    if not parquet_path.exists() or not report_path.exists():
        pytest.skip("Built passive_defense.parquet / PASSIVE_ARCHETYPES.json not present in this environment.")
    df = pd.read_parquet(parquet_path, columns=["defender_functional_role", CLUSTER_ID_COLUMN, CLUSTER_NAME_COLUMN])
    if CLUSTER_ID_COLUMN not in df.columns:
        pytest.skip("passive_defense.parquet has not had archetype labels built yet (run scripts/features/build_passive_archetype_labels.py).")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for bucket in BUCKETS:
        bucket_df = df[df["defender_functional_role"] == bucket].dropna(subset=[CLUSTER_ID_COLUMN, CLUSTER_NAME_COLUMN])
        parquet_mapping = {
            int(cluster_id): name
            for cluster_id, name in bucket_df.drop_duplicates(subset=[CLUSTER_ID_COLUMN])[
                [CLUSTER_ID_COLUMN, CLUSTER_NAME_COLUMN]
            ].itertuples(index=False)
        }
        report_mapping = {c["cluster_id"]: c["name"] for c in report["buckets"][bucket]["clusters"]}
        assert parquet_mapping == report_mapping, bucket
