"""CLI entrypoint: run passive-defence archetype clustering (Phase 8) per
defender_functional_role bucket and write reports/analysis/shot_target/PASSIVE_ARCHETYPES.json.

Usage:
    python -m src.eda.generate_passive_archetypes
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dax.analysis.passive_archetypes import (
    BUCKETS,
    CLUSTER_ID_COLUMN,
    CLUSTER_NAME_COLUMN,
    DEFAULT_SAMPLE_N,
    DEFAULT_SEED,
    DRIFT_SHARE_PP_THRESHOLD,
    DRIFT_SHOT_RATE_PP_THRESHOLD,
    FEATURE_COLUMNS,
    full_population_cluster_stats,
    run_all_buckets,
)
from src.eda.feature_config import PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "PASSIVE_ARCHETYPES.json"


def _merge_full_population_stats(df: pd.DataFrame, results: dict) -> list[dict]:
    """Attach full-population (every labeled row, not just the ≤30,000-row
    fitting sample) cluster stats to each bucket's result in place, and
    return a flat list of any cluster where the full population meaningfully
    disagrees with the sample -- surfaced at the top level of the report
    rather than left buried inside each bucket."""
    if CLUSTER_ID_COLUMN not in df.columns:
        raise ValueError(
            f"{CLUSTER_ID_COLUMN!r} not found in the passive_defense dataset -- run "
            "scripts/features/build_passive_archetype_labels.py to fit and persist the archetype pipelines "
            "and write full-population labels before regenerating this report."
        )

    drift_flags: list[dict] = []
    for bucket, r in results.items():
        full_stats = full_population_cluster_stats(df, bucket, r["clusters"])
        r["full_population"] = full_stats
        for cluster_full in full_stats["clusters_full"]:
            if cluster_full["drift_flag"]:
                drift_flags.append({
                    "bucket": bucket,
                    "cluster_id": cluster_full["cluster_id"],
                    "name": cluster_full["name"],
                    "shot_rate_pct_sample": next(
                        c["shot_rate_pct"] for c in r["clusters"] if c["cluster_id"] == cluster_full["cluster_id"]
                    ),
                    "shot_rate_pct_full": cluster_full["shot_rate_pct_full"],
                    "sample_vs_full_shot_rate_delta_pp": cluster_full["sample_vs_full_shot_rate_delta_pp"],
                    "share_of_bucket_sample_pct": next(
                        c["share_of_bucket_sample"] for c in r["clusters"] if c["cluster_id"] == cluster_full["cluster_id"]
                    ),
                    "share_of_bucket_full_pct": cluster_full["share_of_bucket_full_pct"],
                    "sample_vs_full_share_delta_pp": cluster_full["sample_vs_full_share_delta_pp"],
                })
    return drift_flags


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])

    results = run_all_buckets(df, sample_n=DEFAULT_SAMPLE_N, seed=DEFAULT_SEED)
    drift_flags = _merge_full_population_stats(df, results)
    for r in results.values():
        r.pop("pipeline", None)

    output = {
        "dataset": "passive",
        "parquet_path": PASSIVE["parquet_path"],
        "n_rows_total": len(df),
        "buckets_clustered": BUCKETS,
        "buckets_excluded": ["unclassified"],
        "exclusion_reason": "unclassified is the n<2 structural case (nothing to be \"relative to\"), not a behavioural population to find archetypes within.",
        "feature_columns": FEATURE_COLUMNS,
        "feature_exclusion_note": (
            "defender_x/defender_y are deliberately excluded from the feature set -- those already define bucket "
            "membership via _functional_roles' tercile split. Clustering asks what varies WITHIN a role, not the "
            "role itself."
        ),
        "sampling_limitation": (
            f"For tractability, each bucket is clustered on a sample of at most {DEFAULT_SAMPLE_N:,} rows "
            f"(fixed seed {DEFAULT_SEED}), not the full population -- mid_block alone has 838,270 rows. This is an "
            "explicit, named limitation, not a silent shortcut: cluster sizes/shares/shot-rates below are computed "
            "on the sample, not the full bucket. The fitted pipeline is then reused (transform + predict only, no "
            "refitting) to label every row in the bucket, and each cluster's 'full_population' block reports the "
            "same stats recomputed over that full population, so the two can be compared directly."
        ),
        "full_population_drift_note": (
            f"A cluster's full-population shot rate or share is flagged as meaningful drift from its sample-based "
            f"figure when they differ by >= {DRIFT_SHOT_RATE_PP_THRESHOLD}pp (shot rate) or "
            f">= {DRIFT_SHARE_PP_THRESHOLD}pp (share of bucket) -- see 'full_population_drift_flags' below. A real "
            "disagreement here means the sample wasn't representative of the bucket, which is worth knowing, not "
            "burying inside each bucket's block."
        ),
        "full_population_drift_flags": drift_flags,
        "method_note": (
            "Median-impute missing, StandardScaler. KMeans swept over k in {2,3,4,5} (n_init=10, fixed seed), "
            "selected via the same percentile-rank multi-metric approach as dax.analysis.clustering."
            "evaluate_cluster_solutions (silhouette, Calinski-Harabasz, inverse Davies-Bouldin, size balance) minus "
            "the subsample-stability term. Final model refit at the selected k with n_init=20 for stability. "
            "Cluster archetypes are named from their top standardised-difference features, not by arbitrary KMeans "
            "label index (0/1 are not consistent in meaning across buckets or reruns)."
        ),
        "non_causal_note": (
            "Shot-rate differences between clusters within a bucket are associations, not evidence that any "
            "behaviour \"causes\" or \"prevents\" shots. This applies to every finding here, and especially to the "
            "central_screen engaged-vs-disengaged gap and the wide_cover/advanced_wide wide-lane direction flip, "
            "which are the findings most likely to get over-read as causal in a coaching context."
        ),
        "buckets": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=== Passive archetype clustering ===")
    for bucket, r in results.items():
        print(f"\n{bucket}: n_total={r['n_rows_total']:,} n_sampled={r['n_rows_sampled']:,} selected_k={r['selected_k']}")
        for c in r["clusters"]:
            print(f"  cluster {c['cluster_id']} ({c['name']}): n={c['n']:,} ({c['share_of_bucket_sample']}%) shot_rate={c['shot_rate_pct']}% delta={c['shot_rate_delta_vs_bucket_pp']:+.2f}pp")

    if drift_flags:
        print(f"\n!!! {len(drift_flags)} cluster(s) show meaningful sample-vs-full-population drift:")
        for flag in drift_flags:
            print(
                f"  {flag['bucket']} cluster {flag['cluster_id']} ({flag['name']}): "
                f"shot_rate sample={flag['shot_rate_pct_sample']}% full={flag['shot_rate_pct_full']}% "
                f"(delta={flag['sample_vs_full_shot_rate_delta_pp']:+.2f}pp); "
                f"share sample={flag['share_of_bucket_sample_pct']}% full={flag['share_of_bucket_full_pct']}% "
                f"(delta={flag['sample_vs_full_share_delta_pp']:+.2f}pp)"
            )
    else:
        print("\nNo meaningful sample-vs-full-population drift detected in any bucket/cluster.")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
