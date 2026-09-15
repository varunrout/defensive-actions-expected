"""CLI entrypoint: run passive-defence archetype clustering (Phase 8) per
defender_functional_role bucket and write reports/eda/PASSIVE_ARCHETYPES.json.

Usage:
    python -m src.eda.generate_passive_archetypes
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dax.analysis.passive_archetypes import (
    BUCKETS,
    DEFAULT_SAMPLE_N,
    DEFAULT_SEED,
    FEATURE_COLUMNS,
    run_all_buckets,
)
from src.eda.feature_config import PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "PASSIVE_ARCHETYPES.json"


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])

    results = run_all_buckets(df, sample_n=DEFAULT_SAMPLE_N, seed=DEFAULT_SEED)

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
            "on the sample, not the full bucket."
        ),
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

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
