"""CLI entrypoint: reconstruct what CORRELATION_ANALYSIS.json looked like at
stage 01 -- the original 51/44-feature candidate lists, before any
redundancy-driven drop had landed. Mirrors generate_correlation_analysis.py's
analyze_dataset exactly, except the feature lists come from
feature_config_v1_historical.py instead of feature_config.py's current
(final, locked) lists.

Writes reports/analysis/shot_target/CORRELATION_ANALYSIS_V1_HISTORICAL.json -- a separate
file from CORRELATION_ANALYSIS.json, which stays exactly as stage 14 left
it (the post-lock confirmation pass, scored against the current list). This
script never touches that file.

Usage:
    python -m src.eda.generate_correlation_analysis_v1_historical
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.eda import correlation as corr
from src.eda.feature_config_v1_historical import DATASETS_V1_HISTORICAL
from src.eda.generate_correlation_analysis import METHODOLOGY, _feature_type_map, _print_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V1_HISTORICAL.json"

PASSIVE_SAMPLE_SEED = 42
PASSIVE_SAMPLE_N = 300_000
DISTINCT_SAMPLE_TOP_N = 10


def analyze_dataset(dataset_key: str) -> dict:
    dataset_cfg = DATASETS_V1_HISTORICAL[dataset_key]
    df_full = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    n_rows_full = len(df_full)

    type_map = _feature_type_map(dataset_cfg)
    feature_cols = list(type_map.keys())
    # Deliberately no _assert_no_excluded_columns call here -- this
    # reconstruction's candidate list is, by definition, wider than
    # feature_config.py's current `excluded` set for several columns that
    # were only excluded later in the real pipeline.

    sampled = False
    if dataset_key == "passive" and n_rows_full > PASSIVE_SAMPLE_N:
        df = df_full[feature_cols].sample(n=PASSIVE_SAMPLE_N, random_state=PASSIVE_SAMPLE_SEED)
        sampled = True
    else:
        df = df_full[feature_cols]

    n_rows_used = len(df)

    pairs_by_tier: dict[str, list[dict]] = {"drop": [], "collapse": [], "review": [], "distinct": []}
    skipped: list[dict] = []

    for col_a, col_b in itertools.combinations(feature_cols, 2):
        record = corr.compute_pair(df, col_a, type_map[col_a], col_b, type_map[col_b])
        if record is None:
            skipped.append({"feature_a": col_a, "feature_b": col_b, "reason": "n < 50 after dropping NaNs, or degenerate value"})
            continue
        pairs_by_tier[record["tier"]].append(record)

    tier_counts = {tier: len(records) for tier, records in pairs_by_tier.items()}

    distinct_sorted = sorted(pairs_by_tier["distinct"], key=lambda r: r["abs_value"], reverse=True)
    distinct_sample = distinct_sorted[:DISTINCT_SAMPLE_TOP_N]

    for tier in ("drop", "collapse", "review"):
        pairs_by_tier[tier].sort(key=lambda r: r["abs_value"], reverse=True)

    return {
        "dataset": dataset_key,
        "n_rows_full": n_rows_full,
        "n_rows_used": n_rows_used,
        "sampled": sampled,
        "sample_seed": PASSIVE_SAMPLE_SEED if sampled else None,
        "n_features": len(feature_cols),
        "n_pairs_evaluated": sum(tier_counts.values()),
        "n_pairs_skipped": len(skipped),
        "tier_counts": {**tier_counts, "distinct_total": tier_counts["distinct"]},
        "pairs": {
            "drop": pairs_by_tier["drop"],
            "collapse": pairs_by_tier["collapse"],
            "review": pairs_by_tier["review"],
            "distinct_sample": distinct_sample,
        },
        "skipped_pairs": skipped,
    }


def main() -> None:
    results = {key: analyze_dataset(key) for key in DATASETS_V1_HISTORICAL}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Reconstruction of stage 01's original candidate lists (51 active / 44 passive), not a live "
            "re-run -- see feature_config_v1_historical.py for exactly which columns were added back and why. "
            "CORRELATION_ANALYSIS.json (the file this pipeline actually ran forward from) was overwritten at "
            "stage 14 to score the final locked list instead; this file exists so V1's original scope is "
            "visible again."
        ),
        "methodology": METHODOLOGY,
        "datasets": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    _print_summary(results)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
