"""CLI entrypoint: V3 correlation analysis -- the same final locked feature
lists V1/V2 both score (34 active / 38 passive), but restricted to the
canonical split's TRAIN+VAL matches only, excluding every TEST match.

V1 and V2 both run on the full population (all 115 matches, TEST included)
-- fine for redundancy/collapse decisions, since dropping a duplicate or
collinear column doesn't leak target information the way fitting a scaler
or an encoder would. V3 exists as a check on that assumption: does any
correlation/redundancy verdict actually change once the held-out TEST
matches are removed? If V3's tiers match V2's, V1/V2 having been computed
on the full population was never a problem in practice, not just in theory.

Usage:
    python -m src.eda.generate_correlation_analysis_v3
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.dax.models.splits import load_canonical_split
from src.eda import correlation_v2 as corr
from src.eda.feature_config import DATASETS
from src.eda.generate_correlation_analysis import _assert_no_excluded_columns, _feature_type_map, _print_summary
from src.eda.generate_correlation_analysis_v2 import METHODOLOGY

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V3.json"

PASSIVE_SAMPLE_SEED = 42
PASSIVE_SAMPLE_N = 300_000
DISTINCT_SAMPLE_TOP_N = 10


def analyze_dataset(dataset_key: str, split_assignment: dict[str, str]) -> dict:
    dataset_cfg = DATASETS[dataset_key]
    df_full = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    n_rows_full = len(df_full)

    df_full = df_full.copy()
    df_full["match_id"] = df_full["match_id"].astype(str)
    df_full["_split"] = df_full["match_id"].map(split_assignment)
    train_val = df_full[df_full["_split"] != "test"]
    n_rows_train_val = len(train_val)
    n_test_excluded = n_rows_full - n_rows_train_val

    type_map = _feature_type_map(dataset_cfg)
    feature_cols = list(type_map.keys())
    _assert_no_excluded_columns(df_full, dataset_cfg, feature_cols)

    sampled = False
    if dataset_key == "passive" and n_rows_train_val > PASSIVE_SAMPLE_N:
        df = train_val[feature_cols].sample(n=PASSIVE_SAMPLE_N, random_state=PASSIVE_SAMPLE_SEED)
        sampled = True
    else:
        df = train_val[feature_cols]

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
        "n_rows_train_val": n_rows_train_val,
        "n_rows_test_excluded": n_test_excluded,
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
    split_assignment = load_canonical_split()
    results = {key: analyze_dataset(key, split_assignment) for key in DATASETS}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Same locked feature lists as V1/V2 (34 active / 38 passive), but scored on the canonical split's "
            "92 TRAIN+VAL matches only -- every TEST match excluded before any correlation is computed. Checks "
            "whether V1/V2's full-population redundancy verdicts would have changed with TEST held out."
        ),
        "n_matches_test_excluded": sum(1 for v in split_assignment.values() if v == "test"),
        "n_matches_train_val": sum(1 for v in split_assignment.values() if v != "test"),
        "methodology": METHODOLOGY,
        "datasets": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    _print_summary(results)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
