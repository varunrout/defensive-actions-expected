"""CLI entrypoint: compute the mixed-type correlation/association matrix for
both feature datasets and classify every pair into DROP / COLLAPSE / REVIEW /
DISTINCT.

Usage:
    python -m src.eda.generate_correlation_analysis
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.eda import correlation as corr
from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS.json"

PASSIVE_SAMPLE_SEED = 42
PASSIVE_SAMPLE_N = 300_000
DISTINCT_SAMPLE_TOP_N = 10

METHODOLOGY = {
    "numeric<->numeric": "Spearman rho (continuous+discrete pooled as 'numeric'; robust to the heavy right-skew present in several features, e.g. top_option_1_threat_score skew 7.91)",
    "boolean<->boolean": "phi coefficient (Pearson r on 0/1)",
    "boolean<->numeric": "point-biserial r",
    "categorical<->categorical": "Cramer's V, bias-corrected (boolean treated as a 2-level categorical when paired with a categorical column)",
    "categorical<->numeric": "correlation ratio eta",
    "tiers": {
        "drop": "|r| >= 0.98 (confirmed > 0.999) for numeric/boolean pairs, Cramer's V >= 0.9 (confirmed > 0.99), or eta >= 0.98 (confirmed > 0.995) -- exact/near-exact duplicates, delete one side without further analysis",
        "collapse": "0.90 <= |r| < 0.98, or 0.4 <= Cramer's V < 0.9, or 0.6 <= eta < 0.98 (also includes DROP-candidates that failed the near-1.0 confirm check, flagged downgraded_from_drop)",
        "review": "0.5 <= |r| < 0.90 for numeric/boolean pairs only -- resolved by a separate review-analysis pass, not by this script",
        "distinct": "below all thresholds above -- no action needed",
    },
}


def _feature_type_map(dataset_cfg: dict) -> dict[str, str]:
    types: dict[str, str] = {}
    for col in dataset_cfg["categorical"]:
        types[col] = "categorical"
    for col in dataset_cfg["boolean"]:
        types[col] = "boolean"
    for col in dataset_cfg["continuous"] + dataset_cfg["discrete"]:
        types[col] = "numeric"
    return types


def _assert_no_excluded_columns(df: pd.DataFrame, dataset_cfg: dict, feature_cols: list[str]) -> None:
    excluded = set(dataset_cfg["excluded"].keys())
    leaked = excluded & set(feature_cols)
    if leaked:
        raise AssertionError(
            f"[{dataset_cfg['key']}] Excluded columns found in the candidate feature list passed to "
            f"correlation analysis -- this is a feature_config.py bug, not a data issue: {sorted(leaked)}"
        )


def analyze_dataset(dataset_key: str) -> dict:
    dataset_cfg = DATASETS[dataset_key]
    df_full = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    n_rows_full = len(df_full)

    type_map = _feature_type_map(dataset_cfg)
    feature_cols = list(type_map.keys())
    _assert_no_excluded_columns(df_full, dataset_cfg, feature_cols)

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


def _print_summary(results: dict[str, dict]) -> None:
    header = f"{'dataset':<10} {'drop':>6} {'collapse':>10} {'review':>8} {'distinct':>10} {'skipped':>9}"
    print("\n" + header)
    print("-" * len(header))
    for key, r in results.items():
        tc = r["tier_counts"]
        print(f"{key:<10} {tc['drop']:>6} {tc['collapse']:>10} {tc['review']:>8} {tc['distinct_total']:>10} {r['n_pairs_skipped']:>9}")


def main() -> None:
    results = {key: analyze_dataset(key) for key in DATASETS}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": METHODOLOGY,
        "datasets": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    _print_summary(results)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
