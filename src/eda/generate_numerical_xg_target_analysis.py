"""CLI entrypoint: numerical features vs the CONTINUOUS target
(target_future_xg_10s) -- the follow-up to generate_numerical_target_analysis.py,
which explicitly deferred this ("continuous target gets its own prompt after
this one"). Same discipline, same reconstructed pre-drop numerical pool
(build_pool is reused directly, unchanged -- pool construction is about
which features exist, not which target they're compared against), adapted
for a continuous, heavily zero-inflated target instead of a 0-100% rate.

Kept as a fully separate output set (reports/eda_xg/, its own portal) from
the binary-target reports in reports/eda/ -- different target, different
scale, different thresholds; conflating them into the same folder/portal
would blur that distinction rather than clarify it.

Scale note: target_future_xg_10s has mean ~0.008, std ~0.048, ~92% exact
zeros (see distribution printed at runtime) -- nothing like a 0-100% rate.
Shape-classification and consistency-check thresholds are therefore
RELATIVE to the dataset's own overall mean target value, not fixed
percentage-point margins:
  - flat_margin = 0.5 * overall_mean_target (shape classification)
  - consistency-check trigger = |Spearman rho| >= 0.05 (unchanged, scale-free)
    OR bin range >= 1.0 * overall_mean_target (relative, not 3pp)
Both are stated explicitly in the JSON/HTML output, not just implied.

Usage:
    python -m src.eda.generate_numerical_xg_target_analysis
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.dax.models.splits import load_canonical_split
from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import (
    DISCRETE_CARDINALITY_THRESHOLD,
    N_QUANTILE_BINS,
    RHO_THRESHOLD,
    UNRELIABLE_FEATURES,
    build_pool,
    classify_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "eda_xg"
TARGET = "target_future_xg_10s"

FLAT_MARGIN_RATIO = 0.5   # shape classification: flat_margin = ratio * overall mean target
RANGE_TRIGGER_RATIO = 1.0  # consistency-check trigger: bin range >= ratio * overall mean target


def _bin_table(df: pd.DataFrame, feature: str, is_discrete_cardinality: bool, quantile_edges: np.ndarray | None) -> tuple[list[dict], np.ndarray | None]:
    sub = df[[feature, TARGET]].dropna()
    if is_discrete_cardinality:
        grouped = sub.groupby(feature, observed=True)[TARGET].agg(["mean", "count"]).sort_index()
        bins = [
            {"bin": str(idx), "n": int(row["count"]), "mean_xg": round(float(row["mean"]), 6)}
            for idx, row in grouped.iterrows()
        ]
        return bins, None

    if quantile_edges is None:
        _, edges = pd.qcut(sub[feature], q=N_QUANTILE_BINS, duplicates="drop", retbins=True)
    else:
        edges = quantile_edges
    cats = pd.cut(sub[feature], bins=edges, include_lowest=True, duplicates="drop")
    grouped = sub.groupby(cats, observed=True)[TARGET].agg(["mean", "count"])
    bins = [
        {"bin": str(idx), "n": int(row["count"]), "mean_xg": round(float(row["mean"]), 6)}
        for idx, row in grouped.iterrows()
        if row["count"] > 0
    ]
    return bins, edges


def _consistency_check(df: pd.DataFrame, feature: str, is_discrete_cardinality: bool, quantile_edges: np.ndarray | None, split_assignment: dict, flat_margin: float) -> dict:
    match_ids = df["match_id"].astype(str)
    is_test = match_ids.map(split_assignment) == "test"

    train_val_bins, _ = _bin_table(df[~is_test], feature, is_discrete_cardinality, quantile_edges)
    test_bins, _ = _bin_table(df[is_test], feature, is_discrete_cardinality, quantile_edges)

    train_val_shape = classify_shape([b["bin"] for b in train_val_bins], [b["mean_xg"] for b in train_val_bins], flat_margin=flat_margin)
    test_shape = classify_shape([b["bin"] for b in test_bins], [b["mean_xg"] for b in test_bins], flat_margin=flat_margin)

    return {
        "checked": True,
        "train_val_shape": train_val_shape,
        "test_shape": test_shape,
        "consistent": train_val_shape == test_shape,
        "train_val_bins": train_val_bins,
        "test_bins": test_bins,
    }


def analyze_feature(df: pd.DataFrame, entry: dict, split_assignment: dict, overall_mean_target: float) -> dict:
    feature = entry["feature"]
    sub = df[[feature, TARGET]].dropna()
    n_used = len(sub)
    flat_margin = FLAT_MARGIN_RATIO * overall_mean_target
    range_trigger = RANGE_TRIGGER_RATIO * overall_mean_target

    if n_used < 50 or sub[feature].nunique() < 2:
        return {
            **entry,
            "n_rows_used": n_used,
            "pearson_r": None,
            "spearman_rho": None,
            "bins": [],
            "shape": "insufficient data",
            "consistency_check": {"checked": False, "reason": "n<50 or degenerate column"},
            "unreliable_note": UNRELIABLE_FEATURES.get(entry.get("_dataset", ""), {}).get(feature),
        }

    pearson_r, _ = pearsonr(sub[feature], sub[TARGET])
    spearman_rho, _ = spearmanr(sub[feature], sub[TARGET])

    n_unique = sub[feature].nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD
    bins, edges = _bin_table(df, feature, is_discrete_cardinality, None)
    values = [b["mean_xg"] for b in bins]
    shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin)

    rng = (max(values) - min(values)) if values else 0.0
    needs_check = abs(spearman_rho) >= RHO_THRESHOLD or rng >= range_trigger
    consistency = (
        _consistency_check(df, feature, is_discrete_cardinality, edges, split_assignment, flat_margin)
        if needs_check else
        {"checked": False, "reason": f"|rho|<{RHO_THRESHOLD} and bin range {rng:.5f} < {range_trigger:.5f} (={RANGE_TRIGGER_RATIO}x overall mean)"}
    )

    return {
        **entry,
        "n_rows_used": n_used,
        "n_unique_values": int(n_unique),
        "binning_method": "exact-value (discrete)" if is_discrete_cardinality else f"quantile deciles ({len(bins)} bins actually produced)",
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "bin_range": round(rng, 6),
        "bins": bins,
        "shape": shape,
        "consistency_check": consistency,
        "unreliable_note": UNRELIABLE_FEATURES.get(entry.get("_dataset", ""), {}).get(feature),
    }


def analyze_dataset(dataset_key: str, split_assignment: dict) -> dict:
    dataset_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    pool, pool_meta = build_pool(dataset_key)
    print(f"\n=== {dataset_key} (xG) ===")
    print(f"  reconstructed pool = {pool_meta['n_locked']} + {pool_meta['n_dropped']} = {pool_meta['n_total']} (same pool as the binary-target pass)")

    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"], columns=list({p["feature"] for p in pool}) + [TARGET, "match_id"])
    overall_mean_target = float(df[TARGET].mean())
    print(f"  overall mean {TARGET}: {overall_mean_target:.6f}  (flat_margin={FLAT_MARGIN_RATIO * overall_mean_target:.6f}, range_trigger={RANGE_TRIGGER_RATIO * overall_mean_target:.6f})")

    results = []
    for entry in pool:
        entry = {**entry, "_dataset": dataset_key}
        results.append(analyze_feature(df, entry, split_assignment, overall_mean_target))
        del results[-1]["_dataset"]

    results.sort(key=lambda r: abs(r["spearman_rho"]) if r["spearman_rho"] is not None else -1, reverse=True)
    n_flagged_inconsistent = sum(
        1 for r in results if r["consistency_check"].get("checked") and not r["consistency_check"].get("consistent")
    )

    return {
        "dataset": dataset_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "overall_mean_target": round(overall_mean_target, 6),
        "flat_margin": round(FLAT_MARGIN_RATIO * overall_mean_target, 6),
        "flat_margin_ratio": FLAT_MARGIN_RATIO,
        "range_trigger": round(RANGE_TRIGGER_RATIO * overall_mean_target, 6),
        "range_trigger_ratio": RANGE_TRIGGER_RATIO,
        "pool_construction": pool_meta,
        "n_features_analyzed": len(results),
        "n_features_checked_for_consistency": sum(1 for r in results if r["consistency_check"].get("checked")),
        "n_features_flagged_inconsistent": n_flagged_inconsistent,
        "rho_threshold": RHO_THRESHOLD,
        "features": results,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_assignment = load_canonical_split()

    for dataset_key in ("active", "passive"):
        output = analyze_dataset(dataset_key, split_assignment)
        out_path = OUT_DIR / f"{dataset_key}_numerical_target_atlas.json"
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
