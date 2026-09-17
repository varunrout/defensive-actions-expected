"""CLI entrypoint: numerical features vs target_future_shot_10s -- the last
open piece of base-stage EDA for the binary leg. Category Atlas and Flag
Ledger already document categorical/boolean-vs-target relationships; this
does the same for continuous/discrete features, which the Distribution
Atlas never covered (it only describes each feature's own shape).

Runs on the RECONSTRUCTED pre-drop numerical candidate pool (51/44-era),
not the locked 34/38 -- redundancy/VIF drops were feature-vs-feature
decisions made without ever looking at the target, so a dropped feature can
still carry a real target relationship worth seeing. This does not reopen
the lock: nothing here writes back to feature_config.py, and every feature
is tagged locked/dropped so a dropped-but-strong pattern stands out rather
than blending in.

Pool construction (printed before any analysis runs, not just claimed):
  locked numerical (feature_config.ACTIVE/PASSIVE, continuous+discrete)
  UNION
  dropped numerical = (feature_config_v1_historical's reconstructed
    continuous+discrete) MINUS (locked numerical)
  MINUS, passive only: the 6 raw top_option_n_target_x/y columns -- their
    ball-relative replacements (dx/dy/distance_from_ball/angle_from_ball)
    are already locked and in scope, so keeping both would double-count the
    same underlying signal. This is the only exclusion applied; every other
    dropped numerical column is included even though some (e.g. the
    active-side centroids, superseded by defender_attacker_gap_x/y) are
    arguably a similar case -- left in per the "if unsure, include both"
    instruction rather than guessing.
  has_screened_outcome (passive leakage drop) is boolean, not numerical --
    confirmed absent from both the locked and historical numerical lists,
    not silently missed.

Usage:
    python -m src.eda.generate_numerical_target_analysis
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.dax.models.splits import load_canonical_split
from src.eda.feature_config import ACTIVE, PASSIVE, EXCLUDED_COLUMNS_PASSIVE
from src.eda.feature_config_v1_historical import ACTIVE_V1_HISTORICAL, PASSIVE_V1_HISTORICAL

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = "target_future_shot_10s"

N_QUANTILE_BINS = 10
DISCRETE_CARDINALITY_THRESHOLD = 15
RHO_THRESHOLD = 0.05
RANGE_THRESHOLD_PP = 3.0

COORDINATE_DUPLICATE_EXCLUSIONS = {
    f"top_option_{n}_target_{axis}" for n in (1, 2, 3) for axis in ("x", "y")
}

UNRELIABLE_FEATURES = {
    "active": {
        "nearest_defender_distance": (
            "Flagged in the Distribution Atlas as a likely self-reference bug -- 73.6% of rows are "
            "≈ 0m. Run and reported below like every other feature, but do not read the pattern as clean."
        ),
    },
    "passive": {},
}


def _numerical_set(cfg: dict) -> set[str]:
    return set(cfg["continuous"]) | set(cfg["discrete"])


def build_pool(dataset_key: str) -> list[dict]:
    """Returns [{feature, status, reason, type}], type = 'continuous'|'discrete'
    read from whichever config (locked or historical) actually classified it."""
    locked_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    hist_cfg = {"active": ACTIVE_V1_HISTORICAL, "passive": PASSIVE_V1_HISTORICAL}[dataset_key]

    locked_num = _numerical_set(locked_cfg)
    hist_num = _numerical_set(hist_cfg)
    dropped_num = hist_num - locked_num

    excluded_as_duplicate: set[str] = set()
    if dataset_key == "passive":
        excluded_as_duplicate = dropped_num & COORDINATE_DUPLICATE_EXCLUSIONS
        dropped_num = dropped_num - COORDINATE_DUPLICATE_EXCLUSIONS

    def _type_of(feature: str) -> str:
        return "continuous" if feature in hist_cfg["continuous"] else "discrete"

    pool = []
    for feature in sorted(locked_num):
        pool.append({"feature": feature, "status": "locked", "reason": None, "type": _type_of(feature)})
    for feature in sorted(dropped_num):
        reason = locked_cfg["excluded"].get(feature, "Dropped (reason not found in feature_config.py's excluded dict)")
        pool.append({"feature": feature, "status": "dropped", "reason": reason, "type": _type_of(feature)})

    pool.sort(key=lambda p: p["feature"])
    return pool, {
        "n_locked": len(locked_num),
        "n_dropped": len(dropped_num),
        "n_total": len(locked_num) + len(dropped_num),
        "excluded_as_coordinate_duplicate": sorted(excluded_as_duplicate),
    }


def classify_shape(bin_labels: list, rates: list[float], flat_margin: float = 1.5) -> str:
    """Classify a binned shot-rate curve, ordered by increasing feature value.
    Heuristic, stated explicitly rather than left to be read off a chart:
      - flat: range across bins < flat_margin (default 1.5, in the target's own units --
        percentage points for a 0-100% rate; callers on a different-scale target pass
        a scale-appropriate margin instead)
      - U-shaped / inverse-U: both-ends-vs-middle comparison, margin >= flat_margin
      - monotonic increasing/decreasing: |Spearman rho(bin_index, rate)| >= 0.7
      - otherwise: no-clear-pattern
    """
    n = len(rates)
    if n < 2:
        return "flat/no-clear-pattern (too few bins)"
    rates_arr = np.array(rates, dtype=float)
    rng = float(rates_arr.max() - rates_arr.min())
    if rng < flat_margin:
        return "flat"

    if n >= 4:
        third = max(1, n // 3)
        ends = np.concatenate([rates_arr[:third], rates_arr[-third:]])
        middle = rates_arr[third:-third] if n > 2 * third else rates_arr[third:third + 1]
        if len(middle) > 0:
            end_mean, mid_mean = float(ends.mean()), float(middle.mean())
            if end_mean - mid_mean >= flat_margin:
                return "U-shaped"
            if mid_mean - end_mean >= flat_margin:
                return "inverse-U"

    bin_index = np.arange(n)
    rho_bins, _ = spearmanr(bin_index, rates_arr)
    if rho_bins is not None and not np.isnan(rho_bins):
        if rho_bins >= 0.7:
            return "monotonic increasing"
        if rho_bins <= -0.7:
            return "monotonic decreasing"
    return "no-clear-pattern"


def _bin_table(df: pd.DataFrame, feature: str, is_discrete_cardinality: bool, quantile_edges: np.ndarray | None) -> tuple[list[dict], np.ndarray | None]:
    sub = df[[feature, TARGET]].dropna()
    if is_discrete_cardinality:
        grouped = sub.groupby(feature, observed=True)[TARGET].agg(["mean", "count"]).sort_index()
        bins = [
            {"bin": str(idx), "n": int(row["count"]), "shot_rate_pct": round(float(row["mean"]) * 100, 3)}
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
        {"bin": str(idx), "n": int(row["count"]), "shot_rate_pct": round(float(row["mean"]) * 100, 3)}
        for idx, row in grouped.iterrows()
        if row["count"] > 0
    ]
    return bins, edges


def _consistency_check(df: pd.DataFrame, feature: str, is_discrete_cardinality: bool, quantile_edges: np.ndarray | None, split_assignment: dict) -> dict:
    match_ids = df["match_id"].astype(str)
    is_test = match_ids.map(split_assignment) == "test"

    train_val_bins, _ = _bin_table(df[~is_test], feature, is_discrete_cardinality, quantile_edges)
    test_bins, _ = _bin_table(df[is_test], feature, is_discrete_cardinality, quantile_edges)

    train_val_shape = classify_shape([b["bin"] for b in train_val_bins], [b["shot_rate_pct"] for b in train_val_bins])
    test_shape = classify_shape([b["bin"] for b in test_bins], [b["shot_rate_pct"] for b in test_bins])

    consistent = train_val_shape == test_shape
    return {
        "checked": True,
        "train_val_shape": train_val_shape,
        "test_shape": test_shape,
        "consistent": consistent,
        "train_val_bins": train_val_bins,
        "test_bins": test_bins,
    }


def analyze_feature(df: pd.DataFrame, entry: dict, split_assignment: dict) -> dict:
    feature = entry["feature"]
    sub = df[[feature, TARGET]].dropna()
    n_used = len(sub)

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
    rates = [b["shot_rate_pct"] for b in bins]
    shape = classify_shape([b["bin"] for b in bins], rates)

    rng_pp = (max(rates) - min(rates)) if rates else 0.0
    needs_check = abs(spearman_rho) >= RHO_THRESHOLD or rng_pp > RANGE_THRESHOLD_PP
    consistency = (
        _consistency_check(df, feature, is_discrete_cardinality, edges, split_assignment)
        if needs_check else
        {"checked": False, "reason": f"|rho|<{RHO_THRESHOLD} and bin range {rng_pp:.2f}pp <= {RANGE_THRESHOLD_PP}pp"}
    )

    return {
        **entry,
        "n_rows_used": n_used,
        "n_unique_values": int(n_unique),
        "binning_method": "exact-value (discrete)" if is_discrete_cardinality else f"quantile deciles ({len(bins)} bins actually produced)",
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "bin_range_pp": round(rng_pp, 3),
        "bins": bins,
        "shape": shape,
        "consistency_check": consistency,
        "unreliable_note": UNRELIABLE_FEATURES.get(entry.get("_dataset", ""), {}).get(feature),
    }


def analyze_dataset(dataset_key: str, split_assignment: dict) -> dict:
    dataset_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    pool, pool_meta = build_pool(dataset_key)
    print(f"\n=== {dataset_key} ===")
    print(f"  locked numerical:  {pool_meta['n_locked']}")
    print(f"  dropped numerical: {pool_meta['n_dropped']}")
    if pool_meta["excluded_as_coordinate_duplicate"]:
        print(f"  excluded as coordinate-duplicate (replacement already locked): {pool_meta['excluded_as_coordinate_duplicate']}")
    print(f"  reconstructed pool = {pool_meta['n_locked']} + {pool_meta['n_dropped']} = {pool_meta['n_total']}")

    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"], columns=list({p["feature"] for p in pool}) + [TARGET, "match_id"])

    results = []
    for entry in pool:
        entry = {**entry, "_dataset": dataset_key}
        results.append(analyze_feature(df, entry, split_assignment))
        del results[-1]["_dataset"]

    results.sort(key=lambda r: abs(r["spearman_rho"]) if r["spearman_rho"] is not None else -1, reverse=True)

    n_flagged_inconsistent = sum(
        1 for r in results if r["consistency_check"].get("checked") and not r["consistency_check"].get("consistent")
    )

    return {
        "dataset": dataset_key,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "pool_construction": pool_meta,
        "n_features_analyzed": len(results),
        "n_features_checked_for_consistency": sum(1 for r in results if r["consistency_check"].get("checked")),
        "n_features_flagged_inconsistent": n_flagged_inconsistent,
        "rho_threshold": RHO_THRESHOLD,
        "range_threshold_pp": RANGE_THRESHOLD_PP,
        "features": results,
    }


def _verify_leakage_drop_not_silently_missed() -> None:
    """has_screened_outcome (passive leakage drop) is boolean, not numerical --
    explicitly checked here, not just assumed, per this prompt's acceptance
    criteria: confirm it's absent from both the locked and historical
    continuous/discrete lists (so it correctly never enters this pool) and
    present in the historical boolean list (so it wasn't dropped from the
    type classification entirely, just correctly out of scope for this
    numerical-only pass)."""
    assert "has_screened_outcome" not in _numerical_set(PASSIVE), "should not be in the locked numerical set"
    assert "has_screened_outcome" not in _numerical_set(PASSIVE_V1_HISTORICAL), "should not be in the historical numerical set either"
    assert "has_screened_outcome" in PASSIVE_V1_HISTORICAL["boolean"], "should be classified boolean, not missing entirely"
    assert "has_screened_outcome" in EXCLUDED_COLUMNS_PASSIVE, "should still be a documented leakage exclusion"
    print("Verified: has_screened_outcome (leakage drop) correctly out of scope for this numerical pass, not silently missed.")


def main() -> None:
    _verify_leakage_drop_not_silently_missed()
    split_assignment = load_canonical_split()

    for dataset_key in ("active", "passive"):
        output = analyze_dataset(dataset_key, split_assignment)
        out_path = REPO_ROOT / "reports" / "eda" / f"{dataset_key}_numerical_target_atlas.json"
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
