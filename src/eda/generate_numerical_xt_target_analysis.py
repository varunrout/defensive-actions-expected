"""CLI entrypoint: numerical features vs `target_xt_delta` (ACTIVE-BINARY
LEG ONLY) -- the xT sibling of generate_numerical_xg_target_analysis.py,
which is the confirmed generator of
reports/analysis/xg_target/active_numerical_target_atlas.json (confirmed by
reading it: same `build_pool` import, same `bins`/`shape`/
`consistency_check` key set, same canonical-split source).

Carried over UNCHANGED (scale-free, so the target's shape doesn't touch
them):
  - the reconstructed pre-drop numerical pool (`build_pool`, unmodified)
  - N_QUANTILE_BINS = 10 quantile deciles, DISCRETE_CARDINALITY_THRESHOLD = 15
  - RHO_THRESHOLD = 0.05 as the consistency-check trigger
  - classify_shape()'s U / inverse-U / monotonic / flat heuristic
  - the canonical match-grouped train/test split

ADAPTED, with the reason stated in the output itself:
  - flat_margin and the consistency range-trigger are translated from the
    xg convention via xg's own std-fraction (xt_common Adaptation 1);
    "0.5 x the overall mean" is meaningless for a symmetric target whose
    mean is -0.0013.
  - the shot-conditional panel becomes a non-zero-delta panel (Adaptation 2).
  - correlation/lift is framed and ranked two-directionally (Adaptation 3),
    and each bin also carries its negative/positive share.

Usage:
    python -m src.eda.generate_numerical_xt_target_analysis
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.dax.models.splits import load_canonical_split
from src.eda import xt_common as xc
from src.eda.generate_numerical_target_analysis import (
    DISCRETE_CARDINALITY_THRESHOLD,
    N_QUANTILE_BINS,
    RHO_THRESHOLD,
    UNRELIABLE_FEATURES,
    build_pool,
    classify_shape,
)

OUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET


def _bin_table(df: pd.DataFrame, feature: str, is_discrete_cardinality: bool,
               quantile_edges: np.ndarray | None) -> tuple[list[dict], np.ndarray | None]:
    sub = df[[feature, TARGET]].dropna()

    def _rows(grouped_obj) -> list[dict]:
        out = []
        for idx, s in grouped_obj:
            if len(s) == 0:
                continue
            out.append({
                "bin": str(idx),
                "n": int(len(s)),
                "mean_xt": round(float(s.mean()), 8),
                "pct_negative": round(float((s < 0).mean() * 100), 2),
                "pct_positive": round(float((s > 0).mean() * 100), 2),
            })
        return out

    if is_discrete_cardinality:
        g = sub.groupby(feature, observed=True)[TARGET]
        rows = _rows(g)
        rows.sort(key=lambda r: r["bin"])
        return rows, None

    if len(sub) == 0 or sub[feature].nunique() < 2:
        return [], quantile_edges
    if quantile_edges is None:
        _, edges = pd.qcut(sub[feature], q=N_QUANTILE_BINS, duplicates="drop", retbins=True)
    else:
        edges = quantile_edges
    cats = pd.cut(sub[feature], bins=edges, include_lowest=True, duplicates="drop")
    return _rows(sub.groupby(cats, observed=True)[TARGET]), edges


def _consistency_check(df, feature, is_discrete_cardinality, quantile_edges, split_assignment, flat_margin) -> dict:
    is_test = df["match_id"].astype(str).map(split_assignment) == "test"
    tv_bins, _ = _bin_table(df[~is_test], feature, is_discrete_cardinality, quantile_edges)
    te_bins, _ = _bin_table(df[is_test], feature, is_discrete_cardinality, quantile_edges)
    tv_shape = classify_shape([b["bin"] for b in tv_bins], [b["mean_xt"] for b in tv_bins], flat_margin=flat_margin)
    te_shape = classify_shape([b["bin"] for b in te_bins], [b["mean_xt"] for b in te_bins], flat_margin=flat_margin)
    return {
        "checked": True,
        "train_val_shape": tv_shape,
        "test_shape": te_shape,
        "consistent": tv_shape == te_shape,
        "train_val_bins": tv_bins,
        "test_bins": te_bins,
    }


def _nonzero_stats(df: pd.DataFrame, feature: str, is_discrete_cardinality: bool, flat_margin_nz: float) -> dict:
    """Adaptation 2's conditional panel: `target_xt_delta != 0` rather than
    `target_future_shot_10s == 1`. Fresh bin edges on this subset's own
    feature distribution, exactly as the xg version does."""
    nz = xc.nonzero_subset(df)
    sub = nz[[feature, TARGET]].dropna()
    n_used = len(sub)
    if n_used < 10 or sub[feature].nunique() < 2:
        return {"n_rows_used": n_used, "pearson_r": None, "spearman_rho": None, "bins": [], "shape": None,
                "note": "insufficient non-zero-delta data (n<10 or degenerate column)"}

    pr, _ = pearsonr(sub[feature], sub[TARGET])
    rho, _ = spearmanr(sub[feature], sub[TARGET])
    bins, _ = _bin_table(nz, feature, is_discrete_cardinality, None)
    for b in bins:
        b["small_n"] = bool(b["n"] < xc.SMALL_N_CONDITIONAL_THRESHOLD)
    shape = classify_shape([b["bin"] for b in bins], [b["mean_xt"] for b in bins], flat_margin=flat_margin_nz) if bins else None
    return {"n_rows_used": n_used, "pearson_r": round(float(pr), 4), "spearman_rho": round(float(rho), 4),
            "bins": bins, "shape": shape}


def analyze_feature(df: pd.DataFrame, entry: dict, split_assignment: dict, scale: dict) -> dict:
    feature = entry["feature"]
    sub = df[[feature, TARGET]].dropna()
    n_used = len(sub)
    flat_margin = scale["flat_margin"]
    range_trigger = scale["range_trigger"]
    flat_margin_nz = scale["flat_margin_nonzero"]

    n_unique_all = sub[feature].nunique()
    is_discrete_cardinality = n_unique_all <= DISCRETE_CARDINALITY_THRESHOLD
    nz = _nonzero_stats(df, feature, is_discrete_cardinality, flat_margin_nz)

    base = {
        **entry,
        "n_rows_used": n_used,
        "unreliable_note": UNRELIABLE_FEATURES.get("active", {}).get(feature),
        "n_nonzero_delta": nz["n_rows_used"],
        "pearson_r_nonzero_delta": nz["pearson_r"],
        "spearman_rho_nonzero_delta": nz["spearman_rho"],
        "bins_nonzero_delta": nz["bins"],
        "shape_nonzero_delta": nz.get("shape"),
    }

    if n_used < 50 or n_unique_all < 2:
        return {**base, "pearson_r": None, "spearman_rho": None, "bins": [], "shape": "insufficient data",
                "consistency_check": {"checked": False, "reason": "n<50 or degenerate column"}}

    pr, _ = pearsonr(sub[feature], sub[TARGET])
    rho, _ = spearmanr(sub[feature], sub[TARGET])

    bins, edges = _bin_table(df, feature, is_discrete_cardinality, None)
    values = [b["mean_xt"] for b in bins]
    shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin)
    rng = (max(values) - min(values)) if values else 0.0

    # Adaptation 3: does the binned curve cross zero, i.e. does this feature
    # separate threat-reducing from threat-increasing actions at all, rather
    # than only varying in magnitude on one side?
    crosses_zero = bool(values and min(values) < 0 < max(values))

    needs_check = abs(rho) >= RHO_THRESHOLD or rng >= range_trigger
    consistency = (
        _consistency_check(df, feature, is_discrete_cardinality, edges, split_assignment, flat_margin)
        if needs_check else
        {"checked": False,
         "reason": f"|rho|<{RHO_THRESHOLD} and bin range {rng:.6f} < {range_trigger:.6f} "
                   f"(translated from the xg suite's 1.0x-mean trigger via xg's own std-fraction)"}
    )

    return {
        **base,
        "n_unique_values": int(n_unique_all),
        "binning_method": "exact-value (discrete)" if is_discrete_cardinality else f"quantile deciles ({len(bins)} bins actually produced)",
        "pearson_r": round(float(pr), 4),
        "spearman_rho": round(float(rho), 4),
        "bin_range": round(rng, 8),
        "bin_mean_min": round(min(values), 8) if values else None,
        "bin_mean_max": round(max(values), 8) if values else None,
        "binned_curve_crosses_zero": crosses_zero,
        "bins": bins,
        "shape": shape,
        "consistency_check": consistency,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_assignment = load_canonical_split()

    pool, pool_meta = build_pool("active")
    df = xc.load_active_xt(columns=list({p["feature"] for p in pool}) + ["match_id"])
    scale = xc.target_scale(df)

    print("=== active (xT delta) ===")
    print(f"  reconstructed pool = {pool_meta['n_locked']} + {pool_meta['n_dropped']} = {pool_meta['n_total']}")
    print(f"  mean {TARGET}: {scale['mean']:+.6f}  std: {scale['std']:.6f}")
    print(f"  flat_margin={scale['flat_margin']:.6f}  range_trigger={scale['range_trigger']:.6f} "
          f"(std-fraction {scale['xg_flat_margin_as_fraction_of_xg_std']:.6f} translated from xg)")

    results = [analyze_feature(df, entry, split_assignment, scale) for entry in pool]
    results.sort(key=lambda r: abs(r["spearman_rho"]) if r["spearman_rho"] is not None else -1, reverse=True)

    n_inconsistent = sum(
        1 for r in results if r["consistency_check"].get("checked") and not r["consistency_check"].get("consistent")
    )
    n_crossing = sum(1 for r in results if r.get("binned_curve_crosses_zero"))

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "target_source": "outputs/prototypes/active_binary_xt_delta.parquet (joined read-only by event_id)",
        "target_scale": scale,
        "thresholds_carried_over_unchanged": {
            "n_quantile_bins": N_QUANTILE_BINS,
            "discrete_cardinality_threshold": DISCRETE_CARDINALITY_THRESHOLD,
            "rho_threshold": RHO_THRESHOLD,
            "classify_shape": "U / inverse-U / monotonic / flat heuristic, unmodified",
            "pool_construction": "generate_numerical_target_analysis.build_pool, unmodified",
            "split": "canonical match-grouped split (outputs/models/splits/match_assignment.json)",
        },
        "pool_construction": pool_meta,
        "n_features_analyzed": len(results),
        "n_features_checked_for_consistency": sum(1 for r in results if r["consistency_check"].get("checked")),
        "n_features_flagged_inconsistent": n_inconsistent,
        "n_features_whose_binned_curve_crosses_zero": n_crossing,
        "prompt_64_cross_references": {
            "distribution_shape": xc.PROMPT_64_SHAPE_FINDING,
            "clearance_action_x_artefact": xc.PROMPT_64_CLEARANCE_FINDING,
        },
        "features": results,
    }

    out_path = OUT_DIR / "active_numerical_target_atlas.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"  {n_inconsistent} train/test-inconsistent, {n_crossing}/{len(results)} features' binned curves cross zero")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
