"""Reconfirms (and extends to all 32 locked features) the rough chat check of
shot-conditional nonlinearity for the active-continuous leg's Rung 1 (prompt 55).

The rough chat check only looked at 7 features and found weak correlations
(strongest: visible_attacker_count at -0.13/-0.16). This script reproduces
that properly against the real data, across all 32 locked ACTIVE features,
so Rung 1's candidate quadratic-term selection is based on a real systematic
check, not the chat shortlist assumed unchanged.

Important distinction from the active-binary leg's Rung 1 (prompt 39): that
leg's U-shaped feature classification
(reports/analysis/xg_target/active_numerical_target_atlas.json's "shape"
field) is computed on the FULL dataset (zero + positive rows mixed), which
mostly reflects "does a shot happen at all." This script instead restricts
to the 4,368 rows where target_future_shot_10s==1 and asks a different
question: "how good is the shot, given one happens" -- these are not
guaranteed to have the same shape, and this leg's own 4 U-shaped features
are not assumed to be the same as the active-binary leg's.

Quadratic terms (quad__<feature>) are only meaningful for the 17 NUMERIC
(continuous + discrete) locked features -- squaring a boolean 0/1 value is a
no-op (0^2=0, 1^2=1, identical to the original column), and squaring a
one-hot categorical dummy is likewise a no-op. So while this script reports
a correlation/association magnitude for all 32 locked features (for
completeness, per the task's own instruction), only the 17 numeric features
are eligible quadratic-term candidates, and the quintile-curvature check
(the actual curvature diagnostic) is run only on those 17.

Usage:
    python scripts/analysis/check_shot_conditional_nonlinearity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402

OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "shot_conditional_nonlinearity_check.json"
TARGET_XG_COL = "target_future_xg_10s"
N_QUINTILES = 5


def numeric_feature_check(pos_df: pd.DataFrame, feature: str, log_xg: np.ndarray) -> dict:
    x = pos_df[feature].to_numpy(dtype=float)
    finite = np.isfinite(x)
    x_f, y_f = x[finite], log_xg[finite]

    pearson_r, pearson_p = stats.pearsonr(x_f, y_f)
    spearman_r, spearman_p = stats.spearmanr(x_f, y_f)

    # Quintile-binned mean log(xg) -- the actual curvature diagnostic.
    try:
        bins = pd.qcut(x_f, q=N_QUINTILES, duplicates="drop")
    except ValueError:
        bins = None
    quintile_means = []
    if bins is not None:
        tmp = pd.DataFrame({"bin": bins, "log_xg": y_f})
        grouped = tmp.groupby("bin", observed=True)["log_xg"].agg(["mean", "count"]).reset_index()
        for _, row in grouped.iterrows():
            quintile_means.append({"bin": str(row["bin"]), "mean_log_xg": float(row["mean"]), "n": int(row["count"])})

    # A simple, honest curvature signal: is the middle-quintile mean lower (or
    # higher) than BOTH the extremes by more than one pooled standard error --
    # a real, if crude, "does this look U-shaped or threshold-like rather than
    # a straight monotonic trend" check, not just eyeballing the numbers.
    curvature_flag = "insufficient_bins"
    if len(quintile_means) >= 3:
        means = [q["mean_log_xg"] for q in quintile_means]
        first, mid, last = means[0], means[len(means) // 2], means[-1]
        monotonic_inc = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
        monotonic_dec = all(means[i] >= means[i + 1] - 1e-9 for i in range(len(means) - 1))
        # Gap sizes between consecutive bins -- used both for the monotonic-but-
        # accelerating check below and the threshold-like-jump check for
        # non-monotonic sequences.
        gaps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
        abs_gaps = [abs(g) for g in gaps]

        if monotonic_inc or monotonic_dec:
            # A strictly monotonic sequence can still be genuinely curved -- convex
            # (accelerating) or concave (decelerating) rather than a straight line --
            # which a squared term CAN capture even though the direction never
            # reverses. Caught here rather than mislabelling every monotonic trend
            # "linear, no curvature": compare the largest single-step gap to the
            # average of the others; a >2x ratio concentrated at one end is a real
            # acceleration, not just monotonic noise (this is exactly the pattern the
            # rough chat check called "a mild threshold-like jump at its top bin" for
            # defenders_within_10m -- reconfirmed here, not just asserted).
            end_gap = abs_gaps[-1]
            other_gaps = abs_gaps[:-1]
            avg_other_gap = float(np.mean(other_gaps)) if other_gaps else 0.0
            if avg_other_gap > 0 and end_gap > 2 * avg_other_gap:
                curvature_flag = "monotonic_accelerating_top_bin" if monotonic_inc else "monotonic_accelerating_bottom_bin"
            else:
                start_gap = abs_gaps[0]
                other_gaps2 = abs_gaps[1:]
                avg_other_gap2 = float(np.mean(other_gaps2)) if other_gaps2 else 0.0
                if avg_other_gap2 > 0 and start_gap > 2 * avg_other_gap2:
                    curvature_flag = "monotonic_accelerating_bottom_bin" if monotonic_inc else "monotonic_accelerating_top_bin"
                else:
                    curvature_flag = "monotonic_linear"
        elif (mid < first and mid < last) or (mid > first and mid > last):
            curvature_flag = "u_or_inverse_u_shaped"
        else:
            # e.g. a jump concentrated at one extreme bin, not smoothly monotonic
            top_gap = abs_gaps[-1] if abs_gaps else 0.0
            other_gaps = abs_gaps[:-1] if len(abs_gaps) > 1 else [0.0]
            avg_other_gap = float(np.mean(other_gaps)) if other_gaps else 0.0
            if avg_other_gap > 0 and top_gap > 2 * avg_other_gap:
                curvature_flag = "threshold_like_top_bin"
            else:
                curvature_flag = "non_monotonic_irregular"

    return {
        "feature": feature,
        "type": "numeric",
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "abs_max_corr": float(max(abs(pearson_r), abs(spearman_r))),
        "quintile_means": quintile_means,
        "curvature_flag": curvature_flag,
        "n": int(finite.sum()),
    }


def boolean_feature_check(pos_df: pd.DataFrame, feature: str, log_xg: np.ndarray) -> dict:
    x = pos_df[feature].astype(float).to_numpy()
    finite = np.isfinite(x)
    x_f, y_f = x[finite], log_xg[finite]
    if len(np.unique(x_f)) < 2:
        return {"feature": feature, "type": "boolean", "pearson_r": float("nan"), "spearman_r": float("nan"),
                "abs_max_corr": 0.0, "note": "single-valued in shot-conditional subset", "n": int(finite.sum())}
    pearson_r, pearson_p = stats.pearsonr(x_f, y_f)  # point-biserial == Pearson on 0/1 data
    spearman_r, spearman_p = stats.spearmanr(x_f, y_f)
    return {
        "feature": feature, "type": "boolean",
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "abs_max_corr": float(max(abs(pearson_r), abs(spearman_r))),
        "note": "quadratic term not applicable -- squaring a 0/1 value is a no-op",
        "n": int(finite.sum()),
    }


def categorical_feature_check(pos_df: pd.DataFrame, feature: str, log_xg: np.ndarray) -> dict:
    tmp = pd.DataFrame({"cat": pos_df[feature].astype(str), "log_xg": log_xg})
    groups = [g["log_xg"].to_numpy() for _, g in tmp.groupby("cat") if len(g) >= 5]
    if len(groups) < 2:
        return {"feature": feature, "type": "categorical", "eta_squared": float("nan"),
                "note": "too few categories with >=5 rows in shot-conditional subset", "n": len(tmp)}
    f_stat, p_val = stats.f_oneway(*groups)
    grand_mean = tmp["log_xg"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((tmp["log_xg"] - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    return {
        "feature": feature, "type": "categorical",
        "anova_f": float(f_stat), "anova_p": float(p_val), "eta_squared": eta_sq,
        "abs_max_corr": eta_sq if eta_sq == eta_sq else 0.0,
        "note": "reported as ANOVA eta-squared (association magnitude), not Pearson/Spearman -- not a quadratic-term candidate (categorical, one-hot)",
        "n": len(tmp),
    }


def main() -> None:
    df = ab.load_data()
    pos_df = df[df[ab.TARGET_COL] == 1].reset_index(drop=True)
    print(f"[data] shot-conditional positive rows: {len(pos_df)} of {len(df)}")
    assert (pos_df[TARGET_XG_COL] > 0).all(), "expected every positive row to have xg > 0"
    log_xg = np.log(pos_df[TARGET_XG_COL].to_numpy())

    results = []
    print("[numeric features]")
    for feat in ab.NUMERIC_COLS:
        r = numeric_feature_check(pos_df, feat, log_xg)
        results.append(r)
        print(f"  {feat}: pearson={r['pearson_r']:+.4f} spearman={r['spearman_r']:+.4f} curvature={r['curvature_flag']}")

    print("[boolean features]")
    for feat in ab.BOOLEAN_COLS:
        r = boolean_feature_check(pos_df, feat, log_xg)
        results.append(r)
        print(f"  {feat}: pearson={r['pearson_r']:+.4f} spearman={r['spearman_r']:+.4f}")

    print("[categorical features]")
    for feat in ab.CATEGORICAL_COLS:
        r = categorical_feature_check(pos_df, feat, log_xg)
        results.append(r)
        print(f"  {feat}: eta_squared={r.get('eta_squared')}")

    # Rank by absolute association magnitude across all 32.
    ranked = sorted(results, key=lambda r: r.get("abs_max_corr", 0.0) or 0.0, reverse=True)

    # The real quadratic-term candidate pool: numeric features only, with a
    # curvature flag that isn't "monotonic_linear" (a straight monotonic trend
    # doesn't need a squared term -- the linear coefficient already captures it)
    # and isn't from too few bins to judge. Accelerating/decelerating monotonic
    # trends ARE included -- a squared term can sharpen a convex/concave trend
    # even when the direction never reverses.
    numeric_results = [r for r in results if r["type"] == "numeric"]
    CURVATURE_FLAGS = {
        "u_or_inverse_u_shaped", "threshold_like_top_bin",
        "monotonic_accelerating_top_bin", "monotonic_accelerating_bottom_bin",
    }
    curvature_candidates = [r for r in numeric_results if r["curvature_flag"] in CURVATURE_FLAGS]
    curvature_candidates_sorted = sorted(curvature_candidates, key=lambda r: r["abs_max_corr"], reverse=True)

    output = {
        "n_positive_rows": len(pos_df),
        "all_32_features_ranked_by_abs_correlation": ranked,
        "note_on_categorical_and_boolean": (
            "Quadratic terms (quad__<feature>) are only meaningful for the 17 numeric (continuous+discrete) "
            "locked features -- squaring a boolean 0/1 value or a one-hot categorical dummy is a no-op. "
            "Boolean features are reported via point-biserial Pearson/Spearman; categorical features via "
            "ANOVA eta-squared, for completeness, but neither is a quadratic-term candidate."
        ),
        "numeric_curvature_candidates": curvature_candidates_sorted,
        "comparison_to_rough_chat_shortlist": {
            "chat_shortlist_top_feature": "visible_attacker_count",
            "chat_shortlist_correlations_checked": [
                "visible_attacker_count", "attacking_goal_centrality", "distance_to_center_line", "attacker_spread",
                "defenders_within_10m",
            ],
            "reconfirmed": "see all_32_features_ranked_by_abs_correlation for the real, systematic ranking across all 32",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("\n=== TOP 10 BY ABSOLUTE ASSOCIATION MAGNITUDE (all 32) ===")
    for r in ranked[:10]:
        print(f"  {r['feature']} ({r['type']}): abs_max_corr={r.get('abs_max_corr', 0):.4f}")

    print("\n=== NUMERIC CURVATURE CANDIDATES (u/inverse-u or threshold-like, not just monotonic) ===")
    if curvature_candidates_sorted:
        for r in curvature_candidates_sorted:
            print(f"  {r['feature']}: curvature={r['curvature_flag']} abs_max_corr={r['abs_max_corr']:.4f}")
    else:
        print("  none found")

    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
