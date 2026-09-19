"""Reconfirms the rough chat check of shot-conditional nonlinearity for the
passive-continuous leg's Rung 1 (prompt 61), extended to all 38 locked
PASSIVE features -- mirrors scripts/analysis/check_shot_conditional_nonlinearity.py's
own method for the active leg (prompt 55) exactly, including the same
curvature-classification logic (fixed in prompt 55: distinguishes a straight
monotonic trend from a monotonic-but-accelerating one, not just "monotonic").

The rough chat check only looked at the 26 continuous features (missed the 2
discrete ones) and found every correlation weak: strongest
lane_screening_score_option_1 (pearson +0.057, spearman +0.065), followed by
lane_screening_score_option_2 (+0.053/+0.063) and
engagement_distance_to_carrier (-0.049/-0.049); everything else under 0.04,
overload_score (discrete) near zero (+0.006). This script reproduces that
properly against the real data, across all 28 locked numeric (26 continuous
+ 2 discrete) PASSIVE features plus the 6 boolean and 4 categorical features
for completeness, so Rung 1's candidate quadratic-term selection is based on
a real systematic check, not the chat shortlist assumed unchanged.

Restricted to the 95,048 rows where target_future_shot_10s==1 (all
defender-slot rows, per this leg's standing row-duplication decision -- not
deduped or collapsed here either).

Usage:
    python scripts/analysis/check_shot_conditional_nonlinearity_passive.py
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

import train_passive_binary_baseline as pb  # noqa: E402

OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "shot_conditional_nonlinearity_check_passive.json"
TARGET_XG_COL = "target_future_xg_10s"
N_QUINTILES = 5


def numeric_feature_check(pos_df: pd.DataFrame, feature: str, log_xg: np.ndarray) -> dict:
    x = pos_df[feature].to_numpy(dtype=float)
    finite = np.isfinite(x)
    x_f, y_f = x[finite], log_xg[finite]

    pearson_r, pearson_p = stats.pearsonr(x_f, y_f)
    spearman_r, spearman_p = stats.spearmanr(x_f, y_f)

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

    curvature_flag = "insufficient_bins"
    if len(quintile_means) >= 3:
        means = [q["mean_log_xg"] for q in quintile_means]
        first, mid, last = means[0], means[len(means) // 2], means[-1]
        monotonic_inc = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
        monotonic_dec = all(means[i] >= means[i + 1] - 1e-9 for i in range(len(means) - 1))
        gaps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
        abs_gaps = [abs(g) for g in gaps]

        if monotonic_inc or monotonic_dec:
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
            top_gap = abs_gaps[-1] if abs_gaps else 0.0
            other_gaps = abs_gaps[:-1] if len(abs_gaps) > 1 else [0.0]
            avg_other_gap = float(np.mean(other_gaps)) if other_gaps else 0.0
            if avg_other_gap > 0 and top_gap > 2 * avg_other_gap:
                curvature_flag = "threshold_like_top_bin"
            else:
                curvature_flag = "non_monotonic_irregular"

    return {
        "feature": feature, "type": "numeric",
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
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
    pearson_r, pearson_p = stats.pearsonr(x_f, y_f)
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
    df = pb.load_data()
    pos_df = df[df[pb.TARGET_COL] == 1].reset_index(drop=True)
    print(f"[data] shot-conditional positive rows: {len(pos_df)} of {len(df)} "
          f"({pos_df['event_id'].nunique()} unique events)")
    assert (pos_df[TARGET_XG_COL] > 0).all(), "expected every positive row to have xg > 0"
    log_xg = np.log(pos_df[TARGET_XG_COL].to_numpy())

    results = []
    print(f"[numeric features] ({len(pb.NUMERIC_COLS)}: {len(pb.CONTINUOUS_COLS)} continuous + {len(pb.DISCRETE_COLS)} discrete)")
    for feat in pb.NUMERIC_COLS:
        r = numeric_feature_check(pos_df, feat, log_xg)
        results.append(r)
        print(f"  {feat}: pearson={r['pearson_r']:+.4f} spearman={r['spearman_r']:+.4f} curvature={r['curvature_flag']}")

    print(f"[boolean features] ({len(pb.BOOLEAN_COLS)})")
    for feat in pb.BOOLEAN_COLS:
        r = boolean_feature_check(pos_df, feat, log_xg)
        results.append(r)
        print(f"  {feat}: pearson={r['pearson_r']:+.4f} spearman={r['spearman_r']:+.4f}")

    print(f"[categorical features] ({len(pb.CATEGORICAL_COLS)})")
    for feat in pb.CATEGORICAL_COLS:
        r = categorical_feature_check(pos_df, feat, log_xg)
        results.append(r)
        print(f"  {feat}: eta_squared={r.get('eta_squared')}")

    ranked = sorted(results, key=lambda r: r.get("abs_max_corr", 0.0) or 0.0, reverse=True)

    numeric_results = [r for r in results if r["type"] == "numeric"]
    CURVATURE_FLAGS = {
        "u_or_inverse_u_shaped", "threshold_like_top_bin",
        "monotonic_accelerating_top_bin", "monotonic_accelerating_bottom_bin",
    }
    curvature_candidates = [r for r in numeric_results if r["curvature_flag"] in CURVATURE_FLAGS]
    curvature_candidates_sorted = sorted(curvature_candidates, key=lambda r: r["abs_max_corr"], reverse=True)

    output = {
        "n_positive_rows": len(pos_df),
        "n_positive_unique_events": int(pos_df["event_id"].nunique()),
        "all_38_features_ranked_by_abs_correlation": ranked,
        "note_on_categorical_and_boolean": (
            "Quadratic terms (quad__<feature>) are only meaningful for the 28 numeric (26 continuous + "
            "2 discrete) locked features -- squaring a boolean 0/1 value or a one-hot categorical dummy "
            "is a no-op. Boolean features are reported via point-biserial Pearson/Spearman; categorical "
            "features via ANOVA eta-squared, for completeness, but neither is a quadratic-term candidate."
        ),
        "numeric_curvature_candidates": curvature_candidates_sorted,
        "comparison_to_rough_chat_shortlist": {
            "chat_shortlist_top_3": [
                "lane_screening_score_option_1", "lane_screening_score_option_2", "engagement_distance_to_carrier",
            ],
            "chat_shortlist_correlations_reported": {
                "lane_screening_score_option_1": {"pearson": 0.057, "spearman": 0.065},
                "lane_screening_score_option_2": {"pearson": 0.053, "spearman": 0.063},
                "engagement_distance_to_carrier": {"pearson": -0.049, "spearman": -0.049},
                "overload_score": {"pearson": 0.006},
            },
            "chat_check_gap": "chat check only covered the 26 continuous features, missed both discrete features (overload_score, defender_slot_index) -- this script covers all 28 numeric features",
            "reconfirmed": "see all_38_features_ranked_by_abs_correlation for the real, systematic ranking across all 38",
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("\n=== TOP 10 BY ABSOLUTE ASSOCIATION MAGNITUDE (all 38) ===")
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
