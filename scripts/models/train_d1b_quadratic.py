"""Rung 1 of the passive-continuous leg (prompt 61): d1b_quadratic.

d1_lognormal_glm (Rung 0, prompts 59-60) beats d0_dummy in 5/5 CV folds on
the common-scale metric, but the starting point is genuinely weaker than the
active-continuous leg's own Rung 1 (prompt 55): a systematic check of all 38
locked PASSIVE features against log(xg) on the shot-conditional subset
(target_future_shot_10s==1, n=95,048 -- see
scripts/analysis/check_shot_conditional_nonlinearity_passive.py and its
output, outputs/models/validation/shot_conditional_nonlinearity_check_passive.json)
found every feature correlates very weakly once conditioned on a shot
occurring -- the strongest, lane_screening_score_option_1, tops out at
|r|=0.065, less than half the active leg's own strongest
(visible_attacker_count, |r|=0.16). The likely reason, not assumed but
consistent with the pattern: this leg's rows describe ONE defender's
individual geometry relative to a single event, not the aggregated
whole-defense picture the active leg's features capture -- a lone
defender's positioning plausibly has less individual leverage over
eventual shot quality than the attacking side's own aggregate spatial
features do.

The systematic check's real finding, and it is a narrow one:

  - The 3 strongest-correlated numeric features overall
    (lane_screening_score_option_1 |r|=0.065, lane_screening_score_option_2
    |r|=0.063, lane_screening_score_option_3 |r|=0.048) are ALL flagged
    "monotonic_linear" by the quintile-binning curvature diagnostic -- a
    straight trend, no curvature a squared term would add beyond what the
    linear term already captures. Being top-correlated does not make them
    quadratic-term candidates; the diagnostic says plainly they are not.
  - engagement_distance_to_carrier (|r|=0.049, rank #4 by correlation) is
    the ONLY numeric feature that combines (a) a correlation magnitude in
    the same range as the top 3 and (b) a genuine, non-monotonic-linear
    curvature flag ("monotonic_accelerating_top_bin" -- a real
    diminishing-returns/concave shape: quintile means -2.764, -2.828,
    -2.874, -2.905, -2.919, with the largest single-step gap concentrated
    at the low-x end, >2x the average of the other gaps, on ~19,010 rows
    per quintile -- large enough that this is not a binning artifact).
  - Every OTHER numeric feature flagged with real curvature by the same
    diagnostic has |r| well under 0.04 (many under 0.02) -- indistinguishable
    from noise on a target this weak, and not used as quadratic candidates.

Given how weak this leg's starting signal already is, this rung does NOT
force a multi-feature quadratic set the way the active leg's own Rung 1
did (3 features). Only ONE feature clears both bars (real correlation
magnitude AND real curvature): engagement_distance_to_carrier. This is
explicitly the "pick the single best candidate for a minimal test" case,
not a watered-down version of the active leg's approach.

QUADRATIC_FEATURES_D1B = 1 feature (engagement_distance_to_carrier) -- not
the active leg's 3, not any of the active leg's own quadratic candidates
(which mostly don't even exist in the passive feature set), determined
entirely by this leg's own shot-conditional data.

Usage:
    python scripts/models/train_d1b_quadratic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_continuous_baseline as pcb  # noqa: E402
from dax.models.evaluation import regression_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "d1b_quadratic"
EVENT_COL = "event_id"

# Reconfirmed against this leg's own shot-conditional data (see module
# docstring and outputs/models/validation/shot_conditional_nonlinearity_check_passive.json)
# -- NOT the active-continuous leg's 3 quadratic candidates.
QUADRATIC_FEATURES_D1B = ["engagement_distance_to_carrier"]
for _qf in QUADRATIC_FEATURES_D1B:
    assert _qf in pb.NUMERIC_COLS, f"{_qf} must be one of the locked numeric PASSIVE features"


def fit_predict_d1b(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Same as pcb.fit_predict_regression's d1 branch, but with the
    quadratic-feature-extended DesignMatrixBuilder (which also carries
    forward the prompt-60 defender_functional_role_unclassified fix --
    reused exactly, not reimplemented). Mirrors that function's contract:
    returns (log_pred, sigma2, model, (builder, feature_names))."""
    y_train_log = np.log(train_df[pcb.TARGET_XG_COL].to_numpy())

    builder = pb.DesignMatrixBuilder(
        add_interactions=False, add_quadratic=True, quadratic_features=QUADRATIC_FEATURES_D1B
    ).fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    model = LinearRegression()
    model.fit(x_train, y_train_log)
    pred_log = model.predict(x_test)
    resid = y_train_log - model.predict(x_train)
    sigma2 = float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0
    return pred_log, sigma2, model, (builder, feature_names)


def run_cv_d1b(pos_trainval_df: pd.DataFrame) -> dict:
    folds = canonical_grouped_folds(pos_trainval_df, group_col=pb.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_log_pred = np.full(len(df), np.nan)
    oof_sigma2 = np.full(len(df), np.nan)

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        log_pred, sigma2, _, _ = fit_predict_d1b(train_df, test_df)
        oof_log_pred[test_mask] = log_pred
        oof_sigma2[test_mask] = sigma2

        y_true_log = np.log(test_df[pcb.TARGET_XG_COL].to_numpy())
        y_true_xg = test_df[pcb.TARGET_XG_COL].to_numpy()
        log_resid = log_pred - y_true_log
        xg_naive = pcb.backtransform(log_pred, sigma2, "naive")
        xg_corrected = pcb.backtransform(log_pred, sigma2, "lognormal_corrected")
        common_log_resid = np.log(xg_naive) - np.log(y_true_xg)

        m = {
            "fold": int(fold_id), "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "n_train_events": int(train_df[EVENT_COL].nunique()), "n_test_events": int(test_df[EVENT_COL].nunique()),
            "log_rmse": float(np.sqrt(np.mean(log_resid ** 2))), "log_mae": float(np.mean(np.abs(log_resid))),
            "common_log_rmse": float(np.sqrt(np.mean(common_log_resid ** 2))),
            "common_log_mae": float(np.mean(np.abs(common_log_resid))),
        }
        m.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg, xg_naive).items()})
        m.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg, xg_corrected).items()})
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)

    y_true_log_all = np.log(df[pcb.TARGET_XG_COL].to_numpy())
    oof_log_resid = oof_log_pred - y_true_log_all
    oof_xg_naive = pcb.backtransform(oof_log_pred, oof_sigma2, "naive")
    oof_xg_corrected = pcb.backtransform(oof_log_pred, oof_sigma2, "lognormal_corrected")
    y_true_xg_all = df[pcb.TARGET_XG_COL].to_numpy()
    oof_common_log_resid = np.log(oof_xg_naive) - np.log(y_true_xg_all)

    oof_metrics = {
        "log_rmse": float(np.sqrt(np.mean(oof_log_resid ** 2))), "log_mae": float(np.mean(np.abs(oof_log_resid))),
        "common_log_rmse": float(np.sqrt(np.mean(oof_common_log_resid ** 2))),
        "common_log_mae": float(np.mean(np.abs(oof_common_log_resid))),
    }
    oof_metrics.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg_all, oof_xg_naive).items()})
    oof_metrics.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg_all, oof_xg_corrected).items()})

    return {"fold_metrics": fold_df, "oof_metrics": oof_metrics}


def main() -> None:
    pcb.REG_DIR.mkdir(parents=True, exist_ok=True)
    pcb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    pcb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    pcb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[quadratic features] {QUADRATIC_FEATURES_D1B}")
    df_all = pb.load_data()
    load_canonical_split()
    pos_all = df_all[df_all[pb.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask = canonical_test_mask(pos_all, group_col=pb.GROUP_COL)
    pos_trainval = pos_all.loc[~pos_test_mask].reset_index(drop=True)
    pos_test = pos_all.loc[pos_test_mask].reset_index(drop=True)

    print("[cv] d1b_quadratic...")
    cv_result = run_cv_d1b(pos_trainval)
    comparison_row = {"variant": VARIANT, "n_features": 38 + len(QUADRATIC_FEATURES_D1B)}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(pos_trainval)
    comparison_row["matches"] = pos_trainval[pb.GROUP_COL].nunique()
    comparison_row["events"] = pos_trainval[EVENT_COL].nunique()

    print("[significance] d1b_quadratic vs d1_lognormal_glm (common-scale log RMSE per fold)...")
    d1_cv = pcb.run_cv_regression(pos_trainval, "d1_lognormal_glm")
    fold_common_log_rmse = {
        "d1b_quadratic": cv_result["fold_metrics"]["common_log_rmse"].tolist(),
        "d1_lognormal_glm": d1_cv["fold_metrics"]["common_log_rmse"].tolist(),
    }
    fold_events = cv_result["fold_metrics"][["fold", "n_test", "n_test_events"]].to_dict(orient="records")
    a = np.array(fold_common_log_rmse["d1b_quadratic"])
    b = np.array(fold_common_log_rmse["d1_lognormal_glm"])
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    min_test_events = min(r["n_test_events"] for r in fold_events)
    sig = {
        "metric": "common_log_rmse (per fold, natural-log(xg) scale, lower is better)",
        "fold_common_log_rmse": fold_common_log_rmse,
        "fold_test_rows_and_events": fold_events,
        "mean_diff_d1b_minus_d1": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "effective_sample_size_note": (
            f"Per-fold row counts (thousands) are large, but the effective sample size is closer to "
            f"the per-fold unique EVENT count (minimum across folds: {min_test_events}), since every "
            "defender-slot row sharing an event_id carries the identical target value -- same "
            "standing caveat as significance_d0_vs_d1.json (prompt 59)."
        ),
        "quadratic_features_used": QUADRATIC_FEATURES_D1B,
    }
    (pcb.VALIDATION_DIR / "significance_d1_vs_d1b_quadratic.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (d1b-d1) common_log_rmse: {sig['mean_diff_d1b_minus_d1']:+.4f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")
    print(f"  min per-fold test-event count: {min_test_events}")

    print("[held-out] d1b_quadratic (fit once on all positive trainval rows)...")
    log_pred, sigma2, model, extra = fit_predict_d1b(pos_trainval, pos_test)
    builder, feature_names = extra
    y_true_log = np.log(pos_test[pcb.TARGET_XG_COL].to_numpy())
    y_true_xg = pos_test[pcb.TARGET_XG_COL].to_numpy()
    log_resid = log_pred - y_true_log
    xg_naive = pcb.backtransform(log_pred, sigma2, "naive")
    xg_corrected = pcb.backtransform(log_pred, sigma2, "lognormal_corrected")
    common_log_resid = np.log(xg_naive) - np.log(y_true_xg)

    held_out_row = {
        "variant": VARIANT, "readout": "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION",
        "rows": len(pos_test), "matches": pos_test[pb.GROUP_COL].nunique(),
        "events": pos_test[EVENT_COL].nunique(),
        "log_rmse": float(np.sqrt(np.mean(log_resid ** 2))), "log_mae": float(np.mean(np.abs(log_resid))),
        "common_log_rmse": float(np.sqrt(np.mean(common_log_resid ** 2))),
        "common_log_mae": float(np.mean(np.abs(common_log_resid))),
    }
    held_out_row.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg, xg_naive).items()})
    held_out_row.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg, xg_corrected).items()})

    calib_rows = pcb.calibration_bins(y_true_xg, xg_corrected)

    coefs = model.coef_
    coef_table = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coeff_json = {
        "variant": VARIANT, "intercept": float(model.intercept_), "n_features": len(feature_names),
        "quadratic_features": QUADRATIC_FEATURES_D1B,
        "coefficients": coef_table.to_dict(orient="records"),
        "fitted_on": "all positive-row train+val rows (canonical split); held-out test used once for readout only",
    }
    (pcb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(coeff_json, indent=2), encoding="utf-8")

    pcb.save_regression_variant_charts(
        pcb.REG_CHARTS_DIR / VARIANT, y_true_log, log_pred, y_true_xg, xg_corrected, calib_rows, coef_table
    )
    joblib.dump(model, pcb.REG_DIR / f"{VARIANT}.joblib")

    print(f"  held-out: log_rmse={held_out_row['log_rmse']:.4f} common_log_rmse={held_out_row['common_log_rmse']:.4f} "
          f"naive_mae={held_out_row['naive_mae']:.4f} corrected_mae={held_out_row['corrected_mae']:.4f} "
          f"naive_r2={held_out_row['naive_r2']:.4f} corrected_r2={held_out_row['corrected_r2']:.4f}")

    print("[csv] appending d1b_quadratic rows (additive only -- d0/d1 rows untouched)...")
    pcb.append_to_csv(pcb.COMPARISONS_DIR / "passive_continuous_baseline_comparison.csv", [comparison_row])
    pcb.append_to_csv(pcb.COMPARISONS_DIR / "passive_continuous_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== TOP SURVIVING QUADRATIC TERM COEFFICIENTS ===")
    quad_rows = coef_table[coef_table["feature"].str.startswith("quad__")]
    for _, r in quad_rows.iterrows():
        print(f"  {r['feature']}: {r['coef']:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
