"""Rung 1 of the active-continuous leg (prompt 55): c1b_quadratic.

c1_lognormal_glm (Rung 0, prompt 54) beats c0_dummy in 5/5 CV folds on the
common-scale metric, but the starting point is genuinely weak: a systematic
check of all 32 locked ACTIVE features against log(xg) on the shot-conditional
subset (target_future_shot_10s==1, n=4,368 -- see
scripts/analysis/check_shot_conditional_nonlinearity.py and its output,
outputs/models/validation/shot_conditional_nonlinearity_check.json) found
every feature correlates weakly once conditioned on a shot occurring -- the
strongest, visible_attacker_count, tops out at |r|=0.16.

This is a DIFFERENT question from the active-binary leg's Rung 1 (prompt 39),
which found 4 clean U-shaped features on the FULL dataset (does a shot happen
at all). Those 4 features and that reasoning are NOT reused here -- this
leg's own shot-conditional check is the only basis for candidate selection,
per the explicit instruction not to assume the active-binary leg's shapes
transfer.

The systematic check's real findings (not the 7-feature rough chat
shortlist, reconfirmed and extended to all 32):
  - visible_attacker_count (|r|=0.161, strongest feature overall) and
    defenders_within_10m (|r|=0.088, the specific feature the rough chat
    check flagged) both show a genuine accelerating-monotonic pattern:
    strictly increasing/decreasing bin-to-bin, but with a jump at the top
    quintile bin more than 2x the size of the other gaps -- real curvature
    within an overall monotonic trend, not a straight line, and not a
    binning artifact (both are well above the noise floor of the weaker
    candidates).
  - defender_spread (|r|=0.052) shows a genuine, if weaker and more
    asymmetric, U-shape: log(xg) drops from bin 1 to bin 4, then partially
    recovers at bin 5. Weaker evidence than the top two, included as the
    marginal case it is, not oversold.
  - 7 other numeric features were also flagged "curved" by the same
    quintile-binning diagnostic, but every one of them has |r| < 0.05 (as
    low as 0.005) -- indistinguishable from noise on quintile bins of
    ~870-900 rows each drawn from a target with excess kurtosis ~13-14.
    These are NOT used as quadratic candidates; forcing quadratic terms onto
    features with no real correlation just to pad out this rung's candidate
    list is exactly what this prompt was told not to do.

QUADRATIC_FEATURES_C1B = 3 features (visible_attacker_count,
defenders_within_10m, defender_spread), not a fixed count, not the active
leg's 4 -- this leg's own real data determined the count and the members.

Usage:
    python scripts/models/train_c1b_quadratic.py
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

import train_active_binary_baseline as ab  # noqa: E402
import train_active_continuous_baseline as acb  # noqa: E402
from dax.models.evaluation import regression_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "c1b_quadratic"

# Reconfirmed against this leg's own shot-conditional data (see module
# docstring and outputs/models/validation/shot_conditional_nonlinearity_check.json)
# -- NOT the active-binary leg's 4 U-shaped features.
QUADRATIC_FEATURES_C1B = ["visible_attacker_count", "defenders_within_10m", "defender_spread"]
for _qf in QUADRATIC_FEATURES_C1B:
    assert _qf in ab.NUMERIC_COLS, f"{_qf} must be one of the locked numeric ACTIVE features"


def fit_predict_c1b(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Same as acb.fit_predict_regression's c1 branch, but with the
    quadratic-feature-extended DesignMatrixBuilder. Mirrors that function's
    contract: returns (log_pred, sigma2, model, (builder, feature_names))."""
    y_train_log = np.log(train_df[acb.TARGET_XG_COL].to_numpy())

    builder = ab.DesignMatrixBuilder(
        add_interactions=False, add_quadratic=True, quadratic_features=QUADRATIC_FEATURES_C1B
    ).fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    model = LinearRegression()
    model.fit(x_train, y_train_log)
    pred_log = model.predict(x_test)
    resid = y_train_log - model.predict(x_train)
    sigma2 = float(np.var(resid, ddof=1)) if len(resid) > 1 else 0.0
    return pred_log, sigma2, model, (builder, feature_names)


def run_cv_c1b(pos_trainval_df: pd.DataFrame) -> dict:
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_log_pred = np.full(len(df), np.nan)
    oof_sigma2 = np.full(len(df), np.nan)

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        log_pred, sigma2, _, _ = fit_predict_c1b(train_df, test_df)
        oof_log_pred[test_mask] = log_pred
        oof_sigma2[test_mask] = sigma2

        y_true_log = np.log(test_df[acb.TARGET_XG_COL].to_numpy())
        y_true_xg = test_df[acb.TARGET_XG_COL].to_numpy()
        log_resid = log_pred - y_true_log
        xg_naive = acb.backtransform(log_pred, sigma2, "naive")
        xg_corrected = acb.backtransform(log_pred, sigma2, "lognormal_corrected")
        common_log_resid = np.log(xg_naive) - np.log(y_true_xg)

        m = {
            "fold": int(fold_id), "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "log_rmse": float(np.sqrt(np.mean(log_resid ** 2))), "log_mae": float(np.mean(np.abs(log_resid))),
            "common_log_rmse": float(np.sqrt(np.mean(common_log_resid ** 2))),
            "common_log_mae": float(np.mean(np.abs(common_log_resid))),
        }
        m.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg, xg_naive).items()})
        m.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg, xg_corrected).items()})
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)

    y_true_log_all = np.log(df[acb.TARGET_XG_COL].to_numpy())
    oof_log_resid = oof_log_pred - y_true_log_all
    oof_xg_naive = acb.backtransform(oof_log_pred, oof_sigma2, "naive")
    oof_xg_corrected = acb.backtransform(oof_log_pred, oof_sigma2, "lognormal_corrected")
    y_true_xg_all = df[acb.TARGET_XG_COL].to_numpy()
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
    acb.REG_DIR.mkdir(parents=True, exist_ok=True)
    acb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    acb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    acb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[quadratic features] {QUADRATIC_FEATURES_C1B}")
    df_all = ab.load_data()
    load_canonical_split()
    pos_all = df_all[df_all[ab.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask = canonical_test_mask(pos_all, group_col=ab.GROUP_COL)
    pos_trainval = pos_all.loc[~pos_test_mask].reset_index(drop=True)
    pos_test = pos_all.loc[pos_test_mask].reset_index(drop=True)

    print("[cv] c1b_quadratic...")
    cv_result = run_cv_c1b(pos_trainval)
    comparison_row = {"variant": VARIANT, "n_features": 32 + len(QUADRATIC_FEATURES_C1B)}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(pos_trainval)
    comparison_row["matches"] = pos_trainval[ab.GROUP_COL].nunique()

    print("[significance] c1b_quadratic vs c1_lognormal_glm (common-scale log RMSE per fold)...")
    c1_cv = acb.run_cv_regression(pos_trainval, "c1_lognormal_glm")
    fold_common_log_rmse = {
        "c1b_quadratic": cv_result["fold_metrics"]["common_log_rmse"].tolist(),
        "c1_lognormal_glm": c1_cv["fold_metrics"]["common_log_rmse"].tolist(),
    }
    a = np.array(fold_common_log_rmse["c1b_quadratic"])
    b = np.array(fold_common_log_rmse["c1_lognormal_glm"])
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    sig = {
        "metric": "common_log_rmse (per fold, natural-log(xg) scale, lower is better)",
        "fold_common_log_rmse": fold_common_log_rmse,
        "mean_diff_c1b_minus_c1": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "caution_note": (
            "5 folds only, ~875 positive rows per fold (this leg's much smaller sample than the binary "
            "legs) -- read this test as directionally informative, not as strong statistical proof, the "
            "same honest caveat given to every small-n significance test elsewhere in this project."
        ),
        "quadratic_features_used": QUADRATIC_FEATURES_C1B,
    }
    (acb.VALIDATION_DIR / "significance_c1_vs_c1b_quadratic.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (c1b-c1) common_log_rmse: {sig['mean_diff_c1b_minus_c1']:+.4f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("[held-out] c1b_quadratic (fit once on all positive trainval rows)...")
    log_pred, sigma2, model, extra = fit_predict_c1b(pos_trainval, pos_test)
    builder, feature_names = extra
    y_true_log = np.log(pos_test[acb.TARGET_XG_COL].to_numpy())
    y_true_xg = pos_test[acb.TARGET_XG_COL].to_numpy()
    log_resid = log_pred - y_true_log
    xg_naive = acb.backtransform(log_pred, sigma2, "naive")
    xg_corrected = acb.backtransform(log_pred, sigma2, "lognormal_corrected")
    common_log_resid = np.log(xg_naive) - np.log(y_true_xg)

    held_out_row = {
        "variant": VARIANT, "readout": "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION",
        "rows": len(pos_test), "matches": pos_test[ab.GROUP_COL].nunique(),
        "log_rmse": float(np.sqrt(np.mean(log_resid ** 2))), "log_mae": float(np.mean(np.abs(log_resid))),
        "common_log_rmse": float(np.sqrt(np.mean(common_log_resid ** 2))),
        "common_log_mae": float(np.mean(np.abs(common_log_resid))),
    }
    held_out_row.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg, xg_naive).items()})
    held_out_row.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg, xg_corrected).items()})

    calib_rows = acb.calibration_bins(y_true_xg, xg_corrected)

    coefs = model.coef_
    coef_table = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coeff_json = {
        "variant": VARIANT, "intercept": float(model.intercept_), "n_features": len(feature_names),
        "quadratic_features": QUADRATIC_FEATURES_C1B,
        "coefficients": coef_table.to_dict(orient="records"),
        "fitted_on": "all positive-row train+val rows (canonical split); held-out test used once for readout only",
    }
    (acb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(coeff_json, indent=2), encoding="utf-8")

    acb.save_regression_variant_charts(
        acb.REG_CHARTS_DIR / VARIANT, y_true_log, log_pred, y_true_xg, xg_corrected, calib_rows, coef_table
    )
    joblib.dump(model, acb.REG_DIR / f"{VARIANT}.joblib")

    print(f"  held-out: log_rmse={held_out_row['log_rmse']:.4f} common_log_rmse={held_out_row['common_log_rmse']:.4f} "
          f"naive_mae={held_out_row['naive_mae']:.4f} corrected_mae={held_out_row['corrected_mae']:.4f} "
          f"naive_r2={held_out_row['naive_r2']:.4f} corrected_r2={held_out_row['corrected_r2']:.4f}")

    print("[csv] appending c1b_quadratic rows (additive only -- c0/c1 rows untouched)...")
    acb.append_to_csv(acb.COMPARISONS_DIR / "active_continuous_baseline_comparison.csv", [comparison_row])
    acb.append_to_csv(acb.COMPARISONS_DIR / "active_continuous_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== TOP SURVIVING QUADRATIC TERM COEFFICIENTS ===")
    quad_rows = coef_table[coef_table["feature"].str.startswith("quad__")]
    for _, r in quad_rows.iterrows():
        print(f"  {r['feature']}: {r['coef']:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
