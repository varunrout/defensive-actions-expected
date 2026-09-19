"""Rung 2 of the active-continuous leg (prompt 56): c1d_random_forest.

c1c_systematic_interactions (the rung that would mirror the active-binary
leg's Rung 2, prompt 41) is explicitly SKIPPED, not forgotten, on two
independent pieces of already-existing evidence:

  1. reports/analysis/xg_target/FEATURE_INTERACTION_ANALYSIS.json already ran
     the shot-conditional panel on 5 hand-picked active-leg interaction pairs
     that showed real interaction on the binary target
     (defenders_within_10m x distance_to_attacking_box,
     visible_defender_count x attacker_spread,
     defenders_within_5m x defenders_within_10m, plus 2 more). All 5 came
     back "inconclusive" given a shot -- below the noise threshold, one by
     an order of magnitude. Decided 2026-09-17 (10/10 pairs inconclusive
     across both legs), not loosened to force a classification.
  2. A rough chat check ran the systematic version directly: all 17 numeric
     features, full pairwise interactions + squares
     (PolynomialFeatures(degree=2), 170 columns), Ridge-regularized, 5-fold
     CV on the 4,368 positive rows. Plain additive linear regression scored
     mean CV R^2=0.047; the best Ridge-regularized full-interaction
     expansion (alpha=100) reached only R^2=0.055 -- a gain smaller than the
     fold-to-fold noise (folds individually ranged 0.001-0.09), and lighter
     regularization actively made things worse (one fold went negative).

This rung goes straight to c1d_random_forest, mirroring the active-binary
leg's Rung 3 (prompt 44): does automatic interaction/curvature discovery,
which is more sample-efficient than hand-built polynomial expansion, find
more signal than c1c would have on this leg's small (4,368-row) positive
subset? Same raw (non-standardized, non-polynomial-expanded) design matrix
as the binary leg's own Rung 3 -- the point is what the model finds on its
own from the 32 locked features.

Follows train_c1b_quadratic.py's sibling-script pattern (prompt 55's own
established convention): imports train_active_continuous_baseline.py's
shared helpers rather than reimplementing them, and never touches
c0_dummy/c1_lognormal_glm/c1b_quadratic's existing artifacts or CSV rows.

Usage:
    python scripts/models/train_c1d_random_forest.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402
import train_active_continuous_baseline as acb  # noqa: E402
from dax.models.evaluation import regression_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "c1d_random_forest"
RANDOM_STATE = 42

# Trimmed for this leg's real scale: ~3,500 positive rows per training fold
# (4,368 total, 5 folds), a small fraction of the active-binary leg's own
# Rung 3 grid scale (45,408 train+val rows there). min_samples_leaf is set
# much higher than the binary leg's 5-20 floor -- a leaf of 5-20 rows out of
# ~3,500 would badly overfit this leg's already-weak, noisy signal (Rung 1
# found every feature correlates under |r|=0.16 with log(xg) given a shot).
N_ESTIMATORS_GRID = [200]
MAX_DEPTH_GRID = [4, 8, None]
MIN_SAMPLES_LEAF_GRID = [10, 30, 60]


class RFDesignMatrixBuilder:
    """Same categorical (one-hot) / boolean (passthrough) blocks as c1's
    DesignMatrixBuilder. Numeric features are median-imputed but NOT
    standardized -- trees are scale-invariant, and standardizing would only
    obscure feature_importances_. No engineered interaction/quadratic terms
    -- the whole point of this rung is what the tree finds on its own."""

    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="median")

    def fit(self, train_df: pd.DataFrame) -> "RFDesignMatrixBuilder":
        self.ohe.fit(train_df[ab.CATEGORICAL_COLS])
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(ab.CATEGORICAL_COLS))
        self.num_imputer.fit(train_df[ab.NUMERIC_COLS])
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cat_block = self.ohe.transform(df[ab.CATEGORICAL_COLS])
        cat_block = np.asarray(cat_block.todense()) if hasattr(cat_block, "todense") else np.asarray(cat_block)
        num_block = self.num_imputer.transform(df[ab.NUMERIC_COLS])
        bool_block = df[ab.BOOLEAN_COLS].astype(float).fillna(0.0).to_numpy()
        x = np.column_stack([cat_block, num_block, bool_block])
        names = [*self.cat_feature_names_, *ab.NUMERIC_COLS, *ab.BOOLEAN_COLS]
        return x, names


def fit_predict_c1d(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict):
    """Returns (log_pred, sigma2, model, (builder, feature_names)) -- same
    contract as acb.fit_predict_regression / train_c1b_quadratic's
    fit_predict_c1b, for reuse by run_cv/held-out/significance code."""
    y_train_log = np.log(train_df[acb.TARGET_XG_COL].to_numpy())
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(x_train, y_train_log)
    pred_log = model.predict(x_test)
    resid_train = y_train_log - model.predict(x_train)
    sigma2 = float(np.var(resid_train, ddof=1)) if len(resid_train) > 1 else 0.0
    train_r2 = 1.0 - np.sum(resid_train ** 2) / np.sum((y_train_log - y_train_log.mean()) ** 2)
    return pred_log, sigma2, model, (builder, feature_names), float(train_r2)


def run_cv_c1d(pos_trainval_df: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_log_pred = np.full(len(df), np.nan)
    oof_sigma2 = np.full(len(df), np.nan)
    train_r2s = []

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        log_pred, sigma2, _, _, train_r2 = fit_predict_c1d(train_df, test_df, params)
        train_r2s.append(train_r2)
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
            "train_r2": train_r2,
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
        "mean_train_r2": float(np.mean(train_r2s)),
    }
    oof_metrics.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg_all, oof_xg_naive).items()})
    oof_metrics.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg_all, oof_xg_corrected).items()})

    return {"fold_metrics": fold_df, "oof_metrics": oof_metrics}


def tune_c1d(pos_trainval_df: pd.DataFrame) -> dict:
    grid_results = []
    cv_by_params = {}
    combos = list(itertools.product(N_ESTIMATORS_GRID, MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID))
    for n_est, max_depth, min_leaf in combos:
        params = {"n_estimators": n_est, "max_depth": max_depth, "min_samples_leaf": min_leaf}
        key = (n_est, max_depth, min_leaf)
        print(f"  [tune] {params}")
        result = run_cv_c1d(pos_trainval_df, params)
        oof_common_log_rmse = result["oof_metrics"]["common_log_rmse"]
        train_r2 = result["oof_metrics"]["mean_train_r2"]
        oof_r2 = result["oof_metrics"]["corrected_r2"]
        overfit_gap = train_r2 - oof_r2
        grid_results.append({
            **params,
            "oof_common_log_rmse": oof_common_log_rmse,
            "oof_corrected_r2": oof_r2,
            "mean_train_r2": train_r2,
            "train_minus_oof_r2_gap": overfit_gap,
        })
        cv_by_params[key] = result
        print(f"    common_log_rmse={oof_common_log_rmse:.4f} oof_corrected_r2={oof_r2:.4f} "
              f"train_r2={train_r2:.4f} gap={overfit_gap:.4f}")

    # Winner chosen on OOF common-scale log RMSE (lower is better) -- NOT on
    # training-fold fit quality, precisely to avoid picking an overfit config.
    best = min(grid_results, key=lambda r: r["oof_common_log_rmse"])
    best_key = (best["n_estimators"], best["max_depth"], best["min_samples_leaf"])
    print(f"  [tune] best (by OOF common_log_rmse): {best}")
    return {"grid_results": grid_results, "best": best, "best_key": best_key, "cv_by_params": cv_by_params}


def fold_held_out_permutation_importance(pos_trainval_df: pd.DataFrame, best_params: dict) -> list[dict]:
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    importances_by_feature: dict[str, list[float]] = {}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train_log = np.log(train_df[acb.TARGET_XG_COL].to_numpy())
        y_test_log = np.log(test_df[acb.TARGET_XG_COL].to_numpy())

        builder = RFDesignMatrixBuilder().fit(train_df)
        x_train, feature_names = builder.transform(train_df)
        x_test, _ = builder.transform(test_df)
        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(x_train, y_train_log)

        result = permutation_importance(model, x_test, y_test_log, scoring="neg_mean_squared_error",
                                         n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        for name, mean_val in zip(feature_names, result.importances_mean):
            importances_by_feature.setdefault(name, []).append(float(mean_val))

    return sorted(
        ({"feature": name, "mean_permutation_importance": float(np.mean(vals)), "n_folds": len(vals)}
         for name, vals in importances_by_feature.items()),
        key=lambda r: r["mean_permutation_importance"], reverse=True,
    )


def main() -> None:
    acb.REG_DIR.mkdir(parents=True, exist_ok=True)
    acb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    acb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    acb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = ab.load_data()
    load_canonical_split()
    pos_all = df_all[df_all[ab.TARGET_COL] == 1].reset_index(drop=True)
    print(f"[data] shot-conditional positive rows: {len(pos_all)} of {len(df_all)}")

    pos_test_mask = canonical_test_mask(pos_all, group_col=ab.GROUP_COL)
    pos_trainval = pos_all.loc[~pos_test_mask].reset_index(drop=True)
    pos_test = pos_all.loc[pos_test_mask].reset_index(drop=True)
    print(f"  positive trainval rows: {len(pos_trainval)}; held-out test rows: {len(pos_test)}")

    print(f"[tune] grid-searching RF hyperparameters ({len(N_ESTIMATORS_GRID) * len(MAX_DEPTH_GRID) * len(MIN_SAMPLES_LEAF_GRID)} combos)...")
    tuning = tune_c1d(pos_trainval)
    best_params = {k: tuning["best"][k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
    cv_result = tuning["cv_by_params"][tuning["best_key"]]

    comparison_row = {"variant": VARIANT, "n_features": None}
    comparison_row.update({f"rf_{k}": v for k, v in best_params.items()})
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(pos_trainval)
    comparison_row["matches"] = pos_trainval[ab.GROUP_COL].nunique()

    print("[significance] c1d_random_forest vs c1_lognormal_glm (common-scale log RMSE per fold)...")
    c1_cv = acb.run_cv_regression(pos_trainval, "c1_lognormal_glm")
    fold_common_log_rmse = {
        "c1d_random_forest": cv_result["fold_metrics"]["common_log_rmse"].tolist(),
        "c1_lognormal_glm": c1_cv["fold_metrics"]["common_log_rmse"].tolist(),
    }
    a = np.array(fold_common_log_rmse["c1d_random_forest"])
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
        "mean_diff_c1d_minus_c1": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "caution_note": (
            "5 folds only, ~875 positive rows per fold (this leg's much smaller sample than the binary "
            "legs) -- read this test as directionally informative, not as strong statistical proof, the "
            "same honest caveat given to every small-n significance test elsewhere on this leg."
        ),
        "hyperparameter_grid_search": tuning["grid_results"],
        "chosen_params": best_params,
    }
    (acb.VALIDATION_DIR / "significance_c1_vs_c1d_random_forest.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (c1d-c1) common_log_rmse: {sig['mean_diff_c1d_minus_c1']:+.4f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("[held-out] c1d_random_forest (fit once on all positive trainval rows)...")
    log_pred, sigma2, model, extra, _ = fit_predict_c1d(pos_trainval, pos_test, best_params)
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

    gini = sorted(
        ({"feature": n, "gini_importance": float(v)} for n, v in zip(feature_names, model.feature_importances_)),
        key=lambda r: r["gini_importance"], reverse=True,
    )
    print("[permutation importance] computing on fold-held-out portions (5 folds, not final test)...")
    perm_importance = fold_held_out_permutation_importance(pos_trainval, best_params)

    importance_output = {
        "variant": VARIANT, "best_params": best_params,
        "fitted_on": "all positive-row train+val rows (canonical split); held-out test used once for readout only",
        "top15_gini_importance": gini[:15],
        "top15_permutation_importance_fold_held_out": perm_importance[:15],
        "full_gini_importance": gini,
        "full_permutation_importance_fold_held_out": perm_importance,
    }
    (acb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")

    # Feature-importance chart (in place of the coefficient chart c0/c1/c1b use).
    from dax.models.diagnostics import _ACCENT, _new_axes, _save_fixed, ensure_dir
    chart_dir = acb.REG_CHARTS_DIR / VARIANT
    ensure_dir(chart_dir)

    fig, ax = _new_axes()
    top = gini[:15]
    ax.barh([r["feature"] for r in top][::-1], [r["gini_importance"] for r in top][::-1], color=_ACCENT)
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 15 feature importance (Gini)")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")

    acb.save_regression_variant_charts(chart_dir, y_true_log, log_pred, y_true_xg, xg_corrected, calib_rows, None)
    # save_regression_variant_charts writes its own "no features" placeholder for
    # feature_coefficients.png when coef_table is None -- overwrite it with the
    # real importance chart written above (same filename, kept for chart-set
    # consistency across every rung on this leg).
    fig, ax = _new_axes()
    ax.barh([r["feature"] for r in top][::-1], [r["gini_importance"] for r in top][::-1], color=_ACCENT)
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 15 feature importance (Gini)")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")

    joblib.dump(model, acb.REG_DIR / f"{VARIANT}.joblib")

    print(f"  held-out: log_rmse={held_out_row['log_rmse']:.4f} common_log_rmse={held_out_row['common_log_rmse']:.4f} "
          f"naive_mae={held_out_row['naive_mae']:.4f} corrected_mae={held_out_row['corrected_mae']:.4f} "
          f"naive_r2={held_out_row['naive_r2']:.4f} corrected_r2={held_out_row['corrected_r2']:.4f}")

    print("[csv] appending c1d_random_forest rows (additive only)...")
    acb.append_to_csv(acb.COMPARISONS_DIR / "active_continuous_baseline_comparison.csv", [comparison_row])
    acb.append_to_csv(acb.COMPARISONS_DIR / "active_continuous_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== HYPERPARAMETER GRID (chosen by OOF common_log_rmse) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if (r["n_estimators"], r["max_depth"], r["min_samples_leaf"]) == tuning["best_key"] else ""
        print(f"  n_est={r['n_estimators']} depth={r['max_depth']} leaf={r['min_samples_leaf']} "
              f"common_log_rmse={r['oof_common_log_rmse']:.4f} oof_r2={r['oof_corrected_r2']:.4f} "
              f"train_r2={r['mean_train_r2']:.4f} gap={r['train_minus_oof_r2_gap']:.4f}{marker}")
    print("\n=== TOP 15 GINI IMPORTANCE ===")
    for r in gini[:15]:
        print(f"  {r['feature']}: {r['gini_importance']:.4f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
