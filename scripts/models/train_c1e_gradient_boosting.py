"""Rung 3 of the active-continuous leg (prompt 57): c1e_gradient_boosting.

c1d_random_forest (Rung 2, prompt 56) is this leg's first real win: beats
c1_lognormal_glm on common-scale log RMSE in 5/5 CV folds (paired t
p=0.0126), and surfaced genuine nonlinear signal invisible to the linear
rungs -- distance_to_attacking_box jumped from linear-correlation rank #21
to Gini-importance rank #2. That result is the reason to test gradient
boosting directly rather than stop at RF, mirroring the active-binary leg's
own Rung 4 (prompt 45, train_v1e_gradient_boosting.py): a different
tree-ensemble method worth checking once one tree method already found real
structure -- not assumed to help just because RF did.

Library choice: LightGBM, matching train_v1e_gradient_boosting.py's own
choice for the active-binary leg (confirmed available in this environment,
version 4.6.0) -- kept consistent rather than introducing XGBoost as a
second boosting library for no reason.

Grid trimmed for this leg's small sample, the same way prompt 56 trimmed the
RF grid: the binary leg's own v1e grid (LEARNING_RATE=[0.01,0.05,0.1],
NUM_LEAVES=[15,31,63], MIN_CHILD_SAMPLES=[10,30,100]) was sized for ~36,000
rows per training fold. This leg's positive-row subset trains on ~2,880 rows
per fold (3,600 trainval rows, 5-fold CV) -- num_leaves=63 or
min_child_samples=10 would badly overfit a tree that shallow a dataset.
NUM_LEAVES trimmed down to [7,15,31] (mirrors c1d's max_depth trim from
[8,14,None] to [4,8,None]) and MIN_CHILD_SAMPLES raised to [20,50,100]
(mirrors c1d's min_samples_leaf trim from [5,20] to [10,30,60]).
LEARNING_RATE is left unchanged -- it does not scale with row count the way
leaf-size parameters do, and MAX_N_ESTIMATORS/early stopping self-regulates
ensemble size regardless of sample size.

c1c_systematic_interactions remains explicitly skipped (prompt 56's
decision, cited with real numbers in ACTIVE_CONTINUOUS_MODEL_LADDER.md) --
this rung does not revisit that.

Follows train_c1d_random_forest.py's sibling-script pattern (imports acb/ab
for shared helpers, reuses v1d_mod.RFDesignMatrixBuilder directly rather
than redefining it -- the class is fully generic, keyed only off
ab.CATEGORICAL_COLS/NUMERIC_COLS/BOOLEAN_COLS, already reused this way by
train_active_continuous_baseline.py itself to score the classifier half).

Usage:
    python scripts/models/train_c1e_gradient_boosting.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402
import train_active_continuous_baseline as acb  # noqa: E402
import train_v1d_random_forest as v1d_mod  # noqa: E402
from dax.models.evaluation import regression_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "c1e_gradient_boosting"
RANDOM_STATE = 42

# Trimmed for this leg's ~2,880 rows/training-fold (vs the active-binary
# leg's own v1e grid, sized for ~36,000 rows/fold) -- see module docstring.
LEARNING_RATE_GRID = [0.01, 0.05, 0.1]
NUM_LEAVES_GRID = [7, 15, 31]
MIN_CHILD_SAMPLES_GRID = [20, 50, 100]
MAX_N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50

RFDesignMatrixBuilder = v1d_mod.RFDesignMatrixBuilder


def make_eval_split(df: pd.DataFrame, frac: float = 0.2, seed: int = RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    groups = df[ab.GROUP_COL]
    idx_train, idx_valid = next(gss.split(df, groups=groups))
    return df.iloc[idx_train], df.iloc[idx_valid]


def fit_predict_c1e(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict, n_estimators_fixed: int | None = None):
    """Returns (log_pred, sigma2, model, (builder, feature_names), train_r2, best_iteration).

    When n_estimators_fixed is None (grid search / CV), fits with early
    stopping on a match-grouped carve-out of train_df, mirroring
    train_v1e_gradient_boosting.py's own fit_predict_gbm. When
    n_estimators_fixed is given (the single held-out fit), trains on the
    FULL train_df at that fixed tree count -- no held-out-adjacent carve-out
    needed once the ensemble size is already decided by CV."""
    y_train_log = np.log(train_df[acb.TARGET_XG_COL].to_numpy())
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_test, feature_names = builder.transform(test_df)

    if n_estimators_fixed is None:
        sub_train_df, sub_valid_df = make_eval_split(train_df)
        x_sub_train, _ = builder.transform(sub_train_df)
        x_sub_valid, _ = builder.transform(sub_valid_df)
        y_sub_train = np.log(sub_train_df[acb.TARGET_XG_COL].to_numpy())
        y_sub_valid = np.log(sub_valid_df[acb.TARGET_XG_COL].to_numpy())

        model = LGBMRegressor(
            n_estimators=MAX_N_ESTIMATORS, learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        model.fit(
            x_sub_train, y_sub_train,
            eval_set=[(x_sub_valid, y_sub_valid)], eval_metric="rmse",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_iteration = model.best_iteration_
        x_train_full, _ = builder.transform(train_df)
        train_pred = model.predict(x_train_full)
    else:
        x_train, _ = builder.transform(train_df)
        model = LGBMRegressor(
            n_estimators=n_estimators_fixed, learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        model.fit(x_train, y_train_log)
        best_iteration = n_estimators_fixed
        train_pred = model.predict(x_train)

    pred_log = model.predict(x_test)
    resid_train = y_train_log - train_pred
    sigma2 = float(np.var(resid_train, ddof=1)) if len(resid_train) > 1 else 0.0
    train_r2 = 1.0 - np.sum(resid_train ** 2) / np.sum((y_train_log - y_train_log.mean()) ** 2)
    return pred_log, sigma2, model, (builder, feature_names), float(train_r2), int(best_iteration)


def run_cv_c1e(pos_trainval_df: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_log_pred = np.full(len(df), np.nan)
    oof_sigma2 = np.full(len(df), np.nan)
    train_r2s = []
    best_iterations = []

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        log_pred, sigma2, _, _, train_r2, best_iter = fit_predict_c1e(train_df, test_df, params)
        train_r2s.append(train_r2)
        best_iterations.append(best_iter)
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
            "train_r2": train_r2, "best_iteration": best_iter,
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
        "mean_train_r2": float(np.mean(train_r2s)), "mean_best_iteration": float(np.mean(best_iterations)),
    }
    oof_metrics.update({f"naive_{k}": v for k, v in regression_metrics(y_true_xg_all, oof_xg_naive).items()})
    oof_metrics.update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg_all, oof_xg_corrected).items()})

    return {"fold_metrics": fold_df, "oof_metrics": oof_metrics, "best_iterations": best_iterations}


def tune_c1e(pos_trainval_df: pd.DataFrame) -> dict:
    grid_results = []
    cv_by_params = {}
    combos = list(itertools.product(LEARNING_RATE_GRID, NUM_LEAVES_GRID, MIN_CHILD_SAMPLES_GRID))
    print(f"[tune] {len(combos)} combinations x 5 folds = {len(combos) * 5} fits")
    for lr, leaves, min_child in combos:
        params = {"learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child}
        key = (lr, leaves, min_child)
        result = run_cv_c1e(pos_trainval_df, params)
        oof_common_log_rmse = result["oof_metrics"]["common_log_rmse"]
        train_r2 = result["oof_metrics"]["mean_train_r2"]
        oof_r2 = result["oof_metrics"]["corrected_r2"]
        overfit_gap = train_r2 - oof_r2
        grid_results.append({
            **params,
            "oof_common_log_rmse": oof_common_log_rmse, "oof_corrected_r2": oof_r2,
            "mean_train_r2": train_r2, "train_minus_oof_r2_gap": overfit_gap,
            "mean_best_iteration": result["oof_metrics"]["mean_best_iteration"],
        })
        cv_by_params[key] = result
        print(f"  [tune] lr={lr} leaves={leaves} min_child={min_child} "
              f"common_log_rmse={oof_common_log_rmse:.4f} oof_r2={oof_r2:.4f} "
              f"train_r2={train_r2:.4f} gap={overfit_gap:.4f} "
              f"mean_best_iter={result['oof_metrics']['mean_best_iteration']:.0f}")

    best = min(grid_results, key=lambda r: r["oof_common_log_rmse"])
    best_key = (best["learning_rate"], best["num_leaves"], best["min_child_samples"])
    print(f"[tune] best (by OOF common_log_rmse): {best}")
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
        y_test_log = np.log(test_df[acb.TARGET_XG_COL].to_numpy())

        _, _, model, extra, _, _ = fit_predict_c1e(train_df, test_df, best_params)
        builder, feature_names = extra
        x_test, _ = builder.transform(test_df)

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

    n_combos = len(LEARNING_RATE_GRID) * len(NUM_LEAVES_GRID) * len(MIN_CHILD_SAMPLES_GRID)
    print(f"[tune] grid-searching LightGBM hyperparameters ({n_combos} combos)...")
    tuning = tune_c1e(pos_trainval)
    best_params = {k: tuning["best"][k] for k in ("learning_rate", "num_leaves", "min_child_samples")}
    cv_result = tuning["cv_by_params"][tuning["best_key"]]
    final_n_estimators = int(round(tuning["best"]["mean_best_iteration"]))
    print(f"[tune] chosen params={best_params}, final_n_estimators (mean best_iteration, rounded)={final_n_estimators}")

    comparison_row = {"variant": VARIANT, "n_features": None}
    comparison_row.update({f"gbm_{k}": v for k, v in best_params.items()})
    comparison_row["gbm_n_estimators"] = final_n_estimators
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(pos_trainval)
    comparison_row["matches"] = pos_trainval[ab.GROUP_COL].nunique()

    print("[significance] c1e_gradient_boosting vs c1_lognormal_glm and c1d_random_forest "
          "(common-scale log RMSE per fold)...")
    c1_cv = acb.run_cv_regression(pos_trainval, "c1_lognormal_glm")

    import train_c1d_random_forest as c1d_mod
    c1d_sig = json.loads((acb.VALIDATION_DIR / "significance_c1_vs_c1d_random_forest.json").read_text(encoding="utf-8"))
    c1d_best_params = c1d_sig["chosen_params"]
    c1d_cv = c1d_mod.run_cv_c1d(pos_trainval, c1d_best_params)

    fold_common_log_rmse = {
        "c1e_gradient_boosting": cv_result["fold_metrics"]["common_log_rmse"].tolist(),
        "c1_lognormal_glm": c1_cv["fold_metrics"]["common_log_rmse"].tolist(),
        "c1d_random_forest": c1d_cv["fold_metrics"]["common_log_rmse"].tolist(),
    }

    def paired_test(a_key: str, b_key: str) -> dict:
        a = np.array(fold_common_log_rmse[a_key])
        b = np.array(fold_common_log_rmse[b_key])
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        if len(set(diff.round(12))) > 1:
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float("nan"), float("nan")
        return {
            "mean_diff": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        }

    vs_c1 = paired_test("c1e_gradient_boosting", "c1_lognormal_glm")
    vs_c1d = paired_test("c1e_gradient_boosting", "c1d_random_forest")

    sig = {
        "metric": "common_log_rmse (per fold, natural-log(xg) scale, lower is better)",
        "fold_common_log_rmse": fold_common_log_rmse,
        "c1e_vs_c1": vs_c1,
        "c1e_vs_c1d": vs_c1d,
        "n_folds": 5,
        "caution_note": (
            "5 folds only, ~875 positive rows per fold (this leg's much smaller sample than the "
            "binary legs) -- read this test as directionally informative, not as strong statistical "
            "proof, the same honest caveat given to every small-n significance test elsewhere on "
            "this leg."
        ),
        "hyperparameter_grid_search": tuning["grid_results"],
        "chosen_params": best_params,
        "chosen_n_estimators": final_n_estimators,
        "c1d_params_used_for_comparison": c1d_best_params,
    }
    (acb.VALIDATION_DIR / "significance_c1e_vs_c1_c1d.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  c1e vs c1:  mean_diff={vs_c1['mean_diff']:+.4f}, t_p={vs_c1['paired_t_pvalue']:.4f}, "
          f"wilcoxon_p={vs_c1['wilcoxon_pvalue']}")
    print(f"  c1e vs c1d: mean_diff={vs_c1d['mean_diff']:+.4f}, t_p={vs_c1d['paired_t_pvalue']:.4f}, "
          f"wilcoxon_p={vs_c1d['wilcoxon_pvalue']}")

    print("[held-out] c1e_gradient_boosting (fit once on all positive trainval rows, "
          f"n_estimators fixed at {final_n_estimators} from CV early stopping)...")
    log_pred, sigma2, model, extra, _, _ = fit_predict_c1e(
        pos_trainval, pos_test, best_params, n_estimators_fixed=final_n_estimators
    )
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

    calib_rows_naive = acb.calibration_bins(y_true_xg, xg_naive)
    calib_rows_corrected = acb.calibration_bins(y_true_xg, xg_corrected)

    # Does the log-normal correction still help, or is GBM's own residual
    # structure better served raw? Compare mean |gap| across bins for both.
    naive_mean_abs_gap = float(np.mean([abs(r["gap"]) for r in calib_rows_naive]))
    corrected_mean_abs_gap = float(np.mean([abs(r["gap"]) for r in calib_rows_corrected]))
    correction_helps_calibration = corrected_mean_abs_gap < naive_mean_abs_gap

    gain = sorted(
        ({"feature": n, "gain_importance": float(v)} for n, v in zip(feature_names, model.feature_importances_)),
        key=lambda r: r["gain_importance"], reverse=True,
    )
    print("[permutation importance] computing on fold-held-out portions (5 folds, not final test)...")
    perm_importance = fold_held_out_permutation_importance(pos_trainval, best_params)

    importance_output = {
        "variant": VARIANT, "best_params": best_params, "n_estimators": final_n_estimators,
        "fitted_on": "all positive-row train+val rows (canonical split); held-out test used once for readout only",
        "naive_calibration_bins": calib_rows_naive,
        "corrected_calibration_bins": calib_rows_corrected,
        "naive_mean_abs_calibration_gap": naive_mean_abs_gap,
        "corrected_mean_abs_calibration_gap": corrected_mean_abs_gap,
        "correction_helps_calibration": correction_helps_calibration,
        "top15_gain_importance": gain[:15],
        "top15_permutation_importance_fold_held_out": perm_importance[:15],
        "full_gain_importance": gain,
        "full_permutation_importance_fold_held_out": perm_importance,
    }
    (acb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")

    from dax.models.diagnostics import _ACCENT, _new_axes, _save_fixed, ensure_dir
    chart_dir = acb.REG_CHARTS_DIR / VARIANT
    ensure_dir(chart_dir)

    fig, ax = _new_axes()
    top = gain[:15]
    ax.barh([r["feature"] for r in top][::-1], [r["gain_importance"] for r in top][::-1], color=_ACCENT)
    ax.set_xlabel("Gain importance")
    ax.set_title("Top 15 feature importance (gain)")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")

    acb.save_regression_variant_charts(chart_dir, y_true_log, log_pred, y_true_xg, xg_corrected, calib_rows_corrected, None)
    fig, ax = _new_axes()
    ax.barh([r["feature"] for r in top][::-1], [r["gain_importance"] for r in top][::-1], color=_ACCENT)
    ax.set_xlabel("Gain importance")
    ax.set_title("Top 15 feature importance (gain)")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")

    joblib.dump(model, acb.REG_DIR / f"{VARIANT}.joblib")

    print(f"  held-out: log_rmse={held_out_row['log_rmse']:.4f} common_log_rmse={held_out_row['common_log_rmse']:.4f} "
          f"naive_mae={held_out_row['naive_mae']:.4f} corrected_mae={held_out_row['corrected_mae']:.4f} "
          f"naive_r2={held_out_row['naive_r2']:.4f} corrected_r2={held_out_row['corrected_r2']:.4f}")
    print(f"  calibration: naive_mean_abs_gap={naive_mean_abs_gap:.4f} "
          f"corrected_mean_abs_gap={corrected_mean_abs_gap:.4f} "
          f"correction_helps={correction_helps_calibration}")

    print("[csv] appending c1e_gradient_boosting rows (additive only)...")
    acb.append_to_csv(acb.COMPARISONS_DIR / "active_continuous_baseline_comparison.csv", [comparison_row])
    acb.append_to_csv(acb.COMPARISONS_DIR / "active_continuous_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== HYPERPARAMETER GRID (chosen by OOF common_log_rmse) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if (r["learning_rate"], r["num_leaves"], r["min_child_samples"]) == tuning["best_key"] else ""
        print(f"  lr={r['learning_rate']} leaves={r['num_leaves']} min_child={r['min_child_samples']} "
              f"common_log_rmse={r['oof_common_log_rmse']:.4f} oof_r2={r['oof_corrected_r2']:.4f} "
              f"train_r2={r['mean_train_r2']:.4f} gap={r['train_minus_oof_r2_gap']:.4f} "
              f"best_iter={r['mean_best_iteration']:.0f}{marker}")
    print("\n=== TOP 15 GAIN IMPORTANCE ===")
    for r in gain[:15]:
        print(f"  {r['feature']}: {r['gain_importance']:.1f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
