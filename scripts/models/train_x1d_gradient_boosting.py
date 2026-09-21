"""Rung 3 of the active-xT leg (prompt 74): x1d_gradient_boosting.

`x1c_random_forest` (Rung 2, prompt 73) is this leg's first real win: beats
`x1_two_stage_huber` on RMSE in 5/5 CV folds (paired t p=0.00006), and closed
the tail-calibration gap Rungs 0/1 tracked by 10-35x. That result is the
reason to test gradient boosting directly, mirroring the active-binary leg's
own Rung 4 (`v1e_gradient_boosting`, prompt 45) and the active-continuous
leg's own Rung 3 (`c1e_gradient_boosting`, prompt 57) -- a different
tree-ensemble method worth checking once one tree method already found real
structure, not assumed to help just because RF did.

STEP 0/1 -- confirm x1c's actual architecture before deciding what x1d replaces
---------------------------------------------------------------------------
Checked directly against `outputs/models/regression/x1c_random_forest.json`
and `ACTIVE_XT_MODEL_LADDER.md` section 3.1/3.7 before assuming anything:
`x1c_random_forest` replaced BOTH of `x1`'s surfaces with RandomForest --
`RandomForestClassifier` for P(nonzero) (`clf_params` present in the JSON)
AND `RandomForestRegressor` for E[delta|nonzero] (`reg_params` present too),
each independently tuned. This is confirmed, not assumed -- Prompt 73's own
Step 1 explicitly rejected the "only touch one surface" pattern
`c1d_random_forest` used, because that pattern was a structural artefact of
`c1d`'s classifier being a cross-leg reused model, not evidence that only
one surface needed the upgrade. That same reasoning carries forward
unchanged into this rung: both of `x1c`'s surfaces are this leg's own
models, built together, with no new asymmetric evidence produced since Rung
2 that would justify upgrading only one of them. **Decision: `x1d` replaces
BOTH stages with gradient boosting** -- `LGBMClassifier` for P(nonzero),
`LGBMRegressor` for E[delta|nonzero], mirroring `x1c`'s own two-independent-
models composition, gated against `x1c` specifically (not the older linear
rungs) per this prompt's own instruction.

Library choice: LightGBM, matching both `v1e_gradient_boosting` (active-binary)
and `c1e_gradient_boosting` (active-continuous) -- confirmed available in
this environment (version 4.7.0, `import lightgbm` succeeds), kept
consistent rather than introducing XGBoost as a second boosting library for
no reason.

Grid, sized for this leg's real row counts (confirmed via `xb.load_data()` +
`canonical_test_mask`, not assumed to match either precedent leg): classifier
trains on 45,166 trainval rows (~36k/fold in 5-fold CV); regressor trains on
36,121 nonzero-only trainval rows (~29k/fold). This sits between `c1e`'s
tiny sample (~2,880 rows/fold, trimmed grid `NUM_LEAVES=[7,15,31]`,
`MIN_CHILD_SAMPLES=[20,50,100]`) and `v1e`'s full active-binary scale
(~36,000 rows/fold, `NUM_LEAVES=[15,31,63]`, `MIN_CHILD_SAMPLES=[10,30,100]`)
-- close enough to `v1e`'s own scale to reuse its `NUM_LEAVES_GRID`
unchanged, but `MIN_CHILD_SAMPLES_GRID` is kept at `[20,50,150]` to stay
internally consistent with `x1c`'s own already-established
`MIN_SAMPLES_LEAF_GRID=[20,50,150]` for this exact leg at this exact row
scale (Prompt 73's own reasoning), rather than reusing `v1e`'s `[10,30,100]`
by default. `LEARNING_RATE_GRID=[0.01,0.05,0.1]` unchanged from both
precedents -- it does not scale with row count the way leaf-size parameters
do. `n_estimators` is never grid-searched directly -- chosen per fit via
early stopping (cap 2,000 rounds, 50-round patience) on a match-grouped
validation carve-out of that fit's own training rows, exactly
`v1e`/`c1e`'s own fitting procedure.

Usage:
    .venv/Scripts/python.exe scripts/models/train_x1d_gradient_boosting.py
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
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402
import train_active_xt_baseline as xb  # noqa: E402
import train_x1c_random_forest as x1c_mod  # noqa: E402
from dax.models.diagnostics import _ACCENT, _new_axes, _save_fixed, ensure_dir  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "x1d_gradient_boosting"
RANDOM_STATE = 42

LEARNING_RATE_GRID = [0.01, 0.05, 0.1]
NUM_LEAVES_GRID = [15, 31, 63]
MIN_CHILD_SAMPLES_GRID = [20, 50, 150]
MAX_N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50

RFDesignMatrixBuilder = x1c_mod.RFDesignMatrixBuilder

# x1c's own already-tuned, already-promoted params -- reused to recompute a
# directly comparable x1c CV run on this script's own data construction,
# same reasoning c1e_gradient_boosting recomputed c1d_random_forest's CV
# fresh via c1d's own module function rather than trusting a stored artefact.
X1C_CLF_PARAMS = {"max_depth": None, "min_samples_leaf": 20}
X1C_REG_PARAMS = {"max_depth": None, "min_samples_leaf": 20}


def make_eval_split(df: pd.DataFrame, frac: float = 0.2, seed: int = RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    groups = df[xb.GROUP_COL]
    idx_train, idx_valid = next(gss.split(df, groups=groups))
    return df.iloc[idx_train], df.iloc[idx_valid]


# ---------------------------------------------------------------------------
# Classifier stage (P(nonzero))
# ---------------------------------------------------------------------------

def fit_predict_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict, n_estimators_fixed: int | None = None):
    y_train = (train_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_test, feature_names = builder.transform(test_df)

    if n_estimators_fixed is None:
        sub_train_df, sub_valid_df = make_eval_split(train_df)
        x_sub_train, _ = builder.transform(sub_train_df)
        x_sub_valid, _ = builder.transform(sub_valid_df)
        y_sub_train = (sub_train_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)
        y_sub_valid = (sub_valid_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)

        model = LGBMClassifier(
            n_estimators=MAX_N_ESTIMATORS, learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        model.fit(
            x_sub_train, y_sub_train,
            eval_set=[(x_sub_valid, y_sub_valid)], eval_metric="auc",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_iteration = model.best_iteration_
        x_train_full, _ = builder.transform(train_df)
        train_p = model.predict_proba(x_train_full)[:, 1]
    else:
        x_train, _ = builder.transform(train_df)
        model = LGBMClassifier(
            n_estimators=n_estimators_fixed, learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        best_iteration = n_estimators_fixed
        train_p = model.predict_proba(x_train)[:, 1]

    p_test = model.predict_proba(x_test)[:, 1]
    train_auc = classification_metrics(y_train, train_p)["roc_auc"]
    return p_test, model, (builder, feature_names), float(train_auc), int(best_iteration)


def run_cv_classifier(trainval: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(trainval, group_col=xb.GROUP_COL)
    df = trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = (df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)

    oof_p = np.full(len(df), np.nan)
    train_aucs, best_iterations = [], []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        p_test, _, _, train_auc, best_iter = fit_predict_classifier(train_df, test_df, params)
        oof_p[test_mask] = p_test
        train_aucs.append(train_auc)
        best_iterations.append(best_iter)

    diag = classification_metrics(y_all, oof_p)
    diag["mean_train_roc_auc"] = float(np.mean(train_aucs))
    diag["train_minus_oof_roc_auc_gap"] = float(np.mean(train_aucs) - diag["roc_auc"])
    diag["mean_best_iteration"] = float(np.mean(best_iterations))
    return diag


def tune_classifier(trainval: pd.DataFrame) -> dict:
    grid_results = []
    combos = list(itertools.product(LEARNING_RATE_GRID, NUM_LEAVES_GRID, MIN_CHILD_SAMPLES_GRID))
    print(f"  [tune:clf] {len(combos)} combinations x 5 folds = {len(combos) * 5} fits")
    for lr, leaves, min_child in combos:
        params = {"learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child}
        diag = run_cv_classifier(trainval, params)
        grid_results.append({**params, "oof_roc_auc": diag["roc_auc"], "mean_train_roc_auc": diag["mean_train_roc_auc"],
                              "train_minus_oof_roc_auc_gap": diag["train_minus_oof_roc_auc_gap"],
                              "mean_best_iteration": diag["mean_best_iteration"]})
        print(f"    lr={lr} leaves={leaves} min_child={min_child} oof_roc_auc={diag['roc_auc']:.4f} "
              f"train_roc_auc={diag['mean_train_roc_auc']:.4f} gap={diag['train_minus_oof_roc_auc_gap']:.4f} "
              f"best_iter={diag['mean_best_iteration']:.0f}")
    best = max(grid_results, key=lambda r: r["oof_roc_auc"])
    print(f"  [tune:clf] best (by OOF ROC-AUC): {best}")
    return {"grid_results": grid_results, "best": best}


# ---------------------------------------------------------------------------
# Regressor stage (E[delta|nonzero])
# ---------------------------------------------------------------------------

def fit_predict_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, params: dict, n_estimators_fixed: int | None = None):
    y_train = train_df[xb.TARGET_COL].to_numpy()
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_test, feature_names = builder.transform(test_df)

    if n_estimators_fixed is None:
        sub_train_df, sub_valid_df = make_eval_split(train_df)
        x_sub_train, _ = builder.transform(sub_train_df)
        x_sub_valid, _ = builder.transform(sub_valid_df)
        y_sub_train = sub_train_df[xb.TARGET_COL].to_numpy()
        y_sub_valid = sub_valid_df[xb.TARGET_COL].to_numpy()

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
        model.fit(x_train, y_train)
        best_iteration = n_estimators_fixed
        train_pred = model.predict(x_train)

    pred_test = model.predict(x_test)
    train_resid = y_train - train_pred
    train_r2 = 1.0 - np.sum(train_resid ** 2) / np.sum((y_train - y_train.mean()) ** 2)
    return pred_test, model, (builder, feature_names), float(train_r2), int(best_iteration)


def run_cv_regressor(nonzero_trainval: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(nonzero_trainval, group_col=xb.GROUP_COL)
    df = nonzero_trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[xb.TARGET_COL].to_numpy()

    oof_pred = np.full(len(df), np.nan)
    train_r2s, best_iterations = [], []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        pred, _, _, train_r2, best_iter = fit_predict_regressor(train_df, test_df, params)
        oof_pred[test_mask] = pred
        train_r2s.append(train_r2)
        best_iterations.append(best_iter)

    m = xb.signed_regression_metrics(y_all, oof_pred)
    m["mean_train_r2"] = float(np.mean(train_r2s))
    m["train_minus_oof_r2_gap"] = float(np.mean(train_r2s) - m["r2"])
    m["mean_best_iteration"] = float(np.mean(best_iterations))
    return m


def tune_regressor(nonzero_trainval: pd.DataFrame) -> dict:
    grid_results = []
    combos = list(itertools.product(LEARNING_RATE_GRID, NUM_LEAVES_GRID, MIN_CHILD_SAMPLES_GRID))
    print(f"  [tune:reg] {len(combos)} combinations x 5 folds = {len(combos) * 5} fits")
    for lr, leaves, min_child in combos:
        params = {"learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child}
        m = run_cv_regressor(nonzero_trainval, params)
        grid_results.append({**params, "oof_rmse": m["rmse"], "oof_r2": m["r2"],
                              "mean_train_r2": m["mean_train_r2"], "train_minus_oof_r2_gap": m["train_minus_oof_r2_gap"],
                              "mean_best_iteration": m["mean_best_iteration"]})
        print(f"    lr={lr} leaves={leaves} min_child={min_child} oof_rmse={m['rmse']:.5f} oof_r2={m['r2']:.4f} "
              f"train_r2={m['mean_train_r2']:.4f} gap={m['train_minus_oof_r2_gap']:.4f} "
              f"best_iter={m['mean_best_iteration']:.0f}")
    best = min(grid_results, key=lambda r: r["oof_rmse"])
    print(f"  [tune:reg] best (by OOF RMSE): {best}")
    return {"grid_results": grid_results, "best": best}


# ---------------------------------------------------------------------------
# Combined fit/predict
# ---------------------------------------------------------------------------

def fit_predict_combined(train_df: pd.DataFrame, test_df: pd.DataFrame, clf_params: dict, reg_params: dict,
                          clf_n_estimators: int | None = None, reg_n_estimators: int | None = None):
    p_nonzero, clf, clf_extra, _, _ = fit_predict_classifier(train_df, test_df, clf_params, clf_n_estimators)
    nonzero_train_df = train_df.loc[train_df[xb.TARGET_COL] != 0.0]
    e_delta, reg, reg_extra, _, _ = fit_predict_regressor(nonzero_train_df, test_df, reg_params, reg_n_estimators)
    combined_pred = p_nonzero * e_delta
    extra = {
        "clf": clf, "clf_builder": clf_extra[0], "clf_feature_names": clf_extra[1],
        "reg": reg, "reg_builder": reg_extra[0], "reg_feature_names": reg_extra[1],
    }
    return combined_pred, extra


def run_cv_combined(trainval: pd.DataFrame, clf_params: dict, reg_params: dict,
                     clf_n_estimators: int | None, reg_n_estimators: int | None) -> dict:
    folds = canonical_grouped_folds(trainval, group_col=xb.GROUP_COL)
    df = trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_pred = np.full(len(df), np.nan)
    y_all = df[xb.TARGET_COL].to_numpy()

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        pred, _ = fit_predict_combined(train_df, test_df, clf_params, reg_params, clf_n_estimators, reg_n_estimators)
        oof_pred[test_mask] = pred

        m = xb.signed_regression_metrics(y_all[test_mask], pred)
        m["fold"] = int(fold_id)
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    oof_metrics = xb.signed_regression_metrics(y_all, oof_pred)
    return {"fold_metrics": fold_df, "oof_metrics": oof_metrics, "df": df}


def fold_held_out_permutation_importance(nonzero_trainval: pd.DataFrame, reg_params: dict, reg_n_estimators: int) -> list[dict]:
    folds = canonical_grouped_folds(nonzero_trainval, group_col=xb.GROUP_COL)
    df = nonzero_trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    importances_by_feature: dict[str, list[float]] = {}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_test = test_df[xb.TARGET_COL].to_numpy()

        _, model, extra, _, _ = fit_predict_regressor(train_df, test_df, reg_params, reg_n_estimators)
        builder, feature_names = extra
        x_test, _ = builder.transform(test_df)

        result = permutation_importance(model, x_test, y_test, scoring="neg_mean_squared_error",
                                         n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        for name, mean_val in zip(feature_names, result.importances_mean):
            importances_by_feature.setdefault(name, []).append(float(mean_val))

    return sorted(
        ({"feature": name, "mean_permutation_importance": float(np.mean(vals)), "n_folds": len(vals)}
         for name, vals in importances_by_feature.items()),
        key=lambda r: r["mean_permutation_importance"], reverse=True,
    )


def main() -> None:
    xb.REG_DIR.mkdir(parents=True, exist_ok=True)
    xb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    xb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    xb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = xb.load_data()
    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=xb.GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test = df_all.loc[test_mask_full].reset_index(drop=True)
    nonzero_trainval = trainval.loc[trainval[xb.TARGET_COL] != 0.0].reset_index(drop=True)
    print(f"[data] trainval={len(trainval)} rows ({trainval[xb.GROUP_COL].nunique()} matches); "
          f"nonzero_trainval={len(nonzero_trainval)} rows; held-out test={len(test)} rows")

    print("\n[1/7] tuning classifier stage (LGBMClassifier, OOF ROC-AUC)...")
    clf_tune = tune_classifier(trainval)
    (xb.VALIDATION_DIR / "x1d_classifier_tuning.json").write_text(json.dumps(clf_tune, indent=2), encoding="utf-8")
    clf_params = {k: clf_tune["best"][k] for k in ("learning_rate", "num_leaves", "min_child_samples")}
    clf_n_estimators = int(round(clf_tune["best"]["mean_best_iteration"]))

    print("\n[2/7] tuning regressor stage (LGBMRegressor, OOF RMSE)...")
    reg_tune = tune_regressor(nonzero_trainval)
    (xb.VALIDATION_DIR / "x1d_regressor_tuning.json").write_text(json.dumps(reg_tune, indent=2), encoding="utf-8")
    reg_params = {k: reg_tune["best"][k] for k in ("learning_rate", "num_leaves", "min_child_samples")}
    reg_n_estimators = int(round(reg_tune["best"]["mean_best_iteration"]))

    print(f"\n[3/7] combined CV, x1d_gradient_boosting (clf={clf_params} n_est={clf_n_estimators}, "
          f"reg={reg_params} n_est={reg_n_estimators})...")
    cv_result = run_cv_combined(trainval, clf_params, reg_params, None, None)
    comparison_row = {"variant": VARIANT, "n_features": 32}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(trainval)
    comparison_row["matches"] = trainval[xb.GROUP_COL].nunique()
    comparison_row["clf_learning_rate"] = clf_params["learning_rate"]
    comparison_row["clf_num_leaves"] = clf_params["num_leaves"]
    comparison_row["clf_min_child_samples"] = clf_params["min_child_samples"]
    comparison_row["clf_n_estimators"] = clf_n_estimators
    comparison_row["reg_learning_rate"] = reg_params["learning_rate"]
    comparison_row["reg_num_leaves"] = reg_params["num_leaves"]
    comparison_row["reg_min_child_samples"] = reg_params["min_child_samples"]
    comparison_row["reg_n_estimators"] = reg_n_estimators

    print("\n[4/7] significance test, x1d vs x1c (per-fold RMSE, current best -- NOT x1/x1b)...")
    x1c_cv = x1c_mod.run_cv_combined(trainval, X1C_CLF_PARAMS, X1C_REG_PARAMS)
    a = cv_result["fold_metrics"]["rmse"].to_numpy()
    b = x1c_cv["fold_metrics"]["rmse"].to_numpy()
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    sig = {
        "metric": "rmse (per fold, signed target_xt_delta_v2 scale, lower is better)",
        "baseline_compared_against": "x1c_random_forest (current best, promoted Rung 2 -- NOT x1 or x1b)",
        "fold_rmse_x1d": a.tolist(), "fold_rmse_x1c": b.tolist(),
        "mean_diff_x1d_minus_x1c": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "x1d_clf_params": clf_params, "x1d_reg_params": reg_params,
        "x1d_clf_n_estimators": clf_n_estimators, "x1d_reg_n_estimators": reg_n_estimators,
        "x1c_params_used_for_comparison": {"clf": X1C_CLF_PARAMS, "reg": X1C_REG_PARAMS},
    }
    (xb.VALIDATION_DIR / "significance_x1d_vs_x1c.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (x1d-x1c) rmse: {sig['mean_diff_x1d_minus_x1c']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[5/7] held-out test readout + permutation importance...")
    pred, extra = fit_predict_combined(trainval, test, clf_params, reg_params, clf_n_estimators, reg_n_estimators)
    y_true = test[xb.TARGET_COL].to_numpy()
    held_out_row = xb.signed_regression_metrics(y_true, pred)
    held_out_row["variant"] = VARIANT
    held_out_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    held_out_row["rows"] = len(test)
    held_out_row["matches"] = test[xb.GROUP_COL].nunique()

    calib_rows = xb.calibration_bins(y_true, pred)

    clf_diag_holdout = classification_metrics(
        (test[xb.TARGET_COL].to_numpy() != 0.0).astype(int),
        extra["clf"].predict_proba(extra["clf_builder"].transform(test)[0])[:, 1],
    )
    print(f"  [held-out classifier diagnostic] roc_auc={clf_diag_holdout['roc_auc']:.4f} "
          f"average_precision={clf_diag_holdout['average_precision']:.4f} brier={clf_diag_holdout['brier_score']:.4f}")

    print("[permutation importance] computing on fold-held-out portions (5 folds, not final test)...")
    perm_importance = fold_held_out_permutation_importance(nonzero_trainval, reg_params, reg_n_estimators)

    gain = sorted(
        ({"feature": n, "gain_importance": float(v)} for n, v in zip(extra["reg_feature_names"], extra["reg"].feature_importances_)),
        key=lambda r: r["gain_importance"], reverse=True,
    )

    importance_output = {
        "variant": VARIANT, "clf_params": clf_params, "reg_params": reg_params,
        "clf_n_estimators": clf_n_estimators, "reg_n_estimators": reg_n_estimators,
        "clf_n_features": len(extra["clf_feature_names"]), "reg_n_features": len(extra["reg_feature_names"]),
        "reg_top15_gain_importance": gain[:15],
        "reg_full_gain_importance": gain,
        "reg_top15_permutation_importance_5fold_cv": perm_importance[:15],
        "reg_full_permutation_importance_5fold_cv": perm_importance,
        "clf_feature_importances_gain": {
            name: float(v) for name, v in zip(extra["clf_feature_names"], extra["clf"].feature_importances_)
        },
        "held_out_classifier_diagnostic": clf_diag_holdout,
        "fitted_on": "all trainval rows (canonical split); classifier on all trainval rows, "
                     "regression head on nonzero-delta trainval rows only; held-out test used once for readout only",
    }
    (xb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")
    joblib.dump(extra["clf"], xb.REG_DIR / f"{VARIANT}_classifier.joblib")
    joblib.dump(extra["reg"], xb.REG_DIR / f"{VARIANT}_regressor.joblib")

    xb.save_xt_regression_charts(xb.REG_CHARTS_DIR / VARIANT, y_true, pred, calib_rows, None)
    # save_xt_regression_charts writes a "no features (dummy variant)" placeholder for
    # feature_coefficients.png when coef_table is None -- overwrite it with a real gain-
    # importance bar chart (regression head), same fix applied to x1c_random_forest.
    top = gain[:15]
    chart_dir = ensure_dir(xb.REG_CHARTS_DIR / VARIANT)
    fig, ax = _new_axes()
    ax.barh([r["feature"] for r in top][::-1], [r["gain_importance"] for r in top][::-1], color=_ACCENT)
    ax.set_xlabel("Gain importance")
    ax.set_title("Top 15 feature importance (gain) -- regression head")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")

    print(f"  [held-out] {VARIANT}: rmse={held_out_row['rmse']:.5f} mae={held_out_row['mae']:.5f} "
          f"r2={held_out_row['r2']:.5f} spearman={held_out_row['spearman']:.4f} "
          f"zero_mae={held_out_row['zero_target_mae']:.5f} nonzero_mae={held_out_row['nonzero_target_mae']:.5f}")

    print("\n[6/7] writing comparison CSVs (additive only -- x0/x1/x1b/x1c rows untouched)...")
    xb.append_to_csv(xb.COMPARISONS_DIR / "active_xt_baseline_comparison.csv", [comparison_row])
    xb.append_to_csv(xb.COMPARISONS_DIR / "active_xt_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n[7/7] summary...")
    print("=== Calibration bins, held-out test, x1d (compare against x1/x1b/x1c tables) ===")
    for r in calib_rows:
        print(f"  bin {r['bin']}: n={r['n']} predicted={r['mean_predicted']:+.5f} actual={r['mean_actual']:+.5f} gap={r['gap']:+.5f}")

    print("\n=== TOP 15 GAIN IMPORTANCE (regression head) ===")
    for r in gain[:15]:
        print(f"  {r['feature']}: {r['gain_importance']:.1f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out, regression head) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:+.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
