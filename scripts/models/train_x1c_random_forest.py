"""Rung 2 of the active-xT leg (prompt 73): x1c_random_forest.

`x1_two_stage_huber` (Rung 0, prompt 70) remains the standing baseline --
`x1b_quadratic` (Rung 1, prompt 71) added real, evidence-backed curvature
terms to both surfaces but did not move the actual motivating question (the
tail-calibration gap), and its aggregate-metric gain was too small to matter
(RMSE -0.07% relative). Prompt 71's own read was that the limitation is a
property of `HuberRegressor`'s loss function (its robustness to outliers
caps how far it moves predictions toward extreme observed values), not of
the linear functional form -- i.e. adding curvature to a still-linear-loss
model couldn't fix it. This rung tests the next hypothesis: does swapping the
MODEL FAMILY (random forest, whose leaf-averaging has a fundamentally
different -- not necessarily better, but different -- tail behavior) close
the gap that more features on the same loss function could not.

STEP 1 -- which surface(s) get replaced with a random forest, and why
---------------------------------------------------------------------------
Checked directly against `train_c1d_random_forest.py` before deciding
anything here, per this leg's own standing discipline (Prompt 71 Step 0) of
checking the closer structural precedent rather than defaulting. `c1d`
replaced ONLY `c1`'s regression head with a random forest -- its classifier
half (`v1e_gradient_boosting_calibrated`) was reused EXACTLY as already
fitted, never re-tuned or re-fit with a different family. But the REASON
`c1d` only touched one surface is structural, not evidence-based: `v1e` is a
separately-built, already-promoted model belonging to an entirely different
leg (active-binary), reused via composition across leg boundaries -- it was
never part of `c1`'s own Rung 0/1/2 ladder at all, so there was nothing of
`c1`'s own to leave untouched on the classifier side; the classifier simply
isn't this leg's model to retune.

That reasoning does NOT transfer to `x1`. Both of `x1`'s surfaces (Stage A
classifier for P(nonzero), Stage B regressor for E[delta|nonzero]) were
built together, inside this leg's own Rung 0 (Prompt 70) -- neither is a
cross-leg reused artifact. Prompt 71's own Step 0 already established,
with real numbers, that the SAME candidate features show real curvature in
BOTH the zero-rate (Stage A's target) and the nonzero-delta magnitude
(Stage B's target) -- there is no asymmetric evidence here that would
justify upgrading one surface's model family while leaving the other on a
weaker one for no data-driven reason (unlike `c1d`, where the asymmetry was
a leg boundary, not evidence).

**Decision: replace both stages with RandomForest** --
`RandomForestClassifier` for P(nonzero), `RandomForestRegressor` for
E[delta|nonzero]. Each is tuned independently on its own natural OOF metric
(classifier: ROC-AUC; regressor: RMSE on the raw signed delta, no log
transform, same as x1/x1b) rather than jointly grid-searched as one
combined pipeline -- a joint grid would require fitting every
(classifier-params x regressor-params) combination, which is not needed
since each stage's own hyperparameters only affect that stage's own fit
quality, and the combined prediction is a simple product of the two
independently-best models (same composition x1/x1b already use).

Usage:
    .venv/Scripts/python.exe scripts/models/train_x1c_random_forest.py
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402
import train_active_xt_baseline as xb  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "x1c_random_forest"
RANDOM_STATE = 42

# Scaled for this leg's real row counts: classifier trains on ~36-45k rows/fold
# (full trainval), regressor on ~29-36k rows/fold (nonzero-only trainval) --
# between c1d's small scale (~3,500 rows) and d1d's much larger duplicated-row
# scale (hundreds of thousands). min_samples_leaf floor set well above c1d's
# 10-60 (this leg has far more rows) but well below d1d's 30-300 (this leg's
# rows are NOT defender-slot-duplicated -- 1 row per event, more independent
# signal per row than d1d's ~8x-duplicated grain).
N_ESTIMATORS = 200
MAX_DEPTH_GRID = [6, 10, None]
MIN_SAMPLES_LEAF_GRID = [20, 50, 150]


class RFDesignMatrixBuilder:
    """Same categorical (one-hot) / boolean (passthrough) blocks as
    vb.DesignMatrixBuilder. Numeric features median-imputed but NOT
    standardized -- trees are scale-invariant, and standardizing would only
    obscure feature_importances_. No engineered interaction/quadratic terms
    -- the point of this rung is what the tree finds on its own."""

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


# ---------------------------------------------------------------------------
# Classifier stage: tune on OOF ROC-AUC
# ---------------------------------------------------------------------------

def run_cv_classifier(trainval: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(trainval, group_col=xb.GROUP_COL)
    df = trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_nonzero_all = (df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)

    oof_p = np.full(len(df), np.nan)
    train_aucs = []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train = y_nonzero_all[train_mask]

        builder = RFDesignMatrixBuilder().fit(train_df)
        x_train, _ = builder.transform(train_df)
        x_test, _ = builder.transform(test_df)
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        oof_p[test_mask] = model.predict_proba(x_test)[:, 1]
        train_p = model.predict_proba(x_train)[:, 1]
        train_aucs.append(classification_metrics(y_train, train_p)["roc_auc"])

    diag = classification_metrics(y_nonzero_all, oof_p)
    diag["mean_train_roc_auc"] = float(np.mean(train_aucs))
    diag["train_minus_oof_roc_auc_gap"] = float(np.mean(train_aucs) - diag["roc_auc"])
    return diag


def tune_classifier(trainval: pd.DataFrame) -> dict:
    grid_results = []
    for max_depth, min_leaf in itertools.product(MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID):
        params = {"max_depth": max_depth, "min_samples_leaf": min_leaf}
        print(f"  [tune:clf] {params}")
        diag = run_cv_classifier(trainval, params)
        grid_results.append({**params, "oof_roc_auc": diag["roc_auc"], "mean_train_roc_auc": diag["mean_train_roc_auc"],
                              "train_minus_oof_roc_auc_gap": diag["train_minus_oof_roc_auc_gap"]})
        print(f"    oof_roc_auc={diag['roc_auc']:.4f} train_roc_auc={diag['mean_train_roc_auc']:.4f} "
              f"gap={diag['train_minus_oof_roc_auc_gap']:.4f}")
    best = max(grid_results, key=lambda r: r["oof_roc_auc"])
    print(f"  [tune:clf] best (by OOF ROC-AUC): {best}")
    return {"grid_results": grid_results, "best": best}


# ---------------------------------------------------------------------------
# Regressor stage: tune on OOF RMSE (raw signed delta, no log transform)
# ---------------------------------------------------------------------------

def run_cv_regressor(nonzero_trainval: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(nonzero_trainval, group_col=xb.GROUP_COL)
    df = nonzero_trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[xb.TARGET_COL].to_numpy()

    oof_pred = np.full(len(df), np.nan)
    train_r2s = []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train = y_all[train_mask]

        builder = RFDesignMatrixBuilder().fit(train_df)
        x_train, _ = builder.transform(train_df)
        x_test, _ = builder.transform(test_df)
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        oof_pred[test_mask] = pred
        train_pred = model.predict(x_train)
        train_r2 = 1.0 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
        train_r2s.append(float(train_r2))

    m = xb.signed_regression_metrics(y_all, oof_pred)
    m["mean_train_r2"] = float(np.mean(train_r2s))
    m["train_minus_oof_r2_gap"] = float(np.mean(train_r2s) - m["r2"])
    return m


def tune_regressor(nonzero_trainval: pd.DataFrame) -> dict:
    grid_results = []
    for max_depth, min_leaf in itertools.product(MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID):
        params = {"max_depth": max_depth, "min_samples_leaf": min_leaf}
        print(f"  [tune:reg] {params}")
        m = run_cv_regressor(nonzero_trainval, params)
        grid_results.append({**params, "oof_rmse": m["rmse"], "oof_r2": m["r2"],
                              "mean_train_r2": m["mean_train_r2"], "train_minus_oof_r2_gap": m["train_minus_oof_r2_gap"]})
        print(f"    oof_rmse={m['rmse']:.5f} oof_r2={m['r2']:.4f} train_r2={m['mean_train_r2']:.4f} "
              f"gap={m['train_minus_oof_r2_gap']:.4f}")
    best = min(grid_results, key=lambda r: r["oof_rmse"])
    print(f"  [tune:reg] best (by OOF RMSE): {best}")
    return {"grid_results": grid_results, "best": best}


# ---------------------------------------------------------------------------
# Combined fit/predict with the winning hyperparameters
# ---------------------------------------------------------------------------

def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, clf_params: dict, reg_params: dict):
    y_nonzero_train = (train_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)

    clf_builder = RFDesignMatrixBuilder().fit(train_df)
    x_train_clf, clf_feature_names = clf_builder.transform(train_df)
    x_test_clf, _ = clf_builder.transform(test_df)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=clf_params["max_depth"],
        min_samples_leaf=clf_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(x_train_clf, y_nonzero_train)
    p_nonzero_test = clf.predict_proba(x_test_clf)[:, 1]

    nonzero_train_df = train_df.loc[train_df[xb.TARGET_COL] != 0.0]
    reg_builder = RFDesignMatrixBuilder().fit(nonzero_train_df)
    x_train_reg, reg_feature_names = reg_builder.transform(nonzero_train_df)
    x_test_reg, _ = reg_builder.transform(test_df)
    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=reg_params["max_depth"],
        min_samples_leaf=reg_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg.fit(x_train_reg, nonzero_train_df[xb.TARGET_COL].to_numpy())
    e_delta_given_nonzero_test = reg.predict(x_test_reg)

    combined_pred = p_nonzero_test * e_delta_given_nonzero_test
    extra = {
        "clf": clf, "clf_builder": clf_builder, "clf_feature_names": clf_feature_names,
        "reg": reg, "reg_builder": reg_builder, "reg_feature_names": reg_feature_names,
    }
    return combined_pred, extra


def run_cv_combined(trainval: pd.DataFrame, clf_params: dict, reg_params: dict) -> dict:
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

        pred, _ = fit_predict(train_df, test_df, clf_params, reg_params)
        oof_pred[test_mask] = pred

        m = xb.signed_regression_metrics(y_all[test_mask], pred)
        m["fold"] = int(fold_id)
        m["n_train"] = int(train_mask.sum())
        m["n_test"] = int(test_mask.sum())
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    oof_metrics = xb.signed_regression_metrics(y_all, oof_pred)
    return {"fold_metrics": fold_df, "oof_metrics": oof_metrics, "df": df}


def fold_held_out_permutation_importance(nonzero_trainval: pd.DataFrame, reg_params: dict) -> list[dict]:
    folds = canonical_grouped_folds(nonzero_trainval, group_col=xb.GROUP_COL)
    df = nonzero_trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    importances_by_feature: dict[str, list[float]] = {}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = train_df[xb.TARGET_COL].to_numpy(), test_df[xb.TARGET_COL].to_numpy()

        builder = RFDesignMatrixBuilder().fit(train_df)
        x_train, feature_names = builder.transform(train_df)
        x_test, _ = builder.transform(test_df)
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=reg_params["max_depth"],
            min_samples_leaf=reg_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(x_train, y_train)

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

    print("[1/6] tuning classifier stage (RandomForestClassifier, OOF ROC-AUC)...")
    clf_tune = tune_classifier(trainval)
    (xb.VALIDATION_DIR / "x1c_classifier_tuning.json").write_text(json.dumps(clf_tune, indent=2), encoding="utf-8")
    clf_params = {"max_depth": clf_tune["best"]["max_depth"], "min_samples_leaf": clf_tune["best"]["min_samples_leaf"]}

    print("\n[2/6] tuning regressor stage (RandomForestRegressor, OOF RMSE)...")
    reg_tune = tune_regressor(nonzero_trainval)
    (xb.VALIDATION_DIR / "x1c_regressor_tuning.json").write_text(json.dumps(reg_tune, indent=2), encoding="utf-8")
    reg_params = {"max_depth": reg_tune["best"]["max_depth"], "min_samples_leaf": reg_tune["best"]["min_samples_leaf"]}

    print(f"\n[3/6] combined CV, x1c_random_forest with winning params "
          f"(clf={clf_params}, reg={reg_params})...")
    cv_result = run_cv_combined(trainval, clf_params, reg_params)
    comparison_row = {"variant": VARIANT, "n_features": 32}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(trainval)
    comparison_row["matches"] = trainval[xb.GROUP_COL].nunique()
    comparison_row["clf_max_depth"] = str(clf_params["max_depth"])
    comparison_row["clf_min_samples_leaf"] = clf_params["min_samples_leaf"]
    comparison_row["reg_max_depth"] = str(reg_params["max_depth"])
    comparison_row["reg_min_samples_leaf"] = reg_params["min_samples_leaf"]

    print("\n[4/6] significance test, x1c vs x1 (per-fold RMSE, standing baseline, NOT x1b)...")
    x1_cv = xb.run_cv(trainval, "x1_two_stage_huber")
    a = cv_result["fold_metrics"]["rmse"].to_numpy()
    b = x1_cv["fold_metrics"]["rmse"].to_numpy()
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    sig = {
        "metric": "rmse (per fold, signed target_xt_delta_v2 scale, lower is better)",
        "baseline_compared_against": "x1_two_stage_huber (standing baseline -- NOT x1b_quadratic, which did not win promotion)",
        "fold_rmse_x1c": a.tolist(), "fold_rmse_x1": b.tolist(),
        "mean_diff_x1c_minus_x1": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "clf_params": clf_params, "reg_params": reg_params,
    }
    (xb.VALIDATION_DIR / "significance_x1_vs_x1c_random_forest.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (x1c-x1) rmse: {sig['mean_diff_x1c_minus_x1']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[5/6] held-out test readout + permutation importance...")
    pred, extra = fit_predict(trainval, test, clf_params, reg_params)
    y_true = test[xb.TARGET_COL].to_numpy()
    held_out_row = xb.signed_regression_metrics(y_true, pred)
    held_out_row["variant"] = VARIANT
    held_out_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    held_out_row["rows"] = len(test)
    held_out_row["matches"] = test[xb.GROUP_COL].nunique()

    calib_rows = xb.calibration_bins(y_true, pred)

    clf_diag_holdout = classification_metrics((test[xb.TARGET_COL].to_numpy() != 0.0).astype(int),
                                               extra["clf"].predict_proba(extra["clf_builder"].transform(test)[0])[:, 1])
    print(f"  [held-out classifier diagnostic] roc_auc={clf_diag_holdout['roc_auc']:.4f} "
          f"average_precision={clf_diag_holdout['average_precision']:.4f} brier={clf_diag_holdout['brier_score']:.4f}")

    importances = fold_held_out_permutation_importance(nonzero_trainval, reg_params)
    coeff_json = {
        "variant": VARIANT, "clf_params": clf_params, "reg_params": reg_params,
        "clf_n_features": len(extra["clf_feature_names"]), "reg_n_features": len(extra["reg_feature_names"]),
        "clf_feature_importances_gini": {
            name: float(v) for name, v in zip(extra["clf_feature_names"], extra["clf"].feature_importances_)
        },
        "reg_feature_importances_gini": {
            name: float(v) for name, v in zip(extra["reg_feature_names"], extra["reg"].feature_importances_)
        },
        "reg_permutation_importance_5fold_cv": importances,
        "held_out_classifier_diagnostic": clf_diag_holdout,
        "fitted_on": "all trainval rows (canonical split); classifier on all trainval rows, "
                     "regression head on nonzero-delta trainval rows only; held-out test used once for readout only",
    }
    (xb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(coeff_json, indent=2), encoding="utf-8")
    joblib.dump(extra["clf"], xb.REG_DIR / f"{VARIANT}_classifier.joblib")
    joblib.dump(extra["reg"], xb.REG_DIR / f"{VARIANT}_regressor.joblib")

    xb.save_xt_regression_charts(xb.REG_CHARTS_DIR / VARIANT, y_true, pred, calib_rows, None)
    print(f"  [held-out] {VARIANT}: rmse={held_out_row['rmse']:.5f} mae={held_out_row['mae']:.5f} "
          f"r2={held_out_row['r2']:.5f} spearman={held_out_row['spearman']:.4f} "
          f"zero_mae={held_out_row['zero_target_mae']:.5f} nonzero_mae={held_out_row['nonzero_target_mae']:.5f}")

    print("\n[6/6] writing comparison CSVs (additive only -- x0/x1/x1b rows untouched)...")
    xb.append_to_csv(xb.COMPARISONS_DIR / "active_xt_baseline_comparison.csv", [comparison_row])
    xb.append_to_csv(xb.COMPARISONS_DIR / "active_xt_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== Calibration bins, held-out test, x1c (compare directly against x1's and x1b's tables) ===")
    for r in calib_rows:
        print(f"  bin {r['bin']}: n={r['n']} predicted={r['mean_predicted']:+.5f} actual={r['mean_actual']:+.5f} gap={r['gap']:+.5f}")

    print("\n=== TOP 10 PERMUTATION-IMPORTANCE FEATURES (regression head, nonzero-delta rows) ===")
    for r in importances[:10]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:+.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
