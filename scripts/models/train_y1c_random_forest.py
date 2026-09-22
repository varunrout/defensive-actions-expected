"""Rung 2 of the passive-xT leg (prompt 78): y1c_random_forest.

`y1b_quadratic` (Rung 1, prompt 77) did not clear `y1`: aggregate RMSE moved
in `y1b`'s favor by an amount too small to matter, and the rung's actual
motivating question -- a systematic positive prediction bias present in ALL 5
of `y1`'s calibration bins, not a tail-specific gap (Prompt 77's own Task-0
correction of a mischaracterization in Prompt 76's own report) -- got worse,
not better. This rung tests a different hypothesis: does a model-FAMILY
change (random forest, whose split-based mechanism differs fundamentally
from adding explicit quadratic/interaction TERMS to the same linear model)
reduce that bias where more features on the same linear model could not.

STEP 1 -- what does y1c replace, decided from passive's own evidence
---------------------------------------------------------------------------
Active-xT's own Rung 2 (`x1c_random_forest`, Prompt 73) replaced BOTH of
`x1`'s stages, reasoned from evidence that the SAME candidate features
carried curvature in both target quantities (zero-rate and nonzero-delta
magnitude) -- no asymmetric evidence to justify touching only one surface.
This leg's own Rung 1 (Prompt 77) found the OPPOSITE evidence pattern for
QUADRATIC TERMS specifically: passive's strongest classifier-stage signal
comes from CATEGORICAL structure (`on_ball_event_type`'s ~176x zero-rate
range), while the numeric quadratic candidates showed only weak zero-rate
curvature (~1.1-1.25x) -- the basis for Rung 1's "regression stage only"
decision, since squaring a numeric feature cannot add anything to signal
that is fundamentally categorical.

**That reasoning does not transfer to random forest, and the difference
matters**: a `RandomForestClassifier` does not need explicit quadratic/
interaction TERMS to exploit categorical structure -- it splits on any
locked feature directly, including one-hot categorical columns, and can
discover interactions AMONG categorical features (e.g. does
`on_ball_event_type`'s own effect on P(nonzero) vary by `phase_label`?) that
a plain `LogisticRegression` with only linear one-hot main effects cannot
represent at all. Rung 1's exclusion of the classifier stage was specific to
the TOOL (quadratic terms are numeric-only by construction), not a general
finding that the classifier has nothing left to gain -- if anything, the
classifier's own strong, already-established categorical signal
(Prompt 76's own Step-0 evidence, ~176x zero-rate range) is exactly the kind
of structure a tree-based split mechanism is well suited to exploit further
(categorical-x-categorical interactions), stronger reason to test it here
than to skip it.

**Decision: replace BOTH stages with RandomForest** -- `RandomForestClassifier`
for P(nonzero), `RandomForestRegressor` for E[delta|nonzero]. This lands on
the same practical answer active-xT's own `x1c` reached, but via genuinely
different, leg-specific reasoning (RF's own split mechanism vs. the
numeric-only limitation that shaped Rung 1's decision), not by assuming
active's answer transfers.

Grid, sized for THIS leg's real row counts (confirmed via `yb.load_data()`,
not assumed to match active-xT's `x1c` sizing): classifier trains on
1,275,289 trainval rows (~30x active-xT's own 45,166); regressor trains on
941,234 nonzero-only trainval rows (~26x active-xT's own 36,121). Unlike
active-xT's rows (1 row per event, independent), this leg's rows are NOT
independent -- every defender-slot row sharing an `event_id` carries the
identical target value (~8.03 rows/event, Prompt 68). `MIN_SAMPLES_LEAF_GRID`
is scaled up accordingly, mirroring `d1d_random_forest`'s own reasoning for
this exact leg-wide duplication property (that script used `[30,100,300]`
vs. the non-duplicated legs' `[10,30,100]`/`[20,50,150]`): used here,
`[50,150,500]`. `N_ESTIMATORS` trimmed to 100 (from active-xT's 200) to keep
runtime reasonable at this row count -- more trees mainly reduces variance,
and this leg's much larger sample already provides that. `MAX_DEPTH_GRID`
unchanged, `[6, 10, None]`.

Usage:
    .venv/Scripts/python.exe scripts/models/train_y1c_random_forest.py
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

import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_xt_baseline as yb  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "y1c_random_forest"
RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH_GRID = [6, 10, None]
MIN_SAMPLES_LEAF_GRID = [50, 150, 500]


class RFDesignMatrixBuilder:
    """Same categorical (one-hot)/boolean(passthrough) blocks as
    pb.DesignMatrixBuilder. Numeric features median-imputed but NOT
    standardized -- trees are scale-invariant. No engineered interaction/
    quadratic terms -- the point of this rung is what the tree finds on its
    own from the raw 38 locked features."""

    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="median")

    def fit(self, train_df: pd.DataFrame) -> "RFDesignMatrixBuilder":
        self.ohe.fit(train_df[pb.CATEGORICAL_COLS])
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(pb.CATEGORICAL_COLS))
        self.num_imputer.fit(train_df[pb.NUMERIC_COLS])
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cat_block = self.ohe.transform(df[pb.CATEGORICAL_COLS])
        cat_block = np.asarray(cat_block.todense()) if hasattr(cat_block, "todense") else np.asarray(cat_block)
        num_block = self.num_imputer.transform(df[pb.NUMERIC_COLS])
        bool_block = df[pb.BOOLEAN_COLS].astype(float).fillna(0.0).to_numpy()
        x = np.column_stack([cat_block, num_block, bool_block])
        names = [*self.cat_feature_names_, *pb.NUMERIC_COLS, *pb.BOOLEAN_COLS]
        return x, names


# ---------------------------------------------------------------------------
# Classifier stage: tune on OOF ROC-AUC
# ---------------------------------------------------------------------------

def run_cv_classifier(trainval: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(trainval, group_col=yb.GROUP_COL)
    df = trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_nonzero_all = (df[yb.TARGET_COL].to_numpy() != 0.0).astype(int)

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
# Regressor stage: tune on OOF RMSE
# ---------------------------------------------------------------------------

def run_cv_regressor(nonzero_trainval: pd.DataFrame, params: dict) -> dict:
    folds = canonical_grouped_folds(nonzero_trainval, group_col=yb.GROUP_COL)
    df = nonzero_trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[yb.TARGET_COL].to_numpy()

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

    m = yb.signed_regression_metrics(y_all, oof_pred)
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
    y_nonzero_train = (train_df[yb.TARGET_COL].to_numpy() != 0.0).astype(int)

    clf_builder = RFDesignMatrixBuilder().fit(train_df)
    x_train_clf, clf_feature_names = clf_builder.transform(train_df)
    x_test_clf, _ = clf_builder.transform(test_df)
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=clf_params["max_depth"],
        min_samples_leaf=clf_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(x_train_clf, y_nonzero_train)
    p_nonzero_test = clf.predict_proba(x_test_clf)[:, 1]

    nonzero_train_df = train_df.loc[train_df[yb.TARGET_COL] != 0.0]
    reg_builder = RFDesignMatrixBuilder().fit(nonzero_train_df)
    x_train_reg, reg_feature_names = reg_builder.transform(nonzero_train_df)
    x_test_reg, _ = reg_builder.transform(test_df)
    reg = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=reg_params["max_depth"],
        min_samples_leaf=reg_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg.fit(x_train_reg, nonzero_train_df[yb.TARGET_COL].to_numpy())
    e_delta_given_nonzero_test = reg.predict(x_test_reg)

    combined_pred = p_nonzero_test * e_delta_given_nonzero_test
    extra = {
        "clf": clf, "clf_builder": clf_builder, "clf_feature_names": clf_feature_names,
        "reg": reg, "reg_builder": reg_builder, "reg_feature_names": reg_feature_names,
    }
    return combined_pred, extra


def run_cv_combined(trainval: pd.DataFrame, clf_params: dict, reg_params: dict) -> dict:
    folds = canonical_grouped_folds(trainval, group_col=yb.GROUP_COL)
    df = trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_pred = np.full(len(df), np.nan)
    y_all = df[yb.TARGET_COL].to_numpy()

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        pred, _ = fit_predict(train_df, test_df, clf_params, reg_params)
        oof_pred[test_mask] = pred

        m = yb.signed_regression_metrics(y_all[test_mask], pred)
        m["fold"] = int(fold_id)
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    oof_metrics = yb.signed_regression_metrics(y_all, oof_pred)
    return {"fold_metrics": fold_df, "oof_metrics": oof_metrics, "df": df}


def fold_held_out_permutation_importance(nonzero_trainval: pd.DataFrame, reg_params: dict, n_repeats: int = 5) -> list[dict]:
    """n_repeats trimmed from active-xT's own 10 -- this leg's much larger
    per-fold row count makes even a small number of repeats a stable
    estimate, and keeps runtime reasonable."""
    folds = canonical_grouped_folds(nonzero_trainval, group_col=yb.GROUP_COL)
    df = nonzero_trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    importances_by_feature: dict[str, list[float]] = {}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = train_df[yb.TARGET_COL].to_numpy(), test_df[yb.TARGET_COL].to_numpy()

        builder = RFDesignMatrixBuilder().fit(train_df)
        x_train, feature_names = builder.transform(train_df)
        x_test, _ = builder.transform(test_df)
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, max_depth=reg_params["max_depth"],
            min_samples_leaf=reg_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(x_train, y_train)

        # Subsample the held-out fold for permutation importance -- this leg's
        # per-fold test size (~150-200k rows) makes a full-data permutation
        # pass unnecessarily slow for a diagnostic; a fixed random subsample
        # gives a stable estimate at a fraction of the cost.
        rng = np.random.RandomState(RANDOM_STATE)
        sub_n = min(20000, len(test_df))
        sub_idx = rng.choice(len(test_df), size=sub_n, replace=False)
        result = permutation_importance(model, x_test[sub_idx], y_test[sub_idx], scoring="neg_mean_squared_error",
                                         n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
        for name, mean_val in zip(feature_names, result.importances_mean):
            importances_by_feature.setdefault(name, []).append(float(mean_val))

    return sorted(
        ({"feature": name, "mean_permutation_importance": float(np.mean(vals)), "n_folds": len(vals)}
         for name, vals in importances_by_feature.items()),
        key=lambda r: r["mean_permutation_importance"], reverse=True,
    )


def main() -> None:
    yb.REG_DIR.mkdir(parents=True, exist_ok=True)
    yb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    yb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    yb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = yb.load_data()
    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=yb.GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test = df_all.loc[test_mask_full].reset_index(drop=True)
    nonzero_trainval = trainval.loc[trainval[yb.TARGET_COL] != 0.0].reset_index(drop=True)
    print(f"[data] trainval={len(trainval)} rows ({trainval[yb.GROUP_COL].nunique()} matches); "
          f"nonzero_trainval={len(nonzero_trainval)} rows; held-out test={len(test)} rows")

    print("\n[1/7] tuning classifier stage (RandomForestClassifier, OOF ROC-AUC)...")
    clf_tune = tune_classifier(trainval)
    (yb.VALIDATION_DIR / "y1c_classifier_tuning.json").write_text(json.dumps(clf_tune, indent=2), encoding="utf-8")
    clf_params = {"max_depth": clf_tune["best"]["max_depth"], "min_samples_leaf": clf_tune["best"]["min_samples_leaf"]}

    print("\n[2/7] tuning regressor stage (RandomForestRegressor, OOF RMSE)...")
    reg_tune = tune_regressor(nonzero_trainval)
    (yb.VALIDATION_DIR / "y1c_regressor_tuning.json").write_text(json.dumps(reg_tune, indent=2), encoding="utf-8")
    reg_params = {"max_depth": reg_tune["best"]["max_depth"], "min_samples_leaf": reg_tune["best"]["min_samples_leaf"]}

    print(f"\n[3/7] combined CV, y1c_random_forest with winning params "
          f"(clf={clf_params}, reg={reg_params})...")
    cv_result = run_cv_combined(trainval, clf_params, reg_params)
    comparison_row = {"variant": VARIANT, "n_features": 38}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(trainval)
    comparison_row["matches"] = trainval[yb.GROUP_COL].nunique()
    comparison_row["events"] = trainval[yb.EVENT_COL].nunique()
    comparison_row["clf_max_depth"] = str(clf_params["max_depth"])
    comparison_row["clf_min_samples_leaf"] = clf_params["min_samples_leaf"]
    comparison_row["reg_max_depth"] = str(reg_params["max_depth"])
    comparison_row["reg_min_samples_leaf"] = reg_params["min_samples_leaf"]

    print("\n[4/7] significance test, y1c vs y1 (per-fold RMSE, standing baseline, NOT y1b)...")
    y1_cv = yb.run_cv(trainval, "y1_two_stage_huber")
    a = cv_result["fold_metrics"]["rmse"].to_numpy()
    b = y1_cv["fold_metrics"]["rmse"].to_numpy()
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    sig = {
        "metric": "rmse (per fold, signed target_xt_delta_passive scale, lower is better)",
        "baseline_compared_against": "y1_two_stage_huber (standing baseline -- NOT y1b_quadratic, which did not win promotion)",
        "fold_rmse_y1c": a.tolist(), "fold_rmse_y1": b.tolist(),
        "mean_diff_y1c_minus_y1": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "y1c_clf_params": clf_params, "y1c_reg_params": reg_params,
    }
    (yb.VALIDATION_DIR / "significance_y1_vs_y1c_random_forest.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (y1c-y1) rmse: {sig['mean_diff_y1c_minus_y1']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[5/7] held-out test readout + permutation importance...")
    pred, extra = fit_predict(trainval, test, clf_params, reg_params)
    y_true = test[yb.TARGET_COL].to_numpy()
    held_out_row = yb.signed_regression_metrics(y_true, pred)
    held_out_row["variant"] = VARIANT
    held_out_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    held_out_row["rows"] = len(test)
    held_out_row["matches"] = test[yb.GROUP_COL].nunique()
    held_out_row["events"] = test[yb.EVENT_COL].nunique()

    calib_rows = yb.calibration_bins(y_true, pred)

    clf_diag_holdout = classification_metrics(
        (test[yb.TARGET_COL].to_numpy() != 0.0).astype(int),
        extra["clf"].predict_proba(extra["clf_builder"].transform(test)[0])[:, 1],
    )
    print(f"  [held-out classifier diagnostic] roc_auc={clf_diag_holdout['roc_auc']:.4f} "
          f"average_precision={clf_diag_holdout['average_precision']:.4f} brier={clf_diag_holdout['brier_score']:.4f}")

    print("[permutation importance] computing on fold-held-out portions (5 folds, 20k-row subsample each)...")
    perm_importance = fold_held_out_permutation_importance(nonzero_trainval, reg_params)

    gini = sorted(
        ({"feature": n, "gini_importance": float(v)} for n, v in zip(extra["reg_feature_names"], extra["reg"].feature_importances_)),
        key=lambda r: r["gini_importance"], reverse=True,
    )

    importance_output = {
        "variant": VARIANT, "clf_params": clf_params, "reg_params": reg_params,
        "clf_n_features": len(extra["clf_feature_names"]), "reg_n_features": len(extra["reg_feature_names"]),
        "reg_top15_gini_importance": gini[:15],
        "reg_full_gini_importance": gini,
        "reg_top15_permutation_importance_5fold_cv": perm_importance[:15],
        "reg_full_permutation_importance_5fold_cv": perm_importance,
        "clf_feature_importances_gini": {
            name: float(v) for name, v in zip(extra["clf_feature_names"], extra["clf"].feature_importances_)
        },
        "held_out_classifier_diagnostic": clf_diag_holdout,
        "fitted_on": "all trainval rows (canonical split); classifier on all trainval rows, "
                     "regression head on nonzero-delta trainval rows only; held-out test used once for readout only",
    }
    (yb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")
    joblib.dump(extra["clf"], yb.REG_DIR / f"{VARIANT}_classifier.joblib")
    joblib.dump(extra["reg"], yb.REG_DIR / f"{VARIANT}_regressor.joblib")

    yb.save_yxt_regression_charts(yb.REG_CHARTS_DIR / VARIANT, y_true, pred, calib_rows, None)
    # save_yxt_regression_charts writes a "no features (dummy variant)" placeholder for
    # feature_coefficients.png when coef_table is None -- overwrite it with a real Gini-
    # importance bar chart (regression head), same fix applied to x1c_random_forest.
    from dax.models.diagnostics import _ACCENT, _new_axes, _save_fixed, ensure_dir
    top = gini[:15]
    chart_dir = ensure_dir(yb.REG_CHARTS_DIR / VARIANT)
    fig, ax = _new_axes()
    ax.barh([r["feature"] for r in top][::-1], [r["gini_importance"] for r in top][::-1], color=_ACCENT)
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 15 feature importance (Gini) -- regression head")
    _save_fixed(fig, chart_dir / "feature_coefficients.png")

    print(f"  [held-out] {VARIANT}: rmse={held_out_row['rmse']:.5f} mae={held_out_row['mae']:.5f} "
          f"r2={held_out_row['r2']:.5f} spearman={held_out_row['spearman']:.4f} "
          f"zero_mae={held_out_row['zero_target_mae']:.5f} nonzero_mae={held_out_row['nonzero_target_mae']:.5f} "
          f"prediction_bias={held_out_row['prediction_bias']:+.5f}")

    print("\n[6/7] writing comparison CSVs (additive only -- y0/y1/y1b rows untouched)...")
    yb.append_to_csv(yb.COMPARISONS_DIR / "passive_xt_baseline_comparison.csv", [comparison_row])
    yb.append_to_csv(yb.COMPARISONS_DIR / "passive_xt_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n[7/7] summary...")
    print("=== Calibration bins, held-out test, y1c (compare against y1/y1b tables) ===")
    for r in calib_rows:
        print(f"  bin {r['bin']}: n={r['n']} predicted={r['mean_predicted']:+.5f} actual={r['mean_actual']:+.5f} gap={r['gap']:+.5f}")

    print("\n=== TOP 15 GINI IMPORTANCE (regression head) ===")
    for r in gini[:15]:
        print(f"  {r['feature']}: {r['gini_importance']:.4f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out, regression head) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:+.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
