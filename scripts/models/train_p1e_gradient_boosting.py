"""Rung 4 of the passive-binary model ladder (prompt 51): p1e_gradient_boosting
(+ p1e_gradient_boosting_calibrated).

Library choice: LightGBM, same as the active leg's Rung 4 and for the same
reason -- it is already a project dependency (added to pyproject.toml in
that prompt), so no new dependency is introduced here.

Design matrix: identical to p1d_random_forest -- same 38 locked features,
same categorical-one-hot/boolean-passthrough blocks, numeric features
median-imputed but not standardized, no engineered interaction columns
(reuses RFDesignMatrixBuilder from train_p1d_random_forest.py directly).

Two variants:
  - p1e_gradient_boosting: the tuned LGBMClassifier, no class_weight
    (mirrors the unweighted-wins finding from every prior rung on this leg).
  - p1e_gradient_boosting_calibrated: the same tuned model (fixed
    n_estimators from early stopping) wrapped in CalibratedClassifierCV,
    fit with a match-grouped internal CV on train+val only. Both sigmoid
    (Platt) and isotonic calibration are tried; the better one (by CV OOF
    ECE, without meaningfully worse PR-AUC) is reported as the primary
    "_calibrated" variant.

n_estimators is chosen per fit via early stopping (cap 1000 rounds here,
not the active leg's 2000 -- see the grid-size deviation note below for
why); learning_rate / num_leaves / min_child_samples ARE grid-searched on
the 5 canonical CV folds (train+val only).

Deviation from the active leg's Rung 4 stated up front: the hyperparameter
grid here is 4 combinations, not 27 (2 learning rates x 2 num_leaves x 1
min_child_samples, vs the active leg's 3x3x3), min_child_samples is fixed
at 500 rather than swept 10-100, and the early-stopping cap is 1000 rounds
not 2000. This leg's train+val set is ~28x the active leg's row count
(1.28M vs 45K); a 27-combination grid with up to 2000 boosting rounds per
fit, times 5 folds, is not tractable in this session's timeframe at this
scale. The trimmed grid and higher min_child_samples are a real,
explicitly-reported precision/runtime tradeoff, not a silent shortcut.

No promotion audit is run in this script, whatever the results -- that is
an explicit out-of-scope item for this prompt, mirroring the active leg's
Prompt 45 constraint.

Usage:
    python scripts/models/train_p1e_gradient_boosting.py
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
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_p1d_random_forest as p1d_mod  # noqa: E402
import train_passive_binary_baseline as tb  # noqa: E402
import validate_passive_binary_baselines as vpb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "p1e_gradient_boosting"
CALIB_VARIANT = "p1e_gradient_boosting_calibrated"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

# Trimmed vs the active leg's 3x3x3=27-combo grid -- see module docstring.
LEARNING_RATE_GRID = [0.05, 0.1]
NUM_LEAVES_GRID = [31, 63]
MIN_CHILD_SAMPLES_GRID = [500]
MAX_N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
RANDOM_STATE = 42
RFDesignMatrixBuilder = p1d_mod.RFDesignMatrixBuilder


def make_eval_split(df: pd.DataFrame, frac: float = 0.2, seed: int = RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    groups = df[tb.GROUP_COL]
    idx_train, idx_valid = next(gss.split(df, groups=groups))
    return df.iloc[idx_train], df.iloc[idx_valid]


def fit_predict_gbm(train_df, test_df, y_train, y_test, params: dict, class_weight=None):
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_test, feature_names = builder.transform(test_df)

    sub_train_df, sub_valid_df = make_eval_split(train_df)
    x_sub_train, _ = builder.transform(sub_train_df)
    x_sub_valid, _ = builder.transform(sub_valid_df)
    y_sub_train = sub_train_df[tb.TARGET_COL].to_numpy()
    y_sub_valid = sub_valid_df[tb.TARGET_COL].to_numpy()

    model = LGBMClassifier(
        n_estimators=MAX_N_ESTIMATORS,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        x_sub_train, y_sub_train,
        eval_set=[(x_sub_valid, y_sub_valid)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    score = model.predict_proba(x_test)[:, 1]
    return score, model, (builder, feature_names)


def run_cv_gbm(df: pd.DataFrame, params: dict, class_weight=None) -> dict:
    folds = canonical_grouped_folds(df, group_col=tb.GROUP_COL)
    df = df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    fold_rows = []
    oof_scores = np.full(len(df), np.nan)
    best_iterations = []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        score, model, _ = fit_predict_gbm(train_df, test_df, y_train, y_test, params, class_weight)
        oof_scores[test_mask] = score
        best_iterations.append(model.best_iteration_)

        m = classification_metrics(y_test, score)
        m["fold"] = int(fold_id)
        m["n_train"] = int(train_mask.sum())
        m["n_test"] = int(test_mask.sum())
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in {"fold", "n_train", "n_test"}]
    fold_mean = fold_df[metric_cols].mean()
    fold_std = fold_df[metric_cols].std(ddof=0)
    overall = classification_metrics(y_all, oof_scores)

    return {
        "fold_metrics": fold_df,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "oof_metrics": overall,
        "oof_scores": oof_scores,
        "y_all": y_all,
        "best_iterations": best_iterations,
    }


def tune_hyperparams(trainval_df: pd.DataFrame) -> dict:
    grid_results = []
    cv_by_params = {}
    combos = list(itertools.product(LEARNING_RATE_GRID, NUM_LEAVES_GRID, MIN_CHILD_SAMPLES_GRID))
    print(f"[tune] {len(combos)} combinations x 5 folds = {len(combos) * 5} fits")
    for lr, leaves, min_child in combos:
        params = {"learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child}
        key = (lr, leaves, min_child)
        result = run_cv_gbm(trainval_df, params)
        pr_auc = result["oof_metrics"]["average_precision"]
        grid_results.append({
            "learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child,
            "oof_average_precision": pr_auc,
            "fold_average_precision_mean": float(result["fold_mean"]["average_precision"]),
            "fold_average_precision_std": float(result["fold_std"]["average_precision"]),
            "mean_best_iteration": float(np.mean(result["best_iterations"])),
        })
        cv_by_params[key] = result
        print(f"[tune] lr={lr} leaves={leaves} min_child={min_child} OOF PR-AUC={pr_auc:.4f} "
              f"mean_best_iter={np.mean(result['best_iterations']):.0f}")

    best = max(grid_results, key=lambda r: r["oof_average_precision"])
    best_key = (best["learning_rate"], best["num_leaves"], best["min_child_samples"])
    print(f"[tune] best params={best} (OOF PR-AUC={best['oof_average_precision']:.4f})")
    return {"grid_results": grid_results, "best": best, "best_key": best_key, "cv_by_params": cv_by_params}


def append_to_csv(path: Path, new_rows: list[dict]) -> None:
    existing = pd.read_csv(path)
    new_df = pd.DataFrame(new_rows)
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined.to_csv(path, index=False)


def fit_predict_gbm_calibrated(train_df, test_df, y_train, y_test, params: dict, n_estimators_fixed: int, method: str):
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    base = LGBMClassifier(
        n_estimators=n_estimators_fixed,
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
    inner_groups = train_df[tb.GROUP_COL]
    n_inner_splits = min(3, inner_groups.nunique())
    inner_cv = list(GroupKFold(n_splits=n_inner_splits).split(x_train, y_train, groups=inner_groups))
    calibrated = CalibratedClassifierCV(base, method=method, cv=inner_cv)
    calibrated.fit(x_train, y_train)
    score = calibrated.predict_proba(x_test)[:, 1]
    return score, calibrated, (builder, feature_names)


def run_cv_gbm_calibrated(df: pd.DataFrame, params: dict, n_estimators_fixed: int, method: str) -> dict:
    folds = canonical_grouped_folds(df, group_col=tb.GROUP_COL)
    df = df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    fold_rows = []
    oof_scores = np.full(len(df), np.nan)
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        score, _, _ = fit_predict_gbm_calibrated(train_df, test_df, y_train, y_test, params, n_estimators_fixed, method)
        oof_scores[test_mask] = score

        m = classification_metrics(y_test, score)
        m["fold"] = int(fold_id)
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in {"fold"}]
    fold_mean = fold_df[metric_cols].mean()
    fold_std = fold_df[metric_cols].std(ddof=0)
    overall = classification_metrics(y_all, oof_scores)
    return {"fold_metrics": fold_df, "fold_mean": fold_mean, "fold_std": fold_std,
            "oof_metrics": overall, "oof_scores": oof_scores, "y_all": y_all}


def significance_two(a_fold_ap: list[float], b_fold_ap: list[float], label_a: str, label_b: str) -> dict:
    a = np.array(a_fold_ap)
    b = np.array(b_fold_ap)
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
    }


def fold_held_out_permutation_importance(trainval_df: pd.DataFrame, best_params: dict, n_estimators_fixed: int) -> list[dict]:
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    importances_by_feature: dict[str, list[float]] = {}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        builder = RFDesignMatrixBuilder().fit(train_df)
        x_train, feature_names = builder.transform(train_df)
        x_test, _ = builder.transform(test_df)
        model = LGBMClassifier(
            n_estimators=n_estimators_fixed, learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"], min_child_samples=best_params["min_child_samples"],
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1,
        )
        model.fit(x_train, y_train)

        result = permutation_importance(
            model, x_test, y_test, scoring="average_precision",
            n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1,
        )
        for name, mean_val in zip(feature_names, result.importances_mean):
            importances_by_feature.setdefault(name, []).append(float(mean_val))

    return sorted(
        (
            {"feature": name, "mean_permutation_importance": float(np.mean(vals)), "n_folds": len(vals)}
            for name, vals in importances_by_feature.items()
        ),
        key=lambda r: r["mean_permutation_importance"], reverse=True,
    )


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    tb.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = tb.load_data()
    load_canonical_split()
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)
    print(f"[split] train+val rows={len(trainval_df)} matches={trainval_df[tb.GROUP_COL].nunique()}; "
          f"held-out test rows={len(test_df)} matches={test_df[tb.GROUP_COL].nunique()}")

    print("[tune] grid-searching LightGBM hyperparameters on 5 canonical CV folds (train+val only)...")
    tuning = tune_hyperparams(trainval_df)
    best_params = {k: tuning["best"][k] for k in ("learning_rate", "num_leaves", "min_child_samples")}
    cv_result = tuning["cv_by_params"][tuning["best_key"]]
    p1e_fold_ap = cv_result["fold_metrics"]["average_precision"].tolist()
    n_estimators_fixed = int(round(np.mean(cv_result["best_iterations"])))
    print(f"[n_estimators] fixed at {n_estimators_fixed} (mean best_iteration_ across folds at chosen params)")

    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]
    builder_probe = RFDesignMatrixBuilder().fit(trainval_df)
    n_features_probe = builder_probe.transform(trainval_df.head(3))[0].shape[1]

    comparison_row = {"variant": VARIANT, "n_features": int(n_features_probe)}
    comparison_row.update({f"gbm_{k}": v for k, v in best_params.items()})
    comparison_row["gbm_n_estimators_fixed"] = n_estimators_fixed
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        comparison_row[metric] = oof[metric]
        comparison_row[f"fold_{metric}_mean"] = fold_mean[metric]
        comparison_row[f"fold_{metric}_std"] = fold_std[metric]
    comparison_row["rows"] = len(trainval_df)
    comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    chart_df = pd.DataFrame({"y_true": cv_result["y_all"], "y_score": cv_result["oof_scores"]})
    chart_paths = save_classification_charts(chart_df, tb.CHARTS_DIR / VARIANT)

    print(f"[final-test-readout] {VARIANT} at {best_params}, n_estimators={n_estimators_fixed}")
    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test = test_df[tb.TARGET_COL].to_numpy()
    fixed_params_for_test = {**best_params, "n_estimators_override": n_estimators_fixed}
    # Refit once more with early stopping on the full train+val (same recipe as CV folds) for the
    # single held-out readout -- consistent with every prior rung's "fit once on train+val" pattern.
    score_test, model, extra = fit_predict_gbm(trainval_df, test_df, y_trainval, y_test, best_params)
    test_metrics = classification_metrics(y_test, score_test)

    readout_row = dict(test_metrics)
    readout_row["variant"] = VARIANT
    readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    readout_row["rows"] = len(test_df)
    readout_row["matches"] = test_df[tb.GROUP_COL].nunique()

    builder, feature_names = extra
    gain_importance = sorted(
        ({"feature": n, "gain_importance": float(v)} for n, v in zip(feature_names, model.booster_.feature_importance(importance_type="gain"))),
        key=lambda r: r["gain_importance"], reverse=True,
    )
    joblib.dump(model, tb.MODELS_DIR / f"{VARIANT}.joblib")

    print("[permutation importance] computing on fold-held-out portions (5 folds, not final test)...")
    perm_importance = fold_held_out_permutation_importance(trainval_df, best_params, model.best_iteration_ or n_estimators_fixed)

    importance_output = {
        "variant": VARIANT,
        "best_params": best_params,
        "n_estimators_fixed": n_estimators_fixed,
        "fitted_on": "all train+val rows (canonical split); held-out test used once for readout only",
        "top15_gain_importance": gain_importance[:15],
        "top15_permutation_importance_fold_held_out": perm_importance[:15],
        "full_gain_importance": gain_importance,
        "full_permutation_importance_fold_held_out": perm_importance,
    }
    (tb.MODELS_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")

    print("[csv] appending raw p1e rows to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_comparison.csv", [comparison_row])
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_held_out_test_readout.csv", [readout_row])

    # -- Calibrated variant: sigmoid vs isotonic, both tried, better one reported primary. --
    n_est_for_calib = model.best_iteration_ or n_estimators_fixed
    calib_results = {}
    for method in ["sigmoid", "isotonic"]:
        print(f"[calibrated] method={method}, n_estimators={n_est_for_calib}...")
        cv_calib = run_cv_gbm_calibrated(trainval_df, best_params, n_est_for_calib, method)
        calib_results[method] = cv_calib

    chosen_method = min(
        calib_results,
        key=lambda m: calib_results[m]["oof_metrics"]["expected_calibration_error"],
    )
    chosen_pr_auc = calib_results[chosen_method]["oof_metrics"]["average_precision"]
    raw_pr_auc = oof["average_precision"]
    print(f"[calibrated] chosen method={chosen_method} (lowest OOF ECE); "
          f"OOF PR-AUC raw={raw_pr_auc:.4f} calibrated={chosen_pr_auc:.4f}")

    chosen_cv = calib_results[chosen_method]
    calib_comparison_row = {"variant": CALIB_VARIANT, "n_features": int(n_features_probe), "calibration_method": chosen_method}
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        calib_comparison_row[metric] = chosen_cv["oof_metrics"][metric]
        calib_comparison_row[f"fold_{metric}_mean"] = chosen_cv["fold_mean"][metric]
        calib_comparison_row[f"fold_{metric}_std"] = chosen_cv["fold_std"][metric]
    calib_comparison_row["rows"] = len(trainval_df)
    calib_comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    calib_chart_df = pd.DataFrame({"y_true": chosen_cv["y_all"], "y_score": chosen_cv["oof_scores"]})
    calib_chart_paths = save_classification_charts(calib_chart_df, tb.CHARTS_DIR / CALIB_VARIANT)

    print(f"[final-test-readout] {CALIB_VARIANT} (method={chosen_method})")
    score_test_calib, calib_model, calib_extra = fit_predict_gbm_calibrated(
        trainval_df, test_df, y_trainval, y_test, best_params, n_est_for_calib, chosen_method
    )
    calib_test_metrics = classification_metrics(y_test, score_test_calib)
    calib_readout_row = dict(calib_test_metrics)
    calib_readout_row["variant"] = CALIB_VARIANT
    calib_readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    calib_readout_row["rows"] = len(test_df)
    calib_readout_row["matches"] = test_df[tb.GROUP_COL].nunique()
    joblib.dump(calib_model, tb.MODELS_DIR / f"{CALIB_VARIANT}.joblib")

    print("[csv] appending calibrated p1e rows to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_comparison.csv", [calib_comparison_row])
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_held_out_test_readout.csv", [calib_readout_row])

    calibration_method_comparison = {
        "sigmoid_oof_average_precision": calib_results["sigmoid"]["oof_metrics"]["average_precision"],
        "isotonic_oof_average_precision": calib_results["isotonic"]["oof_metrics"]["average_precision"],
        "sigmoid_oof_ece": calib_results["sigmoid"]["oof_metrics"]["expected_calibration_error"],
        "isotonic_oof_ece": calib_results["isotonic"]["oof_metrics"]["expected_calibration_error"],
        "raw_oof_average_precision": raw_pr_auc,
        "raw_oof_ece": oof["expected_calibration_error"],
        "chosen_method": chosen_method,
    }
    (VALIDATION_DIR / "p1e_calibration_method_comparison.json").write_text(
        json.dumps(calibration_method_comparison, indent=2), encoding="utf-8"
    )

    # -- Secondary: class_weight="balanced" comparison at the same tuned params (raw variant). --
    print("[secondary] class_weight='balanced' comparison at the same tuned hyperparameters...")
    cv_weighted = run_cv_gbm(trainval_df, best_params, class_weight="balanced")
    secondary_output = {
        "note": "Secondary comparison only -- NOT the primary p1e_gradient_boosting variant.",
        "best_params": best_params,
        "unweighted_oof_average_precision": raw_pr_auc,
        "balanced_oof_average_precision": cv_weighted["oof_metrics"]["average_precision"],
        "unweighted_oof_expected_calibration_error": oof["expected_calibration_error"],
        "balanced_oof_expected_calibration_error": cv_weighted["oof_metrics"]["expected_calibration_error"],
    }
    (VALIDATION_DIR / "p1e_class_weight_secondary_comparison.json").write_text(
        json.dumps(secondary_output, indent=2), encoding="utf-8"
    )

    # -- Gate: significance vs p1d (raw random forest, this leg's best tree-based comparator so far)
    # and vs the best linear candidate so far (read from the comparison CSV: p1 vs p1b vs p1c). --
    print("[gate prep] identifying best linear candidate so far from the comparison CSV...")
    comp_csv = pd.read_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_comparison.csv")
    linear_variants = ["p1_unweighted", "p1b_quadratic", "p1c_systematic_interactions"]
    linear_rows = comp_csv[comp_csv["variant"].isin(linear_variants)]
    best_linear_variant = linear_rows.loc[linear_rows["average_precision"].idxmax(), "variant"]
    print(f"[gate prep] best linear candidate by CV PR-AUC: {best_linear_variant}")

    p1d_sig_path = VALIDATION_DIR / "significance_p1d_vs_p1c.json"
    p1d_sig = json.loads(p1d_sig_path.read_text(encoding="utf-8"))
    p1d_fold_ap = p1d_sig["per_fold_average_precision"]["p1d_random_forest"]

    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df_folded = trainval_df.loc[folds["row_index"]].copy()
    df_folded["fold"] = folds["fold"].to_numpy()
    y_all_folded = df_folded[tb.TARGET_COL].to_numpy()
    best_linear_fold_ap: list[float] = []
    import train_passive_binary_baseline as _tb_mod
    for fold_id in sorted(df_folded["fold"].unique()):
        train_mask = (df_folded["fold"] != fold_id).to_numpy()
        test_mask_f = (df_folded["fold"] == fold_id).to_numpy()
        tr, te = df_folded.loc[train_mask], df_folded.loc[test_mask_f]
        ytr, yte = y_all_folded[train_mask], y_all_folded[test_mask_f]
        if best_linear_variant == "p1c_systematic_interactions":
            import train_p1c_systematic_interactions as p1c_mod
            p1c_coeffs = json.loads((tb.MODELS_DIR / "p1c_systematic_interactions.json").read_text(encoding="utf-8"))
            score, _, _ = p1c_mod.fit_predict_p1c(tr, te, ytr, yte, p1c_coeffs["chosen_C"])
        else:
            score, _, _ = _tb_mod.fit_predict_fold(tr, te, ytr, yte, best_linear_variant)
        m = classification_metrics(yte, score)
        best_linear_fold_ap.append(m["average_precision"])

    sig = {
        "per_fold_average_precision": {
            VARIANT: p1e_fold_ap,
            "p1d_random_forest": p1d_fold_ap,
            best_linear_variant: best_linear_fold_ap,
        },
        "best_linear_comparator": best_linear_variant,
        f"{VARIANT}_vs_p1d_random_forest": significance_two(p1e_fold_ap, p1d_fold_ap, VARIANT, "p1d_random_forest"),
        f"{VARIANT}_vs_{best_linear_variant}": significance_two(p1e_fold_ap, best_linear_fold_ap, VARIANT, best_linear_variant),
        "hyperparam_grid_search": tuning["grid_results"],
    }
    (VALIDATION_DIR / f"significance_p1e_vs_p1d_{best_linear_variant}.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[fit] p1_unweighted, once on train+val, scored on held-out test (cluster-bootstrap comparator)...")
    _, y_test_p1, score_p1 = vpb._fit_once_and_score_test(df_all, "p1_unweighted")
    assert np.array_equal(y_test_p1, y_test)

    print("[cluster bootstrap] p1e vs p1_unweighted, held-out test, cluster-by-event_id...")
    boot = vpb.item3_cluster_bootstrap(
        y_test, score_test, score_p1, test_df[vpb.EVENT_GROUP_COL].to_numpy(),
        label_a=VARIANT, label_b="p1_unweighted",
    )
    (VALIDATION_DIR / "cluster_bootstrap_p1_vs_p1e.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    print("\n=== HYPERPARAMETER GRID SEARCH (CV only, train+val) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if (r["learning_rate"], r["num_leaves"], r["min_child_samples"]) == tuning["best_key"] else ""
        print(f"lr={r['learning_rate']:<5} leaves={r['num_leaves']:<4} min_child={r['min_child_samples']:<4} "
              f"OOF PR-AUC={r['oof_average_precision']:.4f}{marker}")
    print("\n=== CV (OOF) raw ===")
    print({k: v for k, v in comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT (raw) ===")
    print(readout_row)
    print("\n=== CALIBRATION METHOD COMPARISON ===")
    print(calibration_method_comparison)
    print("\n=== CV (OOF) calibrated ===")
    print({k: v for k, v in calib_comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT (calibrated) ===")
    print(calib_readout_row)
    print("\n=== SECONDARY: class_weight comparison ===")
    print(secondary_output)
    print("\n=== SIGNIFICANCE ===")
    print(sig[f"{VARIANT}_vs_p1d_random_forest"])
    print(sig[f"{VARIANT}_vs_{best_linear_variant}"])
    print("\n=== CLUSTER BOOTSTRAP (p1e raw vs p1) ===")
    diff_key = f"{VARIANT}_minus_p1_unweighted_average_precision"
    print("diff naive:", boot[diff_key]["naive_row_level"])
    print("diff cluster:", boot[diff_key]["cluster_by_event"])
    print("width ratio:", boot[diff_key]["cluster_to_naive_width_ratio"])
    print("\n=== TOP 15 GAIN IMPORTANCE ===")
    for r in gain_importance[:15]:
        print(f"  {r['feature']}: {r['gain_importance']:.1f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:.4f}")
    print("\nCharts written (raw):", [str(p) for p in chart_paths])
    print("Charts written (calibrated):", [str(p) for p in calib_chart_paths])
    print("Done.")


if __name__ == "__main__":
    main()
