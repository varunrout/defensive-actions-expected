"""Rung 4 of the model ladder (prompt 45): v1e_gradient_boosting.

Rung 3 (v1d_random_forest) found that automatic interaction discovery on
the raw 32 features beats v1c_systematic_interactions's hand-built L1
approach by a wide margin (held-out PR-AUC 0.4085 vs 0.3747), at the cost
of noticeably worse calibration (ECE 0.0183 vs 0.0103). Rung 3's own
verdict: this justifies a real gradient-boosting build, since boosting
(bias-reduction-focused) has a real chance of finding still more signal
than bagging (variance-reduction-focused) did, or of matching RF's
ranking with better-behaved probabilities.

Library choice: LightGBM. Neither LightGBM nor XGBoost was already a
project dependency; LightGBM was trivially installable (`pip install
lightgbm` pulled a pure win_amd64 wheel with no build step, no compiler
needed) and is used here rather than XGBoost for that reason. Added to
pyproject.toml's dependencies.

Design matrix: identical to v1d_random_forest -- same 32 locked features,
same categorical-one-hot/boolean-passthrough blocks, numeric features
median-imputed but not standardized, no engineered interaction columns
(reuses RFDesignMatrixBuilder from train_v1d_random_forest.py directly,
not duplicated). The point, same as rung 3, is what the model finds on
its own from the raw inputs.

Two variants:
  - v1e_gradient_boosting: the tuned LGBMClassifier, no class_weight
    (mirrors the unweighted-wins finding from every prior rung -- also
    checked explicitly with a secondary balanced comparison).
  - v1e_gradient_boosting_calibrated: the same tuned model (fixed
    n_estimators from early stopping, no further early stopping inside
    the calibration wrapper) wrapped in CalibratedClassifierCV, fit with
    a match-grouped internal CV on train+val only. Both sigmoid (Platt)
    and isotonic calibration are tried; the better one (by CV OOF ECE,
    without meaningfully worse PR-AUC) is reported as the primary
    "_calibrated" variant, with both methods' numbers recorded for
    transparency.

n_estimators is never grid-searched directly -- it's chosen per fit via
early stopping (cap 2000 rounds, early_stopping_rounds=50) on a
match-grouped validation carve-out from that fit's own training rows,
monitored on the average_precision metric. learning_rate / num_leaves /
min_child_samples ARE grid-searched on the 5 canonical CV folds
(train+val only); held-out test is read exactly once, after the grid is
fixed, for both variants.

Usage:
    python scripts/train_v1e_gradient_boosting.py
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
import train_v1c_systematic_interactions as v1c_mod  # noqa: E402
import train_v1d_random_forest as v1d_mod  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "v1e_gradient_boosting"
CALIB_VARIANT = "v1e_gradient_boosting_calibrated"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

LEARNING_RATE_GRID = [0.01, 0.05, 0.1]
NUM_LEAVES_GRID = [15, 31, 63]
MIN_CHILD_SAMPLES_GRID = [10, 30, 100]
MAX_N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
RFDesignMatrixBuilder = v1d_mod.RFDesignMatrixBuilder


def make_eval_split(df: pd.DataFrame, frac: float = 0.2, seed: int = RANDOM_STATE):
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    groups = df[tb.GROUP_COL]
    idx_train, idx_valid = next(gss.split(df, groups=groups))
    return df.iloc[idx_train], df.iloc[idx_valid]


def fit_predict_gbm(train_df, test_df, y_train, y_test, params: dict, class_weight=None):
    """Fits with early stopping on a match-grouped carve-out of train_df;
    predicts on test_df using the best iteration."""
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


def significance_vs_others(trainval_df: pd.DataFrame, v1e_fold_ap: list[float]) -> dict:
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    per_fold_ap_v1c: list[float] = []
    per_fold_ap_v1d: list[float] = []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        score_c, _, _ = v1c_mod.fit_predict_v1c(train_df, test_df, y_train, y_test, 0.1)
        per_fold_ap_v1c.append(classification_metrics(y_test, score_c)["average_precision"])

        score_d, _, _ = v1d_mod.fit_predict_rf(train_df, test_df, y_train, y_test,
                                                 {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 5})
        per_fold_ap_v1d.append(classification_metrics(y_test, score_d)["average_precision"])

    result: dict = {
        "per_fold_average_precision": {
            VARIANT: v1e_fold_ap,
            "v1c_systematic_interactions": per_fold_ap_v1c,
            "v1d_random_forest": per_fold_ap_v1d,
        },
    }
    a = np.array(v1e_fold_ap)
    for other_name, other_scores in [("v1c_systematic_interactions", per_fold_ap_v1c), ("v1d_random_forest", per_fold_ap_v1d)]:
        b = np.array(other_scores)
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        if len(set(diff.round(12))) > 1:
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float("nan"), float("nan")
        result[f"{VARIANT}_vs_{other_name}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        }
    return result


def player_disjoint_check(trainval_df: pd.DataFrame, params: dict) -> dict:
    if "player_id" not in trainval_df.columns:
        return {"error": "player_id column not found in the dataset"}

    groups = trainval_df["player_id"].astype(str)
    y = trainval_df[tb.TARGET_COL].to_numpy()
    n_players = int(groups.nunique())
    gkf = GroupKFold(n_splits=5)

    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(trainval_df, y, groups)):
        train_df, test_df = trainval_df.iloc[train_idx], trainval_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        overlap = set(train_df["player_id"]) & set(test_df["player_id"])
        assert len(overlap) == 0, f"GroupKFold leaked {len(overlap)} players into fold {fold_id}"

        score, _, _ = fit_predict_gbm(train_df, test_df, y_train, y_test, params)
        m = classification_metrics(y_test, score)
        m["fold"] = fold_id
        m["n_train"] = int(len(train_df))
        m["n_test"] = int(len(test_df))
        m["n_test_players"] = int(test_df["player_id"].nunique())
        m["player_overlap_train_test"] = int(len(overlap))
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in {"fold", "n_train", "n_test", "n_test_players", "player_overlap_train_test"}]
    return {
        "params": params,
        "n_players_total": n_players,
        "fold_metrics": fold_df.to_dict(orient="records"),
        "mean": fold_df[metric_cols].mean().to_dict(),
        "std": fold_df[metric_cols].std(ddof=0).to_dict(),
        "max_player_overlap_any_fold": int(fold_df["player_overlap_train_test"].max()),
        "player_disjoint_confirmed_every_fold": bool((fold_df["player_overlap_train_test"] == 0).all()),
    }


def fold_held_out_permutation_importance(trainval_df: pd.DataFrame, params: dict) -> list[dict]:
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

        _, model, extra = fit_predict_gbm(train_df, test_df, y_train, y_test, params)
        builder, feature_names = extra
        x_test, _ = builder.transform(test_df)

        result = permutation_importance(
            model, x_test, y_test, scoring="average_precision",
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
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

    print("[tune] grid-searching learning_rate/num_leaves/min_child_samples on 5 canonical CV folds...")
    tuning = tune_hyperparams(trainval_df)
    best_params = {k: tuning["best"][k] for k in ("learning_rate", "num_leaves", "min_child_samples")}
    cv_result = tuning["cv_by_params"][tuning["best_key"]]
    v1e_fold_ap = cv_result["fold_metrics"]["average_precision"].tolist()

    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]
    builder_probe = RFDesignMatrixBuilder().fit(trainval_df)
    n_features_probe = builder_probe.transform(trainval_df.head(3))[0].shape[1]

    comparison_row = {"variant": VARIANT, "split": "cv", "n_features": int(n_features_probe)}
    comparison_row.update({f"gbm_{k}": v for k, v in best_params.items()})
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        comparison_row[metric] = oof[metric]
        comparison_row[f"fold_{metric}_mean"] = fold_mean[metric]
        comparison_row[f"fold_{metric}_std"] = fold_std[metric]
    comparison_row["rows"] = len(trainval_df)
    comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    chart_df = pd.DataFrame({"y_true": cv_result["y_all"], "y_score": cv_result["oof_scores"]})
    chart_paths = save_classification_charts(chart_df, tb.CHARTS_DIR / VARIANT)

    print(f"[final-test-readout] {VARIANT} at {best_params}")
    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test = test_df[tb.TARGET_COL].to_numpy()
    score_test, model, extra = fit_predict_gbm(trainval_df, test_df, y_trainval, y_test, best_params)
    test_metrics = classification_metrics(y_test, score_test)
    final_best_iteration = model.best_iteration_
    print(f"[final] best_iteration_={final_best_iteration}")

    test_row = dict(test_metrics)
    test_row["variant"] = VARIANT
    test_row["split"] = "test"
    test_row["n_features"] = int(n_features_probe)
    test_row["rows"] = len(test_df)
    test_row["matches"] = test_df[tb.GROUP_COL].nunique()

    readout_row = dict(test_metrics)
    readout_row["variant"] = VARIANT
    readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    readout_row["rows"] = len(test_df)
    readout_row["matches"] = test_df[tb.GROUP_COL].nunique()

    builder, feature_names = extra
    gain_importance = sorted(
        ({"feature": n, "gain_importance": float(v)} for n, v in zip(feature_names, model.feature_importances_)),
        key=lambda r: r["gain_importance"], reverse=True,
    )
    joblib.dump(model, tb.MODELS_DIR / f"{VARIANT}.joblib")

    print("[permutation importance] computing on fold-held-out portions (5 folds, not final test)...")
    perm_importance = fold_held_out_permutation_importance(trainval_df, best_params)

    importance_output = {
        "variant": VARIANT,
        "best_params": best_params,
        "final_best_iteration": int(final_best_iteration),
        "fitted_on": "all train+val rows (canonical split); held-out test used once for readout only",
        "top15_gain_importance": gain_importance[:15],
        "top15_permutation_importance_fold_held_out": perm_importance[:15],
        "full_gain_importance": gain_importance,
        "full_permutation_importance_fold_held_out": perm_importance,
    }
    (tb.MODELS_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")

    print("[csv] appending v1e_gradient_boosting rows (cv + test)...")
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_comparison.csv", [comparison_row, test_row])
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv", [readout_row])

    # -- Secondary: class_weight comparison. --
    print("[secondary] class_weight='balanced' comparison at the same tuned hyperparameters...")
    cv_weighted = run_cv_gbm(trainval_df, best_params, class_weight="balanced")
    secondary_output = {
        "note": "Secondary comparison only -- NOT the primary v1e_gradient_boosting variant.",
        "best_params": best_params,
        "unweighted_oof_average_precision": oof["average_precision"],
        "balanced_oof_average_precision": cv_weighted["oof_metrics"]["average_precision"],
        "unweighted_oof_expected_calibration_error": oof["expected_calibration_error"],
        "balanced_oof_expected_calibration_error": cv_weighted["oof_metrics"]["expected_calibration_error"],
    }
    (VALIDATION_DIR / "v1e_class_weight_secondary_comparison.json").write_text(
        json.dumps(secondary_output, indent=2), encoding="utf-8"
    )

    # -- Calibrated variant: try sigmoid and isotonic, pick the better one. --
    print("[calibration] CV comparison: sigmoid vs isotonic (train+val only)...")
    calib_cv_results = {}
    for method in ["sigmoid", "isotonic"]:
        print(f"[calibration] method={method}")
        calib_cv_results[method] = run_cv_gbm_calibrated(trainval_df, best_params, final_best_iteration, method)

    calib_comparison = {
        method: {
            "oof_average_precision": res["oof_metrics"]["average_precision"],
            "oof_expected_calibration_error": res["oof_metrics"]["expected_calibration_error"],
            "oof_calibration_slope": res["oof_metrics"]["calibration_slope"],
            "oof_calibration_intercept": res["oof_metrics"]["calibration_intercept"],
        }
        for method, res in calib_cv_results.items()
    }
    raw_pr_auc = oof["average_precision"]
    chosen_method = min(
        calib_cv_results,
        key=lambda m: (
            calib_cv_results[m]["oof_metrics"]["expected_calibration_error"]
            if calib_cv_results[m]["oof_metrics"]["average_precision"] >= raw_pr_auc - 0.01
            else float("inf")
        ),
    )
    print(f"[calibration] chosen method for {CALIB_VARIANT}: {chosen_method} "
          f"(comparison: {json.dumps(calib_comparison, indent=2)})")

    chosen_calib_result = calib_cv_results[chosen_method]
    calib_fold_mean = chosen_calib_result["fold_mean"]
    calib_fold_std = chosen_calib_result["fold_std"]
    calib_oof = chosen_calib_result["oof_metrics"]

    calib_comparison_row = {"variant": CALIB_VARIANT, "split": "cv", "n_features": int(n_features_probe),
                             "calibration_method": chosen_method, "gbm_n_estimators_fixed": int(final_best_iteration)}
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        calib_comparison_row[metric] = calib_oof[metric]
        calib_comparison_row[f"fold_{metric}_mean"] = calib_fold_mean[metric]
        calib_comparison_row[f"fold_{metric}_std"] = calib_fold_std[metric]
    calib_comparison_row["rows"] = len(trainval_df)
    calib_comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    calib_chart_df = pd.DataFrame({"y_true": chosen_calib_result["y_all"], "y_score": chosen_calib_result["oof_scores"]})
    calib_chart_paths = save_classification_charts(calib_chart_df, tb.CHARTS_DIR / CALIB_VARIANT)

    print(f"[final-test-readout] {CALIB_VARIANT} ({chosen_method})")
    score_test_calib, model_calib, _ = fit_predict_gbm_calibrated(
        trainval_df, test_df, y_trainval, y_test, best_params, final_best_iteration, chosen_method
    )
    test_metrics_calib = classification_metrics(y_test, score_test_calib)

    calib_test_row = dict(test_metrics_calib)
    calib_test_row["variant"] = CALIB_VARIANT
    calib_test_row["split"] = "test"
    calib_test_row["n_features"] = int(n_features_probe)
    calib_test_row["calibration_method"] = chosen_method
    calib_test_row["rows"] = len(test_df)
    calib_test_row["matches"] = test_df[tb.GROUP_COL].nunique()

    calib_readout_row = dict(test_metrics_calib)
    calib_readout_row["variant"] = CALIB_VARIANT
    calib_readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    calib_readout_row["calibration_method"] = chosen_method
    calib_readout_row["rows"] = len(test_df)
    calib_readout_row["matches"] = test_df[tb.GROUP_COL].nunique()

    joblib.dump(model_calib, tb.MODELS_DIR / f"{CALIB_VARIANT}.joblib")

    print("[csv] appending v1e_gradient_boosting_calibrated rows (cv + test)...")
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_comparison.csv", [calib_comparison_row, calib_test_row])
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv", [calib_readout_row])

    calibration_summary = {
        "sigmoid_vs_isotonic_cv_comparison": calib_comparison,
        "chosen_method": chosen_method,
        "chosen_method_selection_rule": "lowest OOF ECE among methods whose OOF PR-AUC is not more than 0.01 below the raw (uncalibrated) OOF PR-AUC",
        "raw_oof_average_precision": raw_pr_auc,
        "final_n_estimators_fixed_from_early_stopping": int(final_best_iteration),
    }
    (VALIDATION_DIR / "v1e_calibration_method_comparison.json").write_text(
        json.dumps(calibration_summary, indent=2), encoding="utf-8"
    )

    # -- Gates: significance (uncalibrated v1e vs v1c and v1d) + player-disjoint. --
    print("[significance] v1e vs v1c, v1e vs v1d...")
    sig = significance_vs_others(trainval_df, v1e_fold_ap)
    sig["gbm_best_params"] = best_params
    sig["hyperparameter_grid_search"] = tuning["grid_results"]
    sig["calibrated_variant_pr_auc_vs_raw_note"] = (
        f"Calibrated ({chosen_method}) OOF PR-AUC={calib_oof['average_precision']:.4f} vs raw OOF "
        f"PR-AUC={raw_pr_auc:.4f} -- calibration is NOT assumed to leave PR-AUC untouched; measured directly."
    )
    (VALIDATION_DIR / "significance_v1e_vs_v1d_v1c.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[player-disjoint] v1e, 5-fold GroupKFold on player_id...")
    disjoint = player_disjoint_check(trainval_df, best_params)
    (VALIDATION_DIR / "player_disjoint_v1e.json").write_text(json.dumps(disjoint, indent=2, default=str), encoding="utf-8")

    print("\n=== CV (OOF) v1e_gradient_boosting (raw) ===")
    print({k: v for k, v in comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT (raw) ===")
    print(readout_row)
    print("\n=== CV (OOF) v1e_gradient_boosting_calibrated ===")
    print({k: v for k, v in calib_comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT (calibrated) ===")
    print(calib_readout_row)
    print("\n=== SECONDARY: class_weight comparison ===")
    print(secondary_output)
    print("\n=== CALIBRATION METHOD COMPARISON ===")
    print(json.dumps(calib_comparison, indent=2))
    print("\n=== SIGNIFICANCE (v1e vs v1c, v1e vs v1d) ===")
    print(sig[f"{VARIANT}_vs_v1c_systematic_interactions"])
    print(sig[f"{VARIANT}_vs_v1d_random_forest"])
    print("\n=== PLAYER-DISJOINT ===")
    print("mean:", disjoint.get("mean"))
    print("player_disjoint_confirmed_every_fold:", disjoint.get("player_disjoint_confirmed_every_fold"))
    print("\n=== TOP 15 GAIN IMPORTANCE ===")
    for r in gain_importance[:15]:
        print(f"  {r['feature']}: {r['gain_importance']:.1f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:.4f}")
    print("\nRaw charts:", [str(p) for p in chart_paths])
    print("Calibrated charts:", [str(p) for p in calib_chart_paths])
    print("Done.")


if __name__ == "__main__":
    main()
