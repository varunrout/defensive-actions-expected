"""Rung 3 of the passive-binary model ladder (prompt 51): p1d_random_forest.

p1c_systematic_interactions (Rung 2) found its gain (if any -- see that
rung's own report) by explicitly engineering pairwise/squared terms from
this leg's 28 numeric features and letting L1 prune them. This rung asks
the question that approach can't answer by itself: does a model that
discovers interactions and non-linearity automatically, straight from the
RAW 38 locked features, find meaningfully more signal than the
hand-engineered approach did?

Random Forest is the diagnostic probe (not a promotion candidate in its own
right, mirroring the active leg's Rung 3 framing): no manual feature
engineering, handles interactions/non-linearity natively via tree splits.

Design matrix: same 38 locked features, same categorical-one-hot /
boolean-passthrough blocks as p1, numeric features median-imputed but NOT
standardized -- and none of p1c's engineered interaction/polynomial
columns. No player_id-based feature at any point (this leg has none).

Primary variant reported as p1d_random_forest: no class_weight (mirrors
the unweighted-wins-on-PR-AUC finding from p1 vs p2/p3). A secondary
class_weight="balanced_subsample" comparison is run at the same tuned
hyperparameters, reported alongside but not as the primary variant.

Deviation from the active leg's Rung 3 stated up front: the hyperparameter
grid here is much smaller (3 combinations, not 12) and min_samples_leaf is
set much higher (200, not 5-20). This leg's train+val set is ~28x the
active leg's row count (1.28M vs 45K rows); a RandomForestClassifier with
300-600 trees and a min_samples_leaf as low as 5 on this many rows is not
tractable in this session's timeframe, and would also badly overfit
individual leaves at this row count regardless. The grid was trimmed and
regularized more aggressively than the active leg's, which is itself a
real finding about how tree hyperparameters should scale with dataset
size, reported here rather than silently applied.

Usage:
    python scripts/models/train_p1d_random_forest.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_p1c_systematic_interactions as p1c_mod  # noqa: E402
import train_passive_binary_baseline as tb  # noqa: E402
import validate_passive_binary_baselines as vpb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "p1d_random_forest"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

# Trimmed vs the active leg's 2x3x2=12-combo grid -- see module docstring.
HYPERPARAM_GRID = [
    {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 200},
    {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 200},
    {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 500},
]
RANDOM_STATE = 42
# The C1c comparator's chosen C is read from its own significance JSON
# (written by train_p1c_systematic_interactions.py) rather than hardcoded,
# so this script never silently assumes a stale tuning result.
P1C_SIGNIFICANCE_PATH = VALIDATION_DIR / "significance_p1c_vs_p1_p1b.json"


class RFDesignMatrixBuilder:
    """Same categorical (one-hot) / boolean (passthrough) blocks as p1's
    DesignMatrixBuilder. Numeric features are median-imputed but NOT
    standardized. No engineered interaction terms."""

    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="median")

    def fit(self, train_df: pd.DataFrame) -> "RFDesignMatrixBuilder":
        self.ohe.fit(train_df[tb.CATEGORICAL_COLS])
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(tb.CATEGORICAL_COLS))
        self.num_imputer.fit(train_df[tb.NUMERIC_COLS])
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cat_block = self.ohe.transform(df[tb.CATEGORICAL_COLS])
        cat_block = np.asarray(cat_block.todense()) if hasattr(cat_block, "todense") else np.asarray(cat_block)
        num_block = self.num_imputer.transform(df[tb.NUMERIC_COLS])
        bool_block = df[tb.BOOLEAN_COLS].astype(float).fillna(0.0).to_numpy()
        x = np.column_stack([cat_block, num_block, bool_block])
        names = [*self.cat_feature_names_, *tb.NUMERIC_COLS, *tb.BOOLEAN_COLS]
        return x, names


def fit_predict_rf(train_df, test_df, y_train, y_test, params: dict, class_weight=None):
    builder = RFDesignMatrixBuilder().fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    return score, model, (builder, feature_names)


def run_cv_rf(df: pd.DataFrame, params: dict, class_weight=None) -> dict:
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

        t0 = time.time()
        score, _, _ = fit_predict_rf(train_df, test_df, y_train, y_test, params, class_weight)
        print(f"    fold {fold_id} ({params}) fit in {time.time() - t0:.1f}s")
        oof_scores[test_mask] = score

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
    }


def tune_hyperparams(trainval_df: pd.DataFrame) -> dict:
    grid_results = []
    cv_by_params = {}
    for params in HYPERPARAM_GRID:
        key = (params["n_estimators"], params["max_depth"], params["min_samples_leaf"])
        print(f"[tune] {params}")
        result = run_cv_rf(trainval_df, params)
        pr_auc = result["oof_metrics"]["average_precision"]
        grid_results.append({
            **params,
            "oof_average_precision": pr_auc,
            "fold_average_precision_mean": float(result["fold_mean"]["average_precision"]),
            "fold_average_precision_std": float(result["fold_std"]["average_precision"]),
        })
        cv_by_params[key] = result

    best = max(grid_results, key=lambda r: r["oof_average_precision"])
    best_key = (best["n_estimators"], best["max_depth"], best["min_samples_leaf"])
    print(f"[tune] best params={best} (OOF PR-AUC={best['oof_average_precision']:.4f})")
    return {"grid_results": grid_results, "best_params": best, "best_key": best_key, "cv_by_params": cv_by_params}


def append_to_csv(path: Path, new_rows: list[dict]) -> None:
    existing = pd.read_csv(path)
    new_df = pd.DataFrame(new_rows)
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined.to_csv(path, index=False)


def significance_vs_p1c(trainval_df: pd.DataFrame, p1d_fold_ap: list[float], p1c_chosen_c: float) -> dict:
    """Paired fold-level PR-AUC comparison: p1d vs p1c, on the same 5
    canonical CV folds."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    per_fold_ap_p1c: list[float] = []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        score, _, _ = p1c_mod.fit_predict_p1c(train_df, test_df, y_train, y_test, p1c_chosen_c)
        m = classification_metrics(y_test, score)
        per_fold_ap_p1c.append(m["average_precision"])

    a = np.array(p1d_fold_ap)
    b = np.array(per_fold_ap_p1c)
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "per_fold_average_precision": {VARIANT: p1d_fold_ap, "p1c_systematic_interactions": per_fold_ap_p1c},
        "p1c_chosen_c": p1c_chosen_c,
        f"{VARIANT}_vs_p1c_systematic_interactions": {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        },
    }


def fold_held_out_permutation_importance(trainval_df: pd.DataFrame, best_params: dict) -> list[dict]:
    """Permutation importance computed on each fold's held-out portion
    (never the final held-out test set), averaged across folds."""
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
        model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"], random_state=RANDOM_STATE, n_jobs=-1,
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

    # Read the actually-chosen C from p1c's own coefficient JSON (ground truth
    # of what was fit), not re-derived or assumed.
    assert P1C_SIGNIFICANCE_PATH.exists(), f"Rung 2 (p1c) must run and commit before Rung 3 -- {P1C_SIGNIFICANCE_PATH} not found"
    p1c_coeffs = json.loads((tb.MODELS_DIR / "p1c_systematic_interactions.json").read_text(encoding="utf-8"))
    p1c_chosen_c = p1c_coeffs["chosen_C"]
    print(f"[p1c comparator] chosen_C={p1c_chosen_c} (read from p1c_systematic_interactions.json)")

    print("[tune] grid-searching RF hyperparameters on 5 canonical CV folds (train+val only)...")
    tuning = tune_hyperparams(trainval_df)
    best_params = {k: tuning["best_params"][k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
    cv_result = tuning["cv_by_params"][tuning["best_key"]]
    p1d_fold_ap = cv_result["fold_metrics"]["average_precision"].tolist()

    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]
    builder_probe = RFDesignMatrixBuilder().fit(trainval_df)
    n_features_probe = builder_probe.transform(trainval_df.head(3))[0].shape[1]

    comparison_row = {"variant": VARIANT, "n_features": int(n_features_probe)}
    comparison_row.update({f"rf_{k}": v for k, v in best_params.items()})
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
    score_test, model, extra = fit_predict_rf(trainval_df, test_df, y_trainval, y_test, best_params)
    test_metrics = classification_metrics(y_test, score_test)

    readout_row = dict(test_metrics)
    readout_row["variant"] = VARIANT
    readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    readout_row["rows"] = len(test_df)
    readout_row["matches"] = test_df[tb.GROUP_COL].nunique()

    builder, feature_names = extra
    gini = sorted(
        ({"feature": n, "gini_importance": float(v)} for n, v in zip(feature_names, model.feature_importances_)),
        key=lambda r: r["gini_importance"], reverse=True,
    )
    joblib.dump(model, tb.MODELS_DIR / f"{VARIANT}.joblib")

    print("[permutation importance] computing on fold-held-out portions (5 folds, not final test)...")
    perm_importance = fold_held_out_permutation_importance(trainval_df, best_params)

    importance_output = {
        "variant": VARIANT,
        "best_params": best_params,
        "fitted_on": "all train+val rows (canonical split); held-out test used once for readout only",
        "top15_gini_importance": gini[:15],
        "top15_permutation_importance_fold_held_out": perm_importance[:15],
        "full_gini_importance": gini,
        "full_permutation_importance_fold_held_out": perm_importance,
    }
    (tb.MODELS_DIR / f"{VARIANT}.json").write_text(json.dumps(importance_output, indent=2), encoding="utf-8")

    print("[csv] appending rows to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_comparison.csv", [comparison_row])
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_held_out_test_readout.csv", [readout_row])

    print("[secondary] class_weight='balanced_subsample' comparison at the same tuned hyperparameters...")
    cv_weighted = run_cv_rf(trainval_df, best_params, class_weight="balanced_subsample")
    secondary_output = {
        "note": "Secondary comparison only -- NOT the primary p1d_random_forest variant.",
        "best_params": best_params,
        "unweighted_oof_average_precision": oof["average_precision"],
        "balanced_subsample_oof_average_precision": cv_weighted["oof_metrics"]["average_precision"],
        "unweighted_oof_expected_calibration_error": oof["expected_calibration_error"],
        "balanced_subsample_oof_expected_calibration_error": cv_weighted["oof_metrics"]["expected_calibration_error"],
    }
    (VALIDATION_DIR / "p1d_class_weight_secondary_comparison.json").write_text(
        json.dumps(secondary_output, indent=2), encoding="utf-8"
    )

    print("[significance] p1d vs p1c...")
    sig = significance_vs_p1c(trainval_df, p1d_fold_ap, p1c_chosen_c)
    sig["hyperparam_grid_search"] = tuning["grid_results"]
    (VALIDATION_DIR / "significance_p1d_vs_p1c.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[fit] p1_unweighted, once on train+val, scored on held-out test (cluster-bootstrap comparator)...")
    _, y_test_p1, score_p1 = vpb._fit_once_and_score_test(df_all, "p1_unweighted")
    assert np.array_equal(y_test_p1, y_test)

    print("[cluster bootstrap] p1d vs p1_unweighted, held-out test, cluster-by-event_id...")
    boot = vpb.item3_cluster_bootstrap(
        y_test, score_test, score_p1, test_df[vpb.EVENT_GROUP_COL].to_numpy(),
        label_a=VARIANT, label_b="p1_unweighted",
    )
    (VALIDATION_DIR / "cluster_bootstrap_p1_vs_p1d.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    print("\n=== HYPERPARAMETER GRID SEARCH (CV only, train+val) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if (r["n_estimators"], r["max_depth"], r["min_samples_leaf"]) == tuning["best_key"] else ""
        print(f"n_est={r['n_estimators']:<5} max_depth={str(r['max_depth']):<5} min_leaf={r['min_samples_leaf']:<4} "
              f"OOF PR-AUC={r['oof_average_precision']:.4f}{marker}")
    print("\n=== CV (OOF) at chosen params ===")
    print({k: v for k, v in comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT ===")
    print(readout_row)
    print("\n=== SECONDARY: class_weight comparison ===")
    print(secondary_output)
    print("\n=== SIGNIFICANCE (p1d vs p1c) ===")
    print(sig[f"{VARIANT}_vs_p1c_systematic_interactions"])
    print("\n=== CLUSTER BOOTSTRAP (p1d vs p1) ===")
    diff_key = f"{VARIANT}_minus_p1_unweighted_average_precision"
    print("diff naive:", boot[diff_key]["naive_row_level"])
    print("diff cluster:", boot[diff_key]["cluster_by_event"])
    print("width ratio:", boot[diff_key]["cluster_to_naive_width_ratio"])
    print("\n=== TOP 15 GINI IMPORTANCE ===")
    for r in gini[:15]:
        print(f"  {r['feature']}: {r['gini_importance']:.4f}")
    print("\n=== TOP 15 PERMUTATION IMPORTANCE (fold held-out) ===")
    for r in perm_importance[:15]:
        print(f"  {r['feature']}: {r['mean_permutation_importance']:.4f}")
    print("\nCharts written:", [str(p) for p in chart_paths])
    print("Done.")


if __name__ == "__main__":
    main()
