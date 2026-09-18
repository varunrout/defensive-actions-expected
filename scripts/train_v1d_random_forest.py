"""Rung 3 of the model ladder (prompt 44): v1d_random_forest.

v1c_systematic_interactions (Rung 2) found its gain by explicitly
engineering 153 pairwise/squared terms from the 17 numeric features and
letting L1 prune them. This rung asks the question that approach can't
answer by itself: does a model that discovers interactions and
non-linearity automatically, straight from the RAW 32 locked features,
find meaningfully more signal than the hand-engineered approach did -- or
has Rung 2 already captured most of what's recoverable with this feature
set?

Random Forest is the diagnostic probe (not a promotion candidate in its
own right, per the brief): no manual feature engineering, handles
interactions/non-linearity natively via tree splits, far less
hyperparameter-sensitive than gradient boosting. Its result decides how
Rung 4 (GBM) should be scoped -- a real investment if RF shows a wide
gap over v1c, or a smaller confirmatory step if RF is roughly level with
or worse than v1c.

Design matrix: same 32 locked features, same categorical-one-hot /
boolean-passthrough blocks as v1, numeric features median-imputed but
NOT standardized (trees don't need it, and standardizing would only
obscure feature importances) -- and critically, NONE of v1c's 153
engineered interaction/polynomial columns. The whole point is to see
what the tree ensemble finds on its own from the same raw inputs v1 had.

Primary variant reported as v1d_random_forest: no class_weight
(mirrors the unweighted-wins-on-PR-AUC finding from v1 vs v2/v3).
A secondary class_weight="balanced_subsample" comparison is run at the
same tuned hyperparameters, reported alongside but not as the primary
variant, to check whether that finding still holds for trees.

n_estimators / max_depth / min_samples_leaf grid-searched on the same
5 canonical CV folds (train+val only); random_state=42 fixed throughout.
Held-out test is read exactly once, after the grid is fixed.

Usage:
    python scripts/train_v1d_random_forest.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "v1d_random_forest"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

N_ESTIMATORS_GRID = [300, 600]
MAX_DEPTH_GRID = [8, 14, None]
MIN_SAMPLES_LEAF_GRID = [5, 20]
RANDOM_STATE = 42


class RFDesignMatrixBuilder:
    """Same categorical (one-hot) / boolean (passthrough) blocks as v1's
    DesignMatrixBuilder. Numeric features are median-imputed but NOT
    standardized -- trees are scale-invariant, and standardizing would
    only obscure feature_importances_. No engineered interaction terms."""

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

        score, _, _ = fit_predict_rf(train_df, test_df, y_train, y_test, params, class_weight)
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
        "df": df,
    }


def tune_hyperparams(trainval_df: pd.DataFrame) -> dict:
    """Grid-search n_estimators/max_depth/min_samples_leaf on the 5
    canonical CV folds, train+val rows only. Held-out test never
    referenced."""
    grid_results = []
    cv_by_params = {}
    combos = list(itertools.product(N_ESTIMATORS_GRID, MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID))
    for n_est, max_depth, min_leaf in combos:
        params = {"n_estimators": n_est, "max_depth": max_depth, "min_samples_leaf": min_leaf}
        key = (n_est, max_depth, min_leaf)
        print(f"[tune] n_estimators={n_est} max_depth={max_depth} min_samples_leaf={min_leaf}")
        result = run_cv_rf(trainval_df, params)
        pr_auc = result["oof_metrics"]["average_precision"]
        grid_results.append({
            "n_estimators": n_est, "max_depth": max_depth, "min_samples_leaf": min_leaf,
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


def significance_vs_v1c(trainval_df: pd.DataFrame, best_params: dict, v1d_fold_ap: list[float]) -> dict:
    """Paired fold-level PR-AUC comparison: v1d vs v1c, on the same 5
    canonical CV folds."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    per_fold_ap_v1c: list[float] = []
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        import train_v1c_systematic_interactions as v1c_mod
        score, _, _ = v1c_mod.fit_predict_v1c(train_df, test_df, y_train, y_test, 0.1)
        m = classification_metrics(y_test, score)
        per_fold_ap_v1c.append(m["average_precision"])

    a = np.array(v1d_fold_ap)
    b = np.array(per_fold_ap_v1c)
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "per_fold_average_precision": {VARIANT: v1d_fold_ap, "v1c_systematic_interactions": per_fold_ap_v1c},
        "best_params": best_params,
        f"{VARIANT}_vs_v1c_systematic_interactions": {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        },
    }


def player_disjoint_check(trainval_df: pd.DataFrame, best_params: dict) -> dict:
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

        score, _, _ = fit_predict_rf(train_df, test_df, y_train, y_test, best_params)
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
        "best_params": best_params,
        "n_players_total": n_players,
        "fold_metrics": fold_df.to_dict(orient="records"),
        "mean": fold_df[metric_cols].mean().to_dict(),
        "std": fold_df[metric_cols].std(ddof=0).to_dict(),
        "max_player_overlap_any_fold": int(fold_df["player_overlap_train_test"].max()),
        "player_disjoint_confirmed_every_fold": bool((fold_df["player_overlap_train_test"] == 0).all()),
    }


def fold_held_out_permutation_importance(trainval_df: pd.DataFrame, best_params: dict) -> list[dict]:
    """Permutation importance computed on each fold's held-out portion
    (never the final held-out test set), averaged across folds -- avoids
    the in-sample optimism of computing it on training data."""
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

    print("[tune] grid-searching RF hyperparameters on 5 canonical CV folds (train+val only)...")
    tuning = tune_hyperparams(trainval_df)
    best_params = {k: tuning["best_params"][k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
    cv_result = tuning["cv_by_params"][tuning["best_key"]]
    v1d_fold_ap = cv_result["fold_metrics"]["average_precision"].tolist()

    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]
    builder_probe = RFDesignMatrixBuilder().fit(trainval_df)
    n_features_probe = builder_probe.transform(trainval_df.head(3))[0].shape[1]

    comparison_row = {"variant": VARIANT, "split": "cv", "n_features": int(n_features_probe)}
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

    print("[csv] appending rows (cv + test) to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_comparison.csv", [comparison_row, test_row])
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv", [readout_row])

    # -- Secondary comparison: class_weight="balanced_subsample" at the same tuned hyperparameters. --
    print("[secondary] class_weight='balanced_subsample' comparison at the same tuned hyperparameters...")
    cv_weighted = run_cv_rf(trainval_df, best_params, class_weight="balanced_subsample")
    secondary_output = {
        "note": "Secondary comparison only -- NOT the primary v1d_random_forest variant. "
                "Checks whether the unweighted-wins-on-PR-AUC finding from v1 vs v2/v3 still holds for trees.",
        "best_params": best_params,
        "unweighted_oof_average_precision": oof["average_precision"],
        "balanced_subsample_oof_average_precision": cv_weighted["oof_metrics"]["average_precision"],
        "unweighted_oof_expected_calibration_error": oof["expected_calibration_error"],
        "balanced_subsample_oof_expected_calibration_error": cv_weighted["oof_metrics"]["expected_calibration_error"],
    }
    (VALIDATION_DIR / "v1d_class_weight_secondary_comparison.json").write_text(
        json.dumps(secondary_output, indent=2), encoding="utf-8"
    )

    print("[significance] v1d vs v1c...")
    sig = significance_vs_v1c(trainval_df, best_params, v1d_fold_ap)
    sig["c_grid_search"] = tuning["grid_results"]
    (VALIDATION_DIR / "significance_v1d_vs_v1c.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[player-disjoint] v1d, 5-fold GroupKFold on player_id...")
    disjoint = player_disjoint_check(trainval_df, best_params)
    (VALIDATION_DIR / "player_disjoint_v1d.json").write_text(json.dumps(disjoint, indent=2, default=str), encoding="utf-8")

    print("\n=== HYPERPARAMETER GRID SEARCH (CV only, train+val) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if (r["n_estimators"], r["max_depth"], r["min_samples_leaf"]) == tuning["best_key"] else ""
        print(f"n_est={r['n_estimators']:<5} max_depth={str(r['max_depth']):<5} min_leaf={r['min_samples_leaf']:<3} "
              f"OOF PR-AUC={r['oof_average_precision']:.4f}{marker}")

    print("\n=== CV (OOF) at chosen params ===")
    print({k: v for k, v in comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT ===")
    print(readout_row)
    print("\n=== SECONDARY: class_weight comparison ===")
    print(secondary_output)
    print("\n=== SIGNIFICANCE (v1d vs v1c) ===")
    print(sig[f"{VARIANT}_vs_v1c_systematic_interactions"])
    print("\n=== PLAYER-DISJOINT ===")
    print("mean:", disjoint.get("mean"))
    print("player_disjoint_confirmed_every_fold:", disjoint.get("player_disjoint_confirmed_every_fold"))
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
