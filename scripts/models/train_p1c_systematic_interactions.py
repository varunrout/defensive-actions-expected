"""Rung 2 of the passive-binary model ladder (prompt 51): p1c_systematic_interactions.

p3_weighted_interactions (prompt 49) tried interactions too, but only 5
manually-chosen pairs, combined with class_weight="balanced" -- which
independently hurts both PR-AUC and calibration on this leg (see
PASSIVE_BINARY_BASELINE_SUMMARY.md section 8), so p3 never isolated whether
interactions specifically help. Rung 1 (p1b_quadratic) tested 6 hand-picked
squared terms in isolation. This rung isolates the interaction question
properly, mirroring the active leg's Rung 2 (prompt 41):

  - Same unweighted objective as p1 (no class_weight).
  - Same 38 locked features, same categorical one-hot / boolean-passthrough
    blocks, unchanged.
  - The 28 numeric features (26 continuous + 2 discrete -- this leg's own
    NUMERIC_COLS from train_passive_binary_baseline.py, NOT the active
    leg's 17) get every pairwise product and every squared term generated
    systematically via sklearn's PolynomialFeatures(degree=2,
    interaction_only=False, include_bias=False) -- C(28,2)=378 pairwise +
    28 squared = 406 new columns -- fit on the standardized numeric block
    ONLY.
  - L1 (Lasso) penalty via LogisticRegression(penalty="l1",
    solver="liblinear"): can zero out the ~406 candidate terms that don't
    earn their place, rather than every one getting a nonzero coefficient.
  - C is grid-searched on the same 5 canonical CV folds (train+val rows
    only); the held-out test set is never touched until the winning C is
    fixed, and is then read exactly once.

Deviation from the active leg's Rung 2 stated up front: the C grid here is
2 values, not 5, and liblinear's tol is relaxed from the sklearn default
1e-4 to 1e-2 (vs the active leg's own 1e-3 relaxation for the same reason).
This leg's train+val set is ~28x the active leg's row count (1.28M vs 45K)
at a similar column count (~459 vs ~234). A timed probe fit on a ~200K-row
subsample (roughly 1/6 of a real CV fold) at tol=1e-2 took ~118s; scaling
to the real ~1M-row folds made a full 5-C grid (25 fits for tuning alone)
impractical for this session, so the grid was trimmed to the 2 endpoints
of the active leg's range (C=0.01, strong regularization; C=1.0, weak) --
enough to see the direction of the regularization-strength effect without
the full sweep. This is a real precision/runtime tradeoff, reported here
rather than silently applied.

Usage:
    python scripts/models/train_p1c_systematic_interactions.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_passive_binary_baseline as tb  # noqa: E402
import validate_passive_binary_baselines as vpb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "p1c_systematic_interactions"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
C_GRID = [0.01, 1.0]
TOL = 1e-2
NONZERO_EPS = 1e-6
N_NUMERIC = len(tb.NUMERIC_COLS)
N_EXPECTED_PAIRWISE = N_NUMERIC * (N_NUMERIC - 1) // 2
N_EXPECTED_SQUARE = N_NUMERIC


class InteractionDesignMatrixBuilder:
    """Same categorical (one-hot) and boolean (passthrough) blocks as p1's
    DesignMatrixBuilder, unchanged. The 28 numeric columns (already
    median-imputed + standardized, same as p1) are additionally expanded
    with PolynomialFeatures(degree=2) fit on the standardized numeric block
    only."""

    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        self.poly_scaler = StandardScaler()

    def fit(self, train_df: pd.DataFrame) -> "InteractionDesignMatrixBuilder":
        self.ohe.fit(train_df[tb.CATEGORICAL_COLS])
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(tb.CATEGORICAL_COLS))

        num_imputed = self.num_imputer.fit_transform(train_df[tb.NUMERIC_COLS])
        num_std = self.num_scaler.fit_transform(num_imputed)
        self.poly.fit(num_std)
        poly_full_train = self.poly.transform(num_std)

        poly_names_raw = list(self.poly.get_feature_names_out(tb.NUMERIC_COLS))
        self.poly_new_mask_ = np.array([("^" in n) or (" " in n) for n in poly_names_raw])
        self.poly_new_names_ = [
            f"poly__{n.replace(' ', '__x__').replace('^2', '__sq')}"
            for n, keep in zip(poly_names_raw, self.poly_new_mask_)
            if keep
        ]
        n_pairwise = sum(1 for n in poly_names_raw if " " in n)
        n_square = sum(1 for n in poly_names_raw if "^" in n)
        assert n_pairwise == N_EXPECTED_PAIRWISE and n_square == N_EXPECTED_SQUARE, (
            f"Expected C({N_NUMERIC},2)={N_EXPECTED_PAIRWISE} pairwise + {N_EXPECTED_SQUARE} squared, "
            f"got {n_pairwise} pairwise + {n_square} squared"
        )
        self.poly_scaler.fit(poly_full_train[:, self.poly_new_mask_])
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cat_block = self.ohe.transform(df[tb.CATEGORICAL_COLS])
        cat_block = np.asarray(cat_block.todense()) if hasattr(cat_block, "todense") else np.asarray(cat_block)

        num_imputed = self.num_imputer.transform(df[tb.NUMERIC_COLS])
        num_std = self.num_scaler.transform(num_imputed)

        poly_full = self.poly.transform(num_std)
        poly_new = self.poly_scaler.transform(poly_full[:, self.poly_new_mask_])

        bool_block = df[tb.BOOLEAN_COLS].astype(float).fillna(0.0).to_numpy()

        blocks = [cat_block, num_std, bool_block, poly_new]
        names = [*self.cat_feature_names_, *tb.NUMERIC_COLS, *tb.BOOLEAN_COLS, *self.poly_new_names_]
        x = np.column_stack(blocks)
        return x, names


def fit_predict_p1c(train_df, test_df, y_train, y_test, C: float):
    builder = InteractionDesignMatrixBuilder().fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=42, C=C, tol=TOL)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    return score, model, (builder, feature_names)


def run_cv_p1c(df: pd.DataFrame, C: float) -> dict:
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
        score, _, _ = fit_predict_p1c(train_df, test_df, y_train, y_test, C)
        print(f"    fold {fold_id} (C={C}) fit in {time.time() - t0:.1f}s")
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


def tune_c(trainval_df: pd.DataFrame) -> dict:
    grid_results = []
    cv_by_c = {}
    for C in C_GRID:
        print(f"[tune] C={C}")
        result = run_cv_p1c(trainval_df, C)
        pr_auc = result["oof_metrics"]["average_precision"]
        grid_results.append({
            "C": C,
            "oof_average_precision": pr_auc,
            "fold_average_precision_mean": float(result["fold_mean"]["average_precision"]),
            "fold_average_precision_std": float(result["fold_std"]["average_precision"]),
        })
        cv_by_c[C] = result

    best = max(grid_results, key=lambda r: r["oof_average_precision"])
    best_c = best["C"]
    print(f"[tune] best C={best_c} (OOF PR-AUC={best['oof_average_precision']:.4f})")
    return {"grid_results": grid_results, "best_c": best_c, "cv_by_c": cv_by_c}


def append_to_csv(path: Path, new_rows: list[dict]) -> None:
    existing = pd.read_csv(path)
    new_df = pd.DataFrame(new_rows)
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined.to_csv(path, index=False)


def significance_vs_others(trainval_df: pd.DataFrame, p1c_fold_ap: list[float]) -> dict:
    """Paired fold-level PR-AUC comparison: p1c vs p1_unweighted, p1c vs
    p1b_quadratic, on the same 5 canonical CV folds."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    other_variants = ["p1_unweighted", "p1b_quadratic"]
    per_fold_ap: dict[str, list[float]] = {v: [] for v in other_variants}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        for variant in other_variants:
            score, _, _ = tb.fit_predict_fold(train_df, test_df, y_train, y_test, variant)
            m = classification_metrics(y_test, score)
            per_fold_ap[variant].append(m["average_precision"])

    a = np.array(p1c_fold_ap)
    result: dict = {"per_fold_average_precision": {VARIANT: p1c_fold_ap, **per_fold_ap}}
    for other in other_variants:
        b = np.array(per_fold_ap[other])
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        if len(set(diff.round(12))) > 1:
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float("nan"), float("nan")
        result[f"{VARIANT}_vs_{other}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        }
    return result


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

    print(f"[numeric block] {N_NUMERIC} features (26 continuous + 2 discrete, this leg's own list -- "
          f"NOT the active leg's 17) -> C({N_NUMERIC},2)={N_EXPECTED_PAIRWISE} pairwise + {N_EXPECTED_SQUARE} squared")

    probe_builder = InteractionDesignMatrixBuilder().fit(trainval_df)
    probe_x, probe_names = probe_builder.transform(trainval_df.head(5))
    n_cat = len(probe_builder.cat_feature_names_)
    n_poly_new = len(probe_builder.poly_new_names_)
    print(f"[design] categorical one-hot={n_cat}, numeric originals={N_NUMERIC}, "
          f"boolean={len(tb.BOOLEAN_COLS)}, new poly terms={n_poly_new}, "
          f"TOTAL design matrix columns={probe_x.shape[1]}")
    assert n_poly_new == N_EXPECTED_PAIRWISE + N_EXPECTED_SQUARE

    print(f"[tune] grid-searching C={C_GRID} on 5 canonical CV folds (train+val only), tol={TOL}...")
    tuning = tune_c(trainval_df)
    best_c = tuning["best_c"]
    cv_result = tuning["cv_by_c"][best_c]
    p1c_fold_ap = cv_result["fold_metrics"]["average_precision"].tolist()

    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]
    comparison_row = {"variant": VARIANT, "n_features": int(probe_x.shape[1]), "chosen_C": best_c}
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        comparison_row[metric] = oof[metric]
        comparison_row[f"fold_{metric}_mean"] = fold_mean[metric]
        comparison_row[f"fold_{metric}_std"] = fold_std[metric]
    comparison_row["rows"] = len(trainval_df)
    comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    chart_df = pd.DataFrame({"y_true": cv_result["y_all"], "y_score": cv_result["oof_scores"]})
    chart_paths = save_classification_charts(chart_df, tb.CHARTS_DIR / VARIANT)

    print(f"[final-test-readout] {VARIANT} at C={best_c}")
    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test = test_df[tb.TARGET_COL].to_numpy()
    score_test, model, extra = fit_predict_p1c(trainval_df, test_df, y_trainval, y_test, best_c)
    test_metrics = classification_metrics(y_test, score_test)

    readout_row = dict(test_metrics)
    readout_row["variant"] = VARIANT
    readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    readout_row["rows"] = len(test_df)
    readout_row["matches"] = test_df[tb.GROUP_COL].nunique()

    builder, feature_names = extra
    coeffs_all = tb.coefficient_json(model, feature_names, VARIANT)
    coeffs_all["chosen_C"] = best_c
    coeffs_all["fitted_on"] = "all train+val rows (canonical split); held-out test used once for readout only"

    poly_coef_rows = [r for r in coeffs_all["coefficients"] if r["feature"].startswith("poly__")]
    n_nonzero = sum(1 for r in poly_coef_rows if abs(r["coef"]) > NONZERO_EPS)
    surviving = sorted(
        (r for r in poly_coef_rows if abs(r["coef"]) > NONZERO_EPS),
        key=lambda r: r["abs_coef"], reverse=True,
    )
    coeffs_all["n_poly_interaction_terms_total"] = len(poly_coef_rows)
    coeffs_all["n_poly_interaction_terms_nonzero"] = n_nonzero
    coeffs_all["surviving_poly_terms_sorted_by_abs_coef"] = surviving

    (tb.MODELS_DIR / f"{VARIANT}.json").write_text(json.dumps(coeffs_all, indent=2), encoding="utf-8")
    joblib.dump(model, tb.MODELS_DIR / f"{VARIANT}.joblib")

    print(f"[coefficients] {n_nonzero}/{len(poly_coef_rows)} interaction/quadratic terms survived L1 (C={best_c})")

    print("[csv] appending rows to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_comparison.csv", [comparison_row])
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_held_out_test_readout.csv", [readout_row])

    print("[significance] p1c vs p1_unweighted, p1c vs p1b_quadratic...")
    sig = significance_vs_others(trainval_df, p1c_fold_ap)
    sig["c_grid_search"] = tuning["grid_results"]
    sig["tol_used"] = TOL
    (VALIDATION_DIR / "significance_p1c_vs_p1_p1b.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[fit] p1_unweighted, once on train+val, scored on held-out test (cluster-bootstrap comparator)...")
    _, y_test_p1, score_p1 = vpb._fit_once_and_score_test(df_all, "p1_unweighted")
    assert np.array_equal(y_test_p1, y_test)

    print("[cluster bootstrap] p1c vs p1_unweighted, held-out test, cluster-by-event_id...")
    boot = vpb.item3_cluster_bootstrap(
        y_test, score_test, score_p1, test_df[vpb.EVENT_GROUP_COL].to_numpy(),
        label_a=VARIANT, label_b="p1_unweighted",
    )
    (VALIDATION_DIR / "cluster_bootstrap_p1_vs_p1c.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    print("\n=== C GRID SEARCH (CV only, train+val) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if r["C"] == best_c else ""
        print(f"C={r['C']:<8} OOF PR-AUC={r['oof_average_precision']:.4f}{marker}")
    print("\n=== CV (OOF) at chosen C ===")
    print({k: v for k, v in comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT ===")
    print(readout_row)
    print("\n=== SIGNIFICANCE ===")
    print(sig[f"{VARIANT}_vs_p1_unweighted"])
    print(sig[f"{VARIANT}_vs_p1b_quadratic"])
    print("\n=== CLUSTER BOOTSTRAP ===")
    diff_key = f"{VARIANT}_minus_p1_unweighted_average_precision"
    print("diff naive:", boot[diff_key]["naive_row_level"])
    print("diff cluster:", boot[diff_key]["cluster_by_event"])
    print("width ratio:", boot[diff_key]["cluster_to_naive_width_ratio"])
    print("\n=== TOP SURVIVING INTERACTION/QUADRATIC TERMS ===")
    for r in surviving[:15]:
        print(f"  {r['feature']}: {r['coef']:+.4f}")
    print("\nCharts written:", [str(p) for p in chart_paths])
    print("Done.")


if __name__ == "__main__":
    main()
