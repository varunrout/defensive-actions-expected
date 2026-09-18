"""Rung 2 of the model ladder (prompt 41): v1c_systematic_interactions.

v3_weighted_interactions (prompt 36) tried interactions too, but only 5
manually-chosen pairs, combined with class_weight="balanced" -- which
independently hurts both PR-AUC and calibration (see
ACTIVE_BINARY_BASELINE_SUMMARY.md sections 2/9), so v3 never isolated
whether interactions specifically help. Rung 1 (prompt 39) tested 4
hand-picked squared terms and found a small CV gain that did not survive
held-out test. This rung isolates the interaction question properly:

  - Same unweighted objective as v1 (no class_weight).
  - Same 32 locked features, same categorical one-hot / boolean-passthrough
    blocks, unchanged.
  - The 17 numeric features (11 continuous + 6 discrete, already
    median-imputed and standardized exactly as DesignMatrixBuilder does)
    get every pairwise product and every squared term generated
    systematically via sklearn's PolynomialFeatures(degree=2,
    interaction_only=False, include_bias=False) -- C(17,2)=136 pairwise +
    17 squared = 153 new columns -- fit on the standardized numeric block
    ONLY (categorical one-hot columns are not combinatorially exploded).
  - L1 (Lasso) penalty via LogisticRegression(penalty="l1",
    solver="liblinear"), not L2: L1 is what makes "systematic" mean
    something -- it can zero out the ~153 candidate terms that don't earn
    their place, rather than every one getting a nonzero coefficient by
    default.
  - C is grid-searched on the same 5 canonical CV folds (train+val rows
    only); the held-out test set is never touched until the winning C is
    fixed, and is then read exactly once.

Usage:
    python scripts/models/train_v1c_systematic_interactions.py
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/<subfolder>/ -> repo root (prompt 48 move: was parents[1] at scripts/ root)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))  # prompt 48: sibling ladder scripts now live in scripts/models/, not scripts/
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "v1c_systematic_interactions"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
NONZERO_EPS = 1e-6


class InteractionDesignMatrixBuilder:
    """Same categorical (one-hot) and boolean (passthrough) blocks as v1's
    DesignMatrixBuilder, unchanged. The 17 numeric columns (already
    median-imputed + standardized, same as v1) are additionally expanded
    with PolynomialFeatures(degree=2) fit on the standardized numeric block
    only -- 153 new squared/pairwise columns kept alongside, not instead
    of, the 17 original standardized numeric columns.
    """

    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        # Products/squares of standardized columns are not themselves unit-
        # variance (e.g. two independent standard-normal columns multiplied
        # have variance 1 but heavier tails; correlated pairs deviate more).
        # Re-standardizing the 153 new poly columns is a numerical-
        # conditioning step for the L1 solver, not a change to which
        # interactions/squares are generated -- liblinear converges far
        # faster on well-conditioned input.
        self.poly_scaler = StandardScaler()

    def fit(self, train_df: pd.DataFrame) -> "InteractionDesignMatrixBuilder":
        self.ohe.fit(train_df[tb.CATEGORICAL_COLS])
        self.cat_feature_names_ = list(self.ohe.get_feature_names_out(tb.CATEGORICAL_COLS))

        num_imputed = self.num_imputer.fit_transform(train_df[tb.NUMERIC_COLS])
        num_std = self.num_scaler.fit_transform(num_imputed)
        self.poly.fit(num_std)
        poly_full_train = self.poly.transform(num_std)

        poly_names_raw = list(self.poly.get_feature_names_out(tb.NUMERIC_COLS))
        # PolynomialFeatures(degree=2, include_bias=False) output = the 17
        # degree-1 originals (no space, no '^') followed by degree-2 terms
        # (either "x^2" or "x0 x1"). Keep only the degree-2 ones -- the
        # degree-1 originals are already carried separately, unchanged.
        self.poly_new_mask_ = np.array([("^" in n) or (" " in n) for n in poly_names_raw])
        self.poly_new_names_ = [
            f"poly__{n.replace(' ', '__x__').replace('^2', '__sq')}"
            for n, keep in zip(poly_names_raw, self.poly_new_mask_)
            if keep
        ]
        n_pairwise = sum(1 for n in poly_names_raw if " " in n)
        n_square = sum(1 for n in poly_names_raw if "^" in n)
        assert n_pairwise == 136 and n_square == 17, (
            f"Expected C(17,2)=136 pairwise + 17 squared, got {n_pairwise} pairwise + {n_square} squared"
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


def fit_predict_v1c(train_df, test_df, y_train, y_test, C: float):
    builder = InteractionDesignMatrixBuilder().fit(train_df)
    x_train, feature_names = builder.transform(train_df)
    x_test, _ = builder.transform(test_df)

    # tol relaxed from sklearn's default 1e-4 to 1e-3: liblinear on this
    # ~234-column, heavily collinear design matrix (products/squares of
    # correlated base features are themselves correlated) takes ~270s/fit
    # at 1e-4 vs ~90s/fit at 1e-3 (tested at C=1.0), which was needed to
    # keep the 5-C x 5-fold grid search plus both gates tractable. This is
    # NOT free: the two tolerances gave a similar nonzero-term count
    # (223 vs 226 at C=1.0) but a non-trivial intercept difference
    # (-2.50 vs -1.65) -- expected L1-on-collinear-features behaviour (the
    # solution path is less sharply determined when candidate columns are
    # correlated), not a bug, but a real precision/runtime tradeoff that is
    # reported as a caveat in the rung-2 writeup rather than glossed over.
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=42, C=C, tol=1e-3)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    return score, model, (builder, feature_names)


def run_cv_v1c(df: pd.DataFrame, C: float) -> dict:
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

        score, _, _ = fit_predict_v1c(train_df, test_df, y_train, y_test, C)
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
    """Grid-search C on the 5 canonical CV folds, train+val rows only. The
    held-out test set is never referenced in this function."""
    grid_results = []
    cv_by_c = {}
    for C in C_GRID:
        print(f"[tune] C={C}")
        result = run_cv_v1c(trainval_df, C)
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


def significance_vs_others(trainval_df: pd.DataFrame, best_c: float, v1c_fold_ap: list[float]) -> dict:
    """Paired fold-level PR-AUC comparison: v1c vs v1_unweighted, v1c vs
    v1b_quadratic, on the same 5 canonical CV folds. Reuses v1c's
    already-computed per-fold scores from the tuning run at best_c (same
    folds, same fitted models -- no need to refit) and refits v1/v1b fresh
    on those folds for the comparison."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    other_variants = ["v1_unweighted", "v1b_quadratic"]
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

    a = np.array(v1c_fold_ap)
    result: dict = {"per_fold_average_precision": {VARIANT: v1c_fold_ap, **per_fold_ap}, "chosen_C": best_c}
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


def player_disjoint_check(trainval_df: pd.DataFrame, best_c: float) -> dict:
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

        score, _, _ = fit_predict_v1c(train_df, test_df, y_train, y_test, best_c)
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
        "chosen_C": best_c,
        "n_players_total": n_players,
        "fold_metrics": fold_df.to_dict(orient="records"),
        "mean": fold_df[metric_cols].mean().to_dict(),
        "std": fold_df[metric_cols].std(ddof=0).to_dict(),
        "max_player_overlap_any_fold": int(fold_df["player_overlap_train_test"].max()),
        "player_disjoint_confirmed_every_fold": bool((fold_df["player_overlap_train_test"] == 0).all()),
    }


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

    # -- 1. Confirm the design-matrix shape once, on the full train+val set. --
    probe_builder = InteractionDesignMatrixBuilder().fit(trainval_df)
    probe_x, probe_names = probe_builder.transform(trainval_df.head(5))
    n_cat = len(probe_builder.cat_feature_names_)
    n_poly_new = len(probe_builder.poly_new_names_)
    print(f"[design] categorical one-hot={n_cat}, numeric originals={len(tb.NUMERIC_COLS)}, "
          f"boolean={len(tb.BOOLEAN_COLS)}, new poly terms={n_poly_new}, "
          f"TOTAL design matrix columns={probe_x.shape[1]}")
    assert n_poly_new == 153, f"Expected 153 new poly/interaction columns, got {n_poly_new}"

    # -- 2. Tune C on CV only; held-out test set not referenced. --
    print("[tune] grid-searching C on 5 canonical CV folds (train+val only)...")
    tuning = tune_c(trainval_df)
    best_c = tuning["best_c"]
    cv_result = tuning["cv_by_c"][best_c]
    v1c_fold_ap = cv_result["fold_metrics"]["average_precision"].tolist()

    # -- 3. CV metrics at the chosen C (reuse the tuning run's CV fit -- same folds, same C). --
    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]
    comparison_row = {"variant": VARIANT, "split": "cv", "n_features": int(probe_x.shape[1]), "chosen_C": best_c}
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        comparison_row[metric] = oof[metric]
        comparison_row[f"fold_{metric}_mean"] = fold_mean[metric]
        comparison_row[f"fold_{metric}_std"] = fold_std[metric]
    comparison_row["rows"] = len(trainval_df)
    comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    chart_df = pd.DataFrame({"y_true": cv_result["y_all"], "y_score": cv_result["oof_scores"]})
    chart_paths = save_classification_charts(chart_df, tb.CHARTS_DIR / VARIANT)

    # -- 4. Final held-out test readout at the chosen C -- single touch. --
    print(f"[final-test-readout] {VARIANT} at C={best_c}")
    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test = test_df[tb.TARGET_COL].to_numpy()
    score_test, model, extra = fit_predict_v1c(trainval_df, test_df, y_trainval, y_test, best_c)
    test_metrics = classification_metrics(y_test, score_test)

    test_row = dict(test_metrics)
    test_row["variant"] = VARIANT
    test_row["split"] = "test"
    test_row["n_features"] = int(probe_x.shape[1])
    test_row["rows"] = len(test_df)
    test_row["matches"] = test_df[tb.GROUP_COL].nunique()

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

    # -- 5. Append rows to the existing comparison / held-out-readout CSVs. --
    print("[csv] appending rows (cv + test) to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_comparison.csv", [comparison_row, test_row])
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv", [readout_row])

    # -- 6. Gate: paired significance vs v1 and v1b_quadratic. --
    print("[significance] v1c vs v1_unweighted, v1c vs v1b_quadratic...")
    sig = significance_vs_others(trainval_df, best_c, v1c_fold_ap)
    sig["c_grid_search"] = tuning["grid_results"]
    (VALIDATION_DIR / "significance_v1c_vs_v1_v1b.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    # -- 7. Gate: player-disjoint re-check at the chosen C. --
    print("[player-disjoint] v1c, 5-fold GroupKFold on player_id...")
    disjoint = player_disjoint_check(trainval_df, best_c)
    (VALIDATION_DIR / "player_disjoint_v1c.json").write_text(json.dumps(disjoint, indent=2, default=str), encoding="utf-8")

    print("\n=== C GRID SEARCH (CV only, train+val) ===")
    for r in tuning["grid_results"]:
        marker = "  <== chosen" if r["C"] == best_c else ""
        print(f"C={r['C']:<8} OOF PR-AUC={r['oof_average_precision']:.4f}{marker}")

    print("\n=== CV (OOF) at chosen C ===")
    print({k: v for k, v in comparison_row.items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT ===")
    print(readout_row)
    print("\n=== SIGNIFICANCE ===")
    print(sig[f"{VARIANT}_vs_v1_unweighted"])
    print(sig[f"{VARIANT}_vs_v1b_quadratic"])
    print("\n=== PLAYER-DISJOINT ===")
    print("mean:", disjoint.get("mean"))
    print("player_disjoint_confirmed_every_fold:", disjoint.get("player_disjoint_confirmed_every_fold"))
    print("\n=== TOP SURVIVING INTERACTION/QUADRATIC TERMS ===")
    for r in surviving[:15]:
        print(f"  {r['feature']}: {r['coef']:+.4f}")
    print("\nCharts written:", [str(p) for p in chart_paths])
    print("Done.")


if __name__ == "__main__":
    main()
