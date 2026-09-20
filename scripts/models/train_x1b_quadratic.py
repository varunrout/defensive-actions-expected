"""Rung 1 of the active-xT leg (prompt 71): x1b_quadratic.

`x1_two_stage_huber` (Rung 0, prompt 70) cleared its dummy floor, but its own
calibration check (Prompt 70 section 7) found a real, unaddressed limitation:
the two most extreme predicted-value bins under-predict the actual magnitude
by 3x or more (bin 0: predicted -0.0078 vs. actual -0.0286; bin 4: predicted
+0.0262 vs. actual +0.0433). This rung asks whether quadratic/interaction
terms narrow that gap -- not whether they move an aggregate metric in
isolation.

STEP 0 -- where do expanded terms go, and why (decided before building)
---------------------------------------------------------------------------
`x1` has two surfaces (Stage A logistic classifier for P(nonzero), Stage B
Huber regression for E[delta|nonzero]), unlike `v1b`'s single classifier or
`c1b`'s single regression head -- `c1b` is the closer precedent structurally
(two-part active-continuous leg), but even `c1b` only ever had ONE surface to
expand (its classifier half, `v1e`, was reused unchanged, never re-fit with
new terms). This leg is the first one where both halves are actually being
fit inside the same rung, so the question of which surface(s) get the new
terms has no direct precedent to just copy -- it is answered here from this
leg's own data, not assumed.

Checked directly against `reports/analysis/xt_target/active_numerical_target_atlas.json`,
which the calibration problem (magnitude of nonzero deltas) motivates reading
its `bins_nonzero_delta` panel specifically: distance_to_attacking_box
(r=0.157 vs the nonzero delta, the single strongest numeric correlate on
this leg) shows a clean, real accelerating-monotonic curve across its 9 bins
(mean delta -0.0122 -> 0.0224, with the last three gaps larger than the
first six); defender_attacker_gap_x (r=-0.062) shows a genuine inverted-U
(rises for 4 bins, peaks, falls for 5 -- not a straight line); visible_defender_count
(r=0.047, U-shaped per the atlas's own `shape` field) shows a real dip-then-
accelerating-rise pattern in its non-thin bins. These are exactly the kind of
real, non-noise curvature `c1b`'s own discipline requires before adding a
quadratic term -- reconfirmed here on the actual bin counts, not assumed from
the correlation coefficient alone (the report's own `small_n` flags rule out
several other numeric features, e.g. defenders_within_10m/attackers_within_10m,
whose apparent tail spikes turn out to sit on n<=23, some n=1, bins).

The SAME atlas also reports each bin's zero-rate for these same three
features (`bins`, the unconditional panel, `pct_zero = 100 - pct_negative -
pct_positive`): distance_to_attacking_box's zero-rate is NOT flat across bins
(14.9% -> 26.5% -> 15.3%, an inverted-U of its own), defender_attacker_gap_x's
zero-rate ranges 15.9%-25.9% with the same rise-then-fall shape, and
visible_defender_count's zero-rate ranges 8.4%-26.6%. **This is the evidence
this decision rests on**: the same three features show real curvature in
BOTH the zero-rate (what Stage A predicts) and the nonzero-delta magnitude
(what Stage B predicts) -- not a coincidence to ignore, and not a default to
"add terms everywhere" either, since the candidate set itself was selected
for its regression-stage evidence first (Prompt 70's own motivating problem)
and only kept for the classifier stage because the SAME features independently
clear a curvature bar there too.

**Decision: expand both stages with the identical feature set.** Not
independently tuned per stage -- there is no evidence in hand that a
different curvature shape applies to one stage vs. the other for these three
features, and inventing a second, classifier-specific candidate list without
its own evidence would be exactly the "add terms everywhere" default this
prompt was told to avoid.

FEATURE-SET CONSTRUCTION (item 1)
---------------------------------------------------------------------------
Quadratic (3 features, not a fixed count, mirroring c1b's own "this leg's own
real data determined the count and the members" discipline):
  QUADRATIC_FEATURES_X1B = ["distance_to_attacking_box", "defender_attacker_gap_x",
                             "visible_defender_count"]

Interaction (1 pair): reused directly from
`reports/analysis/xt_target/FEATURE_INTERACTION_ANALYSIS.json` (Prompt
65/67's own xT-specific interaction findings, target_xt_delta_v2, NOT
re-discovered from scratch, per the prompt's own instruction). Of its 5
curated ACTIVE pairs, 2 are classified "interactive": `defenders_within_5m x
defenders_within_10m` and `visible_defender_count x attacker_spread`. The
first pair is not usable here -- `defenders_within_5m` is excluded from this
leg's 32 locked modelling features (structurally nested inside
`defenders_within_10m`, per Prompt 70's own feature-set decision, unchanged
here). The second pair is fully eligible (both features are locked modelling
features) and is used as this rung's one interaction term:
  INTERACTION_PAIR_X1B = ("visible_defender_count", "attacker_spread")

Usage:
    .venv/Scripts/python.exe scripts/models/train_x1b_quadratic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import HuberRegressor, LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as vb  # noqa: E402
import train_active_xt_baseline as xb  # noqa: E402
from dax.models.evaluation import classification_metrics, safe_spearman  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

VARIANT = "x1b_quadratic"

QUADRATIC_FEATURES_X1B = ["distance_to_attacking_box", "defender_attacker_gap_x", "visible_defender_count"]
for _qf in QUADRATIC_FEATURES_X1B:
    assert _qf in vb.NUMERIC_COLS, f"{_qf} must be one of the 32 locked modelling features"

INTERACTION_PAIR_X1B = ("visible_defender_count", "attacker_spread")
for _f in INTERACTION_PAIR_X1B:
    assert _f in vb.NUMERIC_COLS, f"{_f} must be one of the 32 locked modelling features"


def build_expanded_design_matrix(train_df: pd.DataFrame, df: pd.DataFrame):
    """Fit on train_df, transform df. Reuses vb.DesignMatrixBuilder's own
    quadratic-term machinery (add_quadratic=True, quadratic_features=...),
    then appends ONE extra interaction column
    (visible_defender_count_std * attacker_spread_std) computed from the
    already-fitted, already-standardized numeric block -- same no-extra-
    leakage-risk reasoning vb.DesignMatrixBuilder's own add_interactions
    path uses, just for a single caller-specified pair rather than the
    hardcoded v1-only list."""
    builder = vb.DesignMatrixBuilder(
        add_interactions=False, add_quadratic=True, quadratic_features=QUADRATIC_FEATURES_X1B
    ).fit(train_df)
    x, names = builder.transform(df)

    feat_a, feat_b = INTERACTION_PAIR_X1B
    idx_a, idx_b = names.index(feat_a), names.index(feat_b)
    interaction_col = (x[:, idx_a] * x[:, idx_b]).reshape(-1, 1)
    x = np.hstack([x, interaction_col])
    names = names + [f"interaction__{feat_a}__x__{feat_b}"]
    return builder, x, names


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: str):
    if variant == "x0_dummy":
        return xb.fit_predict(train_df, test_df, variant)

    y_nonzero_train = (train_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)

    clf_builder, x_train_clf, clf_feature_names = build_expanded_design_matrix(train_df, train_df)
    _, x_test_clf, _ = build_expanded_design_matrix(train_df, test_df)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(x_train_clf, y_nonzero_train)
    p_nonzero_test = clf.predict_proba(x_test_clf)[:, 1]

    nonzero_train_df = train_df.loc[train_df[xb.TARGET_COL] != 0.0]
    reg_builder, x_train_reg, reg_feature_names = build_expanded_design_matrix(nonzero_train_df, nonzero_train_df)
    _, x_test_reg, _ = build_expanded_design_matrix(nonzero_train_df, test_df)
    reg = HuberRegressor(max_iter=500)
    reg.fit(x_train_reg, nonzero_train_df[xb.TARGET_COL].to_numpy())
    e_delta_given_nonzero_test = reg.predict(x_test_reg)

    combined_pred = p_nonzero_test * e_delta_given_nonzero_test
    extra = {
        "clf": clf, "clf_builder": clf_builder, "clf_feature_names": clf_feature_names,
        "reg": reg, "reg_builder": reg_builder, "reg_feature_names": reg_feature_names,
    }
    return combined_pred, extra


def run_cv(df: pd.DataFrame, variant: str) -> dict:
    folds = canonical_grouped_folds(df, group_col=xb.GROUP_COL)
    df = df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_pred = np.full(len(df), np.nan)
    y_all = df[xb.TARGET_COL].to_numpy()

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        pred, _ = fit_predict(train_df, test_df, variant)
        oof_pred[test_mask] = pred

        m = xb.signed_regression_metrics(y_all[test_mask], pred)
        m["fold"] = int(fold_id)
        m["n_train"] = int(train_mask.sum())
        m["n_test"] = int(test_mask.sum())
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    oof_metrics = xb.signed_regression_metrics(y_all, oof_pred)
    return {"variant": variant, "fold_metrics": fold_df, "oof_metrics": oof_metrics, "oof_pred": oof_pred, "df": df}


def main() -> None:
    xb.REG_DIR.mkdir(parents=True, exist_ok=True)
    xb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    xb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    xb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[quadratic features] {QUADRATIC_FEATURES_X1B}")
    print(f"[interaction pair] {INTERACTION_PAIR_X1B}")

    df_all = xb.load_data()
    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=xb.GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test = df_all.loc[test_mask_full].reset_index(drop=True)

    print("\n[1/5] cross-validation, x1b_quadratic...")
    cv_result = run_cv(trainval, VARIANT)
    comparison_row = {"variant": VARIANT, "n_features": 32 + len(QUADRATIC_FEATURES_X1B) + 1}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(trainval)
    comparison_row["matches"] = trainval[xb.GROUP_COL].nunique()

    print("\n[2/5] significance test, x1b vs x1 (per-fold RMSE)...")
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
        "fold_rmse_x1b": a.tolist(), "fold_rmse_x1": b.tolist(),
        "mean_diff_x1b_minus_x1": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "quadratic_features_used": QUADRATIC_FEATURES_X1B,
        "interaction_pair_used": list(INTERACTION_PAIR_X1B),
    }
    (xb.VALIDATION_DIR / "significance_x1_vs_x1b_quadratic.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (x1b-x1) rmse: {sig['mean_diff_x1b_minus_x1']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[3/5] classifier-stage diagnostic (P(nonzero), OOF, x1b only -- not the promotion gate)...")
    oof_df = cv_result["df"]
    y_nonzero_oof = (oof_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)
    p_nonzero_oof = np.full(len(oof_df), np.nan)
    for fold_id in sorted(oof_df["fold"].unique()):
        train_mask = (oof_df["fold"] != fold_id).to_numpy()
        test_mask = (oof_df["fold"] == fold_id).to_numpy()
        train_df, test_df = oof_df.loc[train_mask], oof_df.loc[test_mask]
        y_nonzero_train = (train_df[xb.TARGET_COL].to_numpy() != 0.0).astype(int)
        _, x_train_clf, _ = build_expanded_design_matrix(train_df, train_df)
        _, x_test_clf, _ = build_expanded_design_matrix(train_df, test_df)
        clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42).fit(x_train_clf, y_nonzero_train)
        p_nonzero_oof[test_mask] = clf.predict_proba(x_test_clf)[:, 1]
    clf_diag = classification_metrics(y_nonzero_oof, p_nonzero_oof)
    clf_diag["note"] = "Diagnostic only (zero-vs-nonzero classifier stage of x1b, expanded features). Not the promotion gate."
    (xb.VALIDATION_DIR / "x1b_classifier_stage_diagnostic.json").write_text(json.dumps(clf_diag, indent=2), encoding="utf-8")
    print(f"  P(nonzero) OOF: roc_auc={clf_diag['roc_auc']:.4f} average_precision={clf_diag['average_precision']:.4f} "
          f"brier={clf_diag['brier_score']:.4f}")

    print("\n[4/5] held-out test readout (fit once on all trainval rows)...")
    pred, extra = fit_predict(trainval, test, VARIANT)
    y_true = test[xb.TARGET_COL].to_numpy()
    held_out_row = xb.signed_regression_metrics(y_true, pred)
    held_out_row["variant"] = VARIANT
    held_out_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    held_out_row["rows"] = len(test)
    held_out_row["matches"] = test[xb.GROUP_COL].nunique()

    calib_rows = xb.calibration_bins(y_true, pred)

    reg, reg_feature_names = extra["reg"], extra["reg_feature_names"]
    coef_table = pd.DataFrame({
        "feature": reg_feature_names, "coef": reg.coef_, "abs_coef": np.abs(reg.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coeff_json = {
        "variant": VARIANT, "intercept": float(reg.intercept_), "n_features": len(reg_feature_names),
        "quadratic_features": QUADRATIC_FEATURES_X1B, "interaction_pair": list(INTERACTION_PAIR_X1B),
        "regression_head_coefficients": coef_table.to_dict(orient="records"),
        "classifier_head_intercept": float(extra["clf"].intercept_[0]),
        "classifier_head_coefficients": {
            name: float(c) for name, c in zip(extra["clf_feature_names"], extra["clf"].coef_[0])
        },
        "fitted_on": "all trainval rows (canonical split); classifier on all trainval rows, "
                     "regression head on nonzero-delta trainval rows only; held-out test used once for readout only",
    }
    (xb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(coeff_json, indent=2), encoding="utf-8")
    joblib.dump(extra["clf"], xb.REG_DIR / f"{VARIANT}_classifier.joblib")
    joblib.dump(reg, xb.REG_DIR / f"{VARIANT}_regressor.joblib")

    xb.save_xt_regression_charts(xb.REG_CHARTS_DIR / VARIANT, y_true, pred, calib_rows, coef_table)
    print(f"  [held-out] {VARIANT}: rmse={held_out_row['rmse']:.5f} mae={held_out_row['mae']:.5f} "
          f"r2={held_out_row['r2']:.5f} spearman={held_out_row['spearman']:.4f} "
          f"zero_mae={held_out_row['zero_target_mae']:.5f} nonzero_mae={held_out_row['nonzero_target_mae']:.5f}")

    print("\n[5/5] writing comparison CSVs (additive only -- x0/x1 rows untouched)...")
    xb.append_to_csv(xb.COMPARISONS_DIR / "active_xt_baseline_comparison.csv", [comparison_row])
    xb.append_to_csv(xb.COMPARISONS_DIR / "active_xt_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== Calibration bins, held-out test, x1b (compare directly against Prompt 70's x1 table) ===")
    for r in calib_rows:
        print(f"  bin {r['bin']}: n={r['n']} predicted={r['mean_predicted']:+.5f} actual={r['mean_actual']:+.5f} gap={r['gap']:+.5f}")

    print("\n=== TOP SURVIVING QUADRATIC / INTERACTION TERM COEFFICIENTS (regression head) ===")
    extra_rows = coef_table[coef_table["feature"].str.startswith("quad__") | coef_table["feature"].str.startswith("interaction__")]
    for _, r in extra_rows.iterrows():
        print(f"  {r['feature']}: {r['coef']:+.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
