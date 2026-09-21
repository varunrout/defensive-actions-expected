"""Rung 1 of the passive-xT leg (prompt 77): y1b_quadratic.

TASK 0 -- what this rung is actually testing (checked, not assumed)
---------------------------------------------------------------------------
Active-xT's own Rung 1 (Prompt 71) was motivated by a SYMMETRIC tail-under-
prediction pattern in `x1`'s Rung-0 calibration table (both extreme bins'
|predicted| smaller than |actual|). Checked `y1_two_stage_huber`'s own Rung-0
calibration table (`PASSIVE_XT_BASELINE_SUMMARY.md` section 7) directly
before assuming the same motivation carries over -- it does NOT show that
pattern. Recomputed the gap direction per bin explicitly:

    bin 0: predicted=-0.00518, actual=-0.01173, gap(pred-actual)=+0.00655 (pred > actual)
    bin 1: predicted=-0.00096, actual=-0.00252, gap=+0.00156 (pred > actual)
    bin 2: predicted=+0.00012, actual=-0.00048, gap=+0.00060 (pred > actual)
    bin 3: predicted=+0.00120, actual=+0.00007, gap=+0.00113 (pred > actual)
    bin 4: predicted=+0.00712, actual=+0.00306, gap=+0.00406 (pred > actual)

**`predicted > actual` in all 5 bins, not just the tails.** This is a
SYSTEMATIC POSITIVE BIAS across the whole distribution, confirmed independently
by the held-out `prediction_bias` field itself (+0.00278, i.e. the model's
average prediction sits above the true average by more than the true mean's
own magnitude, |0.00278| vs |mean_observed|=0.00232). This is a genuinely
different, more specific finding than active-xT's own symmetric-tail
conservatism -- NOTE, correcting the record: Prompt 76's own
`PASSIVE_XT_BASELINE_SUMMARY.md` section 7 write-up mischaracterized bin 4 as
"under-predicts the magnitude ... the same way" as active's bin 4; the real
direction is the OPPOSITE (`y1` OVER-predicts bin 4's magnitude, it does not
under-predict it) -- both bin 4 AND bin 0 in fact point the same direction as
every other bin (`predicted > actual`), which is what actually makes this a
single coherent systematic-bias finding rather than two independent tail
effects. Corrected in this document and cross-referenced from an updated
note in the baseline summary.

**This rung therefore checks whether quadratic/interaction terms reduce the
systematic positive bias** -- not whether they close a "tail gap" the way
active-xT's Rung 1 did, since passive's own Rung 0 does not show that pattern.

STEP 0 -- which stage(s) get expanded terms, decided from passive's own data
---------------------------------------------------------------------------
Active-xT's own Rung 1 (Prompt 71) expanded BOTH stages because the SAME
candidate numeric features showed real curvature in BOTH the zero-rate
(Stage A's target) and the nonzero-delta magnitude (Stage B's target).
Checked the identical cross-check here, not assumed to transfer: the
regression-stage quadratic candidates below (`ball_x`,
`top_option_3_threat_score`, `angle_to_attacking_goal`) DO show real
nonzero-delta-magnitude curvature (large, clean, non-thin-bin patterns --
see FEATURE-SET CONSTRUCTION below), but their OWN zero-rate curvature
(`reports/analysis/xt_target/passive_numerical_target_atlas.json`'s
unconditional `bins` panel) is much weaker: `ball_x`'s zero-rate ranges only
22.77%-28.33% (~1.25x) across its 10 bins, `top_option_3_threat_score`
24.88%-27.72% (~1.11x), `angle_to_attacking_goal` 24.7%-27.09% (~1.10x) --
nowhere near the ~176x range `on_ball_event_type` (a CATEGORICAL feature,
Prompt 76's own Step-0 evidence) showed for the classifier stage. The
classifier's own real predictive signal on this leg comes from categorical
structure that quadratic/interaction terms (numeric-only by construction --
squaring a one-hot dummy is a no-op, the same reasoning Prompt 39/55
established) cannot add anything to.

**Decision: expand the REGRESSION STAGE ONLY.** This is a genuine departure
from active-xT's own Step-0 answer (which expanded both stages), reached
here because passive's own evidence pattern is different, not because this
rung defaulted to a different answer for its own sake -- the numeric
quadratic candidates' zero-rate curvature here is real but an order of
magnitude weaker than the categorical evidence already driving the
classifier, so there is no comparable data-driven case for touching Stage A.

FEATURE-SET CONSTRUCTION (item 2)
---------------------------------------------------------------------------
Quadratic (3 features): checked all locked PASSIVE numeric features' own
`pearson_r_nonzero_delta` ranking (`passive_numerical_target_atlas.json`).
Top 4 by |r| are `ball_x` (0.290), `defender_x` (0.222),
`top_option_3_threat_score` (0.218), `top_option_2_threat_score` (0.203),
all confirmed real (non-thin, ~116-117k rows/bin, no `small_n` flags) --
but `defender_x` correlates r=0.81 with `ball_x` and
`top_option_2_threat_score` correlates r=0.87 with `top_option_3_threat_score`
(checked directly against the real data, not assumed), so keeping both of
either pair would pad the candidate list with near-duplicate curvature
information rather than 3 genuinely different signals, the same discipline
`c1b`/`x1b` used to avoid padding. Kept the stronger of each redundant pair
plus one more-independent (moderate correlation, r=0.35-0.47 with the other
two) weaker-but-real candidate:
  QUADRATIC_FEATURES_Y1B = ["ball_x", "top_option_3_threat_score", "angle_to_attacking_goal"]

Interaction (3 pairs): reused directly from
`reports/analysis/xt_target/PASSIVE_FEATURE_INTERACTION_ANALYSIS.json`
(Prompt 69's own passive-specific interaction findings, target
`target_xt_delta_passive`, not re-discovered from scratch). All 3 of its
"interactive" (both unconditional AND given-nonzero-delta) pairs are used
directly -- all features involved are locked PASSIVE modelling features, no
exclusion issue (unlike active-xT's own `x1b`, which had to drop one
interactive pair for an excluded feature):
  INTERACTION_PAIRS_Y1B = [
      ("top_option_2_threat_score", "top_option_2_distance_from_ball"),
      ("lane_screening_score_option_2", "engagement_distance_to_carrier"),
      ("marking_tightness", "engagement_distance_to_carrier"),
  ]

Both are appended ONLY to the regression head's design matrix (Step 0),
never the classifier's.

Usage:
    .venv/Scripts/python.exe scripts/models/train_y1b_quadratic.py
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

import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_xt_baseline as yb  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "y1b_quadratic"

QUADRATIC_FEATURES_Y1B = ["ball_x", "top_option_3_threat_score", "angle_to_attacking_goal"]
for _qf in QUADRATIC_FEATURES_Y1B:
    assert _qf in pb.NUMERIC_COLS, f"{_qf} must be one of the 38 locked PASSIVE modelling features"

INTERACTION_PAIRS_Y1B = [
    ("top_option_2_threat_score", "top_option_2_distance_from_ball"),
    ("lane_screening_score_option_2", "engagement_distance_to_carrier"),
    ("marking_tightness", "engagement_distance_to_carrier"),
]
for _a, _b in INTERACTION_PAIRS_Y1B:
    assert _a in pb.NUMERIC_COLS and _b in pb.NUMERIC_COLS, f"{_a}/{_b} must both be locked PASSIVE features"


def build_expanded_reg_design_matrix(train_df: pd.DataFrame, df: pd.DataFrame):
    """Regression-head-only expansion: quadratic terms via pb.DesignMatrixBuilder's
    own override mechanism, plus 3 manually-appended interaction columns
    (same no-extra-leakage-risk reasoning the active-xT x1b script used --
    each interaction column is the product of two already-fitted,
    already-standardized numeric columns)."""
    builder = pb.DesignMatrixBuilder(
        add_interactions=False, add_quadratic=True, quadratic_features=QUADRATIC_FEATURES_Y1B
    ).fit(train_df)
    x, names = builder.transform(df)

    for feat_a, feat_b in INTERACTION_PAIRS_Y1B:
        idx_a, idx_b = names.index(feat_a), names.index(feat_b)
        interaction_col = (x[:, idx_a] * x[:, idx_b]).reshape(-1, 1)
        x = np.hstack([x, interaction_col])
        names = names + [f"interaction__{feat_a}__x__{feat_b}"]
    return builder, x, names


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Classifier: UNCHANGED from y1 (plain 38-feature builder, no expansion --
    Step 0's decision). Regressor: expanded with quadratic + interaction terms."""
    y_nonzero_train = (train_df[yb.TARGET_COL].to_numpy() != 0.0).astype(int)

    clf_builder = pb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(train_df)
    x_train_clf, clf_feature_names = clf_builder.transform(train_df)
    x_test_clf, _ = clf_builder.transform(test_df)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(x_train_clf, y_nonzero_train)
    p_nonzero_test = clf.predict_proba(x_test_clf)[:, 1]

    nonzero_train_df = train_df.loc[train_df[yb.TARGET_COL] != 0.0]
    reg_builder, x_train_reg, reg_feature_names = build_expanded_reg_design_matrix(nonzero_train_df, nonzero_train_df)
    _, x_test_reg, _ = build_expanded_reg_design_matrix(nonzero_train_df, test_df)
    reg = HuberRegressor(max_iter=500)
    reg.fit(x_train_reg, nonzero_train_df[yb.TARGET_COL].to_numpy())
    e_delta_given_nonzero_test = reg.predict(x_test_reg)

    combined_pred = p_nonzero_test * e_delta_given_nonzero_test
    extra = {
        "clf": clf, "clf_builder": clf_builder, "clf_feature_names": clf_feature_names,
        "reg": reg, "reg_builder": reg_builder, "reg_feature_names": reg_feature_names,
    }
    return combined_pred, extra


def run_cv(df: pd.DataFrame) -> dict:
    folds = canonical_grouped_folds(df, group_col=yb.GROUP_COL)
    df = df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    fold_rows = []
    oof_pred = np.full(len(df), np.nan)
    y_all = df[yb.TARGET_COL].to_numpy()

    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]

        pred, _ = fit_predict(train_df, test_df)
        oof_pred[test_mask] = pred

        m = yb.signed_regression_metrics(y_all[test_mask], pred)
        m["fold"] = int(fold_id)
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    oof_metrics = yb.signed_regression_metrics(y_all, oof_pred)
    return {"variant": VARIANT, "fold_metrics": fold_df, "oof_metrics": oof_metrics, "df": df}


def main() -> None:
    yb.REG_DIR.mkdir(parents=True, exist_ok=True)
    yb.REG_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    yb.COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    yb.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[quadratic features, regression head only] {QUADRATIC_FEATURES_Y1B}")
    print(f"[interaction pairs, regression head only] {INTERACTION_PAIRS_Y1B}")

    df_all = yb.load_data()
    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=yb.GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test = df_all.loc[test_mask_full].reset_index(drop=True)

    print("\n[1/5] cross-validation, y1b_quadratic...")
    cv_result = run_cv(trainval)
    comparison_row = {"variant": VARIANT, "n_features": 38 + len(QUADRATIC_FEATURES_Y1B) + len(INTERACTION_PAIRS_Y1B)}
    comparison_row.update(cv_result["oof_metrics"])
    comparison_row["rows"] = len(trainval)
    comparison_row["matches"] = trainval[yb.GROUP_COL].nunique()
    comparison_row["events"] = trainval[yb.EVENT_COL].nunique()

    print("\n[2/5] significance test, y1b vs y1 (per-fold RMSE)...")
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
        "fold_rmse_y1b": a.tolist(), "fold_rmse_y1": b.tolist(),
        "mean_diff_y1b_minus_y1": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
        "quadratic_features_used": QUADRATIC_FEATURES_Y1B,
        "interaction_pairs_used": [list(p) for p in INTERACTION_PAIRS_Y1B],
    }
    (yb.VALIDATION_DIR / "significance_y1_vs_y1b_quadratic.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print(f"  mean_diff (y1b-y1) rmse: {sig['mean_diff_y1b_minus_y1']:+.5f}, "
          f"paired_t_p={sig['paired_t_pvalue']:.4f}, wilcoxon_p={sig['wilcoxon_pvalue']}")

    print("\n[3/5] classifier-stage diagnostic (P(nonzero), OOF -- UNCHANGED from y1, not re-fit "
          "with new terms, Step 0's decision) -- reused directly from y1's own diagnostic...")
    y1_clf_diag = json.loads((yb.VALIDATION_DIR / "y1_classifier_stage_diagnostic.json").read_text(encoding="utf-8"))
    print(f"  P(nonzero) OOF (identical to y1, classifier untouched): roc_auc={y1_clf_diag['roc_auc']:.4f}")

    print("\n[4/5] held-out test readout (fit once on all trainval rows)...")
    pred, extra = fit_predict(trainval, test)
    y_true = test[yb.TARGET_COL].to_numpy()
    held_out_row = yb.signed_regression_metrics(y_true, pred)
    held_out_row["variant"] = VARIANT
    held_out_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    held_out_row["rows"] = len(test)
    held_out_row["matches"] = test[yb.GROUP_COL].nunique()
    held_out_row["events"] = test[yb.EVENT_COL].nunique()

    calib_rows = yb.calibration_bins(y_true, pred)

    reg, reg_feature_names = extra["reg"], extra["reg_feature_names"]
    coef_table = pd.DataFrame({
        "feature": reg_feature_names, "coef": reg.coef_, "abs_coef": np.abs(reg.coef_),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coeff_json = {
        "variant": VARIANT, "intercept": float(reg.intercept_), "n_features": len(reg_feature_names),
        "quadratic_features": QUADRATIC_FEATURES_Y1B, "interaction_pairs": [list(p) for p in INTERACTION_PAIRS_Y1B],
        "regression_head_coefficients": coef_table.to_dict(orient="records"),
        "fitted_on": "all trainval rows (canonical split); classifier UNCHANGED from y1 (not re-fit); "
                     "regression head (expanded) fit on nonzero-delta trainval rows only; held-out test "
                     "used once for readout only",
    }
    (yb.REG_DIR / f"{VARIANT}.json").write_text(json.dumps(coeff_json, indent=2), encoding="utf-8")
    joblib.dump(reg, yb.REG_DIR / f"{VARIANT}_regressor.joblib")

    yb.save_yxt_regression_charts(yb.REG_CHARTS_DIR / VARIANT, y_true, pred, calib_rows, coef_table)
    print(f"  [held-out] {VARIANT}: rmse={held_out_row['rmse']:.5f} mae={held_out_row['mae']:.5f} "
          f"r2={held_out_row['r2']:.5f} spearman={held_out_row['spearman']:.4f} "
          f"zero_mae={held_out_row['zero_target_mae']:.5f} nonzero_mae={held_out_row['nonzero_target_mae']:.5f} "
          f"prediction_bias={held_out_row['prediction_bias']:+.5f}")

    print("\n[5/5] writing comparison CSVs (additive only -- y0/y1 rows untouched)...")
    yb.append_to_csv(yb.COMPARISONS_DIR / "passive_xt_baseline_comparison.csv", [comparison_row])
    yb.append_to_csv(yb.COMPARISONS_DIR / "passive_xt_baseline_held_out_test_readout.csv", [held_out_row])

    print("\n=== Calibration bins, held-out test, y1b (compare directly against y1's own table) ===")
    for r in calib_rows:
        print(f"  bin {r['bin']}: n={r['n']} predicted={r['mean_predicted']:+.5f} actual={r['mean_actual']:+.5f} gap={r['gap']:+.5f}")

    print("\n=== TOP SURVIVING QUADRATIC / INTERACTION TERM COEFFICIENTS (regression head) ===")
    extra_rows = coef_table[coef_table["feature"].str.startswith("quad__") | coef_table["feature"].str.startswith("interaction__")]
    for _, r in extra_rows.iterrows():
        print(f"  {r['feature']}: {r['coef']:+.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
