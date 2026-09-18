"""Rung 1 of the passive-binary model ladder (prompt 51): p1b_quadratic.

p1_unweighted (prompt 49, validated prompt 50) is a plain logistic
regression -- a single slope per feature. This leg's own EDA
(reports/analysis/shot_target/passive_numerical_target_atlas.json) shows 6
of the 38 locked features have a strictly U-shaped, non-monotonic
relationship with target_future_shot_10s: top_option_1/2/3_threat_score
(this leg's 3 strongest-correlated features overall), ball_x, defender_x,
and angle_to_attacking_goal. See QUADRATIC_FEATURES in
train_passive_binary_baseline.py for the exact selection and its
justification -- these are NOT the active leg's 4 U-shaped feature names
(defender_spread, distance_to_attacking_box, visible_defender_count,
defenders_between_ball_and_attacking_goal); this leg's own atlas was
checked directly rather than assuming the active leg's columns transfer.

p1b_quadratic is p1 plus one extra column per U-shaped feature -- the
square of that feature's already-standardized value -- with everything
else held identical: same 38 base features, same preprocessing, same
canonical split, no class_weight (unweighted, like p1 -- this rung isolates
feature *shape*, not reweighting), same LogisticRegression(solver="lbfgs",
max_iter=2000, random_state=42) call.

This script:
  1. Runs the same 5-fold CV + one-time held-out-test readout as the
     baseline run, and appends p1b_quadratic's rows to the existing
     comparison / held-out-readout CSVs (does not touch p0-p3's rows).
  2. Gates the result against p1 with a paired significance test on the
     same 5 canonical CV folds (reusing validate_passive_binary_baselines.py's
     item1-style pattern, applied to p1 vs p1b_quadratic).
  3. Runs the cluster-by-event_id bootstrap established in prompt 50
     (reusing validate_passive_binary_baselines.item3_cluster_bootstrap
     directly) on the held-out test set -- this leg's equivalent of the
     active leg's player-disjoint recheck, since there is no player_id here.
     Confirms whether p1b_quadratic's apparent gain (if any) survives the
     wider, cluster-aware CI, not just the naive row-level one.
  4. Saves the same 4 diagnostic charts under
     outputs/models/classification/charts/p1b_quadratic/.

Usage:
    python scripts/models/train_p1b_quadratic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_passive_binary_baseline as tb  # noqa: E402
import validate_passive_binary_baselines as vpb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "p1b_quadratic"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
N_FEATURES = 38 + len(tb.QUADRATIC_FEATURES)  # 44


def run_cv_and_test(df_all: pd.DataFrame) -> dict:
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)
    print(f"[split] train+val rows={len(trainval_df)} matches={trainval_df[tb.GROUP_COL].nunique()}; "
          f"held-out test rows={len(test_df)} matches={test_df[tb.GROUP_COL].nunique()}")

    print(f"[cv] {VARIANT}")
    cv_result = tb.run_cv(trainval_df, VARIANT)
    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]

    comparison_row = {"variant": VARIANT, "n_features": N_FEATURES}
    for metric in ["log_loss", "brier_score", "average_precision", "roc_auc",
                    "calibration_slope", "calibration_intercept", "expected_calibration_error", "positive_rate"]:
        comparison_row[metric] = oof[metric]
        comparison_row[f"fold_{metric}_mean"] = fold_mean[metric]
        comparison_row[f"fold_{metric}_std"] = fold_std[metric]
    comparison_row["rows"] = len(trainval_df)
    comparison_row["matches"] = trainval_df[tb.GROUP_COL].nunique()

    chart_df = pd.DataFrame({"y_true": cv_result["y_all"], "y_score": cv_result["oof_scores"]})
    chart_paths = save_classification_charts(chart_df, tb.CHARTS_DIR / VARIANT)

    print(f"[final-test-readout] {VARIANT}")
    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test = test_df[tb.TARGET_COL].to_numpy()
    score_test, model, extra = tb.fit_predict_fold(trainval_df, test_df, y_trainval, y_test, VARIANT)
    test_metrics = classification_metrics(y_test, score_test)

    readout_row = dict(test_metrics)
    readout_row["variant"] = VARIANT
    readout_row["readout"] = "FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION"
    readout_row["rows"] = len(test_df)
    readout_row["matches"] = test_df[tb.GROUP_COL].nunique()

    builder, feature_names = extra
    coeffs = tb.coefficient_json(model, feature_names, VARIANT)
    coeffs["fitted_on"] = "all train+val rows (canonical split); held-out test used once for readout only"
    (tb.MODELS_DIR / f"{VARIANT}.json").write_text(json.dumps(coeffs, indent=2), encoding="utf-8")
    joblib.dump(model, tb.MODELS_DIR / f"{VARIANT}.joblib")

    return {
        "comparison_row": comparison_row,
        "readout_row": readout_row,
        "cv_result": cv_result,
        "chart_paths": chart_paths,
        "trainval_df": trainval_df,
        "test_df": test_df,
        "score_test": score_test,
        "y_test": y_test,
    }


def append_to_csv(path: Path, new_rows: list[dict]) -> None:
    existing = pd.read_csv(path)
    new_df = pd.DataFrame(new_rows)
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined.to_csv(path, index=False)


def significance_p1_vs_p1b(trainval_df: pd.DataFrame) -> dict:
    """Paired fold-level PR-AUC comparison: p1_unweighted vs p1b_quadratic,
    on the same 5 canonical CV folds."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    variants = ["p1_unweighted", VARIANT]
    per_fold_ap: dict[str, list[float]] = {v: [] for v in variants}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask_f = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask_f]
        y_train, y_test = y_all[train_mask], y_all[test_mask_f]
        for variant in variants:
            score, _, _ = tb.fit_predict_fold(train_df, test_df, y_train, y_test, variant)
            m = classification_metrics(y_test, score)
            per_fold_ap[variant].append(m["average_precision"])

    a = np.array(per_fold_ap[VARIANT])
    b = np.array(per_fold_ap["p1_unweighted"])
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "assumes_row_level_independence_within_fold": True,
        "per_fold_average_precision": per_fold_ap,
        f"{VARIANT}_vs_p1_unweighted": {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        },
    }


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    tb.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tb.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    df_all = tb.load_data()
    load_canonical_split()

    print(f"[quadratic features] {tb.QUADRATIC_FEATURES}")
    result = run_cv_and_test(df_all)

    print(f"[csv] appending {VARIANT} rows (cv + held-out readout) to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_comparison.csv", [result["comparison_row"]])
    append_to_csv(tb.COMPARISONS_DIR / "passive_binary_baseline_held_out_test_readout.csv", [result["readout_row"]])

    print("[significance] p1_unweighted vs p1b_quadratic, 5 canonical CV folds...")
    sig = significance_p1_vs_p1b(result["trainval_df"])
    (VALIDATION_DIR / "significance_p1_vs_p1b_quadratic.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[fit] p1_unweighted, once on train+val, scored on held-out test (for the cluster bootstrap comparator)...")
    _, y_test_p1, score_p1 = vpb._fit_once_and_score_test(df_all, "p1_unweighted")
    assert np.array_equal(y_test_p1, result["y_test"]), "held-out y_test mismatch between p1 and p1b fits"

    print("[cluster bootstrap] p1b_quadratic vs p1_unweighted, held-out test, cluster-by-event_id...")
    boot = vpb.item3_cluster_bootstrap(
        result["y_test"], result["score_test"], score_p1,
        result["test_df"][vpb.EVENT_GROUP_COL].to_numpy(),
        label_a=VARIANT, label_b="p1_unweighted",
    )
    (VALIDATION_DIR / "cluster_bootstrap_p1_vs_p1b.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    print("\n=== CV (OOF) ===")
    print({k: v for k, v in result["comparison_row"].items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT ===")
    print(result["readout_row"])
    print("\n=== SIGNIFICANCE (p1b_quadratic vs p1_unweighted) ===")
    print(sig[f"{VARIANT}_vs_p1_unweighted"])
    print("\n=== CLUSTER BOOTSTRAP ===")
    print(f"{VARIANT} AP naive:", boot[f"{VARIANT}_average_precision"]["naive_row_level"])
    print(f"{VARIANT} AP cluster:", boot[f"{VARIANT}_average_precision"]["cluster_by_event"])
    diff_key = f"{VARIANT}_minus_p1_unweighted_average_precision"
    print("diff naive:", boot[diff_key]["naive_row_level"])
    print("diff cluster:", boot[diff_key]["cluster_by_event"])
    print("width ratio (cluster/naive):", boot[diff_key]["cluster_to_naive_width_ratio"])
    print("\nCharts written:", [str(p) for p in result["chart_paths"]])
    print("Done.")


if __name__ == "__main__":
    main()
