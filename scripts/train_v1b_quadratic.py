"""Rung 1 of the model ladder (prompt 39): v1b_quadratic.

v1_unweighted (prompts 36-38) is a plain logistic regression -- a single
slope per feature. EDA already confirmed 4 of the 32 locked features have a
U-shaped, non-monotonic relationship with target_future_shot_10s
(defender_spread, distance_to_attacking_box, visible_defender_count,
defenders_between_ball_and_attacking_goal): a single linear coefficient
structurally cannot represent that shape, no matter how much data or
regularization is applied.

v1b_quadratic is v1 plus one extra column per U-shaped feature -- the square
of that feature's already-standardized value (see
train_active_binary_baseline.DesignMatrixBuilder(add_quadratic=True)) -- with
everything else held identical: same 32 base features, same categorical/
numeric/boolean preprocessing, same canonical split, no class_weight
(unweighted, like v1 -- this rung isolates feature *shape*, not
reweighting), same LogisticRegression(solver="lbfgs", max_iter=2000,
random_state=42) call.

This script:
  1. Runs the same 5-fold CV + one-time held-out-test readout as the
     original baseline run, and appends v1b_quadratic's rows to the
     existing comparison / held-out-readout CSVs (does not touch v0-v3's
     rows).
  2. Gates the result against v1 with a paired significance test on the
     same 5 canonical CV folds (reusing the pattern from
     validate_active_binary_baselines.py, applied to v1 vs v1b_quadratic).
  3. Re-runs the player-disjoint GroupKFold(player_id) check for
     v1b_quadratic specifically -- a richer feature space is more
     leakage-prone in principle, so this is not assumed to still hold just
     because v1 cleared it.
  4. Saves the same 4 diagnostic charts under
     outputs/models/classification/charts/v1b_quadratic/.

Usage:
    python scripts/train_v1b_quadratic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
from dax.models.diagnostics import save_classification_charts  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VARIANT = "v1b_quadratic"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"


def run_cv_and_test(df_all: pd.DataFrame) -> dict:
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)
    print(f"[split] train+val rows={len(trainval_df)} matches={trainval_df[tb.GROUP_COL].nunique()}; "
          f"held-out test rows={len(test_df)} matches={test_df[tb.GROUP_COL].nunique()}")

    print(f"[cv] {VARIANT}")
    cv_result = tb.run_cv(trainval_df, VARIANT)
    fold_mean, fold_std, oof = cv_result["fold_mean"], cv_result["fold_std"], cv_result["oof_metrics"]

    comparison_row = {"variant": VARIANT, "split": "cv", "n_features": 36}
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

    test_row = dict(test_metrics)
    test_row["variant"] = VARIANT
    test_row["split"] = "test"
    test_row["n_features"] = 36
    test_row["rows"] = len(test_df)
    test_row["matches"] = test_df[tb.GROUP_COL].nunique()

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
        "test_row": test_row,
        "readout_row": readout_row,
        "cv_result": cv_result,
        "chart_paths": chart_paths,
        "trainval_df": trainval_df,
    }


def append_to_csv(path: Path, new_rows: list[dict]) -> None:
    existing = pd.read_csv(path)
    new_df = pd.DataFrame(new_rows)
    # Preserve existing column order/set; add any new columns at the end.
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = pd.NA
    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined.to_csv(path, index=False)


def significance_v1_vs_v1b(trainval_df: pd.DataFrame) -> dict:
    """Paired fold-level PR-AUC comparison: v1_unweighted vs v1b_quadratic,
    on the same 5 canonical CV folds (same pattern as
    validate_active_binary_baselines.py's item1, applied to this pair)."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    variants = ["v1_unweighted", VARIANT]
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
    b = np.array(per_fold_ap["v1_unweighted"])
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "per_fold_average_precision": per_fold_ap,
        f"{VARIANT}_vs_v1_unweighted": {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        },
    }


def player_disjoint_check(trainval_df: pd.DataFrame) -> dict:
    """Same GroupKFold(player_id) check as validate_active_binary_baselines.py
    item 3, re-run here for v1b_quadratic specifically -- a richer feature
    space is more leakage-prone in principle and is not assumed to still
    clear this just because v1 did."""
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

        score, _, _ = tb.fit_predict_fold(train_df, test_df, y_train, y_test, VARIANT)
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

    result = run_cv_and_test(df_all)

    print(f"[csv] appending {VARIANT} rows (cv + test) to existing comparison CSVs...")
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_comparison.csv",
                  [result["comparison_row"], result["test_row"]])
    append_to_csv(tb.COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv",
                  [result["readout_row"]])

    print("[significance] v1_unweighted vs v1b_quadratic, 5 canonical CV folds...")
    sig = significance_v1_vs_v1b(result["trainval_df"])
    (VALIDATION_DIR / "significance_v1_vs_v1b_quadratic.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[player-disjoint] v1b_quadratic, 5-fold GroupKFold on player_id...")
    disjoint = player_disjoint_check(result["trainval_df"])
    (VALIDATION_DIR / "player_disjoint_v1b_quadratic.json").write_text(json.dumps(disjoint, indent=2, default=str), encoding="utf-8")

    print("\n=== CV (OOF) ===")
    print({k: v for k, v in result["comparison_row"].items() if not k.startswith("fold_")})
    print("\n=== HELD-OUT TEST READOUT ===")
    print(result["readout_row"])
    print("\n=== SIGNIFICANCE (v1b_quadratic vs v1_unweighted) ===")
    print(sig[f"{VARIANT}_vs_v1_unweighted"])
    print("\n=== PLAYER-DISJOINT ===")
    print("n_players_total:", disjoint.get("n_players_total"))
    print("mean:", disjoint.get("mean"))
    print("player_disjoint_confirmed_every_fold:", disjoint.get("player_disjoint_confirmed_every_fold"))
    print("\nCharts written:", [str(p) for p in result["chart_paths"]])
    print("Done.")


if __name__ == "__main__":
    main()
