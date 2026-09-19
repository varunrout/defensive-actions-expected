"""Full promotion validation for c1d_random_forest (prompt 58).

Rung 2's own ladder gate (prompt 56 -- paired significance vs c1_lognormal_glm,
held-out-test confirmation) answered "is this rung's change real?". This
script answers the bigger question, the same depth of scrutiny the binary
legs' own promotion audits went through (prompts 43/46/52) before a rung
replaced the standing reference model: should c1d_random_forest now replace
c1_lognormal_glm as the leg's reference E[xg|shot] component?

Two things make this audit structurally different from the binary legs':

  1. Small sample. ~4,368 shot-positive rows total, ~875/fold -- slice
     checks here are thinner than the binary legs' own, and thin slices are
     flagged plainly rather than papered over.
  2. This component doesn't stand alone. c1_lognormal_glm/c1d_random_forest
     only matter as half of the hurdle pipeline
     (predicted_xg = P(shot) * E[xg|shot]). Item 4 re-runs prompt 54's own
     hurdle_pipeline_readout.json diagnostic with c1d in place of c1 --
     the decisive check for this leg specifically, since a regression-only
     win (already banked, prompt 56) does not automatically mean a
     proportional end-to-end win once P(shot) dominates the product for
     most rows.

Four checks, all on train+val 5-fold canonical CV out-of-fold (OOF)
predictions for items 1-3 (positive rows only, same population every prior
significance/error-slice check on this leg has used) -- the held-out test
set is read again ONLY for item 4's hurdle re-run, reusing exactly the same
held-out set and method prompt 54 already used for this diagnostic:

  1. Tournament-stratified check (WC2022 vs Euro2024) -- common_log_rmse and
     original-scale MAE/RMSE/R^2, c1d vs c1.
  2. Error-slice breakdown by phase_label x position -- MAE and signed bias
     on the xg scale, c1d vs c1, three-way honesty (improves / same /
     regresses) per slice with an n<30 thin-slice caveat.
  3. Feature-shape sanity check -- marginal mean-predicted-log(xg) shape in
     10 bins for c1d's top 8 Gini-importance features, cross-checked
     against prompt 55's 3 quadratic candidates and prompt 56's
     correlation-vs-importance cross-check.
  4. Hurdle pipeline re-run -- v1e_gradient_boosting_calibrated's P(shot)
     (unchanged, reused via joblib.load + .predict_proba only, never
     refit) combined with c1d_random_forest's E[xg|shot] (also reused via
     joblib.load + .predict only, never refit) on the full held-out test
     set, compared against prompt 54's original c1-based hurdle numbers.

Reuse, not re-run: c1d_random_forest's and c1_lognormal_glm's CV-OOF
predictions are regenerated via their own already-established run_cv_c1d /
run_cv_regression functions at their already-locked hyperparameters (no
re-tuning) -- neither has a saved per-row OOF array on disk, only aggregate
metrics (the same situation validate_v1e_promotion.py faced for v1d/v1e,
solved the same way there). The held-out-set joblib models
(c1d_random_forest.joblib, c1_lognormal_glm.joblib,
v1e_gradient_boosting_calibrated.joblib) are loaded and SCORED ONLY
(.predict / .predict_proba), never re-fit, for item 4.

Usage:
    python scripts/models/validate_c1d_promotion.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402
import train_active_continuous_baseline as acb  # noqa: E402
import train_c1d_random_forest as c1d_mod  # noqa: E402
from dax.models.evaluation import regression_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
MATCHES_GLOB = str(REPO_ROOT / "data" / "raw" / "matches" / "*.json")

TOP8_GINI_FEATURES = [
    "attacking_goal_centrality", "distance_to_attacking_box", "visible_attacker_count",
    "match_time_seconds", "defender_spread", "angle_to_attacking_goal",
    "defender_attacker_gap_x", "nearest_attacker_distance",
]
QUADRATIC_CANDIDATES = {"visible_attacker_count", "defenders_within_10m", "defender_spread"}
CORR_IMPORTANCE_CROSS_CHECK = {"attacking_goal_centrality", "distance_to_attacking_box"}


def build_competition_map() -> dict[int, str]:
    comp: dict[int, str] = {}
    for path in glob.glob(MATCHES_GLOB):
        matches = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in matches:
            comp[m["match_id"]] = f"{m['competition_name']} {m['season']}"
    return comp


def ordered_pos_trainval(pos_trainval_df: pd.DataFrame) -> pd.DataFrame:
    """Same row ordering canonical_grouped_folds imposes -- what run_cv_c1d /
    run_cv_regression already use internally."""
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    return pos_trainval_df.loc[folds["row_index"]]


def run_cv_c1d_with_oof(pos_trainval_df: pd.DataFrame, params: dict) -> dict:
    """Same as c1d_mod.run_cv_c1d but also returns per-row OOF xg-scale
    predictions (naive + corrected) aligned to canonical_grouped_folds row
    order, needed for the tournament/error-slice/feature-shape checks."""
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    oof_log_pred = np.full(len(df), np.nan)
    oof_sigma2 = np.full(len(df), np.nan)
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        log_pred, sigma2, _, _, _ = c1d_mod.fit_predict_c1d(train_df, test_df, params)
        oof_log_pred[test_mask] = log_pred
        oof_sigma2[test_mask] = sigma2

    oof_naive = acb.backtransform(oof_log_pred, oof_sigma2, "naive")
    oof_corrected = acb.backtransform(oof_log_pred, oof_sigma2, "lognormal_corrected")
    return {"df_ordered": df, "oof_log_pred": oof_log_pred, "oof_naive": oof_naive, "oof_corrected": oof_corrected}


def run_cv_c1_with_oof(pos_trainval_df: pd.DataFrame) -> dict:
    """Same for c1_lognormal_glm, reusing acb.fit_predict_regression per fold."""
    folds = canonical_grouped_folds(pos_trainval_df, group_col=ab.GROUP_COL)
    df = pos_trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    oof_log_pred = np.full(len(df), np.nan)
    oof_sigma2 = np.full(len(df), np.nan)
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        log_pred, sigma2, _, _ = acb.fit_predict_regression(train_df, test_df, "c1_lognormal_glm")
        oof_log_pred[test_mask] = log_pred
        oof_sigma2[test_mask] = sigma2

    oof_naive = acb.backtransform(oof_log_pred, oof_sigma2, "naive")
    oof_corrected = acb.backtransform(oof_log_pred, oof_sigma2, "lognormal_corrected")
    return {"df_ordered": df, "oof_log_pred": oof_log_pred, "oof_naive": oof_naive, "oof_corrected": oof_corrected}


def tournament_breakdown(df_ordered: pd.DataFrame, y_true_log: np.ndarray, oof_naive: np.ndarray,
                          oof_corrected: np.ndarray, comp_map: dict) -> dict:
    work = df_ordered.copy()
    work["_y_log"] = y_true_log
    work["_y_xg"] = np.exp(y_true_log)
    work["_naive"] = oof_naive
    work["_corrected"] = oof_corrected
    work["competition"] = work[ab.GROUP_COL].map(comp_map)
    out: dict = {}
    for comp, sub in work.groupby("competition", dropna=False):
        matches = sub[ab.GROUP_COL].nunique()
        entry: dict = {"rows": int(len(sub)), "matches": int(matches)}
        if len(sub) < 30:
            entry["note"] = "fewer than 30 positive rows -- comparison on this slice is noisy, read directionally only"
        y_log = sub["_y_log"].to_numpy()
        y_xg = sub["_y_xg"].to_numpy()
        common_log_resid = np.log(sub["_naive"].to_numpy()) - y_log
        entry["common_log_rmse"] = float(np.sqrt(np.mean(common_log_resid ** 2)))
        entry["common_log_mae"] = float(np.mean(np.abs(common_log_resid)))
        entry.update({f"naive_{k}": v for k, v in regression_metrics(y_xg, sub["_naive"].to_numpy()).items()})
        entry.update({f"corrected_{k}": v for k, v in regression_metrics(y_xg, sub["_corrected"].to_numpy()).items()})
        out[str(comp)] = entry
    return out


def error_slice_breakdown(df_ordered: pd.DataFrame, y_true_xg: np.ndarray, pred_corrected: np.ndarray) -> dict:
    work = df_ordered.copy()
    work["_y"] = y_true_xg
    work["_pred"] = pred_corrected
    out: dict = {}
    for group_col in ["phase_label", "position"]:
        rows = []
        for val, sub in work.groupby(group_col, dropna=False):
            y = sub["_y"].to_numpy()
            p = sub["_pred"].to_numpy()
            row: dict = {
                group_col: str(val), "rows": int(len(sub)),
                "mean_actual": float(y.mean()), "mean_predicted": float(p.mean()),
                "mae": float(np.mean(np.abs(p - y))), "mean_signed_error_bias": float(np.mean(p - y)),
            }
            if len(sub) < 30:
                row["skipped_reason"] = "fewer than 30 rows -- comparison should not be trusted"
            rows.append(row)
        out[group_col] = sorted(rows, key=lambda r: -r["rows"])
    return out


def feature_shape_sanity(df_ordered: pd.DataFrame, log_pred: np.ndarray, features: list[str]) -> dict:
    out: dict = {}
    for feat in features:
        values = df_ordered[feat].to_numpy(dtype=float)
        finite = np.isfinite(values)
        try:
            bins = pd.qcut(values[finite], q=10, duplicates="drop")
        except ValueError:
            bins = pd.cut(values[finite], bins=10)
        bin_df = pd.DataFrame({"bin": bins, "log_pred": log_pred[finite], "value": values[finite]})
        grouped = bin_df.groupby("bin", observed=True)
        table = []
        for interval, sub in grouped:
            table.append({
                "bin_left": float(interval.left), "bin_right": float(interval.right),
                "n_rows": int(len(sub)),
                "mean_feature_value": float(sub["value"].mean()),
                "mean_predicted_log_xg": float(sub["log_pred"].mean()),
            })
        table.sort(key=lambda r: r["bin_left"])
        thin_bins = [t for t in table if t["n_rows"] < 30]
        out[feat] = {
            "n_bins": len(table), "n_rows_with_finite_value": int(finite.sum()),
            "n_rows_missing_or_nonfinite": int((~finite).sum()),
            "bins": table,
            "thin_bins_under_30_rows": len(thin_bins),
            "is_rung1_quadratic_candidate": feat in QUADRATIC_CANDIDATES,
            "is_rung2_correlation_importance_cross_check_feature": feat in CORR_IMPORTANCE_CROSS_CHECK,
        }
    return out


def score_c1d_predict_only(pos_trainval_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Loads the already-fitted c1d_random_forest.joblib and scores
    target_df, NEVER calling .fit(). sigma2 (training-fold residual
    variance) is recomputed via .predict() only on pos_trainval_df, the
    same quantity the original held-out fit already computed internally
    but did not persist to disk."""
    model_path = acb.REG_DIR / "c1d_random_forest.joblib"
    model = joblib.load(model_path)
    builder = c1d_mod.RFDesignMatrixBuilder().fit(pos_trainval_df)
    x_train, _ = builder.transform(pos_trainval_df)
    y_train_log = np.log(pos_trainval_df[acb.TARGET_XG_COL].to_numpy())
    train_pred_log = model.predict(x_train)
    sigma2 = float(np.var(y_train_log - train_pred_log, ddof=1))

    x_target, _ = builder.transform(target_df)
    pred_log = model.predict(x_target)
    return pred_log, sigma2


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = ab.load_data()
    load_canonical_split()
    pos_all = df_all[df_all[ab.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask = canonical_test_mask(pos_all, group_col=ab.GROUP_COL)
    pos_trainval = pos_all.loc[~pos_test_mask].reset_index(drop=True)
    pos_test = pos_all.loc[pos_test_mask].reset_index(drop=True)
    print(f"[data] positive trainval rows: {len(pos_trainval)}; held-out test rows: {len(pos_test)}")

    comp_map = build_competition_map()
    trainval_comp = pos_trainval[ab.GROUP_COL].map(comp_map)
    tournaments_present = sorted(trainval_comp.dropna().unique().tolist())
    print(f"[tournaments] present in positive train+val rows: {tournaments_present}")
    assert set(tournaments_present) == {"FIFA World Cup 2022", "UEFA Euro 2024"}, (
        f"Expected exactly WC2022 + Euro2024, got {tournaments_present} -- re-checked, not assumed."
    )

    c1d_sig = json.loads((VALIDATION_DIR / "significance_c1_vs_c1d_random_forest.json").read_text(encoding="utf-8"))
    c1d_params = c1d_sig["chosen_params"]
    print(f"[params] c1d_random_forest locked params (from prompt 56, not re-tuned): {c1d_params}")

    print("\n[1/4] regenerating CV-OOF predictions for c1d_random_forest and c1_lognormal_glm "
          "(locked params, no re-tuning)...")
    c1d_cv = run_cv_c1d_with_oof(pos_trainval, c1d_params)
    c1_cv = run_cv_c1_with_oof(pos_trainval)
    df_ordered = c1d_cv["df_ordered"]
    assert np.array_equal(df_ordered[ab.GROUP_COL].to_numpy(), c1_cv["df_ordered"][ab.GROUP_COL].to_numpy()), (
        "c1d and c1 CV row ordering diverged -- folds must match exactly"
    )
    y_true_log = np.log(df_ordered[acb.TARGET_XG_COL].to_numpy())
    y_true_xg = df_ordered[acb.TARGET_XG_COL].to_numpy()

    print("\n[2/4] tournament-stratified check...")
    c1d_tourn = tournament_breakdown(df_ordered, y_true_log, c1d_cv["oof_naive"], c1d_cv["oof_corrected"], comp_map)
    c1_tourn = tournament_breakdown(df_ordered, y_true_log, c1_cv["oof_naive"], c1_cv["oof_corrected"], comp_map)
    train_composition = {str(k): int(v) for k, v in pos_trainval.assign(competition=trainval_comp)
                          .groupby("competition")[ab.GROUP_COL].nunique().items()}
    tournament_output = {
        "population": "positive-row train+val, 5-fold canonical CV out-of-fold predictions (NOT held-out test)",
        "c1d_params": c1d_params,
        "train_val_match_counts_by_tournament": train_composition,
        "c1_lognormal_glm_by_tournament": c1_tourn,
        "c1d_random_forest_by_tournament": c1d_tourn,
    }
    (VALIDATION_DIR / "tournament_stratified_c1d.json").write_text(
        json.dumps(tournament_output, indent=2, default=str), encoding="utf-8"
    )
    for comp in c1_tourn:
        print(f"  {comp}: rows={c1_tourn[comp]['rows']} "
              f"c1_common_log_rmse={c1_tourn[comp]['common_log_rmse']:.4f} "
              f"c1d_common_log_rmse={c1d_tourn[comp]['common_log_rmse']:.4f} "
              f"c1_corrected_r2={c1_tourn[comp]['corrected_r2']:.4f} "
              f"c1d_corrected_r2={c1d_tourn[comp]['corrected_r2']:.4f}")

    print("\n[3/4] error-slice breakdown (phase_label x position)...")
    c1d_err = error_slice_breakdown(df_ordered, y_true_xg, c1d_cv["oof_corrected"])
    c1_err = error_slice_breakdown(df_ordered, y_true_xg, c1_cv["oof_corrected"])
    error_output = {
        "population": "positive-row train+val, 5-fold canonical CV out-of-fold predictions, corrected back-transform",
        "c1d_params": c1d_params,
        "c1_lognormal_glm": c1_err,
        "c1d_random_forest": c1d_err,
    }
    (VALIDATION_DIR / "error_analysis_c1d_regression.json").write_text(
        json.dumps(error_output, indent=2, default=str), encoding="utf-8"
    )
    verdicts = []
    for group_col in ["phase_label", "position"]:
        c1_by_key = {r[group_col]: r for r in c1_err[group_col]}
        c1d_by_key = {r[group_col]: r for r in c1d_err[group_col]}
        for key in c1_by_key:
            a, b = c1_by_key[key], c1d_by_key.get(key, {})
            if a["rows"] < 30:
                continue
            mae_diff = b.get("mae", float("nan")) - a.get("mae", float("nan"))
            if abs(mae_diff) < 0.005:
                verdict = "about the same"
            elif mae_diff < 0:
                verdict = "c1d improves"
            else:
                verdict = "c1d regresses"
            verdicts.append((group_col, key, a["rows"], a["mae"], b.get("mae"), verdict))
            print(f"  [{group_col}={key}] rows={a['rows']} c1_mae={a['mae']:.4f} "
                  f"c1d_mae={b.get('mae', float('nan')):.4f} -> {verdict}")

    print("\n[4/4] feature-shape sanity (top 8 Gini-importance features, c1d)...")
    sanity = feature_shape_sanity(df_ordered, c1d_cv["oof_log_pred"], TOP8_GINI_FEATURES)
    sanity_output = {
        "population": "positive-row train+val, 5-fold canonical CV out-of-fold predictions",
        "c1d_params": c1d_params,
        "note": "Marginal view only -- one feature binned at a time, nothing else held fixed. "
                "Not a controlled/partial-dependence view. Small positive-row sample makes "
                "spurious bin-level noise a real risk -- row count per bin is reported alongside "
                "every mean so a jagged pattern can be checked against thin-bin size before being "
                "read as a real football relationship.",
        "features": sanity,
    }
    (VALIDATION_DIR / "feature_shape_sanity_c1d.json").write_text(
        json.dumps(sanity_output, indent=2, default=str), encoding="utf-8"
    )
    for feat, info in sanity.items():
        bins = info["bins"]
        first, last = bins[0], bins[-1]
        mid = bins[len(bins) // 2]
        print(f"  {feat}: first_bin_log_xg={first['mean_predicted_log_xg']:.4f} "
              f"mid_bin={mid['mean_predicted_log_xg']:.4f} last_bin={last['mean_predicted_log_xg']:.4f} "
              f"thin_bins(<30)={info['thin_bins_under_30_rows']}")

    print("\n[hurdle] re-running full hurdle pipeline readout with c1d in place of c1 "
          "(held-out test set, single read, reusing joblib artifacts predict-only)...")
    test_mask_full = canonical_test_mask(df_all, group_col=ab.GROUP_COL)
    trainval_full = df_all.loc[~test_mask_full].reset_index(drop=True)
    test_full = df_all.loc[test_mask_full].reset_index(drop=True)

    c1d_log_pred_full, c1d_sigma2 = score_c1d_predict_only(pos_trainval, test_full)
    e_xg_given_shot_c1d = acb.backtransform(c1d_log_pred_full, c1d_sigma2, "lognormal_corrected")
    p_shot_full_test = acb.score_classifier(trainval_full, test_full)

    combined_pred_c1d = p_shot_full_test * e_xg_given_shot_c1d
    hurdle_oof_c1d = pd.DataFrame({
        "observed_future_xg": test_full[acb.TARGET_XG_COL].to_numpy(),
        "combined_future_xg_prediction": combined_pred_c1d,
        "match_id": test_full[ab.GROUP_COL].to_numpy(),
    })
    from dax.models.two_part_xg import compute_hurdle_metrics
    hurdle_metrics_c1d = compute_hurdle_metrics(hurdle_oof_c1d, prediction_col="combined_future_xg_prediction")

    prior_hurdle = json.loads((VALIDATION_DIR / "hurdle_pipeline_readout.json").read_text(encoding="utf-8"))
    c1_hurdle_metrics = prior_hurdle["hurdle_pipeline_metrics"]
    trivial_metrics = prior_hurdle["trivial_baseline_metrics"]

    hurdle_readout_c1d = {
        "held_out_test_rows": len(test_full),
        "held_out_test_matches": int(test_full[ab.GROUP_COL].nunique()),
        "classifier_variant": acb.CLASSIFIER_VARIANT,
        "regression_variant": "c1d_random_forest (lognormal_corrected back-transform)",
        "hurdle_pipeline_metrics": hurdle_metrics_c1d,
        "c1_based_hurdle_metrics_for_comparison": c1_hurdle_metrics,
        "trivial_unconditional_mean_baseline_metrics_for_comparison": trivial_metrics,
        "rmse_change_vs_c1_hurdle": hurdle_metrics_c1d["rmse"] - c1_hurdle_metrics["rmse"],
        "mae_change_vs_c1_hurdle": hurdle_metrics_c1d["mae"] - c1_hurdle_metrics["mae"],
        "r2_change_vs_c1_hurdle": hurdle_metrics_c1d["r2"] - c1_hurdle_metrics["r2"],
        "note": (
            "Decisive check for the promotion decision (prompt 58) -- c1d_random_forest's "
            "regression-only CV win (prompt 56) does not by itself guarantee a proportional "
            "end-to-end hurdle improvement, since P(shot) dominates the product for most rows. "
            "This is the number the promotion decision actually rests on for this leg."
        ),
    }
    (VALIDATION_DIR / "hurdle_pipeline_readout_c1d.json").write_text(
        json.dumps(hurdle_readout_c1d, indent=2), encoding="utf-8"
    )
    print(f"  c1d-based hurdle: rmse={hurdle_metrics_c1d['rmse']:.5f} mae={hurdle_metrics_c1d['mae']:.5f} r2={hurdle_metrics_c1d['r2']:.5f}")
    print(f"  c1-based hurdle (prompt 54, on record): rmse={c1_hurdle_metrics['rmse']:.5f} mae={c1_hurdle_metrics['mae']:.5f} r2={c1_hurdle_metrics['r2']:.5f}")
    print(f"  trivial baseline (on record): rmse={trivial_metrics['rmse']:.5f} mae={trivial_metrics['mae']:.5f} r2={trivial_metrics['r2']:.5f}")
    print(f"  change (c1d - c1): rmse={hurdle_readout_c1d['rmse_change_vs_c1_hurdle']:+.5f} "
          f"mae={hurdle_readout_c1d['mae_change_vs_c1_hurdle']:+.5f} "
          f"r2={hurdle_readout_c1d['r2_change_vs_c1_hurdle']:+.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
