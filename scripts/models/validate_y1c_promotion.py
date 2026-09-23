"""Full promotion validation for y1c_random_forest (prompt 80, part A).

Rung 2's own ladder gate (prompt 78 -- paired significance vs
y1_two_stage_huber, held-out-test confirmation) answered "is this rung's
change real?". This script answers the deeper question every prior
promotion decision on this project has asked (prompts 43/46/52/58/75):
should y1c_random_forest now replace y1_two_stage_huber as the leg's
documented reference model? Mirrors validate_x1c_promotion.py (prompt 75)
exactly in structure, adapted for passive-xT's own architecture and
already-established categorical/slice precedent.

Two things carry over unchanged from x1c's own audit, confirmed rather than
assumed to transfer:

  1. Two-stage architecture, entirely internal to this leg. y1c's classifier
     AND regressor are both this leg's own models -- confirmed directly
     against outputs/models/regression/y1c_random_forest.json (clf_params
     and reg_params both populated, Prompt 78's own Step-1 decision) before
     assuming anything. So item 4's "pipeline re-run" is y1c's own
     classifier x regressor product on the held-out test set, not a mixed
     composite -- same as x1c's own audit, no external hurdle-pipeline
     component on this leg either.
  2. No tournament-stability precedent on ACTUAL MODEL predictions for this
     target yet -- x1c's own audit was the first of this kind on the
     xt-delta family; this is the first on the passive leg specifically.
     Tournament composition is reconfirmed directly below, not assumed to
     match the active leg's own composition.

One thing does NOT carry over unchanged: the error-slice columns. x1c's own
audit used phase_label x position (a 2-column breakdown, ACTIVE's own
categorical set). Passive's own locked categorical columns are different
(on_ball_event_type, phase_label, defender_functional_role, period -- see
train_passive_binary_baseline.PASSIVE_CATEGORICAL_COLS), and this leg's own
EDA/modelling reports (PASSIVE_XT_BASELINE_SUMMARY.md, PASSIVE_XT_MODEL_LADDER.md
section 3.6) have already established which of those columns carry real,
already-documented signal worth slicing on: on_ball_event_type (~176x
zero-rate range, the classifier's own strongest signal, Prompt 76's own
Step-0 evidence), phase_label, and defender_functional_role (the leg's own
role-slicing precedent, PASSIVE_SLICE_STRATIFICATION_V2.json). This prompt's
own instruction names exactly these three ("defender role, phase, on-ball
event type") -- used here as three separate one-way breakdowns (not a joint
cross-tab, matching x1c's own one-way-per-column convention), rather than
x1c's two.

Four checks, all on train+val 5-fold canonical CV out-of-fold (OOF)
predictions for items 1-3 (all 1,275,289 trainval rows -- NOT filtered to
nonzero, since this leg's combined P(nonzero)*E[delta|nonzero] prediction is
defined and evaluated over every row) -- the held-out test set is read again
ONLY for item 4's pipeline re-run, reusing the canonical split's own frozen
test rows, never refit:

  1. Tournament-stratified check (composition reconfirmed, not assumed) --
     signed_regression_metrics + 5-bin calibration, y1c vs y1.
  2. Error-slice breakdown by on_ball_event_type / phase_label /
     defender_functional_role -- MAE, mean signed bias, and this leg's own
     zero_target_mae/nonzero_target_mae split, y1c vs y1, with an n<30
     thin-slice caveat.
  3. Feature-shape sanity check -- marginal mean-predicted-target_xt_delta_passive
     shape for y1c's top 8 features, aggregated back from one-hot Gini
     importance to the leg's own 38 locked PASSIVE features (same
     aggregation reasoning x1c's own audit used -- several literal
     top-ranked one-hot columns are single dummy levels of the same
     categorical feature, e.g. on_ball_event_type_Shot and
     on_ball_event_type_Pass are both on_ball_event_type). Numeric features
     binned via qcut (10 bins); categorical/boolean features grouped by
     their own category/value; cross-checked against
     reports/analysis/xt_target/PASSIVE_LEAKAGE_AUDIT.json's own Part E'
     construction-coupling finding for ball_x/ball_y specifically (restated,
     not re-derived).
  4. Full two-stage pipeline re-run -- y1c_random_forest_classifier.joblib
     and y1c_random_forest_regressor.joblib (both reused via joblib.load +
     .predict_proba/.predict ONLY, never refit) combined on the held-out
     test set, compared against y1_two_stage_huber's own already-recorded
     held-out numbers (Prompt 76's CSV row -- reused directly, NOT rescored
     again, per this prompt's own explicit constraint).

Usage:
    .venv/Scripts/python.exe scripts/models/validate_y1c_promotion.py
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

import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_xt_baseline as yb  # noqa: E402
import train_y1c_random_forest as y1c_mod  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
MATCHES_GLOB = str(REPO_ROOT / "data" / "raw" / "matches" / "*.json")

SLICE_COLUMNS = ["on_ball_event_type", "phase_label", "defender_functional_role"]


def build_competition_map() -> dict[int, str]:
    comp: dict[int, str] = {}
    for path in glob.glob(MATCHES_GLOB):
        matches = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in matches:
            comp[m["match_id"]] = f"{m['competition_name']} {m['season']}"
    return comp


def top8_original_features(gini_by_onehot_col: dict[str, float]) -> list[tuple[str, float]]:
    """Aggregates y1c's regression-head Gini importance (reported per
    one-hot output column in y1c_random_forest.json) back to the 38 locked
    PASSIVE features, longest-categorical-prefix-first -- same
    disambiguation x1c's own audit needed (a categorical feature's own
    dummy levels all start with that feature's name)."""
    cat_cols_sorted = sorted(pb.CATEGORICAL_COLS, key=len, reverse=True)
    agg: dict[str, float] = {}
    for name, val in gini_by_onehot_col.items():
        matched = None
        for c in cat_cols_sorted:
            if name.startswith(c + "_"):
                matched = c
                break
        if matched is None:
            matched = name
        agg[matched] = agg.get(matched, 0.0) + val
    total = sum(agg.values())
    assert abs(total - 1.0) < 1e-6, f"aggregated Gini importance must still sum to 1.0, got {total}"
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    return ranked[:8]


# ---------------------------------------------------------------------------
# CV-OOF regeneration (locked params, never re-tuned)
# ---------------------------------------------------------------------------

def run_cv_y1c_with_oof(trainval: pd.DataFrame, clf_params: dict, reg_params: dict) -> dict:
    folds = canonical_grouped_folds(trainval, group_col=yb.GROUP_COL)
    df = trainval.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()

    oof_pred = np.full(len(df), np.nan)
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask]
        pred, _ = y1c_mod.fit_predict(train_df, test_df, clf_params, reg_params)
        oof_pred[test_mask] = pred
    return {"df_ordered": df, "oof_pred": oof_pred}


def run_cv_y1_with_oof(trainval: pd.DataFrame) -> dict:
    result = yb.run_cv(trainval, "y1_two_stage_huber")
    return {"df_ordered": result["df"], "oof_pred": result["oof_pred"]}


# ---------------------------------------------------------------------------
# Item 1 -- tournament-stratified check
# ---------------------------------------------------------------------------

def tournament_breakdown(df_ordered: pd.DataFrame, y_true: np.ndarray, oof_pred: np.ndarray, comp_map: dict) -> dict:
    work = df_ordered.copy()
    work["_y"] = y_true
    work["_pred"] = oof_pred
    work["competition"] = work[yb.GROUP_COL].map(comp_map)
    out: dict = {}
    for comp, sub in work.groupby("competition", dropna=False):
        entry: dict = {"rows": int(len(sub)), "matches": int(sub[yb.GROUP_COL].nunique())}
        if len(sub) < 30:
            entry["note"] = "fewer than 30 rows -- comparison on this slice is noisy, read directionally only"
        y, p = sub["_y"].to_numpy(), sub["_pred"].to_numpy()
        entry.update(yb.signed_regression_metrics(y, p))
        entry["calibration_bins"] = yb.calibration_bins(y, p)
        out[str(comp)] = entry
    return out


# ---------------------------------------------------------------------------
# Item 2 -- error-slice breakdown
# ---------------------------------------------------------------------------

def error_slice_breakdown(df_ordered: pd.DataFrame, y_true: np.ndarray, pred: np.ndarray) -> dict:
    work = df_ordered.copy()
    work["_y"] = y_true
    work["_pred"] = pred
    out: dict = {}
    for group_col in SLICE_COLUMNS:
        rows = []
        for val, sub in work.groupby(group_col, dropna=False):
            y, p = sub["_y"].to_numpy(), sub["_pred"].to_numpy()
            row: dict = {
                group_col: str(val), "rows": int(len(sub)),
                "zero_rate_pct": float(100 * (y == 0.0).mean()),
                "mean_actual": float(y.mean()), "mean_predicted": float(p.mean()),
                "mae": float(np.mean(np.abs(p - y))), "mean_signed_error_bias": float(np.mean(p - y)),
            }
            row.update({f"overall_{k}": v for k, v in yb.signed_regression_metrics(y, p).items()
                        if k in ("zero_target_mae", "nonzero_target_mae")})
            if len(sub) < 30:
                row["skipped_reason"] = "fewer than 30 rows -- comparison should not be trusted"
            rows.append(row)
        out[group_col] = sorted(rows, key=lambda r: -r["rows"])
    return out


# ---------------------------------------------------------------------------
# Item 3 -- feature-shape sanity
# ---------------------------------------------------------------------------

def feature_shape_sanity(df_ordered: pd.DataFrame, pred: np.ndarray, features: list[str]) -> dict:
    out: dict = {}
    for feat in features:
        is_categorical = feat in pb.CATEGORICAL_COLS
        is_boolean = feat in pb.BOOLEAN_COLS
        if is_categorical or is_boolean:
            work = pd.DataFrame({"cat": df_ordered[feat].astype(str), "pred": pred})
            table = []
            for val, sub in work.groupby("cat", dropna=False):
                table.append({"category": str(val), "n_rows": int(len(sub)), "mean_predicted": float(sub["pred"].mean())})
            table.sort(key=lambda r: -r["n_rows"])
            thin = [t for t in table if t["n_rows"] < 30]
            out[feat] = {
                "type": "categorical" if is_categorical else "boolean",
                "n_categories": len(table), "categories": table,
                "thin_categories_under_30_rows": len(thin),
            }
        else:
            values = df_ordered[feat].to_numpy(dtype=float)
            finite = np.isfinite(values)
            try:
                bins = pd.qcut(values[finite], q=10, duplicates="drop")
            except ValueError:
                bins = pd.cut(values[finite], bins=10)
            bin_df = pd.DataFrame({"bin": bins, "pred": pred[finite], "value": values[finite]})
            table = []
            for interval, sub in bin_df.groupby("bin", observed=True):
                table.append({
                    "bin_left": float(interval.left), "bin_right": float(interval.right),
                    "n_rows": int(len(sub)),
                    "mean_feature_value": float(sub["value"].mean()),
                    "mean_predicted": float(sub["pred"].mean()),
                })
            table.sort(key=lambda r: r["bin_left"])
            thin = [t for t in table if t["n_rows"] < 30]
            out[feat] = {
                "type": "numeric", "n_bins": len(table),
                "n_rows_with_finite_value": int(finite.sum()), "n_rows_missing_or_nonfinite": int((~finite).sum()),
                "bins": table, "thin_bins_under_30_rows": len(thin),
            }
    return out


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = yb.load_data()
    load_canonical_split()
    test_mask_full = canonical_test_mask(df_all, group_col=yb.GROUP_COL)
    trainval = df_all.loc[~test_mask_full].reset_index(drop=True)
    test_full = df_all.loc[test_mask_full].reset_index(drop=True)
    print(f"[data] trainval rows={len(trainval)}; held-out test rows={len(test_full)}")

    y1c_json = json.loads((REPO_ROOT / "outputs" / "models" / "regression" / "y1c_random_forest.json").read_text(encoding="utf-8"))
    clf_params, reg_params = y1c_json["clf_params"], y1c_json["reg_params"]
    print(f"[params] y1c_random_forest locked params (prompt 78, not re-tuned): clf={clf_params} reg={reg_params}")
    print("[step 1 confirmation] y1c_random_forest.json has both clf_params and reg_params populated "
          "-> y1c replaced BOTH of y1's stages (Prompt 78's own Step-1 decision, reconfirmed here) -- "
          "item 4's pipeline re-run uses y1c's own classifier AND regressor, not a mixed composite.")

    print("\n[1/5] regenerating CV-OOF predictions for y1c_random_forest and y1_two_stage_huber "
          "(locked params/architecture, no re-tuning)...")
    y1c_cv = run_cv_y1c_with_oof(trainval, clf_params, reg_params)
    y1_cv = run_cv_y1_with_oof(trainval)
    df_ordered = y1c_cv["df_ordered"]
    assert np.array_equal(df_ordered[yb.GROUP_COL].to_numpy(), y1_cv["df_ordered"][yb.GROUP_COL].to_numpy()), (
        "y1c and y1 CV row ordering diverged -- folds must match exactly"
    )
    y_true = df_ordered[yb.TARGET_COL].to_numpy()

    comp_map = build_competition_map()
    trainval_comp = trainval[yb.GROUP_COL].map(comp_map)
    tournaments_present = sorted(trainval_comp.dropna().unique().tolist())
    print(f"[tournaments] present in trainval rows (reconfirmed, not assumed): {tournaments_present}")

    print("\n[2/5] tournament-stratified check...")
    y1c_tourn = tournament_breakdown(df_ordered, y_true, y1c_cv["oof_pred"], comp_map)
    y1_tourn = tournament_breakdown(df_ordered, y_true, y1_cv["oof_pred"], comp_map)
    train_composition = {str(k): int(v) for k, v in trainval.assign(competition=trainval_comp)
                          .groupby("competition")[yb.GROUP_COL].nunique().items()}
    tournament_output = {
        "population": "all train+val rows (not filtered to nonzero -- this leg's combined prediction "
                       "is defined over every row), 5-fold canonical CV out-of-fold predictions "
                       "(NOT held-out test)",
        "y1c_clf_params": clf_params, "y1c_reg_params": reg_params,
        "tournaments_present": tournaments_present,
        "train_val_match_counts_by_tournament": train_composition,
        "y1_two_stage_huber_by_tournament": y1_tourn,
        "y1c_random_forest_by_tournament": y1c_tourn,
    }
    (VALIDATION_DIR / "tournament_stratified_y1c.json").write_text(
        json.dumps(tournament_output, indent=2, default=str), encoding="utf-8"
    )
    for comp in y1_tourn:
        print(f"  {comp}: rows={y1_tourn[comp]['rows']} "
              f"y1_rmse={y1_tourn[comp]['rmse']:.5f} y1c_rmse={y1c_tourn[comp]['rmse']:.5f} "
              f"y1_r2={y1_tourn[comp]['r2']:.4f} y1c_r2={y1c_tourn[comp]['r2']:.4f}")

    print(f"\n[3/5] error-slice breakdown ({' / '.join(SLICE_COLUMNS)})...")
    y1c_err = error_slice_breakdown(df_ordered, y_true, y1c_cv["oof_pred"])
    y1_err = error_slice_breakdown(df_ordered, y_true, y1_cv["oof_pred"])
    error_output = {
        "population": "all train+val rows, 5-fold canonical CV out-of-fold predictions",
        "y1c_clf_params": clf_params, "y1c_reg_params": reg_params,
        "slice_columns": SLICE_COLUMNS,
        "y1_two_stage_huber": y1_err, "y1c_random_forest": y1c_err,
    }
    (VALIDATION_DIR / "error_analysis_y1c.json").write_text(
        json.dumps(error_output, indent=2, default=str), encoding="utf-8"
    )
    for group_col in SLICE_COLUMNS:
        y1_by_key = {r[group_col]: r for r in y1_err[group_col]}
        y1c_by_key = {r[group_col]: r for r in y1c_err[group_col]}
        for key in y1_by_key:
            a, b = y1_by_key[key], y1c_by_key.get(key, {})
            if a["rows"] < 30:
                continue
            mae_diff = b.get("mae", float("nan")) - a.get("mae", float("nan"))
            verdict = "about the same" if abs(mae_diff) < 0.001 else ("y1c improves" if mae_diff < 0 else "y1c regresses")
            print(f"  [{group_col}={key}] rows={a['rows']} y1_mae={a['mae']:.5f} "
                  f"y1c_mae={b.get('mae', float('nan')):.5f} -> {verdict}")

    print("\n[4/5] feature-shape sanity (top 8 features by y1c's aggregated Gini importance)...")
    gini_by_onehot = {r["feature"]: r["gini_importance"] for r in y1c_json["reg_full_gini_importance"]}
    top8 = top8_original_features(gini_by_onehot)
    print(f"  top 8 original features (aggregated from one-hot columns): {[f for f, _ in top8]}")
    sanity = feature_shape_sanity(df_ordered, y1c_cv["oof_pred"], [f for f, _ in top8])
    sanity_output = {
        "population": "all train+val rows, 5-fold canonical CV out-of-fold predictions",
        "y1c_clf_params": clf_params, "y1c_reg_params": reg_params,
        "note": (
            "Marginal view only -- one feature binned at a time, nothing else held fixed. Not a "
            "controlled/partial-dependence view. Row count per bin/category is reported alongside "
            "every mean so a jagged pattern can be checked against thin-bin size before being read "
            "as a real football relationship. Top-8 features are AGGREGATED from y1c_random_forest.json's "
            "one-hot-column Gini importance back to the 38 locked PASSIVE features (see "
            "top8_original_features() docstring) -- the raw one-hot ranking would have put multiple "
            "dummy levels of on_ball_event_type in the top 8, which is not 8 genuinely different "
            "features."
        ),
        "construction_coupling_caveat": (
            "ball_x/ball_y are restated here from Rung 2's own report (PASSIVE_XT_MODEL_LADDER.md "
            "section 3.6), not re-derived: reports/analysis/xt_target/PASSIVE_LEAKAGE_AUDIT.json's "
            "Part E' already found ball_x at r=+0.52 vs xt_before and r=+0.25 vs the target itself "
            "(the exact two columns xt_before = xT(ball_x, ball_y) is computed from); ball_y "
            "correlates far more weakly (r=-0.004 vs xt_before). A known, already-documented "
            "construction-coupling caveat on this leg, not new information surfaced by this audit."
        ),
        "top8_features_with_aggregated_gini": [{"feature": f, "aggregated_gini_importance": v} for f, v in top8],
        "features": sanity,
    }
    (VALIDATION_DIR / "feature_shape_sanity_y1c.json").write_text(
        json.dumps(sanity_output, indent=2, default=str), encoding="utf-8"
    )
    for feat, info in sanity.items():
        if info["type"] == "numeric":
            bins = info["bins"]
            print(f"  {feat} (numeric): first_bin={bins[0]['mean_predicted']:+.5f} "
                  f"last_bin={bins[-1]['mean_predicted']:+.5f} thin_bins(<30)={info['thin_bins_under_30_rows']}")
        else:
            cats = info["categories"]
            top_cat = max(cats, key=lambda c: c["n_rows"])
            print(f"  {feat} ({info['type']}): {info['n_categories']} categories, "
                  f"largest={top_cat['category']} (n={top_cat['n_rows']}, mean_pred={top_cat['mean_predicted']:+.5f}), "
                  f"thin(<30)={info['thin_categories_under_30_rows']}")

    print("\n[5/5] full two-stage pipeline re-run (held-out test, y1c's own artifacts, predict-only)...")
    reg_dir = yb.REG_DIR
    clf_model = joblib.load(reg_dir / "y1c_random_forest_classifier.joblib")
    reg_model = joblib.load(reg_dir / "y1c_random_forest_regressor.joblib")
    clf_builder = y1c_mod.RFDesignMatrixBuilder().fit(trainval)
    nonzero_trainval = trainval.loc[trainval[yb.TARGET_COL] != 0.0]
    reg_builder = y1c_mod.RFDesignMatrixBuilder().fit(nonzero_trainval)

    x_test_clf, _ = clf_builder.transform(test_full)
    x_test_reg, _ = reg_builder.transform(test_full)
    p_nonzero_test = clf_model.predict_proba(x_test_clf)[:, 1]
    e_delta_test = reg_model.predict(x_test_reg)
    combined_pred_y1c = p_nonzero_test * e_delta_test

    y_true_test = test_full[yb.TARGET_COL].to_numpy()
    y1c_pipeline_metrics = yb.signed_regression_metrics(y_true_test, combined_pred_y1c)
    y1c_pipeline_calib = yb.calibration_bins(y_true_test, combined_pred_y1c)

    y1_held_out_csv = pd.read_csv(yb.COMPARISONS_DIR / "passive_xt_baseline_held_out_test_readout.csv")
    y1_row = y1_held_out_csv[y1_held_out_csv["variant"] == "y1_two_stage_huber"].iloc[0].to_dict()
    print(f"  [reused, not rescored] y1_two_stage_huber's own recorded held-out row (prompt 76): "
          f"rmse={y1_row['rmse']:.5f} mae={y1_row['mae']:.5f} r2={y1_row['r2']:.5f} spearman={y1_row['spearman']:.4f}")

    pipeline_readout = {
        "held_out_test_rows": len(test_full), "held_out_test_matches": int(test_full[yb.GROUP_COL].nunique()),
        "architecture_note": (
            "y1c_random_forest replaced BOTH of y1's stages (confirmed in step [1/5] above), so this "
            "'pipeline re-run' is y1c's own classifier x regressor product -- there is no y1 stage to "
            "reuse unchanged here, unlike a hypothetical single-stage-replaced rung."
        ),
        "y1c_random_forest_combined_pipeline_metrics": y1c_pipeline_metrics,
        "y1c_random_forest_combined_pipeline_calibration_bins": y1c_pipeline_calib,
        "y1_two_stage_huber_recorded_held_out_row_prompt_76": y1_row,
        "rmse_change_vs_y1": y1c_pipeline_metrics["rmse"] - y1_row["rmse"],
        "mae_change_vs_y1": y1c_pipeline_metrics["mae"] - y1_row["mae"],
        "r2_change_vs_y1": y1c_pipeline_metrics["r2"] - y1_row["r2"],
        "spearman_change_vs_y1": y1c_pipeline_metrics["spearman"] - y1_row["spearman"],
        "note": (
            "Decisive check for the promotion decision (prompt 80, part A) -- y1c's regression-only "
            "CV win (prompt 78) is reconfirmed here via an independent, freshly-scored (predict-only, "
            "never refit) held-out combined-pipeline readout, compared directly against y1's own "
            "already-recorded held-out numbers (prompt 76's CSV row, reused unchanged, not rescored "
            "again)."
        ),
    }
    (VALIDATION_DIR / "full_pipeline_readout_y1c.json").write_text(
        json.dumps(pipeline_readout, indent=2, default=str), encoding="utf-8"
    )
    print(f"  [y1c pipeline, held-out] rmse={y1c_pipeline_metrics['rmse']:.5f} mae={y1c_pipeline_metrics['mae']:.5f} "
          f"r2={y1c_pipeline_metrics['r2']:.5f} spearman={y1c_pipeline_metrics['spearman']:.4f}")
    print(f"  [change vs y1] rmse={pipeline_readout['rmse_change_vs_y1']:+.5f} "
          f"mae={pipeline_readout['mae_change_vs_y1']:+.5f} r2={pipeline_readout['r2_change_vs_y1']:+.5f} "
          f"spearman={pipeline_readout['spearman_change_vs_y1']:+.5f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
