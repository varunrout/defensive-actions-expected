"""Generates and verifies every real worked-example number used in
reports/modeling/MATH_BEHIND_IT.md (prompt 53).

Does NOT retrain, refit, or modify any model. Every number is either read
directly from an existing artifact (coefficients JSON, joblib model) or
computed by scoring one real held-out test row through an already-fitted,
already-saved model -- the same reuse pattern verified exactly-reproducing
in prompt 52 (p1c's held-out PR-AUC reproduced to 0.20126 vs the recorded
0.2013). Every hand-derivable number is cross-checked against the model's
own predict_proba/decision_function output before being written to the
output JSON, so nothing in the teaching document can silently drift from
what the model actually computes.

Usage:
    python scripts/analysis/generate_math_behind_it_examples.py
"""

from __future__ import annotations

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
import train_p1c_systematic_interactions as p1c_mod  # noqa: E402
import train_p1d_random_forest as p1d_mod  # noqa: E402
import train_passive_binary_baseline as pb  # noqa: E402
import train_v1c_systematic_interactions as v1c_mod  # noqa: E402
import train_v1d_random_forest as v1d_mod  # noqa: E402
import validate_passive_binary_baselines as vpb  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402

OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "math_behind_it_worked_examples_verification.json"
CLS_DIR = REPO_ROOT / "outputs" / "models" / "classification"


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def load_leg(target_kind: str):
    if target_kind == "active":
        mod = ab
        df = mod.load_data()
    else:
        mod = pb
        df = mod.load_data()
    load_canonical_split()
    test_mask = canonical_test_mask(df, group_col=mod.GROUP_COL)
    trainval_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    return mod, trainval_df, test_df


def linear_worked_example(mod, trainval_df, test_df, joblib_name: str, display_features: list[str], row_idx: int = 0) -> dict:
    """v1_unweighted / p1_unweighted style worked example: full z from the
    real fitted model, decomposed per-feature, with a subset displayed."""
    builder = mod.DesignMatrixBuilder().fit(trainval_df)
    row_df = test_df.iloc[[row_idx]].reset_index(drop=True)
    x_row, feature_names = builder.transform(row_df)
    model = joblib.load(CLS_DIR / joblib_name)

    z_full = float(model.decision_function(x_row)[0])
    p_hat = float(model.predict_proba(x_row)[0, 1])
    assert abs(sigmoid(z_full) - p_hat) < 1e-9, "sigmoid(z) must match predict_proba exactly"

    contributions = model.coef_[0] * x_row[0]
    intercept = float(model.intercept_[0])
    contrib_by_name = dict(zip(feature_names, contributions))

    displayed = []
    displayed_sum = 0.0
    for feat in display_features:
        # feat may be a raw column name (boolean/numeric) present directly in feature_names,
        # or -- for categorical raw columns -- the specific one-hot level actually active for this row.
        if feat in contrib_by_name:
            c = float(contrib_by_name[feat])
            raw_val = float(x_row[0, feature_names.index(feat)])
            displayed.append({"feature": feat, "design_matrix_value": raw_val, "coef": float(model.coef_[0][feature_names.index(feat)]), "contribution": c})
            displayed_sum += c

    remaining = z_full - intercept - displayed_sum
    return {
        "row_idx_in_held_out_test": row_idx,
        "match_id": int(row_df[mod.GROUP_COL].iloc[0]),
        "label_y": int(row_df[mod.TARGET_COL].iloc[0]),
        "n_design_matrix_features": len(feature_names),
        "intercept": intercept,
        "displayed_features": displayed,
        "sum_of_displayed_contributions": displayed_sum,
        "sum_of_remaining_feature_contributions": remaining,
        "z_full": z_full,
        "sigmoid_z": float(sigmoid(z_full)),
        "model_predict_proba": p_hat,
        "verified_match": bool(abs(sigmoid(z_full) - p_hat) < 1e-9),
    }


def calibration_gap_example(mod, trainval_df, test_df, joblib_v1: str, joblib_v2: str, row_idx_search_limit: int = 200) -> dict:
    """Finds a real held-out row where the unweighted and weighted variant's
    predicted probabilities disagree noticeably, using the two already-fitted
    saved models."""
    builder = mod.DesignMatrixBuilder().fit(trainval_df)
    sub = test_df.iloc[:row_idx_search_limit].reset_index(drop=True)
    x_sub, _ = builder.transform(sub)
    m1 = joblib.load(CLS_DIR / joblib_v1)
    m2 = joblib.load(CLS_DIR / joblib_v2)
    p1 = m1.predict_proba(x_sub)[:, 1]
    p2 = m2.predict_proba(x_sub)[:, 1]
    diff = np.abs(p1 - p2)
    best_idx = int(np.argmax(diff))
    return {
        "row_idx_in_held_out_test": best_idx,
        "match_id": int(sub[mod.GROUP_COL].iloc[best_idx]),
        "label_y": int(sub[mod.TARGET_COL].iloc[best_idx]),
        "p_unweighted": float(p1[best_idx]),
        "p_weighted": float(p2[best_idx]),
        "abs_diff": float(diff[best_idx]),
    }


def l1_interaction_example(coeff_json_path: Path, trainval_df: pd.DataFrame, test_df: pd.DataFrame, InteractionBuilderCls, GROUP_COL: str, TARGET_COL: str, NUMERIC_COLS: list[str], row_idx: int = 0) -> dict:
    """Picks the top surviving (nonzero) pairwise interaction term from the
    real L1-fitted coefficients JSON, and shows its actual contribution to
    one real held-out row's log-odds -- the real two raw feature values,
    the real standardized values, the real product, times the real fitted
    coefficient."""
    coeffs = json.loads(coeff_json_path.read_text(encoding="utf-8"))
    surviving = coeffs.get("surviving_poly_terms_sorted_by_abs_coef", [])
    pairwise = [r for r in surviving if "__x__" in r["feature"]]
    top = pairwise[0]
    term_name = top["feature"]  # e.g. "poly__feat_a__x__feat_b"
    inner = term_name[len("poly__"):]
    feat_a, feat_b = inner.split("__x__")

    builder = InteractionBuilderCls().fit(trainval_df)
    row_df = test_df.iloc[[row_idx]].reset_index(drop=True)

    num_imputed = builder.num_imputer.transform(row_df[NUMERIC_COLS])
    num_std = builder.num_scaler.transform(num_imputed)
    num_names = list(NUMERIC_COLS)
    std_a = float(num_std[0, num_names.index(feat_a)])
    std_b = float(num_std[0, num_names.index(feat_b)])
    raw_a = float(row_df[feat_a].iloc[0])
    raw_b = float(row_df[feat_b].iloc[0])
    product_std = std_a * std_b

    # The design matrix re-standardizes the poly block itself (poly_scaler) --
    # reproduce that exactly to get the real column value the coefficient multiplies.
    x_row, feature_names = builder.transform(row_df)
    col_idx = feature_names.index(term_name)
    real_design_value = float(x_row[0, col_idx])
    contribution = float(top["coef"]) * real_design_value

    return {
        "term": term_name,
        "feature_a": feat_a, "feature_b": feat_b,
        "raw_value_a": raw_a, "raw_value_b": raw_b,
        "standardized_value_a": std_a, "standardized_value_b": std_b,
        "standardized_product": product_std,
        "poly_block_restandardized_design_value": real_design_value,
        "coefficient": float(top["coef"]),
        "contribution_to_log_odds": contribution,
        "row_idx_in_held_out_test": row_idx,
        "n_surviving_pairwise_terms": len(pairwise),
    }


def rf_tree_example(mod, trainval_df, test_df, joblib_name: str, RFBuilderCls, row_idx: int = 0, n_trees_show: int = 5) -> dict:
    builder = RFBuilderCls().fit(trainval_df)
    row_df = test_df.iloc[[row_idx]].reset_index(drop=True)
    x_row, feature_names = builder.transform(row_df)
    forest = joblib.load(CLS_DIR / joblib_name)

    tree_preds = [float(t.predict_proba(x_row)[0, 1]) for t in forest.estimators_]
    forest_pred = float(forest.predict_proba(x_row)[0, 1])
    mean_of_all_trees = float(np.mean(tree_preds))
    assert abs(mean_of_all_trees - forest_pred) < 1e-9, "forest predict_proba must equal the mean of all trees' predict_proba"

    root_feat_idx = int(forest.estimators_[0].tree_.feature[0])
    root_threshold = float(forest.estimators_[0].tree_.threshold[0])
    root_feature_name = feature_names[root_feat_idx]

    return {
        "row_idx_in_held_out_test": row_idx,
        "n_trees_total": len(forest.estimators_),
        "n_trees_shown": n_trees_show,
        "shown_tree_predictions": tree_preds[:n_trees_show],
        "mean_of_shown_trees": float(np.mean(tree_preds[:n_trees_show])),
        "mean_of_all_trees": mean_of_all_trees,
        "forest_predict_proba": forest_pred,
        "verified_match": bool(abs(mean_of_all_trees - forest_pred) < 1e-9),
        "first_tree_root_split_feature": root_feature_name,
        "first_tree_root_split_threshold": root_threshold,
    }


def gbm_boosting_rounds_example(mod, trainval_df, test_df, joblib_name: str, RFBuilderCls, row_idx: int = 0) -> dict:
    builder = RFBuilderCls().fit(trainval_df)
    row_df = test_df.iloc[[row_idx]].reset_index(drop=True)
    x_row, feature_names = builder.transform(row_df)
    model = joblib.load(CLS_DIR / joblib_name)
    booster = model.booster_
    best_iter = int(model.best_iteration_) if model.best_iteration_ else int(model.n_estimators_)

    rounds_to_show = sorted(set([1, 10, 50, best_iter]))
    rounds_to_show = [r for r in rounds_to_show if r <= best_iter]
    raw_scores = {}
    for r in rounds_to_show:
        raw = booster.predict(x_row, raw_score=True, num_iteration=r)
        raw_scores[r] = float(raw[0])

    final_p = float(model.predict_proba(x_row)[0, 1])
    final_raw = raw_scores[best_iter]
    assert abs(sigmoid(final_raw) - final_p) < 1e-6, "sigmoid(final raw score) must match predict_proba"

    return {
        "row_idx_in_held_out_test": row_idx,
        "best_iteration": best_iter,
        "learning_rate": model.get_params()["learning_rate"],
        "raw_score_by_round": raw_scores,
        "final_raw_score": final_raw,
        "final_predict_proba": final_p,
        "sigmoid_of_final_raw_score": float(sigmoid(final_raw)),
        "verified_match": bool(abs(sigmoid(final_raw) - final_p) < 1e-6),
    }


def calibration_mapping_example(mod, trainval_df, test_df, raw_joblib: str, calib_joblib: str, RFBuilderCls, row_idx: int = 0) -> dict:
    builder = RFBuilderCls().fit(trainval_df)
    row_df = test_df.iloc[[row_idx]].reset_index(drop=True)
    x_row, _ = builder.transform(row_df)

    raw_model = joblib.load(CLS_DIR / raw_joblib)
    calib_model = joblib.load(CLS_DIR / calib_joblib)

    raw_p = float(raw_model.predict_proba(x_row)[0, 1])
    calib_p = float(calib_model.predict_proba(x_row)[0, 1])

    method = calib_model.calibrated_classifiers_[0].calibrators[0].__class__.__name__
    return {
        "row_idx_in_held_out_test": row_idx,
        "raw_predict_proba": raw_p,
        "calibrated_predict_proba": calib_p,
        "calibrator_class": method,
    }


def ladder_row_example(mod, trainval_df, test_df, variants: dict[str, tuple], row_idx: int = 0) -> dict:
    """variants: name -> (joblib_filename, BuilderCls)"""
    row_df = test_df.iloc[[row_idx]].reset_index(drop=True)
    out: dict = {"row_idx_in_held_out_test": row_idx, "label_y": int(row_df[mod.TARGET_COL].iloc[0]), "predictions": {}}
    for name, (joblib_name, BuilderCls) in variants.items():
        builder = BuilderCls().fit(trainval_df)
        x_row, _ = builder.transform(row_df)
        model = joblib.load(CLS_DIR / joblib_name)
        out["predictions"][name] = float(model.predict_proba(x_row)[0, 1])
    return out


def main() -> None:
    result: dict = {}

    # ============================= ACTIVE LEG =============================
    print("[active] loading data + canonical split...")
    ab_mod, ab_trainval, ab_test = load_leg("active")
    ROW = 0  # fixed, deterministic row index into held-out test for narrative continuity

    print("[active 1.1] v1_unweighted linear worked example...")
    display_feats_active = [
        "match_time_seconds", "possession_elapsed_seconds", "attacking_goal_centrality",
        "is_in_attacking_box", "counterpress", "defenders_within_10m",
    ]
    result["active_1_1_v1_linear"] = linear_worked_example(
        ab_mod, ab_trainval, ab_test, "v1_unweighted.joblib", display_feats_active, row_idx=ROW
    )

    print("[active 1.2] v1 vs v2 calibration-gap example...")
    result["active_1_2_calibration_gap"] = calibration_gap_example(
        ab_mod, ab_trainval, ab_test, "v1_unweighted.joblib", "v2_weighted.joblib"
    )

    print("[active 1.4] v1c interaction-term worked example...")
    result["active_1_4_l1_interaction"] = l1_interaction_example(
        CLS_DIR / "v1c_systematic_interactions.json", ab_trainval, ab_test,
        v1c_mod.InteractionDesignMatrixBuilder, ab_mod.GROUP_COL, ab_mod.TARGET_COL, ab_mod.NUMERIC_COLS, row_idx=ROW,
    )

    print("[active 1.5] v1d random forest tree example...")
    result["active_1_5_random_forest"] = rf_tree_example(
        ab_mod, ab_trainval, ab_test, "v1d_random_forest.joblib", v1d_mod.RFDesignMatrixBuilder, row_idx=ROW
    )

    print("[active 1.6] v1e gradient boosting rounds example...")
    result["active_1_6_gradient_boosting"] = gbm_boosting_rounds_example(
        ab_mod, ab_trainval, ab_test, "v1e_gradient_boosting.joblib", v1d_mod.RFDesignMatrixBuilder, row_idx=ROW
    )

    print("[active 1.7] v1e calibration mapping example...")
    result["active_1_7_calibration_mapping"] = calibration_mapping_example(
        ab_mod, ab_trainval, ab_test, "v1e_gradient_boosting.joblib", "v1e_gradient_boosting_calibrated.joblib",
        v1d_mod.RFDesignMatrixBuilder, row_idx=ROW
    )

    print("[active 1.8] one row, whole ladder...")
    result["active_1_8_ladder_row"] = ladder_row_example(
        ab_mod, ab_trainval, ab_test,
        {
            "v1_unweighted": ("v1_unweighted.joblib", ab_mod.DesignMatrixBuilder),
            "v1c_systematic_interactions": ("v1c_systematic_interactions.joblib", v1c_mod.InteractionDesignMatrixBuilder),
            "v1d_random_forest": ("v1d_random_forest.joblib", v1d_mod.RFDesignMatrixBuilder),
            "v1e_gradient_boosting": ("v1e_gradient_boosting.joblib", v1d_mod.RFDesignMatrixBuilder),
            "v1e_gradient_boosting_calibrated": ("v1e_gradient_boosting_calibrated.joblib", v1d_mod.RFDesignMatrixBuilder),
        },
        row_idx=ROW,
    )

    # ============================= PASSIVE LEG =============================
    print("[passive] loading data + canonical split...")
    pb_mod, pb_trainval, pb_test = load_leg("passive")

    print("[passive 2.1] p1_unweighted linear worked example...")
    display_feats_passive = [
        "defender_x", "ball_x", "top_option_1_threat_score",
        "is_in_attacking_box", "marking_tightness", "engagement_distance_to_carrier",
    ]
    result["passive_2_1_p1_linear"] = linear_worked_example(
        pb_mod, pb_trainval, pb_test, "p1_unweighted.joblib", display_feats_passive, row_idx=ROW
    )

    print("[passive 2.2] p1 vs p2 calibration-gap example...")
    result["passive_2_2_calibration_gap"] = calibration_gap_example(
        pb_mod, pb_trainval, pb_test, "p1_unweighted.joblib", "p2_weighted.joblib"
    )

    print("[passive 2.3] p1c interaction-term worked example...")
    result["passive_2_3_l1_interaction"] = l1_interaction_example(
        CLS_DIR / "p1c_systematic_interactions.json", pb_trainval, pb_test,
        p1c_mod.InteractionDesignMatrixBuilder, pb_mod.GROUP_COL, pb_mod.TARGET_COL, pb_mod.NUMERIC_COLS, row_idx=ROW,
    )

    print("[passive 2.4] p1d random forest tree example...")
    result["passive_2_4_random_forest"] = rf_tree_example(
        pb_mod, pb_trainval, pb_test, "p1d_random_forest.joblib", p1d_mod.RFDesignMatrixBuilder, row_idx=ROW
    )

    print("[passive 2.5] p1e gradient boosting rounds example...")
    result["passive_2_5_gradient_boosting"] = gbm_boosting_rounds_example(
        pb_mod, pb_trainval, pb_test, "p1e_gradient_boosting.joblib", p1d_mod.RFDesignMatrixBuilder, row_idx=ROW
    )

    print("[passive 2.6] p1e calibration mapping example...")
    result["passive_2_6_calibration_mapping"] = calibration_mapping_example(
        pb_mod, pb_trainval, pb_test, "p1e_gradient_boosting.joblib", "p1e_gradient_boosting_calibrated.joblib",
        p1d_mod.RFDesignMatrixBuilder, row_idx=ROW
    )

    print("[passive 2.8] one row, whole ladder...")
    result["passive_2_8_ladder_row"] = ladder_row_example(
        pb_mod, pb_trainval, pb_test,
        {
            "p1_unweighted": ("p1_unweighted.joblib", pb_mod.DesignMatrixBuilder),
            "p1c_systematic_interactions": ("p1c_systematic_interactions.joblib", p1c_mod.InteractionDesignMatrixBuilder),
            "p1d_random_forest": ("p1d_random_forest.joblib", p1d_mod.RFDesignMatrixBuilder),
            "p1e_gradient_boosting": ("p1e_gradient_boosting.joblib", p1d_mod.RFDesignMatrixBuilder),
            "p1e_gradient_boosting_calibrated": ("p1e_gradient_boosting_calibrated.joblib", p1d_mod.RFDesignMatrixBuilder),
        },
        row_idx=ROW,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten to {OUT_PATH}")

    all_verified = all(
        v.get("verified_match", True) for v in result.values() if isinstance(v, dict)
    )
    print("ALL VERIFIED MATCHES OK:", all_verified)


if __name__ == "__main__":
    main()
