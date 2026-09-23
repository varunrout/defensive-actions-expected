"""Prompt 85, Phase 4 addendum: single held-out-test refit of each of the 4
non-xT promoted models at their locked hyperparameters, on Phase-2-corrected
features, for a direct old-vs-new headline-metric table (the held-out test
set is read exactly once per model, matching each leg's own
FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION convention -- not used for any
selection decision, which is made entirely from the Phase-4 CV comparison in
phase4_reconfirm_promoted_legs.py).

Writes outputs/models/validation/phase4_held_out_refit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
import train_c1d_random_forest as c1d_mod  # noqa: E402
import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_continuous_baseline as pcb  # noqa: E402
import train_v1e_gradient_boosting as v1e_mod  # noqa: E402
import train_p1e_gradient_boosting as p1e_mod  # noqa: E402
from dax.models.evaluation import classification_metrics, regression_metrics  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"


def main() -> None:
    load_canonical_split()
    result = {}

    # v1e
    df_all = tb.load_data()
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval, test = df_all.loc[~test_mask].reset_index(drop=True), df_all.loc[test_mask].reset_index(drop=True)
    v1e_params = {"learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 30}
    score, _, _ = v1e_mod.fit_predict_gbm(trainval, test, trainval[tb.TARGET_COL].to_numpy(), test[tb.TARGET_COL].to_numpy(), v1e_params)
    result["v1e_gradient_boosting"] = classification_metrics(test[tb.TARGET_COL].to_numpy(), score)
    print("[v1e] held-out:", result["v1e_gradient_boosting"])

    # p1e
    df_all_p = pb.load_data()
    test_mask_p = canonical_test_mask(df_all_p, group_col=pb.GROUP_COL)
    trainval_p, test_p = df_all_p.loc[~test_mask_p].reset_index(drop=True), df_all_p.loc[test_mask_p].reset_index(drop=True)
    p1e_params = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 500}
    score_p, _, _ = p1e_mod.fit_predict_gbm(trainval_p, test_p, trainval_p[pb.TARGET_COL].to_numpy(), test_p[pb.TARGET_COL].to_numpy(), p1e_params)
    result["p1e_gradient_boosting"] = classification_metrics(test_p[pb.TARGET_COL].to_numpy(), score_p)
    print("[p1e] held-out:", result["p1e_gradient_boosting"])

    # c1d
    pos_all_c = df_all[df_all[tb.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask_c = canonical_test_mask(pos_all_c, group_col=tb.GROUP_COL)
    pos_trainval_c, pos_test_c = pos_all_c.loc[~pos_test_mask_c].reset_index(drop=True), pos_all_c.loc[pos_test_mask_c].reset_index(drop=True)
    c1d_params = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10}
    import numpy as np
    log_pred, sigma2, _, _, _ = c1d_mod.fit_predict_c1d(pos_trainval_c, pos_test_c, c1d_params)
    y_true_xg = pos_test_c["target_future_xg_10s"].to_numpy()
    y_true_log = np.log(y_true_xg)
    log_resid = log_pred - y_true_log
    import importlib
    acb = importlib.import_module("train_active_continuous_baseline")
    xg_naive = acb.backtransform(log_pred, sigma2, "naive")
    xg_corrected = acb.backtransform(log_pred, sigma2, "lognormal_corrected")
    common_log_resid = np.log(xg_naive) - np.log(y_true_xg)
    result["c1d_random_forest"] = {
        "common_log_rmse": float(np.sqrt(np.mean(common_log_resid ** 2))),
        "common_log_mae": float(np.mean(np.abs(common_log_resid))),
    }
    result["c1d_random_forest"].update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg, xg_corrected).items()})
    print("[c1d] held-out:", result["c1d_random_forest"])

    # d1
    df_all_pc = pb.load_data()
    pos_all_d = df_all_pc[df_all_pc[pb.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask_d = canonical_test_mask(pos_all_d, group_col=pb.GROUP_COL)
    pos_trainval_d, pos_test_d = pos_all_d.loc[~pos_test_mask_d].reset_index(drop=True), pos_all_d.loc[pos_test_mask_d].reset_index(drop=True)
    log_pred_d, sigma2_d, _, _ = pcb.fit_predict_regression(pos_trainval_d, pos_test_d, "d1_lognormal_glm")
    y_true_xg_d = pos_test_d["target_future_xg_10s"].to_numpy()
    y_true_log_d = np.log(y_true_xg_d)
    log_resid_d = log_pred_d - y_true_log_d
    xg_naive_d = pcb.backtransform(log_pred_d, sigma2_d, "naive")
    xg_corrected_d = pcb.backtransform(log_pred_d, sigma2_d, "lognormal_corrected")
    common_log_resid_d = np.log(xg_naive_d) - np.log(y_true_xg_d)
    result["d1_lognormal_glm"] = {
        "common_log_rmse": float(np.sqrt(np.mean(common_log_resid_d ** 2))),
        "common_log_mae": float(np.mean(np.abs(common_log_resid_d))),
    }
    result["d1_lognormal_glm"].update({f"corrected_{k}": v for k, v in regression_metrics(y_true_xg_d, xg_corrected_d).items()})
    print("[d1] held-out:", result["d1_lognormal_glm"])

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / "phase4_held_out_refit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("\n[write] phase4_held_out_refit.json")


if __name__ == "__main__":
    main()
