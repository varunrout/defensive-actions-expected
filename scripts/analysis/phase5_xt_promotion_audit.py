"""Prompt 85, Phase 5: fresh promotion audit for the 2 xT legs
(x1c_random_forest / active-xT, y1c_random_forest / passive-xT) on the
Phase-2-corrected features AND Phase-3-corrected targets
(target_xt_delta_v2 / target_xt_delta_passive).

Unlike Phase 4's 4 non-xT legs (where the target itself is proven
frame-invariant and only features changed), both xT targets are themselves
directly recomputed from corrected coordinates (COORDINATE_FRAME_IMPACT_AUDIT.md
section 1b/1c: 69.13% / 67.19% of rows exposed) -- so this is NOT a
before/after diff of the same ground truth. This script re-runs the actual
promotion comparison each leg's own validate_x1c_promotion.py /
validate_y1c_promotion.py made originally (prompts 75 / 80A): the candidate
two-stage RF (x1c/y1c) vs the immediately-preceding non-promoted rung
(x1_two_stage_huber / y1_two_stage_huber), both refit fresh on the corrected
target (the old rung's OLD held-out numbers are NOT reused for this
comparison, because they were scored against the now-provably-wrong target --
reusing them would compare a fresh model's accuracy against corrected ground
truth to a stale model's accuracy against wrong ground truth, which is not a
valid promotion comparison).

Both candidate and comparator are refit AT THEIR ALREADY-LOCKED
hyperparameters (no re-tuning) -- same scope decision and same reasoning as
Phase 4 (see phase4_reconfirm_promoted_legs.py's own docstring): x1c/y1c's
own hyperparameters were tuned in prompts 73/78 against feature-only-shifted
noise, and re-tuning here would conflate "did the fix change what's optimal"
with "did the fix change how good the already-chosen architecture is" --
this script answers the second, narrower, Phase-4-consistent question. The
FULL ladder (x0/x1/x1b/x1d and y0/y1/y1b/y1d) is not re-run -- only the
locked candidate and its immediate promoted-over comparator, per this
leg's own promotion precedent (x1c replaced x1_two_stage_huber directly).

For each leg: 5-fold canonical CV OOF (signed_regression_metrics: rmse, mae,
r2, spearman) for both models on the corrected target, paired significance
test (same as the leg's own original Rung-2 gate: paired t-test + Wilcoxon on
per-fold RMSE), plus a single held-out-test fit/score for both (held-out test
rows are read once, for the final readout only, matching original
methodology).

Writes outputs/models/validation/phase5_xt_promotion_audit.json.

Usage (run AFTER phase3 xT prototype rebuild):
    .venv/Scripts/python.exe scripts/analysis/phase5_xt_promotion_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_xt_baseline as xb  # noqa: E402
import train_passive_xt_baseline as yb  # noqa: E402
import train_x1c_random_forest as x1c_mod  # noqa: E402
import train_y1c_random_forest as y1c_mod  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

X1C_CLF_PARAMS = {"max_depth": None, "min_samples_leaf": 20}
X1C_REG_PARAMS = {"max_depth": None, "min_samples_leaf": 20}
Y1C_CLF_PARAMS = {"max_depth": None, "min_samples_leaf": 150}
Y1C_REG_PARAMS = {"max_depth": None, "min_samples_leaf": 500}

# OLD (pre-fix) held-out headline numbers -- reported for context ONLY (not used
# in the promotion decision itself, since the target changed -- see docstring).
OLD_HELD_OUT_CONTEXT_ONLY = {
    "x1_two_stage_huber": {"rmse": 0.0530853658939473, "r2": 0.1519362938505301},
    "x1c_random_forest": {"rmse": 0.042771188893547, "r2": 0.44946936363749},
    "y1_two_stage_huber": {"rmse": 0.033880219071549, "r2": 0.0611878855320654},
    "y1c_random_forest": {"rmse": 0.0248916092947857, "r2": 0.4932517978581514},
}


def paired_test(a_vals, b_vals):
    a = np.array(a_vals)
    b = np.array(b_vals)
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "mean_diff_a_minus_b": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
    }


def run_active_xt() -> dict:
    print("\n[active-xT] loading corrected data + corrected target_xt_delta_v2...")
    df_all = xb.load_data()
    test_mask = canonical_test_mask(df_all, group_col=xb.GROUP_COL)
    trainval = df_all.loc[~test_mask].reset_index(drop=True)
    test = df_all.loc[test_mask].reset_index(drop=True)
    print(f"[active-xT] trainval={len(trainval)} rows, held-out test={len(test)} rows")

    print("[active-xT] CV: x1_two_stage_huber (comparator, refit on corrected target)...")
    huber_cv = xb.run_cv(trainval, "x1_two_stage_huber")
    print("[active-xT] CV: x1c_random_forest (candidate, locked params, corrected data+target)...")
    x1c_cv = x1c_mod.run_cv_combined(trainval, X1C_CLF_PARAMS, X1C_REG_PARAMS)

    sig = paired_test(x1c_cv["fold_metrics"]["rmse"].tolist(), huber_cv["fold_metrics"]["rmse"].tolist())

    print("[active-xT] held-out test: both models, single fit...")
    huber_pred, _ = xb.fit_predict(trainval, test, "x1_two_stage_huber")
    huber_held_out = xb.signed_regression_metrics(test[xb.TARGET_COL].to_numpy(), huber_pred)
    x1c_pred, _ = x1c_mod.fit_predict(trainval, test, X1C_CLF_PARAMS, X1C_REG_PARAMS)
    x1c_held_out = xb.signed_regression_metrics(test[xb.TARGET_COL].to_numpy(), x1c_pred)

    return {
        "leg": "active-xT",
        "target": "target_xt_delta_v2 (Phase-3-corrected)",
        "candidate": "x1c_random_forest", "candidate_params": {"clf": X1C_CLF_PARAMS, "reg": X1C_REG_PARAMS},
        "comparator": "x1_two_stage_huber",
        "cv_oof_metrics": {"x1c_random_forest": x1c_cv["oof_metrics"], "x1_two_stage_huber": huber_cv["oof_metrics"]},
        "cv_fold_rmse": {"x1c_random_forest": x1c_cv["fold_metrics"]["rmse"].tolist(),
                          "x1_two_stage_huber": huber_cv["fold_metrics"]["rmse"].tolist()},
        "significance_test_x1c_minus_huber_rmse": sig,
        "held_out_test": {"x1c_random_forest": x1c_held_out, "x1_two_stage_huber": huber_held_out},
        "old_pre_fix_held_out_context_only": {
            "x1c_random_forest": OLD_HELD_OUT_CONTEXT_ONLY["x1c_random_forest"],
            "x1_two_stage_huber": OLD_HELD_OUT_CONTEXT_ONLY["x1_two_stage_huber"],
        },
    }


def run_passive_xt() -> dict:
    print("\n[passive-xT] loading corrected data + corrected target_xt_delta_passive...")
    df_all = yb.load_data()
    test_mask = canonical_test_mask(df_all, group_col=yb.GROUP_COL)
    trainval = df_all.loc[~test_mask].reset_index(drop=True)
    test = df_all.loc[test_mask].reset_index(drop=True)
    print(f"[passive-xT] trainval={len(trainval)} rows, held-out test={len(test)} rows")

    print("[passive-xT] CV: y1_two_stage_huber (comparator, refit on corrected target)...")
    huber_cv = yb.run_cv(trainval, "y1_two_stage_huber")
    print("[passive-xT] CV: y1c_random_forest (candidate, locked params, corrected data+target)...")
    y1c_cv = y1c_mod.run_cv_combined(trainval, Y1C_CLF_PARAMS, Y1C_REG_PARAMS)

    sig = paired_test(y1c_cv["fold_metrics"]["rmse"].tolist(), huber_cv["fold_metrics"]["rmse"].tolist())

    print("[passive-xT] held-out test: both models, single fit...")
    huber_pred, _ = yb.fit_predict(trainval, test, "y1_two_stage_huber")
    huber_held_out = yb.signed_regression_metrics(test[yb.TARGET_COL].to_numpy(), huber_pred)
    y1c_pred, _ = y1c_mod.fit_predict(trainval, test, Y1C_CLF_PARAMS, Y1C_REG_PARAMS)
    y1c_held_out = yb.signed_regression_metrics(test[yb.TARGET_COL].to_numpy(), y1c_pred)

    return {
        "leg": "passive-xT",
        "target": "target_xt_delta_passive (Phase-3-corrected)",
        "candidate": "y1c_random_forest", "candidate_params": {"clf": Y1C_CLF_PARAMS, "reg": Y1C_REG_PARAMS},
        "comparator": "y1_two_stage_huber",
        "cv_oof_metrics": {"y1c_random_forest": y1c_cv["oof_metrics"], "y1_two_stage_huber": huber_cv["oof_metrics"]},
        "cv_fold_rmse": {"y1c_random_forest": y1c_cv["fold_metrics"]["rmse"].tolist(),
                          "y1_two_stage_huber": huber_cv["fold_metrics"]["rmse"].tolist()},
        "significance_test_y1c_minus_huber_rmse": sig,
        "held_out_test": {"y1c_random_forest": y1c_held_out, "y1_two_stage_huber": huber_held_out},
        "old_pre_fix_held_out_context_only": {
            "y1c_random_forest": OLD_HELD_OUT_CONTEXT_ONLY["y1c_random_forest"],
            "y1_two_stage_huber": OLD_HELD_OUT_CONTEXT_ONLY["y1_two_stage_huber"],
        },
    }


def main() -> None:
    load_canonical_split()
    result = {"active_xt": run_active_xt(), "passive_xt": run_passive_xt()}

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VALIDATION_DIR / "phase5_xt_promotion_audit.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("\n=== PHASE 5 SUMMARY ===")
    for leg_key, r in result.items():
        print(f"\n{r['leg']}: candidate={r['candidate']} vs comparator={r['comparator']}")
        print(f"  held-out RMSE: candidate={r['held_out_test'][r['candidate']]['rmse']:.5f} "
              f"comparator={r['held_out_test'][r['comparator']]['rmse']:.5f}")
        print(f"  held-out R2:   candidate={r['held_out_test'][r['candidate']]['r2']:.5f} "
              f"comparator={r['held_out_test'][r['comparator']]['r2']:.5f}")
        sig_key = [k for k in r if k.startswith("significance_test")][0]
        print(f"  significance (candidate RMSE - comparator RMSE, per fold): {r[sig_key]}")
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
