"""Prompt 85, Phase 4: re-validate the 4 non-xT legs' promoted reference models
against the Phase-2-corrected features, and Phase 3-corrected targets are NOT
involved here (target_future_shot_10s / target_future_xg_10s are proven
frame-invariant, COORDINATE_FRAME_IMPACT_AUDIT.md section 1a/1d -- 0 rows
affected). Only the 34/38 locked ACTIVE/PASSIVE features change.

For each of the 4 legs' currently-promoted reference model, refit it AT ITS
ALREADY-LOCKED HYPERPARAMETERS (no re-tuning -- see the methodology note
below for why) on the corrected feature parquets, using the exact same
5-fold canonical-grouped CV each leg's own promotion used, and apply the
SAME paired significance test each leg's own train_*.py script already used
to gate its own promotion (scipy.stats.ttest_rel + wilcoxon on the per-fold
primary metric): old (pre-fix, on-disk) fold array vs new (post-fix,
recomputed here) fold array, both on the IDENTICAL canonical fold assignment
(canonical_grouped_folds is a deterministic function of match_id, unaffected
by the coordinate fix).

Methodology note / explicit choice made under ambiguity (per the prompt's own
instruction to make and document a defensible choice rather than stall):
this script refits each promoted model at its LOCKED hyperparameters rather
than re-running each leg's full historical grid-search ladder. Re-running the
full tune_*() grid search (each combo x 5 folds) for all 4 legs was
considered and rejected: (1) the prompt's own Phase 4 instruction explicitly
scopes this to "re-validate the promoted model per leg... not the full
historical ladder", and (2) holding hyperparameters fixed isolates the
coordinate-fix's own effect on the metric from any re-tuning noise, which is
the cleaner scientific comparison for a "did the corrected data move the
metric outside the noise band" question. A full re-tune remains a reasonable
alternative a future prompt could still run; it was not chosen here because
it would answer a different question (does a corrected feature set change
which hyperparameters are optimal) than the one Phase 4 asks (did the fix
move THIS model's metric outside the band that governed its own promotion).

Legs covered, each at its already-recorded promoted/locked configuration:
  - Active-binary:      v1e_gradient_boosting            {learning_rate:0.01, num_leaves:63, min_child_samples:30}
  - Passive-binary:     p1e_gradient_boosting             {learning_rate:0.05, num_leaves:31, min_child_samples:500}
  - Active-continuous:  c1d_random_forest                 {n_estimators:200, max_depth:8, min_samples_leaf:10}
  - Passive-continuous: d1_lognormal_glm                  (no hyperparameters -- plain LinearRegression on log(xg))

Primary metric per leg (matches each leg's own original significance test):
  - v1e / p1e: per-fold average_precision (PR-AUC), higher is better
  - c1d / d1:  per-fold common_log_rmse (natural-log(xg) scale, back-transformed to a
               common naive scale), lower is better

Writes outputs/models/validation/phase4_reconfirmation.json.

Usage:
    .venv/Scripts/python.exe scripts/analysis/phase4_reconfirm_promoted_legs.py
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

import train_active_binary_baseline as tb  # noqa: E402
import train_active_continuous_baseline as acb  # noqa: E402
import train_c1d_random_forest as c1d_mod  # noqa: E402
import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_continuous_baseline as pcb  # noqa: E402
import train_v1e_gradient_boosting as v1e_mod  # noqa: E402
import train_p1e_gradient_boosting as p1e_mod  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"

# --- OLD (pre-fix) per-fold arrays, read directly off this repo's own already-committed
# significance JSONs (frozen before Phase 2 overwrote any parquet -- these numbers are
# untouched by anything this script does; reused for the paired comparison only). ---
OLD_FOLD_METRICS = {
    "v1e_gradient_boosting": {
        "metric": "average_precision",
        "values": [0.42626249168984176, 0.4455479722685746, 0.3921197125604135, 0.44882345590430006, 0.38428205047484354],
        "source": "outputs/models/validation/significance_v1e_vs_v1d_v1c.json:per_fold_average_precision.v1e_gradient_boosting",
    },
    "p1e_gradient_boosting": {
        "metric": "average_precision",
        "values": [0.22300721935126552, 0.23722894878082765, 0.20451168570651523, 0.24426876734743883, 0.2124396935618378],
        "source": "outputs/models/validation/significance_p1e_vs_p1d_p1c_systematic_interactions.json:per_fold_average_precision.p1e_gradient_boosting",
    },
    "c1d_random_forest": {
        "metric": "common_log_rmse",
        "values": [0.9403625757258852, 0.9498993227454405, 1.0558139218446836, 1.055738419067374, 0.8892239399953047],
        "source": "outputs/models/validation/significance_c1_vs_c1d_random_forest.json:fold_common_log_rmse.c1d_random_forest",
    },
    "d1_lognormal_glm": {
        "metric": "common_log_rmse",
        "values": None,  # filled in main() from the backed-up JSON, longer array (passive has more folds/rows)
        "source": "outputs/models/validation/significance_d1_vs_d1d_random_forest.json:fold_common_log_rmse.d1_lognormal_glm",
    },
}

# --- OLD held-out headline metrics (pre-fix), from this repo's own comparison CSVs,
# read before Phase 2/3 touched anything. ---
OLD_HELD_OUT = {
    "v1e_gradient_boosting_calibrated": {
        "average_precision": 0.4281721587160004, "roc_auc": 0.8364320994869592,
        "log_loss": 0.19429926155873148, "brier_score": 0.05224687975285264,
        "expected_calibration_error": 0.009551648597652005,
    },
    "p1e_gradient_boosting_calibrated": {
        "average_precision": 0.2161872368673111, "roc_auc": 0.7888812374081401,
        "log_loss": 0.1818146624721417, "brier_score": 0.047420197649619904,
        "expected_calibration_error": 0.004405561663363075,
    },
    "c1d_random_forest": {
        "common_log_rmse": 0.9553324022208108, "corrected_rmse": 0.1383742903009976,
        "corrected_r2": 0.0924708699793267, "corrected_mae": 0.088960632151991,
    },
    "d1_lognormal_glm": {
        "common_log_rmse": 1.0374442571058435, "corrected_rmse": 0.1300950496166491,
        "corrected_r2": 0.0078358797019425, "corrected_mae": 0.0986279348568879,
    },
}


def paired_test(old_vals, new_vals):
    a = np.array(new_vals)
    b = np.array(old_vals)
    diff = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    if len(set(diff.round(12))) > 1:
        w_stat, w_p = stats.wilcoxon(a, b)
    else:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "old_fold_values": b.tolist(),
        "new_fold_values": a.tolist(),
        "mean_diff_new_minus_old": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
        "n_folds": int(len(a)),
    }


def main() -> None:
    result: dict = {}

    old_d1 = json.loads((REPO_ROOT / ".claude_scratch" / "prefix_backup" / "models_full_backup" / "validation" / "significance_d1_vs_d1d_random_forest.json").read_text(encoding="utf-8"))
    OLD_FOLD_METRICS["d1_lognormal_glm"]["values"] = old_d1["fold_common_log_rmse"]["d1_lognormal_glm"]

    load_canonical_split()

    # ---- v1e_gradient_boosting (active-binary) ----
    print("[v1e] loading corrected active-binary data...")
    df_all = tb.load_data()
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval = df_all.loc[~test_mask].reset_index(drop=True)
    test = df_all.loc[test_mask].reset_index(drop=True)
    v1e_params = {"learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 30}
    print("[v1e] running 5-fold canonical CV at locked params (no re-tuning)...")
    v1e_cv = v1e_mod.run_cv_gbm(trainval, v1e_params)
    v1e_new_fold_ap = v1e_cv["fold_metrics"]["average_precision"].tolist()
    print(f"[v1e] new fold AP: {v1e_new_fold_ap}")
    result["v1e_gradient_boosting"] = {
        "leg": "active-binary", "locked_params": v1e_params,
        "significance_test": paired_test(OLD_FOLD_METRICS["v1e_gradient_boosting"]["values"], v1e_new_fold_ap),
        "new_oof_metrics": v1e_cv["oof_metrics"],
        "old_held_out_calibrated": OLD_HELD_OUT["v1e_gradient_boosting_calibrated"],
    }

    # ---- p1e_gradient_boosting (passive-binary) ----
    print("[p1e] loading corrected passive-binary data...")
    df_all_p = pb.load_data()
    test_mask_p = canonical_test_mask(df_all_p, group_col=pb.GROUP_COL)
    trainval_p = df_all_p.loc[~test_mask_p].reset_index(drop=True)
    p1e_params = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 500}
    print("[p1e] running 5-fold canonical CV at locked params (no re-tuning)...")
    p1e_cv = p1e_mod.run_cv_gbm(trainval_p, p1e_params)
    p1e_new_fold_ap = p1e_cv["fold_metrics"]["average_precision"].tolist()
    print(f"[p1e] new fold AP: {p1e_new_fold_ap}")
    result["p1e_gradient_boosting"] = {
        "leg": "passive-binary", "locked_params": p1e_params,
        "significance_test": paired_test(OLD_FOLD_METRICS["p1e_gradient_boosting"]["values"], p1e_new_fold_ap),
        "new_oof_metrics": p1e_cv["oof_metrics"],
        "old_held_out_calibrated": OLD_HELD_OUT["p1e_gradient_boosting_calibrated"],
    }

    # ---- c1d_random_forest (active-continuous) ----
    print("[c1d] loading corrected active-continuous data (shot-conditional positive rows)...")
    df_all_c = tb.load_data()
    pos_all_c = df_all_c[df_all_c[tb.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask_c = canonical_test_mask(pos_all_c, group_col=tb.GROUP_COL)
    pos_trainval_c = pos_all_c.loc[~pos_test_mask_c].reset_index(drop=True)
    c1d_params = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10}
    print("[c1d] running 5-fold canonical CV at locked params (no re-tuning)...")
    c1d_cv = c1d_mod.run_cv_c1d(pos_trainval_c, c1d_params)
    c1d_new_fold_rmse = c1d_cv["fold_metrics"]["common_log_rmse"].tolist()
    print(f"[c1d] new fold common_log_rmse: {c1d_new_fold_rmse}")
    result["c1d_random_forest"] = {
        "leg": "active-continuous", "locked_params": c1d_params,
        "significance_test": paired_test(OLD_FOLD_METRICS["c1d_random_forest"]["values"], c1d_new_fold_rmse),
        "new_oof_metrics": c1d_cv["oof_metrics"],
        "old_held_out": OLD_HELD_OUT["c1d_random_forest"],
        "lower_is_better": True,
    }

    # ---- d1_lognormal_glm (passive-continuous) ----
    print("[d1] loading corrected passive-continuous data (shot-conditional positive rows)...")
    df_all_d = pb.load_data()
    pos_all_d = df_all_d[df_all_d[pb.TARGET_COL] == 1].reset_index(drop=True)
    pos_test_mask_d = canonical_test_mask(pos_all_d, group_col=pb.GROUP_COL)
    pos_trainval_d = pos_all_d.loc[~pos_test_mask_d].reset_index(drop=True)
    print("[d1] running 5-fold canonical CV (no hyperparameters -- plain LinearRegression)...")
    d1_cv = pcb.run_cv_regression(pos_trainval_d, "d1_lognormal_glm")
    d1_new_fold_rmse = d1_cv["fold_metrics"]["common_log_rmse"].tolist()
    print(f"[d1] new fold common_log_rmse: {d1_new_fold_rmse}")
    result["d1_lognormal_glm"] = {
        "leg": "passive-continuous", "locked_params": "none (plain LinearRegression on log(xg))",
        "significance_test": paired_test(OLD_FOLD_METRICS["d1_lognormal_glm"]["values"], d1_new_fold_rmse),
        "new_oof_metrics": d1_cv["oof_metrics"],
        "old_held_out": OLD_HELD_OUT["d1_lognormal_glm"],
        "lower_is_better": True,
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VALIDATION_DIR / "phase4_reconfirmation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("\n=== PHASE 4 SUMMARY ===")
    for leg, r in result.items():
        sig = r["significance_test"]
        print(f"{leg}: mean_diff(new-old)={sig['mean_diff_new_minus_old']:+.4f} "
              f"paired_t_p={sig['paired_t_pvalue']:.4f} wilcoxon_p={sig['wilcoxon_pvalue']}")
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
