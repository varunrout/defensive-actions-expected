"""Export the model-ladder rung inventory for all 6 modelling legs to
dashboard_data/model_ladders.json.

Unlike export_feature_journey.py and export_analysis_facts.py, this exporter
cannot parse its facts out of machine-readable JSON -- the MODEL_LADDER
reports under reports/modeling/ are narrative markdown. Every rung entry
below was individually verified by reading the actual report text (not
assumed from any pre-written list), including cross-checking against
reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md's Phase 4/5 results
to determine which headline numbers are pre- vs post- the coordinate-frame
fix (Prompt 85/86). `source` on every rung cites the exact report + section.

Specific things independently verified while building this (see notes on the
relevant rungs below for what was actually found, which in a couple of cases
differs from what might be assumed from rung naming alone):
  - c1b IS in the active-continuous ladder (Rung 1, not promoted).
  - c1c (active-continuous) and d1c (passive-continuous) are BOTH
    skipped-by-design, but for two independently-derived reasons -- not the
    same rationale copy-pasted across legs.
  - No passive-continuous gradient-boosting rung was ever attempted; the
    leg's Rung-0 GLM (d1_lognormal_glm) is never beaten by anything.
  - p1e (passive-binary)'s real current headline is PR-AUC 0.2267 / ROC-AUC
    0.8066, post the Prompt-86 coordinate-frame refit. No "CV-only"
    calibration caveat is attached to it anywhere in the corpus; its actual
    caveat is a small (n=259) defender_functional_role="unclassified" slice.
  - "x1" and "x1c" are DIFFERENT rungs on the active-xT leg: x1 is the Rung-0
    two-stage-Huber baseline/comparator (corrected R²=0.11908), x1c is the
    Rung-2 promoted RandomForest candidate (corrected R²=0.37407). The same
    pattern repeats on passive-xT as y1 (R²=0.12473) vs y1c (R²=0.33130).

Run from the repo root:
    .venv/Scripts/python.exe scripts/dashboard/export_model_ladders.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "dashboard_data" / "model_ladders.json"

MODELING = "reports/modeling"


def rung(name, status, metric, note, source, pre_fix=False, headline_metric_name=None):
    return {
        "name": name,
        "status": status,  # promoted | tried_not_promoted | mixed | skipped_by_design
        "headline_metric_name": headline_metric_name,
        "headline_metric_value": metric,
        "pre_fix": pre_fix,
        "note": note,
        "source": source,
    }


LEGS = {
    "active_binary": {
        "label": "Active - Binary (P(shot within 10s))",
        "reference_model": "v1e_gradient_boosting_calibrated",
        "reference_model_status": "promoted, coordinate-frame fix RECONFIRMED (mean fold-AP shift +0.0008, paired-t p=0.929, indistinguishable from zero); original Prompt-46 promotion stands, closed, not reopened.",
        "coordinate_fix_verdict": "RECONFIRMED",
        "rungs": [
            rung(
                "v1_unweighted", "promoted", 0.372, "Rung 0 baseline. Held-out PR-AUC 0.372, ROC-AUC 0.819, ECE 0.011.",
                f"{MODELING}/active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md", headline_metric_name="PR-AUC",
            ),
            rung(
                "v1b_quadratic", "tried_not_promoted", 0.3713,
                "Statistically real CV lift on 4 known U-shaped features (+0.0021 PR-AUC, p=0.0122) but does not clearly survive held-out (-0.0012 vs v1). Not promoted.",
                f"{MODELING}/active_binary/ACTIVE_BINARY_MODEL_LADDER.md #2.4 Bottom line", headline_metric_name="PR-AUC",
            ),
            rung(
                "v1c_systematic_interactions", "promoted", 0.3747,
                "Promoted Prompt 43 (gain survives held-out, unlike Rung 1); later superseded by v1e_gradient_boosting_calibrated (Prompt 46).",
                f"{MODELING}/active_binary/ACTIVE_BINARY_MODEL_LADDER.md #3.6", headline_metric_name="PR-AUC",
            ),
            rung(
                "v1d_random_forest", "tried_not_promoted", 0.4085,
                "Beats v1c on PR-AUC (0.4085) but ECE 0.0183 is 1.8x worse (calibration). Explicitly 'a diagnostic rung, not a promotion candidate in its own right' -- no promotion audit run.",
                f"{MODELING}/active_binary/ACTIVE_BINARY_MODEL_LADDER.md #4.7", headline_metric_name="PR-AUC",
            ),
            rung(
                "v1e_gradient_boosting_calibrated", "promoted", 0.4282,
                "Promoted Prompt 46, current standing reference model, superseding v1c. ROC-AUC 0.8364, Brier 0.0522, ECE 0.0096. Coordinate-fix RECONFIRMED unchanged (paired-t p=0.929).",
                f"{MODELING}/active_binary/ACTIVE_BINARY_MODEL_LADDER.md #5.8", headline_metric_name="PR-AUC",
            ),
        ],
    },
    "active_continuous": {
        "label": "Active - Continuous (E[xG|shot] hurdle component)",
        "reference_model": "c1d_random_forest",
        "reference_model_status": "promoted (Prompt 58), coordinate-frame fix REOPENED (metric moved outside noise, favourable direction) -- rung stands, headline numbers updated post-fix.",
        "coordinate_fix_verdict": "REOPENED",
        "rungs": [
            rung(
                "c1_lognormal_glm", "promoted", 0.9951,
                "Rung 0 baseline. Held-out common log RMSE 0.9951, corrected R²=0.0371.",
                f"{MODELING}/active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md #1", headline_metric_name="common_log_rmse",
            ),
            rung(
                "c1b_quadratic", "tried_not_promoted", 1.0027,
                "Null result: 'does not beat c1 -- on either CV or held-out, on either the common-scale metric or original-scale MAE/R2.' Paired-t p=0.2073, Wilcoxon p=0.1875.",
                f"{MODELING}/active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md #2.4", headline_metric_name="common_log_rmse (CV)",
            ),
            rung(
                "c1c_systematic_interactions", "skipped_by_design", None,
                "Not built. Two independent checks (shot-conditional interaction panel already 10/10 inconclusive; direct systematic interaction-expansion check found gains smaller than fold-to-fold noise, R2 0.047->0.055) both already answer the question this rung would ask. 'This is a closed decision for this leg, not deferred.'",
                f"{MODELING}/active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md #3",
            ),
            rung(
                "c1d_random_forest", "promoted", 0.9560,
                "Promoted Prompt 58, current reference E[xg|shot] component, superseding c1_lognormal_glm. Post-Prompt-86-fix held-out common_log_rmse 0.9560 (pre-fix 0.9553), corrected R2 0.0978 (pre-fix 0.0925). Hurdle-pipeline readout: RMSE 0.04416, R2=0.1693.",
                f"{MODELING}/active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md #6.5", headline_metric_name="common_log_rmse",
            ),
            rung(
                "c1e_gradient_boosting", "tried_not_promoted", 0.9672,
                "Beats c1 baseline (p=0.0402) but loses to c1d (paired-t p=0.31, Wilcoxon p=0.44, mixed 2-of-5 folds). 'c1d_random_forest remains this leg's best regression candidate after Rung 3.'",
                f"{MODELING}/active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md #5.5", headline_metric_name="common_log_rmse",
            ),
        ],
    },
    "passive_binary": {
        "label": "Passive - Binary (P(shot within 10s))",
        "reference_model": "p1e_gradient_boosting_calibrated",
        "reference_model_status": "promoted (Prompt 52), coordinate-frame fix REOPENED (metric moved outside noise, favourable direction) -- rung stands, headline numbers updated post-fix (Prompt 86).",
        "coordinate_fix_verdict": "REOPENED",
        "rungs": [
            rung(
                "p1_unweighted", "promoted", 0.1736,
                "Rung 0 baseline. Held-out PR-AUC 0.1736.",
                f"{MODELING}/passive_binary/PASSIVE_BINARY_MODEL_LADDER.md #1", headline_metric_name="PR-AUC",
            ),
            rung(
                "p1b_quadratic", "tried_not_promoted", 0.1805,
                "Passes all 3 ladder gates and DOES survive held-out test (p=0.0104) -- the headline divergence from the active leg's v1b, which did not survive held-out. Not itself promoted (promotion audit deferred to Rung 4).",
                f"{MODELING}/passive_binary/PASSIVE_BINARY_MODEL_LADDER.md #2", headline_metric_name="PR-AUC",
            ),
            rung(
                "p1c_systematic_interactions", "tried_not_promoted", 0.2013,
                "Passes all 3 gates (vs p1 p=0.0004, vs p1b p=0.00005); not itself promoted, superseded by later rungs.",
                f"{MODELING}/passive_binary/PASSIVE_BINARY_MODEL_LADDER.md #3", headline_metric_name="PR-AUC",
            ),
            rung(
                "p1d_random_forest", "mixed", 0.2078,
                "Mixed: passes gates 2 and 3 (vs Rung-0 baseline) but fails gate 1 (vs immediately-prior rung p1c, p=0.275, not significant).",
                f"{MODELING}/passive_binary/PASSIVE_BINARY_MODEL_LADDER.md #4.4", headline_metric_name="PR-AUC",
            ),
            rung(
                "p1e_gradient_boosting_calibrated", "promoted", 0.2267,
                "Promoted Prompt 52, current standing reference model. Post-Prompt-86 coordinate-fix headline: held-out PR-AUC 0.2267 (pre-fix 0.2162), ROC-AUC 0.8066 (pre-fix 0.7889). One caveat: p1c beats it on the small (n=259, 0.08% of held-out rows) defender_functional_role='unclassified' slice -- 'not disqualifying given its size.' No CV-only calibration caveat exists anywhere in the corpus for this rung; its reported numbers are held-out test numbers.",
                f"{MODELING}/passive_binary/PASSIVE_BINARY_MODEL_LADDER.md #7.6", headline_metric_name="PR-AUC",
            ),
        ],
    },
    "passive_continuous": {
        "label": "Passive - Continuous (E[xG|shot] hurdle component)",
        "reference_model": "d1_lognormal_glm",
        "reference_model_status": "Rung 0, never beaten by any later rung. Coordinate-frame fix RECONFIRMED (borderline, not significant at 0.05) -- original promotion stands, closed, not reopened.",
        "coordinate_fix_verdict": "RECONFIRMED (borderline)",
        "rungs": [
            rung(
                "d1_lognormal_glm", "promoted", 1.0374,
                "Rung 0 baseline AND still the current reference model -- 'Rung 0, never beaten' (ALL_LEGS_SUMMARY.md #2). Held-out common log RMSE 1.0374, corrected R2=0.0078.",
                f"{MODELING}/passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md #1", headline_metric_name="common_log_rmse",
            ),
            rung(
                "d1b_quadratic", "tried_not_promoted", None,
                "'An even more emphatic null result than the active leg's own Rung 1' (p=0.5013). Not promoted.",
                f"{MODELING}/passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md #2.4",
            ),
            rung(
                "d1c_systematic_interactions", "skipped_by_design", None,
                "Not built. Plain additive linear regression already scores negative mean CV R2 (-0.0042); full pairwise-interaction expansion (406 cols) scores -0.0249/-0.0230, worse than plain linear. 'This is a closed decision for this leg, not deferred.' (An independently-derived rationale from c1c's, not a copy-paste.)",
                f"{MODELING}/passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md #3",
            ),
            rung(
                "d1d_random_forest", "tried_not_promoted", 1.0322,
                "Fails significance gate vs d1 (paired-t p=0.576) -- 'the third independent null result on this leg.' Directionally better held-out common log RMSE (1.0322 vs 1.0374) and corrected R2 nearly triples (0.0078->0.0211) but statistically inconclusive.",
                f"{MODELING}/passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md #4.4", headline_metric_name="common_log_rmse",
            ),
        ],
        "note": "No gradient-boosting rung was ever attempted for this leg -- verified by reading the full ladder doc, no LightGBM/GBM mention exists.",
    },
    "active_xt": {
        "label": "Active - xT delta (threat change caused by the action)",
        "reference_model": "x1c_random_forest",
        "reference_model_status": "promoted (Prompt 75), current reference model. Phase 5 of the coordinate-frame fix is a FRESH audit on the corrected target (not a before/after diff, since the xT target itself is directly frame-dependent) -- x1c remains promoted on the corrected data.",
        "coordinate_fix_verdict": "fresh post-fix audit (target itself recomputed); x1c re-promoted",
        "rungs": [
            rung(
                "x1_two_stage_huber", "promoted", 0.11908,
                "Rung 0 baseline/comparator (two-stage Huber classifier x regressor). Pre-fix held-out R2=0.1519; post-Prompt-86-fix corrected refit R2=0.11908 (also below its own pre-fix number, since the comparator's features/target both changed).",
                f"{MODELING}/COORDINATE_FRAME_FIX_AND_REPIPELINE.md Phase 5 (Active-xT)", headline_metric_name="R2", pre_fix=False,
            ),
            rung(
                "x1b_quadratic", "tried_not_promoted", None,
                "Does not clear the bar to replace x1 as standing baseline; the rung's motivating question (tail-calibration gap) is 'unchanged to worse.'",
                f"{MODELING}/active_xt/ACTIVE_XT_MODEL_LADDER.md #2.6",
            ),
            rung(
                "x1c_random_forest", "promoted", 0.37407,
                "Promoted Prompt 75, current reference model, superseding x1_two_stage_huber. Ladder-gate pre-fix held-out R2=0.44947 (p=0.00006 vs x1). Post-Prompt-86-fix fresh audit (target itself corrected): held-out RMSE 0.04630, R2=0.37407 -- lower than the pre-fix 0.44947, but the report frames this as 'a correction, not a regression' since some of the old apparent accuracy came from the model and the frame-corrupted target sharing correlated errors.",
                f"{MODELING}/active_xt/ACTIVE_XT_MODEL_LADDER.md #5.5", headline_metric_name="R2",
            ),
            rung(
                "x1d_gradient_boosting", "mixed", None,
                "Genuine mixed result: beats x1c on RMSE/R2/MAE (small, p=0.0031) but Spearman regresses (-0.0255) and the tail-calibration gap x1c closed widens back 2-5x. 'x1c_random_forest remains the leg's standing baseline.'",
                f"{MODELING}/active_xt/ACTIVE_XT_MODEL_LADDER.md #4.8",
            ),
        ],
    },
    "passive_xt": {
        "label": "Passive - xT delta (threat change caused by the positioning)",
        "reference_model": "y1c_random_forest",
        "reference_model_status": "promoted (Prompt 78 ladder-gate, Prompt 80A full promotion), current reference model. Phase 5 fresh audit on corrected target -- re-promoted.",
        "coordinate_fix_verdict": "fresh post-fix audit (target itself recomputed); y1c re-promoted",
        "rungs": [
            rung(
                "y1_two_stage_huber", "promoted", 0.12473,
                "Rung 0 baseline/comparator. Pre-fix held-out R2=0.0612; post-Prompt-86-fix corrected refit R2=0.12473.",
                f"{MODELING}/COORDINATE_FRAME_FIX_AND_REPIPELINE.md Phase 5 (Passive-xT)", headline_metric_name="R2",
            ),
            rung(
                "y1b_quadratic", "tried_not_promoted", None,
                "Does not clear the bar -- the systematic positive prediction bias it targeted does not shrink (+0.00278 -> +0.00298, slightly worse). 'y1_two_stage_huber remains the standing baseline.'",
                f"{MODELING}/passive_xt/PASSIVE_XT_MODEL_LADDER.md #2.7",
            ),
            rung(
                "y1c_random_forest", "promoted", 0.33130,
                "Promoted (Prompt 78 ladder-gate, Prompt 80A full promotion), current reference model, superseding y1_two_stage_huber. Prediction bias essentially eliminated (+0.00278 -> -0.000018). Pre-fix held-out R2=0.49325; post-Prompt-86-fix corrected refit RMSE 0.02418, R2=0.33130.",
                f"{MODELING}/passive_xt/PASSIVE_XT_MODEL_LADDER.md #5.5", headline_metric_name="R2",
            ),
            rung(
                "y1d_gradient_boosting", "tried_not_promoted", None,
                "First rung on either xT leg to fail significance entirely vs the reference (paired-t p=0.0635, Wilcoxon p=0.125), Spearman regresses ~13%. 'y1c_random_forest remains the leg's standing baseline.'",
                f"{MODELING}/passive_xt/PASSIVE_XT_MODEL_LADDER.md #4.7",
            ),
        ],
    },
}


def main() -> None:
    verification_notes = [
        "c1b_quadratic confirmed present in the active-continuous ladder (Rung 1) -- not promoted.",
        "c1c_systematic_interactions (active-continuous) and d1c_systematic_interactions (passive-continuous) "
        "are both skipped-by-design, with two independently-derived rationales (different evidence, different "
        "numbers), not one rationale copy-pasted across legs.",
        "No gradient-boosting rung was ever attempted on the passive-continuous leg; d1_lognormal_glm (Rung 0) "
        "is never beaten.",
        "p1e_gradient_boosting_calibrated's real current headline (post Prompt-86 coordinate-frame fix): "
        "PR-AUC 0.2267, ROC-AUC 0.8066. No CV-only calibration caveat exists for this rung anywhere in the "
        "corpus (checked); its actual caveat is a small n=259 unclassified-role slice.",
        "x1 (active-xT Rung 0 comparator, corrected R2=0.11908) and x1c (active-xT Rung 2 promoted candidate, "
        "corrected R2=0.37407) are DIFFERENT rungs/metrics from the same Phase-5 refit table -- not two "
        "readings of the same rung. Same x1-vs-x1c pattern repeats as y1 (R2=0.12473) vs y1c (R2=0.33130) "
        "on the passive-xT leg.",
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/dashboard/export_model_ladders.py",
        "verification_notes": verification_notes,
        "legs": LEGS,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    n_rungs = sum(len(leg["rungs"]) for leg in LEGS.values())
    print(f"{len(LEGS)} legs, {n_rungs} rungs total")


if __name__ == "__main__":
    main()
