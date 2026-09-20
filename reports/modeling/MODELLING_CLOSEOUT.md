# Modelling Phase Closeout

Project-wide synthesis, sitting one level above the four legs' own baseline/ladder/closeout docs --
the modelling-phase equivalent of `reports/eda/eda-closeout.html` for the EDA phase. This document
does not replace or duplicate any per-leg doc (in particular
[`active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.md`](active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.md),
which remains the full promotion-audit record for that one leg) -- it compares across all four legs
and records the discipline that applied to more than one of them, linking out to per-leg detail
rather than re-deriving it.

## 1. What this closes out

Modelling is complete across all four legs: two binary `P(shot)` classifiers (active, passive) and
two continuous `E[xg | shot]` regressors (active, passive), combined per leg-pair via the hurdle
architecture -- `predicted_xg = P(shot) * E[xg | shot]`. As of this closeout, **no further
model-ladder rungs are planned for any of the four legs** unless new evidence specifically warrants
revisiting one. This is a stopping decision, not a pause: the passive-continuous leg's own Rungs 1-2
(Prompts 61-62) tested three structurally different approaches (a hand-built quadratic term, a
systematic interaction expansion, a random forest) against its Rung 0 baseline and none cleared
significance -- a convergent, three-angle null result that is the clearest evidence, project-wide,
that diminishing returns had been reached on the weakest leg, not an isolated result specific to it
alone.

## 2. Final state, all four legs

| Leg | Standing/promoted model | Artifact | Headline held-out metric(s) | Ladder rungs before settling |
|---|---|---|---|---|
| Active-binary (`target_future_shot_10s`, active features) | `v1e_gradient_boosting_calibrated` | `outputs/models/classification/v1e_gradient_boosting_calibrated.joblib` | PR-AUC 0.4282, ROC-AUC 0.8364, ECE 0.0096 | 4 rungs beyond Rung 0 (quadratic, interactions, random forest, gradient boosting) |
| Passive-binary (`target_future_shot_10s`, passive features) | `p1e_gradient_boosting_calibrated` | `outputs/models/classification/p1e_gradient_boosting_calibrated.joblib` | PR-AUC 0.2162, ROC-AUC 0.7889, ECE 0.0044 | 4 rungs beyond Rung 0 (quadratic, interactions, random forest, gradient boosting) |
| Active-continuous (`E[xg\|shot]`, active features) | `c1d_random_forest` | `outputs/models/regression/c1d_random_forest.joblib` | Hurdle pipeline (vs `v1e_gradient_boosting_calibrated`'s P(shot)): RMSE 0.04416, MAE 0.01107, R² 0.1693 | 2 rungs beyond Rung 0 (quadratic null, interactions skipped on evidence, random forest promoted); a 3rd rung (gradient boosting) tried and did not beat it |
| Passive-continuous (`E[xg\|shot]`, passive features) | `d1_lognormal_glm` (Rung 0, never beaten) | `outputs/models/regression/d1_lognormal_glm.joblib` | Hurdle pipeline (vs `p1e_gradient_boosting_calibrated`'s P(shot)): RMSE 0.03726, MAE 0.00980, R² 0.0426 | 0 rungs beyond Rung 0 survived; 2 rungs attempted (quadratic, random forest) plus 1 skipped on evidence (interactions), none beat it |

Numbers above confirmed directly against
`outputs/models/comparisons/active_binary_baseline_held_out_test_readout.csv`,
`outputs/models/comparisons/passive_binary_baseline_held_out_test_readout.csv`,
`outputs/models/validation/hurdle_pipeline_readout_c1d.json`, and
`outputs/models/validation/hurdle_pipeline_readout_passive.json` while writing this document, not
carried forward from memory.

## 3. Per-leg summary

**Active-binary -- `v1e_gradient_boosting_calibrated`.** Promoted after a slice-level calibration
check overrode the raw variant's better-looking aggregate numbers: raw `v1e` had a strong aggregate
ECE but was found 6-19x worse than `v1c` on specific `phase_label` slices, a trap the calibrated
variant does not fall into. Full detail:
[`active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.html`](active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.html).

**Passive-binary -- `p1e_gradient_boosting_calibrated`.** Same shape of promotion as the active leg
(raw gradient boosting, then isotonic-calibrated), with one leg-specific caveat carried at promotion
time: the `defender_functional_role="unclassified"` slice (n=259 held-out) is a real but
non-disqualifying weak spot, flagged explicitly rather than dropped. No separate closeout document
exists for this leg -- the ladder doc itself carries the promotion-audit detail. Full detail:
[`passive_binary/PASSIVE_BINARY_MODEL_LADDER.html`](passive_binary/PASSIVE_BINARY_MODEL_LADDER.html).

**Active-continuous -- `c1d_random_forest`.** Replaced the original `c1_lognormal_glm` GLM after a
full promotion audit (tournament-stratified check, error-slice comparison, feature-shape sanity
check, and a hurdle-pipeline re-readout showing a real, if modest, end-to-end improvement over the
GLM-based numbers already on record) -- the same depth of scrutiny the binary legs' own promotions
used, adapted for this leg's small sample and its hurdle-pipeline (not standalone) role. Full detail:
[`active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.html`](active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.html).

**Passive-continuous -- `d1_lognormal_glm` stands.** Rung 0 was never beaten through three
independent Rung 1/2 attempts (a quadratic term, a random forest, plus a systematic-interactions
rung skipped on its own decisive evidence). Attributed to this leg's row grain: each row describes
one individual defender's geometry relative to a single event, not an aggregated picture of the
whole defensive shape the way the active leg's features are -- plausibly less individual leverage
over eventual shot quality than the attacking side's own aggregate spatial features carry. Full
detail:
[`passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.html`](passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.html).

## 4. Cross-cutting discipline established across legs

- **Ranking and calibration are always reported as separate verdicts, never collapsed into one.**
  Established from the active-binary leg's own `v1e` promotion, where the raw and calibrated
  variants split on exactly this axis (raw: better ranking, worse slice-level calibration); reused
  on every subsequent rung on every leg, including the continuous legs' own naive-vs-corrected
  back-transform reporting.
- **The common-scale log-RMSE fix for head-to-head continuous-model gates.** Originally a
  log1p-vs-log scale mismatch between `c0_dummy` and `c1_lognormal_glm` (found and fixed in Prompt
  54), which would otherwise have made a correctly-behaving model look catastrophically worse purely
  from the transform, not the fit. Carried through every rung on both continuous legs.
- **"Skip a rung only with a cited, real-data diligence check, never assumed."** Applied on both
  continuous legs' interactions rungs: the active leg's `c1c` was skipped on a smaller-than-noise
  gain from a real Ridge/PolynomialFeatures check (Prompt 56); the passive leg's `d1c` was skipped
  on a more decisive rejection -- the interaction expansion scored *worse* than the plain linear
  model, not just a smaller gain than noise (Prompt 62).
- **A promotion audit is always deeper than a ladder gate.** Passing a rung's own significance test
  only answers "is this rung's isolated change real?", not "should this replace what everything
  downstream builds on?" -- established in the active-binary leg's `v1c`/`v1e` promotions (Prompts
  43/46) and reused, adapted for a hurdle-pipeline component rather than a standalone classifier, for
  the active-continuous leg's `c1d` promotion (Prompt 58).
- **Locked feature parquets, canonical splits, and promoted models are frozen artifacts; data-quality
  fixes happen at the preprocessing/model-input layer, never by rewriting locked source data.**
  Established by the `defender_functional_role_unclassified` fix (Prompt 60): the shared
  `DesignMatrixBuilder`'s fitted category set was changed, `passive_defense.parquet` itself was not.
- **Effective sample size (unique events, not rows) reported alongside row counts wherever row-level
  duplication exists.** The passive leg's defender-slot grain (~8.9 rows/event) means row counts
  overstate independence; every significance test and CV-overfitting diagnostic on both passive legs
  reports the event count explicitly rather than letting a large row count imply a false sense of
  statistical power.

## 5. Known open items, carried forward from this closeout

Checked directly against the current repo state while writing this document (not assumed from
memory or an earlier prompt's status):

- **Top open item: `p1e_gradient_boosting_calibrated` has not been checked for the specific
  `defender_functional_role_unclassified` artifact mechanism Prompt 60 found in the linear passive
  models.** This needs to be stated precisely, because it is *not* simply "unchecked": the
  `unclassified` slice's **error-slice PR-AUC and calibration gap** *were* examined at promotion time
  (Prompts 51/52, `error_analysis_p1e_calibrated.json`, `slice_level_calibration_p1e_calibrated_vs_p1c_vs_p1.json`)
  and found to be a real but non-disqualifying caveat on a thin (n=259 held-out) slice. What has
  *not* been checked is the mechanism Prompt 60 actually diagnosed: whether GBM shows an analogous
  overfit-to-a-near-singleton pattern (the way `d1_lognormal_glm`'s and `p1_unweighted`/`p2_weighted`/
  `p3_weighted_interactions`' linear coefficients did), and critically, **`p1e_gradient_boosting_calibrated`'s
  own scoring design matrix (`train_p1d_random_forest.py`'s `RFDesignMatrixBuilder`, used to reload
  and score it wherever it is reused, including both continuous legs' hurdle pipelines) does not
  include the Prompt 60 fix at all** -- confirmed directly: that class still fits its `OneHotEncoder`
  on every observed category, `"unclassified"` included, unlike the corrected `DesignMatrixBuilder`
  (`train_passive_binary_baseline.py`) that `d1`/`d1b`/`d1d` now use. `p1e_gradient_boosting_calibrated`
  is a promoted, in-use classifier whose sensitivity to a known data-quality artifact has never
  actually been verified end to end, even though GBMs are generally less prone to this failure mode
  than linear models. Flagged as a recommendation for a future prompt to check, not acted on here.
- **Duplicate `v1_unweighted`/`v1_unweighted_final_fit` artifact pair.** Confirmed still present on
  disk: `outputs/models/classification/v1_unweighted.joblib`/`.json` and
  `v1_unweighted_final_fit.joblib`/`.json` both exist (checked directly, file timestamps unchanged
  since the original Prompt 36/37 runs). Both remain gitignored, reproducible outputs -- harmless,
  not a correctness issue, still not cleaned up pending Varun's explicit OK, exactly as
  `ACTIVE_BINARY_MODELLING_CLOSEOUT.md` section 6 originally flagged.
- **`configs/models.yaml`/`train_models.py`'s `max_rows` cap.** Confirmed unchanged: `run_training`'s
  `max_rows: int | None = None` parameter and the `--max-rows` CLI flag (`src/dax/models/training.py`)
  still exist, still default to no cap, with no commit touching this default since before Prompt 40.
  Still unfixed, deliberately, for the same reason already on record -- the same config drives
  variants that run at real scale and feed the two-part xG pipeline, and the cap is opt-in, not a
  standing limitation of current runs.
- **No new limitation surfaced while writing this document.** The three items above are the same
  ones already on record; nothing additional was found during this closeout's own review that hasn't
  already been stated plainly as a limitation somewhere in the per-leg docs.

## 6. What was not built, and why that's a deliberate stop, not a gap

No fifth rung was attempted on either binary leg past its own promoted gradient-boosting-calibrated
variant, and no further rung was attempted on either continuous leg past what is now promoted or
standing. The passive-continuous leg's three independent nulls (Prompts 61-62: a quadratic term, a
systematic-interactions expansion rejected on its own real-data evidence, and a random forest) are
the clearest single piece of project-wide evidence that diminishing returns had been reached --
not an isolated result specific to that leg's own weak per-defender signal, but the sharpest instance
of a pattern already visible earlier: the active-continuous leg's own gradient-boosting rung (Rung 3)
also failed to beat its Rung 2 random forest, and the active-binary/passive-binary ladders both
stopped at their 4th rung once gradient boosting plus calibration cleared every gate checked. Four
legs, each independently run to the point where further rungs stopped clearing real bars, is the
basis for closing modelling out project-wide now rather than continuing to search for marginal gains.
