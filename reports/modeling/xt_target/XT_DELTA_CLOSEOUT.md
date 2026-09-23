# xT-Delta Family Closeout

Two-leg synthesis, sitting one level above the active-xT and passive-xT legs' own
baseline/ladder/closeout docs -- the xT-delta-family equivalent of
[`../MODELLING_CLOSEOUT.md`](../MODELLING_CLOSEOUT.md)'s own project-wide synthesis for the four
original DAx legs. This document does not replace or duplicate either per-leg doc (in particular
[`../active_xt/ACTIVE_XT_MODEL_LADDER.md`](../active_xt/ACTIVE_XT_MODEL_LADDER.md) and
[`../passive_xt/PASSIVE_XT_MODEL_LADDER.md`](../passive_xt/PASSIVE_XT_MODEL_LADDER.md), which
remain the full ladder/promotion-audit records for each leg) -- it compares across the two legs and
records the discipline that applied to both of them, linking out to per-leg detail rather than
re-deriving it.

## 1. What this closes out

Both legs of the xT-delta target family -- active (`target_xt_delta_v2`) and passive
(`target_xt_delta_passive`) -- now have structurally complete ladders and a promoted reference
model each:

- **Active**: `x0_dummy` -> `x1_two_stage_huber` (Rung 0) -> `x1b_quadratic` (Rung 1, did not
  clear the bar) -> `x1c_random_forest` (Rung 2, cleared the bar, promoted as reference model,
  Prompt 75) -> `x1d_gradient_boosting` (Rung 3, mixed result against `x1c`, not promoted,
  Prompt 74).
- **Passive**: `y0_dummy` -> `y1_two_stage_huber` (Rung 0) -> `y1b_quadratic` (Rung 1, did not
  clear the bar) -> `y1c_random_forest` (Rung 2, cleared the bar, promoted as reference model,
  Prompt 80 part A) -> `y1d_gradient_boosting` (Rung 3, mixed result against `y1c`, not promoted,
  Prompt 79).

As of this closeout, no further model-ladder rungs are planned for either leg unless new evidence
specifically warrants revisiting one. This is a reporting closeout, not a new modelling decision --
every number below was confirmed directly against the artifacts and reports cited, not carried
forward from memory, while writing this document.

## 2. Final state, both legs

| Leg | Reference model | Artifact | Held-out RMSE | Held-out R&sup2; | Held-out Spearman | Rungs beyond Rung 0 |
|---|---|---|---|---|---|---|
| Active-xT (`target_xt_delta_v2`) | `x1c_random_forest` | `outputs/models/regression/x1c_random_forest_{classifier,regressor}.joblib` | 0.04277 | 0.44947 | 0.5947 | 3 (`x1b`, `x1c`, `x1d`) |
| Passive-xT (`target_xt_delta_passive`) | `y1c_random_forest` | `outputs/models/regression/y1c_random_forest_{classifier,regressor}.joblib` | 0.02489 | 0.49325 | 0.3154 | 3 (`y1b`, `y1c`, `y1d`) |

Both reference-model rows are the promotion-audit's own full-pipeline re-run numbers (active:
`full_pipeline_readout_x1c.json`, Prompt 75; passive: `full_pipeline_readout_y1c.json`, Prompt 80
part A) -- an independent, freshly-scored, predict-only combined-pipeline readout on each leg's
own held-out test set, not the Rung-2 ladder-gate's own CV number.

**Improvement over each leg's own Rung-0 baseline:**

| Leg | Rung-0 RMSE | Reference RMSE | RMSE reduction | Rung-0 R&sup2; | Reference R&sup2; | R&sup2; change |
|---|---|---|---|---|---|---|
| Active-xT (`x1_two_stage_huber` -> `x1c_random_forest`) | 0.05309 | 0.04277 | -19.4% | 0.1519 | 0.4495 | +0.298 (~3.0x) |
| Passive-xT (`y1_two_stage_huber` -> `y1c_random_forest`) | 0.03388 | 0.02489 | -26.5% | 0.0612 | 0.4933 | +0.432 (~8.1x) |

Passive's own reference model shows both the larger relative RMSE reduction and the larger
absolute R&sup2; gain of the two legs -- confirmed directly by both legs' own promotion-audit
pipeline re-runs (section 5.4 of each ladder doc), not an artefact of CV-only numbers.

## 3. The real structural parallel: both legs landed on RF at Rung 2, both saw GBM fail at Rung 3 -- for different reasons

Both legs' ladders reached the identical shape: a features-only rung (`x1b`/`y1b`, quadratic and
interaction terms on the same model family) that did not clear the bar, then a model-family change
to random forest (`x1c`/`y1c`) that did, then a further model-family change to gradient boosting
(`x1d`/`y1d`) that also did not clear the bar against the now-promoted RF baseline. **That shape is
real and worth stating plainly. The reason GBM failed to clear RF is not the same on both legs, and
forcing it into one narrative would misstate at least one leg:**

- **Active-xT**: `x1d_gradient_boosting` won on the headline accuracy metrics (RMSE, MAE) but gave
  back a real share of `x1c`'s own tail-calibration win -- the specific diagnostic this leg's own
  ladder has tracked since Rung 0 (the two tail bins under-predicting magnitude). `x1c` had closed
  that gap by 10-35x; `x1d` widened it back out by roughly 2-5x (still far better than `x1b`/`x1`,
  but a real regression against the current reference). Spearman also regressed. The promotion
  call (section 4.8 of `ACTIVE_XT_MODEL_LADDER.md`) explicitly weighed a small, real RMSE win
  against a real regression on the two diagnostics this leg treats as first-class evidence, not a
  footnote, and did not promote.
- **Passive-xT**: `y1d_gradient_boosting` did not even clear the significance bar against `y1c` in
  the first place -- paired-t p=0.0635, Wilcoxon p=0.125, the first rung comparison on either leg's
  ladder that fails to clear significance at all. RMSE/R&sup2; moved marginally in `y1d`'s favor
  (not a "real" win the way active's `x1d` had on those same two metrics), and held-out Spearman
  regressed meaningfully (0.3154 -> 0.2732, ~13% relative) -- a rank-correlation regression with no
  offsetting significant accuracy gain to weigh it against. The prediction-bias gate `y1c` closed
  stayed closed either way (section 4.5 of `PASSIVE_XT_MODEL_LADDER.md`), so that check did not
  distinguish the two variants on this leg -- unlike active-xT, where calibration was the decisive
  axis.

**Same shape, different actual reason on each leg**: active's `x1d` lost on calibration/rank
evidence despite winning on raw accuracy; passive's `y1d` did not even establish a significant
accuracy win to begin with. Reading both as "GBM regressed calibration" would be accurate for
active and wrong for passive; reading both as "GBM wasn't significant" would be wrong for active
(it was significant on RMSE) and accurate for passive. Each leg's own report states its own reason;
this closeout does not average them into one.

## 4. The real asymmetry: both legs' calibration problems turned out to be model-family-shaped, not feature-shaped

This is the strongest piece of cross-leg evidence in the whole xT-delta family, and it appears on
both legs independently, via different motivating diagnostics:

- **Active-xT**: Rung 0's own calibration table showed a *tail* under-prediction gap (the two
  extreme bins under-predicting magnitude by 3x or more). `x1b_quadratic` (Rung 1 -- more terms,
  same `HuberRegressor` model) added curvature-capturing terms specifically aimed at that gap.
  **The tail-calibration gap that motivated the entire rung did not narrow -- it is unchanged to
  worse** (bin 0's gap actually widens slightly). It took `x1c_random_forest` (Rung 2 -- a genuine
  model-family change, same features) to close that gap by 10-35x. The Rung-1 report's own
  conclusion: the tail-calibration problem is a property of `HuberRegressor`'s own loss function
  (which caps large residuals by design), not a curvature-shaped problem more terms in the same
  loss function could fix.
- **Passive-xT**: Rung 0's own calibration table showed a different pattern -- a *systematic
  positive bias across all 5 bins*, not a tail-specific gap (`y1`'s own held-out
  `prediction_bias`: +0.00278). `y1b_quadratic` (Rung 1 -- more terms, same `HuberRegressor` model)
  added curvature-capturing terms specifically aimed at that bias. **The bias did not shrink -- it
  got marginally worse** (+0.00278 -> +0.00298). It took `y1c_random_forest` (Rung 2 -- the same
  kind of model-family change) to close it almost to zero (-0.000018, essentially eliminated).

**Both legs ran the identical experiment (more features, same model, aimed at a calibration
problem) and got the identical null result, then both ran the identical follow-up experiment
(same features, different model family) and got the identical positive result** -- on two
different calibration diagnostics (a tail gap on one leg, a whole-distribution bias on the other),
on two independently-built feature sets, with no shared code path between the two legs' `x1b`/`y1b`
scripts beyond the general two-stage architecture. This convergence is the strongest evidence
anywhere in this project that `HuberRegressor`'s calibration limitations on this target family are
a property of the model family's own loss function, not something addressable by adding more terms
to the same linear-in-parameters model. A single leg showing this pattern could be leg-specific
noise; two independently-motivated legs showing the identical experiment/null/positive-result
sequence is not.

## 5. Where the two reference models draw signal

Top features by Gini importance, regression head, aggregated back to each leg's own original
locked feature set (see each ladder doc's own section 5.3 for the aggregation method):

| Rank | Active-xT (`x1c_random_forest`, aggregated by original feature) | Passive-xT (`y1c_random_forest`, aggregated by original feature) |
|---|---|---|
| 1 | `phase_label_prev_event` (categorical, 35.5%) | `ball_x` (33.9%) |
| 2 | `event_type` (categorical, 14.1%) | `ball_y` (31.5%) |
| 3 | `distance_to_attacking_box` (9.7%) | `on_ball_event_type` (categorical, 29.0%) |
| 4 | `attacking_goal_centrality` (8.8%) | `top_option_3_threat_score` (0.8%) |
| 5 | `nearest_attacker_distance` (8.2%) | `defender_x` (0.8%) |

**Little overlap in kind, and a real divergence in concentration.** Active's top signal is
behavioral/contextual -- what phase the previous event happened in, what kind of event it was,
how far the defensive action is from its own box, how central the attacking goal angle is -- a mix
of two categorical features and three relative-geometry features, with the top 5 combining for
~76% and no long tail of near-zero features. Passive's top signal is dominated by two raw location
columns and one event-type category; the **top 3 alone account for ~94% of total aggregated Gini
importance** (33.9% + 31.5% + 29.0%), an extremely concentrated result with no equivalent on the
active leg -- everything past rank 3 on the passive side is under 1%.

**This concentration is not incidental, and passive's own report already flagged why -- restated
here, not re-derived**: `target_xt_delta_passive`'s own construction bakes in a
`ball_x`/`ball_y`-shaped signal that `target_xt_delta_v2` does not share. `xt_before` on the
passive leg is a direct grid lookup, `xt_before = xT(ball_x, ball_y)`, off two columns that are
*also* locked PASSIVE modelling features; on the active leg, `xt_before` is looked up at the
*previous* event's location, not a feature-table column at all. `reports/analysis/xt_target/PASSIVE_LEAKAGE_AUDIT.json`'s
own Part E' measured `ball_x` at r=+0.52 vs `xt_before` and r=+0.25 vs the target itself (`ball_y`
correlates far more weakly, r=-0.004, consistent with the xT grid varying more along pitch length
than width). `y1c`'s accuracy gain is real (confirmed independently by the Part-A promotion audit,
section 5 of `PASSIVE_XT_MODEL_LADDER.md`), but a meaningful share of what the model leans on is
this already-documented construction-linked signal, not entirely fresh predictive structure. Active
carries no equivalent flagged coupling in its own top features -- its own promotion audit (Prompt
75) found no comparable red flag, a genuine difference between the two legs' feature-importance
profiles, not merely a different presentation of the same caveat.

## 6. What's not done

- **No deeper promotion audit beyond what already ran.** `x1c_random_forest`'s own audit (Prompt
  75) and `y1c_random_forest`'s own audit (this prompt's Part A) are both now on record; neither
  `x1d_gradient_boosting` nor `y1d_gradient_boosting` gets a promotion audit, since neither was
  promoted -- a promotion audit answers "should this replace the reference model," which does not
  apply to a rung that already failed its own ladder gate.
- **No project-wide combined-target comparison.** Whether the xT-delta family's own reference
  models (`x1c_random_forest`, `y1c_random_forest`) should be compared against, combined with, or
  read alongside the four original DAx legs' own reference models (`v1e_gradient_boosting_calibrated`,
  `p1e_gradient_boosting_calibrated`, `c1d_random_forest`, `d1_lognormal_glm` -- see
  `../MODELLING_CLOSEOUT.md`) is not addressed here. That is a separate, later step if the user
  asks for it, not something this closeout resolves.
- **No new ladder rungs.** No `x1e`, no `y1e`. Both legs' ladders are considered structurally
  complete as of this closeout, in the same sense `../MODELLING_CLOSEOUT.md` closed the four
  original legs -- a stopping decision based on the evidence gathered, not a hard rule against ever
  revisiting either leg.
