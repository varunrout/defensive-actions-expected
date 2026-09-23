# All-Legs Cross-Comparison Summary

Project-wide synthesis across all six modelling legs, sitting one level above the two existing
pairwise closeout docs -- [`MODELLING_CLOSEOUT.md`](MODELLING_CLOSEOUT.md) (the four original DAx
legs) and [`xt_target/XT_DELTA_CLOSEOUT.md`](xt_target/XT_DELTA_CLOSEOUT.md) (the two xT legs).
Neither of those puts all six side by side; this document does that, and only that. **This is a
synthesis document, not new analysis.** Every number below is pulled from an existing,
already-verified source and cited inline; nothing here is recomputed, re-derived, or re-audited.
No modelling was re-run, no promotion was re-audited, and no standing reference model changes as a
result of this document.

## 1. What this closes out

Six legs, spanning two target-family shapes: **binary presence** (active-binary
`target_future_shot_10s`, passive-binary `target_future_shot_10s` on passive features) and
**continuous value**, where the continuous shape itself splits into two sub-families with
different architectures -- a **hurdle pipeline** (active-continuous `E[xg|shot]`, passive-continuous
`E[xg|shot]` on passive features, each combined with its own leg-pair's binary classifier via
`P(shot) x E[xg|shot]`) and a **two-stage xT-delta** pipeline (active-xT `target_xt_delta_v2`,
passive-xT `target_xt_delta_passive`, each combined internally via
`P(nonzero) x E[delta|nonzero]`, with no external classifier dependency). All six now stand at a
documented reference model -- four promoted through a full multi-check audit, one (passive-xT)
promoted this same way as of Prompt 80, and one (passive-continuous) still at its own Rung-0
baseline because nothing built on top of it ever cleared significance.

## 2. The six reference models, side by side

| Leg | Target type | Reference model | Model family | Held-out headline metric(s) | Audit status |
|---|---|---|---|---|---|
| Active-binary | Binary | `v1e_gradient_boosting_calibrated` | Gradient boosting (LightGBM) + isotonic calibration | PR-AUC 0.4282, ROC-AUC 0.8364, Brier 0.0522, ECE 0.0096 (source: `active_binary_baseline_held_out_test_readout.csv`, `v1e_gradient_boosting_calibrated` row) | Yes -- 3-check audit (tournament-stratified, error-slice, behavioral-sanity), Prompt 46 (source: `ACTIVE_BINARY_MODELLING_CLOSEOUT.md` §8) |
| Passive-binary | Binary | `p1e_gradient_boosting_calibrated` | Gradient boosting (LightGBM) + isotonic calibration | PR-AUC 0.2162, ROC-AUC 0.7889, Brier 0.0474, ECE 0.0044 (source: `PASSIVE_BINARY_MODEL_LADDER.md` §7 / `MODELLING_CLOSEOUT.md` §2; the underlying `passive_binary_baseline_held_out_test_readout.csv` row shows PR-AUC 0.210773 for this variant -- a 3rd-decimal discrepancy between the CSV and the doc-reported headline that this synthesis does not attempt to resolve, out of scope for a read-only document) | Yes -- 5-check audit (tournament-stratified, error-slice, slice-level calibration, behavioral-sanity, cluster-bootstrap vs `p1c`), Prompt 52 (source: `PASSIVE_BINARY_MODEL_LADDER.md` §7) |
| Active-continuous | Continuous (hurdle pipeline) | `c1d_random_forest` | Random forest | Hurdle pipeline (vs `v1e`'s P(shot)): RMSE 0.04416, MAE 0.01107, R&sup2; 0.1693 (source: `outputs/models/validation/hurdle_pipeline_readout_c1d.json`, `hurdle_pipeline_metrics`) | Yes -- 4-check audit incl. hurdle-pipeline re-run as the decisive check, Prompt 58 (source: `ACTIVE_CONTINUOUS_MODEL_LADDER.md` §6) |
| Passive-continuous | Continuous (hurdle pipeline) | `d1_lognormal_glm` (**Rung 0, never beaten**) | Log-normal GLM (`LinearRegression` on `log(xg)`) | Hurdle pipeline (vs `p1e`'s P(shot)): RMSE 0.03726, MAE 0.00980, R&sup2; 0.0426 (source: `outputs/models/validation/hurdle_pipeline_readout_passive.json`, `hurdle_pipeline_metrics`) | No -- nothing ever cleared significance against Rung 0 to trigger one; `d1d_random_forest` was tried (Rung 2) and failed significance, p=0.576 (source: `PASSIVE_CONTINUOUS_MODEL_LADDER.md` §4.4) |
| Active-xT | Two-stage continuous | `x1c_random_forest` | Random forest (two-stage: classifier + regressor) | Held-out RMSE 0.04277, R&sup2; 0.44947, Spearman 0.5947 (source: `outputs/models/validation/full_pipeline_readout_x1c.json`) | Yes -- 4-check audit incl. full-pipeline re-run, Prompt 75 (source: `ACTIVE_XT_MODEL_LADDER.md` §5) |
| Passive-xT | Two-stage continuous | `y1c_random_forest` | Random forest (two-stage: classifier + regressor) | Held-out RMSE 0.02489, R&sup2; 0.49325, Spearman 0.3154 (source: `outputs/models/validation/full_pipeline_readout_y1c.json`) | Yes -- 4-check audit incl. full-pipeline re-run, Prompt 80 part A (source: `PASSIVE_XT_MODEL_LADDER.md` §5) |

**Note on this table's own premise, stated plainly rather than silently corrected**: this prompt's
own context table listed the passive-continuous reference model as `d1d_random_forest`. That is
not accurate as of the current repository state. `d1d_random_forest` (Rung 2, built and tested) did
**not** clear its own ladder-gate significance test against `d1_lognormal_glm` (paired-t p=0.576,
the third independent null result on this leg -- alongside a quadratic Rung 1 and a skipped
interactions rung, source: `PASSIVE_CONTINUOUS_MODEL_LADDER.md` §4.4), and
`MODELLING_CLOSEOUT.md` §2/§3 (itself already current, not stale) correctly documents
`d1_lognormal_glm` as standing, never beaten. The table above reports the actual current state, not
the prompt's premise.

## 3. The one real cross-leg pattern -- and where it does and doesn't hold

Random forest is the reference model for **3 of 6 legs** (`c1d_random_forest`,
`x1c_random_forest`, `y1c_random_forest`) -- not 4, per the correction in section 2 above. On the
remaining 3 legs, random forest was still **tried** on every single one, and lost out for three
different, leg-specific reasons, not because it was never attempted:

- **Active-binary**: `v1d_random_forest` (Rung 3) beat `v1c` on PR-AUC (+0.0338, held-out
  0.4085 vs 0.3747, `significance_v1d_vs_v1c.json` p=0.0015) but with held-out ECE ~1.8x worse
  (0.0183 vs 0.0103). It was explicitly framed from the start as **"a diagnostic rung, not a
  promotion candidate in its own right"** (source: `ACTIVE_BINARY_MODEL_LADDER.md` §4) -- used to
  size how much accuracy headroom existed before committing to gradient boosting, not run as a
  genuine promotion attempt. No promotion audit was ever run for `v1d` itself.
- **Passive-binary**: `p1d_random_forest` (Rung 3) failed its own ladder-gate significance test
  against `p1c` (mean diff +0.0025 PR-AUC, p=0.275, "NOT significant" -- source:
  `PASSIVE_BINARY_MODEL_LADDER.md` §4) -- an ordinary ladder-rung loss, not a diagnostic-only
  framing the way active-binary's was.
- **Passive-continuous**: `d1d_random_forest` (Rung 2) also failed its own ladder-gate
  significance test against `d1_lognormal_glm` (p=0.576, source: `PASSIVE_CONTINUOUS_MODEL_LADDER.md`
  §4.4), despite being directionally ahead on 4 of 5 CV folds and nearly tripling corrected R&sup2;
  (0.0078 -> 0.0211) -- a real but statistically unreliable signal at this leg's own effective
  sample size, not evidence RF was untested here.

**Stated only as far as the evidence supports**: random forest is not "the winning model family
everywhere" -- it is the model family that won 3 of 6 head-to-head tests it was actually put
through, with the two binary legs' own non-adoption traceable to calibration cost (active) and a
failed significance test (passive), and the third continuous leg's own non-adoption traceable to
the same failed-significance pattern repeating on a smaller, noisier sample. The pattern worth
reporting is not "RF wins," but **"RF was tried everywhere, and won exactly where the leg's own
target had enough signal-to-noise for a tree ensemble's flexibility to pay for itself on held-out
data -- the three continuous/two-stage legs with the largest row counts, not the two binary legs or
the smallest continuous leg."**

## 4. Active vs passive, leg by leg -- is there a consistent direction?

| Target family | Active leg's own improvement over its Rung 0 | Passive leg's own improvement over its Rung 0 | Larger gain |
|---|---|---|---|
| Binary (headline: PR-AUC) | 0.3725 -> 0.4282, &Delta;+0.0557 (+14.9% relative) | 0.1736 -> 0.2162, &Delta;+0.0426 (+24.5% relative) | **Ambiguous, depends on framing** -- active's absolute gain is larger, passive's relative gain is larger |
| Continuous/hurdle (headline: hurdle R&sup2;) | 0.1426 -> 0.1693, &Delta;+0.0267 (+18.7% relative) | 0.0426 -> 0.0426, &Delta;+0.0000 (0%, nothing ever beat Rung 0) | **Active, decisively** -- passive-continuous has zero improvement to compare |
| xT-delta (headline: held-out R&sup2;) | 0.1519 -> 0.4495, &Delta;+0.2976 (~3.0x) | 0.0612 -> 0.4933, &Delta;+0.4321 (~8.1x) | **Passive, decisively** (source: `XT_DELTA_CLOSEOUT.md` §2) |

Sources: active-binary Rung 0/reference PR-AUC from `ACTIVE_BINARY_BASELINE_SUMMARY.md` §5 and
`active_binary_baseline_held_out_test_readout.csv`; passive-binary from
`PASSIVE_BINARY_BASELINE_SUMMARY.md` §7 and the doc-reported headline in section 2 above;
active-continuous hurdle R&sup2; from `hurdle_pipeline_readout_c1d.json`'s own
`c1_based_hurdle_metrics_for_comparison` vs `hurdle_pipeline_metrics` fields; passive-continuous
from section 2 above (Rung 0 is the reference model, so the comparison is to itself); xT-delta
figures from `XT_DELTA_CLOSEOUT.md` §2.

**No, there is not a consistent direction across all three families.** The xT-delta family shows
passive with the clearly larger gain (per `XT_DELTA_CLOSEOUT.md` §2), but that direction does not
hold for the other two families: the hurdle/continuous family shows the opposite pattern, active
improving substantially while passive improved not at all (its own reference model is still Rung
0), and the binary family is genuinely ambiguous -- which leg "wins" depends on whether the
question is asked in absolute PR-AUC points or relative percentage terms. **The passive-xT result
is specific to that family, not part of a project-wide "passive improves more" pattern** -- if
anything, the continuous/hurdle family shows the strongest pattern in the *opposite* direction, and
that is the more statistically striking of the two (a leg with literally zero validated improvement
across three independent modelling attempts, versus a leg with a large one).

## 5. What's genuinely not comparable, stated plainly

**This document does not rank all six legs on one axis, because there is no single metric that
means the same thing across all of them.** Binary-leg metrics (PR-AUC, ROC-AUC, Brier score, ECE)
answer "how well does this model separate and calibrate a yes/no outcome" -- they have no
continuous-leg equivalent. Continuous-leg metrics (RMSE, R&sup2;, Spearman) answer "how well does
this model predict a numeric value's magnitude and ordering" -- they have no binary-leg equivalent.
A PR-AUC of 0.43 and an R&sup2; of 0.45 are not the same kind of number and saying one leg
"performs better" than the other by comparing them would be a category error, not a finding. Even
within the continuous legs, the hurdle-pipeline legs' R&sup2; (active-continuous 0.169,
passive-continuous 0.043) and the two-stage xT legs' R&sup2; (active-xT 0.449, passive-xT 0.493)
are computed over different target distributions with different zero-inflation rates and different
downstream architectures (external classifier dependency vs. fully internal), so even a same-units
comparison across those four legs should be read with that caveat, not as a clean four-way ranking.
Section 2's table is ordered by leg pairing (the order the six legs were built in), not by any
performance ranking, and should not be read as one.

## 6. Links

**Cross-leg closeouts**: [`MODELLING_CLOSEOUT.md`](MODELLING_CLOSEOUT.md) (the four original DAx
legs, project-wide), [`xt_target/XT_DELTA_CLOSEOUT.md`](xt_target/XT_DELTA_CLOSEOUT.md) (the two
xT legs).

**Per-leg baseline/ladder/promotion docs**:
- Active-binary: [`active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md`](active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md),
  [`active_binary/ACTIVE_BINARY_MODEL_LADDER.md`](active_binary/ACTIVE_BINARY_MODEL_LADDER.md),
  [`active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.md`](active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.md)
- Passive-binary: [`passive_binary/PASSIVE_BINARY_BASELINE_SUMMARY.md`](passive_binary/PASSIVE_BINARY_BASELINE_SUMMARY.md),
  [`passive_binary/PASSIVE_BINARY_MODEL_LADDER.md`](passive_binary/PASSIVE_BINARY_MODEL_LADDER.md)
  (promotion audit is ladder doc &sect;7; no separate closeout doc exists for this leg)
- Active-continuous: [`active_continuous/ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md`](active_continuous/ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md),
  [`active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md`](active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md)
  (promotion audit is ladder doc &sect;6; no separate closeout doc exists for this leg)
- Passive-continuous: [`passive_continuous/PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md`](passive_continuous/PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md),
  [`passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md`](passive_continuous/PASSIVE_CONTINUOUS_MODEL_LADDER.md)
  (no promotion doc -- nothing was ever promoted past Rung 0)
- Active-xT: [`active_xt/ACTIVE_XT_BASELINE_SUMMARY.md`](active_xt/ACTIVE_XT_BASELINE_SUMMARY.md),
  [`active_xt/ACTIVE_XT_MODEL_LADDER.md`](active_xt/ACTIVE_XT_MODEL_LADDER.md) (promotion decision
  is ladder doc &sect;5)
- Passive-xT: [`passive_xt/PASSIVE_XT_BASELINE_SUMMARY.md`](passive_xt/PASSIVE_XT_BASELINE_SUMMARY.md),
  [`passive_xt/PASSIVE_XT_MODEL_LADDER.md`](passive_xt/PASSIVE_XT_MODEL_LADDER.md) (promotion
  decision is ladder doc &sect;5)
