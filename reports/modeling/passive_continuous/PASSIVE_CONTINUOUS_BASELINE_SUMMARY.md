# Passive-Continuous Baseline Summary (d0-d1)

Target: `target_future_xg_10s` (the continuous, shot-conditional counterpart of the passive-binary
leg's `target_future_shot_10s`). Dataset: `data/features/passive_defense.parquet`, filtered to the
95,048 rows where `target_future_shot_10s == 1` (reconfirmed directly against the real parquet, not
assumed to carry over from the active leg's own numbers -- matched exactly). This is this leg's
Rung 0: one dummy floor (`d0_dummy`) and one real candidate (`d1_lognormal_glm`), mirroring the
active-continuous leg's own Rung 0 discipline (Prompt 54) and the binary legs' Rung 0 discipline
before that (Prompts 36/49).

*For the full math behind the models this leg reuses (logistic regression, gradient boosting,
calibration), see [`../MATH_BEHIND_IT.md`](../MATH_BEHIND_IT.md).*

> **Data-quality correction (Prompt 60):** `d0_dummy`/`d1_lognormal_glm` were refit after fixing a
> thin-category artifact in `defender_functional_role`'s encoding. See section 0 below for the full
> root-cause writeup. The corrected numbers replace the originals throughout this document; nothing
> else in sections 1-11 changed as a result (confirmed, not assumed -- see section 0's "what actually
> changed" summary).

## 0. Data-quality correction: `defender_functional_role_unclassified` (Prompt 60)

**What was found.** `d1_lognormal_glm`'s (original Prompt 59 fit) largest-magnitude coefficient by a
wide margin was `defender_functional_role_unclassified` (coef -0.58, roughly 2x the next-largest at
-0.31). On the positive-row subset `d1` is fit on, `unclassified` has only **5 rows out of 95,048**
(0.005%); on the full `passive_defense.parquet` (all rows, not just positive), it accounts for
**1,026 of 1,593,181** rows (0.064%) -- still thin, but not as extreme as the 5-row slice made it
look at first glance, both counts reported rather than only the more dramatic one. Their mean xg
(0.0227) really is much lower than every other role (~0.10 across the other 5 categories) -- but
n=5 cannot support that as a real football finding; it is the linear model overfitting to a
near-singleton bucket.

**Root cause, read directly from source rather than assumed.**
`src/dax/features/passive_defense.py`'s `_functional_roles` function assigns `unclassified`
**exclusively when a frame has fewer than 2 visible defenders** -- there is structurally nothing to
measure depth/laterality "relative to" when only 0 or 1 defenders are visible. It is not a geometric
edge case, missing tracking data, or a classification failure on ambiguous input -- it is a
deliberate, distinct "nothing to compare against" bucket. The function's own docstring states this
explicitly: *"unclassified and mid_block mean different things -- couldn't-measure vs.
measured-and-ordinary -- and must never be treated as interchangeable downstream."*
`src/dax/analysis/passive_archetypes.py` independently confirms the same reasoning in its own module
docstring, excluding `unclassified` from its archetype clustering for the identical "n<2 structural
case, not a behavioural population" reason. Two independent pieces of this codebase already agree on
this distinction before this prompt ever looked at it.

**Fold-in into `mid_block` (the modal category) was considered and rejected.** Both source files'
own docstrings explicitly warn against conflating "couldn't measure" with "measured and ordinary."
There is no principled basis to assign these n<2 rows to *any* of the 5 measured roles -- the whole
reason they are unclassified is that role-relative geometry could not be computed for them at all,
not that it was computed and happened to look ordinary.

**Fix implemented (model-input preprocessing only -- `passive_defense.parquet` is untouched, not
regenerated, not rewritten).** In `scripts/models/train_passive_binary_baseline.py`'s
`DesignMatrixBuilder` (the exact class both this leg's and the passive-binary leg's training scripts
import -- one shared place, not duplicated per script), `"unclassified"` is excluded from the
categories the `OneHotEncoder` is fit on for `defender_functional_role` specifically. With
`handle_unknown="ignore"` already set, any row carrying this value -- at fit time or transform time,
on either leg -- now gets an **all-zero** `defender_functional_role_*` block: no invented role label,
no thin dummy column of its own, and no row dropped from the dataset (every other one of the 38
features is still used for that row). A future refit of the passive-binary leg's own `p1`-`p3`
(not done in this prompt -- see section below) would pick up this fix automatically, by design.

**What actually changed after refitting `d0`/`d1` with the fix.**

- *Gate result: essentially unchanged, confirmed not assumed.* Given the fix touches 2 of 77,646
  train+val rows and 3 of 17,402 held-out rows, the expectation was the Rung-0 gate result shouldn't
  move much -- confirmed directly: common-scale log RMSE CV = 1.0345 (was 1.0345), paired t
  p=0.0053 (was 0.0053), 5/5 folds (unchanged), held-out common-scale log RMSE = 1.0374 (was
  1.0374), hurdle pipeline RMSE/MAE/R² = 0.03726/0.00980/0.0426 (all unchanged to displayed
  precision). Every number in sections 1-11 below is the refit number, and none of them moved
  outside rounding noise.
- *Coefficients: the specific fix worked, and nothing else jumped into the gap.* New top-10
  `d1_lognormal_glm` coefficients (`outputs/models/regression/d1_lognormal_glm.json`):

  | Rank | Feature | Coefficient |
  |---|---|---|
  | 1 | `defender_functional_role_advanced_wide` | 0.742 |
  | 2 | `defender_functional_role_last_line` | 0.705 |
  | 3 | `defender_functional_role_wide_cover` | 0.698 |
  | 4 | `defender_functional_role_central_screen` | 0.687 |
  | 5 | `defender_functional_role_mid_block` | 0.648 |
  | 6 | `has_option_3` | -0.312 |
  | 7 | `on_ball_event_type_Shot` | 0.253 |
  | 8 | `has_option_2` | 0.171 |
  | 9 | `on_ball_event_type_Carry` | -0.170 |
  | 10 | `phase_label_counterpress_after_loss` | 0.108 |

  `defender_functional_role_unclassified` no longer exists as its own dummy, as intended. The 5 real
  role coefficients now cluster tightly together (0.648-0.742, a spread of only 0.094) rather than
  one outlier at 2x the next-largest -- this is the expected, healthier consequence of removing a
  near-singleton category from a one-hot encoding that never drops a reference level (every category
  gets its own dummy, so the design matrix is rank-deficient and the intercept/dummy split is not
  uniquely identified; `unclassified`'s near-pure separation was previously absorbing part of that
  redistribution unstably). Every non-role coefficient (`has_option_3`, `on_ball_event_type_Shot`,
  `has_option_2`, etc.) is **identical to 4 decimal places** to the original Prompt 59 fit -- the fix
  is precisely localized to the role encoding, nothing else moved.
- *Scan for other thin-category artifacts, not just re-checking the one already found.* Every
  categorical and boolean column among the 38 locked PASSIVE features was checked for value counts on
  the positive-row subset `d1` fits on. Nothing else approaches `unclassified`'s original severity:
  the next-thinnest category is `has_option_2 == False` (330 of 95,048 positive rows, 0.35%), two
  orders of magnitude less extreme than `unclassified`'s 5 rows, and its coefficient (0.171, rank 8)
  shows no sign of the same overfit-to-a-singleton pattern. `period == 5` is thinner still on the
  *full* parquet (52 of 1,593,181 rows) but has **zero** positive-row representation at all, so it
  never appears as a category in `d1`'s own fit in the first place. No further fix was needed for
  this leg's regression.

**Cross-check: does the same artifact exist on the passive-binary leg? (informational only, no
changes made).** `defender_functional_role` is also a locked feature for the passive-binary leg's
already-promoted `p1e_gradient_boosting_calibrated` (a gradient-boosted model, structurally much less
prone to this failure mode) and for the linear `p1_unweighted`/`p2_weighted`/
`p3_weighted_interactions` variants. Reading their already-fitted coefficient JSONs directly
(read-only -- **none of these were retrained, refit, or touched in this prompt**):

| Variant | `defender_functional_role_unclassified` coefficient | Rank by \|coef\| |
|---|---|---|
| `p1_unweighted` | -0.389 | not top-8, but the largest-magnitude of the 6 role dummies |
| `p2_weighted` | **-2.269** | **#1 of all 51 features** |
| `p3_weighted_interactions` | **-2.343** | **#1 of all features** |

**The same artifact is present, and materially worse on `p2`/`p3`** (this leg's own full dataset,
1,026 of 1,593,181 rows, 0.064% -- thinner in proportion than the positive-row-only 0.005% `d1` saw,
which is the reverse of the direction that would make this less concerning) -- `unclassified` is the
single largest-magnitude coefficient in the entire model for both `p2_weighted` and
`p3_weighted_interactions`, well ahead of `on_ball_event_type_Shot`. **This is reported here as a
follow-up item for a future prompt to decide whether it's worth revisiting**, since
`p1e_gradient_boosting_calibrated` (not `p2`/`p3`) is this leg's actual promoted, standing reference
model and per this prompt's own constraint none of `p1`/`p2`/`p3`/`p1e` were retrained or repromoted
here.

## 1. Why a hurdle architecture, not a single regression

Two things about the passive dataset were reconfirmed directly against real data before writing
this script (not assumed to transfer from the active leg's own Prompt 54 checks).

**Distribution shape transfers almost exactly.** `target_future_xg_10s`'s shot-conditional
distribution on the passive leg (the 95,048 rows where a shot did follow) has **raw skew 3.28 /
excess kurtosis 14.0, log-scale skew 0.19 / excess kurtosis -0.13** -- essentially the same shape as
the active leg's own target (skew ~3.2-3.3, kurtosis ~13-14 raw; near-normal on the log scale). **67
of the 95,048 positive rows exceed `xg=1.0`** (0 rows `<= 0`), same reasoning as the active leg: the
target sums xG across *every* shot in the 10-second window, not one shot's probability. Log-normal
was already confirmed the best-fitting family for this shape on the active leg, and the same raw
distribution check here supports the same conclusion without needing to redo the full
family-comparison analysis. This is why this leg is built as a **two-part ("hurdle") model**, same
architecture as the active-continuous leg: reuse the already-promoted binary classifier
(`p1e_gradient_boosting_calibrated`, Prompt 52) for `P(shot)`, and fit a new regression only on the
shot-positive rows for `E[xg | shot]`, on a log scale.

**Row grain creates real duplication that must be documented, not silently absorbed.**
`passive_defense.parquet` is one row per visible defender-slot per on-ball attacking event. Among
positive rows specifically: **95,048 rows come from only 10,700 unique events** (~8.9 defender-slot
rows per event on average, reconfirmed), and **every row sharing an `event_id` has the identical
`target_future_xg_10s` value** (0 of 10,700 events show more than one distinct value, reconfirmed
directly) -- the same shot outcome, viewed from a different defender's geometry. **Varun's explicit
decision: fit on all defender-slot rows as-is.** This is not a duplication bug to engineer away --
it is this leg's actual premise (does *this* defender's positioning explain the outcome), the same
premise the passive-binary leg's own baseline was built on. The report states the
effective-sample-size caveat plainly rather than letting the large row count imply a false sense of
independence: **~10,700 independent shot events underlie the 95,048 training rows**, and any given
CV fold's aggregate metric implicitly weights events with more visible defenders more heavily than
events with fewer.

**Window length is not reopened here.** Prompt 54's sensitivity check on the active leg (25-match
sample, real `add_future_shot_target`/`add_future_xg_target` code from `dax.targets.short_horizon`)
found the existing 10-second window captures only ~41% of possessions that eventually produce a shot,
and that a full possession-bound framing is worse (some possessions run well past the point where the
original action's own features remain a fair predictor). That check is **leg-agnostic** -- it
operates on event/possession timing via the shared target-generation code, not on the active/passive
feature split -- so the same accepted 10s-window limitation applies unchanged here. This document
does not re-derive that analysis.

## 2. Confirmed inputs

**Classifier half, reused exactly, never retrained.** `p1e_gradient_boosting_calibrated` -- the
passive-binary leg's own promoted standing reference model (Prompt 52). Artifact confirmed on disk at
`outputs/models/classification/p1e_gradient_boosting_calibrated.joblib`. No saved held-out prediction
array existed for it, so its held-out score here was regenerated by loading the saved model and
scoring through a freshly-fit (cheap, deterministic) `RFDesignMatrixBuilder`
(`train_p1d_random_forest.py`'s own class) -- the identical reuse pattern the active-continuous leg's
own baseline script (Prompt 54) already used for its classifier half.

**Feature set.** All **38 locked PASSIVE features** (4 categorical, 6 boolean, 26 continuous, 2
discrete, per `src/eda/feature_config.py`'s `PASSIVE` dict) -- reconfirmed by count, not assumed. No
feature excluded for this leg (unlike the active leg's 34-to-32 exclusion); nothing in this leg's own
data forced a further exclusion beyond the leakage exclusions (`has_screened_outcome`,
`screened_option_was_avoided`) already locked out at the feature-config level.

**Positive-row structure.** 95,048 of 1,593,181 rows, from 10,700 unique events. Split via the
canonical match-grouped assignment (shared with the active leg and both binary legs -- this leg does
not compute a fresh split): 77,646 rows / 92 matches / 8,721 unique events in train+val, 17,402 rows /
23 matches / 1,979 unique events in held-out test.

| Fold | Positive rows | Unique events |
|---|---|---|
| 0 | 15,833 | 1,772 |
| 1 | 18,071 | 2,039 |
| 2 | 13,222 | 1,512 |
| 3 | 15,915 | 1,775 |
| 4 | 14,605 | 1,623 |

**No fold has fewer than 100 unique events** -- the real check on effective sample size (not just row
count, which is always large here given the ~8.9x defender-slot multiplier). Every fold's CV metrics
rest on a stable effective sample size.

## 3. Rung 0 variants

| Variant | Model | Notes |
|---|---|---|
| `d0_dummy` | `DummyRegressor(strategy="mean")` on `log1p(xg)` | training-fold mean of `log1p(xg)`, back-transformed via `expm1` for original-scale reporting |
| `d1_lognormal_glm` | `LinearRegression()` on `log(xg)` | natural log (every positive row has `xg > 0` by construction, reconfirmed: 0 rows `<= 0`), same 38 locked features, same `DesignMatrixBuilder` preprocessing as the passive-binary leg (imported directly, not reimplemented) |

Naming note: `c0`/`c1` already means something on the active-continuous leg, `p0`/`p1` already means
something on the passive-binary leg -- this leg uses **`d0`/`d1`** ("d" for the passive leg's
defender-slot regression), the first letter not already claimed by another leg, to avoid colliding
with either existing prefix.

`d0` and `d1` are fit on different log transforms by this leg's own design (`log1p` for the dummy
floor, natural `log` for the real candidate) -- their own "native" log-scale RMSE/MAE numbers are
therefore not directly comparable to each other, same scale-mismatch risk Prompt 54 found and fixed
on the active leg. Section 4 below uses a **common-scale** metric (natural log of both variants'
back-transformed xg-scale predictions) for the actual head-to-head comparison.

## 4. Cross-validation results (5-fold, OOF metrics)

| Variant | Common-scale log RMSE | Common-scale log MAE | Naive MAE (xg scale) | Corrected MAE (xg scale) | Naive R² | Corrected R² |
|---|---|---|---|---|---|---|
| `d0_dummy` | 1.1484 | 0.9382 | 0.0807 | 0.0807 | -0.0040 | -0.0040 |
| **`d1_lognormal_glm`** | **1.0345** | **0.8216** | **0.0716** | 0.0824 | -0.0898 | **0.0062** |

`d1` beats `d0` on the fair common-scale log RMSE/MAE, on original-scale naive MAE, and on corrected
R² (the only metric where `d0` isn't already at its ceiling, same reason as the active leg -- a
constant prediction has no meaningful log-normal correction to apply). `d1`'s naive R² is negative
(-0.090) despite ranking rows meaningfully better than `d0` (Spearman +0.097 vs effectively 0 for a
constant prediction) -- resolved in section 5 below, same pattern as the active leg's own Rung 0.

## 5. Two back-transforms: naive `exp(pred)` vs log-normal-corrected `exp(pred + σ²/2)`

Same reasoning as the active leg (Prompt 54): exponentiating a log-scale prediction directly
(`exp(pred)`) recovers the **median** of the implied log-normal distribution, not its mean -- and
`E[xg]` is what the downstream hurdle combination actually needs. By Jensen's inequality, the naive
back-transform systematically under-predicts the true conditional mean. The standard correction,
`exp(pred + σ²/2)`, corrects for this.

Real numbers, `d1`, held-out test: fitted `σ² = 1.046` (larger than the active leg's own `σ² = 0.934`,
consistent with this leg's slightly heavier log-scale residual spread), giving a mean correction
factor `exp(1.046/2) ≈ 1.68` -- the corrected prediction runs about 68% higher than the naive one on
average.

**Which one should the hurdle combination use, and why**: the **corrected** back-transform, same
conclusion as the active leg. Naive `exp(pred)` has a *lower* MAE here (CV: 0.0716 vs 0.0824) -- but
MAE is minimized by the **median**, and naive `exp(pred)` is approximately a median-type estimator.
R²-type metrics, which are about matching the **mean**, tell the opposite story: `d1`'s naive R² is
negative (-0.090), while `d1`'s corrected R² is positive (+0.006, small but genuinely better than the
constant-mean floor). Since the hurdle combination needs an unbiased estimate of the conditional
**mean**, the corrected back-transform is used for that purpose throughout this document and in
`hurdle_pipeline_readout_passive.json`.

## 6. Calibration (5 bins by predicted value, held-out test, `d1` corrected predictions)

| Bin | n | Mean predicted xg | Mean actual xg | Gap |
|---|---|---|---|---|
| 0 (lowest predicted) | 3,481 | 0.0784 | 0.0840 | -0.0056 |
| 1 | 3,481 | 0.0884 | 0.0936 | -0.0053 |
| 2 | 3,480 | 0.0958 | 0.1000 | -0.0043 |
| 3 | 3,480 | 0.1045 | 0.1005 | +0.0040 |
| 4 (highest predicted) | 3,480 | 0.1261 | 0.1168 | +0.0093 |

No dramatic systematic bias in either direction -- gaps are small relative to the bin means (roughly
-7% to +8% relative) and don't monotonically worsen at either extreme, the same clean pattern the
active leg's own Rung 0 calibration showed.

## 7. Held-out test readout (single readout, `FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION`)

| Variant | Common-scale log RMSE | Naive MAE | Corrected MAE | Naive R² | Corrected R² |
|---|---|---|---|---|---|
| `d0_dummy` | 1.1691 | 0.0827 | 0.0827 | -0.0013 | -0.0013 |
| **`d1_lognormal_glm`** | **1.0374** | **0.0718** | 0.0839 | -0.0869 | **0.0078** |

Same direction as CV throughout: `d1` beats `d0` on common-scale log RMSE, naive MAE, and corrected
R². `d1`'s corrected R² actually improves slightly from CV (0.0062) to held-out (0.0078) -- no sign
of overfitting on this leg's much larger sample, same pattern the active leg's own Rung 0 showed on
its smaller sample.

## 8. Paired significance test, `d1` vs `d0`

Common-scale log RMSE, per canonical CV fold (`outputs/models/validation/significance_d0_vs_d1.json`):
`d1` beats `d0` in **5 of 5 folds** (mean diff -0.1102, i.e. `d1`'s RMSE is lower). Paired t-test
**p=0.0053** -- significant at the conventional threshold. Wilcoxon signed-rank p=0.0625, the same
5-fold floor (the minimum p-value achievable with 5 same-signed paired samples) seen throughout this
project, unrelated to sample size.

**Effective-sample-size note, read carefully rather than assumed**: this leg's per-fold *row* count
(13,222-18,071) is far larger than the active leg's own (628-830), which might suggest a much more
powerful test. But the real effective sample size for this test is closer to the per-fold *unique
event* count (1,512-2,039, minimum 1,512), since every defender-slot row sharing an `event_id`
carries the identical target value -- rows are not independent observations of different outcomes.
**The test's row-count-implied power should not be read at face value.** That said, even the thinnest
fold's event count (1,512) is still well above the active leg's own per-fold *row* counts (hundreds),
so this remains a real, meaningfully-sized test -- just not as large as the raw row count alone would
suggest, and the active leg's small-n caution note does not mechanically carry over here unexamined.

## 9. Full hurdle pipeline readout (diagnostic, not the Rung-0 gate)

`P(shot)` (from `p1e_gradient_boosting_calibrated`, reused unchanged) times `E[xg|shot]` (from
`d1_lognormal_glm`, corrected back-transform), computed for **every** held-out test row (315,807
rows, not just the positive ones, and not collapsed across defender-slots -- the classifier's own row
grain is confirmed identical to this regression's, both built from the same `passive_defense.parquet`
row index), compared against the real `target_future_xg_10s`
(`outputs/models/validation/hurdle_pipeline_readout_passive.json`):

| | RMSE | MAE | R² |
|---|---|---|---|
| Hurdle pipeline (`P(shot) x E[xg\|shot]`) | **0.03726** | **0.00980** | **0.0426** |
| Trivial baseline (unconditional train-set mean of `target_future_xg_10s`) | 0.03809 | 0.01090 | ~0.0000 |

The two-part architecture beats the trivial constant-mean baseline on every metric -- a real, positive
first read on whether combining the two already-separately-validated halves works end to end on this
leg too. **This is explicitly a diagnostic, not the Rung-0 gate**: the Rung-0 promotion decision for
this leg rests on the regression-only `d1`-vs-`d0` comparison (sections 4/8 above), since that is the
piece this prompt actually built; the classifier half was already gated and promoted on its own leg
(Prompt 52).

## 10. Charts

4 PNGs per variant, saved under `outputs/models/regression/charts/<variant>/` (gitignored --
regenerate by running `scripts/models/train_passive_continuous_baseline.py` if the images below don't
resolve): `predicted_vs_observed_log.png`, `residual_distribution.png`, `calibration_bins.png`,
`feature_coefficients.png` (`d1` only; `d0`'s is an "n/a" placeholder, same convention as every other
leg's Rung 0). See the HTML version of this document for inline images.

## 11. Rung-0 result

**`d1_lognormal_glm` clears Rung 0** against the `d0_dummy` floor: consistent direction across CV and
held-out test, on the fair common-scale log metric and on original-scale MAE/R², with a statistically
significant paired test (p=0.0053, and unlike the active leg's own Rung 0 this is not a small-n
result even after accounting for the event-level effective sample size), and a positive first
end-to-end hurdle readout against a trivial baseline. No model ladder is built in this document --
this is Rung 0 only, mirroring the active-continuous leg's own Prompt 54 discipline and the binary
legs' Prompt 36/49 discipline before that. Whether this leg gets the same quadratic/RF/GBM ladder the
active leg did is a decision for after this Rung 0 is reviewed, not assumed here.

This is the fourth and final leg of the project's model-ladder structure -- all four legs (active
binary, passive binary, active continuous, passive continuous) now have a real, validated Rung 0.

## 12. Rung 1 -- `d1b_quadratic` (prompt 61)

**Structural note on this document, decided explicitly rather than defaulted into**: the
active-continuous leg appended its own Rung 1 (`c1b_quadratic`, prompt 55) directly to its baseline
summary rather than splitting into a separate ladder doc, and only split once a second rung existed
(`ACTIVE_CONTINUOUS_MODEL_LADDER.md`, prompt 56). This leg is at the identical point -- 1 rung beyond
Rung 0 -- so the same timing applies: Rung 1 is appended here, not split out yet. That split will
happen once a second rung is built on this leg, consistent with both the active-continuous leg's own
precedent and the binary legs' before that.

### 12.1 Reconfirming shot-conditional nonlinearity from real data, systematically

A rough chat check of the shot-conditional signal (positive rows only, n=95,048) found every one of
the 26 locked continuous PASSIVE features correlates very weakly with `log(xg)` once conditioned on a
shot occurring -- strongest `lane_screening_score_option_1` (pearson +0.057, spearman +0.065),
followed by `lane_screening_score_option_2` (+0.053/+0.063) and `engagement_distance_to_carrier`
(-0.049/-0.049); everything else under 0.04, `overload_score` near zero (+0.006). This is genuinely
weaker than the active leg's own Rung 1 starting point (active's strongest shot-conditional
correlation, `visible_attacker_count`, was |r|=0.16 -- over 2x stronger). The likely explanation:
this leg's rows describe one individual defender's geometry relative to a single event, not an
aggregated picture of the whole defensive shape the way the active leg's features are -- a lone
defender's positioning plausibly has less individual leverage over eventual shot quality than the
attacking side's own aggregate spatial features do.

`scripts/analysis/check_shot_conditional_nonlinearity_passive.py` reproduces this properly across all
38 locked PASSIVE features (not just the 26 continuous the chat check covered -- it missed both
discrete features, `overload_score` and `defender_slot_index`), computing Pearson/Spearman for the 28
numeric features, point-biserial for boolean features, and ANOVA eta-squared for categorical features.
Full output: `outputs/models/validation/shot_conditional_nonlinearity_check_passive.json`.

**The systematic check confirms the chat shortlist's top-3 by correlation magnitude, but finds none
of them curved.** Among numeric features: `lane_screening_score_option_1` (|r|=0.0646),
`lane_screening_score_option_2` (|r|=0.0630), and `lane_screening_score_option_3` (|r|=0.0478, a
4th chat feature not in the original 3-feature shortlist but the same family) are the 3
strongest-correlated numeric features overall -- reconfirmed, not a stale shortlist. **But the
quintile-binning curvature diagnostic flags all three `monotonic_linear`**: a straight trend, no
curvature a squared term would add beyond what the linear term already captures. Being
top-correlated does not make them quadratic-term candidates here -- the diagnostic says plainly they
are not.

**Only one numeric feature clears both bars** (a correlation magnitude in the same range as the top 3,
*and* a genuine non-linear curvature flag):

| Feature | \|r\| | Curvature shape | Note |
|---|---|---|---|
| `engagement_distance_to_carrier` | 0.049 | monotonic, concentrated gap at the low-x end (>2x the other gaps) | rank #4 by correlation, the only feature combining real magnitude with real curvature |

Quintile means: -2.764, -2.828, -2.874, -2.905, -2.919 (log xg), on ~19,010 rows per bin -- a genuine
diminishing-returns/concave shape, not a binning artifact given the bin size. Every other numeric
feature flagged with curvature by the same diagnostic has |r| well under 0.04 (many under 0.02) --
indistinguishable from noise on a target this weak, and not used as candidates.

`QUADRATIC_FEATURES_D1B = [engagement_distance_to_carrier]` -- **1 feature, not the active leg's 3**,
and none of the active leg's own quadratic candidates (which mostly don't exist in the passive
feature set at all). This is explicitly the minimal-single-candidate case the task brief allowed for
when the underlying signal is this weak, not a watered-down copy of the active leg's approach.

### 12.2 `d1b_quadratic` results

Same 38 locked features, same `DesignMatrixBuilder` and `LinearRegression` as `d1` (including the
`defender_functional_role_unclassified` exclusion fix from Prompt 60, reused exactly, not
reintroduced), plus one `quad__engagement_distance_to_carrier` column.

| Variant | Common log RMSE (CV) | Common log RMSE (held-out) | Naive MAE (held-out) | Corrected R² (held-out) |
|---|---|---|---|---|
| `d1_lognormal_glm` | 1.0345 | 1.0374 | 0.0718 | 0.0078 |
| `d1b_quadratic` | 1.0346 | 1.0374 | 0.0718 | 0.0077 |

**`d1b_quadratic` does not beat `d1` -- on either CV or held-out, on any metric, and the numbers are
not merely close, they are essentially identical** (differences in the 4th-5th decimal place, smaller
than the active leg's own already-small Rung 1 gap). The single surviving quadratic coefficient
(`quad__engagement_distance_to_carrier`) is **+0.0038** -- negligible in magnitude.

### 12.3 Significance test, d1b vs d1

`outputs/models/validation/significance_d1_vs_d1b_quadratic.json`: mean diff (d1b - d1) common-scale
log RMSE = **+0.0000** (indistinguishable), paired t-test **p=0.5013**, Wilcoxon p=0.625. **Does not
clear significance in either direction, and by a wide margin** -- this is an even more emphatic null
result than the active leg's own Rung 1 (p=0.2073 there; p=0.50 here). Per-fold unique-event counts
(minimum 1,512, same as Rung 0) confirm this is not a thin-sample artifact -- the effective sample
size is large enough that a real effect this size would very likely have shown up.

### 12.4 This is a real finding about this leg, not a modelling failure

Stated plainly: **`d1b_quadratic` shows no real improvement over `d1`, and this connects directly to
section 12.1's already-documented weak-correlation finding, not a coincidence.** This leg's
shot-conditional correlations were already markedly weaker than the active leg's own Rung 1 starting
point before any quadratic term was tried (strongest |r|=0.065 here vs |r|=0.16 there) -- one weak
feature's squared term, chosen precisely because it was the *only* feature showing genuine curvature
at all, simply doesn't have enough underlying signal to move a fair, common-scale metric measurably.
This is not a surprise; it is the expected continuation of a pattern already visible in item 1's own
numbers before the model was ever fit.

**`d1_lognormal_glm` remains this leg's best Rung-0/Rung-1 candidate**; `d1b_quadratic` is not
promoted or preferred over it. Whether later rungs (interaction terms, tree-based models) can extract
more signal than a linear model can from this weak a feature set -- a real open question given how
thoroughly this leg's own linear/quadratic signal has now been exhausted -- is left for those rungs
to answer on their own evidence, not assumed from this rung's result.
