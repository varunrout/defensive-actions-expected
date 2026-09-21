# Active-xT Model Ladder

> **Status update (Prompt 74):** `x1c_random_forest` (Rung 2) remains the leg's standing baseline.
> `x1d_gradient_boosting` (Rung 3) was built and shows a small, statistically significant edge on
> RMSE/MAE/R² -- but a real, measurable *regression* on rank correlation and the tail-calibration
> gap this ladder has tracked since Rung 0. See section 4.7 for the full, mixed-result promotion
> call. `x1b_quadratic` (Rung 1) did not clear its own gate and remains documented as the rung
> that didn't win.

*Split out of `ACTIVE_XT_BASELINE_SUMMARY.md` (prompt 73, once a second rung -- Rung 2,
`x1c_random_forest` -- existed beyond Rung 0) so the locked Rung 0 baseline document doesn't keep
growing as more rungs are tried -- mirroring `active_continuous`'s own exact split point
(`ACTIVE_CONTINUOUS_MODEL_LADDER.md`, split at its own Rung 2, `c1d_random_forest`) and the binary
legs' split points before that. Rung 0 -- the two locked variants (`x0_dummy`,
`x1_two_stage_huber`), the architecture decision, and the Rung-0 calibration finding -- stays in
[`ACTIVE_XT_BASELINE_SUMMARY.md`](ACTIVE_XT_BASELINE_SUMMARY.md). This document is everything built
on top of that locked Rung 0 baseline, starting from Rung 1. This is a pure reorganization for
Rung 1's material: no numbers changed, verified by rendering both HTML documents in-browser after
the split, same verification `active_continuous`'s own split commit used.*

## 1. What a rung is

A **rung** is one controlled change to the current standing baseline: a single variable changed --
feature shape, an interaction term, a model family -- with everything else held identical (same 32
locked features unless the rung specifically changes that, same canonical match-grouped 5-fold
split, same signed `target_xt_delta_v2` target, no ad-hoc tuning beyond what a rung's own grid
search covers). A rung is tested against 2 fixed gates before it can be considered for adoption as
the new baseline:

1. **Paired significance test vs the current standing baseline** -- refit both variants on the
   same 5 canonical CV folds, paired t-test + Wilcoxon signed-rank test on the per-fold RMSE
   difference (no log-transform on this leg's signed target, so no common-scale correction is
   needed the way the continuous legs' `log1p`-vs-`log` mismatch required).
2. **Held-out-test confirmation** -- a single, one-time readout on the frozen held-out test set
   (10,595 rows / 23 matches). This is the gate that actually decides the rung's fate: CV picks a
   candidate, the held-out test either confirms or rejects it, and that result is not re-litigated
   after the fact to fit a preferred narrative.

This leg carries a **third, rung-specific check no other leg's ladder needs**: the 5-bin
calibration table first built at Rung 0 (`ACTIVE_XT_BASELINE_SUMMARY.md` section 7), because Rung 0
identified a real, specific limitation (the two tail bins under-predicting magnitude by 3x or
more) that is the actual open question this ladder has been tracking since it started -- an
aggregate metric improving without that gap closing would be an incomplete answer, not a full one,
per Prompt 71/73's own explicit instruction not to let a better-looking aggregate number stand in
for the motivating problem.

Rung 0 (the baseline itself) lives in
[`ACTIVE_XT_BASELINE_SUMMARY.md`](ACTIVE_XT_BASELINE_SUMMARY.md). This ladder builds on that
document's `x1_two_stage_huber` pick (held-out RMSE 0.05309, R² 0.1519) as the baseline every rung
up through Rung 1 is compared against; Rung 2 changes the comparison baseline to `x1c_random_forest`
once it is promoted (section 3.7). *(Fixed cross-reference, prompt 74 -- this note previously
pointed at section 3.5, a stale reference from before this document's final section numbering
settled; the ladder-gate outcome has always lived in section 3.7.)*

## 2. Rung 1 -- `x1b_quadratic` (prompt 71)

### 2.1 Step 0 -- where do quadratic/interaction terms go, and why

`x1` has two surfaces (Stage A logistic classifier for P(nonzero), Stage B Huber regression for
E[delta|nonzero]), unlike `v1b`'s single classifier or `c1b`'s single regression head -- `c1b` is
the closer precedent structurally (also a two-part active-continuous leg), but even `c1b` only
ever had **one** surface to expand: its classifier half (`v1e`) was reused unchanged, never
re-fit with new terms. This leg is the first one where both halves are actually fit inside the
same rung, so which surface(s) get the new terms has no direct precedent to copy -- it is
answered here from this leg's own data, not assumed.

Checked directly against `reports/analysis/xt_target/active_numerical_target_atlas.json`, read
via its `bins_nonzero_delta` panel first (since that is exactly where Prompt 70's motivating
problem -- the tail-calibration gap -- lives: the magnitude of nonzero deltas). Three features
clear a real-curvature bar, the same standard `c1b` used (a genuine, non-monotonic or
accelerating pattern, not a straight line, and not a `small_n`-flagged tail artefact):

| Feature | r vs. nonzero delta | Pattern |
|---|---|---|
| `distance_to_attacking_box` | **0.157** (strongest numeric correlate on this leg) | clean 8-of-8 monotonic increase across its 9 bins, with the final three gaps larger than the first six -- accelerating-monotonic, same shape `c1b`'s own `visible_attacker_count`/`defenders_within_10m` showed |
| `defender_attacker_gap_x` | -0.062 | genuine inverted-U: rises for 4 bins, peaks, falls for 5 -- not a straight line |
| `visible_defender_count` | 0.047 | the atlas's own `shape` field: `U-shaped`; dip then accelerating rise across its non-thin bins |

Several other numeric features show apparent tail spikes (e.g. `defenders_within_10m`'s bin-11
mean of +0.177, `attackers_within_10m`'s bin-9 mean of +0.193) that are **not** used -- reconfirmed
directly against the atlas's own `small_n` flags, every one of those spikes sits on n&le;23 rows,
several on n=1, indistinguishable from noise on a target with excess kurtosis 18.33. Padding the
candidate list with these would be exactly the "quadratic terms on features with no real signal"
mistake `c1b`'s own docstring warns against.

**The same atlas also reports each bin's zero-rate for these same three features** (its
unconditional `bins` panel; `pct_zero = 100 - pct_negative - pct_positive`):
`distance_to_attacking_box`'s zero-rate is not flat across bins (14.9% -> 26.5% -> 15.3%, an
inverted-U of its own), `defender_attacker_gap_x`'s zero-rate ranges 15.9%-25.9% with the same
rise-then-fall shape, and `visible_defender_count`'s zero-rate ranges 8.4%-26.6%. **This is the
evidence the Step-0 decision rests on**: the same three features show real curvature in *both*
the zero-rate (what Stage A predicts) and the nonzero-delta magnitude (what Stage B predicts) --
not a coincidence to ignore, and not a default to "add terms everywhere" either, since the
candidate set was selected for its regression-stage evidence first (Prompt 70's own motivating
problem) and only kept for the classifier stage because the *same* features independently clear
a curvature bar there too.

**Decision: expand both stages with the identical feature set.** Not independently tuned per
stage -- there is no evidence in hand that a different curvature shape applies to one stage vs.
the other for these three features, and inventing a second, classifier-specific candidate list
without its own evidence would be exactly the "add terms everywhere" default this prompt was
told to avoid.

### 2.2 Feature-set construction

**Quadratic (3 features, not a fixed count)**:
`QUADRATIC_FEATURES_X1B = ["distance_to_attacking_box", "defender_attacker_gap_x", "visible_defender_count"]`

**Interaction (1 pair)**: reused directly from
`reports/analysis/xt_target/FEATURE_INTERACTION_ANALYSIS.json` (Prompt 65/67's own xT-specific
interaction findings, target `target_xt_delta_v2`, not re-discovered from scratch). Of its 5
curated ACTIVE pairs, 2 are classified `interactive`: `defenders_within_5m x defenders_within_10m`
and `visible_defender_count x attacker_spread`. The first pair is not usable here --
`defenders_within_5m` is excluded from this leg's 32 locked modelling features (structurally
nested inside `defenders_within_10m`, Prompt 70's own feature-set decision, unchanged here). The
second pair is fully eligible (both features are locked modelling features) and is used:
`INTERACTION_PAIR_X1B = ("visible_defender_count", "attacker_spread")`.

Both terms are appended to both stages' design matrices (35 columns total: 32 locked features + 3
quadratic + 1 interaction).

### 2.3 Cross-validation results (5-fold, OOF metrics)

| Variant | RMSE | MAE | R² | Spearman | Zero-row MAE | Nonzero-row MAE |
|---|---|---|---|---|---|---|
| `x1_two_stage_huber` | 0.05394 | 0.02069 | 0.1514 | **0.5242** | **0.00478** | **0.02468** |
| `x1b_quadratic` | **0.05391** | 0.02070 | **0.1525** | 0.5233 | 0.00481 | 0.02468 |

Movement is real but tiny in every column: RMSE improves by 0.00004 (0.07% relative), R² improves
by 0.0011, MAE and nonzero-row MAE are flat to the 5th decimal, and Spearman and zero-row MAE are
each marginally *worse*. This is not the shape of a rung that meaningfully improved the model.

### 2.4 Paired significance test, `x1b` vs `x1`

Per-fold RMSE (`outputs/models/validation/significance_x1_vs_x1b_quadratic.json`): `x1b` beats
`x1` in 5 of 5 folds, but the per-fold differences are small and consistent enough to produce a
significant paired t-test (t=-7.37, **p=0.0018**) despite the practically negligible effect size
-- mean diff -0.00004, roughly two orders of magnitude smaller than Rung 0's own `x1`-vs-`x0`
mean diff (-0.00461). Wilcoxon signed-rank p=0.0625, the same 5-fold floor seen throughout this
project. **A significant p-value here is not being read as "clears the bar"** -- see section 2.6.

### 2.5 Calibration -- the actual motivating question for this rung

| Bin | `x1` gap (Rung 0) | `x1b` gap (this rung) | Change |
|---|---|---|---|
| 0 (most negative predicted) | +0.02074 | **+0.02113** | worse by 0.00039 |
| 1 | +0.00038 | -0.00019 | flips sign, magnitude similar |
| 2 | +0.00009 | -0.00019 | flips sign, magnitude similar |
| 3 | +0.00059 | +0.00081 | worse by 0.00022 |
| 4 (most positive predicted) | -0.01713 | **-0.01725** | worse by 0.00012 |

**The tail-calibration gap that motivated this entire rung did not narrow -- it is unchanged to
worse.** Bin 0's gap (the more negative-magnitude tail) widens slightly (+0.02074 -> +0.02113);
bin 4's gap widens slightly too (-0.01713 -> -0.01725). The middle bins move by amounts smaller
than the bin-to-bin noise already present in Rung 0's own table. Stated plainly: adding curvature
that the underlying feature-target relationship genuinely has did not translate into `x1b`
predicting the extreme rows any better, because the limitation Prompt 70 identified is a property
of `HuberRegressor`'s own loss function (its robustness to outliers is what caps how far it will
move a prediction toward an extreme observed value), not a property of the linear functional form
the added quadratic/interaction terms were meant to fix. A curved relationship fit with a still-Huber
loss is still capped the same way at the tails.

### 2.6 Promotion call

**`x1b_quadratic` does not clear the bar to replace `x1` as the standing baseline.** The aggregate
metrics move in `x1b`'s favor by an amount too small to matter operationally (RMSE 0.07% lower,
R² +0.0011), and the one metric this rung actually exists to move -- the tail-calibration gap --
did not improve; if anything it is marginally worse. A statistically significant paired test
(section 2.4) is not being treated as sufficient on its own, since a real effect can still be
too small to be worth carrying forward, and this leg's own Step-0 evidence (the tail bins) is a
closer read of practical value than a p-value computed on a sub-0.0001 mean RMSE difference.

**`x1_two_stage_huber` remained the standing baseline** carried forward into Rung 2 below. A
valid, reportable Rung-1 outcome, not a failure to fix before moving on: the evidence in section
2.1 was real (the candidate features do show genuine curvature in both stages' target
quantities), and testing that evidence properly required actually fitting the expanded model
rather than assuming curvature terms would help because the correlation coefficients looked
non-trivial -- they were real, they just were not the right kind of fix for a
loss-function-driven calibration limitation.

## 3. Rung 2 -- `x1c_random_forest` (prompt 73)

`x1b_quadratic` (Rung 1) established that the tail-calibration problem is a property of
`HuberRegressor`'s own loss function, not the linear functional form -- more curved features fed
into the same robust-but-capping loss function didn't move the tail bins. This rung tests the
next, different hypothesis: does changing the **model family** entirely (random forest, whose
leaf-averaging has a fundamentally different mechanism for handling extreme observations) close
the gap that more features on the same loss function could not.

### 3.1 Step 1 -- which surface(s) get replaced with a random forest, and why

Checked directly against `train_c1d_random_forest.py` before deciding anything here, per this
leg's own standing discipline (Prompt 71 Step 0) of checking the closer structural precedent
rather than defaulting. `c1d_random_forest` replaced ONLY `c1`'s regression head with a random
forest -- its classifier half (`v1e_gradient_boosting_calibrated`) was reused EXACTLY as already
fitted, never re-tuned or re-fit with a different family.

But the REASON `c1d` only touched one surface is structural, not evidence-based: `v1e` is a
separately-built, already-promoted model belonging to an entirely different leg (active-binary),
reused via composition across leg boundaries -- it was never part of `c1`'s own Rung 0/1/2 ladder
at all, so there was nothing of `c1`'s own to leave untouched on the classifier side; the
classifier simply isn't that leg's model to retune.

That reasoning does not transfer to `x1`. Both of `x1`'s surfaces (Stage A classifier for
P(nonzero), Stage B regressor for E[delta|nonzero]) were built together, inside this leg's own
Rung 0 (Prompt 70) -- neither is a cross-leg reused artifact. Rung 1's own Step 0 (section 2.1)
already established, with real numbers, that the same candidate features show real curvature in
*both* the zero-rate (Stage A's target) and the nonzero-delta magnitude (Stage B's target) --
there is no asymmetric evidence here that would justify upgrading one surface's model family
while leaving the other on a weaker one for no data-driven reason.

**Decision: replace both stages with RandomForest** -- `RandomForestClassifier` for P(nonzero),
`RandomForestRegressor` for E[delta|nonzero]. Each is tuned independently on its own natural OOF
metric (classifier: ROC-AUC; regressor: RMSE on the raw signed delta, no log transform) rather
than jointly grid-searched as one combined pipeline -- a joint grid would require fitting every
(classifier-params x regressor-params) combination, unneeded since each stage's own
hyperparameters only affect that stage's own fit quality and the combined prediction is a simple
product of the two independently-best models (the same composition `x1`/`x1b` already use).

### 3.2 Hyperparameter grids, sized for this leg's real row counts

Classifier trains on ~36-45k rows/fold (full trainval, 45,166 rows total); regressor trains on
~29-36k rows/fold (nonzero-only trainval, 36,121 rows total) -- between `c1d`'s small scale
(~3,500 rows) and `d1d`'s much larger duplicated-row scale (hundreds of thousands).
`min_samples_leaf` floor set well above `c1d`'s 10-60 (this leg has far more rows) but well below
`d1d`'s 30-300 (this leg's rows are not defender-slot-duplicated -- 1 row per event, more
independent signal per row than `d1d`'s ~8x-duplicated grain). Grid used for both stages:
`N_ESTIMATORS=200` (fixed), `MAX_DEPTH_GRID=[6, 10, None]`, `MIN_SAMPLES_LEAF_GRID=[20, 50, 150]`
(9 combinations each, 18 total).

**Classifier grid** (chosen by OOF ROC-AUC, `outputs/models/validation/x1c_classifier_tuning.json`):

| max_depth | min_samples_leaf | OOF ROC-AUC | Train ROC-AUC | Train &minus; OOF gap |
|---|---|---|---|---|
| 6 | 20 | 0.7360 | 0.7521 | 0.0160 |
| 6 | 50 | 0.7357 | 0.7492 | 0.0134 |
| 6 | 150 | 0.7322 | 0.7430 | 0.0107 |
| 10 | 20 | 0.7553 | 0.8036 | 0.0483 |
| 10 | 50 | 0.7513 | 0.7851 | 0.0337 |
| 10 | 150 | 0.7418 | 0.7615 | 0.0197 |
| **None** | **20** | **0.7635** | 0.8775 | 0.1140 |
| None | 50 | 0.7561 | 0.8145 | 0.0584 |
| None | 150 | 0.7438 | 0.7683 | 0.0245 |

**Regressor grid** (chosen by OOF RMSE, `outputs/models/validation/x1c_regressor_tuning.json`):

| max_depth | min_samples_leaf | OOF RMSE | OOF R² | Train R² | Train &minus; OOF gap |
|---|---|---|---|---|---|
| 6 | 20 | 0.04967 | 0.4240 | 0.4616 | 0.0376 |
| 6 | 50 | 0.05037 | 0.4078 | 0.4376 | 0.0298 |
| 6 | 150 | 0.05165 | 0.3773 | 0.3924 | 0.0151 |
| 10 | 20 | 0.04788 | 0.4648 | 0.5513 | 0.0865 |
| 10 | 50 | 0.04893 | 0.4411 | 0.4862 | 0.0452 |
| 10 | 150 | 0.05121 | 0.3878 | 0.4053 | 0.0175 |
| **None** | **20** | **0.04757** | **0.4717** | 0.5746 | 0.1029 |
| None | 50 | 0.04884 | 0.4433 | 0.4922 | 0.0490 |
| None | 150 | 0.05118 | 0.3885 | 0.4065 | 0.0181 |

Both grids pick the same corner (`max_depth=None, min_samples_leaf=20`) -- the least-constrained
config in the grid, and also the one with the largest train-minus-OOF gap on both stages
(classifier 0.1140, regressor 0.1029). **This gap is real and reported plainly, not hidden**: the
winning config is genuinely overfitting the training folds more than the shallower alternatives.
But the selection rule used throughout this project -- pick by OOF/held-out score, not
training-fold fit -- is the correct one here specifically because it already penalizes overfit
configs on their own terms: every shallower alternative shows a smaller gap *and* a worse OOF
score, so constraining further would trade away real signal to chase a smaller (but not zero)
overfit gap, the same reasoning `c1d`'s own Rung 2 (`ACTIVE_CONTINUOUS_MODEL_LADDER.md` section
4.1) used when it picked `max_depth=8` over shallower or fully-unconstrained alternatives for the
same reason. The gap sizes here (0.10-0.11) sit in a similar range to `c1d`'s own chosen-config gap
(0.30) and well below `c1d`'s worst-in-grid gap (0.46) -- not an alarming level of overfitting by
this project's own established scale.

### 3.3 `x1c_random_forest` vs `x1_two_stage_huber` -- ranking (CV and held-out)

| Variant | RMSE (CV) | RMSE (held-out) | R² (CV) | R² (held-out) | Spearman (held-out) |
|---|---|---|---|---|---|
| `x1_two_stage_huber` | 0.05394 | 0.05309 | 0.1514 | 0.1519 | 0.5180 |
| **`x1c_random_forest`** | **0.04346** | **0.04277** | **0.4491** | **0.4495** | **0.5947** |

A large, consistent win on every metric -- RMSE down ~19.5% (CV) / ~19.4% (held-out), R² roughly
**tripling** (0.15 -> 0.45), Spearman rank correlation improving too (0.518 -> 0.595). CV and
held-out results are near-identical (R² 0.4491 vs 0.4495), no sign of the tuning-selected
overfitting gap (section 3.2) leaking into the final held-out readout.

**Zero-row / nonzero-row MAE, held-out**:

| Variant | Zero-row MAE | Nonzero-row MAE |
|---|---|---|
| `x1_two_stage_huber` | **0.00488** | 0.02469 |
| `x1c_random_forest` | 0.00678 | **0.02066** |

Same pattern Rung 0 already established for `x1` vs `x0`: `x1c`'s classifier stage occasionally
assigns real nonzero probability to rows that turn out to be exactly zero (a real, small cost,
zero-row MAE up 39% relative), but this is more than offset by a large nonzero-row MAE
improvement (16% lower) -- consistent with `x1c`'s much-improved P(nonzero) classifier (ROC-AUC
0.762 held-out vs `x1`'s 0.710) actually distinguishing rows better, which necessarily means
assigning more genuine probability mass to some rows that end up zero, not just to rows that end
up nonzero.

### 3.4 Paired significance test, `x1c` vs `x1`

Per-fold RMSE (`outputs/models/validation/significance_x1_vs_x1c_random_forest.json`), compared
against the **standing baseline `x1_two_stage_huber`, not `x1b_quadratic`** (which did not win
promotion in Rung 1): `x1c` beats `x1` in **5 of 5 folds** (0.0450 vs 0.0569, 0.0456 vs 0.0551,
0.0435 vs 0.0526, 0.0430 vs 0.0533, 0.0396 vs 0.0517), mean diff **-0.01054** (RMSE ~19.5% lower).
Paired t-test t=-17.48, **p=0.00006** -- clearly significant, and roughly two orders of magnitude
larger in effect size than Rung 1's own `x1b`-vs-`x1` result (mean diff -0.00004). Wilcoxon
signed-rank p=0.0625, the same 5-fold floor seen throughout this project.

### 3.5 Calibration -- does random forest's nonlinearity close the gap that quadratic terms couldn't?

| Bin | `x1` gap (Rung 0) | `x1b` gap (Rung 1) | `x1c` gap (this rung) |
|---|---|---|---|
| 0 (most negative predicted) | +0.02074 | +0.02113 | **+0.00194** |
| 1 | +0.00038 | -0.00019 | +0.00065 |
| 2 | +0.00009 | -0.00019 | -0.00004 |
| 3 | +0.00059 | +0.00081 | +0.00026 |
| 4 (most positive predicted) | -0.01713 | -0.01725 | **-0.00049** |

**Yes -- decisively, and this is the headline result of this rung.** Bin 0's gap shrinks from
+0.02074 (`x1`) / +0.02113 (`x1b`) to **+0.00194** -- roughly a **10-11x reduction**. Bin 4's gap
shrinks from -0.01713 / -0.01725 to **-0.00049** -- roughly a **35x reduction**. Every middle bin
also lands closer to zero than either linear-family variant. Where `x1b` (more curved features,
same Huber loss) left the tail gap unchanged to worse, `x1c` (same features, tree-ensemble
averaging instead of a robust linear loss) closes it almost entirely. This directly confirms
Rung 1's own diagnosis: the limitation was the *loss function's* structural cap on how far a
prediction could move toward an extreme observed value, not a lack of curvature in the features
themselves -- a random forest's leaf-averaging mechanism does not carry that same cap, and the
real curvature this leg's features do have (established with real bin evidence at Rung 1) is
exactly what the trees can now exploit.

### 3.6 Feature importance (Gini and fold-held-out permutation importance, regression head)

Top 10 permutation-importance features (`outputs/models/regression/x1c_random_forest.json`,
fold-held-out, nonzero-delta rows):

| Rank | Feature | Mean permutation importance |
|---|---|---|
| 1 | `phase_label_prev_event_box_defence` | +0.001550 |
| 2 | `distance_to_attacking_box` | +0.000686 |
| 3 | `event_type_Block` | +0.000435 |
| 4 | `phase_label_prev_event_high_press_proxy` | +0.000375 |
| 5 | `attacking_goal_centrality` | +0.000353 |
| 6 | `event_type_Pressure` | +0.000309 |
| 7 | `event_type_Ball Recovery` | +0.000289 |
| 8 | `nearest_attacker_distance` | +0.000164 |
| 9 | `defender_spread` | +0.000143 |
| 10 | `action_retained_defensive_team_control` | +0.000135 |

**Cross-check against Rung 1's own quadratic candidates** (`distance_to_attacking_box`,
`defender_attacker_gap_x`, `visible_defender_count`): `distance_to_attacking_box` (Rung 1's
strongest candidate, r=0.157) is confirmed here too -- RF ranks it #2 by permutation importance,
independent agreement that this feature carries real, exploitable signal. `defender_attacker_gap_x`
and `visible_defender_count` (Rung 1's other two candidates) do not appear in the top 10 here --
RF finds more value in `phase_label_prev_event` categories and `event_type` levels (categorical
context) than in those two specific numeric shapes, a genuinely different picture from Rung 1's
own numeric-only curvature check, not a contradiction of it (a permutation-importance ranking
across all 32+ one-hot columns is not the same comparison as a numeric-only correlation ranking).
`attacking_goal_centrality` (rank #5) and `defender_spread` (rank #9) both also appeared in the
active-continuous leg's own `c1d_random_forest` top-10 (`ACTIVE_CONTINUOUS_MODEL_LADDER.md`
section 4.3) -- not the same target or leg, but a recurring pattern of these two features carrying
real signal for shot/threat-adjacent targets on the active side of this project more broadly.

### 3.7 Ladder-gate outcome for this rung, no promotion audit

**`x1c_random_forest` clears the Rung-2 gate against `x1_two_stage_huber`, decisively.** A
consistent (5-of-5 folds), statistically significant (p=0.00006) improvement on RMSE, a near-tripling
of R² on both CV and held-out with no CV-to-held-out drop-off, and -- the result this whole ladder
has been tracking since Rung 0 -- a 10-35x reduction in the tail-calibration gap that neither Rung 0
nor Rung 1 could close. Per this prompt's explicit constraint, no deeper promotion audit
(tournament-stratified check, error-slice comparison, feature-shape sanity check -- the kind of
4-part deep dive `c1d`'s own later promotion prompt, Prompt 58, ran) is performed here; this
section reports the Rung-2 ladder-gate outcome only, the same discipline `c1d_random_forest`'s own
Rung-2 prompt (56) followed before its own separate promotion-audit prompt (58).

**`x1c_random_forest` is now the leg's standing baseline**, replacing `x1_two_stage_huber`.
`x1_two_stage_huber` remains documented as the Rung-0 baseline it was measured against, and
`x1b_quadratic` remains documented as the Rung-1 rung that didn't clear its own bar. No further
rung (gradient boosting or otherwise) is built in this prompt.

## 4. Rung 3 -- `x1d_gradient_boosting` (prompt 74)

`x1c_random_forest` (Rung 2) is this leg's first real win. That result is the reason to test
gradient boosting directly, mirroring the active-binary leg's own Rung 4
(`v1e_gradient_boosting`, prompt 45) and the active-continuous leg's own Rung 3
(`c1e_gradient_boosting`, prompt 57) -- a different tree-ensemble method worth checking once one
tree method already found real structure, not assumed to help just because RF did.

### 4.1 Step 1 -- confirm x1c's actual architecture before deciding what x1d replaces

Checked directly against `outputs/models/regression/x1c_random_forest.json` and this document's
own section 3.1/3.7 before assuming anything: `x1c_random_forest` replaced **both** of `x1`'s
surfaces with RandomForest -- `RandomForestClassifier` for P(nonzero) and
`RandomForestRegressor` for E[delta|nonzero], each independently tuned (the JSON's `clf_params`
and `reg_params` fields are both populated). Section 3.1's own reasoning for why (both surfaces
are this leg's own models, built together, with no asymmetric evidence favoring one over the
other) has not been contradicted by anything since Rung 2. **Decision: `x1d` replaces both
stages with gradient boosting too** -- `LGBMClassifier` for P(nonzero), `LGBMRegressor` for
E[delta|nonzero], mirroring `x1c`'s own two-independent-models composition, gated against `x1c`
specifically (not the older linear rungs) per this prompt's own instruction.

### 4.2 Library and grid, sized for this leg's real row counts

**LightGBM**, matching both `v1e_gradient_boosting` (active-binary) and `c1e_gradient_boosting`
(active-continuous) -- confirmed available in this environment (version 4.7.0), kept consistent
rather than introducing XGBoost as a second boosting library for no reason.

Row counts confirmed directly from this leg's own join (not assumed to match either precedent
leg): classifier trains on 45,166 trainval rows (~36k/fold in 5-fold CV); regressor trains on
36,121 nonzero-only trainval rows (~29k/fold). This sits between `c1e`'s tiny sample (~2,880
rows/fold, trimmed grid) and `v1e`'s full active-binary scale (~36,000 rows/fold) -- close enough
to `v1e`'s own scale to reuse its `NUM_LEAVES_GRID=[15,31,63]` unchanged, but
`MIN_CHILD_SAMPLES_GRID` is kept at `[20,50,150]` to stay internally consistent with `x1c`'s own
already-established `MIN_SAMPLES_LEAF_GRID` for this exact leg at this exact row scale (Prompt
73's own reasoning), rather than reusing `v1e`'s `[10,30,100]` by default.
`LEARNING_RATE_GRID=[0.01,0.05,0.1]` unchanged from both precedents. `n_estimators` is never
grid-searched directly -- chosen per fit via early stopping (cap 2,000 rounds, 50-round patience)
on a match-grouped validation carve-out of that fit's own training rows, exactly
`v1e`/`c1e`'s own fitting procedure. 27 combinations x 5 folds = 135 fits per stage, 270 total.

### 4.3 Hyperparameter grids and CV-internal overfitting diagnostic

**Classifier grid** (chosen by OOF ROC-AUC; full 27-row grid in
`outputs/models/validation/x1d_classifier_tuning.json`) -- best and worst-gap rows:

| learning_rate | num_leaves | min_child_samples | OOF ROC-AUC | Train ROC-AUC | Train &minus; OOF gap | best_iteration |
|---|---|---|---|---|---|---|
| **0.05** | **15** | **50** | **0.7766** | 0.8523 | 0.0758 | 477 |
| 0.01 | 15 | 20 | 0.7717 | 0.8222 | 0.0505 (smallest gap in grid) | 1355 |
| 0.05 | 63 | 50 | 0.7735 | 0.8891 | 0.1156 (largest gap in grid) | 170 |

The chosen config's gap (0.0758) is smaller than `x1c`'s own classifier gap (0.1140) -- gradient
boosting's early stopping is doing real work constraining this stage, more so than RF's grid
search did at Rung 2.

**Regressor grid** (chosen by OOF RMSE; full 27-row grid in
`outputs/models/validation/x1d_regressor_tuning.json`) -- best and worst-gap rows:

| learning_rate | num_leaves | min_child_samples | OOF RMSE | OOF R² | Train R² | Train &minus; OOF gap | best_iteration |
|---|---|---|---|---|---|---|---|
| **0.01** | **63** | **20** | **0.04696** | **0.4853** | 0.6540 | 0.1687 | 531 |
| 0.01 | 15 | 150 | 0.04791 | 0.4641 | 0.5337 | 0.0696 (smallest gap in grid) | 1380 |
| 0.1 | 63 | 20 | 0.04708 | 0.4825 | 0.6700 | 0.1874 (largest gap in grid) | 64 |

**Reported plainly, not hidden**: the regressor's chosen config has a *larger* train-minus-OOF
gap (0.1687) than `x1c`'s own regressor gap (0.1029) -- gradient boosting's sequential,
error-correcting structure is fitting the training folds more readily here than RF's
independent-tree averaging did, the same pattern `c1e`'s own docstring flagged on the
active-continuous leg ("gradient boosting's sequential, error-correcting structure fits training-
fold noise more readily than RF's independent-tree averaging"). The selection rule used
throughout this leg -- pick by OOF score, not training-fold fit -- still applies and still
defends itself on its own terms: the smallest-gap config (0.0696) has a meaningfully worse OOF
RMSE (0.04791 vs 0.04696), so constraining further would trade away real signal to chase a
smaller gap. This is reported as a genuine leg-level pattern to watch, not dismissed.

### 4.4 x1d_gradient_boosting vs x1c_random_forest -- ranking (CV and held-out)

| Variant | RMSE (CV) | RMSE (held-out) | R² (CV) | R² (held-out) | Spearman (held-out) |
|---|---|---|---|---|---|
| `x1c_random_forest` | 0.04346 | 0.04277 | 0.4491 | 0.4495 | **0.5947** |
| `x1d_gradient_boosting` | **0.04285** | **0.04205** | **0.4647** | **0.4680** | 0.5692 |

`x1d` beats `x1c` on RMSE (CV -1.4%, held-out -1.7%) and R² (+0.016 CV, +0.018 held-out) -- real,
but a fraction of the size of `x1c`'s own edge over `x1` at Rung 2 (RMSE -19.5%, R² nearly
tripled). **Spearman rank correlation moves the other way**: 0.5947 -> 0.5692, a real drop
(-0.0255), consistent on both CV (0.5935 -> 0.5641) and held-out.

| Variant | Zero-row MAE | Nonzero-row MAE |
|---|---|---|
| `x1c_random_forest` | 0.00678 | 0.02066 |
| `x1d_gradient_boosting` | **0.00587** | **0.02060** |

Both MAE slices also favor `x1d` -- lower on every accuracy metric measured in absolute-error
terms. The pattern across sections 4.4-4.5 is consistent: **magnitude-accuracy metrics (RMSE,
MAE, R²) favor `x1d`; shape/rank/tail metrics (Spearman, calibration) favor `x1c`.**

### 4.5 Paired significance test, x1d vs x1c

Per-fold RMSE (`outputs/models/validation/significance_x1d_vs_x1c.json`), compared against the
**current best (`x1c_random_forest`), not the older linear rungs**: `x1d` beats `x1c` in **5 of 5
folds** (0.0443 vs 0.0450, 0.0451 vs 0.0456, 0.0429 vs 0.0435, 0.0421 vs 0.0430, 0.0393 vs
0.0396), mean diff **-0.00061**. Paired t-test t=-6.39, **p=0.0031** -- statistically
significant, but the effect size is roughly **one-seventeenth** the size of `x1c`'s own edge over
`x1` (mean diff -0.01054 at Rung 2). Wilcoxon p=0.0625, the same 5-fold floor seen throughout
this project.

### 4.6 Calibration -- does x1d hold onto x1c's tail-calibration win?

| Bin | x1 gap (Rung 0) | x1b gap (Rung 1) | x1c gap (Rung 2) | x1d gap (this rung) |
|---|---|---|---|---|
| 0 (most negative predicted) | +0.02074 | +0.02113 | **+0.00194** | +0.00369 |
| 1 | +0.00038 | -0.00019 | +0.00065 | +0.00094 |
| 2 | +0.00009 | -0.00019 | -0.00004 | +0.00004 |
| 3 | +0.00059 | +0.00081 | +0.00026 | -0.00067 |
| 4 (most positive predicted) | -0.01713 | -0.01725 | **-0.00049** | -0.00233 |

**No -- x1d gives back a real share of x1c's own tail-calibration win, though it stays far ahead
of the pre-RF rungs.** Bin 0's gap widens from +0.00194 (`x1c`) to +0.00369 (`x1d`) -- roughly
**1.9x larger**. Bin 4's gap widens from -0.00049 to -0.00233 -- roughly **4.8x larger**. Both
remain a fraction of `x1`/`x1b`'s own gaps (bin 0: 0.0207/0.0211; bin 4: -0.0171/-0.0173), so
gradient boosting has not undone Rung 2's headline finding entirely -- but it has not matched or
improved on it either, which is the actual question this section exists to answer, per this
prompt's own explicit instruction to keep tracking this diagnostic rather than let an
aggregate-metric win stand in for it.

### 4.7 Feature importance: cross-check against x1c

Top 15 by gain importance and by fold-held-out permutation importance (regression head,
`outputs/models/regression/x1d_gradient_boosting.json`):

| Rank | Gain importance | Permutation importance (fold held-out) |
|---|---|---|
| 1 | `distance_to_attacking_box` (4031.0) | `phase_label_prev_event_box_defence` (0.001482) |
| 2 | `attacking_goal_centrality` (2845.0) | `distance_to_attacking_box` (0.000776) |
| 3 | `nearest_attacker_distance` (2802.0) | `event_type_Block` (0.000417) |
| 4 | `defender_spread` (2283.0) | `attacking_goal_centrality` (0.000388) |
| 5 | `angle_to_attacking_goal` (2193.0) | `action_retained_defensive_team_control` (0.000240) |
| 6 | `defender_attacker_gap_x` (1707.0) | `phase_label_prev_event_high_press_proxy` (0.000230) |
| 7 | `attacker_spread` (1469.0) | `event_type_Pressure` (0.000218) |
| 8 | `attacker_defender_ratio` (1378.0) | `event_type_Ball Recovery` (0.000211) |
| 9 | `possession_elapsed_seconds` (1306.0) | `nearest_attacker_distance` (0.000167) |
| 10 | `match_time_seconds` (1204.0) | `defender_spread` (0.000140) |

**Strong agreement with `x1c`'s own permutation-importance ranking (section 3.6)**:
`phase_label_prev_event_box_defence` is #1 by permutation importance on *both* models,
`distance_to_attacking_box` is #2 on both, `attacking_goal_centrality` and `defender_spread`
appear in both top-10s. Two independently-trained tree-ensemble methods (bagged vs. boosted)
converging on the same top features is stronger evidence than either alone that these are real
drivers, not an artefact of one model family's own inductive bias -- the same cross-model
agreement pattern `c1e`/`c1d` showed on the active-continuous leg (section 3.6/5.4 of that leg's
own ladder doc). `distance_to_attacking_box`'s #1 rank by gain importance (well ahead of anything
else, 4031 vs 2845 for #2) is new information gain importance alone can show that permutation
importance's #2 rank does not emphasize as strongly -- consistent with this feature carrying real,
substantial signal by both measures, just weighted differently by each importance method's own
convention.

### 4.8 Promotion call: a genuine mixed result, not a clean win either direction

**`x1d_gradient_boosting` does not clearly replace `x1c_random_forest` as the standing
baseline**, and the reverse is also not a clean call -- this is reported as the genuinely mixed
result it is, per this prompt's own explicit instruction not to frame a small edge as a win while
ignoring a real cost elsewhere.

**In `x1d`'s favor**: a statistically significant (p=0.0031), 5-of-5-fold-consistent improvement
on RMSE, and a real (if modest) gain on R², MAE, and both zero-/nonzero-row MAE slices. The
classifier stage in isolation is also meaningfully better (held-out ROC-AUC 0.778 vs `x1c`'s
0.762), and feature importance agrees strongly with `x1c`'s own ranking, adding confidence that
both models are finding the same real signal, not different noise.

**Against `x1d`**: Spearman rank correlation drops measurably (-0.0255 held-out, consistent on
CV), and -- the specific diagnostic this whole ladder has tracked since Rung 0 -- the
tail-calibration gap `x1c` closed by 10-35x widens back out by roughly 2-5x (still far better
than the pre-RF rungs, but a real regression relative to the current best, not noise). The
effect size of `x1d`'s own RMSE win (mean diff -0.00061) is about one-seventeenth the size of
`x1c`'s own edge over `x1` (mean diff -0.01054) -- a real signal, but a small one, set against a
real cost on two different diagnostics this leg has specifically prioritized.

**This document's own standing evaluation philosophy (section 1) is the deciding factor**: "an
aggregate metric improving without [the tail-calibration] gap closing would be an incomplete
answer, not a full one." Here the aggregate metric improves *and* the tail-calibration gap
widens -- the harder case that principle was written to cover, not a hypothetical. Weighing a
small, real RMSE win against a real regression on the two diagnostics (rank correlation, tail
calibration) this leg has consistently treated as first-class evidence, not just a footnote:
**`x1c_random_forest` remains the leg's standing baseline.** `x1d_gradient_boosting` is
documented in full above as a legitimate, closely-contested rung that does not win, the same
honest-mixed-result discipline `c1e_gradient_boosting` modeled on the active-continuous leg
(section 5.5 of that leg's own ladder doc) when it also failed to clear its own leg's
random-forest bar. No promotion audit is run for either candidate in this prompt, per this
prompt's own explicit constraint.
