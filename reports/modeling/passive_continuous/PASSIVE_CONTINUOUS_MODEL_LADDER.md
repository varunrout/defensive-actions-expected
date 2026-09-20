# Passive-Continuous Model Ladder

*Split out of `PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md` (prompt 62, once a second rung existed beyond
Rung 0) so the locked Rung 0 baseline document doesn't keep growing as more rungs are tried --
mirroring the active-continuous leg's own split point (`ACTIVE_CONTINUOUS_MODEL_LADDER.md`, prompt
56, 2 rungs beyond Rung 0). Rung 0 -- the two locked variants (`d0_dummy`, `d1_lognormal_glm`), their
CV/held-out results, the hurdle pipeline readout, and the Prompt 60 data-quality correction -- stays
in [`PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md`](PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md). This document
is everything built on top of that locked Rung 0 baseline, starting from Rung 1.*

## 1. What a rung is

A **rung** is one controlled change to the locked Rung 0 baseline (`d1_lognormal_glm`): a single
variable changed -- feature shape, an interaction term, a model family -- with everything else held
identical (same 38 locked features unless the rung specifically changes that, same canonical
match-grouped 5-fold split, same `log(xg)` target on the same shot-positive rows, all defender-slot
rows per the standing row-duplication decision, no ad-hoc tuning). A rung is tested against 2 fixed
gates before it can be considered for adoption as the new baseline:

1. **Paired significance test vs the current best** -- refit both variants on the same 5 canonical CV
   folds, paired t-test + Wilcoxon signed-rank test on the per-fold common-scale log RMSE difference.
2. **Held-out-test confirmation** -- a single, one-time readout on the frozen held-out test set
   (17,402 positive rows / 1,979 unique events). This is the gate that actually decides the rung's
   fate: CV picks a candidate, the held-out test either confirms or rejects it.

Given this leg's row-duplication structure, every significance test on this leg reports fold-level
unique-event counts alongside row counts -- the effective sample size is closer to the event count
than the row count, and that distinction is checked explicitly each time, not assumed away by the
large row counts.

Rung 0 (the baseline itself) lives in
[`PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md`](PASSIVE_CONTINUOUS_BASELINE_SUMMARY.md). This ladder
builds on that document's `d1_lognormal_glm` pick (held-out common-scale log RMSE 1.0374, corrected
R² 0.0078) as the baseline every rung is compared against.

## 2. Rung 1 -- `d1b_quadratic` (prompt 61)

### 2.1 Reconfirming shot-conditional nonlinearity from real data, systematically

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

### 2.2 `d1b_quadratic` results

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

### 2.3 Significance test, d1b vs d1

`outputs/models/validation/significance_d1_vs_d1b_quadratic.json`: mean diff (d1b - d1) common-scale
log RMSE = **+0.0000** (indistinguishable), paired t-test **p=0.5013**, Wilcoxon p=0.625. **Does not
clear significance in either direction, and by a wide margin** -- this is an even more emphatic null
result than the active leg's own Rung 1 (p=0.2073 there; p=0.50 here). Per-fold unique-event counts
(minimum 1,512, same as Rung 0) confirm this is not a thin-sample artifact -- the effective sample
size is large enough that a real effect this size would very likely have shown up.

### 2.4 This is a real finding about this leg, not a modelling failure

Stated plainly: **`d1b_quadratic` shows no real improvement over `d1`, and this connects directly to
section 2.1's already-documented weak-correlation finding, not a coincidence.** This leg's
shot-conditional correlations were already markedly weaker than the active leg's own Rung 1 starting
point before any quadratic term was tried (strongest |r|=0.065 here vs |r|=0.16 there) -- one weak
feature's squared term, chosen precisely because it was the *only* feature showing genuine curvature
at all, simply doesn't have enough underlying signal to move a fair, common-scale metric measurably.
This is not a surprise; it is the expected continuation of a pattern already visible in item 1's own
numbers before the model was ever fit.

**`d1_lognormal_glm` remains this leg's best Rung-0/Rung-1 candidate**; `d1b_quadratic` is not
promoted or preferred over it.

## 3. Rung 2 -- `d1c_systematic_interactions` is SKIPPED (prompt 62)

The rung that would mirror the active-continuous leg's own skipped `c1c` (Prompt 56) is **explicitly
skipped, not forgotten**, on a rough chat check run directly against real data before this rung was
built, mirroring the diligence Prompt 56 applied to the active leg -- more decisively here:

1. **Plain additive linear regression** (28 numeric PASSIVE features, standardized, 5-fold grouped
   CV, positive rows only) scores mean CV **R² = -0.0042** -- already negative, i.e. this simple
   linear model is doing no better than predicting the training mean on held-out folds.
2. **Full pairwise-interaction + squared-term expansion**
   (`PolynomialFeatures(degree=2)`, 406 columns from 28 features), Ridge-regularized at alpha=1 and
   alpha=100, scores mean CV **R² of -0.0249 and -0.0230** respectively -- **worse** than the plain
   linear model, not just a smaller gain than noise (the active leg's own finding). Lighter
   regularization makes it actively worse, same direction as the active leg's own check, but the
   interaction expansion never even matches the (already weak) additive baseline here.

**Decision: `d1c_systematic_interactions` is not built.** Both the plain linear model and every
tested interaction-expansion variant score negative CV R² on this leg's data -- there is no
additive-linear or interaction-expansion signal here worth systematizing. This is a closed decision
for this leg, not deferred; it would only be revisited if new evidence specifically contradicted the
numbers above.

## 4. Rung 2 -- `d1d_random_forest` (prompt 62)

Instead of the interaction rung, Rung 2 goes straight to a `RandomForestRegressor` on the same 38
locked features, raw (non-standardized, non-polynomial-expanded) design matrix -- mirroring the
active-continuous leg's own Rung 2 (`c1d_random_forest`, Prompt 56). The question: can automatic,
sample-efficient nonlinearity/interaction discovery find signal that neither the hand-built quadratic
term (Rung 1) nor the brute-force polynomial expansion (section 3 above) could?

### 4.1 Hyperparameter grid, sized for this leg's real (but duplicated) scale

This leg's per-fold **row** count (~13,200-18,100) is far larger than the active leg's own Rung 2
grid target (~2,880 rows/training-fold), but the effective sample size -- unique **events**,
~1,500-2,000/fold -- is only about 2x the active leg's, not ~5x. The grid used here:
`N_ESTIMATORS_GRID=[200]`, `MAX_DEPTH_GRID=[6, 10, None]`, `MIN_SAMPLES_LEAF_GRID=[30, 100, 300]` --
`min_samples_leaf`'s floor is set well above the active leg's own `[10,30,60]` range specifically
because a leaf of even 60 raw rows here could still be as few as ~7 unique events (rows/event ~8.9),
thinner than it looks from the row count alone.

**CV-internal overfitting is severe at the grid's unconstrained end, and it tracks the event count,
not the row count.** Full grid (chosen by lowest OOF common-scale log RMSE):

| max_depth | min_samples_leaf | OOF common log RMSE | OOF corrected R² | mean train R² | train &minus; OOF gap |
|---|---|---|---|---|---|
| 6 | 30 | 1.0335 | 0.0002 | 0.1369 | 0.1368 |
| 6 | 100 | 1.0324 | 0.0039 | 0.1230 | 0.1191 |
| 6 | 300 | 1.0323 | 0.0069 | 0.0981 | 0.0912 |
| 10 | 30 | 1.0383 | -0.0161 | 0.3818 | 0.3979 |
| 10 | 100 | 1.0339 | -0.0021 | 0.2928 | 0.2949 |
| **10** | **300** | **1.0319** | **0.0063** | 0.1839 | 0.1776 |
| None | 30 | 1.0471 | -0.0577 | **0.8360** | **0.8936** |
| None | 100 | 1.0378 | -0.0149 | 0.4725 | 0.4873 |
| None | 300 | 1.0323 | 0.0053 | 0.2184 | 0.2131 |

`max_depth=None, min_samples_leaf=30` memorizes the training folds almost completely (train R²=0.84)
while scoring *negative* OOF R² (-0.058) -- an 0.89 train-minus-OOF gap, by far the worst overfitting
seen on either continuous leg's own Rung 2 grid, and a direct illustration of the event-vs-row
distinction: with only ~30 raw rows required per leaf (as few as ~3-4 unique events), the deepest,
least-constrained trees are trivially able to memorize individual events' duplicated rows rather than
learning anything general. The chosen config (`max_depth=10, min_samples_leaf=300`) still shows a
real gap (train R²=0.18 vs OOF R²=0.006) but wins cleanly on OOF score, and every config with
`min_samples_leaf=300` (roughly 33+ unique events per leaf at this leg's ~8.9 rows/event ratio)
clusters together with the smallest gaps in the grid -- exactly where the event-count reasoning
predicts the safer configs should land.

### 4.2 `d1d_random_forest` vs `d1_lognormal_glm` -- ranking and calibration, reported separately

**Ranking (common-scale log RMSE) -- `d1d_random_forest` wins directionally on every fold but one,
without clearing significance:**

| Variant | Common log RMSE (CV) | Common log RMSE (held-out) |
|---|---|---|
| `d1_lognormal_glm` | 1.0345 | 1.0374 |
| `d1d_random_forest` | 1.0319 | 1.0322 |

Paired significance test (`outputs/models/validation/significance_d1_vs_d1d_random_forest.json`):
mean diff (d1d &minus; d1) = **-0.0025** (d1d better), paired t-test **p=0.5760**, Wilcoxon
**p=0.4375**. `d1d_random_forest`'s common-scale log RMSE is lower than `d1`'s on **4 of 5 CV folds**
(fold 4 is the exception, 1.0092 vs 0.9979) -- a real, consistent-*direction* edge, but small and not
statistically significant given this leg's effective (event-level) sample size. Per-fold unique-event
counts range 1,512-2,039 (minimum 1,512) -- the same standing caveat as every other test on this leg:
large row counts, more modest effective sample size, checked explicitly rather than assumed away.

**Calibration and original-scale accuracy (held-out test) -- also favors `d1d_random_forest`, more
visibly than the ranking test alone suggests:**

| Variant | Naive MAE | Naive R² | Corrected MAE | Corrected R² |
|---|---|---|---|---|
| `d1_lognormal_glm` | 0.0718 | -0.0869 | 0.0839 | 0.0078 |
| `d1d_random_forest` | 0.0711 | -0.0731 | 0.0797 | **0.0211** |

Corrected R² nearly triples (0.0078 -> 0.0211) and corrected MAE improves meaningfully (0.0839 ->
0.0797). **Ranking and calibration agree in direction here** -- both favor `d1d_random_forest` -- but
the ranking test's own significance gate does not clear, so this is reported as a real but modest,
statistically inconclusive improvement, not a confirmed win.

### 4.3 Feature importance: does RF find what the linear methods couldn't?

Top 10 (of 15 reported) features by Gini importance and by fold-held-out permutation importance
(`outputs/models/regression/d1d_random_forest.json`):

| Rank | Gini importance | Permutation importance (fold held-out) |
|---|---|---|
| 1 | `ball_x` (0.1821) | `ball_x` (0.0370) |
| 2 | `ball_y` (0.0753) | `top_option_3_dx` (0.0066) |
| 3 | `top_option_3_distance_from_ball` (0.0701) | `ball_y` (0.0058) |
| 4 | `top_option_1_dx` (0.0644) | `top_option_3_distance_from_ball` (0.0049) |
| 5 | `top_option_1_threat_score` (0.0612) | `on_ball_event_type_Carry` (0.0045) |
| 6 | `top_option_3_threat_score` (0.0611) | `defender_x` (0.0034) |
| 7 | `top_option_3_dx` (0.0585) | `top_option_1_threat_score` (0.0030) |
| 8 | `top_option_2_distance_from_ball` (0.0525) | `top_option_3_threat_score` (0.0023) |
| 9 | `top_option_1_angle_from_ball` (0.0458) | `top_option_1_angle_from_ball` (0.0023) |
| 10 | `top_option_1_distance_from_ball` (0.0437) | `top_option_1_dx` (0.0022) |

**`ball_x` is the clear headline finding, exactly the "RF found something linear missed" pattern the
active leg's own Rung 2 showed with `distance_to_attacking_box`.** By linear correlation,
`ball_x` ranks only **#15 of 38** (|r|=0.0184, `shot_conditional_nonlinearity_check_passive.json`) --
weak enough that it wasn't remotely close to a Rung 1 quadratic candidate. By RF, it is the single
most important feature by both Gini (0.182, more than 2x the next-highest) and permutation importance
(0.037, also far ahead of #2). The tree found real structure in the ball's absolute pitch position
that a linear/Pearson-Spearman view on this leg's own data could not surface. Two more features with
the same pattern: `top_option_3_distance_from_ball` (correlation rank #6, RF Gini rank #3) and
`top_option_1_dx` (correlation rank #18, RF Gini rank #4) both rank meaningfully higher under RF than
under linear correlation.

**Rung 1's own quadratic candidate, `engagement_distance_to_carrier`, is NOT confirmed by RF --
ranked near the bottom of all 54 encoded features.** Gini rank **39th of 54** (importance 0.000044,
essentially zero) and permutation rank **25th of 54** (importance 0.0000113, also essentially zero),
despite being the 4th-strongest feature by linear correlation and the one feature Rung 1 judged
worth a quadratic term. RF does not see this feature as informative at all once it has access to the
full 38-feature raw design matrix -- consistent with, and reinforcing, Rung 1's own null result: the
weak curvature signal `engagement_distance_to_carrier` showed in isolation does not survive contact
with a model that can also draw on everything else.

### 4.4 Ladder-gate outcome for this rung, no promotion audit

**`d1d_random_forest` does not clear significance against `d1_lognormal_glm`** (p=0.576), the third
independent check on this leg to come back this way: Rung 1's quadratic term (p=0.50), the skipped
`d1c` interaction expansion (negative CV R² even before regularization is loosened), and now Rung 2's
random forest (a real, 4-of-5-fold directional edge and a near-tripling of corrected R², but not
statistically significant given this leg's effective event-level sample size). **This convergence --
three structurally different approaches, none clearing a real bar -- is itself the headline finding
for this leg, not a footnote**: this leg's per-defender features carry meaningfully less
shot-conditional signal than the active leg's aggregate whole-defense features do, and that ceiling
has now been checked from three independent angles rather than assumed from one. Per this prompt's
explicit constraint, no promotion audit is run in this prompt regardless of this result --
`d1_lognormal_glm` remains this leg's best regression candidate, and this section reports the Rung-2
gate outcome only.
