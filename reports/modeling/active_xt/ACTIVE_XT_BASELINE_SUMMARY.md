# Active-xT Baseline Summary (x0-x1)

Target: `target_xt_delta_v2` (active-binary leg, Prompts 64/66/67, fully EDA'd and locked against
the 34 locked ACTIVE features per `reports/analysis/xt_target/FEATURE_LOCK_CONFIRMATION_XT.json`).
Dataset: `data/features/player_defensive_actions.parquet` (56,068 rows, read-only, unmodified)
left-joined in memory, by `event_id`, with `outputs/prototypes/active_binary_xt_delta_v2.parquet`
(also read-only, unmodified) -- neither locked file is touched by this leg's modelling work. This
is this leg's Rung 0: one dummy floor (`x0_dummy`) and one real candidate
(`x1_two_stage_huber`), mirroring every other leg's own Rung 0 discipline (Prompts 36/49/54/59).

**Prompt scope**: active-binary leg only. Passive-xT's own Rung 0 is a separate, later prompt.
This document does not build past Rung 0 -- no gradient boosting, no promotion decision, no full
ladder.

## 1. Architecture decision, with evidence -- the load-bearing question this prompt answers

`target_xt_delta_v2` does not match either architecture this project has used so far.

**Not the lognormal hurdle used by `c1d`/`d1`.** That architecture needs a zero/positive-mass
split feeding a *strictly-positive* continuous regression head. `target_xt_delta_v2` is signed:
reconfirmed directly against the joined data, 35.4%/44.4%/20.2% negative/positive/zero per
`XT_TARGET_PROTOTYPE_V2.md`, and this script's own load step reconfirms 11,238 of 55,761 defined
rows (20.15%) are **exactly** zero. A log transform of a signed quantity does not exist, so a
lognormal head is not an option here, full stop -- not a judgement call.

**Not an ordinary single-distribution continuous target either, prima facie.** The 20.15%
exact-zero mass is structural, not noise: Prompt 66 established that `xt_after` is forced to
`0.0` whenever an action ends its own possession, so `target_xt_delta_v2 = xt_before - 0 =
xt_before` for that entire slice, by construction. Fitting one plain regression across all rows
risks the model learning to predict close to the grand mean everywhere and missing that these are
two genuinely different regimes (denied-to-zero vs. continued-possession) -- but only if that
zero/nonzero split is actually *predictable* from the features a model gets to see. That is the
empirical question this section had to settle before choosing an architecture, not something to
assume either way.

**Checked directly, not re-derived from scratch, per the prompt's own instruction** -- against
`reports/analysis/xt_target/active_category_atlas.json` and `active_flag_ledger.json`:

| Locked feature | Split | pct exactly zero |
|---|---|---|
| `event_type` (categorical) | Ball Recovery | 5.70% |
| `event_type` | Interception | 9.13% |
| `event_type` | Block | 11.97% |
| `event_type` | Clearance | 6.98% |
| `event_type` | 50/50 | 19.29% |
| `event_type` | Duel | 25.50% |
| `event_type` | Pressure | 26.78% |
| `event_type` | Foul Committed | 27.94% |
| `action_retained_defensive_team_control` (boolean) | True | 24.19% |
| `action_retained_defensive_team_control` | False | 10.63% |
| `is_in_attacking_box` (boolean) | True | 12.56% |
| `is_in_attacking_box` | False | 21.04% |
| `is_in_defending_box` (boolean) | True | 13.84% |
| `is_in_defending_box` | False | 20.75% |

`event_type` alone spans a ~5x range (5.70% to 27.94%) in zero-rate, and it is a **locked
modelling feature**, not one of the three off-pool possession flags
(`action_ended_possession`/`action_changed_possession`/`action_won_possession`) that
mechanically *trigger* the zero -- those remain excluded from `feature_config.py` (reconfirmed
still excluded) and are not used anywhere in this script either, since including them would make
a classifier stage trivially perfect by construction rather than a genuine prediction from
football-shape features. The boolean flags above show the same pattern: real, several-fold
variation in zero-rate from features genuinely available to a model at prediction time.

**Decision: two-stage architecture.** Stage A -- a plain `LogisticRegression` classifying
"this row's delta is nonzero" from the 32 locked features (feature set, below). Stage B -- a
`HuberRegressor` (robust to this leg's heavy tails -- excess kurtosis 18.33 -- and explicitly
**not** lognormal, per the prompt's own constraint) fit only on the nonzero-delta rows,
predicting the signed delta directly with no transform. Combined:
`E[delta] = P(nonzero) * E[delta | nonzero]`, since `E[delta | zero] = 0` exactly by
construction -- the zero here is the row's real target value, not a missing observation, so
there is nothing to add back for the zero branch. `HuberRegressor`, `LogisticRegression`, and
quantile regression were all named as acceptable Rung-0 candidates by the prompt; `Huber` was
chosen over plain `LinearRegression` specifically because of this target's heavy tails (a plain
least-squares fit would let the small number of large-magnitude rows, e.g. the -0.3576 carry
noted in `PASSIVE_XT_TARGET_PROTOTYPE.md`'s spot-check analogue on this leg, dominate the fit);
gradient boosting and random forest regression are explicitly out of scope for Rung 0 and saved
for a later rung.

**Naming.** `v` (active-binary shot target), `c` (active-continuous xg), `p` (passive-binary),
`d` (passive-continuous) are all already claimed -- confirmed directly against
`scripts/models/`, `reports/modeling/`, and `outputs/models/`. `x` (for xT) is unclaimed --
checked directly: no existing script, report, or output file anywhere in those three locations
uses an `x0`/`x1`-prefixed variant name. Used from Rung 0 onward: `x0_dummy`,
`x1_two_stage_huber`.

## 2. Confirmed inputs

**Join.** `player_defensive_actions.parquet` (56,068 rows, one row per `event_id`, confirmed no
duplicates) left-joined in memory against `active_binary_xt_delta_v2.parquet` (also one row per
`event_id`, confirmed no duplicates) by `event_id`. Row count unchanged by the join (56,068 in,
56,068 out) -- 307 rows have `target_xt_delta_v2 == NaN` (Prompt 66/67's own count, reconfirmed
here) and are dropped, leaving **55,761 rows** for this leg's modelling.

**Feature set.** The same 32 locked ACTIVE features `v1`/`c1` already use: 34 locked ACTIVE
features (`src/eda/feature_config.py`, target-agnostic) minus `nearest_defender_distance`
(self-reference data bug, ~73.6% of rows approx 0m, a data-quality issue independent of which
target is being predicted) and `defenders_within_5m` (structurally nested inside
`defenders_within_10m`, also target-agnostic). Reused directly from
`train_active_binary_baseline.py`'s own `DesignMatrixBuilder` and column lists -- not
reimplemented.

**Split.** The canonical, frozen match-grouped split in `outputs/models/splits/match_assignment.json`
(5-fold, seed 42), loaded via `dax.models.splits`, never rebuilt -- the same split every other
leg uses, confirmed by importing the loader directly rather than recomputing anything.

| | Rows | Matches |
|---|---|---|
| Train+val | 45,166 | 92 |
| Held-out test | 10,595 | 23 |

Nonzero-delta rows per canonical CV fold: fold 0 = 6,819, fold 1 = 8,085, fold 2 = 6,921, fold 3
= 7,374, fold 4 = 6,922 -- every fold well above the thin-fold floors used elsewhere in this
project.

**Metric-convention flag.** `dax.models.evaluation.regression_metrics` and
`dax.models.two_part_xg.compute_hurdle_metrics` both hardcode their zero/nonzero split as
`y_true > 0` -- correct for a non-negative xg target, **wrong** for this leg's signed target (a
genuine negative row would be miscounted into the "zero" bucket). Neither helper is reused for
the zero/nonzero breakdown here; this script computes its own split on `y_true == 0` /
`y_true != 0`, and overall MAE/RMSE/R²/Spearman are computed directly rather than through either
helper, to avoid mixing a correct overall number with an incorrect zero-split number from the
same call.

## 3. Rung 0 variants

| Variant | Model | Notes |
|---|---|---|
| `x0_dummy` | `ConstantRegressor(stat="mean")` on the raw signed `target_xt_delta_v2` | training-fold grand mean predicted for every row, no transform (the target is already signed and reasonably scaled) |
| `x1_two_stage_huber` | `LogisticRegression` (P(nonzero)) x `HuberRegressor` (E[delta\|nonzero]) | both stages use the same 32 locked features; the regression head is fit only on nonzero-delta training rows, mirroring `c1`/`d1`'s own convention of fitting only on the target-relevant row subset |

Unlike `c0`/`c1` or `d0`/`d1`, neither `x0` nor `x1` uses a log transform, so there is no
native-vs-common-scale mismatch to correct for here -- both variants' RMSE/MAE numbers are
already on the same, directly comparable scale.

## 4. Cross-validation results (5-fold, OOF metrics)

| Variant | RMSE | MAE | R² | Spearman | Zero-row MAE | Nonzero-row MAE |
|---|---|---|---|---|---|---|
| `x0_dummy` | 0.05856 | 0.02344 | -0.0002 | -0.0076 | **0.00346** | 0.02844 |
| **`x1_two_stage_huber`** | **0.05394** | **0.02069** | **0.1514** | **0.5242** | 0.00478 | **0.02468** |

`x1` beats `x0` on RMSE, MAE, R², Spearman, and nonzero-row MAE -- the metrics that matter for a
regression task on a target whose whole point is to separate "how much threat was denied" rows
from each other, not just to average them out. `x0`'s Spearman is undefined in practice (a
constant prediction has no variance to correlate against; the near-zero value shown is numerical
noise from `x0`'s tiny per-fold intercept differences, not a real ranking signal). `x0` slightly
beats `x1` on zero-row MAE (0.00346 vs 0.00478) -- read honestly in section 9, not glossed over:
`x0` always predicts a value close to the overall mean (~0.0035, itself close to zero), so it is
cheaply accurate on the exactly-zero rows almost by accident, while `x1`'s classifier stage
occasionally assigns nonzero probability to a row that turns out to be exactly zero, incurring a
small real cost there. This does not change the overall verdict: `x1`'s big win on nonzero-row
MAE (12.5% lower) and R² (from -0.0002 to +0.151, a real move off the zero-skill floor) more than
compensates.

## 5. Paired significance test, `x1` vs `x0`

Per-fold RMSE (`outputs/models/validation/significance_x0_vs_x1.json`): `x1` beats `x0` in
**5 of 5 folds** (mean diff -0.00461, i.e. `x1`'s RMSE is lower every fold). Paired t-test
t=-20.75, **p=0.00003** -- clearly significant. Wilcoxon signed-rank p=0.0625, the same
5-fold floor (the minimum p-value achievable with 5 same-signed paired samples) seen throughout
this project. No common-scale correction needed for this test, since neither variant is
log-transformed.

## 6. Classifier-stage diagnostic (P(nonzero), OOF) -- diagnostic only, not the Rung-0 gate

| Metric | Value |
|---|---|
| ROC-AUC | 0.7103 |
| Average precision | 0.9093 |
| Brier score | 0.1469 |

A P(nonzero) ROC-AUC of 0.71 is real, moderate signal -- consistent with section 1's evidence
that `event_type` and a handful of boolean flags carry genuine, several-fold variation in
zero-rate, not overwhelming signal (nothing here claims the classifier stage is a strong
standalone model). Average precision is high (0.91) mostly because the positive class
(`nonzero`) is the 79.85% majority, not because the classifier is unusually sharp -- reported
alongside ROC-AUC rather than in isolation, to avoid overstating it. **This is a diagnostic on
one internal piece of `x1`, not the Rung-0 gate**: the Rung-0 promotion decision rests on the
combined-prediction `x1`-vs-`x0` regression comparison in sections 4-5, which is the piece this
prompt actually built and can fairly gate.

## 7. Calibration (5 bins by predicted value, held-out test, `x1`)

| Bin | n | Mean predicted | Mean actual | Gap (predicted - actual) |
|---|---|---|---|---|
| 0 (most negative predicted) | 2,119 | -0.00783 | -0.02857 | +0.02074 |
| 1 | 2,119 | -0.00156 | -0.00194 | +0.00038 |
| 2 | 2,119 | -0.00020 | -0.00030 | +0.00009 |
| 3 | 2,119 | 0.00312 | 0.00252 | +0.00059 |
| 4 (most positive predicted) | 2,119 | 0.02621 | 0.04333 | -0.01713 |

The middle three bins (1-3) track closely, gaps under 0.0006. The two tail bins do not: bin 0
under-predicts the magnitude of the most-negative rows (predicted -0.0078 vs. actual -0.0286,
i.e. actual delta is more than 3x more negative than predicted), and bin 4 under-predicts the
magnitude of the most-positive rows the same way (predicted 0.0262 vs. actual 0.0433). Both tails
point the same direction -- `x1` systematically pulls extreme rows toward the middle. This is a
real, reportable limitation, and a plausible consequence of the architecture choice itself:
`HuberRegressor` is deliberately robust to outliers (that is why it was chosen over plain
least-squares in section 1, to avoid a few large-magnitude rows dominating the fit), and the same
property that protects the fit from being dominated by extreme rows also makes it structurally
conservative about predicting extreme values. This is a genuine Rung-0 finding to carry into
later rungs (e.g. whether a quantile-regression head at the tails, or a non-linear regression
head, narrows this gap), not something to correct at this rung -- no calibration correction is
applied here, mirroring the binary/continuous legs' own Rung-0 discipline of reporting
calibration as observed rather than post-hoc adjusting it.

## 8. Held-out test readout (single readout, `FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION`)

| Variant | RMSE | MAE | R² | Spearman | Zero-row MAE | Nonzero-row MAE |
|---|---|---|---|---|---|---|
| `x0_dummy` | 0.05765 | 0.02327 | -0.0001 | NaN | **0.00347** | 0.02844 |
| **`x1_two_stage_huber`** | **0.05309** | **0.02059** | **0.1519** | **0.5180** | 0.00488 | **0.02469** |

Same direction and near-identical magnitude as CV throughout -- `x1`'s held-out R² (0.1519)
is essentially unchanged from CV (0.1514), and RMSE/MAE both move less than 2% between CV and
held-out. No sign of overfitting on this rung.

## 9. Rung-0 result

**`x1_two_stage_huber` clears Rung 0** against the `x0_dummy` floor: consistent, large-margin
direction across CV and held-out test on RMSE, MAE, R², and Spearman; a statistically significant
5/5-fold paired test; and a real jump off the zero-skill floor (R² -0.0002 -> +0.1514). The one
place `x0` edges `x1` (zero-row MAE) is reported plainly in section 4, not smoothed over, and
does not change the overall verdict given the much larger gain on the majority nonzero-row slice.

The architecture decision (section 1) is the load-bearing result of this document: the zero mass
is real, structural, and meaningfully predictable from locked features that are legitimately
available to a model (not from the excluded possession-flag columns that mechanically cause it),
which is the evidence basis for every later rung on this leg to build on the same two-stage
shape rather than reopening the question each time.

**No model ladder is built in this document** -- this is Rung 0 only, mirroring the binary and
continuous legs' own Prompt 36/49/54/59 discipline. Passive-xT's own Rung 0 (a separate,
later prompt) has not been started here.

*Split note (prompt 71): Rung 1 (`x1b_quadratic`, below) is appended directly to this document,
not split into a separate ladder doc yet -- mirroring `c1`'s own precedent
(`ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md`), which stayed a single document through its own Rung 1
and only split once a second rung (Rung 2, `c1d_random_forest`) existed. The same threshold
applies here: this document remains Rung-0-and-1 until a Rung 2 (random forest, out of scope for
this prompt) is built.*

## 11. Rung 1: `x1b_quadratic` (prompt 71)

### 11.1 Step 0 -- where do quadratic/interaction terms go, and why

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

### 11.2 Feature-set construction

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

### 11.3 Cross-validation results (5-fold, OOF metrics)

| Variant | RMSE | MAE | R² | Spearman | Zero-row MAE | Nonzero-row MAE |
|---|---|---|---|---|---|---|
| `x1_two_stage_huber` | 0.05394 | 0.02069 | 0.1514 | **0.5242** | **0.00478** | **0.02468** |
| `x1b_quadratic` | **0.05391** | 0.02070 | **0.1525** | 0.5233 | 0.00481 | 0.02468 |

Movement is real but tiny in every column: RMSE improves by 0.00004 (0.07% relative), R² improves
by 0.0011, MAE and nonzero-row MAE are flat to the 5th decimal, and Spearman and zero-row MAE are
each marginally *worse*. This is not the shape of a rung that meaningfully improved the model.

### 11.4 Paired significance test, `x1b` vs `x1`

Per-fold RMSE (`outputs/models/validation/significance_x1_vs_x1b_quadratic.json`): `x1b` beats
`x1` in 5 of 5 folds, but the per-fold differences are small and consistent enough to produce a
significant paired t-test (t=-7.37, **p=0.0018**) despite the practically negligible effect size
-- mean diff -0.00004, roughly two orders of magnitude smaller than Rung 0's own `x1`-vs-`x0`
mean diff (-0.00461). Wilcoxon signed-rank p=0.0625, the same 5-fold floor seen throughout this
project. **A significant p-value here is not being read as "clears the bar"** -- see section 11.6.

### 11.5 Calibration -- the actual motivating question for this rung

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

### 11.6 Promotion call

**`x1b_quadratic` does not clear the bar to replace `x1` as the standing baseline.** The aggregate
metrics move in `x1b`'s favor by an amount too small to matter operationally (RMSE 0.07% lower,
R² +0.0011), and the one metric this rung actually exists to move -- the tail-calibration gap --
did not improve; if anything it is marginally worse. A statistically significant paired test
(section 11.4) is not being treated as sufficient on its own, since a real effect can still be
too small to be worth carrying forward, and this leg's own Step-0 evidence (the tail bins) is a
closer read of practical value than a p-value computed on a sub-0.0001 mean RMSE difference.

**`x1_two_stage_huber` remains the standing baseline**, carried forward into the leg's next rung
(random forest, out of scope for this prompt). This is a valid, reportable Rung-1 outcome, not a
failure to fix before moving on: the evidence in section 11.1 was real (the candidate features do
show genuine curvature in both stages' target quantities), and testing that evidence properly
required actually fitting the expanded model rather than assuming curvature terms would help
because the correlation coefficients looked non-trivial -- they were real, they just were not the
right kind of fix for a loss-function-driven calibration limitation.
