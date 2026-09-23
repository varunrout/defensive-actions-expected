# Passive-xT Model Ladder

> **Status update (Prompt 79):** `y1d_gradient_boosting` (Rung 3) does **not** clear this rung's
> ladder gate against `y1c_random_forest`: the paired significance test on 5-fold CV RMSE is not
> significant (paired-t p=0.0635, Wilcoxon p=0.125), and held-out Spearman regresses meaningfully
> (0.315 -> 0.273) even though RMSE/R^2 move slightly in `y1d`'s favor and the prediction-bias gate
> `y1c` closed stays closed. See section 4.7 below. `y1c_random_forest` remains the leg's standing
> baseline.
>
> **Status update (Prompt 78):** `y1c_random_forest` (Rung 2) has cleared this rung's ladder gate
> against `y1_two_stage_huber` and is promoted to the leg's new standing baseline. See section 3.7
> below. `y1b_quadratic` (Rung 1) did not clear its own gate and remains documented as the rung
> that didn't win.

*Split out of `PASSIVE_XT_BASELINE_SUMMARY.md` (prompt 78, once a second rung -- Rung 2,
`y1c_random_forest` -- existed beyond Rung 0) so the locked Rung 0 baseline document doesn't keep
growing as more rungs are tried -- mirroring `active_xt`'s own exact split point
(`ACTIVE_XT_MODEL_LADDER.md`, split at its own Rung 2, `x1c_random_forest`) and
`active_continuous`'s split point before that. Rung 0 -- the two locked variants (`y0_dummy`,
`y1_two_stage_huber`), the architecture decision, and the Rung-0 calibration finding -- stays in
[`PASSIVE_XT_BASELINE_SUMMARY.md`](PASSIVE_XT_BASELINE_SUMMARY.md). This document is everything
built on top of that locked Rung 0 baseline, starting from Rung 1. This is a pure reorganization
for Rung 1's material: no numbers changed, verified by rendering both HTML documents in-browser
after the split, same verification `active_xt`'s own split commit used.*

## 1. What a rung is

A **rung** is one controlled change to the current standing baseline: a single variable changed --
feature shape, an interaction term, a model family -- with everything else held identical (same 38
locked features unless the rung specifically changes that, same canonical match-grouped 5-fold
split, same signed `target_xt_delta_passive` target, no ad-hoc tuning beyond what a rung's own grid
search covers). A rung is tested against 2 fixed gates before it can be considered for adoption as
the new baseline:

1. **Paired significance test vs the current standing baseline** -- refit both variants on the
   same 5 canonical CV folds, paired t-test + Wilcoxon signed-rank test on the per-fold RMSE
   difference (no log-transform on this leg's signed target, so no common-scale correction is
   needed).
2. **Held-out-test confirmation** -- a single, one-time readout on the frozen held-out test set
   (315,311 rows / 23 matches). This is the gate that actually decides the rung's fate: CV picks a
   candidate, the held-out test either confirms or rejects it, and that result is not re-litigated
   after the fact to fit a preferred narrative.

This leg carries a **third, rung-specific check**, established at Rung 1 after correcting a
misread of Rung 0's own calibration table: the 5-bin calibration table, tracked specifically for
whether it shows -- and whether a rung reduces -- the **systematic positive prediction bias**
found in all 5 of `y1`'s own Rung-0 bins (not a symmetric tail gap the way active-xT's own leg
tracks; passive's own Rung 0 does not show that pattern, confirmed directly rather than assumed).
An aggregate metric improving without that bias shrinking would be an incomplete answer, not a
full one, the same standing principle active-xT's own ladder established for its own (different)
motivating diagnostic.

Rung 0 (the baseline itself) lives in
[`PASSIVE_XT_BASELINE_SUMMARY.md`](PASSIVE_XT_BASELINE_SUMMARY.md). This ladder builds on that
document's `y1_two_stage_huber` pick (held-out RMSE 0.033880, R² 0.0612) as the baseline every
rung up through Rung 1 is compared against; Rung 2 changes the comparison baseline to
`y1c_random_forest` once it is promoted (section 3.7).
## 2. Rung 1: `y1b_quadratic` (prompt 77)

### 2.1 Task 0 -- what this rung is actually testing

Active-xT's own Rung 1 (Prompt 71) was motivated by a **symmetric** tail-under-prediction pattern
in `x1`'s Rung-0 calibration table (both extreme bins' `|predicted|` smaller than `|actual|`).
Checked `y1_two_stage_huber`'s own Rung-0 calibration table directly before assuming the same
motivation carries over -- **it does not show that pattern**. Section 7 above has been corrected
(this rung's own grounding work caught the error): `predicted > actual` in **all 5 bins**, not
just the tails -- a single, systematic **positive bias** across the whole distribution, confirmed
independently by the held-out `prediction_bias` field (+0.00278, larger in magnitude than the
true mean itself). **This rung checks whether quadratic/interaction terms reduce that systematic
positive bias** -- not whether they close a "tail gap" the way active-xT's Rung 1 did, since
passive's own Rung 0 does not show that pattern.

### 2.2 Step 0 -- which stage(s) get expanded terms, decided from passive's own data

Active-xT's own Rung 1 expanded **both** stages because the same candidate numeric features
showed real curvature in both the zero-rate (Stage A's target) and the nonzero-delta magnitude
(Stage B's target). Checked the identical cross-check here, not assumed to transfer: the
regression-stage quadratic candidates (below) show real, large-sample nonzero-delta-magnitude
curvature, but their own zero-rate curvature
(`passive_numerical_target_atlas.json`'s unconditional `bins` panel) is much weaker:

| Feature | Zero-rate range (10 bins) | Ratio |
|---|---|---|
| `ball_x` | 22.77% - 28.33% | ~1.25x |
| `top_option_3_threat_score` | 24.88% - 27.72% | ~1.11x |
| `angle_to_attacking_goal` | 24.7% - 27.09% | ~1.10x |

...nowhere near the ~176x range `on_ball_event_type` (a **categorical** feature, Prompt 76's own
Step-0 evidence) showed for the classifier stage. The classifier's own real predictive signal on
this leg comes from categorical structure that quadratic/interaction terms (numeric-only by
construction) cannot add anything to.

**Decision: expand the regression stage only.** A genuine departure from active-xT's own Step-0
answer (which expanded both stages), reached here because passive's own evidence pattern is
different, not because this rung defaulted to a different answer for its own sake.

### 2.3 Feature-set construction

**Quadratic (3 features)**: top locked PASSIVE numeric features by `pearson_r_nonzero_delta` are
`ball_x` (0.290), `defender_x` (0.222), `top_option_3_threat_score` (0.218),
`top_option_2_threat_score` (0.203) -- all confirmed real (non-thin, ~116-117k rows/bin). But
`defender_x` correlates r=0.81 with `ball_x`, and `top_option_2_threat_score` correlates r=0.87
with `top_option_3_threat_score` (checked directly against the real data) -- keeping both of
either pair would pad the list with near-duplicate curvature information. Kept the stronger of
each redundant pair plus one more-independent (r=0.35-0.47 with the other two), weaker-but-real
candidate:
`QUADRATIC_FEATURES_Y1B = ["ball_x", "top_option_3_threat_score", "angle_to_attacking_goal"]`

**Interaction (3 pairs)**: reused directly from
`reports/analysis/xt_target/PASSIVE_FEATURE_INTERACTION_ANALYSIS.json` (Prompt 69's own
passive-specific findings, not re-discovered from scratch). All 3 of its `interactive` pairs are
used directly -- unlike active-xT's own `x1b`, no pair needed dropping for an excluded feature,
since all involved features are locked PASSIVE modelling features:
`("top_option_2_threat_score", "top_option_2_distance_from_ball")`,
`("lane_screening_score_option_2", "engagement_distance_to_carrier")`,
`("marking_tightness", "engagement_distance_to_carrier")`.

Both are appended only to the regression head's design matrix (44 columns total: 38 locked + 3
quadratic + 3 interaction); the classifier stage is byte-for-byte unchanged from `y1`.

### 2.4 Cross-validation results (5-fold, OOF metrics)

| Variant | RMSE | MAE | R² | Spearman | Prediction bias |
|---|---|---|---|---|---|
| `y1_two_stage_huber` | 0.035684 | 0.010853 | 0.0786 | **0.3193** | 0.003106 |
| `y1b_quadratic` | **0.035612** | **0.010837** | **0.0823** | 0.3169 | **0.003291** |

Movement is real but tiny in the accuracy columns: RMSE improves by 0.00007 (0.2% relative), R²
improves by 0.0037, MAE is flat, and Spearman is marginally worse. **Prediction bias -- the
quantity this rung actually exists to move -- gets worse, not better** (0.003106 -> 0.003291).

### 2.5 Paired significance test, `y1b` vs `y1`

Per-fold RMSE (`outputs/models/validation/significance_y1_vs_y1b_quadratic.json`): `y1b` beats
`y1` in 5 of 5 folds, small and consistent enough to produce a significant paired t-test (t=-9.67,
**p=0.0006**) despite a practically negligible effect size -- mean diff -0.00007, roughly 20x
smaller than Rung 0's own `y1`-vs-`y0` mean diff (-0.00148). Wilcoxon p=0.0625, the same 5-fold
floor seen throughout this project. **A significant p-value here is not read as "clears the
bar"** -- see section 2.7.

### 2.6 Calibration -- does `y1b` reduce the systematic positive bias?

| Bin | `y1` gap (Rung 0) | `y1b` gap (this rung) | Change |
|---|---|---|---|
| 0 (most negative predicted) | +0.00656 | **+0.00499** | improves |
| 1 | +0.00155 | +0.00284 | **worse** |
| 2 | +0.00060 | +0.00091 | worse |
| 3 | +0.00113 | +0.00116 | flat |
| 4 (most positive predicted) | +0.00406 | **+0.00501** | **worse** |

**No -- the systematic positive bias does not shrink; if anything it is slightly worse.** Only
bin 0 improves; bins 1, 2, and 4 all move in the wrong direction (bin 1 nearly doubles), and bin 3
is flat. The held-out `prediction_bias` field confirms this at the aggregate level too: +0.00278
(`y1`) -> +0.00298 (`y1b`), worse. Stated plainly, per this project's own standing discipline of
not letting a better-looking aggregate metric stand in for the rung's actual motivating question:
**this rung's real target (the systematic positive bias identified in section 2.1) is not
improved by adding curvature to the regression head.**

### 2.7 Promotion call

**`y1b_quadratic` does not clear the bar to replace `y1` as the standing baseline.** The
aggregate metrics move in `y1b`'s favor by an amount too small to matter operationally (RMSE
0.2% lower), and the one thing this rung actually exists to move -- the systematic prediction
bias -- gets worse, not better, on both the bin-level calibration table and the aggregate
`prediction_bias` metric. A statistically significant paired test (section 2.5) is not treated
as sufficient on its own, the same discipline `x1b_quadratic` established on the active leg.

**`y1_two_stage_huber` remains the standing baseline.** A valid, reportable Rung-1 outcome, not a
failure to fix before moving on: the Step-0 evidence was real (the candidate features do show
genuine curvature in the regression stage's own target quantity), and testing that evidence
properly required actually fitting the expanded model rather than assuming curvature terms would
reduce a bias that, on inspection, is not a curvature-shaped problem in the first place --
`HuberRegressor`'s own systematic offset is a location/scale property of its fit, not something
additional nonlinear terms in the same linear-in-parameters model are positioned to correct. This
mirrors active-xT's own Rung 1 finding (a different model-family change, not more features on the
same model, closed its own calibration gap) -- a candidate hypothesis for this leg's own next
rung, not something this document resolves.

## 3. Rung 2 -- `y1c_random_forest` (prompt 78)

`y1b_quadratic` (Rung 1) did not clear `y1`: aggregate RMSE moved in `y1b`'s favor by an amount
too small to matter, and the rung's actual motivating question -- the systematic positive
prediction bias present in all 5 of `y1`'s calibration bins -- got worse, not better. This rung
tests a different hypothesis: does a model-*family* change (random forest, whose split-based
mechanism differs fundamentally from adding explicit quadratic/interaction terms to the same
linear model) reduce that bias where more features on the same linear model could not.

### 3.1 Step 1 -- what does `y1c` replace, decided from passive's own evidence

Active-xT's own Rung 2 (`x1c_random_forest`, Prompt 73) replaced **both** of `x1`'s stages,
reasoned from evidence that the same candidate features carried curvature in both target
quantities -- no asymmetric evidence to justify touching only one surface. This leg's own Rung 1
(Prompt 77) found the *opposite* evidence pattern for quadratic **terms** specifically: passive's
strongest classifier-stage signal comes from categorical structure (`on_ball_event_type`'s ~176x
zero-rate range), while the numeric quadratic candidates showed only weak zero-rate curvature
(~1.1-1.25x) -- the basis for Rung 1's "regression stage only" decision, since squaring a numeric
feature cannot add anything to signal that is fundamentally categorical.

**That reasoning does not transfer to random forest, and the difference matters**: a
`RandomForestClassifier` does not need explicit quadratic/interaction terms to exploit
categorical structure -- it splits on any locked feature directly, including one-hot categorical
columns, and can discover interactions *among* categorical features (e.g. does
`on_ball_event_type`'s own effect on P(nonzero) vary by `phase_label`?) that a plain
`LogisticRegression` with only linear one-hot main effects cannot represent at all. Rung 1's
exclusion of the classifier stage was specific to the *tool* (quadratic terms are numeric-only by
construction), not a general finding that the classifier has nothing left to gain -- if anything,
the classifier's own strong, already-established categorical signal is exactly the kind of
structure a tree-based split mechanism is well suited to exploit further.

**Decision: replace both stages with RandomForest** -- `RandomForestClassifier` for P(nonzero),
`RandomForestRegressor` for E[delta|nonzero]. This lands on the same practical answer active-xT's
own `x1c` reached, but via genuinely different, leg-specific reasoning (RF's own split mechanism
vs. the numeric-only limitation that shaped Rung 1's decision), not by assuming active's answer
transfers.

### 3.2 Grid, sized for this leg's real row counts

Classifier trains on 1,275,289 trainval rows (~30x active-xT's own 45,166); regressor trains on
941,234 nonzero-only trainval rows (~26x active-xT's own 36,121). Unlike active-xT's rows (1 row
per event, independent), this leg's rows are **not** independent -- every defender-slot row
sharing an `event_id` carries the identical target value (~8.03 rows/event, Prompt 68).
`MIN_SAMPLES_LEAF_GRID` is scaled up accordingly, mirroring `d1d_random_forest`'s own reasoning
for this exact leg-wide duplication property (`[30,100,300]` there vs. the non-duplicated legs'
smaller grids): used here, `[50, 150, 500]`. `N_ESTIMATORS` trimmed to 100 (from active-xT's 200)
to keep runtime reasonable at this row count. `MAX_DEPTH_GRID` unchanged, `[6, 10, None]`.

**Classifier grid** (chosen by OOF ROC-AUC; full 9-row grid in
`outputs/models/validation/y1c_classifier_tuning.json`):

| max_depth | min_samples_leaf | OOF ROC-AUC | Train ROC-AUC | Train &minus; OOF gap |
|---|---|---|---|---|
| 6 | 50 | 0.8034 | 0.8123 | 0.0090 |
| 6 | 150 | 0.8030 | 0.8121 | 0.0091 |
| 6 | 500 | 0.8022 | 0.8111 | 0.0089 |
| 10 | 50 | 0.8121 | 0.8395 | 0.0273 |
| 10 | 150 | 0.8121 | 0.8358 | 0.0237 |
| 10 | 500 | 0.8115 | 0.8296 | 0.0181 |
| None | 50 | 0.8165 | 0.9525 | 0.1360 |
| **None** | **150** | **0.8173** | 0.9136 | 0.0963 |
| None | 500 | 0.8160 | 0.8642 | 0.0482 |

**Regressor grid** (chosen by OOF RMSE; full 9-row grid in
`outputs/models/validation/y1c_regressor_tuning.json`):

| max_depth | min_samples_leaf | OOF RMSE | OOF R² | Train R² | Train &minus; OOF gap |
|---|---|---|---|---|---|
| 6 | 50 | 0.02978 | 0.5255 | 0.5563 | 0.0308 |
| 6 | 150 | 0.02978 | 0.5256 | 0.5509 | 0.0253 |
| 6 | 500 | 0.02970 | 0.5281 | 0.5407 | 0.0126 |
| 10 | 50 | 0.02954 | 0.5331 | 0.6463 | 0.1132 |
| 10 | 150 | 0.02938 | 0.5380 | 0.6054 | 0.0673 |
| 10 | 500 | 0.02934 | 0.5394 | 0.5669 | 0.0275 |
| None | 50 | 0.02967 | 0.5289 | 0.7672 | 0.2382 |
| None | 150 | 0.02935 | 0.5390 | 0.6487 | 0.1097 |
| **None** | **500** | **0.02929** | **0.5408** | 0.5773 | 0.0366 |

Both grids again pick the least-constrained depth (`max_depth=None`), but -- unlike active-xT's
own Rung 2, where the winning config also had the *largest* gap in its grid -- here the chosen
`min_samples_leaf` (150 classifier / 500 regressor, both the *largest* leaf-size option tried)
keeps the train-minus-OOF gap moderate (0.096 / 0.037) rather than picking the worst-overfitting
corner of the grid. The large-leaf requirement this leg's own row-duplication property motivated
(section 3.2) is doing real work: the smallest-leaf configs at `max_depth=None` show gaps of
0.136 (classifier) and 0.238 (regressor) and still lose on OOF score to the larger-leaf configs --
confirming the duplication-aware grid sizing was the right call, not just a precaution.

### 3.3 `y1c_random_forest` vs `y1_two_stage_huber` -- ranking (CV and held-out)

| Variant | RMSE (CV) | RMSE (held-out) | R² (CV) | R² (held-out) | Spearman (held-out) |
|---|---|---|---|---|---|
| `y1_two_stage_huber` | 0.035684 | 0.033880 | 0.0786 | 0.0612 | 0.2936 |
| **`y1c_random_forest`** | **0.026371** | **0.024892** | **0.4968** | **0.4933** | **0.3154** |

A very large, consistent win -- RMSE down ~26.1% (CV) / ~26.5% (held-out), R² **more than
sextupling** (0.079 -> 0.497 CV; 0.061 -> 0.493 held-out), Spearman also meaningfully better.
This is the largest effect size reported anywhere on this leg's own ladder so far, and larger in
relative terms than active-xT's own Rung-2 win (R² roughly tripled there, versus more than
sextupling here).

| Variant | Zero-row MAE | Nonzero-row MAE |
|---|---|---|
| `y1_two_stage_huber` | **0.001842** | 0.013360 |
| `y1c_random_forest` | 0.004474 | **0.011080** |

Same pattern established at active-xT's own Rung 2: `y1c`'s classifier occasionally assigns real
nonzero probability to rows that turn out to be exactly zero (zero-row MAE up ~2.4x), more than
offset by a large nonzero-row MAE improvement (17% lower).

### 3.4 Paired significance test, `y1c` vs `y1`

Per-fold RMSE (`outputs/models/validation/significance_y1_vs_y1c_random_forest.json`), compared
against the **standing baseline (`y1_two_stage_huber`), not `y1b_quadratic`** (which did not win
promotion): `y1c` beats `y1` in **5 of 5 folds** (0.02759 vs 0.03745, 0.02628 vs 0.03576, 0.02639
vs 0.03575, 0.02671 vs 0.03583, 0.02497 vs 0.03375), mean diff **-0.00932**. Paired t-test
t=-51.71, **p&lt;0.0001** -- overwhelmingly significant, an order of magnitude larger effect size
than Rung 1's own result (mean diff -0.00007) and even larger than Rung 0's own `y1`-vs-`y0`
result (-0.00148). Wilcoxon p=0.0625, the same 5-fold floor seen throughout this project.

### 3.5 Calibration -- does `y1c` fix the systematic positive bias?

| Bin | `y1` gap (Rung 0) | `y1b` gap (Rung 1) | `y1c` gap (this rung) |
|---|---|---|---|
| 0 (most negative predicted) | +0.00656 | +0.00499 | **-0.00063** |
| 1 | +0.00155 | +0.00284 | **-0.00015** |
| 2 | +0.00060 | +0.00091 | +0.00006 |
| 3 | +0.00113 | +0.00116 | +0.00014 |
| 4 (most positive predicted) | +0.00406 | +0.00501 | **+0.00050** |

**Yes -- decisively, and this is the headline result of this rung.** Every bin's gap shrinks by
roughly 8-20x in magnitude (bin 0: 0.00656 -> 0.00063, ~10x; bin 1: 0.00155 -> 0.00015, ~10x; bin
4: 0.00406 -> 0.00050, ~8x), and two bins even flip sign (0 and 1 go from `predicted > actual` to
`predicted < actual`, essentially noise-level at this point rather than a directional bias). The
aggregate `prediction_bias` metric confirms this cleanly:

| Variant | Prediction bias (held-out) |
|---|---|
| `y1_two_stage_huber` | +0.002780 |
| `y1b_quadratic` | +0.002982 (worse) |
| **`y1c_random_forest`** | **-0.000018** |

The systematic positive bias that `y1b`'s regression-stage-only expansion could not touch (and
made marginally worse) is **essentially eliminated** by the model-family change -- direct
confirmation that Rung 1's own diagnosis (the offset is a property of `HuberRegressor`'s fit, not
a curvature-shaped problem the same model could be coaxed into fixing with more terms) was
correct: a fundamentally different model family removes it, more features on the same one could
not.

### 3.6 Feature importance -- and a caveat this leg's own EDA already carries

Top 15 by Gini importance and by fold-held-out permutation importance (regression head,
`outputs/models/regression/y1c_random_forest.json`):

| Rank | Gini importance | Permutation importance (fold held-out) |
|---|---|---|
| 1 | `ball_x` (0.3392) | `ball_x` (+0.001362) |
| 2 | `ball_y` (0.3155) | `ball_y` (+0.000553) |
| 3 | `on_ball_event_type_Shot` (0.2705) | `on_ball_event_type_Shot` (+0.000515) |
| 4 | `on_ball_event_type_Pass` (0.0172) | `on_ball_event_type_Pass` (+0.000063) |
| 5 | `top_option_3_threat_score` (0.0083) | `on_ball_event_type_Carry` (+0.000008) |

**`ball_x` and `ball_y` together account for ~65% of Gini importance, and `on_ball_event_type_Shot`
a further ~27%** -- these 3 columns alone explain ~93% of the regression head's total importance,
an extremely concentrated result compared to every other rung on either leg of this project. This
is cross-checked directly against `reports/analysis/xt_target/PASSIVE_LEAKAGE_AUDIT.json`'s own
Part E' (construction-coupling audit, Prompt 69) **before being read as a new finding**:
`ball_x`/`ball_y` are the *exact* two columns `xt_before = xT(ball_x, ball_y)` is computed from,
and that audit already measured `ball_x` at r=+0.52 vs `xt_before` and r=+0.25 vs the target
itself -- a known, already-documented construction-coupling caveat, not new information. `ball_y`
itself correlates far more weakly (r=-0.004 vs `xt_before` per that same audit) despite being the
other grid-lookup input, consistent with the Karun Singh xT grid varying much more along the
pitch-length axis than the width axis. **`on_ball_event_type_Shot`'s importance is separately
sensible and not a coupling artefact**: a Shot event's own `ball_x`/`ball_y` sits deep in the
attacking box (structurally high on the xT grid), and Prompt 76's own Step-0 evidence already
established Shot rows are almost never zero (0.30%) -- large `xt_before`, `action_ended_possession`
usually true, so a large positive delta is the expected, not spurious, pattern for this event type.
**Read `y1c`'s accuracy gain with this caveat attached**: a meaningful share of it is the tree
model successfully exploiting signal this leg's own EDA had already flagged as construction-linked,
not entirely fresh predictive structure discovered from scratch -- reported plainly rather than
presented as an unqualified win on "real" football signal.

### 3.7 Ladder-gate outcome for this rung, no promotion audit

**`y1c_random_forest` clears the Rung-2 gate against `y1_two_stage_huber`, decisively and on
every axis this leg has tracked.** A consistent (5-of-5 folds), overwhelmingly significant
(p&lt;0.0001) improvement on RMSE, R² more than sextupling on both CV and held-out with no
CV-to-held-out drop-off, and -- the specific diagnostic this ladder has tracked since Rung 1's own
corrected Task-0 finding -- the systematic positive prediction bias is essentially eliminated
(+0.00278 -> -0.00002). Per this prompt's explicit constraint, no deeper promotion audit
(tournament-stratified check, error-slice comparison, feature-shape sanity check -- the kind of
deep dive active-xT's own separate promotion prompt, 75, ran) is performed here; this section
reports the Rung-2 ladder-gate outcome only, the same discipline `x1c_random_forest`'s own Rung-2
prompt (73) followed before its own separate promotion-audit prompt (75).

**`y1c_random_forest` is now the leg's standing baseline**, replacing `y1_two_stage_huber`.
`y1_two_stage_huber` remains documented as the Rung-0 baseline it was measured against, and
`y1b_quadratic` remains documented as the Rung-1 rung that didn't clear its own bar. No further
rung (gradient boosting or otherwise) is built in this prompt.

## 4. Rung 3 -- `y1d_gradient_boosting` (prompt 79)

`y1c_random_forest` (Rung 2) is this leg's first real win: RMSE down ~26% (CV and held-out), R^2
more than sextupling (0.079 -> 0.497 CV), and -- critically -- the systematic positive prediction
bias this ladder tracked since Rung 0/1 (`y1`: +0.00278, `y1b`: +0.00298) is essentially eliminated
by `y1c` (-0.000018). This rung tests a different tree-ensemble method -- gradient boosting --
directly against `y1c`, mirroring active-xT's own Rung 3 (`x1d_gradient_boosting`, prompt 74) --
a method worth checking once one tree method already found real structure, not assumed to help
just because RF did.

**Important difference from active-xT's own Rung 3 motivation, stated up front**: on active-xT,
`x1c`'s own tail-calibration gap was still nonzero going into `x1d`, so `x1d`'s calibration check
had something live to test. On passive-xT, `y1c` has already essentially closed the
systematic-bias gate (-0.000018, functionally zero). So `y1d`'s own gate is **not** "does it fix a
bias `y1c` left open" -- there isn't one left open to fix. `y1d`'s gate is: does it beat `y1c` on
RMSE/R^2 without reopening the bias `y1c` just closed, and without a comparable regression
elsewhere. The calibration check in section 4.5 below is a **non-regression check, not a
fix-seeking one** -- no bias-fixing narrative is manufactured here where none applies.

### 4.1 Step 1 -- what does `y1d` replace, decided from this leg's own evidence

Same reasoning `x1d_gradient_boosting` applied on the active leg: `y1c_random_forest` replaced
**both** of `y1`'s stages (confirmed directly against `outputs/models/regression/y1c_random_forest.json`
-- `clf_params` and `reg_params` both present), and no new asymmetric evidence has appeared since
Rung 2 to justify upgrading only one surface. **Decision: `y1d` replaces both stages with gradient
boosting** -- `LGBMClassifier` for P(nonzero), `LGBMRegressor` for E[delta|nonzero], mirroring
`y1c`'s own two-independent-models composition, gated against `y1c` specifically (not `y1`/`y1b`)
per this prompt's own instruction. Library: LightGBM (version 4.6.0, confirmed available), matching
`x1d_gradient_boosting`, `v1e_gradient_boosting`, and `c1e_gradient_boosting` -- kept consistent
rather than introducing XGBoost as a second boosting library for no reason. Feature set: the same
38 locked PASSIVE features `y1c` used (no `y1b` quadratic/interaction terms reused).

### 4.2 Grid, sized for this leg's own row counts and duplication property

Classifier trains on 1,275,289 trainval rows; regressor trains on 941,234 nonzero-only trainval
rows -- the same row counts `y1c` reported, confirmed via `yb.load_data()`. Unlike active-xT's
rows (1 row per event, independent), this leg's rows are **not** independent -- every
defender-slot row sharing an `event_id` carries the identical target value (~8.03 rows/event,
Prompt 68). For a leaf-wise boosted model this matters even more than it does for RF: LightGBM's
leaf-wise growth can carve out a tiny leaf that isolates a handful of duplicate-target rows from
one or two events and fit them near-perfectly -- exactly the kind of spurious "signal" grouped
5-fold CV is meant to catch, but a too-small `min_child_samples` makes that state easy to reach
within a single fold. `MIN_CHILD_SAMPLES_GRID` is therefore set to `[50, 150, 500]` -- **reusing
`y1c`'s own already-established, duplication-aware `MIN_SAMPLES_LEAF_GRID` values directly** (same
leg, same duplication property, same reasoning), rather than `x1d`'s smaller `[20, 50, 150]`
(sized for active-xT's independent, ~30x-smaller-sample rows) or a blind copy of `v1e`'s
`[10, 30, 100]`. `NUM_LEAVES_GRID=[15, 31, 63]` and `LEARNING_RATE_GRID=[0.01, 0.05, 0.1]` are
unchanged from `x1d`/`v1e`/`c1e` -- neither scales with row count or duplication the way leaf-size
parameters do. `n_estimators` is never grid-searched directly -- chosen per fit via early stopping
(cap 2,000 rounds, 50-round patience) on a match-grouped validation carve-out of that fit's own
training rows, exactly `x1d`'s own fitting procedure.

**Classifier grid winner** (chosen by OOF ROC-AUC, full 27-row grid in
`outputs/models/validation/y1d_classifier_tuning.json`): `learning_rate=0.01, num_leaves=63,
min_child_samples=500` -- OOF ROC-AUC 0.8197, train ROC-AUC 0.8566, train-minus-OOF gap 0.0370,
mean best iteration 575. As with `y1c`'s own grid, the largest `min_child_samples` value tried
(500) wins at the most flexible `num_leaves` (63) -- the largest-leaf-size, most-flexible-tree
corner of the grid, but held in check by early stopping rather than by the leaf-size constraint
alone (unlike `y1c`'s RF grid, which has no per-fit early stopping and relies on `min_samples_leaf`
alone to control overfit).

**Regressor grid winner** (chosen by OOF RMSE, full 27-row grid in
`outputs/models/validation/y1d_regressor_tuning.json`): `learning_rate=0.05, num_leaves=15,
min_child_samples=500` -- OOF RMSE 0.02923, OOF R^2 0.5429, train R^2 0.5934, train-minus-OOF gap
0.0505, mean best iteration 376. Here the *smallest* `num_leaves` (15) wins, not the largest --
different from the classifier stage and from `y1c`'s own regressor grid (which picked
`max_depth=None`, its least-constrained option). At every `num_leaves`/`learning_rate` combination
tried, `min_child_samples=500` (the largest, most duplication-conservative option) wins or ties on
OOF RMSE against the smaller options, confirming the duplication-aware upper end of the grid is
doing real work here too, not just at the classifier stage.

### 4.3 `y1d_gradient_boosting` vs `y1c_random_forest` -- ranking (CV and held-out)

| Variant | RMSE (CV) | RMSE (held-out) | R&sup2; (CV) | R&sup2; (held-out) | Spearman (held-out) |
|---|---|---|---|---|---|
| `y1c_random_forest` | 0.026371 | 0.024892 | 0.4968 | 0.4933 | **0.3154** |
| `y1d_gradient_boosting` | **0.026296** | **0.024645** | **0.4996** | **0.5032** | 0.2732 |

A small, mixed result -- not the large, unambiguous win `y1c` itself produced over `y1` at Rung 2.
RMSE improves marginally (CV: -0.28% relative; held-out: -0.99% relative), R^2 improves marginally
(CV +0.0028; held-out +0.0100), but **Spearman regresses meaningfully on the held-out set**
(0.3154 -> 0.2732, a ~13% relative drop) -- the opposite direction from every accuracy metric.
`y1d`'s classifier ranks predictions less consistently with the true ordering than `y1c`'s does,
even while its point predictions are closer on average (lower RMSE/MAE). This is the first rung on
either leg's ladder where the headline accuracy metric (RMSE) and the rank metric (Spearman) move
in opposite directions against the same comparison baseline.

| Variant | Zero-row MAE | Nonzero-row MAE |
|---|---|---|
| `y1c_random_forest` | **0.004474** | 0.011080 |
| `y1d_gradient_boosting` | 0.004011 | 0.011132 |

Zero-row MAE improves (~10% lower); nonzero-row MAE is essentially flat, a marginal 0.5% worse. No
large trade-off pattern here the way `y1c` itself showed over `y1`.

### 4.4 Paired significance test, `y1d` vs `y1c` (not `y1`/`y1b`)

Per-fold RMSE (`outputs/models/validation/significance_y1d_vs_y1c.json`), compared against the
**current standing baseline (`y1c_random_forest`), not the older linear rungs**: `y1d` beats
`y1c` in **4 of 5 folds** (0.027516 vs 0.027593, 0.026227 vs 0.026279, 0.026283 vs 0.026385,
0.026548 vs 0.026710; `y1c` wins fold 4, 0.024974 vs `y1d`'s 0.024991), mean diff
**-0.0000752** -- an order of magnitude smaller than Rung 2's own mean diff against `y1`
(-0.00932). Paired t-test t=-2.548, **p=0.0635** -- not significant at the conventional 0.05
threshold. Wilcoxon p=0.125, also not significant. **Unlike every prior rung comparison on this
leg, this one does not clear the significance bar at all**, on either test.

### 4.5 Calibration -- non-regression check, not a fix-seeking one

As stated in this section's own introduction, `y1c` already closed the systematic-bias gate this
ladder tracked since Rung 1 (-0.000018, functionally zero). This check asks only whether `y1d`
**holds** that closure or **reopens** it -- there is no live bias left for `y1d` to fix.

| Bin | `y1c` gap (Rung 2) | `y1d` gap (this rung) |
|---|---|---|
| 0 (most negative predicted) | -0.00063 | **-0.00025** |
| 1 | -0.00015 | +0.00014 |
| 2 | +0.00006 | +0.00013 |
| 3 | +0.00014 | -0.00011 |
| 4 (most positive predicted) | +0.00050 | **-0.00025** |

| Variant | Prediction bias (held-out) |
|---|---|
| `y1_two_stage_huber` | +0.002780 |
| `y1b_quadratic` | +0.002982 |
| `y1c_random_forest` | -0.000018 |
| `y1d_gradient_boosting` | -0.000068 |

**The bias stays closed.** Every bin's gap remains in the +/-0.00025 range, the same noise-level
magnitude `y1c` established -- roughly 25-40x smaller than Rung 0's own `y1` bin gaps (+0.00060 to
+0.00656). The aggregate `prediction_bias` is -0.000068, larger in raw magnitude than `y1c`'s
-0.000018 (about 3.8x), but both values are two orders of magnitude smaller than `y1`'s own
+0.00278 -- this is noise around zero, not a reopening of the systematic bias this ladder spent
Rungs 1-2 closing. Stated plainly, per this section's own framing: **`y1d` neither fixes a bias
(there was none left to fix) nor reopens one.** This is a clean pass on the non-regression check,
not a finding that moves the promotion decision either way.

### 4.6 Feature importance cross-check

Top gain-importance features (regression head, `outputs/models/regression/y1d_gradient_boosting.json`,
56 one-hot-expanded columns from the same 38 locked features):

| Rank | Feature | Gain importance | Share of total |
|---|---|---|---|
| 1 | `ball_x` | 844.0 | 16.0% |
| 2 | `ball_y` | 539.0 | 10.2% |
| 3 | `top_option_3_threat_score` | 348.0 | 6.6% |
| 4 | `top_option_1_threat_score` | 335.0 | 6.4% |
| 5 | `top_option_1_distance_from_ball` | 248.0 | 4.7% |

On **gain importance alone**, `y1d`'s top-3 columns account for only ~33% of total importance --
much more spread out than `y1c`'s ~93% Gini concentration in its own top 3. Read at face value,
this would suggest `y1d` distributes its predictive weight across a much broader set of features
than `y1c` does. **That reading does not survive a permutation-importance cross-check, and this
leg's own established discipline (section 3.6) requires checking it before reporting a change in
concentration as a real finding:**

| Rank | Feature | Permutation importance (fold held-out) |
|---|---|---|
| 1 | `ball_x` | +0.001110 |
| 2 | `ball_y` | +0.000462 |
| 3 | `on_ball_event_type_Shot` | +0.000420 |
| 4 | `on_ball_event_type_Pass` | +0.000042 |
| 5 | `top_option_3_threat_score` | +0.000017 |

By **permutation importance** -- the metric that actually measures each feature's contribution to
predictive accuracy, rather than how often it is split on -- `ball_x`, `ball_y`, and
`on_ball_event_type_Shot` account for **~95% of total positive permutation importance**, an even
higher concentration than `y1c`'s own ~93% Gini figure. Gain importance's spread-out appearance is
a property of gradient boosting's own split mechanism (many low-signal splits on `top_option_*`
threat-score/distance features accumulate gain without meaningfully moving held-out accuracy), not
evidence that `y1d` relies on genuinely broader structure than `y1c` does. **`y1d` leans on the
same already-flagged construction-linked signal `y1c`'s own report identified**
(`reports/analysis/xt_target/PASSIVE_LEAKAGE_AUDIT.json`'s Part E': `ball_x`/`ball_y` are the exact
two columns `xt_before = xT(ball_x, ball_y)` is computed from, `ball_x` at r=+0.52 vs `xt_before`;
`on_ball_event_type_Shot`'s importance is separately sensible, not a coupling artefact, for the
same reason given in section 3.6 -- Shot rows sit deep in the attacking box and are almost never
zero). This is not new structure discovered by the model-family change; it is the same
construction-linked signal `y1c` already found, reported plainly rather than presented as fresh.

### 4.7 Promotion call

**`y1d_gradient_boosting` does not clear the bar to replace `y1c_random_forest` as the standing
baseline.** The paired significance test against `y1c` (section 4.4) is **not significant** on
either test (paired-t p=0.0635, Wilcoxon p=0.125) -- the first rung comparison on this leg that
fails to clear significance at all, not merely a case of "significant but too small to matter" the
way `y1b` was at Rung 1. RMSE and R^2 do move marginally in `y1d`'s favor on both CV and held-out,
and the calibration non-regression check (section 4.5) passes cleanly -- the bias `y1c` closed
stays closed. But **held-out Spearman regresses meaningfully** (0.3154 -> 0.2732, ~13% relative),
the opposite direction from the accuracy metrics, and a statistically insignificant RMSE edge is
not sufficient on its own to promote, the same standing discipline `x1b`/`y1b` established and
`x1c` staying standing over `x1d` on the active leg confirmed most recently. Per this prompt's own
framing, this is a fully valid, reportable outcome: `y1d`'s test does not manufacture a bias-fixing
narrative that does not apply here, and the honest answer is that gradient boosting does not
improve on random forest by enough -- on this leg, at this rung -- to earn promotion.

**`y1c_random_forest` remains the leg's standing baseline.** `y1_two_stage_huber` remains
documented as the Rung-0 baseline, `y1b_quadratic` as the Rung-1 rung that did not clear its own
bar, and `y1d_gradient_boosting` as this Rung-3 rung that also did not clear its own bar (a
different, and for the first time genuinely mixed, kind of non-promotion than `y1b`'s). No further
rung is built in this prompt; no promotion audit is run for `y1d` since it was not promoted.
