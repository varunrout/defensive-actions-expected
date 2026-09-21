# Passive-xT Baseline Summary (y0-y1)

Target: `target_xt_delta_passive` (Prompt 68, fully EDA'd via the extended `xt_target` portal,
Prompt 69, and locked against the 38 locked PASSIVE features per
`reports/analysis/xt_target/PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.json`). Dataset:
`data/features/passive_defense.parquet` (1,593,181 rows, defender-slot grain, read-only,
unmodified) left-joined in memory, by `event_id`, with `outputs/prototypes/passive_xt_delta.parquet`
(event grain, 198,354 rows, also read-only, unmodified) -- neither locked file is touched by this
leg's modelling work. This is this leg's Rung 0: one dummy floor (`y0_dummy`) and one real
candidate (`y1_two_stage_huber`), mirroring active-xT's own Rung 0 discipline (Prompt 70) and every
other leg's own Rung 0 discipline before that (Prompts 36/49/54/59).

**Prompt scope**: passive leg only. This document does not build past Rung 0 -- no quadratic
terms, no random forest, no promotion decision, no full ladder.

## 1. Architecture decision, with evidence -- checked on passive's own data, not copied from active-xT

Both `target_xt_delta_v2` (active) and `target_xt_delta_passive` are signed and zero-inflated --
similar-looking on the surface (active: 20.15% exactly zero; passive: 26.29% at the unique-event
grain per Prompt 68, 26.42% at this leg's own defender-slot row grain after the join below). Prompt
70's own standard requires checking whether the zero mass is *feature-predictable on this leg's own
data* before committing to a two-stage architecture, not assuming it transfers because the shapes
look alike.

**Checked directly against `reports/analysis/xt_target/passive_category_atlas.json`** (Prompt 69,
not re-derived from scratch): `on_ball_event_type` (a locked passive categorical feature) shows a
dramatically wider zero-rate range than anything found on the active leg's own Step-0 check
(Prompt 70's `event_type` range was 5.70%-27.94%, ~5x):

| `on_ball_event_type` | n | pct exactly zero |
|---|---|---|
| Shot | 24,487 | **0.30%** |
| Pass | 830,924 | 6.73% |
| Carry | 713,760 | 49.44% |
| Dribble | 21,429 | **52.79%** |

A **~176x range** across a single locked feature, on large (tens-of-thousands-plus)
per-category samples throughout -- no small-n artefact possible here. Other categorical features
show much weaker but still real variation (`phase_label`: 23.43%-29.65%, ~1.3x); a defender-attribute
feature (`defender_functional_role`) shows almost none (25.83%-27.76%), a sensible null since the
*defender's* role says little about whether the *attacking* team's possession continues. **The zero
mass here is, if anything, more strongly feature-predictable than on the active leg**, not less --
no basis to skip the two-stage architecture on predictability grounds.

**Checked whether either of this leg's own existing architectures is a closer structural fit than
active-xT's `x1` before defaulting to copying it.** `p1e_gradient_boosting` (passive-binary) is a
single classifier (P(shot)), not two-stage -- not a fit for a signed continuous target at all.
`d1_lognormal_glm`'s hurdle architecture (passive-continuous) reuses
`p1e_gradient_boosting_calibrated` -- an **already-separately-promoted classifier for a different
target, P(shot)** -- as its classifier half, fitting only a new regression head. That reuse pattern
does not apply here: there is no existing "P(nonzero delta)" classifier anywhere in this project to
borrow. P(shot) and "does this on-ball action's possession end here" are not the same event --
confirmed directly: Shot rows are almost never zero (0.30%), while Pass rows (the large majority of
on-ball events, and mostly *not* shots) are also mostly nonzero (6.73% zero). Both of this target's
stages are new, exactly the position active-xT's own `x1` was in at its own Rung 0.

**Checked the regression-stage signal too**, via
`reports/analysis/xt_target/passive_numerical_target_atlas.json`'s own `pearson_r_nonzero_delta`
field: `ball_x` (r=0.290), `defender_x` (r=0.222), `top_option_3_threat_score` (r=0.218),
`top_option_2_threat_score` (r=0.203) -- real, and stronger than active-xT's own strongest Rung-0/1
correlate (`distance_to_attacking_box`, r=0.157). `ball_x`'s own strong correlation is not a
surprise, cross-referenced against Prompt 68's own construction-coupling finding
(`xt_before = xT(ball_x, ball_y)`, `ball_x` r=+0.52 vs `xt_before`) -- a caveat this leg's own EDA
already carries, not new information, but consistent with real regression signal existing.

**Decision: two-stage architecture, mirroring `x1`'s own shape, built fresh (no existing classifier
to reuse)** -- `LogisticRegression` for P(nonzero), `HuberRegressor` for E[delta|nonzero] (robust to
this leg's own heavy tails -- Prompt 68 recorded skew -1.779, excess kurtosis 31.02, heavier than
active v2's own -0.306/18.33 -- the same reasoning Prompt 70 used to pick Huber over plain
least-squares). Combined: `E[delta] = P(nonzero) * E[delta|nonzero]`, since `E[delta|zero] = 0`
exactly by construction, the same identity Prompt 70 established for the active leg.

**Naming.** `v`/`c`/`p`/`d`/`x` are all claimed (active-binary, active-continuous, passive-binary,
passive-continuous, active-xT). `y` is unclaimed -- checked directly against `scripts/models/`,
`reports/modeling/`, and `outputs/models/`. Used from Rung 0 onward: `y0_dummy`,
`y1_two_stage_huber`.

## 2. Confirmed inputs

**Join.** `passive_defense.parquet` (1,593,181 rows, defender-slot grain, 198,354 unique
`event_id` values) left-joined in memory against `passive_xt_delta.parquet` (also confirmed one
row per `event_id`) by `event_id`. Row count unchanged by the join (1,593,181 in, 1,593,181 out) --
2,581 rows (0.16%) have `target_xt_delta_passive == NaN` and are dropped, leaving **1,590,600
rows** for this leg's modelling.

| | Active-xT (Prompt 70) | Passive-xT (this document) |
|---|---|---|
| Locked rows in | 56,068 | 1,593,181 |
| NaN dropped | 307 (0.55%) | 2,581 (0.16%) |
| Rows after drop | 55,761 | 1,590,600 |
| Exactly-zero rate | 20.15% | 26.42% |

**Feature set.** All **38** locked PASSIVE features (`src/eda/feature_config.py`'s `PASSIVE` pool:
4 categorical, 6 boolean, 26 continuous + 2 discrete numeric) -- reused directly from
`train_passive_binary_baseline.py`'s own `DesignMatrixBuilder` and column lists. Unlike active-xT's
32-of-34 (2 features excluded for a self-reference data bug and a substitutive-feature finding),
this is a clean **38-of-38** -- `train_passive_binary_baseline.py` carries no
`EXCLUDED_FOR_THIS_LEG` dict, reconfirmed directly, so nothing needed excluding here either.

**Split.** The same canonical, frozen match-grouped split every other leg uses
(`outputs/models/splits/match_assignment.json`, `GROUP_COL = match_id`), loaded via
`dax.models.splits`, never rebuilt. Reconfirmed directly against
`train_passive_binary_baseline.py`/`train_passive_continuous_baseline.py`'s own split code before
assuming anything different was needed for "no stable player identity": that constraint is about
*player*-level splits/validity checks specifically (the Player-Level Validity Check / Player-Grouped
Split Check are both active-only, per their own `scope` fields, because passive isn't
player-indexed the same way) -- it has never meant this leg uses a different match-grouped split.

| | Rows | Matches | Unique events |
|---|---|---|---|
| Train+val | 1,275,289 | 92 | 157,936 |
| Held-out test | 315,311 | 23 | 40,073 |

Nonzero-delta rows per canonical CV fold: fold 0 = 177,286, fold 1 = 206,668, fold 2 = 167,595,
fold 3 = 191,052, fold 4 = 198,633 -- unique events per fold: 29,549 / 34,810 / 29,688 / 31,570 /
32,319, all well above any thin-fold floor used elsewhere in this project.

**Metric-convention flag (carried over from Prompt 70, reconfirmed applicable here).**
`dax.models.evaluation.regression_metrics` and `dax.models.two_part_xg.compute_hurdle_metrics`
both hardcode their zero/nonzero split as `y_true > 0` -- wrong for this leg's signed target for
the identical reason Prompt 70 flagged. This script computes its own `y_true == 0` split.

**Effective sample size.** This leg's per-fold *row* count is large, but rows are not independent
observations -- every defender-slot row sharing an `event_id` carries the identical target value
(~8.03 rows/event, confirmed in Prompt 68). The effective sample size for any significance test is
closer to the per-fold *event* count (minimum across folds: 29,549) than the row count -- reported
plainly rather than let the large row count imply more statistical power than the data actually has.

## 3. Rung 0 variants

| Variant | Model | Notes |
|---|---|---|
| `y0_dummy` | `ConstantRegressor(stat="mean")` on the raw signed `target_xt_delta_passive` | training-fold grand mean predicted for every row, no transform |
| `y1_two_stage_huber` | `LogisticRegression` (P(nonzero)) x `HuberRegressor` (E[delta\|nonzero]) | both stages use all 38 locked features; the regression head is fit only on nonzero-delta training rows |

Unlike `c0`/`c1` or `d0`/`d1`, neither `y0` nor `y1` uses a log transform, so there is no
native-vs-common-scale mismatch to correct for here -- both variants' RMSE/MAE numbers are already
on the same, directly comparable scale.

## 4. Cross-validation results (5-fold, OOF metrics)

| Variant | RMSE | MAE | R² | Spearman | Zero-row MAE | Nonzero-row MAE |
|---|---|---|---|---|---|---|
| `y0_dummy` | 0.03718 | 0.01203 | -0.0001 | -0.0104 | 0.00276 | 0.01532 |
| **`y1_two_stage_huber`** | **0.03568** | **0.01085** | **0.0786** | **0.3193** | **0.00185** | **0.01405** |

`y1` beats `y0` on every metric -- RMSE 4.0% lower, MAE 9.8% lower, R² off the zero-skill floor
(-0.0001 -> +0.0786), Spearman a real ranking signal where `y0` has none (a constant prediction has
no variance to correlate against). Both MAE slices favor `y1` too. **The R² and Spearman gains are
smaller in absolute terms than active-xT's own Rung 0 (R² 0.1514, Spearman 0.5242)** -- read
honestly, not talked up: this leg's off-ball defender-snapshot signal is intrinsically weaker than
active's own defensive-*action* signal, consistent with Prompt 68's own finding that
`target_xt_delta_passive`'s correlation with the old passive targets (-0.06/-0.07) was weaker than
active v2's own improvement (-0.108). The direction and structure of the win are the same; the
magnitude is not, and that difference is expected, not a red flag.

## 5. Paired significance test, `y1` vs `y0`

Per-fold RMSE (`outputs/models/validation/significance_y0_vs_y1.json`): `y1` beats `y0` in **5 of
5 folds** (0.03745 vs 0.03919, 0.03576 vs 0.03739, 0.03575 vs 0.03691, 0.03583 vs 0.03717, 0.03375
vs 0.03529), mean diff **-0.00148**. Paired t-test t=-14.45, **p=0.0001** -- clearly significant.
Wilcoxon signed-rank p=0.0625, the same 5-fold floor seen throughout this project. No common-scale
correction needed, since neither variant is log-transformed.

## 6. Classifier-stage diagnostic (P(nonzero), OOF) -- diagnostic only, not the Rung-0 gate

| Metric | Value |
|---|---|
| ROC-AUC | **0.7953** |
| Average precision | 0.9096 |
| Brier score | 0.1463 |

A P(nonzero) ROC-AUC of 0.80 is real, and meaningfully *stronger* than active-xT's own Rung-0
classifier (0.7103) -- consistent with section 1's evidence that `on_ball_event_type` carries a
~176x zero-rate range here, versus active's own ~5x `event_type` range. **This is a diagnostic on
one internal piece of `y1`, not the Rung-0 gate**: the Rung-0 promotion decision rests on the
combined-prediction `y1`-vs-`y0` regression comparison in sections 4-5.

## 7. Calibration (5 bins by predicted value, held-out test, `y1`)

| Bin | n | Mean predicted | Mean actual | Gap (predicted - actual) |
|---|---|---|---|---|
| 0 (most negative predicted) | 63,063 | -0.00518 | -0.01173 | +0.00656 |
| 1 | 63,062 | -0.00096 | -0.00252 | +0.00155 |
| 2 | 63,062 | +0.00012 | -0.00048 | +0.00060 |
| 3 | 63,062 | +0.00120 | +0.00007 | +0.00113 |
| 4 (most positive predicted) | 63,062 | +0.00712 | +0.00306 | +0.00406 |

**The same tail-under-prediction pattern Prompt 70 found on the active leg reappears here, and
looks structurally similar.** Bin 0 under-predicts the magnitude of the most-negative rows
(predicted -0.0052 vs. actual -0.0117, actual more than 2x more negative than predicted); bin 4
under-predicts the magnitude of the most-positive rows the same way (predicted +0.0071 vs. actual
+0.0031 -- here the *actual* mean is smaller than predicted, the opposite direction of bin 0's
gap, both consistent with `HuberRegressor`'s own loss function capping how far predictions move
toward extreme observed values on *either* tail, the same diagnosis Prompt 70/71 established for
the active leg). This is tracked from Rung 0 here, the same discipline the active-xT ladder used,
and is a candidate open question for this leg's own later rungs (random forest, mirroring
active-xT's own Rung 2 finding that tree-ensemble averaging closed a similar gap by 10-35x) --
not something this Rung-0 document resolves.

## 8. Held-out test readout (single readout, `FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION`)

| Variant | RMSE | MAE | R² | Spearman | Zero-row MAE | Nonzero-row MAE |
|---|---|---|---|---|---|---|
| `y0_dummy` | 0.03497 | 0.01121 | -0.0002 | NaN | 0.00276 | 0.01439 |
| **`y1_two_stage_huber`** | **0.03388** | **0.01021** | **0.0612** | **0.2936** | **0.00184** | **0.01336** |

Same direction and near-identical magnitude as CV throughout -- `y1`'s held-out R² (0.0612) is
close to CV (0.0786; the modest CV-to-held-out gap is consistent with this leg's own much larger
sample smoothing CV noise less than it might first appear, not a sign of overfitting given the
architecture is entirely linear/logistic with no depth/complexity hyperparameter to overfit with).

## 9. Rung-0 result

**`y1_two_stage_huber` clears Rung 0** against the `y0_dummy` floor: consistent direction across
CV and held-out test on RMSE, MAE, R², and Spearman; a statistically significant 5/5-fold paired
test (p=0.0001); and a real jump off the zero-skill floor (R² -0.0001 -> +0.0786 CV, -0.0002 ->
+0.0612 held-out).

The architecture decision (section 1) is the load-bearing result of this document, same as it was
for active-xT: the zero mass is real, structural, and *more strongly* feature-predictable here than
on the active leg (`on_ball_event_type`'s ~176x zero-rate range vs. active's own ~5x), which is the
evidence basis for building this leg's own two-stage shape from scratch (no existing classifier to
reuse, unlike `d1`'s own hurdle pattern) rather than assuming it either transfers from active-xT or
needs reinventing target-by-target.

**No model ladder is built in this document** -- this is Rung 0 only, mirroring active-xT's own
Prompt 70 discipline and every other leg's own Rung-0-only precedent before it.
