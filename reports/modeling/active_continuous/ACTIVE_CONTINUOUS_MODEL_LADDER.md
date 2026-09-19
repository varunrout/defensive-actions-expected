# Active-Continuous Model Ladder

*Split out of `ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md` (prompt 56, once a second rung existed beyond
Rung 0) so the locked Rung 0 baseline document doesn't keep growing as more rungs are tried --
mirroring the binary legs' own split point (`ACTIVE_BINARY_MODEL_LADDER.md`, Prompt 42). Rung 0 --
the two locked variants (`c0_dummy`, `c1_lognormal_glm`), their CV/held-out results, and the hurdle
pipeline readout -- stays in
[`ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md`](ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md). This document is
everything built on top of that locked Rung 0 baseline, starting from Rung 1.*

## 1. What a rung is

A **rung** is one controlled change to the locked Rung 0 baseline (`c1_lognormal_glm`): a single
variable changed -- feature shape, an interaction term, a model family -- with everything else held
identical (same 32 locked features unless the rung specifically changes that, same canonical
match-grouped 5-fold split, same `log(xg)` target on the same shot-positive rows, no ad-hoc tuning).
A rung is tested against 2 fixed gates before it can be considered for adoption as the new baseline:

1. **Paired significance test vs the current best** -- refit both variants on the same 5 canonical
   CV folds, paired t-test + Wilcoxon signed-rank test on the per-fold common-scale log RMSE
   difference (the fair metric introduced at Rung 0 to avoid comparing variants fit on different log
   transforms).
2. **Held-out-test confirmation** -- a single, one-time readout on the frozen held-out test set (768
   positive rows / 23 matches). This is the gate that actually decides the rung's fate: CV picks a
   candidate, the held-out test either confirms or rejects it, and that result is not re-litigated
   after the fact to fit a preferred narrative.

Given this leg's much smaller per-fold sample than the binary legs' (628-830 positive rows per fold
here vs tens of thousands there), every significance test on this leg carries an explicit small-n
caution: directionally informative, not strong statistical proof on its own.

Rung 0 (the baseline itself) lives in
[`ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md`](ACTIVE_CONTINUOUS_BASELINE_SUMMARY.md). This ladder builds
on that document's `c1_lognormal_glm` pick (held-out common-scale log RMSE 0.9951, corrected R²
0.0371) as the baseline every rung is compared against.

## 2. Rung 1 -- `c1b_quadratic` (prompt 55)

### 2.1 Reconfirming shot-conditional nonlinearity from real data, not the rough chat shortlist

The rough chat check that motivated this rung looked at 7 features and found
`visible_attacker_count` (pearson -0.13, spearman -0.16) as the strongest, with `defenders_within_10m`
flagged for a "mild threshold-like jump at its top bin." `scripts/analysis/check_shot_conditional_nonlinearity.py`
reproduces this properly across all 32 locked features on the real shot-conditional subset (n=4,368),
computing Pearson/Spearman for numeric features, point-biserial for boolean features, and ANOVA
eta-squared for categorical features (quadratic terms are only meaningful for the 17 numeric
features -- squaring a 0/1 boolean or a one-hot categorical dummy is a no-op, stated explicitly
rather than silently skipped). Full output:
`outputs/models/validation/shot_conditional_nonlinearity_check.json`.

**The systematic check confirms the chat shortlist's top feature and finds one it missed.**
`visible_attacker_count` is confirmed as the single strongest association across all 32 features
(|r|=0.161) -- and `visible_defender_count` (|r|=0.135), a feature the 7-feature chat shortlist
never checked, ranks 3rd overall. (The chat shortlist's `distance_to_center_line` is not usable as a
check at all -- reconfirmed here that it is not one of the 32 locked ACTIVE features; it was dropped
from the locked set for redundancy with `attacking_goal_centrality` before this leg's work began.)

**Curvature, specifically**: the quintile-binning diagnostic (5 roughly-equal bins of each numeric
feature, mean `log(xg)` per bin) flags several features as non-linear, but most of those have
near-zero absolute correlation (7 features between |r|=0.005 and |r|=0.05) -- indistinguishable from
sampling noise on ~870-900-row bins drawn from a target with excess kurtosis ~13-14, and **not used
as quadratic candidates**. Only 3 features clear a real signal-plus-shape bar:

| Feature | \|r\| | Curvature shape | Note |
|---|---|---|---|
| `visible_attacker_count` | 0.161 | accelerating monotonic (top-bin jump &gt;2x other gaps) | strongest feature overall, real signal + real curvature |
| `defenders_within_10m` | 0.088 | accelerating monotonic (top-bin jump &gt;2x other gaps) | the specific feature the rough chat check flagged -- reconfirmed, not just asserted |
| `defender_spread` | 0.052 | genuine but asymmetric U-shape | weaker evidence than the top two; included as the marginal case it is |

`QUADRATIC_FEATURES_C1B = [visible_attacker_count, defenders_within_10m, defender_spread]` -- 3
features, not the active-binary leg's 4, and none of the same features (that leg's U-shaped set --
`defender_spread`, `distance_to_attacking_box`, `visible_defender_count`,
`defenders_between_ball_and_attacking_goal` -- was computed on the full zero+positive dataset and is
not reused here; the one name that happens to overlap, `defender_spread`, was independently
reconfirmed on this leg's own shot-conditional data, not carried over by assumption).

### 2.2 `c1b_quadratic` results

Same 32 locked features, same `DesignMatrixBuilder` and `LinearRegression` as `c1`, plus one
`quad__<feature>` column per feature in `QUADRATIC_FEATURES_C1B` above.

| Variant | Common log RMSE (CV) | Common log RMSE (held-out) | Naive MAE (held-out) | Corrected R² (held-out) |
|---|---|---|---|---|
| `c1_lognormal_glm` | 1.0014 | 0.9951 | 0.0778 | 0.0371 |
| `c1b_quadratic` | 1.0027 | 0.9975 | 0.0779 | 0.0349 |

**`c1b_quadratic` does not beat `c1` -- on either CV or held-out, on either the common-scale metric
or original-scale MAE/R².** The common-scale log RMSE is marginally *worse* for `c1b` in both CV
(+0.0013) and held-out (+0.0024), and corrected R² is also marginally worse (0.0349 vs 0.0371).

### 2.3 Significance test, c1b vs c1

`outputs/models/validation/significance_c1_vs_c1b_quadratic.json`: mean diff (c1b - c1) common-scale
log RMSE = **+0.0014** (c1b worse, not better), paired t-test **p=0.2073**, Wilcoxon p=0.1875. **Does
not clear significance in either direction** -- this is not a "significant regression" so much as a
null result: the quadratic terms make essentially no detectable difference, worse or better.

### 2.4 This is a real finding about this leg, not a modelling failure

Stated plainly, per prompt 55's own explicit instruction not to bury a null result under
favourable-sounding framing: **`c1b_quadratic` shows no real improvement over `c1`, and this
connects directly to section 2.1's weak-correlation finding, not a coincidence.** Every one of the 32
locked features correlates weakly with shot quality once a shot is already guaranteed to happen
(strongest at |r|=0.16) -- squared terms on 3 of the strongest-available features simply don't have
much genuine curvature signal to capture once the (already weak) linear terms have done their part.
The surviving quadratic coefficients themselves are small (`quad__defenders_within_10m` +0.022,
`quad__visible_attacker_count` -0.015, `quad__defender_spread` +0.015) -- consistent with "present
but negligible," not "the model found nothing at all."

This is the same kind of honest ladder-rung outcome as the active-binary leg's own Rung 1 (Prompt
39, `v1b_quadratic` also failed to survive held-out test there, for different underlying reasons).
The active-continuous leg's shot-conditional signal is simply weak -- this was flagged as a real
possibility in prompt 55's own brief before any code was written, and the real data bears it out.
**`c1_lognormal_glm` remains this leg's best Rung-0/Rung-1 candidate**; `c1b_quadratic` is not
promoted or preferred over it.

## 3. Rung 2 -- `c1c_systematic_interactions` is SKIPPED (prompt 56)

The rung that would mirror the active-binary leg's own Rung 2 (Prompt 41 -- full pairwise
interactions + squares over the numeric features) is **explicitly skipped, not forgotten**, on two
independent pieces of evidence that already exist and do not need to be re-derived:

**Evidence 1 -- the shot-conditional interaction panel is already 10/10 inconclusive.**
`reports/analysis/xg_target/FEATURE_INTERACTION_ANALYSIS.json` already ran the shot-conditional
(given-shot) version of the interaction-pair panel on the same 5 hand-picked active-leg numeric x
numeric pairs used for the binary-target leg's own interaction analysis:

| Pair | Marginal delta, given shot | vs threshold (0.052053) |
|---|---|---|
| `defenders_within_10m` x `distance_to_attacking_box` | +0.030442 | below (~0.6x) |
| `visible_defender_count` x `attacker_spread` | -0.029744 | below (~0.6x) |
| `defenders_within_5m` x `defenders_within_10m` | -0.004408 | below by ~12x (order of magnitude) |
| `possession_elapsed_seconds` x `match_time_seconds` | +0.003062 | below by ~17x (order of magnitude) |
| `defenders_between_ball_and_attacking_goal` x `attacker_defender_ratio` | -0.006192 | below by ~8x |

All 5 active-leg pairs classify `inconclusive` given a shot (the passive leg's 5 pairs are also
10/10 inconclusive, for the same reason: `xg`-given-shot is a much noisier, lower-signal target than
shot occurrence itself). This was decided 2026-09-17 as a final result, not loosened to force a
classification -- see the file's own `given_shot_threshold_decision` field for the full reasoning
(loosening the threshold specifically because these pairs fail to clear it would be tuning the
threshold to the result, which this project's methodology has consistently avoided).

**Evidence 2 -- a direct systematic check already ran and found nothing worth the complexity.** A
rough chat-derived check ran the actual `c1c`-shaped model directly: all 17 numeric ACTIVE features,
full pairwise interactions + squares (`PolynomialFeatures(degree=2)`, 170 columns), Ridge-regularized,
5-fold CV on the same 4,368 positive rows. Plain additive linear regression scored mean CV
**R²=0.047**; the best Ridge-regularized full-interaction expansion (alpha=100) reached only
**R²=0.055** -- a gain smaller than the fold-to-fold noise itself (individual folds ranged 0.001-0.09
under the same regularization), and lighter regularization actively made results *worse* (one fold
went negative).

**Decision: `c1c_systematic_interactions` is not built.** Two independent checks -- a targeted
5-pair given-shot interaction panel and a full systematic polynomial-expansion regression -- both
already answer the question a `c1c` rung would ask, and both answer it the same way: hand-built or
brute-force interaction/quadratic terms do not find usable signal on this leg's weak,
~4,368-row shot-conditional target. This is a closed decision for this leg, not deferred; it would
only be revisited if new evidence specifically contradicted both findings above.

## 4. Rung 2 -- `c1d_random_forest` (prompt 56)

Instead of the interaction rung, Rung 2 goes straight to a `RandomForestRegressor` on the same 32
locked features, raw (non-standardized, non-polynomial-expanded) design matrix -- mirroring the
active-binary leg's own Rung 3 (`v1d_random_forest`, Prompt 44). The question: can automatic,
sample-efficient nonlinearity/interaction discovery find signal that neither the hand-built quadratic
terms (Rung 1) nor the brute-force polynomial expansion (section 3 above) could?

### 4.1 Hyperparameter grid, sized for this leg's small sample

The active-binary leg's own Rung 3 grid (`N_ESTIMATORS_GRID=[300,600]`,
`MAX_DEPTH_GRID=[8,14,None]`, `MIN_SAMPLES_LEAF_GRID=[5,20]`) was sized for ~9,000 rows per training
fold. This leg's positive-row subset is a fraction of that (3,600 trainval rows total, ~2,880 per
training fold in 5-fold CV) -- reusing the binary leg's `min_samples_leaf` floor of 5 would risk
badly overfit leaves. The grid used here: `N_ESTIMATORS_GRID=[200]` (fixed; more trees rarely hurts
and this dataset is cheap), `MAX_DEPTH_GRID=[4, 8, None]`, `MIN_SAMPLES_LEAF_GRID=[10, 30, 60]` (9
combinations total).

**CV-internal overfitting is real and reported plainly, per prompt 56's own instruction not to
blindly pick the train-CV-best config.** Full grid (chosen by lowest OOF common-scale log RMSE, not
by training-fold fit):

| n_estimators | max_depth | min_samples_leaf | OOF common log RMSE | OOF corrected R² | mean train R² | train &minus; OOF gap |
|---|---|---|---|---|---|---|
| 200 | 4 | 10 | 0.9911 | 0.0500 | 0.1632 | 0.1131 |
| 200 | 4 | 30 | 0.9914 | 0.0459 | 0.1494 | 0.1035 |
| 200 | 4 | 60 | 0.9928 | 0.0405 | 0.1344 | 0.0939 |
| 200 | 8 | 10 | **0.9801** | **0.0597** | 0.3641 | 0.3044 |
| 200 | 8 | 30 | 0.9828 | 0.0535 | 0.2565 | 0.2030 |
| 200 | 8 | 60 | 0.9877 | 0.0452 | 0.1835 | 0.1383 |
| 200 | None | 10 | 0.9804 | 0.0496 | 0.5089 | 0.4594 |
| 200 | None | 30 | 0.9826 | 0.0525 | 0.2779 | 0.2253 |
| 200 | None | 60 | 0.9876 | 0.0451 | 0.1850 | 0.1399 |

`max_depth=None, min_samples_leaf=10` shows the worst overfitting of the grid (train R²=0.51 vs OOF
R²=0.05, a 0.46 gap) yet does *not* win on OOF score -- confirming the deepest/least constrained tree
is memorizing noise, not finding real signal, exactly the failure mode prompt 56 flagged as a risk to
watch for. The chosen config (`max_depth=8, min_samples_leaf=10`) still shows a real gap (train
R²=0.36 vs OOF R²=0.06, 0.30) but wins on OOF score cleanly -- shallower configs (`max_depth=4`) have
smaller gaps but *also* worse OOF scores across the board, so constraining further would trade away
real signal to chase a smaller (but not zero) overfit gap. The selection rule used throughout this
leg -- lowest OOF/held-out score, not training-fold fit -- is the correct one here specifically
because it already penalizes the overfit configs on their own terms.

### 4.2 `c1d_random_forest` vs `c1_lognormal_glm` -- ranking and calibration, reported separately

**Ranking (common-scale log RMSE, lower is better) -- `c1d_random_forest` wins, consistently:**

| Variant | Common log RMSE (CV) | Common log RMSE (held-out) |
|---|---|---|
| `c1_lognormal_glm` | 1.0014 | 0.9951 |
| `c1d_random_forest` | 0.9801 | 0.9553 |

Paired significance test (`outputs/models/validation/significance_c1_vs_c1d_random_forest.json`):
mean diff (c1d &minus; c1) = **-0.0211** (c1d better), paired t-test **p=0.0126**, Wilcoxon
**p=0.0625**. `c1d_random_forest`'s common-scale log RMSE is lower than `c1`'s on **all 5 of 5 CV
folds** (0.940 vs 0.964, 0.950 vs 0.983, 1.056 vs 1.084, 1.056 vs 1.062, 0.889 vs 0.903) -- a
consistent, if modest, directional win, and the paired t-test clears p&lt;0.05 (Wilcoxon is
borderline at p=0.0625, expected with only 5 paired observations). Same small-n caution as every
other significance test on this leg applies: 5 folds is directionally informative, not proof.

**Calibration and original-scale accuracy (held-out test) -- also favors `c1d_random_forest`, with
one caveat:**

| Variant | Naive MAE | Naive R² | Naive Spearman | Corrected MAE | Corrected R² | Corrected prediction bias |
|---|---|---|---|---|---|---|
| `c1_lognormal_glm` | 0.0778 | -0.0638 | 0.3078 | 0.0863 | 0.0371 | -0.0089 |
| `c1d_random_forest` | 0.0745 | -0.0152 | 0.3781 | 0.0786 | 0.0925 | -0.0223 |

`c1d_random_forest` beats `c1` on every accuracy metric here -- lower MAE (both back-transforms),
higher R² (both), and meaningfully better rank correlation (Spearman 0.378 vs 0.308). **The one
place it is worse**: corrected prediction bias is larger in magnitude (-0.0223 vs -0.0089) -- the RF
under-predicts mean xG by about 2.5x as much as `c1` does on the held-out set, even though its
*variance*-driven error (RMSE/MAE) is smaller. Ranking and calibration mostly point the same
direction here (unusual for this leg, where Rung 1 saw them roughly tied and both flat) but are not
identical, and the bias gap is reported rather than smoothed over.

### 4.3 Feature importance cross-check against the linear-correlation ranking and the 3 quadratic candidates

Top 10 (of 15 reported) features by Gini importance and by fold-held-out permutation importance
(`outputs/models/regression/c1d_random_forest.json`):

| Rank | Gini importance | Permutation importance (fold held-out) |
|---|---|---|
| 1 | `attacking_goal_centrality` (0.1178) | `attacking_goal_centrality` (0.0886) |
| 2 | `distance_to_attacking_box` (0.0925) | `distance_to_attacking_box` (0.0354) |
| 3 | `visible_attacker_count` (0.0743) | `event_type_Clearance` (0.0235) |
| 4 | `match_time_seconds` (0.0739) | `visible_attacker_count` (0.0188) |
| 5 | `defender_spread` (0.0642) | `angle_to_attacking_goal` (0.0080) |
| 6 | `angle_to_attacking_goal` (0.0567) | `defender_spread` (0.0076) |
| 7 | `defender_attacker_gap_x` (0.0527) | `visible_defender_count` (0.0076) |
| 8 | `nearest_attacker_distance` (0.0511) | `nearest_attacker_distance` (0.0060) |
| 9 | `attacker_spread` (0.0495) | `event_type_Ball Recovery` (0.0045) |
| 10 | `defender_attacker_gap_y` (0.0482) | `defenders_between_ball_and_attacking_goal` (0.0027) |

**Cross-check against Rung 0/1's systematic correlation ranking
(`shot_conditional_nonlinearity_check.json`)**: the two top RF features tell two different stories.
`attacking_goal_centrality` was already the correlation ranking's #2 feature (|r|=0.1405) -- RF
agrees this is important, no surprise. But `distance_to_attacking_box` ranked only **#21 of 32** by
absolute linear correlation (|r|=0.0205) and jumps to **#2 by both Gini and permutation
importance** -- this is exactly the kind of thing a tree can see that a linear correlation coefficient
cannot: the feature likely carries real information through a nonlinear shape or an interaction with
another feature (e.g. shot angle, or defender positioning) rather than a monotonic linear relationship
with `log(xg)`. This is the headline new finding from this rung, not a footnote.

**Cross-check against the 3 Rung-1 quadratic candidates** (`visible_attacker_count`,
`defenders_within_10m`, `defender_spread`):

- **`visible_attacker_count`** (correlation rank #1, |r|=0.1611): RF ranks it #3 by Gini (0.0743),
  #4 by permutation (0.0188) -- still clearly important under RF, just not the single most important
  feature once the tree can also use `attacking_goal_centrality` and `distance_to_attacking_box`.
  Consistent with a real, strong, roughly-linear-plus-mild-curvature signal (which is exactly what
  `c1b_quadratic`'s own coefficient table found for this feature).
- **`defender_spread`** (correlation rank #9, |r|=0.0521): RF ranks it **higher** relative to its
  linear-correlation position -- #5 by Gini (0.0642), #6 by permutation (0.0076). This is mild
  positive evidence that the tree is finding real (nonlinear/interaction) information in this feature
  beyond what a Pearson correlation captures, consistent with `c1b`'s own U-shape finding for this
  feature even though `c1b`'s linear quadratic term for it was small.
- **`defenders_within_10m`** (correlation rank #7, |r|=0.0879): RF ranks it **#16 by Gini (0.0152)
  and outside the permutation top 20 entirely** -- the tree does *not* confirm this feature as
  strongly informative, despite it being the specific feature the original rough chat check flagged
  for a "threshold-like jump." This lines up with `c1b_quadratic`'s own result, where this feature's
  quadratic coefficient (+0.022) was the smallest and least reliable of the three quadratic terms.

**Interpretation**: 2 of the 3 Rung-1 quadratic candidates (`visible_attacker_count`,
`defender_spread`) get some independent support from RF importance; `defenders_within_10m` does not.
This is a coherent cross-check, not a coincidence -- it agrees with which of `c1b`'s three quadratic
terms actually looked substantive versus negligible.

### 4.4 Ladder-gate outcome for this rung, no promotion audit

**`c1d_random_forest` clears the Rung-2 gate against `c1_lognormal_glm`** -- a consistent (5-of-5
folds), statistically significant-by-paired-t-test (p=0.0126) improvement on the fair common-scale
ranking metric, and a clean win on every held-out accuracy metric except prediction bias magnitude.
This is a genuinely positive result for the leg, in contrast to Rung 1's null result, and it connects
back to this rung's own feature-importance cross-check: `distance_to_attacking_box`'s jump from
correlation rank #21 to importance rank #2 is concrete evidence that there *was* real nonlinear/
interaction signal in this 32-feature set all along -- Rung 1's hand-picked quadratic terms and the
brute-force polynomial expansion (section 3 above) simply weren't the right shape to find it, while a
tree-based model was. Per prompt 56's explicit constraint, no promotion audit is run in this prompt
regardless of this result -- this section reports the Rung-2 gate outcome only, the same discipline
the active-binary leg's own Prompt 44 followed for its own Rung 3.
