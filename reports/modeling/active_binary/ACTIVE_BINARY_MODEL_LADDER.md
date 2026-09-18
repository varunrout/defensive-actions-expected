# Active-Binary Model Ladder

> **Status update (Prompt 46):** Rung 4's calibrated variant (`v1e_gradient_boosting_calibrated`)
> has been promoted to standing reference model for the active-binary leg, superseding Rung 2's
> `v1c_systematic_interactions` (itself promoted in Prompt 43, replacing `v1_unweighted`). See the
> full promotion decision in `reports/eda/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` section 8.

*Split out of `ACTIVE_BINARY_BASELINE_SUMMARY.md` (Prompt 42) so the locked Rung 0 baseline
document doesn't keep growing as more rungs are tried. Rung 0 -- the 4 locked variants
(`v0_dummy`, `v1_unweighted`, `v2_weighted`, `v3_weighted_interactions`), their CV/held-out
results, and the Prompt 38 post-lock validation -- stays in
[`ACTIVE_BINARY_BASELINE_SUMMARY.md`](ACTIVE_BINARY_BASELINE_SUMMARY.md). This document is
everything built on top of that locked baseline, starting from Rung 1.*

## 1. What a rung is

A **rung** is one controlled change to the locked Rung 0 baseline (`v1_unweighted`): a single
variable changed -- feature shape, an interaction term, a model family -- with everything else
held identical (same 32 locked features unless the rung specifically changes that, same
canonical `StratifiedGroupKFold` split, same target, no ad-hoc tuning). A rung is tested against
3 fixed gates before it can be considered for adoption as the new baseline:

1. **Paired significance test vs the current best** -- refit both variants on the same 5
   canonical CV folds, paired t-test + Wilcoxon signed-rank test on the per-fold PR-AUC
   difference.
2. **Held-out-test confirmation** -- a single, one-time readout on the frozen 23-match test set.
   This is the gate that actually decides the rung's fate: CV picks a candidate, the held-out
   test either confirms or rejects it, and that result is not re-litigated after the fact to fit
   a preferred narrative.
3. **Player-disjoint leakage recheck** -- a `GroupKFold(player_id)` fit-and-score run on
   train+val rows only (the frozen held-out test set is never touched by this check). A richer
   or differently-shaped feature space is more leakage-prone in principle, so this is re-run for
   every rung, not assumed to still hold because an earlier rung cleared it.

Rung 0 (the baseline itself) and its own validation live in
[`ACTIVE_BINARY_BASELINE_SUMMARY.md`](ACTIVE_BINARY_BASELINE_SUMMARY.md). This ladder builds on
that document's `v1_unweighted` pick (held-out PR-AUC 0.372, ROC-AUC 0.819, ECE 0.011) as the
baseline every rung is compared against.

## 2. Rung 1 -- quadratic terms for U-shaped features (`v1b_quadratic`)

EDA (`active_numerical_target_atlas.json`, cross-checked in the baseline summary's section 6) already
confirmed 4 of the 32 locked features have a U-shaped, non-monotonic
relationship with the target: `defender_spread`, `distance_to_attacking_box`,
`visible_defender_count`, `defenders_between_ball_and_attacking_goal`. A plain
logistic coefficient is a single slope and structurally cannot represent a
U-shape. `v1b_quadratic` is v1 plus one extra column per feature -- the square
of that feature's already-standardized value -- with everything else held
identical: same 32 base features, same preprocessing, same canonical split,
still **unweighted** (no `class_weight`, isolating shape from reweighting),
same `LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)`.
36 total input columns (32 + 4 quadratic terms). Built by
`scripts/train_v1b_quadratic.py`, which extends
`DesignMatrixBuilder`/`fit_predict_fold` in `train_active_binary_baseline.py`
(`add_quadratic=True`) rather than duplicating that logic.

### 2.1 CV and held-out test results

| | PR-AUC | ROC-AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|
| v1 (CV, OOF) | 0.3521 | 0.8177 | 0.2209 | 0.0614 | 0.0036 |
| v1b_quadratic (CV, OOF) | 0.3542 | 0.8193 | 0.2203 | 0.0613 | 0.0027 |
| v1 (held-out test) | 0.3725 | 0.8193 | 0.2047 | 0.0555 | 0.0107 |
| v1b_quadratic (held-out test) | 0.3713 | 0.8204 | 0.2046 | 0.0556 | 0.0101 |

Calibration stayed good on both readouts (ECE actually improved slightly, CV
0.0036&rarr;0.0027 and test 0.0107&rarr;0.0101) -- as expected, since the
training objective (plain unweighted MLE) didn't change, only the input
features did. No red flag here.

### 2.2 Is the CV lift real? (`significance_v1_vs_v1b_quadratic.json`)

Paired over the same 5 canonical CV folds: mean PR-AUC difference
**+0.0021** (std 0.0011), v1b_quadratic ahead of v1 in **5 of 5 folds**.
Paired t-test: t=4.34, **p=0.0122** -- significant at the conventional 0.05
threshold. Wilcoxon: p=0.0625, the same n=5 exact-test floor seen in the baseline summary's section
8.1 (every fold the same sign, no smaller rank-sum configuration possible
with 5 pairs).

So, statistically: the CV lift is real and directionally consistent (5/5
folds), and the t-test calls it significant. But the *effect size* is small
-- an order of magnitude smaller than v1's own edge over v2/v3 (+0.0021 here
vs +0.0142 / +0.0182 in the baseline summary's section 8.1) -- and it does **not** show up on the
single held-out-test readout at all: v1b_quadratic's test PR-AUC (0.3713) is
marginally *below* v1's (0.3725), a -0.0012 difference. That's within normal
single-readout noise (not a repeated/paired comparison on the test set, which
was deliberately touched only once per variant), but it means the CV-measured
lift does not clearly generalise to an improvement on held-out data.

**Honest read: this is the small/negative-information case, not the
new-best-baseline case.** Curvature on these 4 specific U-shaped features
buys a real but tiny, borderline-material CV improvement and no held-out
improvement. It does not close a meaningful fraction of whatever gap remains
to a strong classifier. That is itself useful: it says the next rung of the
ladder should look at systematic interaction discovery or a different model
family (e.g. gradient boosting, which captures arbitrary feature shape and
interactions natively) rather than more feature engineering on these
specific 4 features.

**What the model actually learned about the 4 features**
(`v1b_quadratic.json` coefficients): the quadratic term for
`distance_to_attacking_box` is by far the largest of the four
(`quad__distance_to_attacking_box` = **+0.278**, vs its linear term at -0.015)
-- confirming the baseline summary's section 6's flagged mismatch (the *linear* v2 coefficient for
this feature had the wrong sign relative to its weak overall Spearman
direction) was indeed a shape problem: a positive quadratic term means the
model now represents elevated risk at both distance extremes, consistent
with a U-shape opening upward. `defender_spread`'s quadratic term is small
and positive (+0.047, same direction as its documented U-shape).
`visible_defender_count` (-0.032) and
`defenders_between_ball_and_attacking_goal` (-0.062) both got small
*negative* quadratic terms, which is the opposite curvature direction from
their documented (weak) U-shapes -- consistent with those two having the
weakest univariate signal of the four to begin with
(`active_numerical_target_atlas.json`: near-zero rho for both), so the
quadratic term is likely fitting noise rather than a real curvature
correction for those two specifically.

### 2.3 Player-disjoint re-check (`player_disjoint_v1b_quadratic.json`)

A richer feature space is more leakage-prone in principle, so this was
re-run for `v1b_quadratic` specifically rather than assumed to still hold
because v1 cleared it. Same protocol as the baseline summary's section 8.3: 5-fold `GroupKFold` on
`player_id`, train+val rows only, held-out test set untouched.
`player_overlap_train_test == 0` confirmed and asserted for every fold.

| | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| v1 player-disjoint CV | 0.3586 &plusmn; 0.0241 | 0.8202 &plusmn; 0.0072 | 0.0054 &plusmn; 0.0012 |
| v1b_quadratic player-disjoint CV | 0.3606 &plusmn; 0.0254 | 0.8216 &plusmn; 0.0084 | 0.0057 &plusmn; 0.0020 |

No meaningful divergence from v1's own player-disjoint numbers -- the 4
quadratic terms did not introduce a detectable leakage signal.

### 2.4 Bottom line

`v1b_quadratic` is **not** promoted to "the" baseline by this prompt (per
constraints, that decision waits for the rest of the ladder). Result:
curvature on the 4 known U-shaped features gives a statistically real but
practically tiny CV lift (+0.0021 PR-AUC, p=0.0122) that does not clearly
survive to the held-out test set (-0.0012, within single-readout noise but
not a positive number). Calibration and player-disjoint robustness both hold
up. Read together, this is negative information in the useful sense: these 4
features' curvature is not where v1's remaining headroom lives, so rung 2
should test systematic interaction discovery or a genuinely non-linear model
family rather than more polynomial terms on these specific features.

Charts: `outputs/models/classification/charts/v1b_quadratic/` (calibration curve, PR curve,
ROC curve, prediction distribution).

## 3. Rung 2 -- systematic interactions (L1) (`v1c_systematic_interactions`)

`v3_weighted_interactions` (Rung 0) tried interactions too, but only 5 manually-chosen pairs,
combined with `class_weight="balanced"` -- which independently hurts both PR-AUC and calibration
(baseline summary sections 2 and 8) -- so v3 never isolated whether interactions specifically
help. This rung isolates that variable properly: same unweighted objective as v1, but the model
discovers interactions systematically instead of 5 hand-picked pairs.

`v1c_systematic_interactions` keeps v1's categorical one-hot and boolean-passthrough blocks
unchanged. The 17 numeric features (already median-imputed and standardized, same as v1) are
additionally expanded with `sklearn.preprocessing.PolynomialFeatures(degree=2,
interaction_only=False, include_bias=False)` fit on the standardized numeric block only: C(17,2) =
136 pairwise products + 17 squared terms = 153 new columns, kept alongside (not instead of) the 17
original standardized numeric columns. Final design matrix: 59 categorical one-hot + 17 numeric
originals + 9 boolean + 153 poly terms = **238 columns total** (confirmed programmatically, not
estimated). No `class_weight` -- unweighted, same objective as v1, isolating this rung from v3's
confound. `LogisticRegression(penalty="l1", solver="liblinear", C=<tuned>, random_state=42)`: L1
so the model can zero out candidate terms that don't earn their place, rather than all 153
getting a nonzero coefficient by default the way L2 would.

### 3.1 Tuning C (CV only, held-out test never touched)

Grid-searched `C` on the 5 canonical CV folds, train+val rows only:

| C | OOF PR-AUC |
|---|---|
| 0.001 | 0.1953 |
| 0.01 | 0.2925 |
| **0.1** | **0.3630 (chosen)** |
| 1.0 | 0.3603 |
| 10.0 | 0.3592 |

The curve brackets a clear interior optimum at C=0.1 (PR-AUC rises steeply from 0.001 to 0.1, then
declines slightly and flattens from 0.1 to 10.0) -- not a boundary solution, so the grid didn't
need widening. The held-out test set was not referenced anywhere in this tuning step.

**A methodological caveat, reported rather than glossed over:** `liblinear`'s default tolerance
(1e-4) took ~270s per fit on this 238-column, heavily collinear design matrix (products/squares of
correlated base features are themselves correlated) -- intractable across a 5C x 5-fold grid plus
two gates. Tolerance was relaxed to 1e-3 (~90s/fit) to make the run tractable. Tested at C=1.0: the
two tolerances gave a similar nonzero-term count (223 vs 226 of the full design matrix) but a
non-trivial intercept difference (-2.50 vs -1.65) -- expected behaviour for L1 on collinear
features (the solution path is less sharply determined when candidate columns are correlated), not
a bug, but a real precision/runtime tradeoff. The qualitative conclusions below (which terms
survive, the ranking of variants, the significance results) are not sensitive to this -- but the
exact coefficient values at the margin should be read as indicative, not four-decimal-precise.

### 3.2 CV and held-out test results, at C=0.1

| | PR-AUC | ROC-AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|
| v1 (CV, OOF) | 0.3521 | 0.8177 | 0.2209 | 0.0614 | 0.0036 |
| v1c (CV, OOF) | **0.3630** | **0.8288** | 0.2172 | 0.0608 | 0.0026 |
| v1 (held-out test) | 0.3725 | 0.8193 | 0.2047 | 0.0555 | 0.0107 |
| v1c (held-out test) | **0.3747** | **0.8321** | 0.2011 | 0.0551 | 0.0103 |

Unlike rung 1, **the gain holds on held-out test**: v1c's PR-AUC is higher than v1's on both CV
(+0.0109) and held-out test (+0.0022) -- same direction both times, not a CV-only artifact.
Calibration stayed as good as v1's or slightly better throughout (ECE 0.0026 CV / 0.0103 test vs
v1's 0.0036 / 0.0107) -- no calibration cost for the added flexibility, unlike v2/v3's
`class_weight="balanced"` cost.

### 3.3 Is the gain real? (`significance_v1c_vs_v1_v1b.json`)

Paired over the same 5 canonical CV folds, v1c beats **both** v1 and v1b_quadratic in **5 of 5
folds**:

| Comparison | Mean &Delta; PR-AUC | Std &Delta; | Paired t (p) | Wilcoxon (p) |
|---|---|---|---|---|
| v1c vs v1 | +0.0102 | 0.0047 | t=4.92, **p=0.0079** | W=0, p=0.0625 |
| v1c vs v1b_quadratic | +0.0081 | 0.0048 | t=3.75, **p=0.0199** | W=0, p=0.0625 |

Both paired t-tests are significant at the conventional 0.05 threshold. Wilcoxon sits at the same
n=5 exact-test floor (p=0.0625) seen in every prior rung comparison in this ladder -- the strongest
signal 5 same-signed pairs can give a rank test, not a weaker result than the t-test, just a
differently-bounded one. This is a materially larger, more consistent effect than rung 1's
(+0.0021 CV, non-significant-on-test): v1c's edge over v1 (+0.0102 CV) is roughly **5x** rung 1's
CV lift, and unlike rung 1, it is confirmed, not contradicted, by the held-out readout.

### 3.4 Player-disjoint re-check (`player_disjoint_v1c.json`)

A ~238-column model is meaningfully more flexible than v1's 32 (or rung 1's 36), so this was
re-checked from scratch rather than assumed to still hold. 5-fold `GroupKFold` on `player_id`,
train+val rows only, held-out test set untouched. `player_overlap_train_test == 0` confirmed and
asserted for every fold (964 total unique players in train+val).

| | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| v1 player-disjoint CV | 0.3586 &plusmn; 0.0241 | 0.8202 &plusmn; 0.0072 | 0.0054 &plusmn; 0.0012 |
| v1c player-disjoint CV | 0.3681 &plusmn; 0.0240 | 0.8311 &plusmn; 0.0087 | 0.0060 &plusmn; 0.0010 |

v1c's player-disjoint PR-AUC (0.368) is close to its own match-grouped CV number (0.363, section
3.2) -- no meaningful drop, and it is *higher* than v1's player-disjoint number, consistent with
v1c's overall edge over v1 elsewhere. No new leakage signal from the larger feature space.

### 3.5 Which interactions survived L1, and does that match what was already flagged?

116 of the 153 candidate interaction/quadratic terms survived with a nonzero coefficient at C=0.1
-- not a highly sparse solution (76% of candidates kept some weight), consistent with C=0.1 being
a fairly mild regularization strength once the more heavily-regularized C=0.001/0.01 were rejected
by the CV grid.

**Top 5 surviving terms by |coefficient|:**

| Term | Coefficient |
|---|---|
| `distance_to_attacking_box`&sup2; | +0.322 |
| `distance_to_attacking_box` &times; `defender_attacker_gap_x` | -0.211 |
| `defender_attacker_gap_x`&sup2; | -0.207 |
| `nearest_attacker_distance`&sup2; | +0.142 |
| `attacking_goal_centrality` &times; `visible_defender_count` | +0.131 |

The largest surviving term, `distance_to_attacking_box`&sup2; (+0.322), is directionally
consistent with rung 1's finding that this feature's linear coefficient had the "wrong" sign
relative to its weak univariate direction because of unrepresented curvature (rung 1, section 2.2)
-- both rungs independently point at the same feature needing a non-linear term, discovered two
different ways.

**Cross-check against `reports/eda/FEATURE_INTERACTION_ANALYSIS.json`'s 4 pre-flagged "interactive"
pairs** (that file only pre-tested 5 candidate pairs total via direct stratified rate analysis, not
a systematic search of all 136 -- so this checks whether the systematic search rediscovers a small,
independently-derived reference set, not the other way around):

| Flagged pair | Survived in v1c? | Coefficient / rank |
|---|---|---|
| `defenders_within_10m` &times; `distance_to_attacking_box` | Yes | -0.038 (rank 44/116) |
| `possession_elapsed_seconds` &times; `match_time_seconds` | Yes | +0.034 (rank 48/116) |
| `visible_defender_count` &times; `attacker_spread` | **No** -- zeroed out by L1 | -- |
| `defenders_between_ball_and_attacking_goal` &times; `attacker_defender_ratio` | **No** -- zeroed out by L1 | -- |

Mixed result, reported honestly: 2 of the 4 pre-flagged pairs survived (both mid-ranked, not
top-15), and 2 were zeroed out. All three possible outcomes the task anticipated actually happened
at once -- the systematic search **rediscovered** some already-flagged pairs, **found new** ones
the small pre-vetted list never tested (the entire top-5 above, e.g. `distance_to_attacking_box`
squared and its interaction with `defender_attacker_gap_x`, neither of which was in the original
5-pair candidate set), and **dropped** 2 of the 4 flagged pairs once given L1's chance to compete
them against 149 other candidates. This is consistent with `FEATURE_INTERACTION_ANALYSIS.json`
having tested a small, manually-selected candidate list rather than an exhaustive one -- most of
what actually matters here was never in that list to begin with.

Charts: `outputs/models/classification/charts/v1c_systematic_interactions/` (calibration curve, PR
curve, ROC curve, prediction distribution).

### 3.6 Bottom line

**Unlike rung 1, this rung's gain survives held-out test.** v1c beats v1 on PR-AUC in CV (+0.0109),
on held-out test (+0.0022, same direction), in 5/5 paired CV folds against both v1 and
v1b_quadratic (paired t p=0.0079 and p=0.0199), and in the player-disjoint recheck -- with
calibration as good as or better than v1's throughout. This rung's own 3-gate ladder check passed.

**Update (Prompt 43): promotion confirmed.** The larger question this section originally left
open -- not just "did this rung's isolated change work" but "should it replace the standing
baseline everyone downstream builds on" -- was run through the same depth of scrutiny
`v1_unweighted` itself went through (tournament-stratified check, error-slice comparison against
v1's known weak spots, an interaction sanity check on the top surviving terms). All 4 promotion
criteria held. **`v1c_systematic_interactions` is now the standing reference model** for the
active-binary leg, replacing `v1_unweighted`. Full promotion evidence and verdict:
`reports/eda/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` section 7.

What this does *not* settle: whether hand-tuned interaction search is now "done" for this target.
116 surviving terms at C=0.1 is not a sparse, interpretable model -- it is evidence that
*something* in the interaction space matters (confirmed by the held-out gain), more than it is a
clean list of "the 5 interactions that matter." A tighter, more interpretable follow-up (e.g. a
stronger C, or L1 with stability selection across resamples) is a reasonable future step, but is
out of scope for this prompt, which only had to answer whether systematic interactions beat
isolated feature curvature (rung 1) -- they did.

## 4. Rung 3 -- Random Forest diagnostic (`v1d_random_forest`)

Rung 2's gain came from *explicitly* engineering 153 interaction/squared terms and letting L1
prune them. This rung asks the question that approach can't answer on its own: does a model that
discovers interactions and non-linearity automatically, straight from the **raw** 32 locked
features (no engineered terms, no standardization -- trees don't need it and it would only
obscure importances), find meaningfully more signal than the hand-built approach did? Explicitly
a **diagnostic rung, not a promotion candidate in its own right** -- the result decides how big an
investment Rung 4 (gradient boosting) should be, not whether to adopt Random Forest itself.

`RandomForestClassifier`, same 32 features, same categorical-one-hot/boolean-passthrough blocks
as v1, `random_state=42`, no `class_weight` as the primary variant (mirrors the
unweighted-wins-on-PR-AUC finding from v1 vs v2/v3 -- checked explicitly for trees too, see 4.2).
`n_estimators`/`max_depth`/`min_samples_leaf` grid-searched on the 5 canonical CV folds
(train+val only); held-out test read exactly once, after the grid was fixed.

### 4.1 Tuning (CV only, held-out test never touched)

| n_estimators | max_depth | min_samples_leaf | OOF PR-AUC |
|---|---|---|---|
| 300 | 8 | 5 | 0.3650 |
| 300 | 8 | 20 | 0.3624 |
| 300 | 14 | 5 | 0.3980 |
| 300 | 14 | 20 | 0.3895 |
| 300 | None | 5 | 0.4007 |
| 300 | None | 20 | 0.3907 |
| 600 | 8 | 5 | 0.3665 |
| 600 | 8 | 20 | 0.3634 |
| 600 | 14 | 5 | 0.3988 |
| 600 | 14 | 20 | 0.3907 |
| **600** | **None** | **5** | **0.4017 (chosen)** |
| 600 | None | 20 | 0.3914 |

Clear pattern: `max_depth` dominates (unconstrained depth beats 14 beats 8 in every
`n_estimators`/`min_samples_leaf` pairing), `min_samples_leaf=5` beats 20 throughout, and
`n_estimators=600` beats 300 only marginally (+0.001 to +0.002) at matched depth/leaf settings --
diminishing returns from more trees, most of the signal comes from letting individual trees grow
deep. Not a boundary-starved grid: 8&rarr;14&rarr;None shows a decelerating but still-positive
trend, consistent with an actual interior-ish optimum rather than "wants to go even deeper."

### 4.2 CV and held-out test results

| | PR-AUC | ROC-AUC | Log loss | Brier | Calib. slope | Calib. intercept | ECE |
|---|---|---|---|---|---|---|---|
| v1c (CV, OOF) | 0.3630 | 0.8288 | 0.2172 | 0.0608 | 1.004 | 0.001 | 0.0026 |
| v1d (CV, OOF) | **0.4017** | 0.8181 | 0.2184 | 0.0598 | 1.180 | 0.314 | 0.0123 |
| v1c (held-out test) | 0.3747 | 0.8321 | 0.2011 | 0.0551 | 1.027 | -0.081 | 0.0103 |
| v1d (held-out test) | **0.4085** | 0.8269 | 0.2017 | 0.0542 | 1.194 | 0.201 | 0.0183 |

**Two things are both true and neither cancels the other out.** PR-AUC: v1d beats v1c by a wide
margin, +0.0387 CV and +0.0338 held-out test (both far larger than rung 2's own margin over v1,
+0.0109/+0.0022) -- this is a real, substantial ranking improvement from letting a tree ensemble
find its own interactions. Calibration: v1d is **markedly worse** than v1c, exactly as the brief
anticipated for raw RF probabilities -- ECE roughly **4.7x worse on CV** (0.0123 vs 0.0026) and
**1.8x worse on held-out test** (0.0183 vs 0.0103), with a calibration slope pulled well above 1.0
(1.18-1.19, vs v1c's ~1.0-1.03) and a positive intercept (0.20-0.31, vs v1c's near-zero) --
the textbook signature of tree-ensemble overconfidence at the probability extremes. This is
reported as a genuine trade-off, not averaged into a vague "better overall": ranking and
calibration are answering different questions, and this project has treated both as first-class
since prompt 36 (recall v2/v3's `class_weight="balanced"` cost 27x on ECE for a *worse* PR-AUC --
different shape of trade-off, same principle of not letting one metric hide the other). That
said, an ECE of 0.012-0.018 is still small in absolute terms and nowhere near v2/v3's catastrophic
0.29 -- "markedly worse than v1c" is not the same claim as "badly broken." Post-hoc calibration
(Platt scaling or isotonic regression) is a standard, well-understood fix for exactly this
symptom, noted here as a live option but deliberately not implemented in this prompt (a rung 4
concern, per the brief).

### 4.3 Secondary comparison: does `class_weight` still lose for trees?

Checked explicitly rather than assumed: at the same tuned hyperparameters,
`class_weight="balanced_subsample"` gives a negligible PR-AUC change (OOF 0.4033 vs unweighted's
0.4017, +0.0016) but an **8x worse** ECE (0.0986 vs 0.0123). The v1-vs-v2/v3 finding holds for
trees too -- balancing the loss buys essentially nothing on ranking here and is meaningfully worse
for calibration, so `v1d_random_forest` (unweighted) is correctly the primary variant. Full
numbers: `outputs/models/validation/v1d_class_weight_secondary_comparison.json`.

### 4.4 Is the gain real? (`significance_v1d_vs_v1c.json`)

Paired over the same 5 canonical CV folds, v1d beats v1c in **5 of 5 folds**: mean PR-AUC
difference **+0.0386** (std 0.0111), paired t-test t=7.77, **p=0.0015**, Wilcoxon at the same n=5
exact-test floor (p=0.0625) seen in every prior rung comparison. This is the largest, most
unambiguous gap seen anywhere in this ladder -- roughly 3.6x rung 2's own margin over v1 in the
same paired-CV framework (+0.0386 vs +0.0109).

### 4.5 Player-disjoint re-check (`player_disjoint_v1d.json`)

A bagged tree ensemble is at least as capable of overfitting to player identity as the
interaction model was -- checked from scratch, not skipped. 5-fold `GroupKFold` on `player_id`,
train+val rows only, held-out test set untouched. `player_overlap_train_test == 0` confirmed and
asserted for every fold. Player-disjoint PR-AUC (0.4172) is actually *higher* than the
match-grouped CV OOF number (0.4017) -- no leakage signal; if anything this is reassuring, not a
concern.

### 4.6 Feature importance: does RF rediscover what v1c's interactions were built from?

**Top 15 by Gini importance** (in-sample, biased toward high-cardinality/continuous features --
reported alongside permutation importance for exactly that reason):

`nearest_attacker_distance`, `defender_spread`, `attacking_goal_centrality`,
`distance_to_attacking_box`, `angle_to_attacking_goal`, `attacker_spread`,
`attacker_defender_ratio`, `defender_attacker_gap_x`, `match_time_seconds`,
`defender_attacker_gap_y`, `possession_elapsed_seconds`,
`defenders_between_ball_and_attacking_goal`, `event_type_Pressure`, `visible_defender_count`,
`defenders_within_10m`.

**Top 15 by permutation importance**, averaged across each fold's held-out portion (never the
final test set -- fit on fold-train, permuted and scored on that fold's held-out rows, 5 repeats
per fold):

`event_type_Pressure`, `attacking_goal_centrality`, `event_type_Ball Recovery`,
`nearest_attacker_distance`, `attacker_defender_ratio`, `distance_to_attacking_box`,
`action_was_under_opponent_possession`, `action_retained_defensive_team_control`,
`angle_to_attacking_goal`, `defender_spread`, `event_type_Clearance`, `event_type_Block`,
`attacker_spread`, `is_in_defending_box`, `defender_attacker_gap_x`.

**Cross-validation between the two rungs' independent signals** (not just their PR-AUC numbers):
strong agreement. Every one of the numeric features that anchored v1c's top-10 surviving
interaction terms (`distance_to_attacking_box`, `defender_attacker_gap_x`,
`nearest_attacker_distance`, `attacking_goal_centrality`, `angle_to_attacking_goal`,
`attacker_spread`, `attacker_defender_ratio`, `defenders_between_ball_and_attacking_goal`) also
appears in RF's Gini top-15, and most of them in the permutation top-15 too -- two structurally
unrelated model families (an L1-pruned expanded-linear model and a tree ensemble), fit
independently, converged on the same handful of features as the most important drivers of shot
risk. That is a meaningfully stronger form of validation than either model's PR-AUC alone.

The Gini/permutation divergence is itself informative and matches the known bias: `event_type`
categories (`Pressure`, `Ball Recovery`) rank low-ish in Gini (`event_type_Pressure` at #13,
`event_type_Ball Recovery` outside the Gini top-15) but **first and third** by permutation
importance -- consistent with Gini's documented bias against low-cardinality one-hot columns in
favour of continuous features with many possible split points. The permutation ranking is the
more trustworthy read for `event_type`'s actual importance; both rankings agree on the
continuous/geometric features.

Charts: `outputs/models/classification/charts/v1d_random_forest/` (calibration curve, PR curve,
ROC curve, prediction distribution).

### 4.7 Bottom line

This is the first of the ladder's three anticipated honest outcomes, not the second or third:
**RF clearly beats v1c on held-out PR-AUC** (+0.0338, confirmed significant, confirmed
player-disjoint-stable), **and calibration, while markedly worse, is workable rather than broken**
(ECE 0.018 on held-out test -- worse than v1c's 0.010, nowhere near v2/v3's 0.29, and fixable
in principle via standard post-hoc calibration, not attempted here). Per the framing this rung was
built to test: **automatic interaction discovery finds more than the hand-built approach did**,
and there is real headroom in this feature set beyond what linear-plus-engineered-interactions
captured. **Rung 4 (gradient boosting) is a well-justified next step with likely real additional
upside**, not merely a smaller confirmatory check -- consistent with RF's own result here
suggesting the ceiling for this feature set is meaningfully above v1c's PR-AUC.

Per the brief, this diagnostic result does **not** trigger a Prompt-43-style promotion audit for
`v1d_random_forest` itself in this prompt -- it reports the ladder-gate result and the resulting
scope for rung 4. Whether Random Forest, gradient boosting, or a calibrated version of either
eventually becomes the standing reference model is a separate, later decision, made only once
each candidate has cleared its own dedicated validation pass, per this project's standing
practice throughout prompts 36-43.

## 5. Rung 4 -- gradient boosting (`v1e_gradient_boosting`)

Rung 3 found that automatic interaction discovery on the raw 32 features (Random Forest) beats
`v1c_systematic_interactions`'s hand-built-plus-L1 approach by a wide margin, at the cost of
noticeably worse calibration. Its own verdict: this justifies a real gradient-boosting build,
since boosting (bias-reduction-focused) has a real chance of finding still more signal than
bagging did, or of matching RF's ranking with better-behaved probabilities. This rung tests both
halves of that claim.

**Library: LightGBM.** Neither LightGBM nor XGBoost was already a project dependency.
`pip install lightgbm` pulled a pure `win_amd64` wheel with no build step and no compiler --
trivially installable -- so LightGBM was used rather than XGBoost for that reason, and added to
`pyproject.toml`. Same raw design matrix as `v1d_random_forest` (categorical one-hot, boolean
passthrough, numeric median-imputed but not standardized, no engineered interaction columns --
reuses `RFDesignMatrixBuilder` directly, not duplicated): the point, same as rung 3, is what the
model finds on its own. No `class_weight` as the primary variant (checked explicitly against a
secondary `class_weight="balanced"` comparison, section 5.2). `n_estimators` was never
grid-searched directly -- chosen per fit via early stopping (cap 2000 rounds,
`early_stopping_rounds=50`, monitored on `average_precision`) on a match-grouped validation
carve-out of that fit's own training rows. `learning_rate` / `num_leaves` / `min_child_samples`
were grid-searched on the 5 canonical CV folds (train+val only, held-out test never touched
during tuning).

Two reported variants: **`v1e_gradient_boosting`** (raw, tuned, uncalibrated) and
**`v1e_gradient_boosting_calibrated`** (the same tuned model, fixed `n_estimators` from early
stopping, wrapped in `CalibratedClassifierCV` fit with a match-grouped internal CV on train+val
only -- both sigmoid/Platt and isotonic tried, better one kept, see 5.3).

### 5.1 Tuning (27 combinations x 5 folds = 135 fits, CV only)

Full grid (`learning_rate` &times; `num_leaves` &times; `min_child_samples`) reported in
`significance_v1e_vs_v1d_v1c.json`'s `hyperparameter_grid_search`. Summary: the winner was
`learning_rate=0.01, num_leaves=63, min_child_samples=30` (OOF PR-AUC **0.4195**, mean early-stop
iteration 194). Two real patterns in the grid, checked against the actual per-setting averages
rather than eyeballed: the highest learning rate tested (0.1, mean OOF PR-AUC 0.4065 across its 9
combinations) is clearly worse and **never** appears in the top 9 combinations by OOF PR-AUC --
0.01 and 0.05 are close on average (0.4125 vs 0.4123) and split the top 9 roughly evenly (4 vs 5),
so "slower learning helps" is real but "0.01 specifically beats 0.05" is closer to a toss-up than
the single winning combination alone suggests. On `min_child_samples`, the most permissive
setting (10, mean 0.4073) is the clear loser; 30 and 100 are close on average (0.4117 vs 0.4122,
with 100 marginally *ahead* on average even though the single winning combination used 30) -- a
mild preference against very small leaves, not a sharp interior optimum. Held-out test was not
referenced anywhere in tuning.

### 5.2 CV and held-out test results

| | PR-AUC | ROC-AUC | Log loss | Brier | Calib. slope | Calib. intercept | ECE |
|---|---|---|---|---|---|---|---|
| v1c (CV, OOF) | 0.3630 | 0.8288 | 0.2172 | 0.0608 | 1.004 | 0.001 | 0.0026 |
| v1d (CV, OOF) | 0.4017 | 0.8181 | 0.2184 | 0.0598 | 1.180 | 0.314 | 0.0123 |
| v1e raw (CV, OOF) | **0.4195** | 0.8242 | 0.2154 | 0.0586 | 1.257 | 0.541 | 0.0123 |
| v1e calibrated (CV, OOF) | **0.4252** | 0.8321 | 0.2104 | 0.0576 | 1.036 | 0.071 | 0.0027 |
| v1c (held-out test) | 0.3747 | 0.8321 | 0.2011 | 0.0551 | 1.027 | -0.081 | 0.0103 |
| v1d (held-out test) | 0.4085 | 0.8269 | 0.2017 | 0.0542 | 1.194 | 0.201 | 0.0183 |
| v1e raw (held-out test) | **0.4305** | 0.8358 | 0.1944 | 0.0523 | 1.027 | -0.037 | 0.0068 |
| v1e calibrated (held-out test) | **0.4282** | 0.8364 | 0.1943 | 0.0522 | 1.030 | -0.080 | 0.0096 |

**Ranking outcome, stated separately from calibration:** gradient boosting beats Random Forest's
held-out PR-AUC (0.4305 vs 0.4085, +0.0220) and beats v1c by an even wider margin (+0.0558). This
is not "roughly level" -- it is the largest gap between adjacent rungs seen anywhere in this
ladder, and it holds in both CV and held-out test, in the same direction. Boosting found real
additional signal beyond what bagging (rung 3) found.

**Calibration outcome, stated separately from ranking, and a genuine surprise worth flagging
plainly:** raw `v1e`'s CV calibration is markedly worse than v1c's (ECE 0.0123 vs 0.0026, slope
1.257 vs 1.004 -- the same rough magnitude of miscalibration RF showed), **but its held-out-test
calibration is unexpectedly good** (ECE 0.0068, actually *better* than v1c's 0.0103, slope 1.027
essentially ideal). This CV-vs-test discrepancy is reported honestly rather than reconciled away:
it most plausibly reflects the extra variance introduced by early stopping's own internal
train/validation carve-out inside each of the 5 (smaller) CV folds versus the one large
train+val-vs-test split, and/or genuine single-readout noise on 23 held-out matches -- not
evidence that the CV number is wrong. The calibrated variant (isotonic won over sigmoid, see 5.3)
brings CV calibration in line with v1c's (ECE 0.0027 vs 0.0026 -- matched almost exactly) while
held-out calibration stays close to v1c's (0.0096 vs 0.0103). Calibration was **not** assumed to
leave PR-AUC untouched -- measured directly: the calibrated variant's held-out PR-AUC (0.4282) is
very slightly below the raw variant's (0.4305, -0.0023), a small real cost from the isotonic
binning step, not zero but not meaningful either.

**The practically important combination the brief flagged in advance actually happened:**
`v1e_gradient_boosting_calibrated` ranks almost as well as the raw model (which beats both v1c
and v1d substantially) *and* calibrates about as well as v1c. This is the strongest candidate
combination of ranking and calibration seen anywhere in the ladder so far -- not a promotion
decision (none is made in this prompt), but the clearest case yet for one being worth running.

### 5.3 Calibration method: sigmoid vs isotonic

Both tried via the same 5-fold canonical CV (match-grouped internal 3-fold CV for the
calibration wrapper itself, train+val only):

| Method | OOF PR-AUC | OOF ECE | OOF calib. slope | OOF calib. intercept |
|---|---|---|---|---|
| Sigmoid (Platt) | 0.4261 | 0.0032 | 1.059 | 0.115 |
| **Isotonic** | 0.4252 | **0.0027** | **1.036** | **0.071** |

Isotonic won narrowly on every calibration metric (lower ECE, slope and intercept both closer to
ideal) for a negligible PR-AUC cost relative to sigmoid (-0.0009) -- selected as
`v1e_gradient_boosting_calibrated`. Both methods comfortably beat the raw model's CV ECE (0.0123),
confirming post-hoc calibration is doing real work here, not just adding noise. Full comparison:
`outputs/models/validation/v1e_calibration_method_comparison.json`.

### 5.4 Secondary comparison: does `class_weight` still lose?

At the same tuned hyperparameters, `class_weight="balanced"` drops OOF PR-AUC from 0.4195 to
0.4049 (worse, not better) and inflates ECE from 0.0123 to 0.2213 -- an **18x** calibration cost
for a *worse* ranking. The unweighted-wins finding holds for gradient boosting too, the third
model family in a row (linear, RF, GBM) to confirm it. Full numbers:
`outputs/models/validation/v1e_class_weight_secondary_comparison.json`.

### 5.5 Is the gain real? (`significance_v1e_vs_v1d_v1c.json`)

Paired over the same 5 canonical CV folds, raw `v1e` beats both prior rungs in **5 of 5 folds**:

| Comparison | Mean &Delta; PR-AUC | Std &Delta; | Paired t (p) | Wilcoxon (p) |
|---|---|---|---|---|
| v1e vs v1c | +0.0560 | 0.0156 | t=8.01, **p=0.0013** | W=0, p=0.0625 |
| v1e vs v1d | +0.0175 | 0.0103 | t=3.81, **p=0.0189** | W=0, p=0.0625 |

Both significant at the conventional 0.05 threshold; Wilcoxon at the same n=5 exact-test floor
seen throughout this ladder. The v1e-vs-v1c gap (+0.0560) is the largest paired-CV margin
recorded anywhere in the ladder so far.

### 5.6 Player-disjoint re-check (`player_disjoint_v1e.json`)

5-fold `GroupKFold` on `player_id`, train+val rows only, held-out test set untouched,
`player_overlap_train_test == 0` confirmed and asserted for every fold. Player-disjoint PR-AUC
(0.4362 &plusmn; 0.0252) is higher than the match-grouped CV OOF number (0.4195) -- consistent
with the pattern seen at every prior rung, no leakage signal.

### 5.7 Feature importance: a three-way cross-check

**Top 15 by gain importance:** `match_time_seconds`, `distance_to_attacking_box`,
`attacking_goal_centrality`, `nearest_attacker_distance`, `defender_spread`, `attacker_spread`,
`possession_elapsed_seconds`, `attacker_defender_ratio`, `defender_attacker_gap_x`,
`defender_attacker_gap_y`, `angle_to_attacking_goal`, `visible_attacker_count`,
`event_type_Pressure`, `action_retained_defensive_team_control`,
`action_was_under_opponent_possession`.

**Top 15 by permutation importance** (fold-held-out, never the final test set):
`attacking_goal_centrality`, `attacker_defender_ratio`, `event_type_Pressure`,
`action_was_under_opponent_possession`, `event_type_Ball Recovery`, `nearest_attacker_distance`,
`distance_to_attacking_box`, `defender_spread`, `action_retained_defensive_team_control`,
`phase_label_settled_mid_block_proxy`, `phase_changed_since_prev_event`, `event_type_Clearance`,
`event_type_Foul Committed`, `phase_label_prev_event_None`, `attacker_spread`.

**Three structurally different model families, fit independently, converge on the same
features** -- the strongest validation signal available in this project so far. Every numeric
feature anchoring v1c's top surviving interaction terms and v1d's top importances
(`distance_to_attacking_box`, `attacking_goal_centrality`, `nearest_attacker_distance`,
`defender_spread`, `attacker_spread`, `attacker_defender_ratio`, `defender_attacker_gap_x`,
`angle_to_attacking_goal`) also appears in v1e's gain top-15, and most of them in its permutation
top-15 too. The same gain/permutation divergence seen in rung 3 repeats here: `event_type`
categories (`Pressure`, `Ball Recovery`) and possession-context booleans
(`action_was_under_opponent_possession`, `action_retained_defensive_team_control`) rank
mid-to-low in gain importance but consistently in the permutation top 5 -- the same documented
bias (gain/Gini favouring continuous features with many split points) showing up identically in
two unrelated tree ensembles, which is itself a form of cross-validation of the bias explanation,
not just the features.

Charts: `outputs/models/classification/charts/v1e_gradient_boosting/` and
`.../v1e_gradient_boosting_calibrated/` (calibration curve, PR curve, ROC curve, prediction
distribution, both variants).

### 5.8 Bottom line

Two separate verdicts, not one, per the discipline this rung was asked to follow:

**Ranking: gradient boosting wins clearly**, beating Random Forest's held-out PR-AUC by +0.0220
and v1c's by +0.0558 -- both confirmed significant (p=0.0189, p=0.0013) and player-disjoint-stable.
Boosting found real additional signal that bagging (rung 3) had not.

**Calibration: the post-hoc-calibrated variant recovers v1c-level quality** (isotonic CV ECE
0.0027 vs v1c's 0.0026 -- matched almost exactly; held-out ECE 0.0096 vs v1c's 0.0103 -- close)
**while keeping nearly all of the PR-AUC gain** (held-out PR-AUC 0.4282, a -0.0023 cost from
calibration, not the near-total erosion that would make the trade-off not worth it). This is the
practically important combination flagged in advance: a model that ranks like the best tree
ensemble tried so far and calibrates like the best linear model tried so far.

Per the brief, **no promotion audit is run in this prompt** for either `v1e_gradient_boosting` or
`v1e_gradient_boosting_calibrated` -- this section reports the ladder-gate result only. Four
candidates now exist across the ladder (v1c, v1d, v1e raw, v1e calibrated); which one, if any, is
put through a Prompt-43-style promotion audit is a separate decision for Varun and a later prompt,
not decided here.

**Update (Prompt 46): promotion confirmed, calibrated variant, not raw.** The full promotion
audit found that raw `v1e`'s good aggregate held-out ECE did not generalise to the slice level --
its CV-OOF calibration gap on `phase_label` slices was 6-19x larger than v1c's on 6 of 7 slices,
a problem the aggregate number hid. The calibrated variant fixes this (bringing slice-level
calibration to within v1c's range or better on most slices, and actually fixing the
`Goalkeeper`/`Right Attacking Midfield` calibration caveat noted when v1c was promoted) while
keeping the ranking gain on the large majority of slices, with one honestly-flagged exception
(`wide_defending_proxy` regresses, judged non-disqualifying). **`v1e_gradient_boosting_calibrated`
is now the standing reference model** for the active-binary leg, superseding
`v1c_systematic_interactions`. Full promotion evidence and verdict:
`reports/eda/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` section 8.
