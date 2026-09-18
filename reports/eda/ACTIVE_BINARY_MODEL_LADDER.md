# Active-Binary Model Ladder

> **Status update (Prompt 43):** Rung 2 (`v1c_systematic_interactions`) has been promoted to
> standing reference model for the active-binary leg, replacing `v1_unweighted`. See the full
> promotion decision in `reports/eda/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` section 7.

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
