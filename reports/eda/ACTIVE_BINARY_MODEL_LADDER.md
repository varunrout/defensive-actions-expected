# Active-Binary Model Ladder

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

## 3. Rung 2 -- systematic interactions (L1)

*Not yet run.*
