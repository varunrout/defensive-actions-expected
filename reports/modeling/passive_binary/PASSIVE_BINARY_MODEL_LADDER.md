# Passive-Binary Model Ladder

*Split out of `PASSIVE_BINARY_BASELINE_SUMMARY.md` into its own document, mirroring the active
leg's Rung-0-doc / ladder-doc split (Prompt 42). Rung 0 -- the 4 locked variants (`p0_dummy`,
`p1_unweighted`, `p2_weighted`, `p3_weighted_interactions`), their CV/held-out results, and the
post-lock validation -- stays in
[`PASSIVE_BINARY_BASELINE_SUMMARY.md`](PASSIVE_BINARY_BASELINE_SUMMARY.md). This document is
everything built on top of that locked baseline, starting from Rung 1. No promotion audit is run
in this document, whatever the results -- that is a separate follow-up once the full ladder is in
and reviewed.*

## 1. What a rung is

A **rung** is one controlled change to the locked Rung 0 baseline (`p1_unweighted`): a single
variable changed -- feature shape, an interaction term, a model family -- with everything else
held identical (same 38 locked features unless the rung specifically changes that, same canonical
`StratifiedGroupKFold` split shared with the active leg, same target). Every rung is tested
against 3 fixed gates:

1. **Paired significance test vs the current best** -- refit both variants on the same 5 canonical
   CV folds, paired t-test + Wilcoxon signed-rank test on the per-fold PR-AUC difference.
2. **Held-out-test confirmation** -- a single, one-time readout on the frozen 23-match test set.
3. **Cluster-by-`event_id` bootstrap** -- this leg's replacement for the active leg's
   player-disjoint recheck, since `passive_defense.parquet` has no `player_id` column. Naive
   row-level and cluster-by-`event_id` bootstrap CIs (1,000 iterations each, method established in
   Prompt 50) are computed side by side for each rung's advantage over `p1_unweighted` on the
   held-out test set, so the row-correlation caveat is checked at every rung, not just once.

This ladder builds on `PASSIVE_BINARY_BASELINE_SUMMARY.md`'s `p1_unweighted` pick (held-out PR-AUC
0.1736, ROC-AUC 0.7498, ECE 0.0051) as the baseline every rung is compared against.

**A structural note carried over from Rung 0 and worth repeating here**: every metric below still
treats each row as an independent trial, even though rows sharing an `event_id` are not
independent (Prompt 50's cluster bootstrap quantified this at roughly a 2-3x understatement of
true uncertainty for Rung 0's own p1-vs-p2 comparison). The same caveat applies to every rung
below -- the cluster-bootstrap gate at each rung is exactly the mechanism for checking whether that
caveat changes the rung's headline conclusion, and it is reported plainly whenever it does or
doesn't.

**Computational-scale note, stated once here rather than repeated at every rung**: this leg's
train+val set (1,277,374 rows) is roughly 28x the active leg's (45,408 rows). Every rung below
trims its hyperparameter/C grid relative to the active leg's equivalent rung, and states the exact
trim and the reasoning (a timed probe fit, where one was run, or the plain arithmetic of grid size
x row count) rather than silently shrinking it. This is a real, reported constraint of running this
ladder at this leg's actual data scale, not a shortcut taken without acknowledgement.

## 2. Rung 1 -- quadratic terms for U-shaped features (`p1b_quadratic`)

**Which features, and why -- verified against this leg's own EDA, not the active leg's.** The
active leg's Rung 1 used 4 features (`defender_spread`, `distance_to_attacking_box`,
`visible_defender_count`, `defenders_between_ball_and_attacking_goal`) that do not exist on the
passive leg's feature list at all. This leg's own
`reports/analysis/shot_target/passive_numerical_target_atlas.json` was read directly: of the 38
locked features, 6 are classified strictly `"U-shaped"` (not the weaker `"inverse-U"` bucket) --
`top_option_1_threat_score`, `top_option_2_threat_score`, `top_option_3_threat_score` (this leg's 3
strongest-correlated features overall, per `FEATURE_LOCK_CONFIRMATION.json`'s own
`numerical_vs_target_rankings.passive_top`), `ball_x`, `defender_x`, and `angle_to_attacking_goal`.
6 features, not forced to match the active leg's count of 4.

`p1b_quadratic` adds one `quad__<feature>` column per feature above -- the square of its
already-standardized value -- to the 38 base features. No `class_weight` (unweighted, like p1;
this rung isolates feature shape, not reweighting).

### 2.1 Results

| | CV PR-AUC (OOF) | Held-out PR-AUC | CV ROC-AUC | Held-out ROC-AUC | CV ECE | Held-out ECE |
|---|---|---|---|---|---|---|
| `p1_unweighted` (Rung 0) | 0.1928 | 0.1736 | 0.7487 | 0.7498 | 0.0024 | 0.0051 |
| `p1b_quadratic` | **0.1983** | **0.1805** | 0.7621 | 0.7658 | 0.0018 | 0.0045 |

**Gate 1 -- significance** (`significance_p1_vs_p1b_quadratic.json`): mean PR-AUC diff +0.0056,
paired t-test t=4.56, **p=0.0104** (significant at 0.05), Wilcoxon p=0.0625 (the same 5-fold floor
seen throughout this project). p1b beats p1 in 5/5 folds.

**Gate 2 -- held-out confirmation**: held-out PR-AUC improves from 0.1736 to 0.1805, the same
direction as CV. **This is the headline divergence from the active leg worth stating plainly**: the
active leg's Rung 1 (`v1b_quadratic`) did *not* survive held-out test (its CV gain evaporated) --
this leg's Rung 1 *does* survive, with a real, consistent gain in both CV and held-out. The two
legs' own EDA-driven feature choices led to genuinely different outcomes; this was checked, not
assumed to repeat the active leg's result in either direction.

**Gate 3 -- cluster-by-`event_id` bootstrap** (`cluster_bootstrap_p1_vs_p1b.json`, held-out test,
1,000 iterations): diff (p1b - p1) naive CI [0.0051, 0.0085] (width 0.0034), cluster CI [0.0044,
0.0097] (width 0.0053), **width ratio 1.57x**. Both CIs are entirely positive -- the gain survives
the more conservative, correlation-aware interval, with less inflation than Rung 0's own p1-vs-p2
comparison (2.15x) showed.

**Verdict: p1b_quadratic passes all 3 gates.** Unlike the active leg's Rung 1, this is a real,
adoptable improvement over Rung 0 on this leg's own evidence.

## 3. Rung 2 -- systematic interactions (L1) (`p1c_systematic_interactions`)

Every pairwise product and squared term (`PolynomialFeatures(degree=2, interaction_only=False,
include_bias=False)`) generated from this leg's own 28-feature numeric block (26 continuous + 2
discrete -- confirmed directly from `train_passive_binary_baseline.py`'s `NUMERIC_COLS`, **not**
the active leg's 17), combined unchanged with the categorical one-hot / boolean-passthrough blocks:
C(28,2)=378 pairwise + 28 squared = 406 new columns, 462 total design-matrix columns.
L1-penalized `LogisticRegression(penalty="l1", solver="liblinear")`, `C` grid-searched on the 5
canonical CV folds only.

**Grid deviation, stated up front**: `C` &isin; {0.01, 1.0} (2 values, not the active leg's 5), and
liblinear's `tol` relaxed to 1e-2 (vs the active leg's 1e-3). A timed probe fit on a ~200K-row
subsample (roughly 1/6 of a real CV fold) took ~118s at this tolerance; scaling to the real
~1M-row folds made a full 5-point grid impractical for this session, so the grid was trimmed to the
active leg's own range endpoints -- enough to see the direction of the regularization-strength
effect without the full sweep.

### 3.1 Results

| | C grid tested | Chosen C | CV PR-AUC | Held-out PR-AUC |
|---|---|---|---|---|
| `p1c_systematic_interactions` | 0.01, 1.0 | **0.01** | **0.2160** | **0.2013** |

C=0.01 (strong regularization) won by a small margin over C=1.0 (OOF PR-AUC 0.2160 vs 0.2153).

**Gate 1 -- significance** (`significance_p1c_vs_p1_p1b.json`): vs p1, mean diff +0.0233, t=10.89,
**p=0.0004**; vs p1b, mean diff +0.0177, t=18.25, **p=0.00005**. Both highly significant, well past
the p1b-vs-p1 comparison's own significance in Rung 1.

**Gate 2 -- held-out confirmation**: held-out PR-AUC 0.2013, clearly ahead of both p1 (0.1736) and
p1b (0.1805), same direction as CV.

**Gate 3 -- cluster bootstrap** (`cluster_bootstrap_p1_vs_p1c.json`): diff (p1c - p1) naive CI
[0.0249, 0.0306] (width 0.0057), cluster CI [0.0226, 0.0335] (width 0.0110), **width ratio 1.92x**.
Both entirely positive.

**Verdict: p1c_systematic_interactions passes all 3 gates**, and by a clearly larger margin than
Rung 1.

### 3.2 What survived L1, and does it match what was already flagged?

At the chosen C=0.01, **286 of 406 poly/interaction terms survived with nonzero coefficient** --
L1 pruned less than a third; this design matrix's terms are, for the most part, not redundant with
each other even under fairly strong regularization. The top 15 surviving terms by `|coef|`:

| Rank | Term | Coefficient |
|---|---|---|
| 1 | `poly__defender_y__x__ball_y` | -0.4326 |
| 2 | `poly__defender_x__sq` | +0.3576 |
| 3 | `poly__defender_x__x__ball_x` | +0.3287 |
| 4 | `poly__ball_y__sq` | -0.2602 |
| 5 | `poly__ball_y__x__top_option_3_dy` | -0.2364 |
| 6 | `poly__ball_x__x__top_option_1_distance_from_ball` | +0.1925 |
| 7 | `poly__top_option_1_dx__x__top_option_3_dx` | +0.1852 |
| 8 | `poly__overload_score__x__defender_slot_index` | +0.1799 |
| 9 | `poly__ball_x__sq` | -0.1616 |
| 10 | `poly__ball_x__x__top_option_1_threat_score` | -0.1518 |
| 11 | `poly__defender_x__x__top_option_1_distance_from_ball` | -0.1517 |
| 12 | `poly__top_option_3_threat_score__sq` | +0.1489 |
| 13 | `poly__ball_y__x__top_option_2_dy` | -0.1484 |
| 14 | `poly__ball_x__x__top_option_3_dx` | +0.1450 |
| 15 | `poly__defender_x__x__defender_slot_index` | -0.1440 |

**Cross-check against `FEATURE_LOCK_CONFIRMATION.json`'s `numeric_interaction` findings**: that
section tested 10 hand-picked pairs from the lane-screening/marking-tightness/overload-score
family (the same family Prompt 49's `p3_weighted_interactions` used). **None of those 10 pairs
appear in the top 15 survivors above.** The model *ignores* that curated family in favour of raw
geometric-coordinate interactions -- `defender_x`/`defender_y`/`ball_x`/`ball_y` and the
`top_option_N_*` distance/angle features dominate instead. This is reported plainly rather than
forced into agreement with the earlier hand-picked set: L1 found a genuinely different structure
than the football-domain-motivated family that Prompt 49 and `FEATURE_LOCK_CONFIRMATION.json`
flagged.

**Cross-check against Rung 1's chosen features**: 3 of Rung 1's 6 quadratic features reappear
directly as top-15 squared terms here -- `defender_x` (rank 2), `ball_y`/`ball_x` (ranks 4 and 9),
and `top_option_3_threat_score` (rank 12). This is a genuine independent confirmation: two
different methods (EDA correlation shape + L1 coefficient survival) agree that these features carry
real nonlinear signal.

## 4. Rung 3 -- Random Forest diagnostic (`p1d_random_forest`)

Diagnostic probe (not a promotion candidate in its own right, same framing as the active leg): raw
38 locked features, no manual feature engineering, median-imputed but not standardized. The
question: does automatic interaction discovery find meaningfully more than Rung 2's hand-engineered
L1 approach?

**Grid deviation, stated up front**: 3 hyperparameter combinations, not the active leg's 12
(`n_estimators` &isin; {150}, `max_depth` &isin; {10, 16}, `min_samples_leaf` &isin; {200, 500}),
and `min_samples_leaf` set far higher than the active leg's 5-20. At this leg's ~28x row count, the
active leg's grid (up to 600 trees, leaves as small as 5 rows) is not tractable in this session's
timeframe and would badly overfit individual leaves regardless of tractability -- a real finding
about how tree hyperparameters should scale with row count, not a silent shortcut.

### 4.1 Results

| Hyperparameters | CV PR-AUC |
|---|---|
| n_est=150, depth=10, leaf=200 | 0.2094 |
| **n_est=150, depth=16, leaf=200** | **0.2182** |
| n_est=150, depth=16, leaf=500 | 0.2138 |

| | CV PR-AUC | Held-out PR-AUC | CV calib. slope | CV calib. intercept | CV ECE | Held-out ECE |
|---|---|---|---|---|---|---|
| `p1d_random_forest` | **0.2182** | **0.2078** | 1.139 | +0.327 | 0.0049 | 0.0070 |

**Gate 1 -- significance vs p1c** (`significance_p1d_vs_p1c.json`): mean diff +0.0025, t=1.26,
**p=0.275 -- NOT significant**. This is the first rung-vs-rung comparison on this leg that fails
to clear conventional significance; p1d is numerically ahead of p1c on both CV (0.2182 vs 0.2160)
and held-out (0.2078 vs 0.2013), but the CV gap is small relative to fold-to-fold noise (mean diff
0.0025 vs std 0.0044) and the paired test does not distinguish it from zero. Reported plainly
rather than glossed over: **p1d does not clear the significance gate against p1c**, even though it
is directionally ahead on both CV and held-out.

**Gate 2 -- held-out confirmation**: held-out PR-AUC 0.2078, ahead of p1c's 0.2013 -- consistent
direction with CV, but see Gate 1's caveat about statistical distinguishability.

**Gate 3 -- cluster bootstrap** (`cluster_bootstrap_p1_vs_p1d.json`, vs `p1_unweighted`): diff
naive CI [0.0315, 0.0373] (width 0.0058), cluster CI [0.0270, 0.0422] (width 0.0152), **width ratio
2.62x**. Both entirely positive -- p1d's advantage over the Rung 0 baseline is real and survives
cluster resampling, even though its advantage over the *prior rung* (p1c) does not clear
significance.

**Calibration, reported separately from ranking (same two-part discipline as the active leg)**: RF
is noticeably worse-calibrated than the linear rungs -- calibration slope 1.139 (vs p1c's 0.988)
and intercept +0.327 (vs p1c's -0.030) in CV, though absolute ECE stays low (0.0049 CV, 0.0070
held-out). This is the same qualitative pattern the active leg found for its RF rung: a ranking
gain (or, here, a directional-but-not-significant one) that comes with a real calibration cost.

**Secondary: `class_weight="balanced_subsample"`** (`p1d_class_weight_secondary_comparison.json`):
OOF PR-AUC 0.2205 vs unweighted 0.2182 -- a small gain, unlike every prior rung's finding that
weighting never helps. But ECE explodes from 0.0049 to 0.2932, the same catastrophic
calibration-destruction mechanism seen at every other rung. The small PR-AUC uptick does not
change this leg's standing conclusion: weighting is not worth its calibration cost.

**Verdict: p1d_random_forest passes gates 2 and 3 (vs the Rung 0 baseline) but fails gate 1 (vs the
immediately-prior rung, p1c)** -- a real, reported outcome rather than a forced pass or fail.

## 5. Rung 4 -- gradient boosting (`p1e_gradient_boosting`, + calibrated)

LightGBM, already a project dependency (added for the active leg's own Rung 4, reused here rather
than adding a new one). Same design matrix as p1d (`RFDesignMatrixBuilder`, reused directly, not
duplicated).

**Grid deviation, stated up front**: 4 hyperparameter combinations (`learning_rate` &isin; {0.05,
0.1} x `num_leaves` &isin; {31, 63} x `min_child_samples`={500}), not the active leg's 27 (3x3x3),
and the early-stopping cap is 1000 rounds not 2000. In practice this leg's early stopping converges
fast regardless of the cap (mean `best_iteration_` 116 rounds at the chosen params) -- the round
cap was not the binding constraint the grid size was.

### 5.1 Raw variant results

| Hyperparameters | CV PR-AUC | Mean best iteration |
|---|---|---|
| lr=0.05, leaves=31, min_child=500 | **0.2239** | 116 |
| lr=0.05, leaves=63, min_child=500 | 0.2213 | -- |
| lr=0.1, leaves=31, min_child=500 | 0.2199 | -- |
| lr=0.1, leaves=63, min_child=500 | 0.2163 | -- |

| | CV PR-AUC | Held-out PR-AUC | CV ROC-AUC | Held-out ROC-AUC |
|---|---|---|---|---|
| `p1e_gradient_boosting` (raw) | **0.2239** | **0.2108** | 0.7829 | 0.7850 |

This is the best CV PR-AUC of any rung so far, and its held-out score (0.2108) edges out p1d's
(0.2078).

**Gate 1a -- significance vs p1d** (raw): mean diff +0.0061, t=1.65, **p=0.174 -- NOT
significant**. The two tree-based rungs are statistically indistinguishable from each other on CV,
consistent with the pattern that started at Rung 3 (statistically-clear jumps happen in the early
rungs; later rungs are directionally ahead but the gaps get harder to distinguish from fold noise).

**Gate 1b -- significance vs the best linear candidate** (`p1c_systematic_interactions`, the
highest-CV-PR-AUC linear variant): mean diff +0.0086, t=3.10, **p=0.036 -- significant** by the
paired t-test, though Wilcoxon lands at p=0.125 (n=5 floor, direction consistent but not
independently significant). Read together: p1e is distinguishable from the best *linear* candidate,
though not from the other *tree-based* candidate (p1d).

**Gate 3 -- cluster bootstrap** (vs `p1_unweighted`): diff naive CI [0.0340, 0.0403] (width 0.0063),
cluster CI [0.0297, 0.0457] (width 0.0161), **width ratio 2.56x**. Both entirely positive.

### 5.2 Calibrated variant

Both sigmoid and isotonic `CalibratedClassifierCV` were fit (match-grouped internal CV, train+val
only); isotonic chosen by lower OOF ECE.

| Method | CV PR-AUC | CV ECE |
|---|---|---|
| raw (uncalibrated) | 0.2239 | 0.0028 |
| sigmoid | 0.2287 | 0.0022 |
| **isotonic (chosen)** | **0.2270** | **0.0018** |

| | CV PR-AUC | Held-out PR-AUC | CV calib. slope | Held-out calib. intercept | CV ECE | Held-out ECE |
|---|---|---|---|---|---|---|
| `p1e_gradient_boosting_calibrated` | **0.2270** | **0.2162** | 1.049 | **+0.006** | 0.0018 | 0.0044 |

Two findings worth stating precisely rather than rounding to a single verdict:

- **Calibration improved ranking too, not just probabilities**: isotonic calibration's CV PR-AUC
  (0.2270) is *higher* than the raw model's (0.2239) -- unusual, since calibration wrappers
  typically trade a little ranking quality for better probabilities, not gain both. The held-out
  PR-AUC (0.2162) is the best of any rung on this leg.
- **Calibration intercept improves to essentially zero** on held-out (+0.006, vs raw's -0.040 and
  RF's +0.28), the tightest bias of any variant tried on this leg. But held-out ECE is *marginally
  higher* for the calibrated variant than the raw one (0.0044 vs 0.0039) -- both are small in
  absolute terms, and this is reported as-is rather than selectively quoting whichever number
  favours the calibrated variant.

### 5.3 Feature importance: four-way cross-method agreement

Gain importance (top 5): `defender_x`, `ball_x`, `top_option_3_threat_score`,
`phase_label_high_press_proxy`, `ball_y`. Permutation importance, fold-held-out (top 5):
`defender_x`, `ball_y`, `on_ball_event_type_Shot`, `ball_x`, `top_option_1_threat_score`.

This closes a four-way independent cross-check across every rung of this ladder: this leg's own
EDA correlation atlas (Rung 1's feature choice), L1 coefficient survival (Rung 2), Random Forest
Gini + permutation importance (Rung 3), and now LightGBM gain + permutation importance all agree
that `defender_x`, `ball_x`/`ball_y`, and the `top_option_N_threat_score` features are this leg's
dominant defensive signal. No single method drove this conclusion in isolation.

### 5.4 Secondary: `class_weight="balanced"`

`p1e_class_weight_secondary_comparison.json`: OOF PR-AUC 0.2158 vs unweighted 0.2239 -- this time
not even a small gain, a real loss -- for the same catastrophic calibration cost (ECE 0.0028 ->
0.3029) seen at every rung on this leg. The unweighted-wins finding holds without qualification
through the whole ladder.

**Verdict: `p1e_gradient_boosting_calibrated` passes gates 2 and 3 comfortably and is the best
held-out PR-AUC and best held-out calibration-intercept result of any rung tried on this leg**, but
gate 1 is mixed: not statistically distinguishable from p1d (the other tree-based rung), but
distinguishable from the best linear candidate. No promotion audit is run here, per this prompt's
explicit constraint -- see section 6 for the full ladder summary and what would need to happen next.

## 6. Ladder summary

| Variant | CV PR-AUC | Held-out PR-AUC | Held-out ECE | Significance vs prior rung | Passes cluster-bootstrap gate vs p1? |
|---|---|---|---|---|---|
| `p1_unweighted` (Rung 0) | 0.1928 | 0.1736 | 0.0051 | -- | -- |
| `p1b_quadratic` (Rung 1) | 0.1983 | 0.1805 | 0.0045 | p=0.0104 (sig.) | Yes (1.57x width ratio) |
| `p1c_systematic_interactions` (Rung 2) | 0.2160 | 0.2013 | 0.0043 | p=0.0004 vs p1b (sig.) | Yes (1.92x width ratio) |
| `p1d_random_forest` (Rung 3) | 0.2182 | 0.2078 | 0.0070 | p=0.275 vs p1c (NOT sig.) | Yes (2.62x width ratio) |
| `p1e_gradient_boosting` (Rung 4, raw) | 0.2239 | 0.2108 | 0.0039 | p=0.174 vs p1d (NOT sig.) | Yes (2.56x width ratio) |
| `p1e_gradient_boosting_calibrated` (Rung 4) | 0.2270 | **0.2162** | 0.0044 | (calibration of raw p1e) | Yes (inherits raw's gate) |

Every rung's held-out PR-AUC is monotonically ahead of the one before it, and every rung's cluster
bootstrap confirms a real advantage over the Rung 0 baseline that survives the row-correlation
caveat. But the *rung-to-rung* significance gate stops being clearable partway through the ladder:
Rungs 1 and 2 each cleared it decisively against their immediate predecessor; Rungs 3 and 4 (both
tree-based) are directionally ahead but not statistically distinguishable from the rung immediately
before them. This is the expected shape of a ladder converging on the same underlying signal from
increasingly flexible model families, not a contradiction -- and it is reported as exactly that
shape, not smoothed into a single "the ladder keeps winning" headline.

**No promotion decision is made in this document.** Per this prompt's explicit constraint, that is
a separate follow-up once the full ladder is in and reviewed against the active leg's own
promotion-audit precedent (Prompts 43/46).
