# Passive-Binary Baseline Summary (p0-p3)

> **Status update (Prompt 52):** `p1_unweighted` (this document's pick) has been superseded as
> the passive-binary leg's standing reference model by `p1e_gradient_boosting_calibrated` -- see
> the promotion decision in `reports/modeling/passive_binary/PASSIVE_BINARY_MODEL_LADDER.md`
> section 7. This document remains the accurate historical record of the Rung 0 comparison and is
> not rewritten.

Target: `target_future_shot_10s`, computed per-row from that row's own anchor timestamp
(never broadcast across a possession -- a leakage bug caught and fixed specifically on
this leg during design; nothing in this baseline re-derives the target). Dataset:
`data/features/passive_defense.parquet` (1,593,181 rows / 115 matches -- confirmed real
scale directly against the parquet, not assumed from prose).

*For the full math behind these models -- derived from first principles, with real worked
examples -- see [`MATH_BEHIND_IT.md`](../MATH_BEHIND_IT.md).*

**Source note:** the prompt for this baseline referenced `claude/passive-defense-build-plan.md`
as context. That file does not exist anywhere in this repository (checked by full-repo
search). Every claim below is instead verified directly against the real, current sources:
`src/eda/feature_config.py` (the locked feature lists), `reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json`
(the lock counts and interaction-pair evidence), and `data/features/passive_defense.parquet`
itself (row/match counts, base rate, column presence). This is a Rung 0 baseline only --
no model ladder in this document (that is a follow-up prompt), mirroring the discipline
already established for the active leg's baseline.

## 1. Feature set

All 38 locked `PASSIVE` features from `src/eda/feature_config.py` (4 categorical + 6
boolean + 26 continuous + 2 discrete = 38). Unlike the active leg (34 locked -> 32 used,
2 excluded for known data issues), no feature is excluded for this leg -- the passive
feature set is used as-is. Verified programmatically at training time (an assertion in
`scripts/models/train_passive_binary_baseline.py` fails the run if the count drifts from 38).

- Categorical (4, one-hot): `on_ball_event_type`, `phase_label`, `defender_functional_role`, `period`
- Boolean (6, passthrough): `is_in_defending_box`, `is_in_attacking_box`, `is_wide_lane`,
  `has_option_2`, `has_option_3`, `is_goal_side_of_nearest_attacker`
- Continuous (26) + discrete (2) = 28 numeric (median-impute + standard-scale)

`screened_option_was_avoided` and `has_screened_outcome` (the two leakage exclusions
confirmed in `feature_config.py`'s `EXCLUDED_COLUMNS_PASSIVE` and
`FEATURE_LOCK_CONFIRMATION.json`) are asserted absent from the locked list defensively
in the training script, not just assumed absent.

**One data-quality note surfaced while building the design matrix:** `is_goal_side_of_nearest_attacker`
is stored as an object column (`True`/`False`/`None`), not a clean boolean -- 279 of
1,593,181 rows (0.018%) are `None` (no nearest attacker to compare against for that
row). These are treated as `0.0` (the majority/negative class) by the same
astype-then-fillna boolean-passthrough pattern already used on the active leg; the
volume is negligible enough not to warrant a separate missingness flag at this stage.

## 2. Row unit: not one-row-per-independent-event

`feature_config.py`'s own `row_description` for this dataset is explicit: **"one row per
visible defender-slot per on-ball attacking event."** Each on-ball attacking event
(a pass, carry, shot, or dribble) produces one row per visible defending-team
player-slot in that frame. Rows sharing the same `event_id` describe the *same*
attacking context -- same ball position, same attacking shape, same target -- differing
only in which defender-slot's geometry (`defender_x`/`defender_y`, `marking_tightness`,
`engagement_distance_to_carrier`, etc.) is being described. This is a structurally
different grain from the active leg, where each row is one independent defensive action.

The canonical `StratifiedGroupKFold` split (grouped by `match_id`, shared unchanged
with the active leg via `dax.models.splits`) still fully prevents cross-split leakage,
since it groups at the match level -- every row for a given match, and therefore every
row for a given `event_id`, lands in exactly one fold or the held-out test set. But the
within-fold row correlation between same-`event_id` rows is real: it is not accounted
for by any of the metrics below (which treat every row as an independent Bernoulli
trial), and it should be read as a property of this leg's data rather than silently
assumed away. A future validation pass that wants an honest effective-sample-size
estimate would need to account for this clustering explicitly (e.g. via a cluster
bootstrap or a per-event aggregation), which this Rung 0 baseline does not attempt.

## 3. Base rate -- not directly comparable to the active leg

5.97% overall (train+val: 6.08%, held-out test: 5.51%) vs the active leg's 7.79%. These
numbers describe different things and should not be read as "passive defending is
associated with fewer shots than active defending" or any other direct comparison --
the active leg's rows are one-per-actual-defensive-action, the passive leg's rows are
one-per-(defender-slot, attacking-event), a fundamentally different unit of observation
and a different denominator.

## 4. Split

The canonical, frozen match-grouped split in `outputs/models/splits/match_assignment.json`
(5-fold, seed 42, computed once by `scripts/pipeline/compute_canonical_split.py`, shared
unchanged with the active leg -- not recomputed here). Train+val: 1,277,374 rows / 92
matches. Held-out test: 315,807 rows / 23 matches. Same 92/23 match split as the active
leg, confirming the split is genuinely match-level and shared, not leg-specific.

## 5. Variants

| Variant | Model | Notes |
|---|---|---|
| `p0_dummy` | `ConstantClassifier` | training-fold positive rate for every row |
| `p1_unweighted` | `LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)` | no `class_weight`, all 38 features |
| `p2_weighted` | `LogisticRegression(class_weight="balanced")` | same features as p1 |
| `p3_weighted_interactions` | p2 + 5 explicit interaction terms | see below |

**Interaction pairs (p3 only):**

| Pair | Evidence |
|---|---|
| `lane_screening_score_option_1` x `marking_tightness` | classified `interactive` in `FEATURE_LOCK_CONFIRMATION.json`'s `numeric_interaction.pairs` |
| `lane_screening_score_option_2` x `engagement_distance_to_carrier` | classified `interactive`, same source |
| `marking_tightness` x `engagement_distance_to_carrier` | classified `interactive`, same source |
| `overload_score` x `attacking_goal_centrality` | classified `interactive`, same source |
| `lane_screening_score_option_3` x `engagement_distance_to_carrier` | not itself tested in the JSON; extends the confirmed option-2 pair to option 3 of the same lane-screening family |

4 of the 5 pairs are pulled directly from `FEATURE_LOCK_CONFIRMATION.json`'s own
`pattern_analysis_findings.numeric_interaction` section (10 passive-side pairs were
tested there; 8 were `interactive`, 1 `substitutive`, 1 `additive` -- these 4 are drawn
from the `interactive` set, excluding the one `additive` pair,
`top_option_2_threat_score` x `top_option_2_distance_from_ball`). The task brief's own
suggested pair `zone_defensive_value` x `engagement_distance_to_carrier` was checked and
**not used**: `zone_defensive_value` was dropped from the locked passive feature set
(`REDUNDANCY_DROPPED_PASSIVE` in `feature_config.py` -- Spearman r=-1.0 vs
`distance_to_defending_goal`, which was kept instead), confirming the brief's own
warning that some prose feature names no longer match the exact locked columns.

## 6. Cross-validation results (5-fold, out-of-fold metrics)

| Variant | PR-AUC (AP) | ROC-AUC | Log loss | Brier | Calib. slope | Calib. intercept | ECE |
|---|---|---|---|---|---|---|---|
| p0_dummy | 0.0578 | 0.4790 | 0.2292 | 0.0571 | 0.882 | -0.322 | 0.0000 |
| **p1_unweighted** | **0.1928** | 0.7487 | **0.2035** | **0.0531** | 0.986 | **-0.032** | **0.0024** |
| p2_weighted | 0.1903 | 0.7507 | 0.5851 | 0.1957 | 0.951 | -2.726 | 0.3542 |
| p3_weighted_interactions | 0.1903 | **0.7516** | 0.5845 | 0.1955 | 0.951 | -2.726 | 0.3538 |

Per-fold average precision mean +/- std: p0 0.0608 +/- 0.0043; p1 0.1924 +/- 0.0147; p2
0.1896 +/- 0.0137; p3 0.1896 +/- 0.0137. Full per-metric fold mean/std for every variant
is in `outputs/models/comparisons/passive_binary_baseline_comparison.csv`.

**PR-AUC ranking: p1 > p2 approx p3 > p0.** This is the same ranking pattern the active
leg found for v1 vs v2/v3, but it was *checked*, not assumed -- p1 (unweighted) has the
best PR-AUC of the three fitted variants in cross-validation, essentially tied with p2
just ahead of it by 0.0025.

## 7. Held-out test readout (all 4 variants, single readout each, NOT for selection)

| Variant | PR-AUC | ROC-AUC | Log loss | Brier | Calib. slope | Calib. intercept | ECE |
|---|---|---|---|---|---|---|---|
| p0_dummy | 0.0551 | 0.5000 | 0.2136 | 0.0521 | 0.916 | -0.335 | 0.0057 |
| **p1_unweighted** | **0.1736** | 0.7498 | **0.1904** | **0.0489** | 0.982 | **-0.134** | **0.0051** |
| p2_weighted | 0.1695 | 0.7515 | 0.5820 | 0.1941 | 0.933 | -2.816 | 0.3572 |
| p3_weighted_interactions | 0.1695 | **0.7526** | 0.5820 | 0.1941 | 0.933 | -2.817 | 0.3571 |

315,807 rows / 23 matches, evaluated exactly once per variant, after all
cross-validation and variant comparison above was finalised -- one readout per variant,
covering all 4 (including p0/p1), unlike the active leg where the readout was split
across two prompts and only covered v2/v3. Consistent with the CV ranking: p1 leads on
PR-AUC and by a wide margin on calibration.

**CV-to-test check:** p1's PR-AUC went from 0.1928 (CV mean) to 0.1736 (held-out test)
-- a **decrease** of about 10% relative. This is the *opposite* direction from the
active leg's v1, which *increased* from CV to held-out test (0.3521 -> 0.3725). Stated
plainly rather than assumed to mirror the active leg: this modest drop is consistent
with ordinary CV-to-test variance at this scale (fold std on CV was +/-0.0147, a
similar order of magnitude to the 0.0192 CV-to-test gap) and with the held-out test set
having a slightly lower base rate (5.51% vs train+val's 6.08%), which mechanically caps
achievable PR-AUC somewhat since precision at any given recall is bounded by the
positive rate. It is not evidence of overfitting on 1.28M training rows with L2
regularization by default -- but it is a real, honestly-reported difference from the
active leg's pattern, not a repeat of it.

## 8. Winner and why

**p1_unweighted wins**, by PR-AUC, on both cross-validation (0.1928, vs 0.1903 for both
p2 and p3) and the held-out readout (0.1736, vs 0.1695 for both p2 and p3). This
reproduces the active leg's v1-wins finding, but the reproduction was checked against
this leg's own numbers rather than assumed to transfer, and the calibration mechanism
behind it is directly confirmed here too, not just presumed:

- **`class_weight="balanced"` (p2/p3) inflates predicted probabilities far past the true
  rate**, exactly as on the active leg. Calibration intercept collapses from -0.03 (p1)
  to -2.73 (p2/p3) in CV, and expected calibration error jumps from **0.0024 to 0.354**
  -- a roughly 150x increase, an even larger relative blowup than the active leg's
  0.004 -> 0.285 (~70x). Log loss and Brier score both roughly triple, the same
  mechanical signature seen on the active leg.
- **ROC-AUC very slightly favours the weighted variants** (0.7516 p3 vs 0.7487 p1 in
  CV, +0.003), the same small-and-not-decision-relevant pattern the active leg showed
  (+0.001 there). ROC-AUC is not the selection metric here for the same reason stated
  on the active leg: this is an imbalanced target (5.97% base rate, more imbalanced
  than the active leg's 7.79%), and PR-AUC is the metric that actually separates these
  variants; ROC-AUC barely moves between any of the three fitted variants.
- **p3's 5 interaction terms do not beat p2 by any meaningful margin** -- p3 is
  marginally *below* p2 on PR-AUC in both CV (0.19028 vs 0.19032) and held-out (0.16947
  vs 0.16954), a difference well inside noise. This mirrors the active leg's v3-vs-v2
  finding (interaction terms added to an already class-weighted linear model don't
  translate the underlying interaction evidence into a better-ranking baseline), though
  here the gap is even smaller than on the active leg (where v3 fell 0.0045 below v2 in
  CV; here it's 0.00004).

**Bottom line:** the active leg's mechanism -- balanced class weighting trades a
negligible ROC-AUC gain for a large, mechanically-explained collapse in calibration,
while genuinely improving nothing about ranking quality on the metric that matters for
an imbalanced target -- reproduces on the passive leg, checked directly against this
leg's own CV and held-out numbers rather than assumed. `p1_unweighted` is this leg's
Rung 0 pick, on the same PR-AUC-plus-calibration reasoning as the active leg's v1.

## 9. Deviations from instructions

- **`claude/passive-defense-build-plan.md` does not exist in this repository.** A
  full-repository search (file names and content) found no trace of this file under
  any name. Every factual claim in this document that the prompt attributed to that
  build plan (row/match counts, base rate, feature count, absence of `player_id`, the
  `screened_option_was_avoided` exclusion, the row-unit description) was instead
  verified directly against the real, current sources: `data/features/passive_defense.parquet`
  itself, `src/eda/feature_config.py`, and `reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json`.
  All of the prompt's specific factual claims checked out against these real sources
  except the interaction-pair suggestion noted in section 5 (`zone_defensive_value`,
  which is not in the locked feature set).
- **Held-out-test readout covers all 4 variants (including p0/p1) in this single run**,
  per this leg's explicit instructions -- unlike the active leg, where Prompts 36/37
  split CV from the held-out readout across two runs and the readout only ever covered
  v2/v3. This was a deliberate instruction difference for this leg, not an inconsistency.
- Generated artifacts under `outputs/models/**` (comparison CSVs, per-variant JSON/joblib,
  charts) are not git-tracked, matching this repository's existing `.gitignore`
  convention (`/outputs/models/**` ignored except `outputs/models/splits/match_assignment.json`).
  They are reproducible with a single command: `python scripts/models/train_passive_binary_baseline.py`.
- No model ladder is built in this document -- Rung 0 only, per the prompt's explicit
  constraint. Ladder rungs and post-lock validation (tournament-stratified check,
  error-slice analysis by `phase_label`/`defender_functional_role`, and an
  event-disjoint or archetype-stability recheck in place of the active leg's
  player-disjoint check, since this leg has no `player_id`) are deferred to a follow-up
  prompt, mirroring the active leg's Prompt 38.

## 10. Non-causal framing

As throughout this project: results in this document describe **correlation with
suppressed shot risk**, not causation. `p1_unweighted`'s coefficients and rankings
describe which defensive geometry and positioning features are statistically
associated with a lower observed rate of a shot in the following 10 seconds -- they do
not establish that any individual defender's positioning *prevents* or *causes* that
outcome, and none of the analysis above attempts to isolate a causal effect.

## 11. Post-lock validation (p1_unweighted)

Run by `scripts/models/validate_passive_binary_baselines.py` (single command, no split
recomputation, no hyperparameter changes, existing comparison/held-out-readout CSVs
untouched). Full outputs under `outputs/models/validation/`. Mirrors the active leg's
Prompt 38, adapted for two real structural differences: no `player_id` to run a
player-disjoint recheck against (replaced by a cluster-aware bootstrap, item 11.3), and
the within-event row correlation flagged in section 2 above, which item 11.3 quantifies
rather than just restates.

### 11.1 Is the p1 pick statistically justified?

Paired over the same 5 canonical CV folds (`significance_p1_vs_p2_p3.json`), refitting
all three variants fold-by-fold:

| Comparison | Mean &Delta; PR-AUC | Std &Delta; | Paired t (p) | Wilcoxon (p) |
|---|---|---|---|---|
| p1 vs p2 | +0.0028 | 0.0014 | t=4.56, **p=0.0104** | W=0, p=0.0625 |
| p1 vs p3 | +0.0028 | 0.0013 | t=4.74, **p=0.0090** | W=0, p=0.0625 |

p1 beats both p2 and p3 in **5 of 5 folds** on PR-AUC. The paired t-test says the gap is
statistically significant at the conventional 0.05 threshold for both comparisons.
Wilcoxon lands at the same 0.0625 floor seen on the active leg -- the minimum p-value
five same-signed paired samples can produce, not a failure to reach significance so
much as the ceiling of what n=5 can show non-parametrically. Read at face value, this
test says the gap is real -- **but it assumes row-level independence within each fold's
PR-AUC estimate**, which section 11.3 directly checks rather than assumes.

### 11.2 Does p1 hold up across tournaments?

`tournament_stratified_p1.json`. Same check as `FEATURE_LOCK_CONFIRMATION.json`'s
tournament-stability section and the active leg's Prompt 38 re-confirmed for this leg's
own matches, not assumed to transfer: **only 2 tournaments are represented** in
`passive_defense.parquet`, not 3. UEFA Euro 2020 (`data/raw/matches/55_43.json`, 51
matches) contributes **zero** rows -- confirmed directly by checking its 51 match IDs
against the dataset's `match_id` column (0 overlap), the same finding as the active leg.
The dataset's 115 matches are FIFA World Cup 2022 (64) and UEFA Euro 2024 (51) only.

| Tournament | Held-out rows/matches | PR-AUC | ROC-AUC | Log loss | Brier | ECE |
|---|---|---|---|---|---|---|
| FIFA World Cup 2022 | 189,333 / 14 | 0.1678 | 0.7497 | 0.1810 | 0.0459 | 0.0063 |
| UEFA Euro 2024 | 126,474 / 9 | 0.1821 | 0.7479 | 0.2044 | 0.0534 | 0.0044 |

Train+val composition: 50 WC2022 matches / 42 Euro2024 matches. p1 performs comparably
on both held-out slices -- PR-AUC is actually slightly *higher* on the smaller Euro 2024
slice (0.182 vs 0.168), ROC-AUC essentially identical (0.748 vs 0.750), and calibration
stays tight on both (ECE 0.0044 / 0.0063). No sign of the model being
tournament-specific. (Both slices clear the &ge;2-matches/&ge;2-positives bar; neither
is flagged as too-thin.)

### 11.3 How much does the row-correlation caveat actually matter?

`cluster_bootstrap_p1_vs_p2.json`. The held-out test set has 315,807 rows across
**40,147 distinct `event_id` groups** (mean 7.87 rows/event, median 8). 1,000 bootstrap
iterations each, seed 42, naive row-level vs cluster-by-`event_id`:

| Quantity | Naive row-level CI (width) | Cluster-by-event CI (width) | Width ratio (cluster/naive) |
|---|---|---|---|
| p1 PR-AUC | [0.1692, 0.1783] (0.00906) | [0.1618, 0.1869] (0.02512) | **2.77x** |
| p1 &minus; p2 PR-AUC | [0.00343, 0.00481] (0.00137) | [0.00275, 0.00570] (0.00295) | **2.15x** |

The cluster-aware CI is **meaningfully wider than the naive one** -- roughly 2.2-2.8x,
not a marginal difference. This is the row-correlation caveat from section 2 made
concrete: treating each row as independent (as item 11.1's paired fold test and every
CV/held-out metric elsewhere in this document does) understates the true uncertainty by
about a factor of 2-3 on this leg's data. **The direction of the finding survives
anyway**: both the naive CI (0.00343 to 0.00481) and the cluster-aware CI (0.00275 to
0.00570) for p1&minus;p2's PR-AUC advantage lie entirely above zero -- p1's edge over p2
is not an artifact of ignoring row correlation, it holds up under the more honest,
wider interval too. What the row-correlation caveat changes is **how confidently the
exact magnitude of that gap, and the precision of item 11.1's p-values, should be
read** -- item 11.1's paired t-test (p=0.0104) is likely overconfident in its stated
precision by roughly this same 2-3x factor, even though its qualitative conclusion
(p1 beats p2/p3) is not overturned by this more careful check.

### 11.4 Where does p1 systematically miss?

`error_analysis_p1.json`, held-out test set (315,807 rows / 23 matches), same p1 fit as
11.2. **Because rows in different `phase_label`/`defender_functional_role` groups can
still share the same `event_id`** (a defending-team slot classified into one group sees
teammates classified into other groups at the same on-ball event), these per-group
counts and metrics are not fully independent of each other either -- the same
row-correlation property quantified in 11.3, restated here rather than assumed away.

**By `phase_label`:** PR-AUC ranges from a low of 0.030 (`wide_defending_proxy`,
n=44,923, positive rate 1.9%) up to 0.270 (`high_press_proxy`, n=35,819, positive rate
13.7%) and 0.240 (`box_defence`, n=20,373, positive rate 13.9%). The same pattern as the
active leg: **PR-AUC tracks positive rate closely** -- phases with the rarest positives
are where the model is worst at ranking them, the expected behaviour of a PR-based
metric on imbalanced slices. Calibration is good almost everywhere (gaps &le;0.009)
except `transition_defence` (gap +0.017, over-predicting by roughly half its 3.5%
positive rate in that slice) -- worth a closer look if this model is used for per-phase
thresholds.

**By `defender_functional_role`:** 5 of 6 categories are broadly consistent (PR-AUC
0.137-0.190, calibration gaps &le;0.007, n=31,289 to 165,541). The 6th,
**`unclassified` (n=259, 0.02% of the test set), is a clear outlier**: PR-AUC 0.013,
ROC-AUC 0.398 (*below* 0.5, i.e. worse than random ranking on this tiny slice), and a
large calibration gap (+0.043, over-predicting roughly 4x its 1.2% positive rate). This
category is reserved by construction for defender-slots with fewer than 2 nearby
teammates to classify against (`defender_functional_role` bucket-fix note in
`FEATURE_LOCK_CONFIRMATION.json`) -- it is the rarest, most structurally unusual slice
in the dataset, and 259 rows (with perhaps only 3 positives, given a 1.2% rate) is far
too few for any metric computed on it to be read as a real finding about the model
rather than noise. Flagged as low-n, not treated as a genuine defect.

### 11.5 Chart integrity

Automated pass, not a visual audit: all 16 expected PNGs (4 variants &times;
{`calibration_curve`, `precision_recall_curve`, `roc_curve`, `prediction_distribution`})
exist, are non-empty, and are pixel-dimension-consistent per chart type across all 4
variants. No missing, near-zero, unreadable, or inconsistently-sized files.

### 11.6 Bottom line

`p1_unweighted`'s pick is well-supported, with one honest caveat quantified rather than
just flagged: its CV edge over p2/p3 is statistically significant by a paired t-test,
and it performs comparably across both tournaments actually present in the dataset. The
one real complication this leg has that the active leg didn't -- correlated rows within
the same `event_id` -- was checked directly rather than left as an unquantified
disclaimer: a cluster-aware bootstrap shows the naive row-level uncertainty estimate is
about 2-3x too narrow, so item 11.1's p-values should be read as directionally correct
but overconfident in their precision. The qualitative conclusion (p1 beats p2/p3) is not
overturned by this more careful accounting -- both the naive and cluster-aware
confidence intervals for p1's advantage over p2 stay entirely positive. p1's main known
weakness is PR-AUC degrading on low-positive-rate phase slices (`wide_defending_proxy`,
`counterpress_after_loss`), the same expected consequence of class imbalance seen on the
active leg, plus one very-low-n `defender_functional_role` category (`unclassified`,
n=259) that should not be read as a real finding given its size.
