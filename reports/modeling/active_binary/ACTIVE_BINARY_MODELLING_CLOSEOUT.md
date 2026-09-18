# Active-Binary Baseline Modelling Closeout — Prompts 36–46

*Scope: the first phase of modelling on the active-binary leg (`target_future_shot_10s`,
`data/features/player_defensive_actions.parquet`), built on top of the feature/target/split
lock covered by `reports/eda/PATTERN_ANALYSIS_CLOSEOUT.md`. This is a decision record, not a
re-listing of every number already in `ACTIVE_BINARY_BASELINE_SUMMARY.md` — that document and
the validation JSONs under `outputs/models/validation/` are the source of truth for full tables;
this one states what was decided and why, and links out rather than duplicating. Sections 1–6
written 2026-09-18 (Prompts 36–39); section 7 added 2026-09-18 (Prompt 43); section 8 added
2026-09-18 (Prompt 46), reading only already-committed outputs plus CV-OOF predictions
regenerated at already-locked (not re-tuned) hyperparameters -- no model re-run/retrain for this
report or these sections.*

> **Status (Prompt 46, current):** `v1e_gradient_boosting_calibrated` was promoted to standing
> reference model for the active-binary leg, superseding `v1c_systematic_interactions` (itself
> promoted in Prompt 43) — see section 8. Sections 1–7 below describe how `v1_unweighted` and
> then `v1c_systematic_interactions` earned standing-reference status in turn and remain accurate
> as history; they are not rewritten to reflect the later promotion.

## 1. What was decided

**`v1_unweighted`** — a plain, unweighted logistic regression on the 32 locked active
features, no `class_weight`, no interaction terms — is the standing baseline for the
active-binary leg.

| Metric | Held-out test value |
| --- | --- |
| PR-AUC (primary) | **0.372** |
| ROC-AUC | 0.819 |
| ECE (calibration) | 0.011 |

Full CV/test tables, coefficients, and charts: `ACTIVE_BINARY_BASELINE_SUMMARY.md` sections
1–6 (Prompt 36) and section 5 (Prompt 37 held-out addendum).

## 2. Why unweighted beat weighted

`class_weight="balanced"` (`v2_weighted`, `v3_weighted_interactions`) reweights the training
loss as if positives and negatives were 50/50, when the true class balance is 7.2% positive.
That mismatch does two things at once: it distorts the ranking the model learns (both weighted
variants scored *lower* PR-AUC than the unweighted one, 0.338 and 0.333 vs 0.352 in CV), and it
badly breaks calibration, because the model's raw predicted probabilities are now calibrated to
the artificial 50/50 world, not the real one — ECE jumps from 0.011 (v1, held-out test) to
0.29 (v2/v3), a roughly **27x** difference. For this target, plain maximum-likelihood logistic
regression was simply the better choice; balancing bought nothing on ranking and cost a lot on
calibration.

## 3. Why the pick is trustworthy, not lucky

Prompt 38 ran 4 independent checks on `v1_unweighted` specifically to stress-test the CV
result before treating it as settled (full numbers: `ACTIVE_BINARY_BASELINE_SUMMARY.md`
section 8, and the JSONs under `outputs/models/validation/`):

- **Statistically real edge over v2/v3** — v1 beats both in 5/5 CV folds; paired t-test
  significant (p=0.0005 vs v2, p=0.0044 vs v3). Wilcoxon sits at the exact floor a 5-fold
  paired test can reach (p=0.0625) given every fold agrees in direction — reported as such,
  not rounded up to "significant" by both tests.
- **Tournament-stable** — performs comparably on both tournaments actually present in the
  active dataset. (Data-coverage fact worth remembering: UEFA Euro 2020 contributes **zero**
  rows to this dataset — it's WC2022 + Euro2024 only, confirmed by match-ID overlap, not a
  bug in any check.)
- **Player-disjoint-stable** — a genuine `GroupKFold(player_id)` fit-and-score run (not just
  the earlier fold-balance diagnostic from Prompt 31) reproduces the match-grouped CV result
  within noise (PR-AUC 0.359 vs 0.352). No player-identity leakage.
- **One real calibration caveat** — per-`phase_label` error analysis found PR-AUC tracking
  positive rate almost directly (expected behaviour for a PR-based metric on imbalanced
  slices, not a phase-specific weakness). One genuine flag: `transition_defence` over-predicts
  by roughly 60% relative to its own positive rate — worth watching if this model is ever used
  for per-phase probability thresholds.

## 4. Rung 1 verdict (Prompt 39): quadratic terms did not move the needle

`v1b_quadratic` added squared-standardized terms for the 4 features EDA already flagged as
U-shaped (`defender_spread`, `distance_to_attacking_box`, `visible_defender_count`,
`defenders_between_ball_and_attacking_goal`), isolating feature *shape* as the one changed
variable versus v1.

**Result: not adopted.** The CV gain was real but small (+0.0021 PR-AUC, paired t p=0.012,
Wilcoxon at the same n=5 floor as section 3) — and it did **not** hold up on the held-out
test set: `v1b_quadratic` scored **0.3713** vs v1's **0.3725**, marginally *lower*. Per the
same discipline used throughout Prompts 36–38 — CV picks a candidate, the held-out test is
the one-time confirmation, and it is never re-litigated after the fact to fit a preferred
narrative — that is the answer to the question rung 1 was built to ask. This is not "promising,
needs more testing"; the held-out readout already happened, once, and it said no.

Read as a finding rather than a failure: **curvature on these 4 specific features, in
isolation, is not the bottleneck** on v1's remaining gap. That is useful negative information
— it rules out the cheapest possible next step (more polynomial terms on already-identified
U-shaped features) and points the ladder toward something structurally different.

## 5. What this rules in/out for the next phase

Rung 1 isolated *single-feature shape* as a variable and found it wasn't where the headroom
is. Two candidate explanations remain open, and this report does not pick between them:

- **(a) Interactions matter more than any single feature's shape.** `v3_weighted_interactions`
  only tested 5 manually chosen interaction pairs, under `class_weight="balanced"`, which
  itself already made ranking worse for unrelated reasons (section 2). A systematic,
  regularization-driven interaction search on top of the *unweighted* objective hasn't been
  tried yet.
- **(b) The ceiling for a linear-in-features model is close to reached**, and the real next
  step is a model family that discovers higher-order interactions and non-linear shape
  automatically — gradient boosting, most obviously — rather than continuing to hand-engineer
  terms into a logistic regression.

Both are live hypotheses. Which one is right is the open question for the next prompt in the
ladder, not a foregone conclusion either way.

## 6. Known open items, carried forward (not re-litigated here)

- **Duplicate v1 artifact pair.** `outputs/models/classification/v1_unweighted.joblib`/`.json`
  (the original Prompt 36 fit) and `v1_unweighted_final_fit.joblib`/`.json` (a near-identical
  refit produced during Prompt 37's held-out readout, since the original's preprocessing state
  wasn't persisted alongside it — see `ACTIVE_BINARY_BASELINE_SUMMARY.md` section 5's
  provenance note) both exist on disk. Flagged, not cleaned up — needs Varun's OK before
  either is deleted, since neither is git-tracked and the duplication is currently harmless
  (both are gitignored, reproducible outputs, not a correctness issue).
- **`configs/models.yaml` / `train_models.py`'s `max_rows` cap.** This is what produced the
  original stale, fixture-scale (`rows=20, matches=4`) `b0`-`b8` baselines retired in Prompt
  36. Still unfixed, deliberately — the same config also drives `b5_360_geometry`,
  `b7_full_with_360`, and `r4_full_with_360`, which run at real scale and feed the two-part
  xG hurdle-model sensitivity analysis and a production-fixture test. Fixing the cap is a
  separate, scoped task, not part of this leg's baseline work.

## 7. Promotion decision: `v1c_systematic_interactions` (Prompt 43)

Rung 2 (`v1c_systematic_interactions`, Prompt 41) cleared its own 3-gate ladder check: paired
significance vs v1 and v1b_quadratic (p=0.0079, p=0.0199, 5/5 folds), a held-out-test gain that
actually survived (PR-AUC 0.3747 vs v1's 0.3725 — rung 1's didn't), and a player-disjoint
recheck with zero overlap (full detail: `ACTIVE_BINARY_MODEL_LADDER.md` section 3). That gate
answers "is this rung's isolated change real?" — a smaller question than "should this replace
the standing baseline everyone downstream builds on?", which needs the same depth of scrutiny
`v1_unweighted` itself went through in section 3 above, not just the ladder gate. This section
runs that scrutiny and states the verdict.

**A population note, stated plainly:** Prompt 38's original tournament/error-slice numbers for
v1 (section 3 above) were built by scoring the *held-out test set* with v1's final fit — verified
by row count (10,660 rows / 23 matches, matching the test set exactly, not train+val's 45,408 /
92). Since v1c's held-out test set was already read once in Prompt 41 and this prompt's
constraints forbid reading it again, the checks below instead use genuine 5-fold cross-validated
out-of-fold (OOF) predictions on train+val for **both** v1 and v1c — an identical, fair
population for both models, though not a literal replay of Prompt 38's held-out-test-based v1
numbers (which remain unchanged and available in `tournament_stratified_v1.json` /
`error_analysis_v1.json` for reference). Full detail and the population note in full:
`outputs/models/validation/tournament_stratified_v1c.json` and
`.../error_analysis_v1c.json`.

### 7.1 Tournament-stratified check

Confirmed again (not assumed): train+val still spans only FIFA World Cup 2022 and UEFA Euro
2024. v1c beats v1 on **both** tournaments, by almost identical absolute margins:

| Tournament | v1 PR-AUC (CV OOF) | v1c PR-AUC (CV OOF) | &Delta; |
|---|---|---|---|
| FIFA World Cup 2022 | 0.3276 | 0.3383 | +0.0108 |
| UEFA Euro 2024 | 0.3765 | 0.3873 | +0.0108 |

Calibration slope moved *closer* to the ideal 1.0 for v1c on both tournaments (0.956&rarr;1.001
WC2022, 0.984&rarr;1.002 Euro2024); ECE stayed small for both models on both slices (v1:
0.0039/0.0059, v1c: 0.0061/0.0042 — a small, mixed shift, not a calibration breakdown on either
tournament). **Verdict: passes.** The gain is not concentrated in one tournament's quirks.

### 7.2 Error-slice comparison against v1's known weak spots

Prompt 38 flagged `wide_defending_proxy` (PR-AUC 0.077) and `settled_mid_block_proxy` (0.090) as
the weakest `phase_label` slices, and `transition_defence` as a real calibration concern
(+0.0254 over-prediction, held-out test). On the CV-OOF population used here (absolute numbers
differ from those held-out-test figures because the population differs, per the note above — the
comparison that matters is v1 vs v1c *within this same population*):

| Slice | v1 PR-AUC | v1c PR-AUC | Verdict |
|---|---|---|---|
| `wide_defending_proxy` | 0.046 | 0.052 | (a) improves |
| `settled_mid_block_proxy` | 0.068 | 0.082 | (a) improves |
| `transition_defence` | 0.277 | 0.311 | (a) improves substantially |
| `high_press_proxy` | 0.455 | 0.449 | (c) small regression (-0.006) |
| remaining 3 `phase_label` slices | -- | -- | (a) all improve |

6 of 7 `phase_label` slices improve; the one regression (`high_press_proxy`) is small and not a
new serious weak spot (still the strongest-performing phase for both models). Calibration gaps
stay under 0.003 in absolute value for both models across every `phase_label` slice on this
population — no calibration collapse anywhere.

`position` (23 slices, full table in the JSON): the large majority improve, a handful show small
regressions (largest: `Center Forward` -0.027, `Left Midfield` -0.026) — none approaching the
scale of `transition_defence`'s improvement, and none flip a previously-fine slice into a bad
one. Two low-row-count slices (`Goalkeeper`, n=701; `Right Attacking Midfield`, n=547) show a
calibration gap increase from ~0.002 to ~0.016-0.018 for v1c — small in absolute terms, on the
two thinnest slices in the table, flagged honestly as a minor caveat rather than treated as
disqualifying. **Verdict: passes**, with that one noted, non-disqualifying caveat.

### 7.3 Interaction sanity check (top 10 surviving terms)

Going one level deeper than `ACTIVE_BINARY_MODEL_LADDER.md` section 3.5's first pass, on the top
10 surviving terms by `|coefficient|` only:

1. `distance_to_attacking_box`&sup2; (+0.322) — U-shaped box-distance effect (elevated risk very
   close to goal and in deep transitions). Matches the EDA-documented shape. **Sensible.**
2. `distance_to_attacking_box` &times; `defender_attacker_gap_x` (-0.211) — zone-dependent
   defensive-gap effect; plausible but the exact sign of `defender_attacker_gap_x`'s convention
   makes a fully confident read hard from the coefficient alone. **Plausible, not fully
   verifiable.**
3. `defender_attacker_gap_x`&sup2; (-0.207) — same caveat as #2. **Plausible, not fully
   verifiable.**
4. `nearest_attacker_distance`&sup2; (+0.142) — elevated risk both very close to (compact
   danger) and very far from (open space, transitions) the nearest attacker. **Sensible.**
5. `attacking_goal_centrality` &times; `visible_defender_count` (+0.131) — central position with
   many defenders visible; most plausibly a phase proxy (congested central/box-defence moments
   are themselves dangerous) rather than "more defenders cause more danger" literally.
   **Sensible as a situational proxy, not a literal causal reading.**
6. `distance_to_attacking_box` &times; `defenders_between_ball_and_attacking_goal` (-0.124) —
   far from goal + well-defended reduces risk, close + poorly-defended raises it. **Sensible.**
7. `nearest_attacker_distance` &times; `attacker_defender_ratio` (+0.123) — space plus a
   numerical attacking overload raises risk. **Sensible.**
8. `nearest_attacker_distance` &times; `attacker_spread` (+0.123) — space plus a stretched
   defence raises risk. **Sensible.**
9. `visible_defender_count` &times; `attackers_within_5m` (-0.120) — defensive presence near
   the attack lowers risk. **Sensible.**
10. `angle_to_attacking_goal`&sup2; (-0.117) — risk falls at extreme (wide) shooting angles
    relative to central ones: the textbook shooting-angle effect. **Sensible.**

**Verdict: passes.** 8 of the top 10 are clearly football-sensible; the other 2 are plausible but
not confidently verifiable from the coefficient sign alone (both involve the same engineered
feature, `defender_attacker_gap_x`, whose sign convention this check doesn't have enough context
to fully pin down) — neither looks like an implausible sign or an out-of-scale magnitude (all 10
sit in a similar ~0.11-0.32 range). No evidence of the top terms chasing noise.

### 7.4 Verdict: promoted

All 4 criteria hold:

- [x] Beats v1 on **both** tournament slices, by nearly identical margins (+0.0108 each) — not
  reliant on one tournament.
- [x] Error-slice comparison shows net improvement (6/7 `phase_label` slices, the large majority
  of `position` slices) with only small regressions and no new serious weak spot.
- [x] Top 10 surviving interaction terms are mostly (8/10) clearly football-sensible, the
  remaining 2 plausible-but-unverifiable rather than noise-shaped.
- [x] Calibration stays as good as v1's on the tournament and slice breakdowns too (calibration
  slope closer to 1.0 on both tournaments; ECE small and mixed rather than degraded; one minor,
  low-n calibration caveat noted in 7.2, not disqualifying).

**`v1c_systematic_interactions` is promoted to standing reference model for the active-binary
leg.** `v1_unweighted` remains documented as the Rung 0 baseline it was built from and measured
against throughout this promotion process (sections 1-6 above are unchanged history, not
rewritten). Downstream work on this leg should now build on `v1c_systematic_interactions`
(`outputs/models/classification/v1c_systematic_interactions.joblib`, C=0.1, 238-column design
matrix per `scripts/train_v1c_systematic_interactions.py`) rather than `v1_unweighted`.

## 8. Promotion decision: `v1e_gradient_boosting` (Prompt 46)

Rung 4 (`v1e_gradient_boosting`, Prompt 45) cleared its own 3-gate ladder check against v1c and
v1d: held-out PR-AUC 0.4305 (vs v1c's 0.3747), significant (p=0.0013 vs v1c, p=0.0189 vs v1d,
5/5 folds both), player-disjoint-stable. Unusually, the raw variant's *aggregate* held-out ECE
(0.0068) looked better than v1c's own (0.0103) -- no obvious calibration trade-off at the
aggregate level, unlike rung 3's Random Forest. That gate answers "is this rung's change real?"
-- not "should this replace the standing reference model?", which needs the same depth of
scrutiny `v1c` itself went through in section 7 before it replaced `v1_unweighted`. This section
runs that scrutiny, and it produces a more nuanced answer than Rung 4's own numbers suggested.

All three checks below run on train+val 5-fold canonical CV out-of-fold (OOF) predictions, the
same population and method Prompt 43 used for v1c's promotion audit -- the held-out test set is
not read again anywhere in this section. `v1`'s and `v1c`'s CV-OOF breakdowns are reused directly
from `tournament_stratified_v1c.json` / `error_analysis_v1c.json` (Prompt 43), not recomputed;
`v1d`'s and `v1e`'s OOF are regenerated at their already-locked hyperparameters (no re-tuning),
since neither prompt 44 nor 45 persisted per-row OOF scores to disk.

### 8.1 Tournament-stratified check

Confirmed again: train+val still spans only FIFA World Cup 2022 and UEFA Euro 2024. Both `v1e`
variants beat `v1c` decisively on **both** tournaments:

| Tournament | v1c | v1d | v1e raw | v1e calibrated |
|---|---|---|---|---|
| FIFA World Cup 2022 | 0.3383 | 0.3701 | 0.3897 | 0.3937 |
| UEFA Euro 2024 | 0.3873 | 0.4327 | 0.4476 | 0.4555 |

The calibrated variant's tournament-level ECE is small on both slices (0.0039 / 0.0040). Not a
single-tournament artifact. **Verdict: passes**, for both variants. Full numbers:
`outputs/models/validation/tournament_stratified_v1e.json`.

### 8.2 Error-slice comparison -- the check that changed the outcome

This is where the promotion decision actually turned. Prompt 45's aggregate held-out ECE for raw
`v1e` (0.0068) looked *better* than v1c's -- but on CV-OOF `phase_label` slices, raw `v1e`'s
calibration gap is **6-19x larger than v1c's on 6 of the 7 slices** (e.g. `box_defence`:
&plusmn;0.0010 for v1c vs &plusmn;0.0191 for raw v1e; `wide_defending_proxy`: &plusmn;0.0012 vs
&plusmn;0.0122). The aggregate held-out number did not generalise to the slice level -- exactly
the failure mode this prompt was written to check for explicitly rather than assume away.

That finding triggered the brief's override condition: checking whether
`v1e_gradient_boosting_calibrated` fixes a slice-specific problem the aggregate numbers hid. It
does. Isotonic calibration brings every `phase_label` slice's calibration gap back down --
typically to within v1c's range or better (`box_defence` &plusmn;0.0045, `high_press_proxy`
&plusmn;0.0039 -- still slightly wider than v1c's near-zero gaps there, but an ~4-5x improvement
over raw; `counterpress_after_loss`, `transition_defence`, `wide_defending_proxy` all calibrate
*better* than v1c). The two slices flagged as a caveat when v1c was promoted (`Goalkeeper`,
`Right Attacking Midfield`) are **fixed, not just improved**: calibrated `v1e`'s gap (0.0094,
0.0112) is smaller than v1c's own (0.0182, 0.0159) on both.

On ranking (PR-AUC), the calibrated variant is at least as good as raw on every `phase_label`
slice (isotonic's monotonic rescaling re-ranks slightly, but never worse), and beats v1c on 5 of
7 slices, several substantially (`box_defence` +0.091, `high_press_proxy` +0.085). Two slices are
a genuine, honestly-reported exception: `settled_mid_block_proxy` is essentially flat (0.0821
&rarr; 0.0779, a small regression) and **`wide_defending_proxy` regresses meaningfully**
(0.0524 &rarr; 0.0403, roughly -23% relative) -- this was already the single weakest slice under
v1c and gets worse, not better, under `v1e_gradient_boosting_calibrated`. Per the same standard
applied throughout this project (a slice going from bad to worse is a real concern even when the
aggregate number improves), this is flagged plainly as a caveat, not smoothed over -- it is the
one place this promotion is not a clean win. It is judged non-disqualifying because it is a
single already-known-weak, low-positive-rate slice (not a new weak spot), the regression is
partial rather than catastrophic, and every other slice-level signal (5/7 phase slices, both
tournament slices, both previously-caveated position slices) points the other way. **Verdict:
passes, with one explicitly non-disqualifying caveat** (`wide_defending_proxy`). Full numbers:
`outputs/models/validation/error_analysis_v1e.json`.

### 8.3 Behavioral sanity check (top 8 gain-importance features, marginal 10-bin view)

Of `v1e`'s top 8 features by gain importance, 6 have a documented monotonic direction in the EDA
atlas and reproduce it cleanly in this marginal (uncontrolled, one-feature-at-a-time) view, with
no discontinuous or noise-shaped jumps anywhere: `match_time_seconds`, `attacking_goal_centrality`
and `possession_elapsed_seconds` increasing; `attacker_spread` and `attacker_defender_ratio`
decreasing, matching their documented directions exactly.

Of the 4 features flagged U-shaped since Rung 1, 2 are in this top-8 list, with a mixed but
informative result, reported honestly rather than rounded to a clean "confirmed":

- **`distance_to_attacking_box` reproduces the documented U-shape cleanly and strongly**: mean
  predicted probability falls from 0.135 near the goal to a minimum of 0.037 at mid-range
  (30-50m) and rises back to 0.141 at the far end -- a textbook U, and consistent with v1c's
  largest surviving quadratic term (`distance_to_attacking_box`&sup2; = +0.322, rung 2, section
  3.5) and rung 1's shape-mismatch finding for this feature's linear coefficient. Three
  independent methods (L1 quadratic term, this marginal GBM view, and the original EDA
  correlation shape) now agree.
- **`defender_spread` does *not* show the documented U-shape in this view** -- it is a clean,
  smooth, strictly monotonic *decrease* across all 10 bins (0.176 &rarr; 0.036), with no upturn
  at high values. This is not noise or a discontinuity (the decline is smooth and monotonic
  throughout), but it is a genuine discrepancy against the atlas's documented U-shape for this
  feature (rho=-0.188, "strongest active correlate," per `active_numerical_target_atlas.json`)
  and against rung 1's small positive quadratic coefficient (+0.047) for it. Read plainly: an
  uncontrolled marginal view is not the same claim as a univariate correlation or a linear
  model's isolated quadratic term, and the three don't fully agree here -- worth a note for
  future work, not resolved in this prompt.

`nearest_attacker_distance` (not one of the 4 documented U-shaped features, but flagged in Prompt
43's coefficient check as carrying a v2 coefficient "not well-supported by univariate evidence")
shows a mild, real dip-then-rise across its 10 bins (0.075 &rarr; 0.069 &rarr; 0.108) -- a weak
but structured non-monotonic pattern, independently observed via a completely different
(non-parametric, marginal) method, consistent with Prompt 43's "likely a multivariate effect"
hypothesis for this feature.

**Verdict: passes, with one qualified exception** (`defender_spread`'s marginal shape doesn't
match its documented univariate U-shape) that is noted rather than disqualifying -- nothing in
the top 8 looks like noise-chasing, and the majority of directional/shape claims, including the
strongest and cleanest one (`distance_to_attacking_box`), are confirmed. Full bin tables:
`outputs/models/validation/behavioral_sanity_v1e.json`.

### 8.4 Which variant, and the verdict

Per the brief's explicit instruction to default toward the raw variant *unless* a slice-specific
calibration problem is found that the calibrated variant fixes: section 8.2 found exactly that
problem, and section 8.2 also confirmed the calibrated variant fixes it. **The calibrated
variant, not raw, is the one promoted.**

All 4 criteria hold for `v1e_gradient_boosting_calibrated`:

- [x] Beats v1c on **both** tournament slices, decisively (+0.055 WC2022, +0.068 Euro2024) --
  not reliant on one tournament.
- [x] Error-slice comparison shows net improvement on the large majority of slices (5/7
  `phase_label`, both previously-caveated `Goalkeeper`/`Right Attacking Midfield` slices actually
  fixed), with one honestly-flagged, non-disqualifying caveat (`wide_defending_proxy` regresses).
- [x] Top-8 marginal shapes are mostly football-sensible and none look noise-shaped; the
  strongest of the 4 previously-documented U-shaped features (`distance_to_attacking_box`)
  reproduces cleanly, with one qualified exception (`defender_spread`) noted rather than
  disqualifying.
- [x] Calibration is at or near v1c's level on the tournament and error slices specifically, not
  just in aggregate -- confirmed only after checking explicitly, since the raw variant's
  aggregate number did not generalise to the slice level and would have been a misleading basis
  for this decision on its own.

**`v1e_gradient_boosting_calibrated` is now the reference model for the active-binary leg,
superseding `v1c_systematic_interactions` (Prompt 43).** `v1c_systematic_interactions` and
`v1_unweighted` remain documented as the rungs/baseline they were measured against (sections 1-7
above are unchanged history, not rewritten). Downstream work on this leg should now build on
`v1e_gradient_boosting_calibrated`
(`outputs/models/classification/v1e_gradient_boosting_calibrated.joblib`, isotonic calibration,
`n_estimators=382` fixed, `learning_rate=0.01`/`num_leaves=63`/`min_child_samples=30`, per
`scripts/train_v1e_gradient_boosting.py`) rather than `v1c_systematic_interactions`.

The raw, uncalibrated `v1e_gradient_boosting` is explicitly **not** the promoted variant, despite
having the single best aggregate held-out PR-AUC and ECE of anything built so far -- this
section's slice-level check found that its good aggregate calibration number did not hold at the
slice level, which the calibrated variant corrects.
