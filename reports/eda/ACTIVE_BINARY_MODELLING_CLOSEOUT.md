# Active-Binary Baseline Modelling Closeout — Prompts 36–39

*Scope: the first phase of modelling on the active-binary leg (`target_future_shot_10s`,
`data/features/player_defensive_actions.parquet`), built on top of the feature/target/split
lock covered by `reports/eda/PATTERN_ANALYSIS_CLOSEOUT.md`. This is a decision record, not a
re-listing of every number already in `ACTIVE_BINARY_BASELINE_SUMMARY.md` — that document and
the Prompt 38 validation JSONs under `outputs/models/validation/` are the source of truth for
full tables; this one states what was decided and why, and links out rather than duplicating.
Written 2026-09-18, reading only already-committed outputs (no models re-run for this report).*

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
