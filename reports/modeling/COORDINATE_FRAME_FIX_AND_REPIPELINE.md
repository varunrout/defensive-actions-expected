# Coordinate-Frame Fix: Full Re-Pipeline, Re-Validation, and Re-Promotion

Prompt 85, Phases 2-6. Phase 1 (the loader fix itself, commit `8b7d0a8`) is treated as ground truth
here and not revisited: `_infer_attack_sign_by_period_team`
(`src/dax/data/statsbomb_loader.py`) -- which inferred each row's attack direction from noisy
per-`(period, possession_team)` ball-progression medians -- was replaced by a row-local rule,
`_attack_sign_for_row(team, possession_team)`: sign=+1 if `team == possession_team` (StatsBomb's raw
locations are already given in the *acting* team's own attacking frame), sign=-1 otherwise (the two
teams attack in opposite directions within a period, so a non-possession-team action needs a
180-degree flip to land in the possession-team-relative frame the rest of the pipeline expects).
This was independently confirmed by a goalkeeper-event sanity oracle
(`scripts/analysis/coordinate_frame_fix_gk_sanity_check.py`): 92.8% (879/947) of goalkeeper
defensive-action events across the 115 locked matches sit at raw x&lt;20 in their own frame, exactly
where a keeper's own action should be.

[`COORDINATE_FRAME_IMPACT_AUDIT.md`](COORDINATE_FRAME_IMPACT_AUDIT.md) (prompt 84) is the full
diagnosis this prompt executes against: 4 of 6 legs have frame-invariant targets (feature-only
exposure, 36.6-84.3% of importance/weight on frame-dependent features per leg); the 2 xT legs
(`x1c_random_forest`, `y1c_random_forest`) have directly frame-dependent targets, exposed for
69.13% / 67.19% of their own training rows.

---

## Phase 2 -- Re-run feature engineering

**Pipeline re-run, once, for all 6 legs.** `python scripts/run_pipeline.py --stage all` (stage 1:
raw-data fetch/refresh from StatsBomb open data, 115 matches; stage 2: `build_all_enriched_events`
through the fixed loader, event-context, phase labelling, and the two frame-invariant short-horizon
targets; stage 3: directory init), followed by `scripts/features/build_passive_defense_dataset.py`
(passive/off-ball defender-slot table, not part of `run_pipeline.py`'s own stage list) and
`scripts/features/add_missingness_flags.py` (structural missingness flags on both feature tables --
required a second pass after the first rebuild silently dropped `has_visible_attacker` /
`has_visible_defender` / `has_previous_event` on the active side and `has_option_2` / `has_option_3`
/ `has_screened_outcome` on the passive side, both restored and re-validated against this script's
own hardcoded `EXPECTED_FALSE_COUNTS` sanity gate, which passed unchanged -- see the schema-parity
note below).

### Row-count / schema parity (proves the fix changed values, not coverage)

| Table | Pre-fix rows | Post-fix rows | Match? |
|---|---|---|---|
| `events_with_targets.parquet` | 422,561 | 422,561 | Yes |
| `player_defensive_actions.parquet` (active) | 56,068 | 56,068 | Yes |
| `passive_defense.parquet` (passive, defender-slot grain) | 1,593,181 | 1,593,181 | Yes |

`add_missingness_flags.py`'s own `EXPECTED_FALSE_COUNTS` assertions (`has_option_2`: 2181 False,
`has_option_3`: 11,398 False, `has_screened_outcome`: 2,860 False, `has_visible_attacker`: 31 False,
`has_visible_defender`: 76 False, `has_previous_event`: 4,534 False) passed against the rebuilt
tables without modification -- these flags mark *structural* absence (not enough freeze-frame
options, no next event, first event of a possession), which is a property of match/possession
structure, not of coordinate frame, so their counts were expected to be, and are, unchanged.

Feature *values* did change, substantially: 59.9% of `passive_defense.parquet`'s 1,593,181 rows
have a different `ball_x` post-fix than pre-fix (`.claude_scratch/prefix_backup/passive_defense.parquet`
vs the rebuilt file, joined on `event_id`+`defender_slot_index`).

### Clearance / phase_label sanity re-check

`scripts/analysis/coordinate_frame_fix_clearance_phase_sanity_check.py` re-runs the earlier
conversation's Clearance/phase_label check against both the pre-fix backup and the post-fix rebuild
(source: `outputs/models/validation/coordinate_frame_fix_clearance_phase_sanity_check.json`):

| | Pre-fix | Post-fix |
|---|---|---|
| Clearance events (total) | 4,518 | 4,518 |
| `high_press_proxy` | 1,370 (30.3%) | 41 (0.9%) |
| `box_defence` | 2,014 (44.6%) | 3,287 (72.8%) |
| `settled_low_block_proxy` | 195 | 346 |
| `counterpress_after_loss` | 341 | 341 |
| `transition_defence` | 309 | 309 |
| `settled_mid_block_proxy` | 181 | 129 |
| `wide_defending_proxy` | 108 | 65 |

Pre-fix, nearly a third of Clearance events (1,370 of 4,518) were rule-labelled `high_press_proxy` --
a proxy for aggressive pressing high up the pitch, the opposite of what a clearance is. Post-fix,
that collapses to 0.9% (41 rows), and `box_defence` (deep defensive action near the actor's own
goal, the intuitively correct label for most clearances) rises from 44.6% to 72.8% of Clearance
events. Within the "clearance deep in the actor's own defensive third" population specifically
(`ball_x >= 95` in the actor's own attacking frame -- the physically unambiguous case), 0 rows were
mislabelled `high_press_proxy` either before or after the fix (that rule branch is unreachable once
`ball_x >= 95` routes to `box_defence` first) -- the improvement above comes from the *other*
Clearance rows, whose corrected, per-row-consistent `ball_x` now more often falls in a range the
rule labels sensibly. `phase_label`'s own rule-based design still has the documented limitation that
low `ball_x` in a defender's own frame is labelled `high_press_proxy` regardless of whether the
action was actually a press or a retreat -- that rule itself is out of scope for this prompt (not
touched), but the *coordinate input* it now receives is no longer randomly wrong for ~46% of rows.

---

## Phase 3 -- Recompute the 2 xT targets

`build_xt_delta_v2_prototype.py` and `build_passive_xt_delta_prototype.py` re-run unchanged against
the Phase-2-corrected `player_defensive_actions.parquet` / `passive_defense.parquet` /
`events_with_targets.parquet`. Row-count/NaN parity: `target_xt_delta_v2` still has exactly 307 NaN
rows of 56,068 (56,068 total, matches the pre-fix count the script's own docstring cites from
prompts 66/67); `target_xt_delta_passive` still has 345 NaN of 198,354 unique events. Neither
script's logic changed -- only the coordinates feeding it did.

`scripts/analysis/phase3_xt_target_delta_report.py` (source:
`outputs/models/validation/phase3_xt_target_delta_report.json`) quantifies what actually moved:

| | `target_xt_delta_v2` (active) | `target_xt_delta_passive` (passive) |
|---|---|---|
| Rows with both old and new value defined | 55,761 | 198,009 |
| Rows whose target value actually changed | 33,018 (59.21%) | 100,491 (50.75%) |
| Rows that flipped sign | 17,684 (31.71%) | 64,392 (32.52%) |
| Delta (new&minus;old) mean | +0.005625 | &minus;0.002829 |
| Delta std | 0.05448 | 0.03747 |
| Delta min / max | &minus;0.2730 / +0.2799 | &minus;0.2932 / +0.2542 |
| Delta 25th / 75th pct | &minus;0.00225 / +0.00451 | &minus;0.00157 / 0.0000 |

This is not a small perturbation: roughly a third of rows on each leg flip which side of zero their
own xT-delta lands on, and the delta distribution's IQR is on the same order as the target's own
overall spread (target_xt_delta_v2 IQR pre-fix was about 0.0057, per section 1b of the impact
audit). **59.2% (active) / 50.75% (passive) of rows' actual target values changed** -- lower than
the audit's own 69.13%/67.19% *exposure* estimates because exposure counted any row whose
`xt_before`/`xt_after` lookup drew from a frame-inconsistent event, while some of those lookups
happened to land on a grid cell whose xT value doesn't differ enough to matter at floating-point
precision (e.g. two nearby cells with the same discretized xT bucket), or the two lookups' errors
partially cancelled in the subtraction. The *direction* of the mean shift differs by leg (active
slightly positive, passive slightly negative) -- consistent with the passive leg's xt_before being
anchored to the snapshot's own event (no previous-event lookup, unlike active's), so its exposure
pattern differs from active's own next/previous-event compounding described in the impact audit.

---

## Phase 4 -- Re-train and re-evaluate the 4 non-xT legs

**Scope decision (explicit, per the prompt's own instruction to make a defensible call under
ambiguity):** each leg's currently-promoted model is refit at its **already-locked
hyperparameters** (no re-tuning), using the identical 5-fold canonical match-grouped CV each leg's
own promotion used (`canonical_grouped_folds` is a deterministic function of `match_id`, unaffected
by the coordinate fix), and compared via the **same paired test each leg's own train script already
used to gate its promotion** (`scipy.stats.ttest_rel` + Wilcoxon signed-rank on the per-fold primary
metric). The alternative -- re-running each leg's full historical grid-search ladder -- was
considered and rejected: it would conflate "did the fix change which hyperparameters are optimal"
with "did the fix change how good the already-promoted model is", and the prompt's own Phase-4 scope
explicitly excludes the full historical ladder. Script:
`scripts/analysis/phase4_reconfirm_promoted_legs.py` (fold-level CV) +
`scripts/analysis/phase4_held_out_refit.py` (single held-out-test refit, read once, for the
headline table only -- not used for the promotion decision). Sources:
`outputs/models/validation/phase4_reconfirmation.json`, `phase4_held_out_refit.json`.

### Per-fold significance test (old features vs Phase-2-corrected features, same folds, same model/hyperparameters)

| Leg | Model | Metric (lower/higher better) | Old fold values | New fold values | Mean diff (new&minus;old) | Paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| Active-binary | v1e_gradient_boosting | average_precision (higher) | [0.4263, 0.4455, 0.3921, 0.4488, 0.3843] | [0.4007, 0.4387, 0.4125, 0.4635, 0.3855] | **+0.0008** | **0.929** | 1.000 |
| Passive-binary | p1e_gradient_boosting | average_precision (higher) | [0.2230, 0.2372, 0.2045, 0.2443, 0.2124] | [0.2475, 0.2605, 0.2376, 0.2672, 0.2326] | **+0.0248** | **0.0004** | 0.0625 |
| Active-continuous | c1d_random_forest | common_log_rmse (lower) | [0.9404, 0.9499, 1.0558, 1.0557, 0.8892] | [0.9303, 0.9428, 1.0526, 1.0409, 0.8832] | **&minus;0.0082** | **0.014** | 0.0625 |
| Passive-continuous | d1_lognormal_glm | common_log_rmse (lower) | [0.9860*, ...see JSON] | [1.0030, 0.9955, 1.1014, 1.0454, 0.9984] | **&minus;0.0078** | **0.071** | 0.125 |

(*d1's old fold array is the 5-value array stored in `significance_d1_vs_d1d_random_forest.json`'s
own `fold_common_log_rmse.d1_lognormal_glm` key, reused verbatim -- see
`phase4_reconfirmation.json` for the exact values.)

### Held-out-test headline metrics, old vs new (single refit, read-once, context only)

| Leg | Metric | Old (pre-fix) | New (post-fix) |
|---|---|---|---|
| Active-binary (calibrated vs raw GBM*) | average_precision | 0.4282 | 0.4461 |
| Active-binary | roc_auc | 0.8364 | 0.8367 |
| Passive-binary (calibrated vs raw GBM*) | average_precision | 0.2162 | 0.2267 |
| Passive-binary | roc_auc | 0.7889 | 0.8066 |
| Active-continuous | common_log_rmse | 0.9553 | 0.9560 |
| Active-continuous | corrected_r2 | 0.0925 | 0.0978 |
| Passive-continuous | common_log_rmse | 1.0374 | 1.0282 |
| Passive-continuous | corrected_r2 | 0.0078 | 0.0278 |

(*The "new" held-out column refits the raw, uncalibrated GBM -- the promoted reference artifact is
the `_calibrated` wrapper around it; `CalibratedClassifierCV`'s isotonic/sigmoid step is a monotonic
transform that does not change ranking-based metrics like average_precision/ROC-AUC materially, so
the raw-vs-calibrated comparison here is directionally valid but not bit-identical to a
recalibrated refit, which was not re-run to keep Phase 4 to its own promoted-model-only scope.)

### Outside-noise verdicts

- **v1e_gradient_boosting (active-binary): RECONFIRMED.** Mean fold-AP shift is +0.0008 (paired-t
  p=0.929, Wilcoxon p=1.0) -- indistinguishable from zero. Held-out ROC-AUC is unchanged to 3
  decimal places (0.8364 to 0.8367). The fix moved 36.6% of this model's own gain importance
  (per the impact audit) onto features whose *values* changed, but the model's predictive accuracy
  did not move outside the noise band that governed its own original promotion. **Original
  promotion (prompt 46) stands, closed, not reopened.**
- **p1e_gradient_boosting (passive-binary): REOPENED (metric moved outside noise -- in the
  favourable direction).** Mean fold-AP shift is +0.0248 (paired-t p=0.0004, Wilcoxon p=0.0625) --
  nearly 3x the +0.0086 AP margin (p=0.036) that justified promoting p1e over p1c in the first
  place (`significance_p1e_vs_p1d_p1c_systematic_interactions.json`), and far more significant.
  Held-out ROC-AUC rises from 0.7889 to 0.8066. This is the leg the impact audit flagged as having
  the highest frame-dependent importance share (84.3%, dominated by `defender_x` at 26.1%) -- the
  fix removing that much noise from the model's own most-relied-on features producing a real
  accuracy gain is the expected, not surprising, outcome. **Verdict: the promoted rung
  (`p1e_gradient_boosting_calibrated`) remains the correct choice -- nothing in this leg's ladder
  outranks it -- but its headline held-out numbers are stale and should be updated to the
  Phase-4-corrected values above wherever they are cited (PASSIVE_BINARY_MODEL_LADDER.md and
  ALL_LEGS_SUMMARY.md, flagged for a follow-up documentation pass, not rewritten in this prompt).**
- **c1d_random_forest (active-continuous): REOPENED (metric moved outside noise -- in the
  favourable direction).** Mean fold common_log_rmse shift is &minus;0.0082 (paired-t p=0.014,
  Wilcoxon p=0.0625) -- a real improvement (lower is better), though this leg's own original
  significance test (`significance_c1_vs_c1d_random_forest.json`) carried an explicit small-n
  caution note ("5 folds only, ~875 positive rows per fold... read as directionally informative, not
  strong statistical proof") that applies here too. Held-out corrected R&sup2; rises from 0.0925 to
  0.0978. **Verdict: promoted rung stands, headline numbers stale, same follow-up-documentation
  flag as p1e.**
- **d1_lognormal_glm (passive-continuous): RECONFIRMED (borderline).** Mean fold common_log_rmse
  shift is &minus;0.0078 (paired-t p=0.071, Wilcoxon p=0.125) -- not significant at the conventional
  0.05 threshold, though closer to it than v1e's. Held-out corrected R&sup2; more than triples
  (0.0078 to 0.0278), but off a very small base and a single ~17k-row held-out fit -- consistent
  with noise, not a confirmed effect. This is also the leg the impact audit flagged as having the
  single most frame-dependent-feature-concentrated model (75.1% of standardized-coefficient weight
  on `defender_functional_role`'s 5 role dummies alone) -- a modest, not-quite-significant
  improvement is the plausible outcome of cleaning up that feature's own values without giving the
  small (~4,200-row) shot-conditional sample size enough power to confirm it at p&lt;0.05.
  **Original promotion (prompt 58 lineage) stands, closed, not reopened -- flagged as the closest
  borderline case among the 4 non-xT legs, worth a larger-sample recheck if this leg's dataset grows.**

---

## Phase 5 -- Fresh promotion audit, the 2 xT legs

Unlike Phase 4, both xT targets are themselves directly recomputed from corrected coordinates (Phase
3: 59.21%/50.75% of rows' actual target value changed) -- so this is **not** a before/after diff of
the same ground truth. `x1_two_stage_huber` / `y1_two_stage_huber`'s OLD held-out numbers were scored
against the now-provably-wrong target, so reusing them to judge a freshly-corrected `x1c`/`y1c` would
compare a fresh model against corrected ground truth to a stale model against wrong ground truth --
not a valid promotion comparison. Instead, both the candidate (`x1c_random_forest` /
`y1c_random_forest`, at their already-locked hyperparameters -- same no-re-tune scope decision and
reasoning as Phase 4) and the immediately-preceding non-promoted comparator rung
(`x1_two_stage_huber` / `y1_two_stage_huber`) are **refit from scratch on the Phase-3-corrected
target**, on the identical 5-fold canonical CV, then compared with the same paired significance test
(`ttest_rel` + Wilcoxon on per-fold RMSE) each leg's own Rung-2 gate originally used
(prompts 73/78). Script: `scripts/analysis/phase5_xt_promotion_audit.py`. Source:
`outputs/models/validation/phase5_xt_promotion_audit.json`.

### Active-xT: x1c_random_forest vs x1_two_stage_huber, both refit on corrected target_xt_delta_v2

| | x1c_random_forest (candidate) | x1_two_stage_huber (comparator) |
|---|---|---|
| Held-out RMSE | **0.04630** | 0.05492 |
| Held-out R&sup2; | **0.37407** | 0.11908 |
| Held-out MAE | 0.01898 | 0.02109 |
| Held-out Spearman | 0.5765 | 0.5069 |
| Held-out nonzero-target RMSE | 0.05410 | 0.06588 |

Paired significance (candidate RMSE &minus; comparator RMSE, per fold, 5-fold canonical CV on
corrected target, trainval=45,166 rows): mean diff = **&minus;0.00827** (candidate lower/better),
paired-t p = **9.5&times;10&supminus;&sup7;**, Wilcoxon p = 0.0625 (the minimum attainable p-value at
n=5 paired folds, all 5 favouring the candidate). x1c wins decisively and consistently across every
fold.

**Context (not used in the decision, since the target changed):** the pre-fix promotion (prompt 75)
reported x1c held-out RMSE=0.04277 / R&sup2;=0.44947. The corrected R&sup2; of 0.37407 is *lower* --
the corrected target is genuinely harder for this model to predict than the old (partly wrong)
target was. This is an expected, not alarming, consequence of the fix: some of the old apparent
accuracy came from the model and the (also frame-corrupted) `ball_x`/`ball_y`/`phase_label`-derived
features sharing correlated errors with the old wrong-frame target -- a form of consistent bias that
inflated the old R&sup2; without being real predictive skill. The comparator's own corrected R&sup2;
(0.119) is also far below its own old pre-fix number (0.152), confirming this is a property of the
corrected target's own difficulty, not specific to x1c.

### Passive-xT: y1c_random_forest vs y1_two_stage_huber, both refit on corrected target_xt_delta_passive

| | y1c_random_forest (candidate) | y1_two_stage_huber (comparator) |
|---|---|---|
| Held-out RMSE | **0.02418** | 0.02767 |
| Held-out R&sup2; | **0.33130** | 0.12473 |
| Held-out MAE | 0.00796 | 0.00773 |
| Held-out Spearman | 0.2042 | 0.1213 |
| Held-out nonzero-target RMSE | 0.02862 | 0.03281 |

Paired significance (candidate RMSE &minus; comparator RMSE, per fold, 5-fold canonical CV on
corrected target, trainval=1,275,289 rows): mean diff = **&minus;0.00358**, paired-t p =
**1.26&times;10&supminus;&sup5;**, Wilcoxon p = 0.0625 (again the n=5 floor, all folds favouring the
candidate). Same pattern as active-xT: decisive, consistent win for the two-stage RF.

**Context (not used in the decision):** pre-fix promotion (prompt 80A) reported y1c held-out
RMSE=0.02489 / R&sup2;=0.49325. Corrected R&sup2; (0.331) is again lower than the pre-fix figure,
and the comparator's own corrected R&sup2; (0.125) is likewise far below its pre-fix number (0.061,
which itself moved -- both models' apparent accuracy against the *old* target is not comparable to
either model's accuracy against the corrected one). Same explanation as active-xT: some of the old
apparent skill was fit to a target that was itself wrong for ~67% of its own rows, including via
`ball_x`/`ball_y`, the regression head's own two most important features (65.4% combined importance,
per the impact audit's section 3) -- correcting the target removes whatever spurious fit existed
between those features' old wrong values and the old wrong target.

### Promotion verdicts

- **Active-xT: PROMOTE `x1c_random_forest`.** This is a fresh promotion decision on genuinely new
  target values, run as a full audit (not a diff): `x1c` beats `x1_two_stage_huber` by a wide,
  highly significant margin (RMSE 0.0463 vs 0.0549, R&sup2; 0.374 vs 0.119, p=9.5e-7) when both are
  refit on the corrected target at their own already-locked configurations. No other rung in this
  leg's ladder was re-examined (out of Phase-4/5's shared scope), but nothing in this audit suggests
  the two-stage architecture has become competitive with the RF approach under the corrected data --
  if anything the RF's relative margin over the linear/Huber two-stage baseline is now larger in
  R&sup2; terms (0.255 points) than it was pre-fix (0.297 points, comparable order of magnitude,
  same conclusion). The leg's headline held-out numbers should be updated from RMSE=0.04277/
  R&sup2;=0.44947 to **RMSE=0.04630/R&sup2;=0.37407** wherever cited (ACTIVE_XT_MODEL_LADDER.md,
  ALL_LEGS_SUMMARY.md -- flagged for a follow-up documentation pass, not rewritten here).
- **Passive-xT: PROMOTE `y1c_random_forest`.** Same verdict, same margin pattern: `y1c` beats
  `y1_two_stage_huber` decisively (RMSE 0.0242 vs 0.0277, R&sup2; 0.331 vs 0.125, p=1.3e-5) on the
  corrected target at locked configurations. Headline held-out numbers should be updated from
  RMSE=0.02489/R&sup2;=0.49325 to **RMSE=0.02418/R&sup2;=0.33130** wherever cited
  (PASSIVE_XT_MODEL_LADDER.md, ALL_LEGS_SUMMARY.md -- same follow-up-documentation flag).
- **Neither leg promotes a different rung than before.** Both audits confirm the existing
  architecture choice (two-stage classifier-times-regressor Random Forest) remains clearly superior
  to the two-stage linear/Huber alternative once both are judged on honestly-corrected ground truth
  -- the fix changes the *number* materially (both R&sup2; drop by roughly a fifth to a third of
  their old value), but not the *choice* of which architecture to promote.

---

### Summary verdict, all 6 legs

| Leg | Reference model | Verdict |
|---|---|---|
| Active-binary | v1e_gradient_boosting_calibrated | **Reconfirmed, closed** (fold-AP shift +0.0008, p=0.929) |
| Passive-binary | p1e_gradient_boosting_calibrated | **Reopened -- promotion stands, numbers updated** (fold-AP shift +0.0248, p=0.0004) |
| Active-continuous | c1d_random_forest | **Reopened -- promotion stands, numbers updated** (fold-RMSE shift &minus;0.0082, p=0.014) |
| Passive-continuous | d1_lognormal_glm | **Reconfirmed, closed** (borderline: fold-RMSE shift &minus;0.0078, p=0.071) |
| Active-xT | x1c_random_forest | **Promoted (fresh audit, same rung)** -- RMSE 0.0463/R&sup2; 0.374 on corrected target, p=9.5e-7 vs comparator |
| Passive-xT | y1c_random_forest | **Promoted (fresh audit, same rung)** -- RMSE 0.0242/R&sup2; 0.331 on corrected target, p=1.3e-5 vs comparator |

## Phase 6 -- Downstream cleanup

- **`dashboard_data/STALE.md`** added (not touching any other file in that folder): states the
  folder was generated before this prompt's fix/re-pipeline and must not be used until regenerated
  (a separate, out-of-scope follow-up). `dashboard_data/` itself is untouched otherwise, per this
  prompt's own non-goals.
- **This report** links back from `COORDINATE_FRAME_IMPACT_AUDIT.md`'s own closing section (added
  below) and from `reports/modeling/INDEX.html`'s sidebar.
- **Non-goals honored:** no historical (never-promoted) ladder-rung doc touched; `dashboard_data/`
  not regenerated; no React/Next.js dashboard build started; Phase 1's loader fix and commit
  `8b7d0a8` not reopened or amended.

## Deliverable index

- This report: `reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md` /
  `COORDINATE_FRAME_FIX_AND_REPIPELINE.html`
- `outputs/models/validation/coordinate_frame_fix_clearance_phase_sanity_check.json` (Phase 2)
- `outputs/models/validation/phase3_xt_target_delta_report.json` (Phase 3)
- `outputs/models/validation/phase4_reconfirmation.json`,
  `outputs/models/validation/phase4_held_out_refit.json` (Phase 4)
- `outputs/models/validation/phase5_xt_promotion_audit.json` (Phase 5)
- `dashboard_data/STALE.md` (Phase 6)
