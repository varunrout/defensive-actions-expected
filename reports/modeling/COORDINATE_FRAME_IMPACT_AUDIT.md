# Coordinate-Frame Bug: Full Impact Audit

Read-only diagnostic audit, Prompt 84. Prompt 83 (dashboard data export) found that
`_infer_attack_sign_by_period_team` (`src/dax/data/statsbomb_loader.py`) infers each row's
attack-direction sign from the aggregate (median) ball-progression direction of its own
`(period, possession_team)` group, under an implicit assumption that raw StatsBomb coordinates
need direction-inferring at all. They do not: StatsBomb's raw event locations are already given
event-by-event in the *acting* team's own attacking frame. The loader's premise is itself the bug,
and it produces a wrong flip for **26,039 of 56,068 rows (46.4%)** of
`player_defensive_actions.parquet` (Prompt 83's own published figure, reconfirmed byte-for-byte
below).

**This document does not fix anything.** No retraining, no locked-file edits, no rewriting of any
existing report. Every number below is freshly computed by
[`scripts/analysis/coordinate_frame_impact_audit.py`](../../scripts/analysis/coordinate_frame_impact_audit.py)
(Task 1) and
[`scripts/analysis/coordinate_frame_model_exposure.py`](../../scripts/analysis/coordinate_frame_model_exposure.py)
(Task 3), writing
`outputs/models/validation/coordinate_frame_impact_audit.json` and
`outputs/models/validation/coordinate_frame_model_exposure.json` respectively, both cited inline
below. See section 6 for what this audit deliberately does not do.

## 1. Are the targets themselves affected, or only the features? (the decisive question)

**Answer: it depends on the leg, and the split is exact and traceable.** Four of the six targets
are **provably frame-invariant by construction** (0% of their own rows affected, proven by code
inspection, not sampled). The two xT-delta targets are **directly, heavily frame-dependent** --
this is a more serious category of problem than a feature-quality issue, and it affects the
*ground truth* `x1c_random_forest`/`y1c_random_forest` were trained and evaluated against, not
just what those models were shown.

### 1a. `target_future_shot_10s` / `target_future_xg_10s` -- frame-invariant, proven by derivation

Both targets are computed in `src/dax/targets/short_horizon.py`
(`add_future_shot_target`/`add_future_xg_target`), the only code that constructs them. Reading
both function bodies in full: every column referenced --
`match_id`, `period`, the possession-group column, `index`, `event_time_seconds`,
`event_type`/`type`, `attacking_team_before_action`, and `shot_statsbomb_xg` (StatsBomb's own
precomputed shot-xG value, not derived from this project's coordinates) -- is either temporal,
team-identity, or event-type. **No `x`/`y`/`location`/`ball_x`/`ball_y`/`action_x`/`action_y`
column is read anywhere in either function.** The binary target is a pure occurrence flag ("does a
`Shot`-type event happen within 10s, same possession, same team"); the continuous target sums
StatsBomb's own xG values gated by that same occurrence logic. Neither can be affected by which
frame a coordinate is stored in, because neither ever looks at a coordinate. **0 of their own
training rows are affected**, for:

- **Active-binary** and **passive-binary** (`target_future_shot_10s` directly)
- **Active-continuous** and **passive-continuous** (`target_future_xg_10s`, and the `P(shot)`
  stage each hurdle pipeline reuses from its own leg-pair's binary classifier)

(source: `src/dax/targets/short_horizon.py`, read in full; no computation needed since there is no
coordinate reference to check exposure against)

### 1b. `target_xt_delta_v2` (active-xT) -- directly frame-dependent, 69.13% of rows exposed

`target_xt_delta_v2 = xt_before - xt_after` (`scripts/analysis/build_xt_delta_v2_prototype.py`).
`xt_before` is a grid lookup (`get_xt`, `src/dax/targets/expected_threat.py`) at the **previous
event's** `ball_x`/`ball_y`; `xt_after` is `0.0` if the action ends its own possession, else the
**next event's** own `shot_statsbomb_xg` if that next event is a `Shot` (frame-invariant --
StatsBomb's own value), else a grid lookup at the **next event's** `ball_x`/`ball_y`. Both
`ball_x`/`ball_y` columns are read from `data/processed/events_with_targets.parquet` -- the
already-flipped output of the buggy loader. Neither `xt_before` nor `xt_after` re-derives or
re-checks direction; both trust the loader's frame was correct.

Computed by generalizing Prompt 83's own row-level frame-consistency check
(`scripts/dashboard/audit_frames_and_profile_matches.py`) from defensive-action rows to **every**
row of `events_with_targets.parquet`, then joining that per-event flag onto the previous-event and
(when actually used) next-event lookups each locked row's `xt_before`/`xt_after` draws from:

| | Rows | Share of 56,068 |
|---|---|---|
| `xt_before` drawn from a frame-inconsistent event | 29,532 | 52.7% |
| `xt_after` drawn from a frame-inconsistent event (only counted when a grid lookup is actually used -- i.e. not `action_ended_possession`, not a Shot) | 26,479 | 47.2% |
| **Exposed via either lookup (target value itself is wrong)** | **38,758** | **69.13%** |

(source: `outputs/models/validation/coordinate_frame_impact_audit.json`, key
`task1_target_xt_delta_v2`; methodology validated against Prompt 83's own published figure --
this script's general event-level method reproduces **26,039 of 56,068 (0.4644)** exactly when
restricted to the defensive-action population Prompt 83 itself audited, key
`sanity_check_vs_prompt83`)

**69.13% is higher than the 46.4% feature-level figure because a row's target can be exposed via
either of two independent lookups** (the previous event, the next event), each with its own
independent chance of being frame-inconsistent -- compounding, not capping, the exposure rate.

### 1c. `target_xt_delta_passive` (passive-xT) -- directly frame-dependent, 66.5-67.2% of rows exposed

`target_xt_delta_passive = xt_before - xt_after` (`scripts/analysis/build_passive_xt_delta_prototype.py`).
Unlike active, `xt_before` here is a grid lookup at the **snapshot's own** on-ball event's
`ball_x`/`ball_y` (no previous-event lookup); `xt_after` uses the identical next-event rule active
uses. Same source table, same bug exposure:

| | Count | Share |
|---|---|---|
| `xt_before` (the snapshot's own on-ball event) frame-inconsistent | 117,370 of 198,354 unique events | 59.2% |
| `xt_after` drawn from a frame-inconsistent event (grid-lookup cases only) | 112,778 of 198,354 | 56.9% |
| **Exposed via either lookup, unique events** | **131,863 of 198,354** | **66.48%** |
| **Exposed via either lookup, defender-slot rows** (the population `y1c_random_forest` actually trains/evaluates on) | **1,070,428 of 1,593,181** | **67.19%** |

(source: `coordinate_frame_impact_audit.json`, key `task1_target_xt_delta_passive`; the
unique-event share and the defender-slot-row share are close but not identical because the
~8.03-rows-per-event duplication factor is not perfectly uniform across frame-consistent vs
frame-inconsistent events)

### 1d. Summary -- this is the single most important result in this prompt

| Target | Frame-dependent? | Rows affected |
|---|---|---|
| `target_future_shot_10s` (active-binary, passive-binary) | **No** -- proven by code inspection | 0 |
| `target_future_xg_10s` (active-continuous, passive-continuous) | **No** -- proven by code inspection | 0 |
| `target_xt_delta_v2` (active-xT) | **Yes** | 38,758 / 56,068 (69.13%) |
| `target_xt_delta_passive` (passive-xT) | **Yes** | 1,070,428 / 1,593,181 defender-slot rows (67.19%) |

**The four binary/hurdle-continuous legs' problem is scoped to feature quality and
interpretation -- serious, but bounded: their own ground truth is correct, only some of what the
model was shown to predict it is wrong.** **The two xT legs have a materially worse problem: for
roughly two-thirds of their own training rows, the number the model was trained to match is itself
computed from a misframed location and is therefore sometimes wrong.** This is not a feature-shape
or interpretation issue on those two legs -- it is a target-correctness issue, and it applies to
`x1c_random_forest` and `y1c_random_forest`'s own promoted-reference-model status, not merely to
some downstream chart.

## 2. Inventory of frame-dependent locked features, per leg

Locked feature lists are shared across legs on the same side (confirmed via
`src/eda/feature_config.py`'s `ACTIVE`/`PASSIVE` dicts, imported identically by every active-side
and passive-side leg's own training/EDA code) and cited per leg's own feature-lock confirmation
doc: `reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json` (active-binary + passive-binary,
`confirmed_counts: {active: 34, passive: 38}`), `reports/analysis/xg_target/FEATURE_LOCK_CONFIRMATION_XG.json`
(active-continuous + passive-continuous, same counts), `reports/analysis/xt_target/FEATURE_LOCK_CONFIRMATION_XT.json`
(active-xT, `active: 34`) and `PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.json` (passive-xT, `passive: 38`).

**Classification principle** (applied consistently, not per-feature guesswork -- full reasoning
and worked derivations in `scripts/analysis/coordinate_frame_model_exposure.py`'s own docstring):
a feature that stores or derives from a raw coordinate, references a *fixed external* pitch point
or boundary (attacking goal at (120,40), box edge at x=102/18), or is a *signed*
direction-dependent quantity (a signed gap/dx/dy, a goal-side inequality, an absolute bearing
angle) is **frame-dependent**. A feature that is a purely *relative*, isometry-invariant quantity
between simultaneously-flipped points (a Euclidean distance/magnitude, a count within a radius, a
spread, a count-ratio) is **frame-independent even under the bug**, because the loader's flip is
applied to every spatial input in a row at once (`ball_x`/`ball_y` and every freeze-frame player
position together), and a point-reflection through the pitch centre is a distance-preserving
isometry. Two non-obvious invariances are worth stating explicitly, since they run against the
instinct that "any spatial feature must be suspect":

- **`is_wide_lane`** (`y <= 12 or y >= 68`): symmetric about `y=40`, so the rule evaluates
  identically whether `y` or `80-y` is supplied -- invariant despite being spatial.
- **`top_option_*_distance_from_ball`** (`hypot(dx, dy)`): `dx` and `dy` both flip sign together
  under the bug, and `hypot(-dx,-dy) == hypot(dx,dy)` -- the magnitude survives even though the
  individual signed components (`dx`, `dy` themselves, and `angle_from_ball`) do not.

### Active leg (34 locked features -- active-binary, active-continuous, active-xT)

| Frame-dependent (11) | Frame-independent (23) |
|---|---|
| `phase_label`, `phase_label_prev_event` -- rule-based labeller assumes the possession team attacks toward x=120 (the bug's own motivating Clearance example) | `position`, `event_type`, `play_pattern`, `period` -- categorical labels, not coordinate-derived |
| `is_in_defending_box`, `is_in_attacking_box` -- fixed zone boundaries | `counterpress`, `action_was_under_opponent_possession`, `action_retained_defensive_team_control`, `has_previous_event`, `has_visible_attacker`, `has_visible_defender` -- non-spatial flags |
| `phase_changed_since_prev_event` -- derived from `phase_label` | `match_time_seconds`, `possession_elapsed_seconds` -- temporal |
| `angle_to_attacking_goal`, `distance_to_attacking_box`, `attacking_goal_centrality` -- reference the fixed point/boundary near (120,40) | `nearest_attacker_distance`, `nearest_defender_distance`, `attacker_spread`, `defender_spread` -- Euclidean distance/dispersion, isometry-invariant |
| `defender_attacker_gap_x`, `defender_attacker_gap_y` -- signed centroid differences, flip sign under the bug | `attacker_defender_ratio`, `visible_attacker_count`, `visible_defender_count`, `attackers_within_5m`, `defenders_within_5m`, `attackers_within_10m`, `defenders_within_10m` -- counts/ratios, isometry-invariant |
| `defenders_between_ball_and_attacking_goal` -- counts defenders with `px >= ball_x`, direction-dependent | |

### Passive leg (38 locked features -- passive-binary, passive-continuous, passive-xT)

| Frame-dependent (22) | Frame-independent (16) |
|---|---|
| `phase_label` -- same rule-based labeller | `on_ball_event_type`, `period` -- categorical, not coordinate-derived |
| `defender_functional_role` -- relative x-ranking among defenders (e.g. "last_line" = deepest) assumes a fixed attack direction | `is_wide_lane` -- proven invariant above |
| `is_in_defending_box`, `is_in_attacking_box` -- fixed zone boundaries | `has_option_2`, `has_option_3` -- existence flags |
| `is_goal_side_of_nearest_attacker` -- signed x-comparison, flips under the bug | `marking_tightness`, `engagement_distance_to_carrier` -- Euclidean distance, isometry-invariant |
| `defender_x`, `defender_y`, `ball_x`, `ball_y` -- raw coordinates | `lane_screening_score_option_{1,2,3}` -- relative-distance-based, isometry-invariant |
| `angle_to_attacking_goal`, `attacking_goal_centrality` -- fixed-point reference | `top_option_{1,2,3}_distance_from_ball` -- magnitude, proven invariant above |
| `top_option_{1,2,3}_threat_score` -- includes `1/(1+distance_to_attacking_goal)`, a fixed-point distance | `overload_score`, `defender_slot_index` -- counts/structural index, not spatial values |
| `top_option_{1,2,3}_dx`, `top_option_{1,2,3}_dy` -- signed differences, flip sign | |
| `top_option_{1,2,3}_angle_from_ball` -- absolute bearing, forward/backward interpretation flips | |

(sources: `src/eda/feature_config.py` `ACTIVE`/`PASSIVE` dicts for the locked lists;
`src/dax/features/player_defense.py`'s `_goal_metrics`/`_support_features` and
`src/dax/features/passive_defense.py`'s `_marking_features`/`_rank_option_candidates`/
`_ball_relative_option_features` for the per-feature formulas the classification above is derived
from)

## 3. Quantify exposure per reference model

Computed from each of the 6 already-promoted reference models' own already-computed
importance/coefficient artifact (Gini importance for the RF legs, gain importance for the
gradient-boosted binary legs, absolute standardized coefficient for the GLM leg), aggregated back
from one-hot columns to the original locked feature (longest-categorical-prefix matching, the same
method this project's own promotion audits already use), classified per section 2's table, and
summed. **No estimation -- every share below is an exact sum over the model's own full importance
list.** (source: `outputs/models/validation/coordinate_frame_model_exposure.json`)

| Leg | Reference model | Share of importance/weight on frame-dependent features | Top frame-dependent feature(s) and individual share |
|---|---|---|---|
| Active-binary | `v1e_gradient_boosting_calibrated` | **36.6%** | `distance_to_attacking_box` (8.8%), `attacking_goal_centrality` (7.5%), `defender_attacker_gap_x` (5.6%) |
| Passive-binary | `p1e_gradient_boosting_calibrated` | **84.3%** | `defender_x` (26.1%), `ball_x` (7.8%), `top_option_3_threat_score` (7.7%) |
| Active-continuous | `c1d_random_forest` | **42.1%** | `attacking_goal_centrality` (11.8%), `distance_to_attacking_box` (9.2%), `angle_to_attacking_goal` (5.7%) |
| Passive-continuous | `d1_lognormal_glm` | **75.1%** | `defender_functional_role` one-hot levels (13.0%, 12.4%, 12.3%, 12.1%, 11.4% across its 5 role dummies) |
| Active-xT | `x1c_random_forest` -- regression head | **60.7%** | `phase_label_prev_event_box_defence` (25.9%), `distance_to_attacking_box` (9.7%), `attacking_goal_centrality` (8.8%) |
| Active-xT | `x1c_random_forest` -- classifier head | **35.3%** | `attacking_goal_centrality` (7.0%), `distance_to_attacking_box` (4.5%), `angle_to_attacking_goal` (4.4%) |
| Passive-xT | `y1c_random_forest` -- regression head | **70.1%** | `ball_x` (33.9%), `ball_y` (31.5%), `top_option_3_threat_score` (0.8%) |
| Passive-xT | `y1c_random_forest` -- classifier head | **19.1%** | `ball_y` (1.7%), `ball_x` (1.5%), `top_option_3_dx` (1.4%) |

**`y1c`'s own Rung-2 report claimed `ball_x`+`ball_y`+`on_ball_event_type_Shot` together account
for ~93% of regression-head importance -- confirmed here that `ball_x`+`ball_y` alone (the
frame-dependent two-thirds of that figure) are 33.9%+31.5%=65.4%, and `on_ball_event_type_Shot` is
NOT frame-dependent (a categorical event-type label, not a coordinate) -- so the leg's own prior
93% figure was never entirely a frame-exposure number, but the 65.4% spatial portion of it is now
confirmed exposed, not assumed.**

**The single largest surprise in this table is not the xT legs -- it is that exposure is
consistently and substantially higher on the passive side than the active side, across every
target family, not just xT**: passive-binary (84.3%) vs active-binary (36.6%); passive-continuous
(75.1%) vs active-continuous (42.1%); passive-xT regression head (70.1%) vs active-xT regression
head (60.7%). This tracks the passive leg's own locked feature set containing more raw-coordinate
and signed-geometry features relative to categorical/count features than the active leg's does
(22 of 38 vs 11 of 34 in section 2's own tally) -- not something specific to any one target family.

## 4. Inventory of affected claims already in the report set

A search across all 6 legs' own ladder/baseline/closeout docs plus the two cross-leg summaries
(`reports/modeling/**/*.md`) for language interpreting a frame-dependent feature's tactical
meaning returns **216 lines** referencing a frame-dependent feature name or a `phase_label` level
name. The table below groups the clearest, most consequential *interpretive* claims (asserting
what a feature or a monotonic/curved shape "means" or "represents" tactically, not merely
reporting a number) by document and topic -- a representative inventory for a follow-up
remediation prompt to work from, not a literal transcription of all 216 lines (which are mostly
plain data tables that happen to name a frame-dependent feature; those are lower-priority for
caveat/correction than the claims below).

| Doc | Section | Claim | Frame-dependent? |
|---|---|---|---|
| `active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md` | Coefficient-sign confirmation table (&sect;~5) | `attacking_goal_centrality`'s coefficient sign "confirms" its documented monotonic direction, "largest continuous coefficient, matches strongest-correlate ranking" | Yes |
| `active_binary/ACTIVE_BINARY_BASELINE_SUMMARY.md` | Same table | `distance_to_attacking_box`'s coefficient sign "doesn't even match" the (weak) Spearman direction, attributed to non-monotonicity | Yes |
| `active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` | &sect;~7 (interaction-sanity) | "8 of the top 10 [surviving interaction terms] are clearly football-sensible" | Mixed -- depends which terms; several likely involve `distance_to_attacking_box`/`attacking_goal_centrality`/gap features |
| `active_binary/ACTIVE_BINARY_MODELLING_CLOSEOUT.md` | &sect;~8.3 (behavioral sanity, `v1e`) | "smooth, strictly monotonic decrease across all 10 bins... not noise or a discontinuity" for a top-8 gain-importance feature | Likely yes -- top-8 gain features for `v1e` include `distance_to_attacking_box`/`attacking_goal_centrality` per section 3's own table above |
| `active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md` | &sect;~4.3/5.3 (curvature/feature-shape) | `attacking_goal_centrality`: "mostly monotonic increasing toward the top bins -- matches its..." | Yes |
| `active_continuous/ACTIVE_CONTINUOUS_MODEL_LADDER.md` | Same | `distance_to_attacking_box`: "non-monotonic (dips through the middle bins, highest at both the [extremes])... the relationship isn't monotonic" | Yes |
| `active_xt/ACTIVE_XT_MODEL_LADDER.md` | &sect;2 (Rung 1 Step-0), &sect;5.3 (feature-shape sanity) | `distance_to_attacking_box`: "clean 8-of-8/9-of-9 monotonic increase... accelerating-monotonic" (correlation atlas cross-check, r=0.157) | Yes |
| `active_xt/ACTIVE_XT_MODEL_LADDER.md` | &sect;5.3 | `angle_to_attacking_goal`: "clean monotonic decrease... matches the atlas's own direction" | Yes |
| `passive_binary/PASSIVE_BINARY_MODEL_LADDER.md` | &sect;~7.4 (behavioral sanity) | `phase_label`: "`high_press_proxy` (0.150) and `box_defence` (0.138) highest; `wide_defending_proxy` (0.024) lowest -- matches the same phase ranking seen throughout this project" | Yes |
| `passive_binary/PASSIVE_BINARY_MODEL_LADDER.md` | Same | `top_option_1_threat_score`: "mostly monotonic decreasing... broadly consistent with the documented shape" | Yes |
| `passive_xt/PASSIVE_XT_MODEL_LADDER.md` | &sect;3.6 / &sect;5.3 (feature importance, both audits) | `ball_x`/`ball_y`/`on_ball_event_type_Shot` dominance -- ~93% combined, with a construction-coupling caveat vs `xt_before`'s own construction (already partially self-flagged -- see note below) | Yes (`ball_x`/`ball_y`); already carries a related but distinct caveat |
| `passive_xt/PASSIVE_XT_MODEL_LADDER.md` | &sect;5.3 | `ball_x`: "clean monotonic increase"; `top_option_1/2/3_threat_score`: "all clean monotonic increases (deeper/more-threatening positions -> higher predicted delta)" | Yes |
| `passive_xt/PASSIVE_XT_MODEL_LADDER.md` | &sect;3.6, &sect;4.6/5.3 | Shot rows "sit deep in the attacking box (structurally high on the xT grid)" -- a tactical/geometric explanation for `on_ball_event_type_Shot`'s importance | Indirectly -- the explanation itself references box position, though the feature named is not itself frame-dependent |
| Every `phase_label` level name used as if descriptive (`high_press_proxy`, `box_defence`, `settled_low_block_proxy`, `settled_mid_block_proxy`, `wide_defending_proxy`, `transition_defence`, `counterpress_after_loss`) | All 6 legs' docs, wherever `phase_label` appears in a table or a sentence | The label names themselves are tactical assertions ("this row is high-pressing", "this row is box defence") produced by the very rule-based labeller the frame bug corrupts for 46.4%+ of rows | Yes, definitionally |

**One important existing caveat already on record, not newly found here**: `passive_xt/PASSIVE_XT_MODEL_LADDER.md`
sections 3.6 and 5.3 already flag that `ball_x`/`ball_y`'s dominance is partly a *construction-coupling*
artefact (because `xt_before = xT(ball_x, ball_y)` shares its own two input columns with the
target's own construction) -- that is a real, separate, already-honestly-reported caveat and
remains valid. It is not the same caveat as the coordinate-frame bug this audit covers (a
construction-coupling caveat says "the model may be partly re-deriving its own target's formula";
the frame-bug caveat says "the underlying `ball_x`/`ball_y` values themselves, and the target
computed from them, are sometimes wrong"). Both apply simultaneously and should both be carried
forward by a remediation prompt, not treated as redundant.

## 5. Severity verdict, per leg

- **Active-binary (`v1e_gradient_boosting_calibrated`)**: **Metrics valid, interpretation
  compromised.** Target (`target_future_shot_10s`) is proven frame-invariant (section 1a) --
  held-out PR-AUC/ROC-AUC/Brier/ECE numbers stand exactly as reported. 36.6% of the model's own
  gain importance sits on frame-dependent features (section 3), concentrated in
  `distance_to_attacking_box`, `attacking_goal_centrality`, and the signed
  `defender_attacker_gap_x/y` pair -- meaningful but a minority share, and several of this leg's
  own interpretive claims (coefficient-sign confirmation, behavioral-sanity monotonic-shape
  claims, section 4) rest specifically on those features and need a caveat or re-check, not the
  model's headline numbers.
- **Passive-binary (`p1e_gradient_boosting_calibrated`)**: **Metrics valid, interpretation
  compromised -- more severely than the active leg.** Target is proven frame-invariant (section
  1a), so held-out PR-AUC/ROC-AUC/Brier/ECE stand. But **84.3%** of this model's own gain
  importance sits on frame-dependent features -- the highest share of any of the four non-xT
  legs, dominated by `defender_x` alone (26.1%) plus `ball_x`/`ball_y`/threat-score/`phase_label`
  features. This leg's own behavioral-sanity claim (`phase_label` ranking "matches the same phase
  ranking seen throughout this project") is a claim about the exact rule-based labeller the bug
  corrupts, and should be treated as needing re-verification before being cited again, even though
  the model's own predictive accuracy numbers are untouched.
- **Active-continuous (`c1d_random_forest`)**: **Metrics valid, interpretation compromised.**
  Target (`target_future_xg_10s`, plus the reused `v1e` P(shot) stage) is proven frame-invariant
  (section 1a) -- the hurdle-pipeline RMSE/MAE/R&sup2; numbers stand. 42.1% of `c1d`'s own Gini
  importance is frame-dependent, again led by `attacking_goal_centrality`/`distance_to_attacking_box`/
  `angle_to_attacking_goal`. This leg's own feature-shape claims about those three features'
  monotonic/non-monotonic shapes (section 4) need the same caveat.
- **Passive-continuous (`d1_lognormal_glm`)**: **Metrics valid, interpretation compromised --
  most severely of the four non-xT legs.** Target is proven frame-invariant (section 1a) --
  held-out RMSE/R&sup2; stand. **75.1%** of this GLM's own standardized-coefficient weight sits on
  frame-dependent features, and unusually, the *entire* top-5 is one categorical feature's own
  one-hot levels: `defender_functional_role`'s 5 role dummies, each individually 11-13% of total
  weight, and `defender_functional_role` is itself frame-dependent (it is a relative x-ranking
  among defenders that assumes a fixed attack direction, section 2). This is the clearest case in
  the whole audit of a single frame-dependent feature dominating a model's story -- worth flagging
  specifically for a remediation prompt, even though (again) the RMSE/R&sup2; numbers themselves
  do not change.
- **Active-xT (`x1c_random_forest`)**: **Targets themselves partly wrong -- the most severe
  category found in this audit.** `target_xt_delta_v2` is directly frame-dependent for 69.13% of
  its own training rows (section 1b) -- this is not a feature or interpretation problem, it is a
  ground-truth problem: for roughly two in three rows, the number `x1c` was trained and scored
  against is itself computed from a misframed `ball_x`/`ball_y` lookup. On top of that, 60.7% of
  the regression head's own importance and 35.3% of the classifier head's own importance sit on
  frame-dependent features. The held-out RMSE 0.04277 / R&sup2; 0.44947 that promoted this model
  (Prompt 75) cannot be read as "accuracy against correct ground truth" the way the four non-xT
  legs' numbers can -- it is accuracy against a target that is itself wrong for most of the rows
  it was measured on. This leg's promotion decision and every interpretive claim building on it
  need re-examination, not just a caveat.
- **Passive-xT (`y1c_random_forest`)**: **Targets themselves partly wrong -- same severe category
  as active-xT, and with a higher feature-exposure share on top.** `target_xt_delta_passive` is
  directly frame-dependent for 67.19% of its own defender-slot training rows (section 1c) -- the
  same ground-truth problem active-xT has. The regression head's own importance is 70.1%
  frame-dependent (the highest regression-head share of any of the six models), concentrated
  almost entirely in `ball_x`+`ball_y` (65.4% combined) -- the exact two columns this leg's own
  Rung-2 report already flagged for a *different* reason (construction-coupling, section 4). The
  held-out RMSE 0.02489 / R&sup2; 0.49325 that promoted this model (Prompt 80 part A) has the same
  problem active-xT's does: it is measured against a target that is itself wrong for roughly
  two-thirds of the rows it covers.

## 6. What this prompt does not do

- **No fix.** The loader's `_infer_attack_sign_by_period_team` function is not modified. No
  corrected coordinate, feature, or target value is written anywhere.
- **No retraining.** All 6 reference models' artifacts (`.joblib`, their own importance JSONs) are
  read, never refit. All numbers in section 3 come from each model's own already-computed
  importance/coefficient list.
- **No locked-file modification.** `player_defensive_actions.parquet`, `passive_defense.parquet`,
  `events_with_targets.parquet`, and both xT prototype parquets are read-only throughout this
  audit.
- **No editing of any existing ladder/baseline/closeout doc.** Section 4 is an inventory of
  claims that need attention, not a rewrite of them -- every doc cited above is left exactly as it
  was before this prompt.
- **No re-scoring of held-out test performance under a corrected frame.** This audit establishes
  *how much* is affected and *how severely*, not what the corrected numbers would be. That is
  necessarily a follow-up remediation prompt's job, once these findings are reviewed and a fix
  strategy (re-run the loader, or a targeted re-derivation of only the affected rows/columns) is
  chosen.

## 7. Closing update (prompt 85) -- the fix, the re-pipeline, and the re-promotion decisions

The remediation this section 6 called for has been done. Prompt 85 fixed the loader (Phase 1,
commit `8b7d0a8`: `_infer_attack_sign_by_period_team`'s noisy per-`(period, possession_team)`
inference replaced by the row-local `_attack_sign_for_row(team, possession_team)` rule), then
re-ran the full pipeline, re-computed both xT targets, and re-validated all 6 legs against the
corrected data. Full detail, every number cited to a script/JSON artifact, in
[`COORDINATE_FRAME_FIX_AND_REPIPELINE.md`](COORDINATE_FRAME_FIX_AND_REPIPELINE.md).

Headline outcome, matching this audit's own section 5 severity split almost exactly:

- **Active-binary (`v1e_gradient_boosting_calibrated`) -- reconfirmed, closed.** Fold-AP shift
  +0.0008 (p=0.929) -- indistinguishable from zero, as this audit's "metrics valid" verdict
  predicted for a frame-invariant target.
- **Passive-binary (`p1e_gradient_boosting_calibrated`) -- reopened, promotion stands, numbers
  updated.** Fold-AP shift +0.0248 (p=0.0004) -- the leg this audit flagged as having the highest
  frame-dependent importance share (84.3%) saw the largest, most significant improvement of the 4
  non-xT legs once that share's own feature values were corrected.
- **Active-continuous (`c1d_random_forest`) -- reopened, promotion stands, numbers updated.**
  Fold-RMSE shift &minus;0.0082 (p=0.014), a real but smaller improvement than passive-binary's.
- **Passive-continuous (`d1_lognormal_glm`) -- reconfirmed, closed (borderline).** Fold-RMSE shift
  &minus;0.0078 (p=0.071) -- not significant at p&lt;0.05, the closest borderline case of the 4,
  consistent with this leg's own small (~4,200-row) shot-conditional sample limiting statistical
  power even though it had the single most frame-dependent-feature-concentrated model (75.1%
  weight on `defender_functional_role` alone).
- **Active-xT (`x1c_random_forest`) -- newly promoted on corrected target_xt_delta_v2.** This
  audit's most severe finding -- 69.13% of training rows' own ground truth was itself wrong -- was
  real: `x1c`'s corrected held-out R&sup2; (0.374) is materially lower than its pre-fix number
  (0.449), because some of the old apparent accuracy was fit to a target computed from the same
  misframed coordinates as the model's own features. `x1c` still beats the two-stage Huber
  comparator decisively when both are refit on the corrected target (p=9.5&times;10&supminus;&sup7;)
  -- the architecture choice survives even though the absolute number does not.
- **Passive-xT (`y1c_random_forest`) -- newly promoted on corrected target_xt_delta_passive.** Same
  pattern: corrected R&sup2; (0.331) below the pre-fix number (0.493), `y1c` still decisively beats
  its own two-stage comparator (p=1.3&times;10&supminus;&sup5;).

No leg promoted a different rung than it had before the fix. `dashboard_data/` (prompt 83, all 6
legs' pre-fix predictions) is marked stale (`dashboard_data/STALE.md`) pending a separate
regeneration prompt.
