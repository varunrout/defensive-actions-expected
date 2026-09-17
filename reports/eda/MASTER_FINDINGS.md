# Master Findings -- Defensive Action Expected

**Bottom line up front:** EDA is complete across both datasets and both target types; modelling has not started. The locked feature set (34 active / 38 passive) is confirmed internally consistent for both binary and continuous targets, with one open redundancy decision (Cluster 5), one confirmed identity-leakage risk feature (`position`, active legs only), and a set of named interaction-term candidates and target-specific caveats per model-leg in section 6 -- read that section before writing any model code.

## 1. What this is

Two datasets describe the same matches from two different vantage points: `player_defensive_actions.parquet` (**active**, 34 locked features, one row per actual defensive action -- Pressure/Duel/Clearance/Block/Interception/Ball Recovery/Foul Committed/50-50 -- by an identifiable player) and `passive_defense.parquet` (**passive**, 38 locked features, one row per anonymous defender-slot per on-ball attacking event -- off-ball positioning, no player identity by design).

They're kept separate because they answer different questions with different row semantics: active asks "what happened at the moment a defender acted", passive asks "where was every nearby defender, at every attacking touch, whether or not they did anything". Merging them would blur a one-action-per-row table into a many-slots-per-event one and lose both.

Four model-legs follow from crossing {active, passive} x {binary target `target_future_shot_10s`, continuous target `target_future_xg_10s`}: active-binary, active-continuous, passive-binary, passive-continuous.

> **current stage** -- EDA is complete. Modelling has not started. Everything in this document is descriptive/observational -- no causal claims, anywhere.

## 2. Data engineering & feature engineering, in decisions

Every decision below changed the candidate feature lists or is a named validation checkpoint in the pipeline, in the order it ran (source: `src/eda/generate_pipeline_log.py`'s STAGE_HISTORY, which asserts its own cumulative total against `feature_config.py`'s live counts on every render -- it cannot silently drift).

| Decision | Reason |
| --- | --- |
| Drop-to-one, active Cluster 1 (goal-proximity) | distance_to_attacking_goal/box, distance_to_defending_goal/box, action_zone mutually r/eta &ge; 0.94 -- kept <code>distance_to_attacking_box</code> (is_in_attacking_box lift +11.99pp is the most predictive single cut). |
| Merge (not drop), active Cluster 2 (centroids) | attacker_centroid_x/y r=0.984 with defender_centroid_x/y -- replaced by <code>defender_attacker_gap_x/y</code> (defensive compactness relative to attacking shape) since the two centroids describe different teams' shapes. |
| Drop-to-one, active Cluster 3 (possession clock) | events_elapsed_in_possession / phase_transitions_observed_so_far mutually r=0.90-0.95 with possession_elapsed_seconds -- kept <code>possession_elapsed_seconds</code> (most granular, continuous). |
| Drop-to-one, passive Cluster 4 (defender-position) | distance_to_attacking_box/box, distance_to_attacking_goal/defending_goal, defender_zone mutually r/eta &ge; 0.92 -- kept <code>defender_x</code> (the most primitive measurement, upstream of every derived distance/zone column). |
| Cluster 5 (passive option threat-score ranks) -- STILL OPEN | top_option_2_threat_score &lt;-&gt; top_option_3_threat_score, r=0.902. Deliberately left unresolved, gated behind a baseline model's feature importance (PASSIVE_COLLAPSE_OPTION_RANKS=False). See section 3. |
| Passive raw option coordinates replaced | top_option_n_target_x/y were a coordinate-frame artefact (r=0.88 with ball_x) -- replaced with ball-relative dx/dy/distance_from_ball/angle_from_ball, kept alongside during a transition period then dropped once confirmed working (6 columns removed). |
| V2 methodology gaps fixed | Categorical pairs had no REVIEW band at all, and continuous&lt;-&gt;continuous REVIEW pairs had no resolution rule. Both fixed in separate _v2 modules; V1 candidate lists untouched. |
| Functional-role bucket fix (4&rarr;5 categories) | 62.4% "unclassified" wasn't sparse visibility -- a missing advanced+wide bucket plus narrow terciles. Added <code>advanced_wide</code>, renamed the n&ge;2 fallback from unclassified to <code>mid_block</code>. unclassified now reserved exclusively for n&lt;2 (structural, not a category-count add/drop). |
| VIF drop (active, -2 features) | local_numerical_balance_5m/10m: pairwise correlation cleared this 6-column cluster (max pairwise r &lt;0.90), VIF didn't -- joint linear dependency (condition number ~1.36e15 pre-drop) invisible to pairwise correlation. Both dropped; exactly recoverable from attackers_within_Nm - defenders_within_Nm, which stay. |
| Leakage drop (passive, -1 feature) | <code>has_screened_outcome</code> is a censoring-mechanism proxy for target_future_shot_10s's own 10s window being truncated at end-of-period/match (chi2=136.5, p=1.54e-31) -- not defensive signal. Excluded. |
| Football sanity check -- substitution + named gap | Phase 7 calls for checking feature values against match video, which isn't available outside this repo's own tooling. Substitute used here: reimplement each documented formula independently from raw geometry, then compare against the real stored output on every row of the actual rebuilt parquet -- a different, arguably stricter check than eyeballing a sample of clips, since it covers every row rather than a handful someone picked to look at. This is still only a partial substitute: real video validation of specific flagged moments is a separate, still-open step. Named gap: marking_tightness and is_goal_side_of_nearest_attacker are NOT independently checkable by this script. Both need the full list of visible attackers per frame, and this parquet only exports the top-3 ranked passing options, which aren't necessarily the same set as "all visible attackers" (an attacker can be visible without being one of the top-3 ranked pass targets). Recomputing them from what's actually exported would not be an independent check -- it would just re-derive the same already-stored inputs. This is an open gap, not a silent omission. |
| Passive archetype clustering -- fit/refit resolution | KMeans swept over k in {2,3,4,5} on a fixed-seed sample (tractability -- mid_block alone has 838,270 rows), best k selected by a multi-metric score, then the FINAL model refit at that k with n_init=20 for stability. That fitted pipeline (transform+predict only, no further refitting) then labels every row in the full bucket population; sample-vs-full-population drift is checked explicitly (&ge;3pp shot-rate or &ge;2pp share triggers a flag) -- no drift found in any of the 5 clustered buckets, confirming the sample generalised. |

## 3. Feature selection, in decisions

V1/V2's correlation-resolution logic ran two review passes (categorical vs categorical, and continuous vs continuous had gaps V1 didn't cover -- see section 2) but resolved into the same candidate lists. Every resolved cluster, with the actual kept/dropped names (read from `feature_config.py`'s `REDUNDANCY_DROPPED_ACTIVE`/`REDUNDANCY_DROPPED_PASSIVE` dicts directly):

| Cluster | Dropped | Kept | Resolution type |
| --- | --- | --- | --- |
| Active Cluster 1 (goal-proximity) | distance_to_attacking_goal, distance_to_defending_goal, distance_to_defending_box, action_zone | distance_to_attacking_box | drop-to-one |
| Active Cluster 2 (centroids) | attacker_centroid_x/y, defender_centroid_x/y | defender_attacker_gap_x/y (new merge) | merge |
| Active Cluster 3 (possession clock) | events_elapsed_in_possession, phase_transitions_observed_so_far | possession_elapsed_seconds | drop-to-one |
| Active VIF cluster | local_numerical_balance_5m, local_numerical_balance_10m | attackers_within_Nm, defenders_within_Nm (both kept, unaffected) | VIF drop (joint, not pairwise) |
| Passive Cluster 4 (defender-position) | distance_to_attacking_box/box, distance_to_attacking_goal, distance_to_defending_goal, defender_zone | defender_x | drop-to-one |
| Passive zone_defensive_value pair | zone_defensive_value | distance_to_defending_goal (which is itself dropped into defender_x above -- the final surviving name is defender_x, not distance_to_defending_goal) | drop-to-one, chained |
| Passive raw option coordinates | top_option_1/2/3_target_x/y (6 columns) | top_option_1/2/3_dx/dy/distance_from_ball/angle_from_ball | superseded |
| Passive leakage drop | has_screened_outcome | (excluded, not replaced -- censoring artefact) | leakage exclusion |
| Cluster 5 -- STILL OPEN | neither dropped yet | top_option_2_threat_score, top_option_3_threat_score (r=0.902) | unresolved, gated on baseline feature importance |

> **Cluster 5 evidence attached since prompt 24/29** -- Confound tests (reports/eda/CONFOUND_ANALYSIS.json) checked whether each option's threat-score U-shape is explained by defender_x (the zone-danger confound): option 2's U-shape is <b>partially</b> explained (verdict 'partially' -- survives in some strata, flattens in others); option 3's is <b>not</b> explained (verdict 'no' -- survives in every stratum). This is asymmetric evidence that option 3 carries more independent signal than option 2, but it tests each option against a third variable (zone danger), not against each other -- it does not by itself resolve whether ranks 2 and 3 are redundant with EACH OTHER, which is what Cluster 5 is actually about. Still gated behind PASSIVE_COLLAPSE_OPTION_RANKS' baseline-model feature-importance check.

Correlation diff at post-lock confirmation: empty for both datasets -- zero new DROP/COLLAPSE tier crossings since the locked lists were finalised.

## 4. The locked split

Canonical split: 115 matches, 23 held out as TEST, 92 in TRAIN+VAL split into 5 `StratifiedGroupKFold` folds grouped by match_id. Stratified on the per-match shot rate (combined percentile rank across both legs), so no single match's rows appear in more than one split -- shared between all 4 model-legs, frozen at `outputs/models/splits/match_assignment.json`.

| Leg | Total rows | Overall shot rate | Test shot rate |
| --- | --- | --- | --- |
| Active | 56,068 | 7.791% | 7.205% |
| Passive | 1,593,181 | 5.966% | 5.51% |

> **player-level caveat (prompt 31) -- a split-validity finding, stated here** -- The match-grouped split does not group by player. 1012 distinct players in the active dataset, Gini=0.4638 (moderate concentration, not extreme), but 89.98% of test-set players also appear in train+val. Cross-referencing with per-feature player-identity association (ICC / Cramer's V against player_id): <b>position</b> is high risk (0.7714 &gt; 0.3, the same combination of high identity-association and high train/test overlap the prompt named as the actual risk case). Active-dataset only -- passive has no player_id column, so this caveat does not apply to either passive leg.

## 5. Pattern analysis, every finding, however small

This section is deliberately exhaustive, not a highlight reel -- every test run in prompts 21-31 is named individually below, whatever it found.

### Confound tests -- binary target, all 10

| Test | Verdict |
| --- | --- |
| Marking tightness reversal | no |
| Lane screening reversal | no |
| top_option_2_threat_score U-shape vs zone_defensive_value | partially |
| top_option_3_threat_score U-shape vs zone_defensive_value | no |
| Lane screening (option 2) reversal | no |
| Lane screening (option 3) reversal | no |
| defenders_within_10m vs box proximity | no |
| has_option_2 shot-rate gap vs top_option_1_threat_score | no |
| has_option_3 shot-rate gap vs top_option_1_threat_score | no |
| has_option_2 shot-rate gap vs defender_x | no |

### Confound tests -- continuous target (xG), all 5

5 tests, not 2 -- prompt 29 already extended this file with 3 has_option_2/3 tests; all 5 reported here.

| Test | Verdict |
| --- | --- |
| Marking tightness reversal | partially |
| Lane screening reversal | no |
| has_option_2 shot-rate gap vs top_option_1_threat_score | no |
| has_option_3 shot-rate gap vs top_option_1_threat_score | no |
| has_option_2 shot-rate gap vs defender_x | no |

### Tournament stability -- binary target, all 10

| Feature | Dataset | Verdict |
| --- | --- | --- |
| defenders_within_10m | active | genuine tournament-level difference |
| defenders_within_5m | active | genuine tournament-level difference |
| events_elapsed_in_possession | active | genuine tournament-level difference |
| local_numerical_balance_5m | active | arbitrary train/test noise |
| match_time_seconds | active | genuine tournament-level difference |
| attackers_within_5m | active | genuine tournament-level difference |
| top_option_1_distance_from_ball | passive | arbitrary train/test noise |
| top_option_3_dx | passive | arbitrary train/test noise |
| top_option_3_dy | passive | arbitrary train/test noise |
| angle_to_attacking_goal | passive | arbitrary train/test noise |

### Tournament stability -- continuous target (xG), all 10

NOT target-agnostic -- 3/10 features (defenders_within_5m, local_numerical_balance_5m, attackers_within_5m) get a different verdict than reports/eda/FEATURE_LOCK_CONFIRMATION.json's tournament_stability section. Reported here in full, xg-specific, not cross-referenced.

| Feature | Dataset | Verdict |
| --- | --- | --- |
| defenders_within_10m | active | genuine tournament-level difference |
| defenders_within_5m | active | arbitrary train/test noise |
| events_elapsed_in_possession | active | genuine tournament-level difference |
| local_numerical_balance_5m | active | genuine tournament-level difference |
| match_time_seconds | active | genuine tournament-level difference |
| attackers_within_5m | active | arbitrary train/test noise |
| top_option_1_distance_from_ball | passive | arbitrary train/test noise |
| top_option_3_dx | passive | arbitrary train/test noise |
| top_option_3_dy | passive | arbitrary train/test noise |
| angle_to_attacking_goal | passive | arbitrary train/test noise |

### Slicer redundancy -- all 3 named candidate pairs, plus the headline

Headline: of the 21 slicer pairs tested (15 active + 6 passive), only <b>1 of 21</b> is confirmed redundant by both Cramer's V and NMI agreeing -- <code>phase_label</code> &harr; <code>phase_label_prev_event</code> (active), the constructed 1-event-lag pair. Genuinely target-agnostic (never references a target column).

| Named candidate pair | Cramer's V | NMI | Verdict |
| --- | --- | --- | --- |
| phase_label &harr; phase_label_prev_event (active) | 0.6503 | 0.5149 | highly redundant |
| event_type &harr; play_pattern (active) | 0.0894 | 0.0156 | independent |
| phase_label &harr; defender_functional_role (passive) | 0.0343 | 0.0019 | independent |

Active: redundant clusters [['phase_label', 'phase_label_prev_event']], independent: ['event_type', 'period', 'play_pattern', 'position']. Passive: redundant clusters (none), independent: ['defender_functional_role', 'on_ball_event_type', 'period', 'phase_label'].

### Numeric x numeric interaction -- all 10 classifications, binary target

NOT target-agnostic despite the prompt's claim -- verified against the xg version's unconditional classification directly: 6/10 pairs differ. This section reports the binary-target numbers specifically; see FEATURE_LOCK_CONFIRMATION_XG.json's sibling section for the xg numbers. Counts: {'interactive': 8, 'substitutive': 1, 'additive': 1}.

| Feature A | Feature B | Dataset | Classification |
| --- | --- | --- | --- |
| defenders_within_10m | distance_to_attacking_box | active | interactive |
| visible_defender_count | attacker_spread | active | interactive |
| defenders_within_5m | defenders_within_10m | active | substitutive |
| possession_elapsed_seconds | match_time_seconds | active | interactive |
| defenders_between_ball_and_attacking_goal | attacker_defender_ratio | active | interactive |
| top_option_2_threat_score | top_option_2_distance_from_ball | passive | additive |
| lane_screening_score_option_2 | engagement_distance_to_carrier | passive | interactive |
| marking_tightness | engagement_distance_to_carrier | passive | interactive |
| overload_score | attacking_goal_centrality | passive | interactive |
| lane_screening_score_option_1 | marking_tightness | passive | interactive |

### Numeric x numeric interaction -- all 10 classifications, continuous target (xG)

NOT target-agnostic -- 6/10 pairs' unconditional classification differs from reports/eda/FEATURE_LOCK_CONFIRMATION.json's numeric_interaction section. Reported here in full, xg-specific, not cross-referenced. Unconditional counts: {'interactive': 6, 'additive': 1, 'inconclusive': 3}. Given-shot counts: {'inconclusive': 10} (all 10 inconclusive -- the shot-only sample is too small at this threshold to resolve additive/interactive/substitutive on chance quality). Agreement between unconditional and given-shot: 3/10.

| Feature A | Feature B | Dataset | Unconditional |
| --- | --- | --- | --- |
| defenders_within_10m | distance_to_attacking_box | active | interactive |
| visible_defender_count | attacker_spread | active | interactive |
| defenders_within_5m | defenders_within_10m | active | interactive |
| possession_elapsed_seconds | match_time_seconds | active | additive |
| defenders_between_ball_and_attacking_goal | attacker_defender_ratio | active | inconclusive |
| top_option_2_threat_score | top_option_2_distance_from_ball | passive | interactive |
| lane_screening_score_option_2 | engagement_distance_to_carrier | passive | interactive |
| marking_tightness | engagement_distance_to_carrier | passive | inconclusive |
| overload_score | attacking_goal_centrality | passive | inconclusive |
| lane_screening_score_option_1 | marking_tightness | passive | interactive |

### Slice stratification -- unconditional vs shot-conditional (xG)

V1 categorical-slicer grid: 118 cells, 320 genuine divergences (binary). On xG: 363 unconditional &rarr; 90 given-shot, 130 conditioning disagreements -- most unconditional xG divergences are about whether a shot happens, not its quality.

### Archetype vs boolean slicers -- the comparison prompt 28 set out to make

defender_archetype_name alone produced divergence at 1.46/cell (binary) and 2.00/cell (xG) vs the 15 boolean slicers combined at 1.22/cell (binary) and 1.12/cell (xG) -- the single archetype slicer matches or exceeds the combined per-cell divergence rate of all 15 boolean flags together, on both targets.

### Player-level validity -- the one risk pair

| Feature | ICC/Cramer's V | Risk level |
| --- | --- | --- |
| position | 0.7714 | high |

## 6. What each model should learn

### Active-binary (target_future_shot_10s)

- Clean signal, use directly: <code>defender_spread</code> (rho=-0.188, strongest active correlate, U-shaped), <code>attacker_spread</code>, <code>attacking_goal_centrality</code>, <code>attacker_defender_ratio</code> -- established monotonic/U-shaped patterns from prompt 21, none flagged unreliable.
- Interaction terms to build, named pairs (all confirmed 'interactive' on this target specifically, prompt 30): <code>defenders_within_10m &times; distance_to_attacking_box</code>, <code>visible_defender_count &times; attacker_spread</code>, <code>possession_elapsed_seconds &times; match_time_seconds</code>, <code>defenders_between_ball_and_attacking_goal &times; attacker_defender_ratio</code>. Also: <code>defenders_within_10m &times; phase_label</code> diverges in 7/7 phase categories -- the single strongest slice-level interaction signal in the whole corpus.
- Special handling -- redundancy: <code>defenders_within_5m &times; defenders_within_10m</code> is substitutive (nested by construction, 5m is a subset of 10m) -- consider dropping or de-weighting defenders_within_5m rather than treating both as independent.
- Special handling -- tournament-dependence: <code>defenders_within_5m</code> and <code>attackers_within_5m</code> show a genuine tournament-level difference (WC2022 vs Euro2024) on this target -- validate with a tournament-aware check alongside the match-grouped CV, not just in-sample.
- Special handling -- identity leakage: <code>position</code> is high-risk (Cramer's V=0.77 vs player_id, 90% train/test player overlap) -- consider a player-grouped CV fold in addition to the match-grouped one, or drop/de-weight position, before trusting its apparent predictive strength.
- Open decisions bearing on this leg: video validation of defender_functional_role and related geometry is still a named gap (the football sanity check reimplemented from raw geometry, not from actual match video). Cluster 5 does not apply (passive-only).

### Active-continuous (target_future_xg_10s)

- Clean signal: near-identical ranking to active-binary (defender_spread rho=-0.188, attacking_goal_centrality, distance_to_center_line, attacker_spread, attacker_defender_ratio) -- the numerical-vs-target ordering is stable across targets even though individual interaction findings are not (see below).
- Structural-zero caveat: target_future_xg_10s == 0 exactly wherever target_future_shot_10s == 0 -- any unconditional xG finding on this leg should be treated as re-deriving occurrence unless separately confirmed on the shot-conditional (given-shot) panel.
- Interaction terms -- do NOT reuse the binary leg's list unchanged: <code>defenders_within_10m &times; distance_to_attacking_box</code> and <code>visible_defender_count &times; attacker_spread</code> are interactive here too (agree with binary), but <code>defenders_within_5m &times; defenders_within_10m</code> is interactive on xG (not substitutive like binary -- a genuine target-dependent disagreement), and <code>possession_elapsed_seconds &times; match_time_seconds</code> is additive on xG (not interactive like binary). <code>defenders_between_ball_and_attacking_goal &times; attacker_defender_ratio</code> is inconclusive on xG.
- Special handling -- given-shot interaction findings are all inconclusive (10/10) at the current threshold -- don't make an interaction-term decision for this leg's chance-quality modelling based on the given-shot panel alone; only 3/10 pairs agree between unconditional and given-shot.
- Special handling -- tournament-dependence differs from binary for <code>defenders_within_5m</code> and <code>attackers_within_5m</code> (binary: genuine difference, xG: arbitrary noise) -- don't assume the binary leg's tournament caveat carries over unchanged.
- Special handling -- identity leakage: same <code>position</code> risk pair applies (target-agnostic finding).
- has_screened_outcome's leakage effect is present but materially weaker here (1.3x, p=0.043 vs binary's ~12x at p=1.5e-31) -- the drop still holds on structural grounds, just less strongly evidenced on this leg specifically.

### Passive-binary (target_future_shot_10s)

- Clean signal: <code>top_option_3_threat_score</code> (rho=-0.085, strongest passive correlate, U-shaped), and both established reversals -- marking_tightness and lane_screening_score_option_1 -- survive every confound test run against them ('no' verdict, not explained away).
- Occurrence vs quality (prompt 23/26): marking_tightness's reversal is <b>occurrence-only</b> (flat once conditioned on a shot happening) -- treat it as a shot-occurs signal for this binary leg, but see the continuous leg below for why it doesn't transfer as a quality signal. lane_screening_score_option_1's reversal is <b>occurrence+quality</b> -- genuinely useful signal either way.
- has_option_2/has_option_3 (prompt 29): both survive conditioning on top_option_1_threat_score AND on defender_x (all 'no' verdicts) -- real signal about the current freeze frame, not a proxy for possession danger or box proximity. Safe to use directly.
- Interaction terms to build, named pairs (prompt 30): <code>lane_screening_score_option_2 &times; engagement_distance_to_carrier</code>, <code>marking_tightness &times; engagement_distance_to_carrier</code>, <code>overload_score &times; attacking_goal_centrality</code>, <code>lane_screening_score_option_1 &times; marking_tightness</code>. <code>top_option_2_threat_score &times; top_option_2_distance_from_ball</code> is additive -- safe to use both independently, no interaction term needed there.
- Open decision -- Cluster 5: top_option_2_threat_score and top_option_3_threat_score (r=0.902) remain unresolved as a pair. Confound evidence is asymmetric (option 2's U-shape is 'partially' explained by defender_x, option 3's is 'no' -- not explained), suggesting option 3 carries more independent signal, but this doesn't resolve whether ranks 2/3 are redundant WITH EACH OTHER -- still gated on baseline-model feature importance.
- Modelling-stage candidate: <code>defender_archetype_name</code> matches or exceeds all 15 boolean slicers combined on a per-cell divergence basis (1.46 vs 1.22/cell) -- worth treating as a first-class categorical covariate/interaction candidate, not just a diagnostic curiosity. Its 'unclassified' category (n=1026) is structural (n&lt;2 visible defenders) -- never a 5th behavioural role.
- Open validation gap: marking_tightness and is_goal_side_of_nearest_attacker could not be independently re-derived by the football sanity check (need the full visible-attacker list; only the top-3 ranked options are exported) -- still open, specific to these two features on this leg.

### Passive-continuous (target_future_xg_10s)

- Structural-zero framing matters most on this leg: 363 unconditional genuine divergences collapse to 90 given-shot, with 130 conditioning disagreements -- treat any unconditional passive xG finding as probably about occurrence, not quality, until checked given-shot.
- marking_tightness vs defender_x: weaker than binary here -- 'partially' both unconditionally and given-shot (binary was a clean 'no'). Treat marking_tightness's confound-survival as more fragile on this leg specifically.
- lane_screening_score_option_1 vs top_option_1_threat_score: robust on both counts ('no' unconditionally and given-shot) -- safe to use for occurrence AND quality here.
- has_option_2 vs top_option_1_threat_score flips to occurrence-only on this leg specifically (unconditional 'no', given-shot 'yes') -- don't use has_option_2 as a chance-quality signal via this pathway on xG, even though it's a real, robust occurrence signal (per the binary leg above). has_option_3 stays robust both ways ('no'/'no'). has_option_2 vs defender_x is mixed ('no'/'partially').
- Interaction terms -- do not reuse the binary leg's list unchanged: <code>top_option_2_threat_score &times; top_option_2_distance_from_ball</code> flips to interactive on xG (was additive on binary) -- a genuine target-dependent difference. <code>marking_tightness &times; engagement_distance_to_carrier</code> and <code>overload_score &times; attacking_goal_centrality</code> both drop to inconclusive on xG (were interactive on binary) -- weaker evidence here, don't force an interaction-term decision from either pair alone for this leg.
- All 10 pairs' given-shot classification is inconclusive at the current threshold -- no numeric-interaction decision for chance-quality on this leg should rest on the given-shot panel alone yet.
- defender_archetype_name is even more informative relative to boolean slicers on this leg than on binary (2.00 vs 1.12 divergences/cell) -- the strongest case across all 4 legs for treating it as a first-class covariate.

## 7. Open items, prioritised

- <b>1. Cluster 5 (top_option_2_threat_score vs top_option_3_threat_score, r=0.902)</b> -- resolve before or during baseline modelling: run a baseline passive model with both ranks present, check feature importance, then decide drop-to-one vs keep-both under PASSIVE_COLLAPSE_OPTION_RANKS. Blocks a clean answer to "how many passive threat-score features actually matter" until resolved.
- <b>2. Video validation of the football-sanity gap</b> -- marking_tightness and is_goal_side_of_nearest_attacker were never independently checked against real match video (the substitute -- reimplementation from raw geometry -- doesn't cover them, since both need a full visible-attacker list this parquet doesn't export). Should happen before those two specific features are trusted at face value in modelling.
- <b>3. Player-grouped CV for the active legs</b> -- position is a confirmed identity-leakage risk (Cramer's V=0.77, 90% train/test player overlap). Add a player-grouped fold alongside the existing match-grouped one specifically to stress-test position (and, opportunistically, anything else) before reporting active-leg validation metrics as final.
- <b>4. Numeric-interaction given-shot resolution</b> -- both xG legs' given-shot interaction classifications are 10/10 inconclusive at the current flat-margin threshold. Either accept that chance-quality interaction decisions wait for more data/a looser threshold, or revisit the threshold explicitly (with the tradeoffs stated) before the continuous legs' feature engineering locks in.
- <b>5. Repo cleanup</b> -- housekeeping only, no analytical blocker: stray/duplicate artefacts noticed but not itemised here (e.g. the `Claude outputs/` folder alongside `reports/`) should be swept before this becomes a shared/public repo.

## 8. Document map

Every source document this page synthesises, so a reader wanting more depth on any one point knows exactly where to go.

| Document | What it covers |
| --- | --- |
| claude/passive-defense-data-exploration-context.md | Working notes on missingness flags -- NOT a build plan (that file doesn't exist in this session). |
| src/eda/generate_pipeline_log.py -&gt; EDA_PIPELINE_LOG.html | The actual phase/stage history this document's section 2 is built from. |
| EDA_ANALYSIS.html | Base category/flag/distribution atlases; original marking-tightness/lane-screening reversal discovery. |
| FEATURE_LOCK_CONFIRMATION.json / .html | Binary-target lock confirmation + pattern-analysis findings (prompts 14, 34). |
| FEATURE_LOCK_CONFIRMATION_XG.json / .html | Continuous-target lock confirmation + pattern-analysis findings (prompts 33, 34). |
| CORRELATION_ANALYSIS.json, REVIEW_ANALYSIS.json (+ _V2, _V3) | Full pairwise correlation tiers and REVIEW resolutions behind section 3. |
| VIF_ANALYSIS.json | Multicollinearity check behind the active VIF drop. |
| LEAKAGE_AUDIT.json (both targets) | has_screened_outcome and has_option_2/3 leakage checks. |
| FOOTBALL_SANITY_CHECK.json | Independent-reimplementation validation substitute and its named coverage gap. |
| PASSIVE_ARCHETYPES.json | Archetype clustering method, buckets, and sample/full-population drift check. |
| SPLIT_VALIDATION.json | Canonical match-grouped split counts. |
| PLAYER_LEVEL_VALIDITY_CHECK.json (both portals) | Row concentration, per-feature ICC, train/test player overlap. |
| CONFOUND_ANALYSIS.json (both targets) | All confound-test verdicts, sections 3 and 5. |
| TOURNAMENT_STABILITY_CHECK.json (both targets) | All tournament-stability verdicts, section 5. |
| SLICE_STRATIFICATION.json / _V2.json (both targets) | Categorical/archetype/boolean slice-divergence counts and worked examples. |
| SLICER_REDUNDANCY.json | All 21 slicer-pair association tests, section 5. |
| FEATURE_INTERACTION_ANALYSIS.json (both targets) | All 10 numeric-interaction classifications, sections 5 and 6. |
