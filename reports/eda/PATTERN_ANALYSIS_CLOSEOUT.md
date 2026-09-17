# Pattern Analysis Closeout — Defensive Actions Expected

*Scope: prompts 21–31 only (the pattern-analysis stage). Feature-selection-stage findings (correlation clustering, VIF, leakage audit) live in `EDA_ANALYSIS.md`, not here. Numbers below are read directly from the generated JSON reports listed in each section, re-verified at the time this document was written (2026-09-17), not carried over from any earlier draft or chat.*

## 1. Overview / bottom line up front

Every locked numerical feature diverges in shape or magnitude on at least one categorical slicer — 0 features are fully stable across either dataset (26/26 active+passive features tested in V1 have at least one diverging slicer; V2's archetype+boolean-flag pass adds another 25 features tested, again 0 fully stable). The two original "reversal" findings — `marking_tightness` and `lane_screening_score_option_1` both rising in shot rate as their raw values suggest *less* danger — survive rigorous confound-testing in every stratum and are genuine football signal, not artefacts of a confound. `defenders_within_10m` and `defenders_within_5m` carry the largest raw signal on the active-binary leg (46.8pp and 44.1pp range) but this is confirmed to be a genuine tournament-level difference (WC2022 vs Euro2024), not a stable, context-free pattern — it needs tournament-aware handling, not face-value use. One feature, `position` (active dataset only), is a confirmed player-identity-leakage risk (Cramér's V=0.7714 against `player_id`, with 89.98% of test-set players also appearing in train+val).

## 2. Numerical features vs target (prompts 21–22)

Top 6 locked features by |Spearman ρ|, per dataset per target type, with shape and train/test consistency.

**Active, `target_future_shot_10s`** (source: `reports/eda/active_numerical_target_atlas.json`)

| Feature | ρ | Shape | Train/val vs test consistent |
| --- | --- | --- | --- |
| `defender_spread` | -0.1881 | U-shaped | yes |
| `attacker_spread` | -0.1703 | monotonic decreasing | yes |
| `attacking_goal_centrality` | 0.1701 | monotonic increasing | yes |
| `attacker_defender_ratio` | -0.1562 | monotonic decreasing | yes |
| `defenders_within_10m` | 0.1516 | monotonic increasing (overall) | **no** — U-shaped train/val, inverse-U test |
| `visible_attacker_count` | -0.0877 | inverse-U | yes |

**Passive, `target_future_shot_10s`** (source: `reports/eda/passive_numerical_target_atlas.json`)

| Feature | ρ | Shape | Train/val vs test consistent |
| --- | --- | --- | --- |
| `top_option_3_threat_score` | -0.0853 | U-shaped | yes |
| `top_option_2_threat_score` | -0.0766 | U-shaped | yes |
| `top_option_1_threat_score` | -0.0671 | U-shaped | yes |
| `lane_screening_score_option_3` | 0.0656 | monotonic increasing | yes |
| `attacking_goal_centrality` | 0.0648 | monotonic increasing | yes |
| `top_option_3_distance_from_ball` | -0.0634 | monotonic decreasing | yes |

**Active, `target_future_xg_10s`** (source: `reports/eda_xg/active_numerical_target_atlas.json`)

| Feature | ρ | Shape | Train/val vs test consistent |
| --- | --- | --- | --- |
| `defender_spread` | -0.1883 | U-shaped (overall) | **no** — monotonic decreasing train/val, U-shaped test |
| `attacking_goal_centrality` | 0.1712 | monotonic increasing | yes |
| `attacker_spread` | -0.1711 | monotonic decreasing | yes |
| `attacker_defender_ratio` | -0.1568 | monotonic decreasing | yes |
| `defenders_within_10m` | 0.1523 | inverse-U (overall) | **no** — U-shaped train/val, inverse-U test |
| `visible_attacker_count` | -0.0895 | monotonic decreasing (overall) | **no** — inverse-U on test |

**Passive, `target_future_xg_10s`** (source: `reports/eda_xg/passive_numerical_target_atlas.json`)

| Feature | ρ | Shape | Train/val vs test consistent |
| --- | --- | --- | --- |
| `top_option_3_threat_score` | -0.0852 | no-clear-pattern | yes |
| `top_option_2_threat_score` | -0.0765 | no-clear-pattern | yes |
| `top_option_1_threat_score` | -0.0669 | U-shaped | yes |
| `lane_screening_score_option_3` | 0.0659 | monotonic increasing | yes |
| `attacking_goal_centrality` | 0.0649 | monotonic increasing | yes |
| `lane_screening_score_option_2` | 0.0636 | monotonic increasing | yes |

Notable cross-target divergence: `defenders_within_10m` and `defender_spread` both show train/val-vs-test shape *inconsistency* on both targets — this is not new information (it's the same tournament-composition effect examined properly in section 5), but it shows up as a flat consistency-check failure here before that deeper investigation exists.

## 3. Structural zero & shot-conditional xG (prompt 23)

Confirmed exactly (source: `reports/eda_xg/STRUCTURAL_ZERO_CHECK.json`): `target_future_xg_10s == 0` in every single row where `target_future_shot_10s == 0`, on both datasets — active (56,068 rows, 0 mismatches) and passive (1,593,181 rows, 0 mismatches). This means any *unconditional* xG finding mostly re-derives the binary occurrence signal; the genuinely new information lives in the shot-conditional (given-shot) panel.

Worked example — occurrence vs quality, passive dataset (source: `reports/eda_xg/passive_numerical_target_atlas.json`):

- `marking_tightness`: unconditional ρ = -0.0358 (monotonic decreasing). Shot-conditional ρ = **0.0112** — essentially flat, mean xG barely moves across bins (0.1026 → 0.0994 → 0.1004 → ... → 0.0991). Classification: **occurrence-only** — it predicts whether a shot happens, but once a shot does happen, marking tightness tells you nothing about how good the chance is.
- `lane_screening_score_option_1`: unconditional ρ = 0.0595 (monotonic increasing). Shot-conditional ρ = **0.0646** — the monotonic climb persists, mean xG rising cleanly from 0.0917 (least-screened bin) to 0.1129 (most-screened bin). Classification: **occurrence+quality** — it predicts both whether a shot happens and, given that it does, how dangerous it is.

## 4. Confound testing (prompt 24 part A, prompt 29)

All 10 tests in `reports/eda/CONFOUND_ANALYSIS.json`:

| Test | Verdict |
| --- | --- |
| `marking_tightness` vs `zone_defensive_value` | no (survives 4/4 strata) |
| `lane_screening_score_option_1` vs `top_option_1_threat_score` | no (survives 4/4) |
| `top_option_2_threat_score` vs `zone_defensive_value` | partially (survives 3/4) |
| `top_option_3_threat_score` vs `zone_defensive_value` | no (survives 4/4) |
| `lane_screening_score_option_2` vs `top_option_2_threat_score` | no (survives 4/4) |
| `lane_screening_score_option_3` vs `top_option_3_threat_score` | no (survives 4/4) |
| `defenders_within_10m` vs `box_proximity` | no (survives 2/2) — **flagged unreliable**, see below |
| `has_option_2` vs `top_option_1_threat_score` | no (survives all strata) |
| `has_option_3` vs `top_option_1_threat_score` | no (survives all strata) |
| `has_option_2` vs `defender_x` | no (survives all strata) |

Worked example — marking-tightness reversal (source: `reports/eda/CONFOUND_ANALYSIS.json`, test `marking_tightness_vs_zone_defensive_value`): marginal shot rate runs **Q1 (tightest marking) 7.582% → Q2 6.136% → Q3 5.280% → Q4 (loosest marking) 4.866%** — tighter marking is associated with a *higher* shot rate, the reverse of naive intuition. Stratified by `zone_defensive_value` quartile, the Q1-to-Q4 delta stays negative in every one of the 4 strata (deltas: -2.888pp, -0.052pp, -0.096pp, -3.430pp) — the reversal is not an artefact of zone danger.

Worked example — lane-screening reversal (test `lane_screening_vs_top_option_1_threat_score`): marginal shot rate runs Q1 (least screened) 4.076% → Q2 5.713% → Q3 6.702% → Q4 (most screened) 7.373% — more screening associates with *more* shots, again counter to naive intuition. Survives conditioning on `top_option_1_threat_score` in every stratum tested.

Worked example — `has_option_2`/`has_option_3` (test `has_option_2_vs_top_option_1_threat_score`): marginal rate is 15.131% when no second option exists (n=2,181) vs 5.953% when one does (n=1,591,000) — a large gap that survives stratification on `top_option_1_threat_score` in every tier (e.g. T1 stratum delta -3.459pp, all strata negative). `has_option_3` shows the same pattern, more strongly (marginal 17.091% vs 5.886%, n=11,398 vs 1,581,783; T1 stratum delta -7.996pp, T2 -13.26pp).

Reliability caveat on `defenders_within_10m_vs_box_proximity`: this test's own JSON carries an explicit `reliability_note` stating it is **unreliable, not a clean finding** — Part B (section 5 below) found `defenders_within_10m` is a genuine tournament-level difference (U-shaped in WC2022, inverse-U in Euro2024, same bin edges). This confound test runs on the pooled 115-match population, mixing two populations with opposite-direction patterns. Its marginal/stratified numbers describe the pooled data, not a single stable phenomenon.

## 5. Tournament stability (prompt 24 part B)

All 10 investigated features (source: `reports/eda/TOURNAMENT_STABILITY_CHECK.json`):

| Feature | Verdict |
| --- | --- |
| `defenders_within_10m` | genuine tournament-level difference |
| `defenders_within_5m` | genuine tournament-level difference |
| `events_elapsed_in_possession` | genuine tournament-level difference |
| `match_time_seconds` | genuine tournament-level difference |
| `attackers_within_5m` | genuine tournament-level difference |
| `local_numerical_balance_5m` | arbitrary train/test noise |
| `top_option_1_distance_from_ball` | arbitrary train/test noise |
| `top_option_3_dx` | arbitrary train/test noise |
| `top_option_3_dy` | arbitrary train/test noise |
| `angle_to_attacking_goal` | arbitrary train/test noise |

Worked example — `defenders_within_10m`: WC2022 (31,041 rows, 64 matches) shows a **U-shaped** curve, range 46.974pp (bin 0 shot rate 3.026% down through intermediate bins, then back up to 45.455% at bin 10, based on very small n at the tail). Euro2024 (25,027 rows, 51 matches) shows an **inverse-U** curve, range 21.001pp. Same feature, same bin edges, opposite shape across the two tournaments in this dataset — a genuine tournament-level difference, not train/test noise. This is the direct link to the reliability caveat on the `defenders_within_10m_vs_box_proximity` confound test in section 4: that test's pooled population mixes these two opposite-shaped populations, which is why its result is flagged unreliable rather than read at face value.

## 6. Slice stratification (prompts 24–25/26, 28)

Aggregate counts:

| | V1 (binary) | V1 (xg) | V2 (binary, archetype+boolean) |
| --- | --- | --- | --- |
| Cells tested | 118 | 118 | 199 |
| Features with ≥1 diverging slicer | 26/26 | 26/26 | 25/26 |
| Features fully stable | 0 | 0 | 0 |
| Genuine divergences (unconditional) | 320 | 363 | 245 |
| Genuine divergences (shot-conditional) | n/a | 90 | n/a |
| Conditioning disagreements | n/a | 130 | n/a |

Worked example — stable pattern, `defender_spread` × `position_group` (source: `reports/eda/SLICE_STRATIFICATION.json`): overall shape U-shaped, range 17.089pp. Every position group holds the same U-shape and does not diverge: `centre_back` (n=10,588, range 18.696pp), `defensive_midfielder` (n=10,837, range 20.786pp), `forward` (n=7,073, range 16.672pp), `fullback_wingback` (n=10,368, range 18.709pp), `midfielder` (n=10,218, range 16.089pp), `other` (n=6,110, range 18.752pp) — all `diverges_from_overall: false`. The one exception is `goalkeeper` (n=874, small_n flagged), which shows `no-clear-pattern`, range only 2.128pp, `diverges_from_overall: true` — expected given goalkeepers occupy a structurally different role and a much smaller sample.

Worked example — diverging pattern, `defender_spread` × `phase_label`: overall shape is the same U-shape (range 17.089pp), but 4 of 7 phase categories diverge. `box_defence` (n=9,144, range 19.232pp) and `settled_low_block_proxy` (n=7,348, range 10.473pp) both flip to **monotonic decreasing**. `settled_mid_block_proxy` (n=9,868, range only 3.331pp) also flips to monotonic decreasing, but with a much flatter range. `wide_defending_proxy` (n=5,093, range 2.062pp) flips the *other* direction, to **monotonic increasing** — a genuine direction reversal versus the overall U-shape, not just a flattening. `counterpress_after_loss`, `high_press_proxy` and `transition_defence` all retain the overall U-shape and do not diverge.

## 7. Slicer redundancy (prompt 27)

21 pairs tested (15 active, 6 passive; source: `reports/eda/SLICER_REDUNDANCY.json`). Only one pair is genuinely redundant on both metrics: `phase_label` × `phase_label_prev_event` — Cramér's V = 0.6503, NMI = 0.5149, both verdicts "highly redundant" (expected: a 1-event lag of the same signal). Two other pairs were suspected going in and cleared: `event_type` × `play_pattern` (V=0.0894, NMI=0.0156, independent) and `phase_label` × `defender_functional_role` (V=0.0343, NMI=0.0019, independent).

## 8. Numeric×numeric interaction (prompt 30)

All 10 pairs (source: `reports/eda/FEATURE_INTERACTION_ANALYSIS.json`):

| Pair | Classification | Evidence |
| --- | --- | --- |
| `defenders_within_10m` × `distance_to_attacking_box` | interactive | Sign flips across B-strata |
| `visible_defender_count` × `attacker_spread` | interactive | Magnitude ratio 4.39, exceeds 2.0 threshold |
| `defenders_within_5m` × `defenders_within_10m` | substitutive | Mean within-B-stratum delta (2.087pp) is 0.13× the pooled marginal delta (+15.899pp), below the 0.3 threshold |
| `possession_elapsed_seconds` × `match_time_seconds` | interactive | Magnitude ratio 2.89 |
| `defenders_between_ball_and_attacking_goal` × `attacker_defender_ratio` | interactive | Sign flips across B-strata |
| `top_option_2_threat_score` × `top_option_2_distance_from_ball` | additive | Every B-stratum delta shares the marginal delta's sign; max/min magnitude ratio 1.32, within 2.0 threshold |
| `lane_screening_score_option_2` × `engagement_distance_to_carrier` | interactive | Magnitude ratio 5.80 |
| `marking_tightness` × `engagement_distance_to_carrier` | interactive | Magnitude ratio 3.45 |
| `overload_score` × `attacking_goal_centrality` | interactive | Sign flips across B-strata |
| `lane_screening_score_option_1` × `marking_tightness` | interactive | Magnitude ratio 2.43 |

7 of 10 are interactive (real interaction-term candidates), 1 additive, 1 substitutive (`defenders_within_5m`/`_10m` carry mostly the same information — 5m is nested inside 10m by construction).

## 9. Player-level validity (prompt 31, active-only)

Scope limitation, stated directly in the source JSON: this check is **active-dataset only** — `passive_defense.parquet` has no player identity column (one row per anonymous defender-slot per on-ball event, not per identified player), so player-level ICC and train/test player overlap cannot be computed on the passive side. Not an oversight — there is no workaround that recovers player identity from an anonymous slot.

Row concentration (source: `reports/eda/PLAYER_LEVEL_VALIDITY_CHECK.json`): 56,068 active rows across 1,012 distinct players. Top 10 players account for 4.83% of rows, top 25 for 10.42%, top 50 for 18.2%. Gini coefficient 0.4638. Max rows for a single player: 344. Median rows per player: 43.

ICC/identity-flavour ranking (34 locked features, method: Cramér's V of `player_id` vs feature, bias-corrected, threshold 0.3): `position` ranks 1st at **0.7714**, flagged `player_identity_flavoured: true` — the only feature above threshold. Rank 2 is `period` at 0.2754 (below threshold, not flagged).

Train/test player overlap: 23 test matches / 92 train+val matches; 479 test players, 964 train+val players, 431 overlapping — **89.98%** of test-set players also appear in train+val. This is expected in a 115-match, 2-tournament dataset (players play multiple matches) and not inherently a problem on its own.

Risk pairs: only `position` is flagged, at `high` risk — the combination of being player-identity-flavoured (0.7714 > 0.3) **and** having 89.98% train/test player overlap is what earns the flag, not either fact alone.

## 10. Modelling-readiness synthesis

**Active-binary** (`target_future_shot_10s`): trust `defender_spread`, `attacker_spread`, `attacking_goal_centrality`, `attacker_defender_ratio` directly — clean, train/test-consistent shapes (section 2). Build interaction terms for the 4 confirmed-interactive pairs in section 8, plus the specific `defenders_within_10m` × `phase_label` slice divergence (section 6). Watch `defenders_within_5m`/`defenders_within_10m` for substitutive redundancy (section 8) and for genuine tournament-dependence (section 5) — don't treat their raw range as a stable pattern. Watch `position` for identity-leakage (section 9).

**Active-continuous** (`target_future_xg_10s`): same clean-signal ranking as binary (section 2), but the structural zero (section 3) means any unconditional finding should be checked shot-conditional before trusting it as a quality signal, not just an occurrence signal. Interaction findings partially disagree with binary (section 8's classifications are per-dataset but xG-specific reruns show some pairs flip) — don't reuse the binary leg's interaction list unchanged.

**Passive-binary** (`target_future_shot_10s`): `top_option_3_threat_score` is the strongest correlate (section 2); both reversal findings (`marking_tightness`, `lane_screening_score_option_1`) are genuine, confound-survived signal (section 4), safe to use. `has_option_2`/`_3` are also genuine and confound-survived (section 4). Build the 4 confirmed-interactive pairs from section 8. `defender_archetype_name`, while not a locked feature, is at least as informative per cell as the 15 combined boolean slicers (section 6/V2) and is worth treating as a first-class categorical covariate at the modelling stage even though it can't be a raw input feature.

**Passive-continuous** (`target_future_xg_10s`): structural zero (section 3) applies here too — 363 unconditional divergences collapse to 90 shot-conditional (section 6), so treat unconditional passive xG findings as probably about occurrence rather than quality until checked shot-conditional. `marking_tightness` = occurrence-only (flat given-shot rho 0.0112), `lane_screening_score_option_1` = occurrence+quality (given-shot rho 0.0646, mean xG still climbs) — the clearest worked distinction in the whole corpus (section 3).

## 11. Still open

- **Cluster 5 decision** (top_option_2_threat_score vs top_option_3_threat_score, r=0.902): section 4's confound evidence is asymmetric — option 2's U-shape is only partially explained by `zone_defensive_value`/`defender_x` (survives 3/4 strata), option 3's is not explained at all (survives 4/4 strata). This document originally stated the evidence without deciding it (that was the intent of prompts 21–31 in isolation). By the time this document was written, the decision had already been made using exactly this evidence: **keep both features, permanently** (`PASSIVE_COLLAPSE_OPTION_RANKS = False` in `feature_config.py`, reasoning recorded there and in `reports/eda/MASTER_FINDINGS.md` sections 2/3/6). Not an open item any more — noted here for continuity with the evidence trail above, not as a pending decision.
- **Real video validation**: Phase 7's named gap for `marking_tightness` and `is_goal_side_of_nearest_attacker` (neither independently checkable by the football-sanity-check's geometry-reimplementation approach, since both need the full visible-attacker list and this parquet only exports the top-3 ranked options) has since been closed — checked manually against real match video and confirmed correct (`reports/eda/FOOTBALL_SANITY_CHECK.json`'s `manual_video_validation` field, `MASTER_FINDINGS.md` sections 2/6). Not an open item any more.
- **Repo cleanup Part B**: scoped back in prompt 18, still never produced. This one remains genuinely open.
- **This document's own scope boundary**: it re-covers pattern-analysis findings (prompts 21–31) only. Feature-selection-stage findings (correlation clustering V1/V2, VIF, leakage audit, the original lock decisions) live in `reports/eda/EDA_ANALYSIS.md` and are not repeated here.

---
*Changelog: written 2026-09-17, reflects prompts 21–31 (pattern-analysis stage). Prompt 32.*
