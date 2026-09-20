# Master Findings -- xT-Delta Target (Active-Binary Leg Only)

> **Scope: the ACTIVE-BINARY leg only.** The passive leg is explicitly out of scope for this target and no passive-side xT-delta report exists. Nothing in this document, or in this portal, covers it.

**Bottom line up front.** `target_xt_delta` is a roughly symmetric, heavy-tailed, sign-carrying target (47.0% negative, 41.7% positive, 11.3% exactly zero, skew +0.093) -- structurally unlike either target this project has analysed before. The xg_target report suite's methodology transfers, but only after three explicit adaptations, each stated in every report it touches. Both of Prompt 64's carried-forward findings resurfaced exactly where expected: the Clearance/`action_x` artefact in the Category Atlas (Clearance ranks last of 8 event types) and again as a new confound test and a new leakage-audit part, and the symmetric/negative-capable distribution shape in the Distribution Atlas. No feature was added, dropped or re-tiered, no model artefact was touched, and the five target-independent reports were linked rather than rebuilt.

## 1. Step 0 -- confirmed generation methodology, report by report

**This section is a required deliverable, not scratch notes.** For each target-dependent report type below, the specific script that actually produced the corresponding file in `reports/analysis/xg_target/` was identified and confirmed by READING ITS LOGIC and checking it against the file on disk -- column names, thresholds, JSON key sets, chart types -- not guessed from a filename.

| Report | Confirmed xg generator(s) | Confirmation status | New xT sibling | How it was confirmed / what differs |
| --- | --- | --- | --- | --- |
| Active -- Category Atlas | `src/eda/generate_reports_xg.py` | CONFIRMED from source | `src/eda/generate_reports_xt.py` | Reads `feature_config_v1_historical.ACTIVE_V1_HISTORICAL` for the reconstructed pre-drop pool, tags each column locked/dropped with the reason from `ACTIVE['excluded']`, emits the `catbar-*` markup and the `categories` / `categories_given_shot` JSON key pair that the file on disk actually has. |
| Active -- Flag Ledger | `src/eda/generate_reports_xg.py` (same script) | CONFIRMED from source | `src/eda/generate_reports_xt.py` | Same script emits both; `boolean_xg_lift` from `compute_stats_xg.py` produces the exact `lift`/`pct_true`/`small_n` key set in `active_flag_ledger.json`, ranked by `abs(lift)`, `SMALL_N_THRESHOLD=500`. |
| Active -- Distribution Atlas (+ reconstructed variant) | `src/eda/generate_distribution_atlas_reconstructed.py` | CONFIRMED from source, with a stale-path artefact found | `src/eda/generate_distribution_atlas_xt.py` | Target-INDEPENDENT by construction (`compute_stats.continuous_distribution` / `discrete_distribution` never reference a target). Its filename branch tests `out_dir.name == "eda_xg"`, which no longer matches the current directory name `xg_target` -- so the file on disk is `active_distribution_atlas.html` while a re-run today would write `..._reconstructed.html`. Noted, NOT fixed there (that script stays untouched so the xg portal remains reproducible). |
| Active -- Numerical Target Atlas | `generate_numerical_xg_target_analysis.py` (compute) + `generate_numerical_xg_target_reports.py` (render) | CONFIRMED from source | `generate_numerical_xt_target_analysis.py` + `_reports.py` | Compute step imports `build_pool`, `classify_shape`, `N_QUANTILE_BINS=10`, `DISCRETE_CARDINALITY_THRESHOLD=15`, `RHO_THRESHOLD=0.05` and `load_canonical_split`; the render step imports `NUM_TARGET_CSS`. Both key sets match the JSON/HTML on disk exactly. |
| Review Methodology | `generate_review_analysis_xg.py` + `generate_review_report_xg.py` | CONFIRMED from source | `generate_review_analysis_xt.py` + `generate_review_report_xt.py` | Reads `shot_target/CORRELATION_ANALYSIS_V1_HISTORICAL.json`; Types 1/3/4 imported unchanged from `generate_review_analysis`; only Type 2 uses target evidence. `TYPE1_R_THRESHOLD`, `TYPE2_SUBSET_CONTAINMENT`, `NEAR_IDENTICAL_RATIO=0.5`, `DISTINCT_SIGNAL_RATIO=1.5` all carried over. |
| Leakage Audit | `generate_leakage_audit_xg.py` + `generate_leakage_report_xg.py` | CONFIRMED from source -- but PASSIVE-ONLY | `generate_leakage_audit_xt.py` + `generate_leakage_report_xt.py` | Parts B/C/D test `screened_option_was_avoided`, `has_screened_outcome`, `has_option_2/3` -- none exist in the active parquet, and NO active-side leakage audit exists anywhere in this repo for any target. Part A reused by import. Parts B'/C' apply the same method to active analogues; Part D' is new and labelled new. |
| Confound (Reversal) Testing | `generate_confound_analysis_xg.py` (2 tests) + `generate_confound_analysis_part_d_xg.py` (3 more) + `generate_confound_report_xg.py` | CONFIRMED from source -- but all 5 are PASSIVE | `generate_confound_analysis_xt.py` + `generate_confound_report_xt.py` | Exactly ONE active confound test exists project-wide: `defenders_within_10m_vs_box_proximity`, from `generate_confound_analysis_part_a.py`, in the BINARY portal only -- the xg mirror never received it (its own docstring says so). Test 1 mirrors it exactly. `_verdict()` reused byte-for-byte; `QCUT_N=4` unchanged. |
| Tournament Stability Check | `generate_tournament_stability_check_xg.py` + `generate_tournament_stability_report_xg.py` | CONFIRMED from source | `generate_tournament_stability_check_xt.py` + `_report_xt.py` | Imports `INVESTIGATE` (10 features, 6 active) and `load_match_tournament_map`; bin edges computed once on pooled data and reused across tournaments; `nearest_defender_distance` excluded for the known self-reference bug. All carried over; the 4 passive features are dropped as out of scope. |
| Slice Stratification V1 | `generate_slice_stratification_xg.py` + `generate_slice_stratification_xg_expand.py` (Part A: 104 cells, Part B: conditional retrofit) + `generate_slice_stratification_report_xg.py` | CONFIRMED from source | `generate_slice_stratification_xt.py` + `generate_slice_stratification_report_xt.py` | `FEATURE_SLICER_PLAN`, `build_plan()`, `SMALL_N_THRESHOLD=1000` and `STRUCTURAL_CAUTION_CATEGORIES` all imported unchanged. The xg portal needs three scripts for historical reasons; this portal has no such history, so the same cells are built in one pass -- the same simplification the xg V2 script itself makes. |
| Slice Stratification V2 | `generate_slice_stratification_v2_xg.py` + `generate_slice_stratification_v2_report_xg.py` | CONFIRMED from source | `generate_slice_stratification_v2_xt.py` + `generate_slice_stratification_report_xt.py` | `ACTIVE_FEATURES`, `ACTIVE_BOOLEAN_SLICERS`, `ARCHETYPE_SLICER` imported unchanged. The archetype slicer is a passive column, so the active half is the boolean-slicer half only and V2's closing archetype-vs-boolean comparison is NOT reproducible -- reported as not applicable. |
| Feature Interaction Analysis | `generate_feature_interaction_analysis_xg.py` + `generate_feature_interaction_report_xg.py` | CONFIRMED from source | `generate_feature_interaction_analysis_xt.py` + `_report_xt.py` | `PAIRS`, `Q=4` and `_bin_labeled` imported unchanged; pairs NOT re-ranked for this target, so the three targets stay comparable. `SUBSTITUTIVE_RATIO_MAX=0.3`, `ADDITIVE_MAGNITUDE_RATIO_MAX=2.0`, `MIN_CELL_N_FOR_DELTA=20` carried over -- they are ratios, so scale- and sign-free. |
| Feature Lock Confirmation + Pattern Findings | `generate_feature_lock_confirmation_xg.py` + `generate_feature_lock_pattern_findings.py` (appends the findings section) + `generate_feature_lock_report_xg.py` | CONFIRMED from source | `generate_feature_lock_confirmation_xt.py` + `generate_feature_lock_report_xt.py` | The xg portal's HTML carries BOTH the confirmation and the pattern findings in one document, so both are produced here in one file too. `correlation_diff` is read directly from the binary confirmation and labelled identical-by-construction -- the same discipline the xg script applies. |
| Master Findings (this document) | `src/eda/generate_master_findings.py` (the binary portal's; the xg portal has none of its own) | CONFIRMED from source | `src/eda/generate_master_findings_xt.py` | The xg portal's INDEX links out to the binary portal's MASTER_FINDINGS rather than having one. This portal gets its own. The binary generator's block model (`p`/`ul`/`table`/`note`/`h3` -> `render_blocks_md`/`render_blocks_html`) is imported and reused, so the .md and .html cannot drift. |

### Where a methodology could NOT be confirmed from a surviving script

Every one of the thirteen report types above traces to a surviving, readable generator script. **There is no report in this suite whose methodology had to be hand-reproduced from a prompt file because the script was lost.** That is the honest answer, and it is stated rather than padded.

Three genuine gaps were found, and none of them is a missing script -- they are all SCOPE gaps, reported in the individual reports as well as here:

- **Leakage Audit.** The xg (and binary) leakage audits are passive-dataset-only. No active-side leakage audit exists anywhere in this repo, for any target. Part A's method was reused by direct import; the xg audit's Parts B/C/D are reported as HAVING NO ACTIVE COUNTERPART, and substitute parts B'/C' apply the same method to the active dataset's own analogous columns. Part D' is new and is labelled as new inside the report itself.
- **Confound Testing.** All five tests in the xg portal are passive. Exactly one active confound test exists project-wide, in the binary portal only -- `defenders_within_10m_vs_box_proximity` -- and the xg portal never received it. Test 1 mirrors it exactly; Test 2 is new and labelled new.
- **Slice Stratification V2.** Its archetype slicer is a passive column, so V2's closing archetype-vs-boolean-slicer comparison cannot be reproduced on the active leg. Reported as not applicable rather than replaced with a different comparison that would read like the same conclusion.

### One place where a carried-forward finding does NOT surface through the mirrored methodology

Mirroring a methodology faithfully means inheriting its blind spots too. Where a report's own methodology fails to surface one of Prompt 64's findings even though a reader would expect it to, that omission is itself recorded here rather than quietly patched over. One such case was found.

**The Flag Ledger does not show the three `action_*_possession` flags at all.** Prompt 64's motivating result is that those three stop being degenerate zeros on this target, so the Flag Ledger is exactly where a reader would expect it to appear. It does not, and the reason is pool construction, not a failed finding: the ledger is built from the reconstructed pre-drop 51-era pool (`feature_config_v1_historical.ACTIVE_V1_HISTORICAL`), and all three were already out of the candidate list before that snapshot was taken -- only `action_was_under_opponent_possession` survived into it. The pool-construction logic is mirrored unchanged from `generate_reports_xg.py`, so it structurally cannot show them.

Rather than let the finding disappear on that technicality, the same `boolean_xt_lift` function the ledger uses for every pooled row was run on the three columns OFF-POOL. The numbers and the gap are both stated in the rendered Flag Ledger, kept under a separate `off_pool_possession_flags` key in its JSON so they are never mistaken for pooled rows, and the flags are also covered properly in the Leakage Audit's Part C'. The numbers are in section 4.

### One stale-path artefact found in an existing script, reported not fixed

`generate_distribution_atlas_reconstructed.py` chooses its output filename with `out_dir.name == "eda_xg"`. The directory has since been renamed to `xg_target`, so that branch no longer matches: the file on disk is `active_distribution_atlas.html`, but a re-run today would write `active_distribution_atlas_reconstructed.html`. Left untouched deliberately -- editing it would change what the xg portal regenerates, and no existing `_xg` script was modified in this work.

### Closeout document: folded in, not separate

`reports/analysis/shot_target/` carries a separate `PATTERN_ANALYSIS_CLOSEOUT.{md,html}` alongside its MASTER_FINDINGS. A separate closeout is **not** produced for this portal: that document exists to close out a multi-prompt pattern-analysis campaign spanning two targets and both datasets, whereas this portal is a single-pass, single-leg suite whose entire findings set fits in section 4 below without crowding anything out. Duplicating it would create two documents that could drift apart for no reader benefit.

> **scope** -- **This portal covers the ACTIVE-BINARY LEG ONLY.** The passive leg is explicitly out of scope and no passive-side xT-delta report exists. Where a report's xg counterpart covers both datasets, only the active half is reproduced, and that is stated in the report.

## 2. The target: what target_xt_delta is and what shape it has

`target_xt_delta` = `xt_before - xt_after`, an Expected-Threat value-delta computed in Prompt 64 for the active-binary leg, to replace the structurally flawed `target_future_shot_10s`/`target_future_xg_10s` pair on that leg. It lives in `outputs/prototypes/active_binary_xt_delta.parquet` (56,068 rows, row-aligned to the locked `player_defensive_actions.parquet` by `event_id`). Every report in this portal JOINS the two in memory and never writes either back.

| Statistic | Value | Meaning |
| --- | --- | --- |
| rows | 56,068 | the full active-binary leg |
| defined | 56,036 | 32 NaN, explained in Prompt 64 section 2 (preceding event has no location) |
| mean | -0.001316 | near zero, and NEGATIVE |
| std | 0.064986 | the scale-setting statistic this portal uses in place of the mean |
| skew | +0.093 | essentially symmetric |
| excess kurtosis | 8.85 | heavy-tailed -- most mass near zero, a few large swings |
| share exactly zero | 11.31% | the action did not cross an xT grid-cell boundary |
| share negative | 47.00% | xT ROSE across the action |
| share positive | 41.69% | xT FELL across the action (threat reduced) |

> **carried forward from Prompt 64 -- confirmed independently** -- Prompt 64 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE.md section 3.2): target_xt_delta is roughly symmetric (skew +0.093), heavy-tailed (excess kurtosis 8.85), 47.0% negative / 41.7% positive / 11.3% exactly zero. This is a fundamentally different shape from target_future_xg_10s, which is strictly non-negative, zero-inflated and heavily right-skewed. Any methodology that assumed non-negativity had to be adapted explicitly -- see this module&#x27;s Adaptation 1/2/3 notes, restated in every report. Every figure in the table above was recomputed on this run and matches Prompt 64's.

## 3. What methodology was adapted, and exactly why

The xg_target suite's methodology assumes a strictly non-negative, zero-inflated, right-skewed target throughout. Three things had to change, and one thing that might have been expected to change did not. All four are stated in every report that they touch, not just here.

### Adaptation 1 -- the scale-setting statistic

ADAPTATION 1. target_xt_delta is roughly symmetric with a near-zero, NEGATIVE mean (-0.001316), so the xg_target suite&#x27;s convention of expressing every threshold as a multiple of the overall mean target value is meaningless here (it would flip signs and collapse to nothing). Instead the xg convention is TRANSLATED, not replaced: xg&#x27;s own flat_margin (0.5 x mean xg = 0.004055 on these same rows) is re-expressed as a fraction of xg&#x27;s own standard deviation (0.048033), giving 0.084425, and that fraction is applied to target_xt_delta&#x27;s standard deviation (0.064986). Every number in this calculation is computed at runtime from the real parquet, not carried over from a writeup. Scale-free thresholds (|Spearman rho| &gt;= 0.05, small-n row counts, the substitutive/additive ratio cutoffs) are carried over completely unchanged.

| Quantity | Value on this run |
| --- | --- |
| xg flat margin on the same rows (0.5 x mean xg) | 0.004055 |
| as a fraction of xg's own std | 0.084425 |
| translated flat margin for xT delta | 0.005486 |
| translated consistency range trigger | 0.010973 |
| translated review near-identical threshold | 0.005486 |
| translated review distinct-signal threshold | 0.016459 |
| flat margin on the non-zero-delta subset | 0.005826 |

Scale-free thresholds were carried over **completely unchanged**: |Spearman rho| &ge; 0.05, small-n row counts (500 and 1000), the substitutive (0.3) and additive (2.0) ratio cutoffs, `MIN_CELL_N_FOR_DELTA=20`, `QCUT_N=4`, 10 quantile deciles, and the discrete-cardinality cutoff of 15.

### Adaptation 2 -- the conditional panel

ADAPTATION 2. The xg_target suite&#x27;s second panel conditions on target_future_shot_10s == 1 to strip the structural zero mass. target_xt_delta has no shot column behind it; its structural-zero analogue is the 11.3% of rows with an EXACTLY zero delta (the action did not cross an xT grid-cell boundary). The conditional panel throughout this suite is therefore &#x27;given a non-zero delta&#x27; (target_xt_delta != 0), never &#x27;given a shot&#x27;.

One knock-on rename: the slice-stratification reports' `occurrence_vs_quality` field encodes an xg-specific distinction that simply does not exist for this target. It is renamed `zero_mass_vs_magnitude` (values: zero-mass-driven / magnitude-driven / inconclusive) rather than reused under a name that would be actively misleading.

### Adaptation 3 -- two-directional framing

ADAPTATION 3. Sign carries the football meaning: POSITIVE target_xt_delta = xT fell across the action (threat reduced, good for the defence), NEGATIVE = xT rose. Rankings are by |magnitude| across BOTH directions, never one-sided &#x27;higher is worse&#x27; lift, and every table also reports the negative/positive/zero share behind its mean.

Concretely this changes four things: every bar chart in the portal diverges from a zero baseline instead of filling from the left (a left-anchored fill renders every negative value as zero width); every sparkline carries a dashed zero line; the interaction heatmaps use a diverging green/red ramp instead of a single-hue alpha ramp; and several reports record a new field with no xg counterpart -- whether a binned curve CROSSES zero, and whether within-stratum deltas change SIGN. Neither question can be asked of a non-negative target at all.

### Not an adaptation -- log-transforms

NOT an adaptation: checked directly rather than assumed -- the xg_target report suite never log-transforms its target anywhere (every xg report uses the raw groupby mean of target_future_xg_10s). The log1p framing in this project belongs to the modelling legs, not to this report suite, so there was no log-transform to remove.

## 4. Findings

### Prompt 64's two carried-forward findings -- where they resurfaced

> **Clearance / action_x artefact -- RESURFACED, in the Category Atlas and the Confound report** -- Prompt 64&#x27;s Clearance/action_x artefact resurfaces exactly where it was predicted to: Clearance ranks 8 of 8 event_type categories at -0.032482 mean delta over n=4,184 rows (53.1% negative). Prompt 64 measured -0.0315 inside the action_ended_possession slice; this is the same artefact on the full active dataset.

It also drives a NEW confound test in this portal (`clearance_vs_defending_box_proximity`), which asks whether Clearance's negative mean delta survives conditioning on defending-box proximity. The verdict is **partially** -- it survives in one stratum and not the other, so the artefact is neither purely a deep-box location effect nor independent of location. And Part D' of the Leakage Audit identifies the underlying mechanism: because `xt_after` is looked up from `action_x`/`action_y`, EVERY location-derived feature inherits the same definitional choice, with `distance_to_attacking_box` the strongest at r=+0.5551 against the target.

> **distribution shape -- CONFIRMED, in the Distribution Atlas** -- The Distribution Atlas adds a target-shape section the xg portal's atlas has no equivalent of (that atlas covers feature shapes only and is target-independent). Skew, kurtosis and the negative/positive/zero shares all reproduce Prompt 64's figures. The histogram there is clipped SYMMETRICALLY at the 99th percentile of |delta|; the atlas's usual min-to-p99 convention would have silently discarded this target's entire negative tail.

### The motivating problem, confirmed fixed on the full dataset (measured off-pool)

The three `action_*_possession` flags are 100% identical zeros against `target_future_shot_10s` -- which is the stated reason feature_config.py excludes them. Against this target, on the full active dataset rather than only the two slices Prompt 64 examined:

| Flag | n (True) | mean delta (True) | % exactly zero (True) | lift |
| --- | --- | --- | --- | --- |
| `action_changed_possession` | 4,656 | -0.001211 | 23.5% | +0.000114 |
| `action_ended_possession` | 4,656 | -0.001211 | 23.5% | +0.000114 |
| `action_won_possession` | 3,600 | -0.003521 | 21.0% | -0.002357 |

> **measured off-pool -- see section 1** -- These three are not rows in the Flag Ledger, because they predate the reconstructed pre-drop pool the ledger is built from. The same lift function was run on them off-pool so the finding is not lost to a pool-construction technicality. The Leakage Audit's Part C' covers them properly.

**These columns stay excluded in feature_config.py and nothing in this portal changes that.** The finding is that their stated exclusion reason is target-specific and does not transfer; acting on it would be a separate decision requiring its own leakage review.

### Numerical features vs the target

31/31 features' binned curves cross zero -- almost all of them separate threat-reducing from threat-increasing actions rather than only varying in magnitude. 4 feature(s) are flagged train/test-inconsistent.

| Feature | Spearman rho | Shape | Curve crosses zero |
| --- | --- | --- | --- |
| `distance_to_attacking_box` | 0.8397 | monotonic increasing | True |
| `distance_to_defending_box` | -0.8383 | monotonic decreasing | True |
| `distance_to_attacking_goal` | 0.8352 | monotonic increasing | True |
| `distance_to_defending_goal` | -0.8326 | monotonic decreasing | True |
| `attacker_centroid_x` | -0.753 | monotonic decreasing | True |

### Flag Ledger -- top 5 by |lift|, both directions

| Flag | lift | mean True | mean False | n True |
| --- | --- | --- | --- | --- |
| `is_in_attacking_box` | -0.110621 | -0.100327 | +0.010294 | 5,881 |
| `is_in_defending_box` | +0.087427 | +0.078558 | -0.008869 | 4,841 |
| `is_high_zone` | -0.069151 | -0.053798 | +0.015352 | 13,507 |
| `is_deep_zone` | +0.065508 | +0.050821 | -0.014687 | 11,438 |
| `has_visible_defender` | -0.016558 | -0.001338 | +0.015220 | 55,960 |

### Confound (reversal) tests

| Test | Unconditional | Given a non-zero delta | New test |
| --- | --- | --- | --- |
| defenders_within_10m vs box proximity | partially | partially | no |
| Clearance's negative mean delta vs defending-box proximity | partially | partially | yes |

### Tournament stability

3/6 active features show a genuine tournament-level difference rather than arbitrary train/test noise.

| Feature | Verdict |
| --- | --- |
| `defenders_within_10m` | genuine tournament-level difference |
| `defenders_within_5m` | genuine tournament-level difference |
| `events_elapsed_in_possession` | arbitrary train/test noise |
| `local_numerical_balance_5m` | arbitrary train/test noise |
| `match_time_seconds` | arbitrary train/test noise |
| `attackers_within_5m` | genuine tournament-level difference |

### Slice stratification

V1: 72 cells, 295 genuine divergences unconditionally and 289 once the zero mass is removed, with 20 categories where that removal flips the conclusion. V2: 108 cells, 149 / 144.

### Numeric x numeric interaction

5/5 pairs classify the same way unconditionally and on the non-zero-delta subset. 4/5 have within-stratum deltas that change SIGN -- feature A's effect reverses direction across levels of feature B. That is a distinct finding from 'interactive' and cannot arise on the xG target at all.

| Feature A | Feature B | Classification | Stratum deltas change sign |
| --- | --- | --- | --- |
| `defenders_within_10m` | `distance_to_attacking_box` | interactive | yes |
| `defenders_between_ball_and_attacking_goal` | `attacker_defender_ratio` | interactive | no |
| `defenders_within_5m` | `defenders_within_10m` | substitutive | yes |
| `visible_defender_count` | `attacker_spread` | inconclusive | yes |
| `possession_elapsed_seconds` | `match_time_seconds` | inconclusive | yes |

### Review tier

57 ACTIVE review pairs; 35 remain needs_human_call (dominated, as on every target, by continuous-continuous pairs no rule covers); 1 Type-2 pair(s) resolve on the opposite-sign rule, which carries more weight here than on xg -- a sign disagreement means the two flags disagree about the DIRECTION of the threat swing, not merely about being above or below a base rate.

## 5. Caveats recorded, and open questions this portal does not answer

Two caveats are **recorded, not resolved**. Neither is a reason to change the locked feature set, and this portal changes nothing in `feature_config.py`.

### Caveat 1 -- construction coupling (the most important one, no xG or binary counterpart)

No feature is dropped on this evidence, and none is proposed for dropping. The locked set stays as-is. The finding is interpretive: because xt_after is looked up from action_x/action_y, location-derived locked features share a definitional term with half the target, so a strong correlation in the Numerical Target Atlas is partly recovery-by-definition rather than discovered defensive signal. This has no counterpart in the xg or binary confirmations, and is the single most important caveat this portal carries.

### Caveat 2 -- this target's effects can sit in its zero mass

has_previous_event stays a locked candidate feature. The finding is about how this target must be MEASURED (its effects can sit in the zero mass rather than the mean), not about whether the feature belongs in the list. A mean-difference test alone -- the instrument the xg audit uses -- returns p=0.0645 here and would have reported nothing, while the exactly-zero share differs by +27.3pp.

> **methodological warning for anyone extending this suite** -- Caveat 2 is the one most likely to bite a future report. The instrument the xg suite reaches for first -- a difference of means -- is the right one for a strictly non-negative target whose signal lives in its centre, and the wrong one for a symmetric target with 11.3% of its mass at exactly zero. Every table in this portal therefore reports the negative / zero / positive shares alongside its mean.

### Open questions this portal does NOT answer

- Whether `action_x`/`action_y` is the right definition of `xt_after` at all. Prompt 64 flagged that a clearance's RESULTING ball location would be more football-accurate; this portal quantifies the consequences of the current definition but does not change it.
- Whether the three `action_*_possession` flags should be unlocked now that their stated exclusion reason no longer holds. That needs its own leakage review.
- What model architecture suits a roughly symmetric, heavy-tailed, sign-carrying target. Prompt 64 flagged that the existing hurdle architecture (P(shot) x E[xg|shot]) does not apply as-is. This is an EDA portal and takes no position; no model artefact was touched.
- The passive leg, entirely. Out of scope for this work.

## 6. Document map

### Built here (target-dependent -- rebuilt against target_xt_delta)

| Report | Files |
| --- | --- |
| Active -- Category Atlas | `active_category_atlas.html` / `.json` |
| Active -- Flag Ledger | `active_flag_ledger.html` / `.json` |
| Active -- Distribution Atlas | `active_distribution_atlas.html`, `active_distribution_atlas_reconstructed.html`, `active_distribution_atlas.json` |
| Active -- Numerical Target Atlas | `active_numerical_target_atlas.html` / `.json` |
| Review Methodology | `REVIEW_METHODOLOGY.html`, `REVIEW_ANALYSIS.json` |
| Leakage Audit | `LEAKAGE_AUDIT.html` / `.json` |
| Confound (Reversal) Testing | `CONFOUND_ANALYSIS.html` / `.json` |
| Tournament Stability Check | `TOURNAMENT_STABILITY_CHECK.html` / `.json` |
| Slice Stratification V1 | `SLICE_STRATIFICATION.html` / `.json` |
| Slice Stratification V2 | `SLICE_STRATIFICATION_V2.html` / `.json` |
| Feature Interaction Analysis | `FEATURE_INTERACTION_ANALYSIS.html` / `.json` |
| Feature Lock Confirmation + Pattern Findings | `FEATURE_LOCK_CONFIRMATION_XT.html` / `.json` |
| Master Findings (this document) | `MASTER_FINDINGS.md` / `.html` |
| Portal index | `INDEX.html` |

### Linked, NOT rebuilt (target-independent -- properties of the features, not of the target)

These five are properties of the 32 locked ACTIVE features and are unaffected by which target the features are measured against. The portal index links out to the existing files in `../xg_target/` rather than duplicating them.

- Correlation Atlas V1 / V2 / V3 -- correlation clustering is feature-vs-feature and never references a target column; re-running it here would produce byte-identical pairs and tiers.
- VIF Analysis -- multicollinearity is a property of the feature matrix alone.
- Slicer Redundancy Check -- slicer-vs-slicer, no target term.
- Player-Level Validity Check -- row concentration and train/test player overlap, no target term.
- Player-Grouped Split Check -- an identity-leakage stress test on the split itself.

### New generator scripts (all added as siblings; no existing script was modified)

Every script below is new. Nothing under `src/eda/` that already existed was edited, so the `xg_target` and `shot_target` portals remain exactly reproducible from their own unmodified generators.

- `src/eda/xt_common.py` -- the loader, the three adaptations, and the xT statistics functions
- `src/eda/generate_reports_xt.py`
- `src/eda/generate_distribution_atlas_xt.py`
- `src/eda/generate_numerical_xt_target_analysis.py` + `generate_numerical_xt_target_reports.py`
- `src/eda/generate_review_analysis_xt.py` + `generate_review_report_xt.py`
- `src/eda/generate_leakage_audit_xt.py` + `generate_leakage_report_xt.py`
- `src/eda/generate_confound_analysis_xt.py` + `generate_confound_report_xt.py`
- `src/eda/generate_tournament_stability_check_xt.py` + `generate_tournament_stability_report_xt.py`
- `src/eda/generate_slice_stratification_xt.py` + `generate_slice_stratification_v2_xt.py` + `generate_slice_stratification_report_xt.py`
- `src/eda/generate_feature_interaction_analysis_xt.py` + `generate_feature_interaction_report_xt.py`
- `src/eda/generate_feature_lock_confirmation_xt.py` + `generate_feature_lock_report_xt.py`
- `src/eda/generate_master_findings_xt.py` + `generate_eda_portal_xt.py`
