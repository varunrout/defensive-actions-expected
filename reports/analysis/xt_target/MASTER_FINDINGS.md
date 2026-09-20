# Master Findings -- the xT-Delta Target Family (Active + Passive Legs)

> **Both legs are covered.** The ACTIVE-BINARY leg is built against `target_xt_delta_v2` (Prompts 64/66/67) and is unchanged by this pass; the prior pass against `target_xt_delta` (v1) is preserved unchanged at `reports/analysis/xt_target_v1_superseded/`. The PASSIVE leg is built against `target_xt_delta_passive` (Prompt 68) and is new here. Neither leg is out of scope any more.

> **Reading order.** Section 1 is the shared target-family preamble. Section 2 is the Step-0 table. Section 3 covers the active leg BY REFERENCE -- it cites and links rather than restating tables that already exist in the linked documents and in this file's own prior version. Section 4 is this pass's own passive findings, in full.

**Bottom line up front.** **This portal now covers BOTH legs of the xT-delta target family.** Its active half (`target_xt_delta_v2`, Prompts 64/66/67) is unchanged by this pass and is summarised by reference in section 3. Its new passive half (`target_xt_delta_passive`, Prompt 68) is written out in full in section 4. **Step 0 found that all twelve target-dependent report types needed a genuine passive build** -- none had a passive-side xT equivalent anywhere, and eight of them are single files carrying both legs in the `xg_target` precedent, a convention this pass could not mirror without editing frozen active-leg output. Three findings dominate the passive half. **One:** the target is the hardest distribution in the portal -- skew -2.604 and excess kurtosis 34.84 at the defender-slot grain, against Prompt 68's -1.779/31.02 at the event grain and active v2's -0.306/18.33 -- and the std-derived flat margin consequently spans **62.1% of all rows** against active's 53.4%, with 21 of 34 features classifying differently under a robust alternative. It was RECOMPUTED from this leg's own std, not transferred. **Two:** construction coupling is LIVE here, unlike on the corrected active leg -- `xt_before = xT(ball_x, ball_y)` and both columns are locked features, leaving `ball_x` at r=+0.2521 against the delta, between active's v1 (+0.5551) and v2 (+0.1431). **Three:** the target is one value per event repeated across ~8 defender rows, so ranks and effect sizes read normally but row-level p-values do not, and no verdict here is gated on one. No feature was added, dropped or re-tiered on either leg, no model artefact was touched, and the target-independent reports were linked rather than rebuilt.

## 1. The xT target family: what both legs share, and where they differ

**Expected Threat (xT)** assigns every location on the pitch a value: the probability that a possession starting there ends in a goal. This project's grid is 8x12 over 105x68m, 96 cells, with a maximum cell value of 0.2575. An **xT delta** is the change in that value across an event: `xt_before - xt_after`. A POSITIVE delta means threat FELL (good for the defence); a NEGATIVE delta means it ROSE.

### The definition both legs share

The `xt_after` term is a **possession-outcome** definition, and it is the same rule on both legs:

- `0.0` if the event ends its own possession -- the attack broke down, so all of the threat that existed a moment ago was denied;
- otherwise that event's own `shot_statsbomb_xg` if the NEXT same-possession event is a Shot -- a real chance value, which the 96-cell grid could never produce on its own;
- otherwise the next same-possession event's own grid value.

> **why the active leg needed two attempts and this leg needed one** -- Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md preamble and section 0): this leg has NO v1/v2 supersession to carry, and that is a property of the data rather than good luck. Active's v1 mistake was using the defensive action's own recorded location (action_x/action_y) as xt_after. passive_defense.parquet's ball_x/ball_y is already the ball's location at the on-ball event the snapshot is anchored to -- confirmed identical to events_with_targets.parquet's own ball_x/ball_y at the same event_id to floating-point exactness -- which is the correct 'before' state by the exact reasoning Prompt 66 established. The corrected possession-outcome xt_after was therefore implemented directly, on the first pass. Nothing in this half of the portal supersedes an earlier passive pass, because there isn't one.

### Where the two legs' targets genuinely differ

|  | `target_xt_delta_v2` (active-binary) | `target_xt_delta_passive` (passive) |
| --- | --- | --- |
| what a row is | one actual defensive action | one visible defender-slot per on-ball attacking event |
| rows | 56,068 | 1,593,181 |
| target computed at | row grain (one action, one target) | **event grain** (198,009 defined unique events), left-joined onto every defender row of that event |
| `xt_before` | the PREVIOUS event's grid cell (not a column in the active feature table) | **`xT(ball_x, ball_y)` at the snapshot itself** -- and `ball_x`/`ball_y` are both LOCKED passive features |
| mean | +0.003381 (positive: actions deny threat) | -0.002674 (negative: an arbitrary moment in a live attack has no reason to lose value) |
| skew / excess kurtosis | -0.306 / 18.33 | -2.604 / 34.84 at the row grain (-1.779 / 31.02 at the event grain) |
| % exactly zero | 20.2% | 26.4% |
| r vs `target_future_xg_10s` | -0.1082 | -0.0736 |
| has a superseded v1 pass | Yes -- preserved at `reports/analysis/xt_target_v1_superseded/` | **No.** Corrected definition from the first pass. |

**The negative passive mean is not a discrepancy to reconcile.** Active-binary rows are interventions that should, on average, deny threat. Passive rows are snapshots at an arbitrary on-ball event during a live attacking phase, where there is no inherent reason attacking possession should lose value from one touch to the next -- if anything a slightly negative mean, threat creeping up as attacks progress, is the more plausible prior. Prompt 68 made this call and this portal reproduces it rather than re-litigating it.

## 2. Step 0 -- which report types are per-leg vs shared, confirmed from real files

**This section is a required deliverable, not scratch notes.** Before anything was built, `reports/analysis/xg_target/INDEX.html` and every JSON behind it were read directly, along with the current `reports/analysis/xt_target/` listing and the target-independent reports across `reports/analysis/`. Each of the thirteen target-dependent report types Prompt 65/67 built for the active leg was classified: is it inherently per-leg, or is it a single file carrying both legs' sections in the established precedent? And does a passive equivalent already exist anywhere in this project?

> **the headline result of Step 0** -- **All twelve target-dependent report types needed a genuine passive build.** Not one of them had a passive-side xT equivalent anywhere, and not one could be satisfied by reusing the active leg's content. Eight of them are SHARED-FILE reports in the xg precedent, which would normally mean adding a passive section to the existing file -- but this portal's files of those names are Prompt 67's active-only output and are frozen by this pass's constraints, so the passive halves are separate `PASSIVE_*` files and this document says so rather than leaving a reader to wonder why the two portals are shaped differently. Of the five target-independent report types, three already have a passive half and are linked; two are ACTIVE-ONLY project-wide for a structural reason, and that is stated rather than treated as a gap.

### The thirteen target-dependent report types

| Report type | Per-leg or shared (in the `xg_target` precedent) | Passive equivalent exists already? | Action needed here |
| --- | --- | --- | --- |
| **Category Atlas** | **Per-leg** | No -- xT active only | **GENUINE PASSIVE BUILD.** `xg_target/` ships a literal `active_category_atlas.html` / `passive_category_atlas.html` pair (+ `.json`). Inherently per-leg: the two datasets share only `period` and `phase_label` among their categorical columns. Built: `passive_category_atlas.html/.json`. |
| **Flag Ledger** | **Per-leg** | No -- xT active only | **GENUINE PASSIVE BUILD.** Literal `active_flag_ledger.*` / `passive_flag_ledger.*` pair in `xg_target/`. Built: `passive_flag_ledger.html/.json`. |
| **Distribution Atlas** | **Per-leg** | No -- xT active only | **GENUINE PASSIVE BUILD, both variants.** `xg_target/` ships the `active_`/`passive_` pair; `shot_target/` additionally ships `passive_distribution_atlas_reconstructed.html`, confirming the two-variant convention DOES apply to the passive leg (checked by listing both directories, not assumed from the active xT portal). Built: `passive_distribution_atlas.html`, `passive_distribution_atlas_reconstructed.html`, `passive_distribution_atlas.json`. |
| **Numerical Target Atlas** | **Per-leg** | No -- xT active only | **GENUINE PASSIVE BUILD.** Literal pair in `xg_target/`. The pools barely overlap: passive carries the screening / lane / option columns, active the local counts and spreads. Built: `passive_numerical_target_atlas.html/.json`. |
| **Review Methodology** | Shared FILE, both legs inside | Yes, for xg -- `xg_target/REVIEW_ANALYSIS.json` has `datasets.active` AND `datasets.passive` | **GENUINE PASSIVE BUILD, separate file.** The xg one-file convention could not be mirrored: this portal's `REVIEW_ANALYSIS.json` / `REVIEW_METHODOLOGY.html` are Prompt 67's ACTIVE-ONLY output and are frozen by this pass's constraints. Built: `PASSIVE_REVIEW_METHODOLOGY.html` + `PASSIVE_REVIEW_ANALYSIS.json`. |
| **Leakage Audit** | Shared NAME, but **passive-only content** | Yes, for xg -- and it is the ACTIVE leg that has no native counterpart anywhere in this repo | **GENUINE PASSIVE BUILD -- and the one report type where the leg asymmetry runs the other way.** `xg_target/LEAKAGE_AUDIT.json` carries `"dataset": "passive"` and its Parts B/C/D test three passive-only columns; `shot_target/` is the same. Prompt 65 had to INVENT active substitutes (Parts B'/C'/D'). This pass needs none: Parts A/B/C/D mirror literally. Part E' (construction coupling through `xt_before = xT(ball_x, ball_y)`) is new. Built: `PASSIVE_LEAKAGE_AUDIT.html/.json`. |
| **Confound Testing** | Shared NAME, but **passive-only content** | Yes, for xg -- all 5 of its tests are passive-dataset tests | **GENUINE PASSIVE BUILD, zero invention.** Prompt 65 found exactly ONE active confound test in the whole repo and had to write two of its own. All five established passive tests mirror here with no substitution. Built: `PASSIVE_CONFOUND_ANALYSIS.html/.json`. |
| **Tournament Stability Check** | Shared FILE, both legs inside | Yes, for xg -- `flat_margins_by_dataset` has an `active` and a `passive` entry | **GENUINE PASSIVE BUILD, separate file** (same frozen-active-file reason as Review Methodology). `INVESTIGATE`'s 4 passive entries, imported not re-derived. Built: `PASSIVE_TOURNAMENT_STABILITY_CHECK.html/.json`. |
| **Slice Stratification V1** | Shared FILE, both legs inside | Yes, for xg -- `results.active` AND `results.passive` | **GENUINE PASSIVE BUILD, separate file.** `FEATURE_SLICER_PLAN`'s and `build_plan()`'s passive entries, imported. Built: `PASSIVE_SLICE_STRATIFICATION.html/.json`. |
| **Slice Stratification V2** | Shared FILE, both legs inside | Yes, for xg -- `results.active` AND `results.passive` | **GENUINE PASSIVE BUILD, separate file -- and the one report where this leg can do MORE than the active one.** `defender_archetype_name` is passive-only, so the closing archetype-vs-boolean comparison that the active half reports as NOT APPLICABLE is made here. Built: `PASSIVE_SLICE_STRATIFICATION_V2.html/.json`. |
| **Feature Interaction Analysis** | Shared FILE, both legs inside | Yes, for xg -- `thresholds.min_marginal_delta_by_dataset` has an entry per leg | **GENUINE PASSIVE BUILD, separate file.** `PAIRS`'s 5 passive entries, imported and deliberately not re-ranked. All five involve screening / option / engagement columns with no active counterpart. Built: `PASSIVE_FEATURE_INTERACTION_ANALYSIS.html/.json`. |
| **Feature Lock Confirmation (+ Pattern Findings)** | Shared FILE, both legs inside | Yes, for xg -- `confirmed_counts` carries `active: 34` and `passive: 38` | **GENUINE PASSIVE BUILD, separate file.** The renderer had to be written rather than relabelled: the active one's leakage section is hardcoded to Parts B'/C'/D' and its pattern section to the Clearance artefact and the possession flags, none of which exists on this leg. Built: `PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.html/.json`. |
| **Master Findings + portal index** | Shared document | n/a | **REWRITTEN for both legs** (this document) and the portal index regrouped into Active + Passive, mirroring `xg_target/INDEX.html`'s own structure. `reports/analysis/INDEX.html`'s xT card updated. |

### The target-independent reports (properties of the features, not of the target)

Seven files, five report types. None was rebuilt, on either leg, for the same reason Prompt 65 and Prompt 67 gave: **a change to which target the features are measured against cannot move a feature-vs-feature statistic**, so rebuilding them would produce byte-identical output. What Step 0 adds here is a per-type check of whether a PASSIVE half already exists -- which was not obvious from the titles, and in two cases turned out to be No.

| Report type | Per-leg or shared | Passive equivalent exists already? | Action needed here |
| --- | --- | --- | --- |
| Correlation Atlas V1 / V2 / V3 | Shared, both legs | **YES** -- `CORRELATION_ANALYSIS*.json` carries `datasets.active` and `datasets.passive` | **LINK, do not rebuild.** Correlation clustering is feature-vs-feature and never references a target, so re-running it against either xT target would produce byte-identical pairs and tiers. |
| VIF Analysis | Shared, both legs | **YES** -- `VIF_ANALYSIS.json` carries `datasets.active`/`.passive` | **LINK.** Multicollinearity is a property of the feature matrix alone. |
| Slicer Redundancy Check | Shared, both legs | **YES** -- `SLICER_REDUNDANCY.json` carries `datasets.active`/`.passive` | **LINK.** Slicer-vs-slicer, no target term. |
| Player-Level Validity Check | **ACTIVE-ONLY**, project-wide | **NO** -- `shot_target/PLAYER_LEVEL_VALIDITY_CHECK.json` carries `"dataset": "active"` plus its own `scope_limitation` field | **LINK, and state the asymmetry.** Not built here, and that is not a gap this pass could close: the check is about row concentration per `player_id` and train/test player overlap, and the passive leg is not player-indexed the same way. It is also target-independent, so it falls outside this pass's remit either way. |
| Player-Grouped Split Check | **ACTIVE-ONLY**, project-wide | **NO** -- its own `scope` field reads "Active leg only (player_id / position are active-dataset concepts; the passive leg is not player-indexed the same way)" | **LINK, and state the asymmetry.** Confirmed by reading the `scope` field, not inferred from the title -- unlike the Player-Level Validity Check, this one's name does not announce its scope. Same reasons as above. |

### Two conventions checked rather than assumed

- **The reconstructed-pool Distribution Atlas variant.** The active xT portal ships both `active_distribution_atlas.html` and `active_distribution_atlas_reconstructed.html`, but `xg_target/` ships only a single `passive_distribution_atlas.html`. That could have meant the passive leg gets one variant. It does not: `shot_target/` ships `passive_distribution_atlas_reconstructed.html` too, so the two-variant convention applies to both legs, and the missing xg file is the stale filename-branch artefact Prompt 65 already recorded for the active side. Both variants are built here.
- **The shared-name reports' file convention.** The prompt asked whether `xg_target`'s own convention for reports like the Leakage Audit is a single file covering both legs or separate files. Checked directly: it is a SINGLE file in every case. That convention is deliberately NOT mirrored here, and the reason is a constraint rather than a preference -- merging a passive section into `LEAKAGE_AUDIT.json`/`.html` and its seven siblings would mean editing active-leg report content that this pass leaves exactly as Prompt 67 rebuilt it. The divergence is recorded here, in every affected report's own scope note, and on the portal index.

## 3. The ACTIVE leg (Prompts 64/66/67) -- by reference

> **this section is deliberately short -- it cites rather than restates** -- The active leg's findings are Prompts 64, 66 and 67's, and they are already written out in full in two places: the linked source documents, and the prior version of THIS file, which git preserves. Restating their tables here would create a second copy that can drift from the first. What follows is a pointer list with the headline number for each finding, so a reader knows what is there and where to go for it. **Nothing in this section is new, re-measured, or changed by this pass**, and no active-side report file was edited.

### Where to read the active leg's findings in full

| Finding | Headline | Read it in |
| --- | --- | --- |
| The Clearance artefact REVERSED | 8th of 8 event types (-0.032482) under v1 -> **1 of 8 (+0.024706)** under v2, and it survives conditioning on both defending-box proximity and the `xt_after = 0` construction rule | `active_category_atlas.html`; `CONFOUND_ANALYSIS.html` (tests 2 and 3); `outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md` s.2.2 |
| The distribution got HARDER, not easier | mean flips -0.00132 -> +0.00338, but skew +0.093 -> -0.306 and excess kurtosis 8.85 -> 18.33; zero share 11.3% -> 20.2% | `active_distribution_atlas.html`; `XT_TARGET_PROTOTYPE_V2.md` s.2.3 |
| Possession-ending collapse | `action_ended_possession == True` (n=4,656) mean +0.03051, identical to that slice's own `xt_before` (`np.allclose`) -- construction, not discovery | `active_flag_ledger.html`; `LEAKAGE_AUDIT.html` Part C' |
| Construction coupling substantially RETIRED | strongest locked-feature correlation with the target falls from r=+0.5551 (v1) to **+0.1431** (v2), 74% lower -- v1's single most important caveat | `LEAKAGE_AUDIT.html` Part D'; `FEATURE_LOCK_CONFIRMATION_XT.html` |
| One threshold does not transfer | the std-derived flat margin spans **53.4%** of v2's rows (35.4% on v1) and 24 of 31 features classify differently under a robust alternative | every active report's `target_scale.threshold_transfer_warning`; section 1 of this document's prior version |
| Correlation with the old target | -0.016 (v1) -> **-0.1082** (v2): an order of magnitude stronger and correctly signed, still modest | `active_numerical_target_atlas.html`; `XT_TARGET_PROTOTYPE_V2.md` s.2.4 |

### The superseded v1 portal

`reports/analysis/xt_target_v1_superseded/` holds Prompt 65's complete portal against `target_xt_delta` (v1), moved there with `git mv` so its history is preserved. Its contents are unchanged apart from a supersession banner. Every number in it is the accurate record of what was measured against v1 and is deliberately not corrected. **This pass did not touch it.**

### What this pass changed on the active side

The portal index and this document, and nothing else. Every active-leg report file under `reports/analysis/xt_target/` stands exactly as Prompt 67 rebuilt it, byte-for-byte. Every active-leg generator script under `src/eda/` is unedited and still reproduces its output. No parquet, no model artefact, and no entry in `feature_config.py` was modified by this pass on either leg.

## 4. The PASSIVE leg (this pass) -- in full

### 4.1 The target, at the grain this suite measures it

`target_xt_delta_passive` lives in `outputs/prototypes/passive_xt_delta.parquet` at the UNIQUE-EVENT grain (198,009 defined values) and is left-joined READ-ONLY onto `data/features/passive_defense.parquet`'s 1,593,181 defender-slot rows by `event_id` in memory. Neither file is ever written back.

| Statistic | Unique event (Prompt 68) | Defender slot (this suite) | Active v2, for reference |
| --- | --- | --- | --- |
| mean | -0.00181 | -0.00267 | +0.00338 |
| std | 0.03828 | 0.03675 | 0.05838 |
| skew | -1.779 | -2.604 | -0.306 |
| excess kurtosis | 31.02 | 34.84 | 18.33 |
| % exactly zero | 26.3% | 26.4% | 20.2% |
| % negative | 36.9% | 37.2% | 35.4% |
| % positive | 36.9% | 36.4% | 44.4% |
| IQR | n/a | 0.003591 | 0.008829 |
| MAD | n/a | 0.001809 | 0.003853 |
| n defined | 198,009 | 1,590,600 | 55,761 |

> **Prompt 68's shape finding -- CONFIRMED, and MORE extreme at the grain this suite works at** -- These are the SAME target measured at two grains, not a discrepancy to reconcile. Prompt 68 reported the unique-event distribution because that is the grain the target is computed at. Every report in this suite measures it at the defender-slot grain, because that is the grain the features live at and the grain a model on this leg would train on. The row-level distribution is MORE extreme on every shape statistic -- measured on this run, skew -2.6036 against Prompt 68's -1.779 and excess kurtosis 34.8444 against 31.02 -- and the mechanism is understood rather than guessed at: events with more visible defender slots contribute more rows, so the row-level distribution is the event-level one re-weighted by visible-defender count, and the extreme shot-xG-tail events are exactly the crowded, well-observed ones that carry the most slots. The directional shares barely move (26.4% zero / 37.2% negative / 36.4% positive here against 26.3 / 36.9 / 36.9 there), which is the reassuring part: the re-weighting lengthens the tails without changing which way the target points. Wherever this suite cites a Prompt 68 figure it is labelled as the unique-event figure.

### 4.2 The shared-target property, and what it does and does not affect

> **carried forward from Prompt 68, and the single most important thing to hold while reading this half of the portal** -- Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md sections 1 and 2.5): target_xt_delta_passive is computed ONCE PER UNIQUE event_id (198,354 rows) and left-joined onto passive_defense.parquet's 1,593,181 defender-slot rows, so the identical value repeats across every defender row sharing an event_id (mean ~8.03 rows per event). This is a deliberate, pre-existing property of this leg, not something to fix: target_future_shot_10s and target_future_xg_10s already work exactly the same way, checked directly across all 198,354 events with 0 showing more than one distinct value. No per-defender attribution scheme was built and building one is explicitly out of scope, the same call Prompt 68 made. The consequence for this suite is a measurement caveat rather than a defect: rows are not independent within an event, so row-level p-values are computed on an effective sample size far smaller than their nominal n. Effect sizes, shares and ranks are unaffected.

**What this does NOT affect:** effect sizes, directional shares, ranks, and every threshold in this suite, all of which are computed from means and counts that a repeated y value paired with genuinely varying x values still estimates correctly. **What it DOES affect:** the standard error behind any row-level test. Welch's t-test appears in the Leakage Audit's Parts B/C/D and its nominal n there is roughly eight times its effective one. That caveat is stated in the report itself, and **no verdict anywhere in this half of the portal is gated on a p-value** -- every one turns on an effect size, a share, or a structural argument about how a column is built. The existing passive xg reports state the row grain but carry no such caveat (checked directly in `reports/analysis/xg_target/`), so this is an addition rather than a mirrored convention.

**No per-defender attribution scheme was invented, and that is explicitly out of scope** -- the same call Prompt 68 made. A lift or a category mean on this leg says 'events at which at least this defender was positioned this way tend to see the ball's threat move this way', not 'this defender caused it'.

### 4.3 The threshold that had to be recomputed for this leg

> **the std-derived flat margin transfers WORSE here than on the active leg -- recomputed, not inherited** -- THIS THRESHOLD DOES NOT TRANSFER CLEANLY, AND ON THIS LEG IT TRANSFERS WORSE THAN ON ACTIVE -- recomputed on the real passive parquet rather than inherited from active's numbers, and reported rather than forced to fit. Prompt 65 tuned the translated flat_margin against a target with excess kurtosis 8.85, where the margin was 0.53 x the MAD and spanned 35.4% of rows -- a genuinely narrow 'too small to matter' band. Prompt 67 found that on target_xt_delta_v2 (excess kurtosis 18.33) the same std-derived margin had grown to 1.28 x the MAD and 53.4% of rows, because the shot-xG injection inflates std while the bulk tightens. This leg is heavier-tailed again (excess kurtosis 34.8444, IQR 0.003591, MAD 0.001809), so the same mechanism applies more strongly: the translated margin is 0.002736, which is 1.51 x the MAD, 0.76 x the IQR, and spans 62.1% of all rows. A flat band covering this much of the data is not doing its job. A NAIVE TRANSFER OF ACTIVE'S OWN ABSOLUTE MARGIN (0.004929) WOULD HAVE BEEN WORSE STILL, since this leg's std is smaller (0.038 vs 0.058) while its bulk is tighter again -- the margin is therefore recomputed from this leg's own std, not copied. The literal translated margin is KEPT as the headline number so the active and passive halves of this portal stay like-for-like and the xg suite's convention is still the one being followed, but the MAD-derived robust alternative (0.000200) is published beside it on every run and every shape classification in this half of the portal should be read with this inflation in mind.

| Quantity | v1 (active) | v2 (active) | passive (this pass) |
| --- | --- | --- | --- |
| translated flat margin | 0.005486 | 0.004929 | 0.002736 |
| as a multiple of MAD | 0.53x | 1.28x | 1.51x |
| as a multiple of IQR | 0.29x | 0.56x | 0.76x |
| share of all rows inside it | 35.4% | 53.4% | **62.1%** |
| robust (MAD-derived) alternative | 0.001286 | 0.000482 | 0.000200 |
| features classifying differently under the robust margin | 15 of 31 | 24 of 31 | **21 of 34** |
| excess kurtosis behind it | 8.85 | 18.33 | 34.84 |

**A naive transfer of the active leg's own absolute margin would have been wrong in a specific, identifiable direction.** Active v2's margin is 0.004929 against a std of 0.058385; this leg's std is 0.036748, so the correctly-translated margin is 0.002736 -- about 1.8x smaller. Reusing active's number would have made the 'too small to matter' band nearly twice as wide as it should be, and almost every verdict in this half of the portal is a SHAPE comparison, so a wider flat band makes categories and tournaments look more alike and understates divergence. The margin was recomputed from this leg's own standard deviation everywhere it is used (Numerical Target Atlas, Tournament Stability, both Slice Stratifications, Feature Interaction, Review Methodology). **The literal translated margin is KEPT as the headline** so the two halves of this portal stay like-for-like and the xg suite's convention is still the one being followed; the robust alternative is published beside it everywhere.

**One report needed no recomputation, and that was checked rather than assumed.** NO THRESHOLD IN THIS REPORT NEEDED PASSIVE-SPECIFIC RECOMPUTATION, and that is a checked result rather than an omission. The only tunable quantities here are QCUT_N (a count) and _verdict()'s sign comparison (sign-free of magnitude). Neither can be distorted by this leg's heavier tails. The reports in this portal that DID need the flat_margin recomputed are the Numerical Target Atlas, the Tournament Stability Check, both Slice Stratifications and the Feature Interaction Analysis -- every one of which uses classify_shape() or an absolute marginal-delta cutoff.

### 4.4 Correlation with the old passive targets -- weaker than active's, stated plainly

> **carried forward from Prompt 68** -- Prompt 68 finding carried forward (outputs/prototypes/PASSIVE_XT_TARGET_PROTOTYPE.md section 2.3): correlation with the two OLD passive targets is correctly signed but numerically WEAKER than the active leg's own improvement -- Pearson r = -0.0629 against target_future_shot_10s and -0.0715 against target_future_xg_10s, against active v2's -0.108. More threat denied is still associated with less future attacking output, which is the sensible direction, but the relationship is looser on this leg. Stated plainly rather than talked up. A plausible reason, recorded by Prompt 68 and not chased here: a passive snapshot is one defender's off-ball positioning at an arbitrary point in a live attack, one step further removed from the moment-to-moment outcome than an active defensive ACTION is.

Re-measured on this run at the defender-slot grain: r=-0.0736 against `target_future_xg_10s` and r=-0.0651 against `target_future_shot_10s`, over n=1,590,600 rows. Prompt 68 reported -0.0715 and -0.0629 at the unique-event grain; the small difference is the same defender-count re-weighting that makes the row-level distribution heavier-tailed, not a disagreement. Both are correctly signed and both are weaker than the active leg's own -0.1082 against the same xg target. **This matters for reading the Numerical Target Atlas:** the two old targets are what the existing passive feature rankings were built against, so a feature that ranks highly here and not there is not necessarily contradicting anything -- the targets are only loosely related. It also matters because Adaptation 1 borrows `target_future_xg_10s`'s own standard-deviation fraction to set every threshold in this suite, so this is the one place where the old target still has a hand in the new one's numbers.

### 4.5 Construction coupling -- LIVE on this leg, unlike the corrected active leg

> **this is the passive leg's own most important caveat, and it has no counterpart in the xg or binary portals** -- target_xt_delta_passive = xt_before - xt_after, and on THIS leg xt_before = xT(ball_x, ball_y) at the snapshot itself -- a deterministic 8x12 grid lookup at two columns that are both in PASSIVE['continuous'] and therefore both LOCKED CANDIDATE FEATURES. That is a structurally different situation from the active leg, where xt_before is a lookup at the PREVIOUS event's location (not a column in the active feature table at all) and where Prompt 66 removed the xt_after coupling entirely. Here, one of the target's two terms is a pure function of two locked features, so it must be measured rather than assumed harmless.

| Measured on this run | Value |
| --- | --- |
| strongest locked feature vs `xt_before` | `ball_x` at r=+0.5199 |
| strongest locked feature vs the DELTA | `ball_x` at r=+0.2521 |
| active leg, v1, for scale | `distance_to_attacking_box` at r=+0.5551 vs the target |
| active leg, v2, for scale | `distance_to_attacking_box` at r=+0.1431 vs the target (74% lower) |

The first row is not itself a finding -- it is the definition of the term, since `xt_before` IS a grid lookup at `ball_x`/`ball_y`. The finding is what survives the subtraction, and it sits **between** the active leg's v1 and v2 figures, reached through a completely different mechanism. Prompt 67 was able to call the active leg's construction-coupling caveat substantially retired; **the equivalent caveat is LIVE here**, and a location-derived feature's relationship to this target should be read knowing it shares a definitional term with half of it. **No feature is dropped or proposed for dropping on this evidence** and `feature_config.py` is unchanged. Whether a model on this leg should use `ball_x`/`ball_y` at all for this target, and whether `xt_before` should be offered as a feature in its own right, are modelling-ladder questions and are explicitly not decided here.

### 4.6 Findings, report by report

**Category Atlas.** The active leg's headline Clearance finding does not transfer and is not forced into a counterpart: `event_type` does not exist in `passive_defense.parquet` (checked against the live schema), and this leg's nearest relative `on_ball_event_type` describes the ATTACKING action the snapshot is anchored to. On its own terms: across 4 categories, `Dribble` is the most threat-reducing at +0.001548 (n=21,429) and `Shot` the most threat-increasing at -0.066888 (n=24,487).

**Flag Ledger.** The active ledger's off-pool possession-flag section has no counterpart here: none of the three `action_*_possession` columns exists in the passive parquet, asserted against the live schema on this run. `action_ended_possession` still sets `xt_after = 0` for this target, but it is read from `events_with_targets.parquet` during Prompt 68's build and never lands in the passive feature table, so **no passive feature carries that construction coupling**. Top 5 by |lift|:

| Flag | lift | mean True | mean False | n True |
| --- | --- | --- | --- | --- |
| `is_in_defending_box` | -0.030452 | -0.030941 | -0.000488 | 114,164 |
| `has_screened_outcome` | +0.026806 | -0.002669 | -0.029475 | 1,590,321 |
| `is_deep_zone` | -0.016466 | -0.015726 | +0.000740 | 329,799 |
| `has_option_2` | +0.014704 | -0.002654 | -0.017358 | 1,588,420 |
| `is_in_attacking_box` | +0.013758 | +0.010439 | -0.003319 | 74,562 |

**Numerical Target Atlas.** 22/34 features' binned curves cross zero -- they separate threat-reducing from threat-increasing events rather than only varying in magnitude. 1 feature(s) flagged train/test-inconsistent. Top 5 by |Spearman rho|:

| Feature | Spearman rho | Shape | Curve crosses zero |
| --- | --- | --- | --- |
| `ball_x` | 0.3386 | inverse-U | True |
| `top_option_1_threat_score` | 0.2537 | inverse-U | True |
| `top_option_3_threat_score` | 0.2361 | monotonic increasing | True |
| `top_option_2_threat_score` | 0.2347 | monotonic increasing | True |
| `distance_to_attacking_goal` | -0.2314 | inverse-U | True |

> **read these shape labels with section 4.3 in hand** -- 21 of 34 features classify differently under the MAD-derived flat margin than under the std-derived one used above (against 24 of 31 on the active leg's v2 target and 15 of 31 on v1). The **Shape** column is indicative on this target, not decided.

**Leakage Audit.** Part A clean. Parts B/C/D mirror the passive xg audit's own columns literally -- no substitution was needed, because leakage auditing in this project IS a passive methodology and it is the ACTIVE leg that has no native counterpart. `has_screened_outcome`'s exactly-zero share gap is -8.5pp with a mean difference of +0.026806; the exclusion holds on the censoring MECHANISM, which is a property of how the column is built rather than of which target it is measured against, so this target's number is the same structural argument measured a third time and should not be cited as fresh evidence. `has_option_2`/`has_option_3` stay IN the candidate list, the same disposition the binary and xg audits reached. Part E' is section 4.5 above.

**Confound (Reversal) Testing.** All five established passive tests mirrored literally -- same columns, same labels, same `_verdict()` function reused byte-for-byte. **This report contains no new test and needed no invention**, in direct contrast to the active half, where Prompt 65 found exactly one active confound test in the whole repo and had to write two of its own. Verdicts on this target:

| Test | Unconditional | Given a non-zero delta |
| --- | --- | --- |
| Marking tightness reversal | partially | partially |
| Lane screening reversal | no | no |
| has_option_2 shot-rate gap vs top_option_1_threat_score | partially | partially |
| has_option_3 shot-rate gap vs top_option_1_threat_score | partially | partially |
| has_option_2 shot-rate gap vs defender_x | partially | partially |

**Tournament Stability Check.** 0 of 4 passive features show a genuine tournament-level difference rather than arbitrary train/test noise, and 0 of 4 verdicts would change under the MAD-derived robust margin -- a check added here that neither the xg nor the active xT version carries, precisely because this leg's margin is wide enough that a shape verdict could rest on it.

| Feature | Verdict | Flips under the robust margin |
| --- | --- | --- |
| `top_option_1_distance_from_ball` | arbitrary train/test noise | False |
| `top_option_3_dx` | arbitrary train/test noise | False |
| `top_option_3_dy` | arbitrary train/test noise | False |
| `angle_to_attacking_goal` | arbitrary train/test noise | False |

**Slice Stratification.** V1: 46 cells, 127 genuine divergences unconditionally and 126 once the zero mass is removed, with 23 categories where that removal flips the conclusion. V2: 91 cells, 152 / 148.

> **the one comparison this leg can make and the active leg cannot** -- APPLICABLE ON THIS LEG, unlike the active half. Unconditionally, defender_archetype_name produced 52 genuine divergence(s) across 13 cells (4.00/cell) versus the 6 boolean slicers' 100 across 78 cells (1.28/cell). This is the comparison the ACTIVE half of this portal reports as NOT APPLICABLE, because defender_archetype_name is a passive-only column that does not exist in the active parquet. It is made here as this leg's own content, not presented as something the portal previously had. Read it as a descriptive rate, not a significance test: the archetype slicer has 8 non-null categories against each boolean slicer's 2, and more categories mechanically offer more chances to diverge, so a higher per-cell rate is expected before any tactical interpretation is reached for.

**Feature Interaction Analysis.** 5/5 pairs classify the same way unconditionally and on the non-zero-delta subset; 1/5 have within-stratum deltas that change SIGN -- feature A's effect reverses DIRECTION across levels of feature B, a finding that cannot arise on the xG target at all. All five pairs involve this leg's own screening, option or engagement columns and could not have been run on the active half.

| Feature A | Feature B | Classification | Stratum deltas change sign |
| --- | --- | --- | --- |
| `top_option_2_threat_score` | `top_option_2_distance_from_ball` | interactive | no |
| `lane_screening_score_option_2` | `engagement_distance_to_carrier` | interactive | no |
| `marking_tightness` | `engagement_distance_to_carrier` | interactive | no |
| `lane_screening_score_option_1` | `marking_tightness` | additive | no |
| `overload_score` | `attacking_goal_centrality` | inconclusive | yes |

**Review Methodology.** 104 PASSIVE review pairs; 80 remain needs_human_call (dominated, as on every target and both legs, by continuous-continuous pairs no rule covers); 0 Type-2 pair(s) resolve on the opposite-sign rule. Types 1, 3 and 4 are target-independent and their resolvers are imported unchanged, so those verdicts match the binary and xg passive reviews **by construction, not by coincidence**.

**Feature Lock Confirmation.** The locked PASSIVE feature set (38 features, matching the expected 38) is confirmed internally consistent. The correlation diff is reused from the binary confirmation and labelled identical-by-construction rather than silently recomputed. **No feature is added, dropped or re-tiered**, on either leg.

## 5. Caveats recorded, and open questions this portal does not answer

**Nothing in this portal changes `feature_config.py`, on either leg, and no model artefact was touched.** No feature is added, dropped or re-tiered.

### Passive-leg caveats recorded by this pass

- **Caveat P1 -- construction coupling, LIVE on this leg.** `xt_before = xT(ball_x, ball_y)` and both columns are locked features, so one of the target's two terms is a deterministic function of the candidate set. What survives the subtraction is measured in section 4.5 and sits between the active leg's v1 and v2 figures. Unlike the active leg's, this caveat is NOT retired.
- **Caveat P2 -- effects can sit in the zero mass.** The same caveat Prompt 65 recorded for the active leg, applying to MORE rows here: 26.4% of passive rows are exactly zero against 20.2% of active v2 rows. Every table in this half of the portal reports negative / zero / positive shares beside its mean, and the Leakage Audit's Parts B/C/D are read on the zero-share gap as well as the mean.
- **Caveat P3 -- the flat-margin threshold describes this distribution even less well than active's.** It spans 62.1% of rows (1.51x the MAD) and 21 of 34 features classify differently under the robust alternative. Recomputed from this leg's own std rather than transferred, and reported rather than swapped.
- **Caveat P4 -- rows are not independent within an event.** One target value per `event_id`, repeated across ~8 defender slots. Row-level p-values are optimistic by roughly that factor. No verdict in this half of the portal is gated on one.

> **methodological warning for anyone extending the passive half** -- This target is the hardest distribution anywhere in this portal to work with -- excess kurtosis ~35 at the row grain against active v2's 18.33 and v1's 8.85, with a strongly left-skewed tail. Prefer rank-based, count-based, ratio-based and sign-based instruments: every one of them transferred to this leg without a scratch (the confound suite's sign-only `_verdict()`, `SMALL_N_THRESHOLD`'s row count, the substitutive/additive ratios, `MIN_CELL_N_FOR_DELTA`, the quantile-decile binning). The ONE instrument that did not transfer is the only standard-deviation-based one in the suite, and it did not transfer on the active leg either. That pattern has now held across three targets.

### Open questions this portal does NOT answer

- **Per-defender attribution for the passive target.** Explicitly out of scope, the same call Prompt 68 made. The target is a property of the event and no scheme for apportioning it across the defenders present was built or is proposed.
- **Whether a model on the passive leg should use `ball_x`/`ball_y` for this target**, given Caveat P1, and whether `xt_before` should be offered as a feature in its own right rather than left implicit. Both are modelling-ladder questions.
- **What model architecture suits a left-skewed, very heavy-tailed, sign-carrying target** -- Prompt 68 found this leg _less_ like the existing hurdle architecture's shape than active v2, which was already less like it than v1. This is an EDA portal and takes no position.
- **Whether the passive leg's `xt_before` should be defined at the previous event's location**, as the active leg's is, rather than at the snapshot's own. Prompt 68 made the opposite choice deliberately and justified it (the passive snapshot's `ball_x`/`ball_y` IS the ball's location at the anchoring event), but the asymmetry between the two legs' `xt_before` definitions is recorded here rather than resolved.
- **Whether the log-transform question should be reopened for this leg specifically.** Not applied anywhere in this report suite, on either leg, because it would break comparability with the xg portal and with this portal's own other half. This leg's tails make it the most tempting case in the project. Recorded for the modelling legs.
- **A passive Player-Level Validity Check and Player-Grouped Split Check.** Both are ACTIVE-ONLY project-wide for a structural reason (the passive leg is not player-indexed the same way), and both are target-independent, so building them would be outside this pass's remit even if the data supported it. Stated in the Step-0 table rather than left as an unexplained absence.

## 6. Document map

### Active leg (built by Prompts 65/67 -- untouched by this pass)

| Report | Files |
| --- | --- |
| Active -- Category Atlas | `active_category_atlas.html` / `.json` |
| Active -- Flag Ledger | `active_flag_ledger.html` / `.json` |
| Active -- Distribution Atlas | `active_distribution_atlas.html`, `active_distribution_atlas_reconstructed.html`, `.json` |
| Active -- Numerical Target Atlas | `active_numerical_target_atlas.html` / `.json` |
| Active -- Review Methodology | `REVIEW_METHODOLOGY.html`, `REVIEW_ANALYSIS.json` |
| Active -- Leakage Audit | `LEAKAGE_AUDIT.html` / `.json` |
| Active -- Confound (Reversal) Testing | `CONFOUND_ANALYSIS.html` / `.json` |
| Active -- Tournament Stability Check | `TOURNAMENT_STABILITY_CHECK.html` / `.json` |
| Active -- Slice Stratification V1 | `SLICE_STRATIFICATION.html` / `.json` |
| Active -- Slice Stratification V2 | `SLICE_STRATIFICATION_V2.html` / `.json` |
| Active -- Feature Interaction Analysis | `FEATURE_INTERACTION_ANALYSIS.html` / `.json` |
| Active -- Feature Lock Confirmation | `FEATURE_LOCK_CONFIRMATION_XT.html` / `.json` |

### Passive leg (built by this pass)

| Report | Files |
| --- | --- |
| Passive -- Category Atlas | `passive_category_atlas.html` / `.json` |
| Passive -- Flag Ledger | `passive_flag_ledger.html` / `.json` |
| Passive -- Distribution Atlas | `passive_distribution_atlas.html`, `passive_distribution_atlas_reconstructed.html`, `.json` |
| Passive -- Numerical Target Atlas | `passive_numerical_target_atlas.html` / `.json` |
| Passive -- Review Methodology | `PASSIVE_REVIEW_METHODOLOGY.html`, `PASSIVE_REVIEW_ANALYSIS.json` |
| Passive -- Leakage Audit | `PASSIVE_LEAKAGE_AUDIT.html` / `.json` |
| Passive -- Confound (Reversal) Testing | `PASSIVE_CONFOUND_ANALYSIS.html` / `.json` |
| Passive -- Tournament Stability Check | `PASSIVE_TOURNAMENT_STABILITY_CHECK.html` / `.json` |
| Passive -- Slice Stratification V1 | `PASSIVE_SLICE_STRATIFICATION.html` / `.json` |
| Passive -- Slice Stratification V2 | `PASSIVE_SLICE_STRATIFICATION_V2.html` / `.json` |
| Passive -- Feature Interaction Analysis | `PASSIVE_FEATURE_INTERACTION_ANALYSIS.html` / `.json` |
| Passive -- Feature Lock Confirmation | `PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.html` / `.json` |

### Shared

| Report | Files |
| --- | --- |
| Master Findings (this document, both legs) | `MASTER_FINDINGS.md` / `.html` |
| Portal index (Active + Passive groups) | `INDEX.html` |

### Linked, NOT rebuilt (target-independent -- properties of the features, not of the target)

See the Step-0 table in section 2 for which of these have a passive half and which are ACTIVE-ONLY project-wide. A change to which target the features are measured against cannot move a feature-vs-feature statistic, so rebuilding any of them would produce byte-identical output.

### New generator scripts (all added as siblings; no existing script was edited)

Nothing under `src/eda/` that already existed was modified. Prompt 65's 21 `*_xt.py` generators, Prompt 67's 8 `*_xt_v2*.py` siblings, and the `xg_target` and `shot_target` portals' own generators all remain exactly reproducible.

- `src/eda/xt_passive_common.py` -- the passive re-point layer, Adaptation 1 recomputed for this leg, and Prompt 68's carried-forward findings
- `src/eda/generate_reports_xt_passive.py` -- Category Atlas + Flag Ledger
- `src/eda/generate_distribution_atlas_xt_passive.py` -- Distribution Atlas, both variants
- `src/eda/generate_numerical_xt_target_passive.py` -- Numerical Target Atlas (compute + render)
- `src/eda/generate_leakage_audit_xt_passive.py` -- Leakage Audit (compute + render)
- `src/eda/generate_confound_analysis_xt_passive.py` -- Confound Testing (compute + render)
- `src/eda/generate_tournament_stability_check_xt_passive.py` -- Tournament Stability
- `src/eda/generate_slice_stratification_xt_passive.py` -- Slice Stratification V1 + V2
- `src/eda/generate_feature_interaction_analysis_xt_passive.py` -- Feature Interaction
- `src/eda/generate_review_analysis_xt_passive.py` -- Review Methodology
- `src/eda/generate_feature_lock_confirmation_xt_passive.py` -- Feature Lock Confirmation
- `src/eda/generate_master_findings_xt_passive.py` (this document)
- `src/eda/generate_eda_portal_xt_passive.py` -- portal shell for BOTH legs, AND the driver that makes the passive re-point work

Rebuild the passive half and the shared documents with: `.venv/Scripts/python.exe -m src.eda.generate_eda_portal_xt_passive --all`. Rebuild the active half, unchanged, with: `.venv/Scripts/python.exe -m src.eda.generate_eda_portal_xt_v2 --all`. **Run them in separate processes** -- both re-point the same `xt_common` module attributes, and whichever imports last would win.
