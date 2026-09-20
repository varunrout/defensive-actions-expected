# Master Findings -- xT-Delta v2 Target (Active-Binary Leg Only)

> **Supersedes the Prompt 65 portal.** This portal is built against `target_xt_delta_v2`. The prior pass against `target_xt_delta` (v1) is preserved unchanged at `reports/analysis/xt_target_v1_superseded/`.

> **Scope: the ACTIVE-BINARY leg only.** The passive leg is explicitly out of scope for this target and no passive-side xT-delta report exists. Nothing in this document, or in this portal, covers it.

**Bottom line up front.** **This portal supersedes an earlier pass and should be read with that in mind.** Prompt 65 built these same thirteen reports against `target_xt_delta` (v1), whose `xt_after` term scored the defensive action's OWN location; Prompt 66 found that wrong and replaced it with a possession-outcome definition. The headline consequence reproduces here exactly: **Clearance moves from 8th of 8 event types (-0.032482) to 1 of 8 event_type categories at +0.024706**, and it survives conditioning on both defending-box proximity and on the construction rule that could have made it self-fulfilling. v1's single most important caveat, construction coupling, is **74% weaker** and substantially retired. But `target_xt_delta_v2` is a HARDER distribution than v1 -- skew -0.306, excess kurtosis 18.33, 20.2% exactly zero -- and **one v1-tuned threshold does not transfer: the std-derived flat margin now spans 53.4% of all rows, and 24 of 31 features change shape classification under a robust alternative. That is reported, not forced to fit.** No feature was added, dropped or re-tiered, no model artefact was touched, and the seven target-independent reports were linked rather than rebuilt.

## 1. What changed from v1, and why

> **this portal supersedes an earlier pass -- read this first** -- Prompt 65 built these same thirteen reports against `target_xt_delta` (v1). Prompt 66 found that target's `xt_after` term wrong and corrected it. This portal is the rebuild against the corrected target. **The v1 portal is preserved UNCHANGED at `reports/analysis/xt_target_v1_superseded/`** -- every number and finding in it is the accurate historical record of what Prompt 65 measured against v1, not something to correct or update.

### The correction itself

`xt_before` is **identical** in v1 and v2 -- the previous event's location, looked up in the same 8x12 xT grid. Only `xt_after` changed:

|  | v1 (Prompt 64) | v2 (Prompt 66) |
| --- | --- | --- |
| `xt_after` definition | `xT(action_x, action_y)` -- the defensive action's OWN recorded location | `0.0` if the action ends its own possession; else the next same-possession event's grid value; else that event's own `shot_statsbomb_xg` if it is a Shot |
| rows affected | all 56,068 | 4,662 possession-ending / 954 shot-follows / 50,452 ordinary |
| the problem it fixes | for a Clearance, `action_x`/`action_y` is where the ball WAS when cleared -- often deep in the defending box, a high-xT cell by construction | a clearance that ends the attack now scores `xt_after = 0`: all of the threat is denied |

### The three consequences that matter for reading this portal

> **1. The Clearance artefact REVERSED -- it did not merely shrink** -- Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.2): Prompt 64's Clearance artefact -- Clearance looking like the WORST event type because xt_after was looked up at the clearance's own deep-in-the-box location -- is not merely reduced under the corrected xt_after, it FULLY REVERSES. Clearance moves from 8th of 8 event types (worst) at -0.0325 mean delta under v1 to 1st of 8 (best) at +0.0247 under v2, and in the motivating action_ended_possession slice from -0.0315 to +0.0406. The mechanism is understood, not assumed: possession-ending clearances now get xt_after = 0 by construction, so the dangerous coordinate is never looked up at all. Any v1-era statement that Clearance ranks last, or that clearances carry a systematically negative delta, is superseded and must not be repeated. (Aside recorded by Prompt 66 and not chased here: Block, at -0.0185, is now the lowest-mean event type under v2.)

> **2. The distribution moved the right way on its mean and the WRONG way on its shape** -- Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.3): target_xt_delta_v2's mean flips from v1's slightly negative -0.00132 to a clearly positive +0.00338 -- on average, active defensive actions now genuinely deny threat, the football-sensible direction v1 never achieved. But the shape got HARDER, not easier: skew moves from +0.093 (near-symmetric) to -0.306 (mildly left-skewed) and excess kurtosis from 8.85 to 18.33 (much heavier tails), while the exactly-zero share rises from 11.3% to 20.2% and the negative share falls from 47.0% to 35.4%. The heavier tails are a direct, understood consequence of injecting real shot_statsbomb_xg values (up to 0.897) as xt_after for the 954 rows -- 1.9% of continuing-possession rows -- where a Shot immediately follows the defensive action; the 96-cell xT grid's own maximum cell is only 0.2575, so it could never produce such values. This REINFORCES rather than resolves Prompt 64's open flag that the existing hurdle architecture does not fit this target's shape as-is: v2 is if anything less like that shape than v1 was.

> **3. Correlation with the old target improved, but is still modest** -- Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.4): correlation with the old target_future_xg_10s improves from v1's essentially-zero -0.016 to -0.108 -- still modest, but now an order of magnitude stronger AND correctly signed: more threat denied by a defensive action is associated with less future attacking output in the next 10 seconds. v2 and v1 are themselves moderately correlated at +0.571, which is sensible given they share the identical xt_before term and differ only in xt_after.

### What that did to Prompt 65's two recorded caveats

Prompt 65 closed with two caveats, recorded but not resolved. The correction moves them in opposite directions, and neither outcome was assumed -- both were re-measured on this run:

- **Caveat 1, construction coupling -- SUBSTANTIALLY RETIRED.** v1's single most important caveat was that because `xt_after` was looked up from `action_x`/`action_y`, every location-derived locked feature shared a definitional term with half the target; the strongest sat at r=+0.5551 (`distance_to_attacking_box`). v2 never looks up the acting row's own coordinates at all, and the strongest locked-feature correlation falls to r=+0.1431 -- **74% lower**. The Numerical Target Atlas no longer needs the heavy recovery-by-definition discount v1 attached to it.
- **Caveat 2, effects living in the zero mass -- SURVIVES, and matters more.** A mean-difference test misses effects that sit in this target's exactly-zero share. That share rose from 11.3% (v1) to 20.2% (v2), so the caveat applies to more rows than before, not fewer. Every table in this portal still reports negative / zero / positive shares beside its mean.

### One threshold that does NOT transfer cleanly -- stated, not forced to fit

> **the std-derived flat margin now spans the majority of the distribution** -- THIS THRESHOLD DOES NOT TRANSFER CLEANLY FROM v1, and that is reported rather than forced to fit. Prompt 65 tuned its translated flat_margin against a target with excess kurtosis 8.85, an IQR of 0.019119 and a MAD of 0.010277; the resulting margin (0.005486) was 0.53 x that MAD and spanned 35.4% of rows -- a genuinely narrow 'too small to matter' band. v2's tails are far heavier (excess kurtosis 18.3289) because Prompt 66 injects real shot xG values up to 0.897 for the 954 rows where a Shot follows the defensive action, while v2's BULK is much tighter (IQR 0.008829, MAD 0.003853). Because the translation is std-derived and std is inflated by exactly those tails, the margin barely moves (0.004929) while the distribution it is meant to describe more than halved: it is now 1.28 x the MAD, 0.56 x the IQR, and spans 53.4% of all rows. A flat band covering the majority of the data is not doing its job. The literal translated margin is KEPT as the headline number so v1-vs-v2 comparisons stay like-for-like and the xg suite's convention is still the one being followed, but the MAD-derived robust alternative (0.000482) is published beside it on every run and every shape classification in this suite should be read with this inflation in mind.

This is not a cosmetic point. `flat_margin` feeds `classify_shape()` in five reports. Measured directly on this run: of the 31 features in the Numerical Target Atlas pool, **24 classify differently** under the std-derived margin than under the MAD-derived one -- against **15 of 31** for the same comparison on v1. **Shape labels in this portal are therefore materially less robust than the v1 portal's were, and should be read as indicative rather than decided.** The std-derived margin is kept as the headline so v1-vs-v2 cells stay comparable and the xg suite's convention is still the one being followed; the robust alternative is published beside it in every report's `target_scale.robust_scale` block.

### Closeout document: folded in again, and why the call did not change

Prompt 65 decided against a separate `PATTERN_ANALYSIS_CLOSEOUT`-style document for this portal (`reports/analysis/shot_target/` has one), on the grounds that it exists to close out a multi-prompt campaign spanning two targets and both datasets, whereas this portal is a single-pass, single-leg suite whose findings fit in one section. **The same call is made here, for the same reason plus one more:** a separate closeout would now also have to carry the whole v1-vs-v2 comparison, and splitting that across two documents is precisely how the two would drift apart. Everything is in this file.

## 2. Step 0 -- script-by-script disposition: re-point vs rewrite

**This section is a required deliverable, not scratch notes.** All 21 of Prompt 65's `generate_*_xt.py` scripts plus `xt_common.py` were read in full and classified one by one: can this be re-pointed at `active_binary_xt_delta_v2.parquet`/`target_xt_delta_v2` by a parameter change alone, or does it bake in a v1-specific scale or direction assumption that needs real new logic?

> **the mechanism that made most of this a re-point** -- Prompt 65's suite has one very fortunate property, found by reading rather than assumed: **every single generator reaches the target through `from src.eda import xt_common as xc`**, and binds the target name, parquet path, output directory, statistics helpers and finding strings from that one module. None opens the prototype parquet directly; none hardcodes the target column name in a computation. So the suite re-points by rebinding `xt_common`'s attributes before the generators are imported -- which is exactly what `xt_v2_common.py` does. **No file Prompt 65 wrote was edited.** All 21 still regenerate the preserved v1 portal if run against an un-re-pointed `xt_common`.

**Result: 14 re-pointed, 8 rewritten as new siblings.** Of the 14 re-points, five additionally needed a declared, asserted correction to one or two hardcoded sentences that asserted v1's distribution shape as fact -- applied at fragment level by `xt_v2_prose_fixups.py` and `generate_feature_lock_report_xt_v2.py` rather than by duplicating whole renderers. Each correction asserts that it matched, so a future change to a Prompt-65 script fails loudly instead of silently republishing a stale claim.

| Script | Re-point vs rewrite | What changed, and why |
| --- | --- | --- |
| `src/eda/xt_common.py` | **REWRITE** -> `src/eda/xt_v2_common.py` | The one module every other script reaches the target through. Re-points `TARGET`, `XT_PARQUET` and the carried-forward finding strings, and wraps `target_scale()`. Its Adaptation-1 prose asserted a "near-zero, NEGATIVE mean" -- v2's mean is POSITIVE, so the justification is restated (the translation is kept, for the reason that still holds). Adds the `robust_scale` block and `threshold_transfer_warning`. Prompt 64's two finding strings are replaced with Prompt 66's, because the Clearance one now asserts something v2 disproves. |
| `generate_reports_xt.py` (Category Atlas + Flag Ledger) | **REWRITE** -> `generate_reports_xt_v2.py` | All computation re-points cleanly. Three pieces of prose did not: the Clearance note announced the artefact as "confirmed present" (it reverses); the possession-flag note quoted v1's "77-79% non-zero" (under v2 those flags collapse onto `xt_before` by construction, checked here with `np.allclose`); and the scale card said "roughly symmetric" (skew is -0.306). Pool construction, lift functions and the diverging-bar geometry are imported unchanged. |
| `generate_distribution_atlas_xt.py` | RE-POINT | Fully data-driven -- its target-shape section recomputes skew/kurtosis/shares from the parquet, so it reports v2's own figures. Its one v1-specific design decision, clipping the histogram SYMMETRICALLY at the p99 of |delta| instead of min-to-p99, becomes MORE necessary under v2's heavier tails, not less. |
| `generate_numerical_xt_target_analysis.py` | RE-POINT | `flat_margin` and `range_trigger` are computed at runtime from the real parquet, so they recompute against v2's own std automatically. The resulting margin is the threshold that does not transfer cleanly -- reported in section 1 and in every `target_scale` block, not silently swapped. |
| `generate_numerical_xt_target_reports.py` | RE-POINT + declared prose correction | Renders entirely from the compute step's JSON. Three hardcoded sentences asserted v1's shape ("roughly symmetric", "near-zero, negative mean"); corrected by `xt_v2_prose_fixups.py` at fragment level with an assertion that each matched, rather than by duplicating a 338-line renderer. |
| `generate_review_analysis_xt.py` | RE-POINT + declared prose correction | Types 1/3/4 resolvers are target-independent and imported unchanged; only Type 2 uses target evidence, and its two thresholds recompute at runtime. One sentence in the JSON `note` asserted the negative mean; corrected as above. |
| `generate_review_report_xt.py` | RE-POINT | Pure renderer over the JSON. Re-run after the prose correction so the two cannot disagree. |
| `generate_leakage_audit_xt.py` | **REWRITE** -> `generate_leakage_audit_xt_v2.py` | **The largest genuine logic change in this rebuild.** Parts A/B'/C' keep their methods (Part A imported unchanged; `_split_stats` reused byte-for-byte). **Part D' was rewritten from scratch: its premise is false under v2.** It existed to quantify "xt_after is looked up from action_x/action_y, so location-derived features share a definitional term with half the target" -- a mechanism Prompt 66 removed entirely. Coupling is re-measured rather than restated, and a NEW v2-only channel is tested (coupling moved from a location term to the `action_ended_possession` flag). |
| `generate_leakage_report_xt.py` | **REWRITE** -> `generate_leakage_report_xt_v2.py` | Cannot render the v2 JSON at all -- Part D's key set changed with the rewrite. Part A, the missing-parts section and all shared table helpers are imported and reused. |
| `generate_confound_analysis_xt.py` | **REWRITE** -> `generate_confound_analysis_xt_v2.py` | All machinery reused by import, including `_verdict()` byte-for-byte and `QCUT_N=4`. `_verdict()` compares only the SIGN of first-to-last deltas, never a magnitude -- re-checked, not assumed, and that is exactly why this report needed no threshold change despite v2's heavier tails. Test 1's spec is unchanged. **Test 2 was re-framed**: it asked whether Clearance's NEGATIVE mean survives conditioning, and that mean is now positive. **Test 3 is NEW** -- whether Clearance's positive mean is itself an artefact of the `xt_after = 0` construction rule, a mechanism that did not exist under v1. |
| `generate_confound_report_xt.py` | RE-POINT | Renders whatever tests the JSON carries, including the new third one, with no change. |
| `generate_tournament_stability_check_xt.py` | RE-POINT + declared prose correction | Imports `INVESTIGATE` and the tournament map; bin edges still computed once on pooled data. Only `flat_margin` varies and it recomputes at runtime. One JSON `methodology` sentence asserted the negative mean; corrected as above. |
| `generate_tournament_stability_report_xt.py` | RE-POINT | Pure renderer. Re-run after the prose correction. |
| `generate_slice_stratification_xt.py` | RE-POINT | `FEATURE_SLICER_PLAN`, `build_plan()`, `SMALL_N_THRESHOLD=1000` and `STRUCTURAL_CAUTION_CATEGORIES` all imported unchanged. Small-n thresholds are ROW COUNTS -- scale-free, so v2's heavier tails cannot touch them. Only `flat_margin` moves, and it recomputes. |
| `generate_slice_stratification_v2_xt.py` | RE-POINT | Same as above for the boolean-slicer half. The archetype slicer is still a passive column, so V2's closing archetype-vs-boolean comparison is still not applicable on the active leg -- reported as such, exactly as in the v1 portal. |
| `generate_slice_stratification_report_xt.py` | RE-POINT | Pure renderer over both slice JSONs. |
| `generate_feature_interaction_analysis_xt.py` | RE-POINT | `PAIRS`, `Q=4` and `_bin_labeled` imported unchanged; pairs deliberately NOT re-ranked, so all three targets stay comparable. `SUBSTITUTIVE_RATIO_MAX=0.3`, `ADDITIVE_MAGNITUDE_RATIO_MAX=2.0` and `MIN_CELL_N_FOR_DELTA=20` are ratios and row counts -- scale- and sign-free, so they transfer untouched. |
| `generate_feature_interaction_report_xt.py` | RE-POINT | Pure renderer over the JSON. |
| `generate_feature_lock_confirmation_xt.py` | **REWRITE** -> `generate_feature_lock_confirmation_xt_v2.py` | Every roll-up re-points correctly and the target-agnostic `correlation_diff` is still read from the binary confirmation. Four narrative strings stated v1 conclusions as fact -- that the Clearance artefact "resurfaces exactly where predicted", and that construction coupling is "the single most important caveat this portal carries". Both are now wrong; replaced with the measured v1->v2 comparison. |
| `generate_feature_lock_report_xt.py` | RE-POINT + declared label corrections | Renders correctly from the v2 JSON. Three card LABELS are literals in the renderer and state v1 conclusions ("Prompt 64 -- CONFIRMED / symmetric", "Prompt 64 -- RESURFACED", "confirmed fixed"). Corrected by `generate_feature_lock_report_xt_v2.py` with an assertion per label. |
| `generate_master_findings_xt.py` (this document) | **REWRITE** -> `generate_master_findings_xt_v2.py` | Almost every sentence is a Prompt-65 conclusion and about half concern findings v2 reversed. The block model is imported unchanged so .md and .html still cannot drift. New section 1 runs BEFORE the Step-0 table so a cold reader learns this supersedes a prior pass up front. |
| `generate_eda_portal_xt.py` | **REWRITE** -> `generate_eda_portal_xt_v2.py` | Portal CSS/JS/shell imported and reused unchanged, so this portal looks identical to the v1 and xg ones. Sidebar names `target_xt_delta_v2`, a "Superseded" group links to the v1 portal, and the footer states the supersession rather than presenting v2 as if it had always existed. This file also gained a new role: it is the DRIVER that imports `xt_v2_common` before any generator, which is what makes the re-point work at all. |

### The rewrites, in one line each

- **Premise falsified.** The Leakage Audit's Part D' and the Confound suite's Test 2 were both built on v1's Clearance/`action_x` mechanism. v2 removes that mechanism, so neither question is about anything that still exists. Re-measured and re-framed respectively.
- **Conclusion inverted.** The Category Atlas's Clearance note, the Feature Lock Confirmation's pattern findings, and this document all asserted findings that v2 reverses.
- **Shape descriptor wrong.** "Roughly symmetric" and "near-zero, NEGATIVE mean" appear across the suite. v2 is left-skewed with a positive mean.
- **Shell and provenance.** The portal index and this document had to state the supersession rather than silently replace the v1 portal.

### Scope gaps -- unchanged from v1, restated rather than quietly dropped

None of these is a missing script; all three are SCOPE gaps that the v2 correction does not touch, and each is reported inside the affected report as well as here:

- **Leakage Audit.** The xg and binary leakage audits are passive-dataset-only. No active-side leakage audit exists in this repo for any target. Part A is reused by import; Parts B/C/D are reported as having no active counterpart; B'/C' substitute the active dataset's own analogues.
- **Confound Testing.** All five xg-portal tests are passive. Exactly one active confound test exists project-wide, in the binary portal only. Test 1 mirrors it; Tests 2 and 3 are this portal's own.
- **Slice Stratification V2.** Its archetype slicer is a passive column, so V2's closing archetype-vs-boolean comparison still cannot be reproduced on the active leg. Reported as not applicable rather than replaced with a different comparison that would read like the same conclusion.

### A finding the Flag Ledger still cannot surface -- same gap, different reason to care

As in the v1 portal, none of the three `action_*_possession` flags appears as a Flag Ledger row: the ledger is built from the reconstructed pre-drop 51-era pool and all three were out of the candidate list before that snapshot. The pool logic is mirrored unchanged from `generate_reports_xg.py`, so it structurally cannot show them, and the same `boolean_xt_lift` is run OFF-POOL so the finding is not lost to a technicality. What changed is what the off-pool numbers mean: under v1 they showed the flags were no longer degenerate; under v2 they show the flags are **construction inputs** to the target.

## 3. The target: what target_xt_delta_v2 is and what shape it has

`target_xt_delta_v2` = `xt_before - xt_after`, an Expected-Threat value-delta for the active-binary leg. It lives in `outputs/prototypes/active_binary_xt_delta_v2.parquet`, row-aligned to the locked `player_defensive_actions.parquet` by `event_id`. Every report in this portal JOINS the two in memory and never writes either back.

| Statistic | v1 | v2 (this portal) | Meaning |
| --- | --- | --- | --- |
| rows | 56,068 | 56,068 | the full active-binary leg |
| defined | 56,036 | 55,761 | 307 NaN: 32 from `xt_before` (unchanged from v1) plus 277 from `xt_after`'s next-event lookup. Not zero-filled. |
| mean | -0.001316 | +0.003381 | **flips sign** -- active defensive actions now deny threat on average |
| std | 0.064986 | 0.058385 | the scale-setting statistic, and see the caveat below |
| skew | +0.093 | -0.306 | from near-symmetric to mildly LEFT-skewed |
| excess kurtosis | 8.85 | 18.33 | much heavier tails -- from the 954 injected shot-xG rows |
| share exactly zero | 11.31% | 20.15% | next same-possession event lands in the same grid cell |
| share negative | 47.00% | 35.43% | xT ROSE across the action |
| share positive | 41.69% | 44.42% | xT FELL (threat reduced) |
| IQR | 0.019119 | 0.008829 | the bulk more than halved while the tails grew |
| MAD | 0.010277 | 0.003853 | same story, robustly measured |

> **why std and IQR move in opposite directions -- the whole threshold problem in one line** -- The standard deviation fell only 10% while the IQR fell 54%. Both are describing the same distribution. The gap is the 954 rows where a Shot follows the defensive action and `xt_after` takes that shot's own xG -- up to 0.897, against the xT grid's maximum cell of 0.2575. Those rows inflate std without touching the middle. Any threshold derived from std therefore describes v2's bulk much worse than it described v1's.

## 4. What methodology was adapted, and what did not transfer

Prompt 65 made three adaptations to the xg_target suite's methodology, each stated in every report it touched. All three are re-examined here against v2 rather than inherited. Two survive unchanged in method; one keeps its mechanism but loses its original justification and gains a warning.

### Adaptation 1 -- the scale-setting statistic (JUSTIFICATION RESTATED, WARNING ADDED)

ADAPTATION 1, RE-EXAMINED FOR v2. Prompt 65 translated the xg suite's mean-multiple threshold convention onto the xT target via xg's own standard-deviation fraction, because v1's mean was near-zero and NEGATIVE (-0.001316), which would have flipped the sign of every threshold. v2's mean is now POSITIVE (+0.003381), so that literal justification no longer applies -- checked, not assumed. The translation is KEPT anyway, for a reason that does still hold: at +0.003381 against a std of 0.058385 the mean is only 5.8% of one standard deviation, so a mean-multiple threshold would still be an arbitrary sliver, and keeping the same translation is what makes this portal numerically comparable with the v1 portal it supersedes. Mechanically: xg's own flat_margin (0.5 x mean xg = 0.004055 on these same rows) re-expressed as a fraction of xg's own std (0.048033) gives 0.084425, applied to target_xt_delta_v2's std (0.058385) to give 0.004929. Every number is computed at runtime from the real parquet. See `threshold_transfer_warning` in this same block: the std-derived margin is materially less appropriate on v2 than it was on v1, and that is stated rather than hidden. Scale-free thresholds (|Spearman rho| >= 0.05, small-n row counts, the substitutive/additive ratio cutoffs) are carried over completely unchanged and are unaffected.

| Quantity | v1 | v2 on this run |
| --- | --- | --- |
| xg flat margin on the same rows (0.5 x mean xg) | 0.004055 | 0.004055 |
| as a fraction of xg's own std | 0.084425 | 0.084425 |
| translated flat margin | 0.005486 | 0.004929 |
| translated consistency range trigger | 0.010973 | 0.009858 |
| translated review near-identical threshold | 0.005486 | 0.004929 |
| translated review distinct-signal threshold | 0.016459 | 0.014787 |
| flat margin on the non-zero-delta subset | 0.005826 | 0.005514 |
| **flat margin as a multiple of MAD** | **0.53x** | **1.28x** |
| **share of all rows inside the flat margin** | **35.4%** | **53.4%** |
| robust (MAD-derived) alternative margin | 0.001286 | 0.000482 |
| _r vs the old `target_future_xg_10s`, measured on these rows_ | -0.016 | **-0.1082** |

That last row is recorded here rather than in the findings section on purpose. **Adaptation 1 is the one place this entire suite still reaches for the old target** -- it borrows `target_future_xg_10s`'s own standard-deviation fraction to set every threshold in every report. So wherever a report in this portal cites an xg-derived threshold, this is how the two targets actually relate on the same rows. Prompt 66's figure of -0.108 reproduces here at -0.1082 over n=55,761. It is an order of magnitude stronger than v1's -0.016 and, unlike v1's, correctly signed -- more threat denied now goes with less future attacking output. **It is still modest, and that is expected rather than disappointing:** the two targets measure genuinely different things (an instantaneous value swing versus a forward-looking 10-second outcome), so a strong correlation would have been evidence of redundancy, not of validity. This block is published in every report's `target_scale.correlation_with_old_target`.

> **THIS THRESHOLD DOES NOT TRANSFER CLEANLY -- the explicit finding, not a footnote** -- THIS THRESHOLD DOES NOT TRANSFER CLEANLY FROM v1, and that is reported rather than forced to fit. Prompt 65 tuned its translated flat_margin against a target with excess kurtosis 8.85, an IQR of 0.019119 and a MAD of 0.010277; the resulting margin (0.005486) was 0.53 x that MAD and spanned 35.4% of rows -- a genuinely narrow 'too small to matter' band. v2's tails are far heavier (excess kurtosis 18.3289) because Prompt 66 injects real shot xG values up to 0.897 for the 954 rows where a Shot follows the defensive action, while v2's BULK is much tighter (IQR 0.008829, MAD 0.003853). Because the translation is std-derived and std is inflated by exactly those tails, the margin barely moves (0.004929) while the distribution it is meant to describe more than halved: it is now 1.28 x the MAD, 0.56 x the IQR, and spans 53.4% of all rows. A flat band covering the majority of the data is not doing its job. The literal translated margin is KEPT as the headline number so v1-vs-v2 comparisons stay like-for-like and the xg suite's convention is still the one being followed, but the MAD-derived robust alternative (0.000482) is published beside it on every run and every shape classification in this suite should be read with this inflation in mind.

Scale-free thresholds were carried over **completely unchanged**, and were checked rather than assumed to be scale-free: |Spearman rho| &ge; 0.05, small-n ROW COUNTS (500 and 1000), the substitutive (0.3) and additive (2.0) ratio cutoffs, `MIN_CELL_N_FOR_DELTA=20`, `QCUT_N=4`, 10 quantile deciles, and the discrete-cardinality cutoff of 15. None of these can be distorted by kurtosis: they are counts, ranks or ratios. The confound suite's `_verdict()` is in the same category -- it compares only signs.

### Adaptation 2 -- the conditional panel (UNCHANGED in method)

ADAPTATION 2, unchanged in method and re-measured on v2. The xg_target suite's second panel conditions on target_future_shot_10s == 1 to strip the structural zero mass. target_xt_delta_v2 has no shot column behind it; its structural-zero analogue is the 20.2% of rows with an EXACTLY zero delta. That share nearly doubled from v1's 11.3%, for an understood reason: under v2 a delta is exactly zero whenever the next same-possession event lands in the same xT grid cell as the previous one, which is far more common than v1's 'the action's own location shares a cell with the previous event'. The conditional panel throughout this suite remains 'given a non-zero delta' (target_xt_delta_v2 != 0), never 'given a shot'.

The knock-on rename Prompt 65 introduced is kept: the slice-stratification reports' `occurrence_vs_quality` field encodes an xg-specific distinction that does not exist for this target, and remains `zero_mass_vs_magnitude`.

### Adaptation 3 -- two-directional framing (UNCHANGED in method, MORE load-bearing)

ADAPTATION 3, unchanged in method and re-measured on v2. Sign carries the football meaning: POSITIVE target_xt_delta_v2 = xT fell across the action (threat reduced, good for the defence), NEGATIVE = xT rose. On v2 the balance shifts markedly towards the defence -- 44.4% positive vs 35.4% negative, where v1 was 41.7% / 47.0% -- but the target is NO LONGER close to symmetric (skew -0.306, vs v1's +0.093), so two-directional framing matters more here, not less. Rankings are by |magnitude| across BOTH directions, and every table also reports the negative/positive/zero share behind its mean.

The four concrete consequences are unchanged and still apply: diverging bar charts instead of left-anchored fills; a dashed zero line on every sparkline; diverging green/red interaction heatmaps; and the two fields with no xg counterpart -- whether a binned curve CROSSES zero, and whether within-stratum deltas change SIGN.

### Not an adaptation -- log-transforms

NOT an adaptation, re-checked rather than inherited: the xg_target report suite never log-transforms its target anywhere, and neither did Prompt 65's v1 xT suite. The log1p framing in this project belongs to the modelling legs, not to this report suite. v2's heavier tails make a transform more tempting than it was on v1 -- and it is still NOT applied here, because doing so would silently break comparability with both the xg portal and the v1 xT portal this one supersedes. Recorded as an open question for the modelling legs, not resolved by an EDA suite.

## 5. Findings

### Prompt 66's findings -- where each one resurfaced in this portal

> **Clearance reversal -- CONFIRMED in the Category Atlas, and tested twice in the Confound report** -- Prompt 64's Clearance/action_x artefact does NOT resurface -- it reverses, exactly as Prompt 66 reported. Clearance ranks 1 of 8 event_type categories at +0.024706 mean delta over n=4,096 rows (38.8% negative). The v1 edition of this confirmation recorded the same category at 8 of 8 and -0.032482, and flagged it as a measurement artefact of the xt_after definition. Under the corrected xt_after the sign flips and the rank inverts. Confirmed independently on this run, not quoted.

The Category Atlas is where Prompt 66's headline number had to reappear, and it does, independently recomputed: Clearance at **+0.024706** over n=4,096, ranked **1 of 8 event_type categories**. Prompt 66 reported +0.0247 and 1st of 8 -- matched exactly. The Flag Ledger carries the same correction from the other direction, through the possession flags. Two confound tests then ask whether the new positive mean is real:

| Confound test | Unconditional | Given a non-zero delta | New? |
| --- | --- | --- | --- |
| defenders_within_10m vs box proximity | partially | partially | no |
| Clearance's POSITIVE mean delta vs defending-box proximity | no | no | no |
| Clearance's positive mean delta vs the xt_after = 0 construction rule | no | no | yes |

Both Clearance tests return **no** -- the pattern does **not** reverse within either stratum. Read plainly: Clearance's positive mean survives conditioning on defending-box proximity, and it survives conditioning on whether the action ended its possession. **That second result is the one worth having.** It was the obvious way the correction could have been self-fulfilling -- clearances end possessions more often than average, possession-enders get `xt_after = 0` by construction, so Clearance could have looked good purely as an artefact of the fix. It does not: the effect holds inside the stratum where the construction rule does not apply.

> **distribution shape -- CONFIRMED in the Distribution Atlas** -- Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.3): target_xt_delta_v2's mean flips from v1's slightly negative -0.00132 to a clearly positive +0.00338 -- on average, active defensive actions now genuinely deny threat, the football-sensible direction v1 never achieved. But the shape got HARDER, not easier: skew moves from +0.093 (near-symmetric) to -0.306 (mildly left-skewed) and excess kurtosis from 8.85 to 18.33 (much heavier tails), while the exactly-zero share rises from 11.3% to 20.2% and the negative share falls from 47.0% to 35.4%. The heavier tails are a direct, understood consequence of injecting real shot_statsbomb_xg values (up to 0.897) as xt_after for the 954 rows -- 1.9% of continuing-possession rows -- where a Shot immediately follows the defensive action; the 96-cell xT grid's own maximum cell is only 0.2575, so it could never produce such values. This REINFORCES rather than resolves Prompt 64's open flag that the existing hurdle architecture does not fit this target's shape as-is: v2 is if anything less like that shape than v1 was.

> **possession-ending collapse -- CONFIRMED in the Flag Ledger and Leakage Audit Part C'** -- Prompt 66 finding carried forward (outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md section 2.1): on the two possession-ending slices, target_xt_delta_v2 collapses EXACTLY onto xt_before's own distribution -- action_ended_possession == True (n=4,656) mean +0.03051, action_won_possession == True (n=3,600) mean +0.03066, both confirmed identical to the same slice's xt_before mean with np.allclose. This is logically REQUIRED, not a new degenerate finding: xt_after is forced to 0.0 for exactly these rows, so the delta is xt_before by construction -- '100% of the threat that existed a moment ago is denied'. Unlike the old target_future_shot_10s, which collapsed these same rows to an identical uninformative zero, this collapses to a real, row-by-row varying distribution (mean +0.031, std ~0.050): the actual danger level that was denied. Any report that shows these slices must present the identity as construction, not as discovery.

### The possession flags, measured off-pool

| Flag | n (True) | mean delta (True) | % exactly zero (True) | lift |
| --- | --- | --- | --- | --- |
| `action_changed_possession` | 4,656 | +0.030515 | 0.0% | +0.029605 |
| `action_ended_possession` | 4,656 | +0.030515 | 0.0% | +0.029605 |
| `action_won_possession` | 3,600 | +0.030656 | 0.0% | +0.029157 |

All three True groups were confirmed on this run to equal their own `xt_before` exactly (`np.allclose`). **These columns stay excluded in `feature_config.py` and nothing in this portal changes that** -- but the reason has strengthened. v1 left an open question about unlocking them, since their stated exclusion reason is target-specific. v2 closes that question the other way: a column that deterministically sets one term of the target is a construction input, and modelling on it would be circular.

### Construction coupling -- the v1 caveat, re-measured

Strongest locked-feature correlation with the target: `distance_to_attacking_box` at r=+0.5551 under v1, falling to `distance_to_attacking_box` at r=+0.1431 under v2 -- **74% lower**. The same feature sits at r=+0.2652 against `xt_before`, which is unchanged from v1 and is where the residual, ordinary location relationship now lives.

### Numerical features vs the target

21/31 features' binned curves cross zero -- they separate threat-reducing from threat-increasing actions rather than only varying in magnitude. 2 feature(s) are flagged train/test-inconsistent. **Note the collapse in rho relative to v1**, which is the same construction-coupling story: v1's top five all sat above |rho| = 0.75.

| Feature | Spearman rho | Shape | Curve crosses zero |
| --- | --- | --- | --- |
| `attacker_centroid_x` | -0.229 | monotonic decreasing | True |
| `defender_centroid_x` | -0.2271 | monotonic decreasing | True |
| `distance_to_attacking_box` | 0.1689 | monotonic increasing | True |
| `distance_to_attacking_goal` | 0.168 | monotonic increasing | True |
| `distance_to_defending_box` | -0.1675 | monotonic decreasing | True |

> **read these shape labels with the section-1 caveat in hand** -- 24 of these 31 features classify differently under the MAD-derived flat margin than under the std-derived one used above (against 15 of 31 on v1). The `Shape` column is indicative on this target, not decided.

### Flag Ledger -- top 5 by |lift|, both directions

| Flag | lift | mean True | mean False | n True |
| --- | --- | --- | --- | --- |
| `has_visible_attacker` | -0.024969 | +0.003367 | +0.028337 | 55,730 |
| `is_in_attacking_box` | -0.021682 | -0.016034 | +0.005648 | 5,830 |
| `is_in_defending_box` | +0.018122 | +0.019947 | +0.001825 | 4,790 |
| `is_deep_zone` | +0.016512 | +0.016529 | +0.000018 | 11,359 |
| `has_previous_event` | -0.016494 | +0.002045 | +0.018539 | 51,244 |

### Tournament stability

3/6 active features show a genuine tournament-level difference rather than arbitrary train/test noise.

| Feature | Verdict |
| --- | --- |
| `defenders_within_10m` | genuine tournament-level difference |
| `defenders_within_5m` | arbitrary train/test noise |
| `events_elapsed_in_possession` | arbitrary train/test noise |
| `local_numerical_balance_5m` | genuine tournament-level difference |
| `match_time_seconds` | arbitrary train/test noise |
| `attackers_within_5m` | genuine tournament-level difference |

### Slice stratification

V1: 72 cells, 300 genuine divergences unconditionally and 285 once the zero mass is removed, with 24 categories where that removal flips the conclusion. V2: 108 cells, 147 / 151.

### Numeric x numeric interaction

5/5 pairs classify the same way unconditionally and on the non-zero-delta subset. 2/5 have within-stratum deltas that change SIGN -- feature A's effect reverses direction across levels of feature B. That is a distinct finding from 'interactive' and cannot arise on the xG target at all.

| Feature A | Feature B | Classification | Stratum deltas change sign |
| --- | --- | --- | --- |
| `visible_defender_count` | `attacker_spread` | interactive | no |
| `defenders_within_5m` | `defenders_within_10m` | interactive | no |
| `possession_elapsed_seconds` | `match_time_seconds` | additive | no |
| `defenders_within_10m` | `distance_to_attacking_box` | inconclusive | yes |
| `defenders_between_ball_and_attacking_goal` | `attacker_defender_ratio` | inconclusive | yes |

### Review tier

57 ACTIVE review pairs; 35 remain needs_human_call (dominated, as on every target, by continuous-continuous pairs no rule covers); 1 Type-2 pair(s) resolve on the opposite-sign rule, which carries more weight here than on xg -- a sign disagreement means the two flags disagree about the DIRECTION of the threat swing, not merely about being above or below a base rate.

## 6. Caveats recorded, and open questions this portal does not answer

**Nothing in this portal changes `feature_config.py`, and no model artefact was touched.** No feature is added, dropped or re-tiered.

### Caveat 1 -- construction coupling: SUBSTANTIALLY RETIRED (was v1's most important caveat)

No feature is dropped on this evidence, and none is proposed for dropping -- the locked set stays as-is, exactly as under v1. What changed is the weight of the caveat attached to reading the Numerical Target Atlas. v1 recorded construction coupling as the single most important caveat its portal carried, with no xG or binary counterpart: because xt_after was looked up from action_x/action_y, location-derived locked features shared a definitional term with half the target, and the strongest sat at r=+0.5551. Prompt 66 removed that mechanism entirely, and the strongest locked-feature correlation falls to r=+0.1431 (74% lower). The caveat is substantially RETIRED rather than carried forward. A residual location relationship remains through xt_before, which is unchanged from v1 and is still a grid lookup at the previous event's location -- that is ordinary and expected, not recovery-by-definition of half the target.

### Caveat 2 -- this target's effects can sit in its zero mass: SURVIVES, and widened

A mean-difference test alone -- the instrument the xg audit reaches for first -- can miss an effect that lives in the exactly-zero share. v1 recorded this via `has_previous_event`, where the mean test returned p=0.0645 and would have reported nothing while the zero share differed by +27.3pp. The caveat is not retired by the correction, and the zero share it concerns grew from 11.3% to 20.2% of rows.

### Caveat 3 -- NEW: the flat-margin threshold no longer describes this distribution

Covered in full in sections 1 and 4. In short: the std-derived translated margin now spans the majority of rows, 24 of 31 Numerical Atlas features classify differently under a robust alternative, and shape labels in this portal are correspondingly less decisive than v1's. This is reported rather than papered over, and the threshold was not silently swapped, because doing so would have made every v2 cell incomparable with the v1 portal this one supersedes.

> **methodological warning for anyone extending this suite** -- v2 is a harder distribution to work with than v1, not an easier one. Its mean moved in the football-sensible direction, which is the headline improvement -- but its skew, kurtosis and zero share all moved AWAY from anything a standard-deviation-based convention handles well. Prefer rank-based, count-based, ratio-based and sign-based instruments on this target; they were the ones that transferred from v1 without a scratch, and the one instrument that did not transfer is the only std-based one in the suite.

### Open questions this portal does NOT answer

- Whether `Block`, now the lowest-mean event type at -0.0185, is itself an artefact. Prompt 66 recorded the handover from Clearance and explicitly left it unchased as out of scope; this portal surfaces it in the Category Atlas and takes no position.
- Whether the three `action_*_possession` flags should be formally re-documented in `feature_config.py` as CONSTRUCTION INPUTS rather than structural zeros. The stated exclusion reason is wrong for this target in both v1 and v2; the exclusion itself is right. Changing the recorded reason is a `feature_config.py` edit and is out of this suite's remit.
- What model architecture suits a left-skewed, very heavy-tailed, sign-carrying target. Prompt 66 found v2 is _less_ like the existing hurdle architecture's shape than v1 was, not more. This is an EDA portal and takes no position; no model artefact was touched.
- Whether `xt_before` should also be redefined. Prompt 66 scoped its correction to `xt_after` only, and `xt_before` is still a grid lookup at the previous event's location with no period-boundary restriction -- unlike the corrected `xt_after`, which respects them. That asymmetry is recorded, not resolved.
- The passive leg, entirely. Out of scope for this work.

## 7. Document map

### Built here (target-dependent -- rebuilt against target_xt_delta_v2)

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

These seven are properties of the locked ACTIVE features and are unaffected by which target the features are measured against. **A correction to `xt_after` cannot move a feature-vs-feature statistic**, so rebuilding them would have produced byte-identical output. The portal index links out to the existing files in `../xg_target/` and `../shot_target/`.

- Correlation Atlas V1 / V2 / V3 -- correlation clustering is feature-vs-feature and never references a target column.
- VIF Analysis -- multicollinearity is a property of the feature matrix alone.
- Slicer Redundancy Check -- slicer-vs-slicer, no target term.
- Player-Level Validity Check -- row concentration and train/test player overlap, no target term.
- Player-Grouped Split Check -- an identity-leakage stress test on the split itself.

### The superseded v1 portal

`reports/analysis/xt_target_v1_superseded/` holds Prompt 65's complete portal against `target_xt_delta` (v1), moved there with `git mv` so its history is preserved. **Its contents are unchanged apart from a supersession banner on its index and master findings.** Every number in it is the accurate record of what was measured against v1 and is deliberately not corrected.

### New generator scripts (all added as siblings; no Prompt-65 script was modified)

Nothing under `src/eda/` that already existed was edited. Prompt 65's 21 `*_xt.py` generators, and the `xg_target` and `shot_target` portals' own generators, all remain exactly reproducible.

- `src/eda/xt_v2_common.py` -- the re-point layer, the re-examined Adaptation 1, and Prompt 66's carried-forward finding strings
- `src/eda/generate_reports_xt_v2.py` -- Category Atlas + Flag Ledger
- `src/eda/generate_leakage_audit_xt_v2.py` + `generate_leakage_report_xt_v2.py`
- `src/eda/generate_confound_analysis_xt_v2.py`
- `src/eda/generate_feature_lock_confirmation_xt_v2.py` + `generate_feature_lock_report_xt_v2.py`
- `src/eda/xt_v2_prose_fixups.py` -- declared, asserted corrections to five stale sentences in re-pointed outputs
- `src/eda/generate_master_findings_xt_v2.py` (this document)
- `src/eda/generate_eda_portal_xt_v2.py` -- portal shell AND the driver that makes the re-point work

Rebuild the whole portal with: `.venv/Scripts/python.exe -m src.eda.generate_eda_portal_xt_v2 --all`
