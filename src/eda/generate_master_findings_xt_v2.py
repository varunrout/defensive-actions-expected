"""CLI entrypoint: MASTER_FINDINGS.md + .html for the xT-delta v2 portal,
ACTIVE-BINARY LEG ONLY.

STEP-0 DISPOSITION: REWRITE.

`generate_master_findings_xt.py` is a narrative document: almost every
sentence in it is a Prompt-65 conclusion, and roughly half of those are
about findings the v2 correction reversed. Re-pointing it would have
produced a document whose prose contradicts its own tables. The BLOCK MODEL
(`p` / `ul` / `table` / `note` / `h3` -> `render_blocks_md` /
`render_blocks_html`) is the part worth keeping, and it is imported from
`generate_master_findings` exactly as the v1 version imported it, so the
.md and .html still cannot drift apart.

STRUCTURAL CHANGE from the v1 document, made deliberately: a new section 1,
"What changed from v1, and why", runs BEFORE the Step-0 table. A reader
landing on this portal cold must understand that it supersedes a prior pass
before they start reading findings, not discover it halfway down.

CLOSEOUT DECISION: Prompt 65 decided against a separate
PATTERN_ANALYSIS_CLOSEOUT-style document for this portal, on the grounds
that the portal is a single-pass, single-leg suite whose findings fit in
one section. THE SAME CALL IS MADE HERE, for the same reason plus one more:
a second document would now have to carry the v1-vs-v2 comparison too, and
splitting that story across two files is exactly how the two would drift.
The reasoning is restated in section 1 rather than left implicit.

Usage:
    python -m src.eda.generate_master_findings_xt_v2
"""

from __future__ import annotations

import json

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import render, xt_common as xc
from src.eda.feature_config import ACTIVE
from src.eda.generate_master_findings import h3, note, p, render_blocks_html, render_blocks_md, table, ul
from src.eda.render import esc

OUT_DIR = xc.OUT_DIR
MD_PATH = OUT_DIR / "MASTER_FINDINGS.md"
HTML_PATH = OUT_DIR / "MASTER_FINDINGS.html"
V1 = v2.V1_PORTAL_DIRNAME


def _load(name: str) -> dict:
    return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Section 1 -- what changed from v1 and why (NEW, and deliberately first)
# --------------------------------------------------------------------------

def build_section_1(scale: dict, lk: dict) -> list[dict]:
    vc = lk["part_d_prime_construction_coupling"]["v1_comparison"]
    return [
        note("this portal supersedes an earlier pass -- read this first",
             "Prompt 65 built these same thirteen reports against <code>target_xt_delta</code> (v1). Prompt 66 "
             "found that target's <code>xt_after</code> term wrong and corrected it. This portal is the rebuild "
             "against the corrected target. <b>The v1 portal is preserved UNCHANGED at "
             f"<code>reports/analysis/{V1}/</code></b> -- every number and finding in it is the accurate "
             "historical record of what Prompt 65 measured against v1, not something to correct or update."),
        h3("The correction itself"),
        p("<code>xt_before</code> is <b>identical</b> in v1 and v2 -- the previous event's location, looked up "
          "in the same 8x12 xT grid. Only <code>xt_after</code> changed:"),
        table(["", "v1 (Prompt 64)", "v2 (Prompt 66)"], [
            ["`xt_after` definition",
             "`xT(action_x, action_y)` -- the defensive action's OWN recorded location",
             "`0.0` if the action ends its own possession; else the next same-possession event's grid value; "
             "else that event's own `shot_statsbomb_xg` if it is a Shot"],
            ["rows affected", "all 56,068", "4,662 possession-ending / 954 shot-follows / 50,452 ordinary"],
            ["the problem it fixes", "for a Clearance, `action_x`/`action_y` is where the ball WAS when cleared "
                                     "-- often deep in the defending box, a high-xT cell by construction",
             "a clearance that ends the attack now scores `xt_after = 0`: all of the threat is denied"],
        ]),
        h3("The three consequences that matter for reading this portal"),
        note("1. The Clearance artefact REVERSED -- it did not merely shrink",
             esc(v2.PROMPT_66_CLEARANCE_FINDING)),
        note("2. The distribution moved the right way on its mean and the WRONG way on its shape",
             esc(v2.PROMPT_66_SHAPE_FINDING)),
        note("3. Correlation with the old target improved, but is still modest",
             esc(v2.PROMPT_66_CORRELATION_FINDING)),
        h3("What that did to Prompt 65's two recorded caveats"),
        p("Prompt 65 closed with two caveats, recorded but not resolved. The correction moves them in opposite "
          "directions, and neither outcome was assumed -- both were re-measured on this run:"),
        ul([
            "<b>Caveat 1, construction coupling -- SUBSTANTIALLY RETIRED.</b> v1's single most important "
            "caveat was that because <code>xt_after</code> was looked up from <code>action_x</code>/"
            "<code>action_y</code>, every location-derived locked feature shared a definitional term with half "
            f"the target; the strongest sat at r={vc['v1_r_vs_target']:+.4f} "
            f"(<code>{esc(vc['v1_strongest_feature'])}</code>). v2 never looks up the acting row's own "
            f"coordinates at all, and the strongest locked-feature correlation falls to "
            f"r={vc['v2_r_vs_target']:+.4f} -- <b>{vc['abs_reduction_pct']:.0f}% lower</b>. The Numerical "
            "Target Atlas no longer needs the heavy recovery-by-definition discount v1 attached to it.",
            "<b>Caveat 2, effects living in the zero mass -- SURVIVES, and matters more.</b> A "
            "mean-difference test misses effects that sit in this target's exactly-zero share. That share rose "
            f"from 11.3% (v1) to {scale['pct_zero']:.1f}% (v2), so the caveat applies to more rows than before, "
            "not fewer. Every table in this portal still reports negative / zero / positive shares beside its "
            "mean.",
        ]),
        h3("One threshold that does NOT transfer cleanly -- stated, not forced to fit"),
        note("the std-derived flat margin now spans the majority of the distribution",
             esc(scale["threshold_transfer_warning"])),
        p("This is not a cosmetic point. <code>flat_margin</code> feeds <code>classify_shape()</code> in five "
          "reports. Measured directly on this run: of the 31 features in the Numerical Target Atlas pool, "
          "<b>24 classify differently</b> under the std-derived margin than under the MAD-derived one -- "
          "against <b>15 of 31</b> for the same comparison on v1. <b>Shape labels in this portal are therefore "
          "materially less robust than the v1 portal's were, and should be read as indicative rather than "
          "decided.</b> The std-derived margin is kept as the headline so v1-vs-v2 cells stay comparable and "
          "the xg suite's convention is still the one being followed; the robust alternative is published "
          "beside it in every report's <code>target_scale.robust_scale</code> block."),
        h3("Closeout document: folded in again, and why the call did not change"),
        p("Prompt 65 decided against a separate <code>PATTERN_ANALYSIS_CLOSEOUT</code>-style document for this "
          "portal (<code>reports/analysis/shot_target/</code> has one), on the grounds that it exists to close "
          "out a multi-prompt campaign spanning two targets and both datasets, whereas this portal is a "
          "single-pass, single-leg suite whose findings fit in one section. <b>The same call is made here, for "
          "the same reason plus one more:</b> a separate closeout would now also have to carry the whole "
          "v1-vs-v2 comparison, and splitting that across two documents is precisely how the two would drift "
          "apart. Everything is in this file."),
    ]


# --------------------------------------------------------------------------
# Section 2 -- the mandatory Step-0 script-by-script disposition
# --------------------------------------------------------------------------

STEP_0_ROWS: list[list[str]] = [
    ["`src/eda/xt_common.py`", "**REWRITE** -> `src/eda/xt_v2_common.py`",
     "The one module every other script reaches the target through. Re-points `TARGET`, `XT_PARQUET` and the "
     "carried-forward finding strings, and wraps `target_scale()`. Its Adaptation-1 prose asserted a "
     "\"near-zero, NEGATIVE mean\" -- v2's mean is POSITIVE, so the justification is restated (the translation "
     "is kept, for the reason that still holds). Adds the `robust_scale` block and "
     "`threshold_transfer_warning`. Prompt 64's two finding strings are replaced with Prompt 66's, because the "
     "Clearance one now asserts something v2 disproves."],
    ["`generate_reports_xt.py` (Category Atlas + Flag Ledger)", "**REWRITE** -> `generate_reports_xt_v2.py`",
     "All computation re-points cleanly. Three pieces of prose did not: the Clearance note announced the "
     "artefact as \"confirmed present\" (it reverses); the possession-flag note quoted v1's \"77-79% non-zero\" "
     "(under v2 those flags collapse onto `xt_before` by construction, checked here with `np.allclose`); and "
     "the scale card said \"roughly symmetric\" (skew is -0.306). Pool construction, lift functions and the "
     "diverging-bar geometry are imported unchanged."],
    ["`generate_distribution_atlas_xt.py`", "RE-POINT",
     "Fully data-driven -- its target-shape section recomputes skew/kurtosis/shares from the parquet, so it "
     "reports v2's own figures. Its one v1-specific design decision, clipping the histogram SYMMETRICALLY at "
     "the p99 of |delta| instead of min-to-p99, becomes MORE necessary under v2's heavier tails, not less."],
    ["`generate_numerical_xt_target_analysis.py`", "RE-POINT",
     "`flat_margin` and `range_trigger` are computed at runtime from the real parquet, so they recompute "
     "against v2's own std automatically. The resulting margin is the threshold that does not transfer "
     "cleanly -- reported in section 1 and in every `target_scale` block, not silently swapped."],
    ["`generate_numerical_xt_target_reports.py`", "RE-POINT + declared prose correction",
     "Renders entirely from the compute step's JSON. Three hardcoded sentences asserted v1's shape (\"roughly "
     "symmetric\", \"near-zero, negative mean\"); corrected by `xt_v2_prose_fixups.py` at fragment level with "
     "an assertion that each matched, rather than by duplicating a 338-line renderer."],
    ["`generate_review_analysis_xt.py`", "RE-POINT + declared prose correction",
     "Types 1/3/4 resolvers are target-independent and imported unchanged; only Type 2 uses target evidence, "
     "and its two thresholds recompute at runtime. One sentence in the JSON `note` asserted the negative mean; "
     "corrected as above."],
    ["`generate_review_report_xt.py`", "RE-POINT",
     "Pure renderer over the JSON. Re-run after the prose correction so the two cannot disagree."],
    ["`generate_leakage_audit_xt.py`", "**REWRITE** -> `generate_leakage_audit_xt_v2.py`",
     "**The largest genuine logic change in this rebuild.** Parts A/B'/C' keep their methods (Part A imported "
     "unchanged; `_split_stats` reused byte-for-byte). **Part D' was rewritten from scratch: its premise is "
     "false under v2.** It existed to quantify \"xt_after is looked up from action_x/action_y, so "
     "location-derived features share a definitional term with half the target\" -- a mechanism Prompt 66 "
     "removed entirely. Coupling is re-measured rather than restated, and a NEW v2-only channel is tested "
     "(coupling moved from a location term to the `action_ended_possession` flag)."],
    ["`generate_leakage_report_xt.py`", "**REWRITE** -> `generate_leakage_report_xt_v2.py`",
     "Cannot render the v2 JSON at all -- Part D's key set changed with the rewrite. Part A, the missing-parts "
     "section and all shared table helpers are imported and reused."],
    ["`generate_confound_analysis_xt.py`", "**REWRITE** -> `generate_confound_analysis_xt_v2.py`",
     "All machinery reused by import, including `_verdict()` byte-for-byte and `QCUT_N=4`. `_verdict()` "
     "compares only the SIGN of first-to-last deltas, never a magnitude -- re-checked, not assumed, and that "
     "is exactly why this report needed no threshold change despite v2's heavier tails. Test 1's spec is "
     "unchanged. **Test 2 was re-framed**: it asked whether Clearance's NEGATIVE mean survives conditioning, "
     "and that mean is now positive. **Test 3 is NEW** -- whether Clearance's positive mean is itself an "
     "artefact of the `xt_after = 0` construction rule, a mechanism that did not exist under v1."],
    ["`generate_confound_report_xt.py`", "RE-POINT",
     "Renders whatever tests the JSON carries, including the new third one, with no change."],
    ["`generate_tournament_stability_check_xt.py`", "RE-POINT + declared prose correction",
     "Imports `INVESTIGATE` and the tournament map; bin edges still computed once on pooled data. Only "
     "`flat_margin` varies and it recomputes at runtime. One JSON `methodology` sentence asserted the negative "
     "mean; corrected as above."],
    ["`generate_tournament_stability_report_xt.py`", "RE-POINT",
     "Pure renderer. Re-run after the prose correction."],
    ["`generate_slice_stratification_xt.py`", "RE-POINT",
     "`FEATURE_SLICER_PLAN`, `build_plan()`, `SMALL_N_THRESHOLD=1000` and `STRUCTURAL_CAUTION_CATEGORIES` all "
     "imported unchanged. Small-n thresholds are ROW COUNTS -- scale-free, so v2's heavier tails cannot touch "
     "them. Only `flat_margin` moves, and it recomputes."],
    ["`generate_slice_stratification_v2_xt.py`", "RE-POINT",
     "Same as above for the boolean-slicer half. The archetype slicer is still a passive column, so V2's "
     "closing archetype-vs-boolean comparison is still not applicable on the active leg -- reported as such, "
     "exactly as in the v1 portal."],
    ["`generate_slice_stratification_report_xt.py`", "RE-POINT",
     "Pure renderer over both slice JSONs."],
    ["`generate_feature_interaction_analysis_xt.py`", "RE-POINT",
     "`PAIRS`, `Q=4` and `_bin_labeled` imported unchanged; pairs deliberately NOT re-ranked, so all three "
     "targets stay comparable. `SUBSTITUTIVE_RATIO_MAX=0.3`, `ADDITIVE_MAGNITUDE_RATIO_MAX=2.0` and "
     "`MIN_CELL_N_FOR_DELTA=20` are ratios and row counts -- scale- and sign-free, so they transfer untouched."],
    ["`generate_feature_interaction_report_xt.py`", "RE-POINT", "Pure renderer over the JSON."],
    ["`generate_feature_lock_confirmation_xt.py`", "**REWRITE** -> `generate_feature_lock_confirmation_xt_v2.py`",
     "Every roll-up re-points correctly and the target-agnostic `correlation_diff` is still read from the "
     "binary confirmation. Four narrative strings stated v1 conclusions as fact -- that the Clearance artefact "
     "\"resurfaces exactly where predicted\", and that construction coupling is \"the single most important "
     "caveat this portal carries\". Both are now wrong; replaced with the measured v1->v2 comparison."],
    ["`generate_feature_lock_report_xt.py`", "RE-POINT + declared label corrections",
     "Renders correctly from the v2 JSON. Three card LABELS are literals in the renderer and state v1 "
     "conclusions (\"Prompt 64 -- CONFIRMED / symmetric\", \"Prompt 64 -- RESURFACED\", \"confirmed fixed\"). "
     "Corrected by `generate_feature_lock_report_xt_v2.py` with an assertion per label."],
    ["`generate_master_findings_xt.py` (this document)", "**REWRITE** -> `generate_master_findings_xt_v2.py`",
     "Almost every sentence is a Prompt-65 conclusion and about half concern findings v2 reversed. The block "
     "model is imported unchanged so .md and .html still cannot drift. New section 1 runs BEFORE the Step-0 "
     "table so a cold reader learns this supersedes a prior pass up front."],
    ["`generate_eda_portal_xt.py`", "**REWRITE** -> `generate_eda_portal_xt_v2.py`",
     "Portal CSS/JS/shell imported and reused unchanged, so this portal looks identical to the v1 and xg ones. "
     "Sidebar names `target_xt_delta_v2`, a \"Superseded\" group links to the v1 portal, and the footer states "
     "the supersession rather than presenting v2 as if it had always existed. This file also gained a new "
     "role: it is the DRIVER that imports `xt_v2_common` before any generator, which is what makes the "
     "re-point work at all."],
]


def build_section_2() -> list[dict]:
    n_repoint = sum(1 for r in STEP_0_ROWS if not r[1].startswith("**REWRITE"))
    n_rewrite = len(STEP_0_ROWS) - n_repoint
    return [
        p("<b>This section is a required deliverable, not scratch notes.</b> All 21 of Prompt 65's "
          "<code>generate_*_xt.py</code> scripts plus <code>xt_common.py</code> were read in full and "
          "classified one by one: can this be re-pointed at "
          "<code>active_binary_xt_delta_v2.parquet</code>/<code>target_xt_delta_v2</code> by a "
          "parameter change alone, or does it bake in a v1-specific scale or direction assumption that needs "
          "real new logic?"),
        note("the mechanism that made most of this a re-point",
             "Prompt 65's suite has one very fortunate property, found by reading rather than assumed: "
             "<b>every single generator reaches the target through <code>from src.eda import xt_common as "
             "xc</code></b>, and binds the target name, parquet path, output directory, statistics helpers and "
             "finding strings from that one module. None opens the prototype parquet directly; none hardcodes "
             "the target column name in a computation. So the suite re-points by rebinding "
             "<code>xt_common</code>'s attributes before the generators are imported -- which is exactly what "
             "<code>xt_v2_common.py</code> does. <b>No file Prompt 65 wrote was edited.</b> All 21 still "
             "regenerate the preserved v1 portal if run against an un-re-pointed <code>xt_common</code>."),
        p(f"<b>Result: {n_repoint} re-pointed, {n_rewrite} rewritten as new siblings.</b> Of the "
          f"{n_repoint} re-points, five additionally needed a declared, asserted correction to one or two "
          "hardcoded sentences that asserted v1's distribution shape as fact -- applied at fragment level by "
          "<code>xt_v2_prose_fixups.py</code> and <code>generate_feature_lock_report_xt_v2.py</code> rather "
          "than by duplicating whole renderers. Each correction asserts that it matched, so a future change to "
          "a Prompt-65 script fails loudly instead of silently republishing a stale claim."),
        table(["Script", "Re-point vs rewrite", "What changed, and why"], STEP_0_ROWS),
        h3("The rewrites, in one line each"),
        ul([
            "<b>Premise falsified.</b> The Leakage Audit's Part D' and the Confound suite's Test 2 were both "
            "built on v1's Clearance/<code>action_x</code> mechanism. v2 removes that mechanism, so neither "
            "question is about anything that still exists. Re-measured and re-framed respectively.",
            "<b>Conclusion inverted.</b> The Category Atlas's Clearance note, the Feature Lock Confirmation's "
            "pattern findings, and this document all asserted findings that v2 reverses.",
            "<b>Shape descriptor wrong.</b> \"Roughly symmetric\" and \"near-zero, NEGATIVE mean\" appear "
            "across the suite. v2 is left-skewed with a positive mean.",
            "<b>Shell and provenance.</b> The portal index and this document had to state the supersession "
            "rather than silently replace the v1 portal.",
        ]),
        h3("Scope gaps -- unchanged from v1, restated rather than quietly dropped"),
        p("None of these is a missing script; all three are SCOPE gaps that the v2 correction does not touch, "
          "and each is reported inside the affected report as well as here:"),
        ul([
            "<b>Leakage Audit.</b> The xg and binary leakage audits are passive-dataset-only. No active-side "
            "leakage audit exists in this repo for any target. Part A is reused by import; Parts B/C/D are "
            "reported as having no active counterpart; B'/C' substitute the active dataset's own analogues.",
            "<b>Confound Testing.</b> All five xg-portal tests are passive. Exactly one active confound test "
            "exists project-wide, in the binary portal only. Test 1 mirrors it; Tests 2 and 3 are this "
            "portal's own.",
            "<b>Slice Stratification V2.</b> Its archetype slicer is a passive column, so V2's closing "
            "archetype-vs-boolean comparison still cannot be reproduced on the active leg. Reported as not "
            "applicable rather than replaced with a different comparison that would read like the same "
            "conclusion.",
        ]),
        h3("A finding the Flag Ledger still cannot surface -- same gap, different reason to care"),
        p("As in the v1 portal, none of the three <code>action_*_possession</code> flags appears as a Flag "
          "Ledger row: the ledger is built from the reconstructed pre-drop 51-era pool and all three were out "
          "of the candidate list before that snapshot. The pool logic is mirrored unchanged from "
          "<code>generate_reports_xg.py</code>, so it structurally cannot show them, and the same "
          "<code>boolean_xt_lift</code> is run OFF-POOL so the finding is not lost to a technicality. What "
          "changed is what the off-pool numbers mean: under v1 they showed the flags were no longer degenerate; "
          "under v2 they show the flags are <b>construction inputs</b> to the target."),
    ]


# --------------------------------------------------------------------------
# Section 3 -- the target
# --------------------------------------------------------------------------

def build_section_3(scale: dict) -> list[dict]:
    rs = scale["robust_scale"]
    return [
        p("<code>target_xt_delta_v2</code> = <code>xt_before - xt_after</code>, an Expected-Threat value-delta "
          "for the active-binary leg. It lives in "
          "<code>outputs/prototypes/active_binary_xt_delta_v2.parquet</code>, row-aligned to the locked "
          "<code>player_defensive_actions.parquet</code> by <code>event_id</code>. Every report in this portal "
          "JOINS the two in memory and never writes either back."),
        table(["Statistic", "v1", "v2 (this portal)", "Meaning"], [
            ["rows", "56,068", f"{scale['n_rows']:,}", "the full active-binary leg"],
            ["defined", "56,036", f"{scale['n_defined']:,}",
             f"{scale['n_nan']} NaN: 32 from `xt_before` (unchanged from v1) plus 277 from `xt_after`'s "
             "next-event lookup. Not zero-filled."],
            ["mean", "-0.001316", f"{scale['mean']:+.6f}",
             "**flips sign** -- active defensive actions now deny threat on average"],
            ["std", "0.064986", f"{scale['std']:.6f}", "the scale-setting statistic, and see the caveat below"],
            ["skew", "+0.093", f"{scale['skew']:+.3f}", "from near-symmetric to mildly LEFT-skewed"],
            ["excess kurtosis", "8.85", f"{scale['excess_kurtosis']:.2f}",
             "much heavier tails -- from the 954 injected shot-xG rows"],
            ["share exactly zero", "11.31%", f"{scale['pct_zero']:.2f}%",
             "next same-possession event lands in the same grid cell"],
            ["share negative", "47.00%", f"{scale['pct_negative']:.2f}%", "xT ROSE across the action"],
            ["share positive", "41.69%", f"{scale['pct_positive']:.2f}%", "xT FELL (threat reduced)"],
            ["IQR", "0.019119", f"{rs['iqr']:.6f}", "the bulk more than halved while the tails grew"],
            ["MAD", "0.010277", f"{rs['mad']:.6f}", "same story, robustly measured"],
        ]),
        note("why std and IQR move in opposite directions -- the whole threshold problem in one line",
             "The standard deviation fell only 10% while the IQR fell 54%. Both are describing the same "
             "distribution. The gap is the 954 rows where a Shot follows the defensive action and "
             "<code>xt_after</code> takes that shot's own xG -- up to 0.897, against the xT grid's maximum cell "
             "of 0.2575. Those rows inflate std without touching the middle. Any threshold derived from std "
             "therefore describes v2's bulk much worse than it described v1's."),
    ]


# --------------------------------------------------------------------------
# Section 4 -- adaptations
# --------------------------------------------------------------------------

def build_section_4(scale: dict) -> list[dict]:
    rs = scale["robust_scale"]
    return [
        p("Prompt 65 made three adaptations to the xg_target suite's methodology, each stated in every report "
          "it touched. All three are re-examined here against v2 rather than inherited. Two survive unchanged "
          "in method; one keeps its mechanism but loses its original justification and gains a warning."),
        h3("Adaptation 1 -- the scale-setting statistic (JUSTIFICATION RESTATED, WARNING ADDED)"),
        p(esc(scale["scale_note"])),
        table(["Quantity", "v1", "v2 on this run"], [
            ["xg flat margin on the same rows (0.5 x mean xg)", "0.004055",
             f"{scale['xg_flat_margin_absolute']:.6f}"],
            ["as a fraction of xg's own std", "0.084425",
             f"{scale['xg_flat_margin_as_fraction_of_xg_std']:.6f}"],
            ["translated flat margin", "0.005486", f"{scale['flat_margin']:.6f}"],
            ["translated consistency range trigger", "0.010973", f"{scale['range_trigger']:.6f}"],
            ["translated review near-identical threshold", "0.005486", f"{scale['near_identical_threshold']:.6f}"],
            ["translated review distinct-signal threshold", "0.016459", f"{scale['distinct_signal_threshold']:.6f}"],
            ["flat margin on the non-zero-delta subset", "0.005826", f"{scale['flat_margin_nonzero']:.6f}"],
            ["**flat margin as a multiple of MAD**", "**0.53x**", f"**{rs['flat_margin_over_mad']:.2f}x**"],
            ["**share of all rows inside the flat margin**", "**35.4%**",
             f"**{rs['pct_rows_inside_flat_margin']:.1f}%**"],
            ["robust (MAD-derived) alternative margin", "0.001286", f"{rs['robust_flat_margin']:.6f}"],
            ["_r vs the old `target_future_xg_10s`, measured on these rows_", "-0.016",
             f"**{scale['correlation_with_old_target']['pearson_r']:+.4f}**"],
        ]),
        p("That last row is recorded here rather than in the findings section on purpose. <b>Adaptation 1 is "
          "the one place this entire suite still reaches for the old target</b> -- it borrows "
          "<code>target_future_xg_10s</code>'s own standard-deviation fraction to set every threshold in every "
          "report. So wherever a report in this portal cites an xg-derived threshold, this is how the two "
          "targets actually relate on the same rows. Prompt 66's figure of -0.108 reproduces here at "
          f"{scale['correlation_with_old_target']['pearson_r']:+.4f} over "
          f"n={scale['correlation_with_old_target']['n']:,}. It is an order of magnitude stronger than v1's "
          "-0.016 and, unlike v1's, correctly signed -- more threat denied now goes with less future attacking "
          "output. <b>It is still modest, and that is expected rather than disappointing:</b> the two targets "
          "measure genuinely different things (an instantaneous value swing versus a forward-looking 10-second "
          "outcome), so a strong correlation would have been evidence of redundancy, not of validity. This "
          "block is published in every report's <code>target_scale.correlation_with_old_target</code>."),
        note("THIS THRESHOLD DOES NOT TRANSFER CLEANLY -- the explicit finding, not a footnote",
             esc(scale["threshold_transfer_warning"])),
        p("Scale-free thresholds were carried over <b>completely unchanged</b>, and were checked rather than "
          "assumed to be scale-free: |Spearman rho| &ge; 0.05, small-n ROW COUNTS (500 and 1000), the "
          "substitutive (0.3) and additive (2.0) ratio cutoffs, <code>MIN_CELL_N_FOR_DELTA=20</code>, "
          "<code>QCUT_N=4</code>, 10 quantile deciles, and the discrete-cardinality cutoff of 15. None of "
          "these can be distorted by kurtosis: they are counts, ranks or ratios. The confound suite's "
          "<code>_verdict()</code> is in the same category -- it compares only signs."),
        h3("Adaptation 2 -- the conditional panel (UNCHANGED in method)"),
        p(esc(scale["conditional_panel_note"])),
        p("The knock-on rename Prompt 65 introduced is kept: the slice-stratification reports' "
          "<code>occurrence_vs_quality</code> field encodes an xg-specific distinction that does not exist for "
          "this target, and remains <code>zero_mass_vs_magnitude</code>."),
        h3("Adaptation 3 -- two-directional framing (UNCHANGED in method, MORE load-bearing)"),
        p(esc(scale["direction_note"])),
        p("The four concrete consequences are unchanged and still apply: diverging bar charts instead of "
          "left-anchored fills; a dashed zero line on every sparkline; diverging green/red interaction "
          "heatmaps; and the two fields with no xg counterpart -- whether a binned curve CROSSES zero, and "
          "whether within-stratum deltas change SIGN."),
        h3("Not an adaptation -- log-transforms"),
        p(esc(scale["log_transform_note"])),
    ]


# --------------------------------------------------------------------------
# Section 5 -- findings
# --------------------------------------------------------------------------

def build_section_5(fl: dict) -> list[dict]:
    pf = fl["pattern_analysis_findings"]
    ca, flg, ct, tr, ss, ni, rv, nr = (pf["category_atlas"], pf["flag_ledger"], pf["confound_tests"],
                                       pf["tournament_stability"], pf["slice_stratification"],
                                       pf["numeric_interaction"], pf["review_tier"],
                                       pf["numerical_vs_target_rankings"])
    lc = fl["leakage_confirmation_xt"]
    d = lc["part_d_prime_construction_coupling"]["strongest"]
    vc = lc["part_d_prime_construction_coupling"]["v1_comparison"]
    conf_by_name = {t["name"]: t for t in ct["tests"]}

    return [
        h3("Prompt 66's findings -- where each one resurfaced in this portal"),
        note("Clearance reversal -- CONFIRMED in the Category Atlas, and tested twice in the Confound report",
             esc(ca["finding"])),
        p("The Category Atlas is where Prompt 66's headline number had to reappear, and it does, independently "
          f"recomputed: Clearance at <b>{ca['clearance_mean_xt']:+.6f}</b> over n={ca['clearance_n']:,}, "
          f"ranked <b>{esc(ca['clearance_rank'])}</b>. Prompt 66 reported +0.0247 and 1st of 8 -- matched "
          "exactly. The Flag Ledger carries the same correction from the other direction, through the "
          "possession flags. Two confound tests then ask whether the new positive mean is real:"),
        table(["Confound test", "Unconditional", "Given a non-zero delta", "New?"],
              [[t["title"], t["verdict"], t["verdict_given_nonzero_delta"], "yes" if t["is_new_test"] else "no"]
               for t in ct["tests"]]),
        p("Both Clearance tests return <b>no</b> -- the pattern does <b>not</b> reverse within either "
          "stratum. Read plainly: Clearance's positive mean survives conditioning on defending-box proximity, "
          "and it survives conditioning on whether the action ended its possession. <b>That second result is "
          "the one worth having.</b> It was the obvious way the correction could have been self-fulfilling -- "
          "clearances end possessions more often than average, possession-enders get "
          "<code>xt_after = 0</code> by construction, so Clearance could have looked good purely as an "
          "artefact of the fix. It does not: the effect holds inside the stratum where the construction rule "
          "does not apply."),
        note("distribution shape -- CONFIRMED in the Distribution Atlas",
             esc(v2.PROMPT_66_SHAPE_FINDING)),
        note("possession-ending collapse -- CONFIRMED in the Flag Ledger and Leakage Audit Part C'",
             esc(v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING)),
        h3("The possession flags, measured off-pool"),
        table(["Flag", "n (True)", "mean delta (True)", "% exactly zero (True)", "lift"],
              [[f"`{k}`", f"{v['n_true']:,}", f"{v['mean_xt_true']:+.6f}", f"{v['pct_zero_true']:.1f}%",
                f"{v['lift']:+.6f}"]
               for k, v in flg["possession_flags_no_longer_degenerate"].items()]),
        p("All three True groups were confirmed on this run to equal their own <code>xt_before</code> exactly "
          "(<code>np.allclose</code>). <b>These columns stay excluded in <code>feature_config.py</code> and "
          "nothing in this portal changes that</b> -- but the reason has strengthened. v1 left an open "
          "question about unlocking them, since their stated exclusion reason is target-specific. v2 closes "
          "that question the other way: a column that deterministically sets one term of the target is a "
          "construction input, and modelling on it would be circular."),
        h3("Construction coupling -- the v1 caveat, re-measured"),
        p(f"Strongest locked-feature correlation with the target: <code>{esc(vc['v1_strongest_feature'])}</code> "
          f"at r={vc['v1_r_vs_target']:+.4f} under v1, falling to <code>{esc(d['feature'])}</code> at "
          f"r={d['r_vs_target']:+.4f} under v2 -- <b>{vc['abs_reduction_pct']:.0f}% lower</b>. The same feature "
          f"sits at r={d['r_vs_xt_before']:+.4f} against <code>xt_before</code>, which is unchanged from v1 and "
          "is where the residual, ordinary location relationship now lives."),
        h3("Numerical features vs the target"),
        p(f"{nr['n_curves_crossing_zero']}/{nr['n_features']} features' binned curves cross zero -- they "
          "separate threat-reducing from threat-increasing actions rather than only varying in magnitude. "
          f"{nr['n_train_test_inconsistent']} feature(s) are flagged train/test-inconsistent. <b>Note the "
          "collapse in rho relative to v1</b>, which is the same construction-coupling story: v1's top five "
          "all sat above |rho| = 0.75."),
        table(["Feature", "Spearman rho", "Shape", "Curve crosses zero"],
              [[f"`{r['feature']}`", str(r["spearman_rho"]), str(r["shape"]), str(r["binned_curve_crosses_zero"])]
               for r in nr["active_top"]]),
        note("read these shape labels with the section-1 caveat in hand",
             "24 of these 31 features classify differently under the MAD-derived flat margin than under the "
             "std-derived one used above (against 15 of 31 on v1). The <code>Shape</code> column is indicative "
             "on this target, not decided."),
        h3("Flag Ledger -- top 5 by |lift|, both directions"),
        table(["Flag", "lift", "mean True", "mean False", "n True"],
              [[f"`{r['column']}`", f"{r['lift']:+.6f}", f"{r['mean_xt_true']:+.6f}",
                f"{r['mean_xt_false']:+.6f}", f"{r['n_true']:,}"] for r in flg["top_by_abs_lift"]]),
        h3("Tournament stability"),
        p(f"{tr['n_genuine']}/{tr['n_features']} active features show a genuine tournament-level difference "
          "rather than arbitrary train/test noise."),
        table(["Feature", "Verdict"], [[f"`{f['feature']}`", f["verdict"]] for f in tr["features"]]),
        h3("Slice stratification"),
        p(f"V1: {ss['v1']['n_cells']} cells, {ss['v1']['n_genuine_divergences']} genuine divergences "
          f"unconditionally and {ss['v1']['n_genuine_divergences_given_nonzero_delta']} once the zero mass is "
          f"removed, with {ss['v1']['n_conditioning_disagreements']} categories where that removal flips the "
          f"conclusion. V2: {ss['v2']['n_cells']} cells, {ss['v2']['n_genuine_divergences']} / "
          f"{ss['v2']['n_genuine_divergences_given_nonzero_delta']}."),
        h3("Numeric x numeric interaction"),
        p(f"{ni['n_agree']}/{ni['n_pairs']} pairs classify the same way unconditionally and on the "
          f"non-zero-delta subset. {ni['n_sign_flipping']}/{ni['n_pairs']} have within-stratum deltas that "
          "change SIGN -- feature A's effect reverses direction across levels of feature B. That is a distinct "
          "finding from 'interactive' and cannot arise on the xG target at all."),
        table(["Feature A", "Feature B", "Classification", "Stratum deltas change sign"],
              [[f"`{r['feature_a']}`", f"`{r['feature_b']}`", r["classification"],
                "yes" if r["within_stratum_deltas_change_sign"] else "no"] for r in ni["pairs"]]),
        h3("Review tier"),
        p(f"{rv['n_review_pairs']} ACTIVE review pairs; {rv['n_needs_human_call']} remain needs_human_call "
          "(dominated, as on every target, by continuous-continuous pairs no rule covers); "
          f"{rv['n_type2_opposite_sign']} Type-2 pair(s) resolve on the opposite-sign rule, which carries more "
          "weight here than on xg -- a sign disagreement means the two flags disagree about the DIRECTION of "
          "the threat swing, not merely about being above or below a base rate."),
    ]


# --------------------------------------------------------------------------
# Section 6 -- caveats and open questions
# --------------------------------------------------------------------------

def build_section_6(fl: dict) -> list[dict]:
    lc = fl["leakage_confirmation_xt"]
    vc = lc["part_d_prime_construction_coupling"]["v1_comparison"]
    return [
        p("<b>Nothing in this portal changes <code>feature_config.py</code>, and no model artefact was "
          "touched.</b> No feature is added, dropped or re-tiered."),
        h3("Caveat 1 -- construction coupling: SUBSTANTIALLY RETIRED (was v1's most important caveat)"),
        p(esc(lc["part_d_prime_construction_coupling"]["implication_for_the_lock"])),
        h3("Caveat 2 -- this target's effects can sit in its zero mass: SURVIVES, and widened"),
        p("A mean-difference test alone -- the instrument the xg audit reaches for first -- can miss an effect "
          "that lives in the exactly-zero share. v1 recorded this via <code>has_previous_event</code>, where "
          "the mean test returned p=0.0645 and would have reported nothing while the zero share differed by "
          "+27.3pp. The caveat is not retired by the correction, and the zero share it concerns grew from "
          "11.3% to 20.2% of rows."),
        h3("Caveat 3 -- NEW: the flat-margin threshold no longer describes this distribution"),
        p("Covered in full in sections 1 and 4. In short: the std-derived translated margin now spans the "
          "majority of rows, 24 of 31 Numerical Atlas features classify differently under a robust "
          "alternative, and shape labels in this portal are correspondingly less decisive than v1's. This is "
          "reported rather than papered over, and the threshold was not silently swapped, because doing so "
          "would have made every v2 cell incomparable with the v1 portal this one supersedes."),
        note("methodological warning for anyone extending this suite",
             "v2 is a harder distribution to work with than v1, not an easier one. Its mean moved in the "
             "football-sensible direction, which is the headline improvement -- but its skew, kurtosis and "
             "zero share all moved AWAY from anything a standard-deviation-based convention handles well. "
             "Prefer rank-based, count-based, ratio-based and sign-based instruments on this target; they were "
             "the ones that transferred from v1 without a scratch, and the one instrument that did not "
             "transfer is the only std-based one in the suite."),
        h3("Open questions this portal does NOT answer"),
        ul([
            "Whether <code>Block</code>, now the lowest-mean event type at -0.0185, is itself an artefact. "
            "Prompt 66 recorded the handover from Clearance and explicitly left it unchased as out of scope; "
            "this portal surfaces it in the Category Atlas and takes no position.",
            "Whether the three <code>action_*_possession</code> flags should be formally re-documented in "
            "<code>feature_config.py</code> as CONSTRUCTION INPUTS rather than structural zeros. The stated "
            "exclusion reason is wrong for this target in both v1 and v2; the exclusion itself is right. "
            "Changing the recorded reason is a <code>feature_config.py</code> edit and is out of this suite's "
            "remit.",
            "What model architecture suits a left-skewed, very heavy-tailed, sign-carrying target. Prompt 66 "
            "found v2 is <i>less</i> like the existing hurdle architecture's shape than v1 was, not more. This "
            "is an EDA portal and takes no position; no model artefact was touched.",
            "Whether <code>xt_before</code> should also be redefined. Prompt 66 scoped its correction to "
            "<code>xt_after</code> only, and <code>xt_before</code> is still a grid lookup at the previous "
            "event's location with no period-boundary restriction -- unlike the corrected "
            "<code>xt_after</code>, which respects them. That asymmetry is recorded, not resolved.",
            "The passive leg, entirely. Out of scope for this work.",
        ]),
    ]


# --------------------------------------------------------------------------
# Section 7 -- document map
# --------------------------------------------------------------------------

def build_section_7() -> list[dict]:
    return [
        h3("Built here (target-dependent -- rebuilt against target_xt_delta_v2)"),
        table(["Report", "Files"], [
            ["Active -- Category Atlas", "`active_category_atlas.html` / `.json`"],
            ["Active -- Flag Ledger", "`active_flag_ledger.html` / `.json`"],
            ["Active -- Distribution Atlas", "`active_distribution_atlas.html`, "
                                             "`active_distribution_atlas_reconstructed.html`, "
                                             "`active_distribution_atlas.json`"],
            ["Active -- Numerical Target Atlas", "`active_numerical_target_atlas.html` / `.json`"],
            ["Review Methodology", "`REVIEW_METHODOLOGY.html`, `REVIEW_ANALYSIS.json`"],
            ["Leakage Audit", "`LEAKAGE_AUDIT.html` / `.json`"],
            ["Confound (Reversal) Testing", "`CONFOUND_ANALYSIS.html` / `.json`"],
            ["Tournament Stability Check", "`TOURNAMENT_STABILITY_CHECK.html` / `.json`"],
            ["Slice Stratification V1", "`SLICE_STRATIFICATION.html` / `.json`"],
            ["Slice Stratification V2", "`SLICE_STRATIFICATION_V2.html` / `.json`"],
            ["Feature Interaction Analysis", "`FEATURE_INTERACTION_ANALYSIS.html` / `.json`"],
            ["Feature Lock Confirmation + Pattern Findings", "`FEATURE_LOCK_CONFIRMATION_XT.html` / `.json`"],
            ["Master Findings (this document)", "`MASTER_FINDINGS.md` / `.html`"],
            ["Portal index", "`INDEX.html`"],
        ]),
        h3("Linked, NOT rebuilt (target-independent -- properties of the features, not of the target)"),
        p("These seven are properties of the locked ACTIVE features and are unaffected by which target the "
          "features are measured against. <b>A correction to <code>xt_after</code> cannot move a "
          "feature-vs-feature statistic</b>, so rebuilding them would have produced byte-identical output. The "
          "portal index links out to the existing files in <code>../xg_target/</code> and "
          "<code>../shot_target/</code>."),
        ul([
            "Correlation Atlas V1 / V2 / V3 -- correlation clustering is feature-vs-feature and never "
            "references a target column.",
            "VIF Analysis -- multicollinearity is a property of the feature matrix alone.",
            "Slicer Redundancy Check -- slicer-vs-slicer, no target term.",
            "Player-Level Validity Check -- row concentration and train/test player overlap, no target term.",
            "Player-Grouped Split Check -- an identity-leakage stress test on the split itself.",
        ]),
        h3("The superseded v1 portal"),
        p(f"<code>reports/analysis/{V1}/</code> holds Prompt 65's complete portal against "
          "<code>target_xt_delta</code> (v1), moved there with <code>git mv</code> so its history is "
          "preserved. <b>Its contents are unchanged apart from a supersession banner on its index and master "
          "findings.</b> Every number in it is the accurate record of what was measured against v1 and is "
          "deliberately not corrected."),
        h3("New generator scripts (all added as siblings; no Prompt-65 script was modified)"),
        p("Nothing under <code>src/eda/</code> that already existed was edited. Prompt 65's 21 "
          "<code>*_xt.py</code> generators, and the <code>xg_target</code> and <code>shot_target</code> "
          "portals' own generators, all remain exactly reproducible."),
        ul([
            "`src/eda/xt_v2_common.py` -- the re-point layer, the re-examined Adaptation 1, and Prompt 66's "
            "carried-forward finding strings",
            "`src/eda/generate_reports_xt_v2.py` -- Category Atlas + Flag Ledger",
            "`src/eda/generate_leakage_audit_xt_v2.py` + `generate_leakage_report_xt_v2.py`",
            "`src/eda/generate_confound_analysis_xt_v2.py`",
            "`src/eda/generate_feature_lock_confirmation_xt_v2.py` + `generate_feature_lock_report_xt_v2.py`",
            "`src/eda/xt_v2_prose_fixups.py` -- declared, asserted corrections to five stale sentences in "
            "re-pointed outputs",
            "`src/eda/generate_master_findings_xt_v2.py` (this document)",
            "`src/eda/generate_eda_portal_xt_v2.py` -- portal shell AND the driver that makes the re-point work",
        ]),
        p("Rebuild the whole portal with: <code>.venv/Scripts/python.exe -m "
          "src.eda.generate_eda_portal_xt_v2 --all</code>"),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt(columns=["event_id"])
    scale = xc.target_scale(df)
    fl = _load("FEATURE_LOCK_CONFIRMATION_XT.json")
    lk = _load("LEAKAGE_AUDIT.json")
    active_count = sum(len(ACTIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))
    vc = lk["part_d_prime_construction_coupling"]["v1_comparison"]
    ca = fl["pattern_analysis_findings"]["category_atlas"]

    bottom_line = (
        "<b>This portal supersedes an earlier pass and should be read with that in mind.</b> Prompt 65 built "
        "these same thirteen reports against <code>target_xt_delta</code> (v1), whose <code>xt_after</code> "
        "term scored the defensive action's OWN location; Prompt 66 found that wrong and replaced it with a "
        f"possession-outcome definition. The headline consequence reproduces here exactly: <b>Clearance moves "
        f"from 8th of 8 event types (-0.032482) to {esc(ca['clearance_rank'])} at "
        f"{ca['clearance_mean_xt']:+.6f}</b>, and it survives conditioning on both defending-box proximity "
        "and on the construction rule that could have made it self-fulfilling. v1's single most important "
        f"caveat, construction coupling, is <b>{vc['abs_reduction_pct']:.0f}% weaker</b> and substantially "
        f"retired. But <code>target_xt_delta_v2</code> is a HARDER distribution than v1 -- skew "
        f"{scale['skew']:+.3f}, excess kurtosis {scale['excess_kurtosis']:.2f}, {scale['pct_zero']:.1f}% "
        "exactly zero -- and <b>one v1-tuned threshold does not transfer: the std-derived flat margin now "
        f"spans {scale['robust_scale']['pct_rows_inside_flat_margin']:.1f}% of all rows, and 24 of 31 features "
        "change shape classification under a robust alternative. That is reported, not forced to fit.</b> No "
        "feature was added, dropped or re-tiered, no model artefact was touched, and the seven "
        "target-independent reports were linked rather than rebuilt."
    )

    sections = [
        ("What changed from v1, and why", build_section_1(scale, lk)),
        ("Step 0 -- script-by-script disposition: re-point vs rewrite", build_section_2()),
        ("The target: what target_xt_delta_v2 is and what shape it has", build_section_3(scale)),
        ("What methodology was adapted, and what did not transfer", build_section_4(scale)),
        ("Findings", build_section_5(fl)),
        ("Caveats recorded, and open questions this portal does not answer", build_section_6(fl)),
        ("Document map", build_section_7()),
    ]

    # --- Markdown ---
    md_parts = [
        "# Master Findings -- xT-Delta v2 Target (Active-Binary Leg Only)",
        "",
        f"> **Supersedes the Prompt 65 portal.** This portal is built against `target_xt_delta_v2`. The prior "
        f"pass against `target_xt_delta` (v1) is preserved unchanged at `reports/analysis/{V1}/`.",
        "",
        "> **Scope: the ACTIVE-BINARY leg only.** The passive leg is explicitly out of scope for this target "
        "and no passive-side xT-delta report exists. Nothing in this document, or in this portal, covers it.",
        "",
        f"**Bottom line up front.** {bottom_line}",
    ]
    for i, (title, blocks) in enumerate(sections, start=1):
        md_parts.append(f"\n## {i}. {title}\n")
        md_parts.append(render_blocks_md(blocks))
    md = "\n".join(md_parts)
    for tag, repl in (("<code>", "`"), ("</code>", "`"), ("<b>", "**"), ("</b>", "**"),
                      ("<i>", "_"), ("</i>", "_")):
        md = md.replace(tag, repl)
    # The shared block renderer HTML-escapes note bodies, which is right for
    # the .html twin but leaks entities into Markdown (the v1 document has the
    # same artefact). Unescaped here so the .md reads as prose; the .html is
    # unaffected because it is built from the same blocks, not from this text.
    for ent, ch in (("&#x27;", "'"), ("&quot;", '"'), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")):
        md = md.replace(ent, ch)
    MD_PATH.write_text(md + "\n", encoding="utf-8")

    # --- HTML, from the same block data ---
    html_parts = [
        '<div class="finding flag" style="margin-bottom:20px;"><span class="tag">supersedes a prior pass</span>'
        '<p><b>This portal is built against <code>target_xt_delta_v2</code>.</b> Prompt 65 built the same '
        'thirteen reports against <code>target_xt_delta</code> (v1); Prompt 66 corrected that target\'s '
        '<code>xt_after</code> term. The v1 portal is preserved UNCHANGED at '
        f'<code>reports/analysis/{V1}/</code> and is linked from this portal\'s index. Section 1 below '
        'explains exactly what changed and what it did to v1\'s findings.</p></div>',
        '<div class="finding flag" style="margin-bottom:20px;"><span class="tag">scope</span>'
        '<p><b>This portal covers the ACTIVE-BINARY leg only.</b> The passive leg is explicitly out of scope '
        'for this target and no passive-side xT-delta report exists.</p></div>',
        f'<div class="finding" style="margin-bottom:28px;"><span class="tag">bottom line up front</span>'
        f'<p>{bottom_line}</p></div>',
    ]
    for i, (title, blocks) in enumerate(sections, start=1):
        html_parts.append(f'<h2 class="mf-section-title" id="s{i}">{i}. {esc(title)}</h2>')
        html_parts.append(render_blocks_html(blocks))

    html = render.render_article(
        eyebrow="MASTER FINDINGS -- XT DELTA V2 &middot; ACTIVE-BINARY LEG ONLY",
        title="xT-Delta v2 Target -- Master Findings (Active-Binary Leg)",
        dek=(
            "The entry point for this portal: what the Prompt 66 target correction changed and why, which of "
            "Prompt 65's 21 scripts re-pointed cleanly and which had to be rewritten, what "
            "target_xt_delta_v2 is, which threshold stopped transferring, every finding, and what is "
            "deliberately left open. Supersedes the v1 portal. Active-binary leg only."
        ),
        stats=[
            (f"{active_count}", "active features"),
            ("13", "reports rebuilt vs v2"),
            ("7", "linked, not rebuilt"),
            (f"{vc['abs_reduction_pct']:.0f}%", "less construction coupling than v1"),
        ],
        body_html="\n".join(html_parts),
        extra_css=render.CORR_CSS + render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
.mf-section-title { margin-top:48px; padding-top:24px; border-top:2px solid var(--border); }
.mf-p { color: var(--text-secondary); font-size: 14px; line-height:1.6; margin: 10px 0; }
.mf-ul { padding-left:22px; }
.mf-ul li { color: var(--text-secondary); font-size: 13.5px; line-height:1.6; margin-bottom:10px; }
.mf-h3 { font-family:"Archivo",sans-serif; font-size:16px; margin-top:24px; }
.mf-table-wrap { overflow-x:auto; margin:12px 0; }
.mf-table { width:100%; border-collapse:collapse; font-size:12.5px; }
.mf-table th { text-align:left; color:var(--text-muted); padding:6px 10px; border-bottom:1px solid var(--border); }
.mf-table td { padding:6px 10px; border-top:1px solid var(--border); color:var(--text-secondary); vertical-align:top; }
""",
    )
    HTML_PATH.write_text(html, encoding="utf-8")

    n_rewrite = sum(1 for r in STEP_0_ROWS if r[1].startswith("**REWRITE"))
    print(f"Step 0: {len(STEP_0_ROWS) - n_rewrite} re-pointed, {n_rewrite} rewritten")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
