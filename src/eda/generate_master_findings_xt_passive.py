"""CLI entrypoint: MASTER_FINDINGS.md + .html for the xT-delta portal --
BOTH LEGS.

STEP-0 DISPOSITION: REWRITE of the overview document (a sibling, so
`generate_master_findings_xt_v2.py` still reproduces Prompt 67's
active-only version byte-for-byte against an un-re-pointed `xt_common`).

The document Prompt 67 wrote opens by declaring the passive leg out of
scope and closes by listing it as an open question. Both statements are
now false, so re-pointing or patching was never an option -- the framing
had to be rebuilt. The BLOCK MODEL (`p` / `ul` / `table` / `note` / `h3`
-> `render_blocks_md` / `render_blocks_html`) is imported from
`generate_master_findings` exactly as both prior versions import it, so the
.md and .html still cannot drift apart.

STRUCTURE, decided here and stated so a reader can navigate it:
  1. A shared preamble on the xT target FAMILY -- what an xT delta is, the
     possession-outcome definition both legs share, and how the two legs'
     targets differ.
  2. The Step-0 table: which report types are inherently per-leg, which are
     shared in the xg precedent, which already have a passive equivalent
     somewhere in the project, and what this pass had to build.
  3. The ACTIVE section, BY REFERENCE. Prompts 64/66/67's findings are
     cited and linked, not restated -- they are in the linked source
     documents and in this file's own prior version, which git preserves.
  4. The PASSIVE section, in full. This pass's own findings.
  5. Caveats and open questions, per leg.
  6. Document map.

CLOSEOUT DECISION, unchanged for the third time: no separate
PATTERN_ANALYSIS_CLOSEOUT-style document. Prompt 65 declined one because
the portal was a single-pass single-leg suite; Prompt 67 declined one
because a closeout would also have had to carry the v1-vs-v2 comparison.
The same call is made here, and the reason is now stronger still: a
closeout would have to carry BOTH legs and the v1-vs-v2 arc, and splitting
that across two documents is exactly how they would drift apart.

Usage:
    python -m src.eda.generate_master_findings_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import render, xt_common as xc
from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_master_findings import h3, note, p, render_blocks_html, render_blocks_md, table, ul
from src.eda.render import esc

OUT_DIR = xc.OUT_DIR
MD_PATH = OUT_DIR / "MASTER_FINDINGS.md"
HTML_PATH = OUT_DIR / "MASTER_FINDINGS.html"
V1_PORTAL = "xt_target_v1_superseded"


def _load(name: str) -> dict:
    return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Section 2 -- the Step-0 table. Written in the same format and spirit as
# Prompt 65's own Step-0 table (a disposition row per unit of work, with the
# evidence that produced the disposition stated inline).
# --------------------------------------------------------------------------

STEP_0_TARGET_DEPENDENT: list[list[str]] = [
    ["**Category Atlas**", "**Per-leg**", "No -- xT active only",
     "**GENUINE PASSIVE BUILD.** `xg_target/` ships a literal `active_category_atlas.html` / "
     "`passive_category_atlas.html` pair (+ `.json`). Inherently per-leg: the two datasets share only "
     "`period` and `phase_label` among their categorical columns. Built: `passive_category_atlas.html/.json`."],
    ["**Flag Ledger**", "**Per-leg**", "No -- xT active only",
     "**GENUINE PASSIVE BUILD.** Literal `active_flag_ledger.*` / `passive_flag_ledger.*` pair in "
     "`xg_target/`. Built: `passive_flag_ledger.html/.json`."],
    ["**Distribution Atlas**", "**Per-leg**", "No -- xT active only",
     "**GENUINE PASSIVE BUILD, both variants.** `xg_target/` ships the `active_`/`passive_` pair; "
     "`shot_target/` additionally ships `passive_distribution_atlas_reconstructed.html`, confirming the "
     "two-variant convention DOES apply to the passive leg (checked by listing both directories, not assumed "
     "from the active xT portal). Built: `passive_distribution_atlas.html`, "
     "`passive_distribution_atlas_reconstructed.html`, `passive_distribution_atlas.json`."],
    ["**Numerical Target Atlas**", "**Per-leg**", "No -- xT active only",
     "**GENUINE PASSIVE BUILD.** Literal pair in `xg_target/`. The pools barely overlap: passive carries the "
     "screening / lane / option columns, active the local counts and spreads. Built: "
     "`passive_numerical_target_atlas.html/.json`."],
    ["**Review Methodology**", "Shared FILE, both legs inside",
     "Yes, for xg -- `xg_target/REVIEW_ANALYSIS.json` has `datasets.active` AND `datasets.passive`",
     "**GENUINE PASSIVE BUILD, separate file.** The xg one-file convention could not be mirrored: this "
     "portal's `REVIEW_ANALYSIS.json` / `REVIEW_METHODOLOGY.html` are Prompt 67's ACTIVE-ONLY output and are "
     "frozen by this pass's constraints. Built: `PASSIVE_REVIEW_METHODOLOGY.html` + "
     "`PASSIVE_REVIEW_ANALYSIS.json`."],
    ["**Leakage Audit**", "Shared NAME, but **passive-only content**",
     "Yes, for xg -- and it is the ACTIVE leg that has no native counterpart anywhere in this repo",
     "**GENUINE PASSIVE BUILD -- and the one report type where the leg asymmetry runs the other way.** "
     "`xg_target/LEAKAGE_AUDIT.json` carries `\"dataset\": \"passive\"` and its Parts B/C/D test three "
     "passive-only columns; `shot_target/` is the same. Prompt 65 had to INVENT active substitutes (Parts "
     "B'/C'/D'). This pass needs none: Parts A/B/C/D mirror literally. Part E' (construction coupling through "
     "`xt_before = xT(ball_x, ball_y)`) is new. Built: `PASSIVE_LEAKAGE_AUDIT.html/.json`."],
    ["**Confound Testing**", "Shared NAME, but **passive-only content**",
     "Yes, for xg -- all 5 of its tests are passive-dataset tests",
     "**GENUINE PASSIVE BUILD, zero invention.** Prompt 65 found exactly ONE active confound test in the whole "
     "repo and had to write two of its own. All five established passive tests mirror here with no "
     "substitution. Built: `PASSIVE_CONFOUND_ANALYSIS.html/.json`."],
    ["**Tournament Stability Check**", "Shared FILE, both legs inside",
     "Yes, for xg -- `flat_margins_by_dataset` has an `active` and a `passive` entry",
     "**GENUINE PASSIVE BUILD, separate file** (same frozen-active-file reason as Review Methodology). "
     "`INVESTIGATE`'s 4 passive entries, imported not re-derived. Built: "
     "`PASSIVE_TOURNAMENT_STABILITY_CHECK.html/.json`."],
    ["**Slice Stratification V1**", "Shared FILE, both legs inside",
     "Yes, for xg -- `results.active` AND `results.passive`",
     "**GENUINE PASSIVE BUILD, separate file.** `FEATURE_SLICER_PLAN`'s and `build_plan()`'s passive entries, "
     "imported. Built: `PASSIVE_SLICE_STRATIFICATION.html/.json`."],
    ["**Slice Stratification V2**", "Shared FILE, both legs inside",
     "Yes, for xg -- `results.active` AND `results.passive`",
     "**GENUINE PASSIVE BUILD, separate file -- and the one report where this leg can do MORE than the active "
     "one.** `defender_archetype_name` is passive-only, so the closing archetype-vs-boolean comparison that "
     "the active half reports as NOT APPLICABLE is made here. Built: "
     "`PASSIVE_SLICE_STRATIFICATION_V2.html/.json`."],
    ["**Feature Interaction Analysis**", "Shared FILE, both legs inside",
     "Yes, for xg -- `thresholds.min_marginal_delta_by_dataset` has an entry per leg",
     "**GENUINE PASSIVE BUILD, separate file.** `PAIRS`'s 5 passive entries, imported and deliberately not "
     "re-ranked. All five involve screening / option / engagement columns with no active counterpart. Built: "
     "`PASSIVE_FEATURE_INTERACTION_ANALYSIS.html/.json`."],
    ["**Feature Lock Confirmation (+ Pattern Findings)**", "Shared FILE, both legs inside",
     "Yes, for xg -- `confirmed_counts` carries `active: 34` and `passive: 38`",
     "**GENUINE PASSIVE BUILD, separate file.** The renderer had to be written rather than relabelled: the "
     "active one's leakage section is hardcoded to Parts B'/C'/D' and its pattern section to the Clearance "
     "artefact and the possession flags, none of which exists on this leg. Built: "
     "`PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.html/.json`."],
    ["**Master Findings + portal index**", "Shared document", "n/a",
     "**REWRITTEN for both legs** (this document) and the portal index regrouped into Active + Passive, "
     "mirroring `xg_target/INDEX.html`'s own structure. `reports/analysis/INDEX.html`'s xT card updated."],
]

STEP_0_TARGET_INDEPENDENT: list[list[str]] = [
    ["Correlation Atlas V1 / V2 / V3", "Shared, both legs",
     "**YES** -- `CORRELATION_ANALYSIS*.json` carries `datasets.active` and `datasets.passive`",
     "**LINK, do not rebuild.** Correlation clustering is feature-vs-feature and never references a target, so "
     "re-running it against either xT target would produce byte-identical pairs and tiers."],
    ["VIF Analysis", "Shared, both legs", "**YES** -- `VIF_ANALYSIS.json` carries `datasets.active`/`.passive`",
     "**LINK.** Multicollinearity is a property of the feature matrix alone."],
    ["Slicer Redundancy Check", "Shared, both legs",
     "**YES** -- `SLICER_REDUNDANCY.json` carries `datasets.active`/`.passive`",
     "**LINK.** Slicer-vs-slicer, no target term."],
    ["Player-Level Validity Check", "**ACTIVE-ONLY**, project-wide",
     "**NO** -- `shot_target/PLAYER_LEVEL_VALIDITY_CHECK.json` carries `\"dataset\": \"active\"` plus its own "
     "`scope_limitation` field",
     "**LINK, and state the asymmetry.** Not built here, and that is not a gap this pass could close: the "
     "check is about row concentration per `player_id` and train/test player overlap, and the passive leg is "
     "not player-indexed the same way. It is also target-independent, so it falls outside this pass's remit "
     "either way."],
    ["Player-Grouped Split Check", "**ACTIVE-ONLY**, project-wide",
     "**NO** -- its own `scope` field reads \"Active leg only (player_id / position are active-dataset "
     "concepts; the passive leg is not player-indexed the same way)\"",
     "**LINK, and state the asymmetry.** Confirmed by reading the `scope` field, not inferred from the title "
     "-- unlike the Player-Level Validity Check, this one's name does not announce its scope. Same reasons as "
     "above."],
]


def build_section_step0() -> list[dict]:
    return [
        p("<b>This section is a required deliverable, not scratch notes.</b> Before anything was built, "
          "<code>reports/analysis/xg_target/INDEX.html</code> and every JSON behind it were read directly, "
          "along with the current <code>reports/analysis/xt_target/</code> listing and the target-independent "
          "reports across <code>reports/analysis/</code>. Each of the thirteen target-dependent report types "
          "Prompt 65/67 built for the active leg was classified: is it inherently per-leg, or is it a single "
          "file carrying both legs' sections in the established precedent? And does a passive equivalent "
          "already exist anywhere in this project?"),
        note("the headline result of Step 0",
             "<b>All twelve target-dependent report types needed a genuine passive build.</b> Not one of them "
             "had a passive-side xT equivalent anywhere, and not one could be satisfied by reusing the active "
             "leg's content. Eight of them are SHARED-FILE reports in the xg precedent, which would normally "
             "mean adding a passive section to the existing file -- but this portal's files of those names are "
             "Prompt 67's active-only output and are frozen by this pass's constraints, so the passive halves "
             "are separate <code>PASSIVE_*</code> files and this document says so rather than leaving a reader "
             "to wonder why the two portals are shaped differently. Of the five target-independent report "
             "types, three already have a passive half and are linked; two are ACTIVE-ONLY project-wide for a "
             "structural reason, and that is stated rather than treated as a gap."),
        h3("The thirteen target-dependent report types"),
        table(["Report type", "Per-leg or shared (in the `xg_target` precedent)",
               "Passive equivalent exists already?", "Action needed here"],
              STEP_0_TARGET_DEPENDENT),
        h3("The target-independent reports (properties of the features, not of the target)"),
        p("Seven files, five report types. None was rebuilt, on either leg, for the same reason Prompt 65 and "
          "Prompt 67 gave: <b>a change to which target the features are measured against cannot move a "
          "feature-vs-feature statistic</b>, so rebuilding them would produce byte-identical output. What Step "
          "0 adds here is a per-type check of whether a PASSIVE half already exists -- which was not obvious "
          "from the titles, and in two cases turned out to be No."),
        table(["Report type", "Per-leg or shared", "Passive equivalent exists already?", "Action needed here"],
              STEP_0_TARGET_INDEPENDENT),
        h3("Two conventions checked rather than assumed"),
        ul([
            "<b>The reconstructed-pool Distribution Atlas variant.</b> The active xT portal ships both "
            "<code>active_distribution_atlas.html</code> and "
            "<code>active_distribution_atlas_reconstructed.html</code>, but <code>xg_target/</code> ships only "
            "a single <code>passive_distribution_atlas.html</code>. That could have meant the passive leg gets "
            "one variant. It does not: <code>shot_target/</code> ships "
            "<code>passive_distribution_atlas_reconstructed.html</code> too, so the two-variant convention "
            "applies to both legs, and the missing xg file is the stale filename-branch artefact Prompt 65 "
            "already recorded for the active side. Both variants are built here.",
            "<b>The shared-name reports' file convention.</b> The prompt asked whether "
            "<code>xg_target</code>'s own convention for reports like the Leakage Audit is a single file "
            "covering both legs or separate files. Checked directly: it is a SINGLE file in every case. That "
            "convention is deliberately NOT mirrored here, and the reason is a constraint rather than a "
            "preference -- merging a passive section into "
            "<code>LEAKAGE_AUDIT.json</code>/<code>.html</code> and its seven siblings would mean editing "
            "active-leg report content that this pass leaves exactly as Prompt 67 rebuilt it. The divergence "
            "is recorded here, in every affected report's own scope note, and on the portal index.",
        ]),
    ]


# --------------------------------------------------------------------------
# Section 1 -- shared preamble on the target family
# --------------------------------------------------------------------------

def build_section_family(a_scale: dict, p_scale: dict) -> list[dict]:
    g = p_scale["grain_comparison"]["prompt_68_unique_event"]
    return [
        p("<b>Expected Threat (xT)</b> assigns every location on the pitch a value: the probability that a "
          "possession starting there ends in a goal. This project's grid is 8x12 over 105x68m, 96 cells, with "
          "a maximum cell value of 0.2575. An <b>xT delta</b> is the change in that value across an event: "
          "<code>xt_before - xt_after</code>. A POSITIVE delta means threat FELL (good for the defence); a "
          "NEGATIVE delta means it ROSE."),
        h3("The definition both legs share"),
        p("The <code>xt_after</code> term is a <b>possession-outcome</b> definition, and it is the same rule "
          "on both legs:"),
        ul([
            "<code>0.0</code> if the event ends its own possession -- the attack broke down, so all of the "
            "threat that existed a moment ago was denied;",
            "otherwise that event's own <code>shot_statsbomb_xg</code> if the NEXT same-possession event is a "
            "Shot -- a real chance value, which the 96-cell grid could never produce on its own;",
            "otherwise the next same-possession event's own grid value.",
        ]),
        note("why the active leg needed two attempts and this leg needed one",
             esc(pc.PROMPT_68_NO_V1_DETOUR_FINDING)),
        h3("Where the two legs' targets genuinely differ"),
        table(["", "`target_xt_delta_v2` (active-binary)", "`target_xt_delta_passive` (passive)"], [
            ["what a row is", "one actual defensive action", "one visible defender-slot per on-ball attacking event"],
            ["rows", "56,068", f"{p_scale['n_rows']:,}"],
            ["target computed at", "row grain (one action, one target)",
             f"**event grain** ({g['n_defined']:,} defined unique events), left-joined onto every defender "
             "row of that event"],
            ["`xt_before`", "the PREVIOUS event's grid cell (not a column in the active feature table)",
             "**`xT(ball_x, ball_y)` at the snapshot itself** -- and `ball_x`/`ball_y` are both LOCKED passive "
             "features"],
            ["mean", "+0.003381 (positive: actions deny threat)",
             f"{p_scale['mean']:+.6f} (negative: an arbitrary moment in a live attack has no reason to lose "
             "value)"],
            ["skew / excess kurtosis", "-0.306 / 18.33",
             f"{p_scale['skew']:+.3f} / {p_scale['excess_kurtosis']:.2f} at the row grain "
             f"({g['skew']:+.3f} / {g['excess_kurtosis']:.2f} at the event grain)"],
            ["% exactly zero", "20.2%", f"{p_scale['pct_zero']:.1f}%"],
            ["r vs `target_future_xg_10s`", "-0.1082",
             f"{p_scale['correlation_with_old_target']['pearson_r']:+.4f}"],
            ["has a superseded v1 pass", "Yes -- preserved at `reports/analysis/" + V1_PORTAL + "/`",
             "**No.** Corrected definition from the first pass."],
        ]),
        p("<b>The negative passive mean is not a discrepancy to reconcile.</b> Active-binary rows are "
          "interventions that should, on average, deny threat. Passive rows are snapshots at an arbitrary "
          "on-ball event during a live attacking phase, where there is no inherent reason attacking possession "
          "should lose value from one touch to the next -- if anything a slightly negative mean, threat "
          "creeping up as attacks progress, is the more plausible prior. Prompt 68 made this call and this "
          "portal reproduces it rather than re-litigating it."),
    ]


# --------------------------------------------------------------------------
# Section 3 -- the ACTIVE leg, BY REFERENCE
# --------------------------------------------------------------------------

def build_section_active() -> list[dict]:
    return [
        note("this section is deliberately short -- it cites rather than restates",
             "The active leg's findings are Prompts 64, 66 and 67's, and they are already written out in full "
             "in two places: the linked source documents, and the prior version of THIS file, which git "
             "preserves. Restating their tables here would create a second copy that can drift from the first. "
             "What follows is a pointer list with the headline number for each finding, so a reader knows what "
             "is there and where to go for it. <b>Nothing in this section is new, re-measured, or changed by "
             "this pass</b>, and no active-side report file was edited."),
        h3("Where to read the active leg's findings in full"),
        table(["Finding", "Headline", "Read it in"], [
            ["The Clearance artefact REVERSED", "8th of 8 event types (-0.032482) under v1 -> **1 of 8 "
                                                "(+0.024706)** under v2, and it survives conditioning on both "
                                                "defending-box proximity and the `xt_after = 0` construction "
                                                "rule",
             "`active_category_atlas.html`; `CONFOUND_ANALYSIS.html` (tests 2 and 3); "
             "`outputs/prototypes/XT_TARGET_PROTOTYPE_V2.md` s.2.2"],
            ["The distribution got HARDER, not easier",
             "mean flips -0.00132 -> +0.00338, but skew +0.093 -> -0.306 and excess kurtosis 8.85 -> 18.33; "
             "zero share 11.3% -> 20.2%",
             "`active_distribution_atlas.html`; `XT_TARGET_PROTOTYPE_V2.md` s.2.3"],
            ["Possession-ending collapse", "`action_ended_possession == True` (n=4,656) mean +0.03051, "
                                           "identical to that slice's own `xt_before` (`np.allclose`) -- "
                                           "construction, not discovery",
             "`active_flag_ledger.html`; `LEAKAGE_AUDIT.html` Part C'"],
            ["Construction coupling substantially RETIRED",
             "strongest locked-feature correlation with the target falls from r=+0.5551 (v1) to **+0.1431** "
             "(v2), 74% lower -- v1's single most important caveat",
             "`LEAKAGE_AUDIT.html` Part D'; `FEATURE_LOCK_CONFIRMATION_XT.html`"],
            ["One threshold does not transfer",
             "the std-derived flat margin spans **53.4%** of v2's rows (35.4% on v1) and 24 of 31 features "
             "classify differently under a robust alternative",
             "every active report's `target_scale.threshold_transfer_warning`; section 1 of this document's "
             "prior version"],
            ["Correlation with the old target", "-0.016 (v1) -> **-0.1082** (v2): an order of magnitude "
                                                "stronger and correctly signed, still modest",
             "`active_numerical_target_atlas.html`; `XT_TARGET_PROTOTYPE_V2.md` s.2.4"],
        ]),
        h3("The superseded v1 portal"),
        p(f"<code>reports/analysis/{V1_PORTAL}/</code> holds Prompt 65's complete portal against "
          "<code>target_xt_delta</code> (v1), moved there with <code>git mv</code> so its history is "
          "preserved. Its contents are unchanged apart from a supersession banner. Every number in it is the "
          "accurate record of what was measured against v1 and is deliberately not corrected. "
          "<b>This pass did not touch it.</b>"),
        h3("What this pass changed on the active side"),
        p("The portal index and this document, and nothing else. Every active-leg report file under "
          "<code>reports/analysis/xt_target/</code> stands exactly as Prompt 67 rebuilt it, byte-for-byte. "
          "Every active-leg generator script under <code>src/eda/</code> is unedited and still reproduces its "
          "output. No parquet, no model artefact, and no entry in <code>feature_config.py</code> was modified "
          "by this pass on either leg."),
    ]


# --------------------------------------------------------------------------
# Section 4 -- the PASSIVE leg, in full
# --------------------------------------------------------------------------

def build_section_passive(scale, cat, flag, atlas, lk, cf, tr, v1, v2, inter, rev, fl) -> list[dict]:
    rs = scale["robust_scale"]
    g = scale["grain_comparison"]
    e = lk["part_e_prime_construction_coupling"]
    et = cat["columns"]["on_ball_event_type"]["categories"]
    flag_ranked = sorted(({"column": k, **v["lift_stats"]} for k, v in flag["columns"].items()),
                         key=lambda r: abs(r["lift"]), reverse=True)[:5]
    blocks = [
        h3("4.1 The target, at the grain this suite measures it"),
        p("<code>target_xt_delta_passive</code> lives in "
          "<code>outputs/prototypes/passive_xt_delta.parquet</code> at the UNIQUE-EVENT grain "
          f"({g['prompt_68_unique_event']['n_defined']:,} defined values) and is left-joined READ-ONLY onto "
          f"<code>data/features/passive_defense.parquet</code>'s {scale['n_rows']:,} defender-slot rows by "
          "<code>event_id</code> in memory. Neither file is ever written back."),
        table(["Statistic", "Unique event (Prompt 68)", "Defender slot (this suite)", "Active v2, for reference"], [
            ["mean", f"{g['prompt_68_unique_event']['mean']:+.5f}", f"{scale['mean']:+.5f}", "+0.00338"],
            ["std", f"{g['prompt_68_unique_event']['std']:.5f}", f"{scale['std']:.5f}", "0.05838"],
            ["skew", f"{g['prompt_68_unique_event']['skew']:+.3f}", f"{scale['skew']:+.3f}", "-0.306"],
            ["excess kurtosis", f"{g['prompt_68_unique_event']['excess_kurtosis']:.2f}",
             f"{scale['excess_kurtosis']:.2f}", "18.33"],
            ["% exactly zero", f"{g['prompt_68_unique_event']['pct_zero']:.1f}%", f"{scale['pct_zero']:.1f}%",
             "20.2%"],
            ["% negative", f"{g['prompt_68_unique_event']['pct_negative']:.1f}%",
             f"{scale['pct_negative']:.1f}%", "35.4%"],
            ["% positive", f"{g['prompt_68_unique_event']['pct_positive']:.1f}%",
             f"{scale['pct_positive']:.1f}%", "44.4%"],
            ["IQR", "n/a", f"{rs['iqr']:.6f}", "0.008829"],
            ["MAD", "n/a", f"{rs['mad']:.6f}", "0.003853"],
            ["n defined", f"{g['prompt_68_unique_event']['n_defined']:,}", f"{scale['n_defined']:,}", "55,761"],
        ]),
        note("Prompt 68's shape finding -- CONFIRMED, and MORE extreme at the grain this suite works at",
             esc(g["note"])),
        h3("4.2 The shared-target property, and what it does and does not affect"),
        note("carried forward from Prompt 68, and the single most important thing to hold while reading this "
             "half of the portal",
             esc(pc.PROMPT_68_SHARED_TARGET_FINDING)),
        p("<b>What this does NOT affect:</b> effect sizes, directional shares, ranks, and every threshold in "
          "this suite, all of which are computed from means and counts that a repeated y value paired with "
          "genuinely varying x values still estimates correctly. <b>What it DOES affect:</b> the standard "
          "error behind any row-level test. Welch's t-test appears in the Leakage Audit's Parts B/C/D and its "
          "nominal n there is roughly eight times its effective one. That caveat is stated in the report "
          "itself, and <b>no verdict anywhere in this half of the portal is gated on a p-value</b> -- every "
          "one turns on an effect size, a share, or a structural argument about how a column is built. The "
          "existing passive xg reports state the row grain but carry no such caveat (checked directly in "
          "<code>reports/analysis/xg_target/</code>), so this is an addition rather than a mirrored "
          "convention."),
        p("<b>No per-defender attribution scheme was invented, and that is explicitly out of scope</b> -- the "
          "same call Prompt 68 made. A lift or a category mean on this leg says 'events at which at least this "
          "defender was positioned this way tend to see the ball's threat move this way', not 'this defender "
          "caused it'."),
        h3("4.3 The threshold that had to be recomputed for this leg"),
        note("the std-derived flat margin transfers WORSE here than on the active leg -- recomputed, not "
             "inherited",
             esc(scale["threshold_transfer_warning"])),
        table(["Quantity", "v1 (active)", "v2 (active)", "passive (this pass)"], [
            ["translated flat margin", "0.005486", "0.004929", f"{scale['flat_margin']:.6f}"],
            ["as a multiple of MAD", "0.53x", "1.28x", f"{rs['flat_margin_over_mad']:.2f}x"],
            ["as a multiple of IQR", "0.29x", "0.56x", f"{rs['flat_margin_over_iqr']:.2f}x"],
            ["share of all rows inside it", "35.4%", "53.4%",
             f"**{rs['pct_rows_inside_flat_margin']:.1f}%**"],
            ["robust (MAD-derived) alternative", "0.001286", "0.000482", f"{rs['robust_flat_margin']:.6f}"],
            ["features classifying differently under the robust margin",
             "15 of 31", "24 of 31",
             f"**{atlas['n_features_classifying_differently_under_robust_margin']} of "
             f"{atlas['n_features_analyzed']}**"],
            ["excess kurtosis behind it", "8.85", "18.33", f"{scale['excess_kurtosis']:.2f}"],
        ]),
        p("<b>A naive transfer of the active leg's own absolute margin would have been wrong in a specific, "
          "identifiable direction.</b> Active v2's margin is 0.004929 against a std of 0.058385; this leg's "
          f"std is {scale['std']:.6f}, so the correctly-translated margin is {scale['flat_margin']:.6f} -- "
          "about 1.8x smaller. Reusing active's number would have made the 'too small to matter' band nearly "
          "twice as wide as it should be, and almost every verdict in this half of the portal is a SHAPE "
          "comparison, so a wider flat band makes categories and tournaments look more alike and understates "
          "divergence. The margin was recomputed from this leg's own standard deviation everywhere it is used "
          "(Numerical Target Atlas, Tournament Stability, both Slice Stratifications, Feature Interaction, "
          "Review Methodology). <b>The literal translated margin is KEPT as the headline</b> so the two halves "
          "of this portal stay like-for-like and the xg suite's convention is still the one being followed; "
          "the robust alternative is published beside it everywhere."),
        p("<b>One report needed no recomputation, and that was checked rather than assumed.</b> "
          + esc(cf["threshold_note"])),
        h3("4.4 Correlation with the old passive targets -- weaker than active's, stated plainly"),
        note("carried forward from Prompt 68", esc(pc.PROMPT_68_CORRELATION_FINDING)),
        p("Re-measured on this run at the defender-slot grain: "
          f"r={scale['correlation_with_old_target']['pearson_r']:+.4f} against "
          "<code>target_future_xg_10s</code> and "
          f"r={scale['correlation_with_old_target']['pearson_r_vs_shot_target']:+.4f} against "
          "<code>target_future_shot_10s</code>, over "
          f"n={scale['correlation_with_old_target']['n']:,} rows. Prompt 68 reported -0.0715 and -0.0629 at "
          "the unique-event grain; the small difference is the same defender-count re-weighting that makes the "
          "row-level distribution heavier-tailed, not a disagreement. Both are correctly signed and both are "
          "weaker than the active leg's own -0.1082 against the same xg target. <b>This matters for reading "
          "the Numerical Target Atlas:</b> the two old targets are what the existing passive feature rankings "
          "were built against, so a feature that ranks highly here and not there is not necessarily "
          "contradicting anything -- the targets are only loosely related. It also matters because Adaptation "
          "1 borrows <code>target_future_xg_10s</code>'s own standard-deviation fraction to set every "
          "threshold in this suite, so this is the one place where the old target still has a hand in the new "
          "one's numbers."),
        h3("4.5 Construction coupling -- LIVE on this leg, unlike the corrected active leg"),
        note("this is the passive leg's own most important caveat, and it has no counterpart in the xg or "
             "binary portals",
             esc(e["why_this_part_exists"])),
        table(["Measured on this run", "Value"], [
            ["strongest locked feature vs `xt_before`",
             f"`{e['strongest_vs_xt_before']['feature']}` at r={e['strongest_vs_xt_before']['r_vs_xt_before']:+.4f}"],
            ["strongest locked feature vs the DELTA",
             f"`{e['strongest_vs_target']['feature']}` at r={e['strongest_vs_target']['r_vs_target']:+.4f}"],
            ["active leg, v1, for scale", "`distance_to_attacking_box` at r=+0.5551 vs the target"],
            ["active leg, v2, for scale", "`distance_to_attacking_box` at r=+0.1431 vs the target (74% lower)"],
        ]),
        p("The first row is not itself a finding -- it is the definition of the term, since "
          "<code>xt_before</code> IS a grid lookup at <code>ball_x</code>/<code>ball_y</code>. The finding is "
          "what survives the subtraction, and it sits <b>between</b> the active leg's v1 and v2 figures, "
          "reached through a completely different mechanism. Prompt 67 was able to call the active leg's "
          "construction-coupling caveat substantially retired; <b>the equivalent caveat is LIVE here</b>, and "
          "a location-derived feature's relationship to this target should be read knowing it shares a "
          "definitional term with half of it. <b>No feature is dropped or proposed for dropping on this "
          "evidence</b> and <code>feature_config.py</code> is unchanged. Whether a model on this leg should "
          "use <code>ball_x</code>/<code>ball_y</code> at all for this target, and whether "
          "<code>xt_before</code> should be offered as a feature in its own right, are modelling-ladder "
          "questions and are explicitly not decided here."),
        h3("4.6 Findings, report by report"),
        p("<b>Category Atlas.</b> The active leg's headline Clearance finding does not transfer and is not "
          "forced into a counterpart: <code>event_type</code> does not exist in "
          "<code>passive_defense.parquet</code> (checked against the live schema), and this leg's nearest "
          "relative <code>on_ball_event_type</code> describes the ATTACKING action the snapshot is anchored "
          f"to. On its own terms: across {len(et)} categories, <code>{esc(et[0]['category'])}</code> is the "
          f"most threat-reducing at {et[0]['mean_xt']:+.6f} (n={et[0]['n']:,}) and "
          f"<code>{esc(et[-1]['category'])}</code> the most threat-increasing at {et[-1]['mean_xt']:+.6f} "
          f"(n={et[-1]['n']:,})."),
        p("<b>Flag Ledger.</b> The active ledger's off-pool possession-flag section has no counterpart here: "
          "none of the three <code>action_*_possession</code> columns exists in the passive parquet, asserted "
          "against the live schema on this run. <code>action_ended_possession</code> still sets "
          "<code>xt_after = 0</code> for this target, but it is read from "
          "<code>events_with_targets.parquet</code> during Prompt 68's build and never lands in the passive "
          "feature table, so <b>no passive feature carries that construction coupling</b>. Top 5 by |lift|:"),
        table(["Flag", "lift", "mean True", "mean False", "n True"],
              [[f"`{r['column']}`", f"{r['lift']:+.6f}", f"{r['mean_xt_true']:+.6f}",
                f"{r['mean_xt_false']:+.6f}", f"{r['n_true']:,}"] for r in flag_ranked]),
        p("<b>Numerical Target Atlas.</b> "
          f"{atlas['n_features_whose_binned_curve_crosses_zero']}/{atlas['n_features_analyzed']} features' "
          "binned curves cross zero -- they separate threat-reducing from threat-increasing events rather than "
          f"only varying in magnitude. {atlas['n_features_flagged_inconsistent']} feature(s) flagged "
          "train/test-inconsistent. Top 5 by |Spearman rho|:"),
        table(["Feature", "Spearman rho", "Shape", "Curve crosses zero"],
              [[f"`{f['feature']}`", str(f["spearman_rho"]), str(f["shape"]),
                str(f.get("binned_curve_crosses_zero"))] for f in atlas["features"][:5]]),
        note("read these shape labels with section 4.3 in hand",
             f"{atlas['n_features_classifying_differently_under_robust_margin']} of "
             f"{atlas['n_features_analyzed']} features classify differently under the MAD-derived flat margin "
             "than under the std-derived one used above (against 24 of 31 on the active leg's v2 target and 15 "
             "of 31 on v1). The <b>Shape</b> column is indicative on this target, not decided."),
        p("<b>Leakage Audit.</b> Part A clean. Parts B/C/D mirror the passive xg audit's own columns "
          "literally -- no substitution was needed, because leakage auditing in this project IS a passive "
          "methodology and it is the ACTIVE leg that has no native counterpart. "
          f"<code>has_screened_outcome</code>'s exactly-zero share gap is "
          f"{lk['part_c_has_screened_outcome']['zero_share_gap_pp']:+.1f}pp with a mean difference of "
          f"{lk['part_c_has_screened_outcome']['delta']:+.6f}; the exclusion holds on the censoring MECHANISM, "
          "which is a property of how the column is built rather than of which target it is measured against, "
          "so this target's number is the same structural argument measured a third time and should not be "
          "cited as fresh evidence. <code>has_option_2</code>/<code>has_option_3</code> stay IN the candidate "
          "list, the same disposition the binary and xg audits reached. Part E' is section 4.5 above."),
        p("<b>Confound (Reversal) Testing.</b> All five established passive tests mirrored literally -- same "
          "columns, same labels, same <code>_verdict()</code> function reused byte-for-byte. <b>This report "
          "contains no new test and needed no invention</b>, in direct contrast to the active half, where "
          "Prompt 65 found exactly one active confound test in the whole repo and had to write two of its "
          "own. Verdicts on this target:"),
        table(["Test", "Unconditional", "Given a non-zero delta"],
              [[t["title"], t["verdict"]["verdict"], t["given_nonzero_delta"]["verdict"]["verdict"]]
               for t in cf["tests"]]),
        p("<b>Tournament Stability Check.</b> "
          f"{sum(1 for f in tr['features'] if f['verdict'] == 'genuine tournament-level difference')} of "
          f"{len(tr['features'])} passive features show a genuine tournament-level difference rather than "
          "arbitrary train/test noise, and "
          f"{tr['threshold_recomputation']['n_features_whose_verdict_flips_under_robust_margin']} of "
          f"{len(tr['features'])} verdicts would change under the MAD-derived robust margin -- a check added "
          "here that neither the xg nor the active xT version carries, precisely because this leg's margin is "
          "wide enough that a shape verdict could rest on it."),
        table(["Feature", "Verdict", "Flips under the robust margin"],
              [[f"`{f['feature']}`", f["verdict"],
                str(f["robust_margin_recheck"]["verdict_flips_under_robust_margin"])]
               for f in tr["features"]]),
        p("<b>Slice Stratification.</b> "
          f"V1: {v1['n_cells']} cells, {v1['cross_dataset_summary']['n_genuine_divergences']} genuine "
          "divergences unconditionally and "
          f"{v1['cross_dataset_summary']['n_genuine_divergences_given_nonzero_delta']} once the zero mass is "
          f"removed, with {v1['cross_dataset_summary']['n_conditioning_disagreements']} categories where that "
          f"removal flips the conclusion. V2: {v2['n_cells']} cells, "
          f"{v2['cross_dataset_summary']['n_genuine_divergences']} / "
          f"{v2['cross_dataset_summary']['n_genuine_divergences_given_nonzero_delta']}."),
        note("the one comparison this leg can make and the active leg cannot",
             esc(v2["cross_dataset_summary"]["closing_note_archetype_vs_boolean"])),
        p("<b>Feature Interaction Analysis.</b> "
          f"{inter['n_pairs_where_unconditional_and_conditional_agree']}/{inter['n_pairs_total']} pairs "
          "classify the same way unconditionally and on the non-zero-delta subset; "
          f"{inter['n_pairs_with_sign_flipping_stratum_deltas']}/{inter['n_pairs_total']} have within-stratum "
          "deltas that change SIGN -- feature A's effect reverses DIRECTION across levels of feature B, a "
          "finding that cannot arise on the xG target at all. All five pairs involve this leg's own screening, "
          "option or engagement columns and could not have been run on the active half."),
        table(["Feature A", "Feature B", "Classification", "Stratum deltas change sign"],
              [[f"`{r['feature_a']}`", f"`{r['feature_b']}`", r["classification"],
                "yes" if r["within_stratum_deltas_change_sign"] else "no"]
               for r in inter["summary_table"]]),
        p("<b>Review Methodology.</b> "
          f"{rev['datasets']['passive']['n_review_pairs']} PASSIVE review pairs; "
          f"{len(rev['datasets']['passive']['needs_human_call'])} remain needs_human_call (dominated, as on "
          "every target and both legs, by continuous-continuous pairs no rule covers); "
          f"{rev['datasets']['passive']['n_type2_opposite_sign_pairs']} Type-2 pair(s) resolve on the "
          "opposite-sign rule. Types 1, 3 and 4 are target-independent and their resolvers are imported "
          "unchanged, so those verdicts match the binary and xg passive reviews <b>by construction, not by "
          "coincidence</b>."),
        p("<b>Feature Lock Confirmation.</b> The locked PASSIVE feature set "
          f"({fl['confirmed_counts']['passive']} features, matching the expected "
          f"{fl['confirmed_counts']['passive_expected']}) is confirmed internally consistent. The correlation "
          "diff is reused from the binary confirmation and labelled identical-by-construction rather than "
          "silently recomputed. <b>No feature is added, dropped or re-tiered</b>, on either leg."),
    ]
    return blocks


# --------------------------------------------------------------------------
# Section 5 -- caveats and open questions
# --------------------------------------------------------------------------

def build_section_caveats(scale, atlas) -> list[dict]:
    rs = scale["robust_scale"]
    return [
        p("<b>Nothing in this portal changes <code>feature_config.py</code>, on either leg, and no model "
          "artefact was touched.</b> No feature is added, dropped or re-tiered."),
        h3("Passive-leg caveats recorded by this pass"),
        ul([
            "<b>Caveat P1 -- construction coupling, LIVE on this leg.</b> <code>xt_before = "
            "xT(ball_x, ball_y)</code> and both columns are locked features, so one of the target's two terms "
            "is a deterministic function of the candidate set. What survives the subtraction is measured in "
            "section 4.5 and sits between the active leg's v1 and v2 figures. Unlike the active leg's, this "
            "caveat is NOT retired.",
            "<b>Caveat P2 -- effects can sit in the zero mass.</b> The same caveat Prompt 65 recorded for the "
            f"active leg, applying to MORE rows here: {scale['pct_zero']:.1f}% of passive rows are exactly "
            "zero against 20.2% of active v2 rows. Every table in this half of the portal reports negative / "
            "zero / positive shares beside its mean, and the Leakage Audit's Parts B/C/D are read on the "
            "zero-share gap as well as the mean.",
            "<b>Caveat P3 -- the flat-margin threshold describes this distribution even less well than "
            f"active's.</b> It spans {rs['pct_rows_inside_flat_margin']:.1f}% of rows "
            f"({rs['flat_margin_over_mad']:.2f}x the MAD) and "
            f"{atlas['n_features_classifying_differently_under_robust_margin']} of "
            f"{atlas['n_features_analyzed']} features classify differently under the robust alternative. "
            "Recomputed from this leg's own std rather than transferred, and reported rather than swapped.",
            "<b>Caveat P4 -- rows are not independent within an event.</b> One target value per "
            "<code>event_id</code>, repeated across ~8 defender slots. Row-level p-values are optimistic by "
            "roughly that factor. No verdict in this half of the portal is gated on one.",
        ]),
        note("methodological warning for anyone extending the passive half",
             "This target is the hardest distribution anywhere in this portal to work with -- excess kurtosis "
             "~35 at the row grain against active v2's 18.33 and v1's 8.85, with a strongly left-skewed tail. "
             "Prefer rank-based, count-based, ratio-based and sign-based instruments: every one of them "
             "transferred to this leg without a scratch (the confound suite's sign-only "
             "<code>_verdict()</code>, <code>SMALL_N_THRESHOLD</code>'s row count, the substitutive/additive "
             "ratios, <code>MIN_CELL_N_FOR_DELTA</code>, the quantile-decile binning). The ONE instrument that "
             "did not transfer is the only standard-deviation-based one in the suite, and it did not transfer "
             "on the active leg either. That pattern has now held across three targets."),
        h3("Open questions this portal does NOT answer"),
        ul([
            "<b>Per-defender attribution for the passive target.</b> Explicitly out of scope, the same call "
            "Prompt 68 made. The target is a property of the event and no scheme for apportioning it across "
            "the defenders present was built or is proposed.",
            "<b>Whether a model on the passive leg should use <code>ball_x</code>/<code>ball_y</code> for this "
            "target</b>, given Caveat P1, and whether <code>xt_before</code> should be offered as a feature in "
            "its own right rather than left implicit. Both are modelling-ladder questions.",
            "<b>What model architecture suits a left-skewed, very heavy-tailed, sign-carrying target</b> -- "
            "Prompt 68 found this leg <i>less</i> like the existing hurdle architecture's shape than active v2, "
            "which was already less like it than v1. This is an EDA portal and takes no position.",
            "<b>Whether the passive leg's <code>xt_before</code> should be defined at the previous event's "
            "location</b>, as the active leg's is, rather than at the snapshot's own. Prompt 68 made the "
            "opposite choice deliberately and justified it (the passive snapshot's <code>ball_x</code>/"
            "<code>ball_y</code> IS the ball's location at the anchoring event), but the asymmetry between the "
            "two legs' <code>xt_before</code> definitions is recorded here rather than resolved.",
            "<b>Whether the log-transform question should be reopened for this leg specifically.</b> Not "
            "applied anywhere in this report suite, on either leg, because it would break comparability with "
            "the xg portal and with this portal's own other half. This leg's tails make it the most tempting "
            "case in the project. Recorded for the modelling legs.",
            "<b>A passive Player-Level Validity Check and Player-Grouped Split Check.</b> Both are "
            "ACTIVE-ONLY project-wide for a structural reason (the passive leg is not player-indexed the same "
            "way), and both are target-independent, so building them would be outside this pass's remit even "
            "if the data supported it. Stated in the Step-0 table rather than left as an unexplained absence.",
        ]),
    ]


# --------------------------------------------------------------------------
# Section 6 -- document map
# --------------------------------------------------------------------------

def build_section_map() -> list[dict]:
    return [
        h3("Active leg (built by Prompts 65/67 -- untouched by this pass)"),
        table(["Report", "Files"], [
            ["Active -- Category Atlas", "`active_category_atlas.html` / `.json`"],
            ["Active -- Flag Ledger", "`active_flag_ledger.html` / `.json`"],
            ["Active -- Distribution Atlas",
             "`active_distribution_atlas.html`, `active_distribution_atlas_reconstructed.html`, `.json`"],
            ["Active -- Numerical Target Atlas", "`active_numerical_target_atlas.html` / `.json`"],
            ["Active -- Review Methodology", "`REVIEW_METHODOLOGY.html`, `REVIEW_ANALYSIS.json`"],
            ["Active -- Leakage Audit", "`LEAKAGE_AUDIT.html` / `.json`"],
            ["Active -- Confound (Reversal) Testing", "`CONFOUND_ANALYSIS.html` / `.json`"],
            ["Active -- Tournament Stability Check", "`TOURNAMENT_STABILITY_CHECK.html` / `.json`"],
            ["Active -- Slice Stratification V1", "`SLICE_STRATIFICATION.html` / `.json`"],
            ["Active -- Slice Stratification V2", "`SLICE_STRATIFICATION_V2.html` / `.json`"],
            ["Active -- Feature Interaction Analysis", "`FEATURE_INTERACTION_ANALYSIS.html` / `.json`"],
            ["Active -- Feature Lock Confirmation", "`FEATURE_LOCK_CONFIRMATION_XT.html` / `.json`"],
        ]),
        h3("Passive leg (built by this pass)"),
        table(["Report", "Files"], [
            ["Passive -- Category Atlas", "`passive_category_atlas.html` / `.json`"],
            ["Passive -- Flag Ledger", "`passive_flag_ledger.html` / `.json`"],
            ["Passive -- Distribution Atlas",
             "`passive_distribution_atlas.html`, `passive_distribution_atlas_reconstructed.html`, `.json`"],
            ["Passive -- Numerical Target Atlas", "`passive_numerical_target_atlas.html` / `.json`"],
            ["Passive -- Review Methodology",
             "`PASSIVE_REVIEW_METHODOLOGY.html`, `PASSIVE_REVIEW_ANALYSIS.json`"],
            ["Passive -- Leakage Audit", "`PASSIVE_LEAKAGE_AUDIT.html` / `.json`"],
            ["Passive -- Confound (Reversal) Testing", "`PASSIVE_CONFOUND_ANALYSIS.html` / `.json`"],
            ["Passive -- Tournament Stability Check", "`PASSIVE_TOURNAMENT_STABILITY_CHECK.html` / `.json`"],
            ["Passive -- Slice Stratification V1", "`PASSIVE_SLICE_STRATIFICATION.html` / `.json`"],
            ["Passive -- Slice Stratification V2", "`PASSIVE_SLICE_STRATIFICATION_V2.html` / `.json`"],
            ["Passive -- Feature Interaction Analysis",
             "`PASSIVE_FEATURE_INTERACTION_ANALYSIS.html` / `.json`"],
            ["Passive -- Feature Lock Confirmation",
             "`PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.html` / `.json`"],
        ]),
        h3("Shared"),
        table(["Report", "Files"], [
            ["Master Findings (this document, both legs)", "`MASTER_FINDINGS.md` / `.html`"],
            ["Portal index (Active + Passive groups)", "`INDEX.html`"],
        ]),
        h3("Linked, NOT rebuilt (target-independent -- properties of the features, not of the target)"),
        p("See the Step-0 table in section 2 for which of these have a passive half and which are ACTIVE-ONLY "
          "project-wide. A change to which target the features are measured against cannot move a "
          "feature-vs-feature statistic, so rebuilding any of them would produce byte-identical output."),
        h3("New generator scripts (all added as siblings; no existing script was edited)"),
        p("Nothing under <code>src/eda/</code> that already existed was modified. Prompt 65's 21 "
          "<code>*_xt.py</code> generators, Prompt 67's 8 <code>*_xt_v2*.py</code> siblings, and the "
          "<code>xg_target</code> and <code>shot_target</code> portals' own generators all remain exactly "
          "reproducible."),
        ul([
            "`src/eda/xt_passive_common.py` -- the passive re-point layer, Adaptation 1 recomputed for this "
            "leg, and Prompt 68's carried-forward findings",
            "`src/eda/generate_reports_xt_passive.py` -- Category Atlas + Flag Ledger",
            "`src/eda/generate_distribution_atlas_xt_passive.py` -- Distribution Atlas, both variants",
            "`src/eda/generate_numerical_xt_target_passive.py` -- Numerical Target Atlas (compute + render)",
            "`src/eda/generate_leakage_audit_xt_passive.py` -- Leakage Audit (compute + render)",
            "`src/eda/generate_confound_analysis_xt_passive.py` -- Confound Testing (compute + render)",
            "`src/eda/generate_tournament_stability_check_xt_passive.py` -- Tournament Stability",
            "`src/eda/generate_slice_stratification_xt_passive.py` -- Slice Stratification V1 + V2",
            "`src/eda/generate_feature_interaction_analysis_xt_passive.py` -- Feature Interaction",
            "`src/eda/generate_review_analysis_xt_passive.py` -- Review Methodology",
            "`src/eda/generate_feature_lock_confirmation_xt_passive.py` -- Feature Lock Confirmation",
            "`src/eda/generate_master_findings_xt_passive.py` (this document)",
            "`src/eda/generate_eda_portal_xt_passive.py` -- portal shell for BOTH legs, AND the driver that "
            "makes the passive re-point work",
        ]),
        p("Rebuild the passive half and the shared documents with: <code>.venv/Scripts/python.exe -m "
          "src.eda.generate_eda_portal_xt_passive --all</code>. Rebuild the active half, unchanged, with: "
          "<code>.venv/Scripts/python.exe -m src.eda.generate_eda_portal_xt_v2 --all</code>. <b>Run them in "
          "separate processes</b> -- both re-point the same <code>xt_common</code> module attributes, and "
          "whichever imports last would win."),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cat = _load("passive_category_atlas.json")
    flag = _load("passive_flag_ledger.json")
    atlas = _load("passive_numerical_target_atlas.json")
    lk = _load("PASSIVE_LEAKAGE_AUDIT.json")
    cf = _load("PASSIVE_CONFOUND_ANALYSIS.json")
    tr = _load("PASSIVE_TOURNAMENT_STABILITY_CHECK.json")
    v1 = _load("PASSIVE_SLICE_STRATIFICATION.json")
    v2 = _load("PASSIVE_SLICE_STRATIFICATION_V2.json")
    inter = _load("PASSIVE_FEATURE_INTERACTION_ANALYSIS.json")
    rev = _load("PASSIVE_REVIEW_ANALYSIS.json")
    fl = _load("PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.json")
    a_atlas = _load("active_numerical_target_atlas.json")

    scale = atlas["target_scale"]
    a_scale = a_atlas["target_scale"]
    rs = scale["robust_scale"]
    e = lk["part_e_prime_construction_coupling"]

    active_count = sum(len(ACTIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))
    passive_count = sum(len(PASSIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))

    bottom_line = (
        "<b>This portal now covers BOTH legs of the xT-delta target family.</b> Its active half "
        "(<code>target_xt_delta_v2</code>, Prompts 64/66/67) is unchanged by this pass and is summarised by "
        "reference in section 3. Its new passive half (<code>target_xt_delta_passive</code>, Prompt 68) is "
        "written out in full in section 4. <b>Step 0 found that all twelve target-dependent report types "
        "needed a genuine passive build</b> -- none had a passive-side xT equivalent anywhere, and eight of "
        "them are single files carrying both legs in the <code>xg_target</code> precedent, a convention this "
        "pass could not mirror without editing frozen active-leg output. Three findings dominate the passive "
        f"half. <b>One:</b> the target is the hardest distribution in the portal -- skew {scale['skew']:+.3f} "
        f"and excess kurtosis {scale['excess_kurtosis']:.2f} at the defender-slot grain, against Prompt 68's "
        "-1.779/31.02 at the event grain and active v2's -0.306/18.33 -- and the std-derived flat margin "
        f"consequently spans <b>{rs['pct_rows_inside_flat_margin']:.1f}% of all rows</b> against active's "
        f"53.4%, with {atlas['n_features_classifying_differently_under_robust_margin']} of "
        f"{atlas['n_features_analyzed']} features classifying differently under a robust alternative. It was "
        "RECOMPUTED from this leg's own std, not transferred. <b>Two:</b> construction coupling is LIVE here, "
        "unlike on the corrected active leg -- <code>xt_before = xT(ball_x, ball_y)</code> and both columns "
        f"are locked features, leaving <code>{esc(e['strongest_vs_target']['feature'])}</code> at "
        f"r={e['strongest_vs_target']['r_vs_target']:+.4f} against the delta, between active's v1 (+0.5551) "
        "and v2 (+0.1431). <b>Three:</b> the target is one value per event repeated across ~8 defender rows, "
        "so ranks and effect sizes read normally but row-level p-values do not, and no verdict here is gated "
        "on one. No feature was added, dropped or re-tiered on either leg, no model artefact was touched, and "
        "the target-independent reports were linked rather than rebuilt."
    )

    sections = [
        ("The xT target family: what both legs share, and where they differ",
         build_section_family(a_scale, scale)),
        ("Step 0 -- which report types are per-leg vs shared, confirmed from real files",
         build_section_step0()),
        ("The ACTIVE leg (Prompts 64/66/67) -- by reference", build_section_active()),
        ("The PASSIVE leg (this pass) -- in full",
         build_section_passive(scale, cat, flag, atlas, lk, cf, tr, v1, v2, inter, rev, fl)),
        ("Caveats recorded, and open questions this portal does not answer",
         build_section_caveats(scale, atlas)),
        ("Document map", build_section_map()),
    ]

    # --- Markdown ---
    md_parts = [
        "# Master Findings -- the xT-Delta Target Family (Active + Passive Legs)",
        "",
        "> **Both legs are covered.** The ACTIVE-BINARY leg is built against `target_xt_delta_v2` (Prompts "
        f"64/66/67) and is unchanged by this pass; the prior pass against `target_xt_delta` (v1) is preserved "
        f"unchanged at `reports/analysis/{V1_PORTAL}/`. The PASSIVE leg is built against "
        "`target_xt_delta_passive` (Prompt 68) and is new here. Neither leg is out of scope any more.",
        "",
        "> **Reading order.** Section 1 is the shared target-family preamble. Section 2 is the Step-0 table. "
        "Section 3 covers the active leg BY REFERENCE -- it cites and links rather than restating tables that "
        "already exist in the linked documents and in this file's own prior version. Section 4 is this pass's "
        "own passive findings, in full.",
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
    # the .html twin but leaks entities into Markdown (both prior versions of
    # this document have the same artefact). Unescaped here so the .md reads
    # as prose; the .html is unaffected because it is built from the same
    # blocks, not from this text.
    for ent, ch in (("&#x27;", "'"), ("&quot;", '"'), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")):
        md = md.replace(ent, ch)
    MD_PATH.write_text(md + "\n", encoding="utf-8")

    # --- HTML, from the same block data ---
    html_parts = [
        '<div class="finding" style="margin-bottom:20px;"><span class="tag">scope -- BOTH legs</span>'
        '<p><b>This portal covers the ACTIVE-BINARY leg and the PASSIVE leg.</b> The active half is built '
        'against <code>target_xt_delta_v2</code> (Prompts 64/66/67) and is unchanged by this pass; its '
        'superseded v1 pass is preserved at <code>reports/analysis/' + V1_PORTAL + '/</code>. The passive '
        'half is built against <code>target_xt_delta_passive</code> (Prompt 68) and is new here. Any earlier '
        'statement in this portal that the passive leg is out of scope is superseded.</p></div>',
        '<div class="finding flag" style="margin-bottom:20px;"><span class="tag">how to read this document'
        '</span><p>Section 1 is the shared target-family preamble. Section 2 is the Step-0 table. Section 3 '
        'covers the active leg <b>by reference</b> -- it cites and links rather than restating tables that '
        'already exist in the linked documents and in this file\'s own prior version. Section 4 is this '
        'pass\'s own passive findings, in full.</p></div>',
        f'<div class="finding" style="margin-bottom:28px;"><span class="tag">bottom line up front</span>'
        f'<p>{bottom_line}</p></div>',
    ]
    for i, (title, blocks) in enumerate(sections, start=1):
        html_parts.append(f'<h2 class="mf-section-title" id="s{i}">{i}. {esc(title)}</h2>')
        html_parts.append(render_blocks_html(blocks))

    html = render.render_article(
        eyebrow="MASTER FINDINGS -- XT DELTA &middot; ACTIVE + PASSIVE LEGS",
        title="The xT-Delta Target Family -- Master Findings (Both Legs)",
        dek=(
            "The entry point for this portal: what an xT delta is and what the two legs share, the Step-0 "
            "disposition of every report type, the active leg's findings by reference, and the passive leg's "
            "findings in full -- including the threshold that had to be recomputed for it, the construction "
            "coupling that is live on it, and the shared-target row grain every number on it depends on."
        ),
        stats=[
            (f"{active_count}", "active features"),
            (f"{passive_count}", "passive features"),
            ("12", "passive reports built here"),
            ("5", "target-independent types linked"),
            (f"{rs['pct_rows_inside_flat_margin']:.0f}%", "rows inside the passive flat margin"),
            (f"{e['strongest_vs_target']['r_vs_target']:+.3f}", "strongest passive coupling vs the delta"),
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

    print(f"Step 0: {len(STEP_0_TARGET_DEPENDENT)} target-dependent report types classified "
          f"({sum(1 for r in STEP_0_TARGET_DEPENDENT if 'GENUINE PASSIVE BUILD' in r[3])} needed a genuine "
          f"passive build), {len(STEP_0_TARGET_INDEPENDENT)} target-independent types checked "
          f"({sum(1 for r in STEP_0_TARGET_INDEPENDENT if r[2].startswith('**YES'))} already have a passive "
          "half)")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
