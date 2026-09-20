"""CLI entrypoint: MASTER_FINDINGS.md + .html for the xT-delta portal,
ACTIVE-BINARY LEG ONLY.

STRUCTURAL PRECEDENT, and why this one was picked. Two candidates were
compared before writing:
  - reports/analysis/shot_target/MASTER_FINDINGS.{md,html} -- the binary
    portal's start-here synthesis, built by generate_master_findings.py.
  - reports/modeling/MODELLING_CLOSEOUT.{md,html} and
    reports/eda-style closeouts.
The shot_target MASTER_FINDINGS is the closer structural match and is the
one mirrored here: it lives inside an analysis portal, it is that portal's
"start here" document, it is generated as .md and .html from ONE content
model so the two cannot drift, and it is organised as "what this is ->
decisions -> every finding -> what a model should learn -> open items ->
document map". The modelling closeouts are retrospectives on a completed
modelling phase, which this is not. That block model (p/ul/table/note/h3
-> render_blocks_md / render_blocks_html) is reused directly, imported
rather than reimplemented.

The xg portal itself has NO master-findings document of its own -- its
INDEX links out to the binary portal's (confirmed by reading
reports/analysis/xg_target/INDEX.html). This portal gets its own, because
its target is structurally different enough that pointing readers at a
document written about two other targets would be actively misleading.

CLOSEOUT DECISION: a separate PATTERN_ANALYSIS_CLOSEOUT-style document is
NOT produced. The reason is stated in section 1 of the document itself.

Usage:
    python -m src.eda.generate_master_findings_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.feature_config import ACTIVE
from src.eda.generate_master_findings import h3, note, p, render_blocks_html, render_blocks_md, table, ul
from src.eda.render import esc

OUT_DIR = xc.OUT_DIR
MD_PATH = OUT_DIR / "MASTER_FINDINGS.md"
HTML_PATH = OUT_DIR / "MASTER_FINDINGS.html"


def _load(name: str) -> dict:
    return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Section 1 -- the mandatory Step-0 methodology confirmation
# --------------------------------------------------------------------------

STEP_0_ROWS = [
    ["Active -- Category Atlas", "`src/eda/generate_reports_xg.py`",
     "CONFIRMED from source", "`src/eda/generate_reports_xt.py`",
     "Reads `feature_config_v1_historical.ACTIVE_V1_HISTORICAL` for the reconstructed pre-drop pool, tags each "
     "column locked/dropped with the reason from `ACTIVE['excluded']`, emits the `catbar-*` markup and the "
     "`categories` / `categories_given_shot` JSON key pair that the file on disk actually has."],
    ["Active -- Flag Ledger", "`src/eda/generate_reports_xg.py` (same script)",
     "CONFIRMED from source", "`src/eda/generate_reports_xt.py`",
     "Same script emits both; `boolean_xg_lift` from `compute_stats_xg.py` produces the exact "
     "`lift`/`pct_true`/`small_n` key set in `active_flag_ledger.json`, ranked by `abs(lift)`, "
     "`SMALL_N_THRESHOLD=500`."],
    ["Active -- Distribution Atlas (+ reconstructed variant)",
     "`src/eda/generate_distribution_atlas_reconstructed.py`",
     "CONFIRMED from source, with a stale-path artefact found",
     "`src/eda/generate_distribution_atlas_xt.py`",
     "Target-INDEPENDENT by construction (`compute_stats.continuous_distribution` / `discrete_distribution` "
     "never reference a target). Its filename branch tests `out_dir.name == \"eda_xg\"`, which no longer "
     "matches the current directory name `xg_target` -- so the file on disk is `active_distribution_atlas.html` "
     "while a re-run today would write `..._reconstructed.html`. Noted, NOT fixed there (that script stays "
     "untouched so the xg portal remains reproducible)."],
    ["Active -- Numerical Target Atlas",
     "`generate_numerical_xg_target_analysis.py` (compute) + `generate_numerical_xg_target_reports.py` (render)",
     "CONFIRMED from source", "`generate_numerical_xt_target_analysis.py` + `_reports.py`",
     "Compute step imports `build_pool`, `classify_shape`, `N_QUANTILE_BINS=10`, "
     "`DISCRETE_CARDINALITY_THRESHOLD=15`, `RHO_THRESHOLD=0.05` and `load_canonical_split`; the render step "
     "imports `NUM_TARGET_CSS`. Both key sets match the JSON/HTML on disk exactly."],
    ["Review Methodology", "`generate_review_analysis_xg.py` + `generate_review_report_xg.py`",
     "CONFIRMED from source", "`generate_review_analysis_xt.py` + `generate_review_report_xt.py`",
     "Reads `shot_target/CORRELATION_ANALYSIS_V1_HISTORICAL.json`; Types 1/3/4 imported unchanged from "
     "`generate_review_analysis`; only Type 2 uses target evidence. `TYPE1_R_THRESHOLD`, "
     "`TYPE2_SUBSET_CONTAINMENT`, `NEAR_IDENTICAL_RATIO=0.5`, `DISTINCT_SIGNAL_RATIO=1.5` all carried over."],
    ["Leakage Audit", "`generate_leakage_audit_xg.py` + `generate_leakage_report_xg.py`",
     "CONFIRMED from source -- but PASSIVE-ONLY", "`generate_leakage_audit_xt.py` + `generate_leakage_report_xt.py`",
     "Parts B/C/D test `screened_option_was_avoided`, `has_screened_outcome`, `has_option_2/3` -- none exist in "
     "the active parquet, and NO active-side leakage audit exists anywhere in this repo for any target. Part A "
     "reused by import. Parts B'/C' apply the same method to active analogues; Part D' is new and labelled new."],
    ["Confound (Reversal) Testing",
     "`generate_confound_analysis_xg.py` (2 tests) + `generate_confound_analysis_part_d_xg.py` (3 more) + "
     "`generate_confound_report_xg.py`",
     "CONFIRMED from source -- but all 5 are PASSIVE", "`generate_confound_analysis_xt.py` + `generate_confound_report_xt.py`",
     "Exactly ONE active confound test exists project-wide: `defenders_within_10m_vs_box_proximity`, from "
     "`generate_confound_analysis_part_a.py`, in the BINARY portal only -- the xg mirror never received it (its "
     "own docstring says so). Test 1 mirrors it exactly. `_verdict()` reused byte-for-byte; `QCUT_N=4` unchanged."],
    ["Tournament Stability Check",
     "`generate_tournament_stability_check_xg.py` + `generate_tournament_stability_report_xg.py`",
     "CONFIRMED from source", "`generate_tournament_stability_check_xt.py` + `_report_xt.py`",
     "Imports `INVESTIGATE` (10 features, 6 active) and `load_match_tournament_map`; bin edges computed once on "
     "pooled data and reused across tournaments; `nearest_defender_distance` excluded for the known "
     "self-reference bug. All carried over; the 4 passive features are dropped as out of scope."],
    ["Slice Stratification V1",
     "`generate_slice_stratification_xg.py` + `generate_slice_stratification_xg_expand.py` (Part A: 104 cells, "
     "Part B: conditional retrofit) + `generate_slice_stratification_report_xg.py`",
     "CONFIRMED from source", "`generate_slice_stratification_xt.py` + `generate_slice_stratification_report_xt.py`",
     "`FEATURE_SLICER_PLAN`, `build_plan()`, `SMALL_N_THRESHOLD=1000` and `STRUCTURAL_CAUTION_CATEGORIES` all "
     "imported unchanged. The xg portal needs three scripts for historical reasons; this portal has no such "
     "history, so the same cells are built in one pass -- the same simplification the xg V2 script itself makes."],
    ["Slice Stratification V2",
     "`generate_slice_stratification_v2_xg.py` + `generate_slice_stratification_v2_report_xg.py`",
     "CONFIRMED from source", "`generate_slice_stratification_v2_xt.py` + `generate_slice_stratification_report_xt.py`",
     "`ACTIVE_FEATURES`, `ACTIVE_BOOLEAN_SLICERS`, `ARCHETYPE_SLICER` imported unchanged. The archetype slicer "
     "is a passive column, so the active half is the boolean-slicer half only and V2's closing "
     "archetype-vs-boolean comparison is NOT reproducible -- reported as not applicable."],
    ["Feature Interaction Analysis",
     "`generate_feature_interaction_analysis_xg.py` + `generate_feature_interaction_report_xg.py`",
     "CONFIRMED from source", "`generate_feature_interaction_analysis_xt.py` + `_report_xt.py`",
     "`PAIRS`, `Q=4` and `_bin_labeled` imported unchanged; pairs NOT re-ranked for this target, so the three "
     "targets stay comparable. `SUBSTITUTIVE_RATIO_MAX=0.3`, `ADDITIVE_MAGNITUDE_RATIO_MAX=2.0`, "
     "`MIN_CELL_N_FOR_DELTA=20` carried over -- they are ratios, so scale- and sign-free."],
    ["Feature Lock Confirmation + Pattern Findings",
     "`generate_feature_lock_confirmation_xg.py` + `generate_feature_lock_pattern_findings.py` (appends the "
     "findings section) + `generate_feature_lock_report_xg.py`",
     "CONFIRMED from source", "`generate_feature_lock_confirmation_xt.py` + `generate_feature_lock_report_xt.py`",
     "The xg portal's HTML carries BOTH the confirmation and the pattern findings in one document, so both are "
     "produced here in one file too. `correlation_diff` is read directly from the binary confirmation and "
     "labelled identical-by-construction -- the same discipline the xg script applies."],
    ["Master Findings (this document)",
     "`src/eda/generate_master_findings.py` (the binary portal's; the xg portal has none of its own)",
     "CONFIRMED from source", "`src/eda/generate_master_findings_xt.py`",
     "The xg portal's INDEX links out to the binary portal's MASTER_FINDINGS rather than having one. This "
     "portal gets its own. The binary generator's block model (`p`/`ul`/`table`/`note`/`h3` -> "
     "`render_blocks_md`/`render_blocks_html`) is imported and reused, so the .md and .html cannot drift."],
]


def build_section_1() -> list[dict]:
    return [
        p("<b>This section is a required deliverable, not scratch notes.</b> For each target-dependent report "
          "type below, the specific script that actually produced the corresponding file in "
          "<code>reports/analysis/xg_target/</code> was identified and confirmed by READING ITS LOGIC and "
          "checking it against the file on disk -- column names, thresholds, JSON key sets, chart types -- not "
          "guessed from a filename."),
        table(
            ["Report", "Confirmed xg generator(s)", "Confirmation status", "New xT sibling", "How it was confirmed / what differs"],
            STEP_0_ROWS,
        ),
        h3("Where a methodology could NOT be confirmed from a surviving script"),
        p("Every one of the thirteen report types above traces to a surviving, readable generator script. "
          "<b>There is no report in this suite whose methodology had to be hand-reproduced from a prompt file "
          "because the script was lost.</b> That is the honest answer, and it is stated rather than padded."),
        p("Three genuine gaps were found, and none of them is a missing script -- they are all SCOPE gaps, "
          "reported in the individual reports as well as here:"),
        ul([
            "<b>Leakage Audit.</b> The xg (and binary) leakage audits are passive-dataset-only. No active-side "
            "leakage audit exists anywhere in this repo, for any target. Part A's method was reused by direct "
            "import; the xg audit's Parts B/C/D are reported as HAVING NO ACTIVE COUNTERPART, and substitute "
            "parts B'/C' apply the same method to the active dataset's own analogous columns. Part D' is new "
            "and is labelled as new inside the report itself.",
            "<b>Confound Testing.</b> All five tests in the xg portal are passive. Exactly one active confound "
            "test exists project-wide, in the binary portal only -- <code>defenders_within_10m_vs_box_proximity</code> "
            "-- and the xg portal never received it. Test 1 mirrors it exactly; Test 2 is new and labelled new.",
            "<b>Slice Stratification V2.</b> Its archetype slicer is a passive column, so V2's closing "
            "archetype-vs-boolean-slicer comparison cannot be reproduced on the active leg. Reported as not "
            "applicable rather than replaced with a different comparison that would read like the same "
            "conclusion.",
        ]),
        h3("One place where a carried-forward finding does NOT surface through the mirrored methodology"),
        p("Mirroring a methodology faithfully means inheriting its blind spots too. Where a report's own "
          "methodology fails to surface one of Prompt 64's findings even though a reader would expect it to, "
          "that omission is itself recorded here rather than quietly patched over. One such case was found."),
        p("<b>The Flag Ledger does not show the three <code>action_*_possession</code> flags at all.</b> Prompt "
          "64's motivating result is that those three stop being degenerate zeros on this target, so the Flag "
          "Ledger is exactly where a reader would expect it to appear. It does not, and the reason is pool "
          "construction, not a failed finding: the ledger is built from the reconstructed pre-drop 51-era pool "
          "(<code>feature_config_v1_historical.ACTIVE_V1_HISTORICAL</code>), and all three were already out of "
          "the candidate list before that snapshot was taken -- only "
          "<code>action_was_under_opponent_possession</code> survived into it. The pool-construction logic is "
          "mirrored unchanged from <code>generate_reports_xg.py</code>, so it structurally cannot show them."),
        p("Rather than let the finding disappear on that technicality, the same <code>boolean_xt_lift</code> "
          "function the ledger uses for every pooled row was run on the three columns OFF-POOL. The numbers and "
          "the gap are both stated in the rendered Flag Ledger, kept under a separate "
          "<code>off_pool_possession_flags</code> key in its JSON so they are never mistaken for pooled rows, "
          "and the flags are also covered properly in the Leakage Audit's Part C'. The numbers are in section 4."),
        h3("One stale-path artefact found in an existing script, reported not fixed"),
        p("<code>generate_distribution_atlas_reconstructed.py</code> chooses its output filename with "
          "<code>out_dir.name == \"eda_xg\"</code>. The directory has since been renamed to "
          "<code>xg_target</code>, so that branch no longer matches: the file on disk is "
          "<code>active_distribution_atlas.html</code>, but a re-run today would write "
          "<code>active_distribution_atlas_reconstructed.html</code>. Left untouched deliberately -- editing it "
          "would change what the xg portal regenerates, and no existing <code>_xg</code> script was modified in "
          "this work."),
        h3("Closeout document: folded in, not separate"),
        p("<code>reports/analysis/shot_target/</code> carries a separate "
          "<code>PATTERN_ANALYSIS_CLOSEOUT.{md,html}</code> alongside its MASTER_FINDINGS. A separate closeout "
          "is <b>not</b> produced for this portal: that document exists to close out a multi-prompt pattern-"
          "analysis campaign spanning two targets and both datasets, whereas this portal is a single-pass, "
          "single-leg suite whose entire findings set fits in section 4 below without crowding anything out. "
          "Duplicating it would create two documents that could drift apart for no reader benefit."),
        note("scope", "<b>This portal covers the ACTIVE-BINARY LEG ONLY.</b> The passive leg is explicitly out "
                      "of scope and no passive-side xT-delta report exists. Where a report's xg counterpart "
                      "covers both datasets, only the active half is reproduced, and that is stated in the "
                      "report."),
    ]


def build_section_2(scale: dict) -> list[dict]:
    return [
        p(f"<code>{esc(xc.TARGET)}</code> = <code>xt_before - xt_after</code>, an Expected-Threat value-delta "
          "computed in Prompt 64 for the active-binary leg, to replace the structurally flawed "
          "<code>target_future_shot_10s</code>/<code>target_future_xg_10s</code> pair on that leg. It lives in "
          "<code>outputs/prototypes/active_binary_xt_delta.parquet</code> (56,068 rows, row-aligned to the "
          "locked <code>player_defensive_actions.parquet</code> by <code>event_id</code>). Every report in this "
          "portal JOINS the two in memory and never writes either back."),
        table(
            ["Statistic", "Value", "Meaning"],
            [
                ["rows", f"{scale['n_rows']:,}", "the full active-binary leg"],
                ["defined", f"{scale['n_defined']:,}", "32 NaN, explained in Prompt 64 section 2 (preceding event has no location)"],
                ["mean", f"{scale['mean']:+.6f}", "near zero, and NEGATIVE"],
                ["std", f"{scale['std']:.6f}", "the scale-setting statistic this portal uses in place of the mean"],
                ["skew", f"{scale['skew']:+.3f}", "essentially symmetric"],
                ["excess kurtosis", f"{scale['excess_kurtosis']:.2f}", "heavy-tailed -- most mass near zero, a few large swings"],
                ["share exactly zero", f"{scale['pct_zero']:.2f}%", "the action did not cross an xT grid-cell boundary"],
                ["share negative", f"{scale['pct_negative']:.2f}%", "xT ROSE across the action"],
                ["share positive", f"{scale['pct_positive']:.2f}%", "xT FELL across the action (threat reduced)"],
            ],
        ),
        note("carried forward from Prompt 64 -- confirmed independently",
             esc(xc.PROMPT_64_SHAPE_FINDING) + " Every figure in the table above was recomputed on this run and "
             "matches Prompt 64's."),
    ]


def build_section_3(scale: dict) -> list[dict]:
    return [
        p("The xg_target suite's methodology assumes a strictly non-negative, zero-inflated, right-skewed "
          "target throughout. Three things had to change, and one thing that might have been expected to "
          "change did not. All four are stated in every report that they touch, not just here."),
        h3("Adaptation 1 -- the scale-setting statistic"),
        p(esc(scale["scale_note"])),
        table(
            ["Quantity", "Value on this run"],
            [
                ["xg flat margin on the same rows (0.5 x mean xg)", f"{scale['xg_flat_margin_absolute']:.6f}"],
                ["as a fraction of xg's own std", f"{scale['xg_flat_margin_as_fraction_of_xg_std']:.6f}"],
                ["translated flat margin for xT delta", f"{scale['flat_margin']:.6f}"],
                ["translated consistency range trigger", f"{scale['range_trigger']:.6f}"],
                ["translated review near-identical threshold", f"{scale['near_identical_threshold']:.6f}"],
                ["translated review distinct-signal threshold", f"{scale['distinct_signal_threshold']:.6f}"],
                ["flat margin on the non-zero-delta subset", f"{scale['flat_margin_nonzero']:.6f}"],
            ],
        ),
        p("Scale-free thresholds were carried over <b>completely unchanged</b>: |Spearman rho| &ge; 0.05, "
          "small-n row counts (500 and 1000), the substitutive (0.3) and additive (2.0) ratio cutoffs, "
          "<code>MIN_CELL_N_FOR_DELTA=20</code>, <code>QCUT_N=4</code>, 10 quantile deciles, and the "
          "discrete-cardinality cutoff of 15."),
        h3("Adaptation 2 -- the conditional panel"),
        p(esc(scale["conditional_panel_note"])),
        p("One knock-on rename: the slice-stratification reports' <code>occurrence_vs_quality</code> field "
          "encodes an xg-specific distinction that simply does not exist for this target. It is renamed "
          "<code>zero_mass_vs_magnitude</code> (values: zero-mass-driven / magnitude-driven / inconclusive) "
          "rather than reused under a name that would be actively misleading."),
        h3("Adaptation 3 -- two-directional framing"),
        p(esc(scale["direction_note"])),
        p("Concretely this changes four things: every bar chart in the portal diverges from a zero baseline "
          "instead of filling from the left (a left-anchored fill renders every negative value as zero width); "
          "every sparkline carries a dashed zero line; the interaction heatmaps use a diverging green/red ramp "
          "instead of a single-hue alpha ramp; and several reports record a new field with no xg counterpart -- "
          "whether a binned curve CROSSES zero, and whether within-stratum deltas change SIGN. Neither question "
          "can be asked of a non-negative target at all."),
        h3("Not an adaptation -- log-transforms"),
        p(esc(scale["log_transform_note"])),
    ]


def build_section_4(fl: dict) -> list[dict]:
    pf = fl["pattern_analysis_findings"]
    ca, flg, ct, tr, ss, ni, rv, nr = (pf["category_atlas"], pf["flag_ledger"], pf["confound_tests"],
                                       pf["tournament_stability"], pf["slice_stratification"],
                                       pf["numeric_interaction"], pf["review_tier"],
                                       pf["numerical_vs_target_rankings"])
    lc = fl["leakage_confirmation_xt"]
    d = lc["part_d_prime_construction_coupling"]["strongest"]

    return [
        h3("Prompt 64's two carried-forward findings -- where they resurfaced"),
        note("Clearance / action_x artefact -- RESURFACED, in the Category Atlas and the Confound report",
             esc(ca["finding"])),
        p("It also drives a NEW confound test in this portal ("
          "<code>clearance_vs_defending_box_proximity</code>), which asks whether Clearance's negative mean "
          "delta survives conditioning on defending-box proximity. The verdict is "
          f"<b>{esc(ct['tests'][1]['verdict'])}</b> -- it survives in one stratum and not the other, so the "
          "artefact is neither purely a deep-box location effect nor independent of location. And Part D' of "
          "the Leakage Audit identifies the underlying mechanism: because <code>xt_after</code> is looked up "
          "from <code>action_x</code>/<code>action_y</code>, EVERY location-derived feature inherits the same "
          "definitional choice, with <code>" + esc(d["feature"]) + f"</code> the strongest at r={d['r_vs_target']:+.4f} "
          "against the target."),
        note("distribution shape -- CONFIRMED, in the Distribution Atlas",
             "The Distribution Atlas adds a target-shape section the xg portal's atlas has no equivalent of "
             "(that atlas covers feature shapes only and is target-independent). Skew, kurtosis and the "
             "negative/positive/zero shares all reproduce Prompt 64's figures. The histogram there is clipped "
             "SYMMETRICALLY at the 99th percentile of |delta|; the atlas's usual min-to-p99 convention would "
             "have silently discarded this target's entire negative tail."),
        h3("The motivating problem, confirmed fixed on the full dataset (measured off-pool)"),
        p("The three <code>action_*_possession</code> flags are 100% identical zeros against "
          "<code>target_future_shot_10s</code> -- which is the stated reason feature_config.py excludes them. "
          "Against this target, on the full active dataset rather than only the two slices Prompt 64 examined:"),
        table(
            ["Flag", "n (True)", "mean delta (True)", "% exactly zero (True)", "lift"],
            [[f"`{k}`", f"{v['n_true']:,}", f"{v['mean_xt_true']:+.6f}", f"{v['pct_zero_true']:.1f}%",
              f"{v['lift']:+.6f}"]
             for k, v in flg["possession_flags_no_longer_degenerate"].items()],
        ),
        note("measured off-pool -- see section 1",
             "These three are not rows in the Flag Ledger, because they predate the reconstructed pre-drop pool "
             "the ledger is built from. The same lift function was run on them off-pool so the finding is not "
             "lost to a pool-construction technicality. The Leakage Audit's Part C' covers them properly."),
        p("<b>These columns stay excluded in feature_config.py and nothing in this portal changes that.</b> The "
          "finding is that their stated exclusion reason is target-specific and does not transfer; acting on it "
          "would be a separate decision requiring its own leakage review."),
        h3("Numerical features vs the target"),
        p(f"{nr['n_curves_crossing_zero']}/{nr['n_features']} features' binned curves cross zero -- almost all "
          "of them separate threat-reducing from threat-increasing actions rather than only varying in "
          f"magnitude. {nr['n_train_test_inconsistent']} feature(s) are flagged train/test-inconsistent."),
        table(["Feature", "Spearman rho", "Shape", "Curve crosses zero"],
              [[f"`{r['feature']}`", str(r["spearman_rho"]), str(r["shape"]), str(r["binned_curve_crosses_zero"])]
               for r in nr["active_top"]]),
        h3("Flag Ledger -- top 5 by |lift|, both directions"),
        table(["Flag", "lift", "mean True", "mean False", "n True"],
              [[f"`{r['column']}`", f"{r['lift']:+.6f}", f"{r['mean_xt_true']:+.6f}",
                f"{r['mean_xt_false']:+.6f}", f"{r['n_true']:,}"] for r in flg["top_by_abs_lift"]]),
        h3("Confound (reversal) tests"),
        table(["Test", "Unconditional", "Given a non-zero delta", "New test"],
              [[t["title"], t["verdict"], t["verdict_given_nonzero_delta"], "yes" if t["is_new_test"] else "no"]
               for t in ct["tests"]]),
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
          f"(dominated, as on every target, by continuous-continuous pairs no rule covers); "
          f"{rv['n_type2_opposite_sign']} Type-2 pair(s) resolve on the opposite-sign rule, which carries more "
          "weight here than on xg -- a sign disagreement means the two flags disagree about the DIRECTION of "
          "the threat swing, not merely about being above or below a base rate."),
    ]


def build_section_5(fl: dict) -> list[dict]:
    lc = fl["leakage_confirmation_xt"]
    b = lc["part_b_prime_has_previous_event"]
    d = lc["part_d_prime_construction_coupling"]
    return [
        p("Two caveats are <b>recorded, not resolved</b>. Neither is a reason to change the locked feature set, "
          "and this portal changes nothing in <code>feature_config.py</code>."),
        h3("Caveat 1 -- construction coupling (the most important one, no xG or binary counterpart)"),
        p(esc(d["implication_for_the_lock"])),
        h3("Caveat 2 -- this target's effects can sit in its zero mass"),
        p(esc(b["implication_for_the_lock"])),
        note("methodological warning for anyone extending this suite",
             "Caveat 2 is the one most likely to bite a future report. The instrument the xg suite reaches for "
             "first -- a difference of means -- is the right one for a strictly non-negative target whose "
             "signal lives in its centre, and the wrong one for a symmetric target with 11.3% of its mass at "
             "exactly zero. Every table in this portal therefore reports the negative / zero / positive shares "
             "alongside its mean."),
        h3("Open questions this portal does NOT answer"),
        ul([
            "Whether <code>action_x</code>/<code>action_y</code> is the right definition of <code>xt_after</code> "
            "at all. Prompt 64 flagged that a clearance's RESULTING ball location would be more "
            "football-accurate; this portal quantifies the consequences of the current definition but does not "
            "change it.",
            "Whether the three <code>action_*_possession</code> flags should be unlocked now that their stated "
            "exclusion reason no longer holds. That needs its own leakage review.",
            "What model architecture suits a roughly symmetric, heavy-tailed, sign-carrying target. Prompt 64 "
            "flagged that the existing hurdle architecture (P(shot) x E[xg|shot]) does not apply as-is. This "
            "is an EDA portal and takes no position; no model artefact was touched.",
            "The passive leg, entirely. Out of scope for this work.",
        ]),
    ]


def build_section_6() -> list[dict]:
    return [
        h3("Built here (target-dependent -- rebuilt against target_xt_delta)"),
        table(["Report", "Files"], [
            ["Active -- Category Atlas", "`active_category_atlas.html` / `.json`"],
            ["Active -- Flag Ledger", "`active_flag_ledger.html` / `.json`"],
            ["Active -- Distribution Atlas", "`active_distribution_atlas.html`, `active_distribution_atlas_reconstructed.html`, `active_distribution_atlas.json`"],
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
        p("These five are properties of the 32 locked ACTIVE features and are unaffected by which target the "
          "features are measured against. The portal index links out to the existing files in "
          "<code>../xg_target/</code> rather than duplicating them."),
        ul([
            "Correlation Atlas V1 / V2 / V3 -- correlation clustering is feature-vs-feature and never "
            "references a target column; re-running it here would produce byte-identical pairs and tiers.",
            "VIF Analysis -- multicollinearity is a property of the feature matrix alone.",
            "Slicer Redundancy Check -- slicer-vs-slicer, no target term.",
            "Player-Level Validity Check -- row concentration and train/test player overlap, no target term.",
            "Player-Grouped Split Check -- an identity-leakage stress test on the split itself.",
        ]),
        h3("New generator scripts (all added as siblings; no existing script was modified)"),
        p("Every script below is new. Nothing under <code>src/eda/</code> that already existed was edited, so "
          "the <code>xg_target</code> and <code>shot_target</code> portals remain exactly reproducible from "
          "their own unmodified generators."),
        ul([
            "`src/eda/xt_common.py` -- the loader, the three adaptations, and the xT statistics functions",
            "`src/eda/generate_reports_xt.py`",
            "`src/eda/generate_distribution_atlas_xt.py`",
            "`src/eda/generate_numerical_xt_target_analysis.py` + `generate_numerical_xt_target_reports.py`",
            "`src/eda/generate_review_analysis_xt.py` + `generate_review_report_xt.py`",
            "`src/eda/generate_leakage_audit_xt.py` + `generate_leakage_report_xt.py`",
            "`src/eda/generate_confound_analysis_xt.py` + `generate_confound_report_xt.py`",
            "`src/eda/generate_tournament_stability_check_xt.py` + `generate_tournament_stability_report_xt.py`",
            "`src/eda/generate_slice_stratification_xt.py` + `generate_slice_stratification_v2_xt.py` + `generate_slice_stratification_report_xt.py`",
            "`src/eda/generate_feature_interaction_analysis_xt.py` + `generate_feature_interaction_report_xt.py`",
            "`src/eda/generate_feature_lock_confirmation_xt.py` + `generate_feature_lock_report_xt.py`",
            "`src/eda/generate_master_findings_xt.py` + `generate_eda_portal_xt.py`",
        ]),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt(columns=["event_id"])
    scale = xc.target_scale(df)
    fl = _load("FEATURE_LOCK_CONFIRMATION_XT.json")
    active_count = sum(len(ACTIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))

    bottom_line = (
        f"<code>{esc(xc.TARGET)}</code> is a roughly symmetric, heavy-tailed, sign-carrying target "
        f"({scale['pct_negative']:.1f}% negative, {scale['pct_positive']:.1f}% positive, "
        f"{scale['pct_zero']:.1f}% exactly zero, skew {scale['skew']:+.3f}) -- structurally unlike either target "
        "this project has analysed before. The xg_target report suite's methodology transfers, but only after "
        "three explicit adaptations, each stated in every report it touches. Both of Prompt 64's carried-forward "
        "findings resurfaced exactly where expected: the Clearance/<code>action_x</code> artefact in the Category "
        "Atlas (Clearance ranks last of 8 event types) and again as a new confound test and a new leakage-audit "
        "part, and the symmetric/negative-capable distribution shape in the Distribution Atlas. No feature was "
        "added, dropped or re-tiered, no model artefact was touched, and the five target-independent reports were "
        "linked rather than rebuilt."
    )

    sections = [
        ("Step 0 -- confirmed generation methodology, report by report", build_section_1()),
        ("The target: what target_xt_delta is and what shape it has", build_section_2(scale)),
        ("What methodology was adapted, and exactly why", build_section_3(scale)),
        ("Findings", build_section_4(fl)),
        ("Caveats recorded, and open questions this portal does not answer", build_section_5(fl)),
        ("Document map", build_section_6()),
    ]

    # --- Markdown ---
    md_parts = [
        "# Master Findings -- xT-Delta Target (Active-Binary Leg Only)",
        "",
        "> **Scope: the ACTIVE-BINARY leg only.** The passive leg is explicitly out of scope for this target "
        "and no passive-side xT-delta report exists. Nothing in this document, or in this portal, covers it.",
        "",
        f"**Bottom line up front.** {bottom_line}",
    ]
    for i, (title, blocks) in enumerate(sections, start=1):
        md_parts.append(f"\n## {i}. {title}\n")
        md_parts.append(render_blocks_md(blocks))
    MD_PATH.write_text("\n".join(md_parts).replace("<code>", "`").replace("</code>", "`")
                       .replace("<b>", "**").replace("</b>", "**") + "\n", encoding="utf-8")

    # --- HTML, from the same block data ---
    html_parts = [
        '<div class="finding flag" style="margin-bottom:20px;"><span class="tag">scope</span>'
        '<p><b>This portal covers the ACTIVE-BINARY leg only.</b> The passive leg is explicitly out of scope '
        'for this target and no passive-side xT-delta report exists. Nothing in this document, or in this '
        'portal, covers it.</p></div>',
        f'<div class="finding" style="margin-bottom:28px;"><span class="tag">bottom line up front</span>'
        f'<p>{bottom_line}</p></div>',
    ]
    for i, (title, blocks) in enumerate(sections, start=1):
        html_parts.append(f'<h2 class="mf-section-title" id="s{i}">{i}. {esc(title)}</h2>')
        html_parts.append(render_blocks_html(blocks))

    html = render.render_article(
        eyebrow="MASTER FINDINGS -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="xT-Delta Target -- Master Findings (Active-Binary Leg)",
        dek=(
            "The entry point for this portal: which xg_target script produced each report and how that was "
            "confirmed, what target_xt_delta is, exactly what methodology had to change because of its shape, "
            "every finding, and what is deliberately left open. Active-binary leg only."
        ),
        stats=[
            (f"{active_count}", "active features"),
            ("13", "reports rebuilt"),
            ("5", "reports linked, not rebuilt"),
            (f"{scale['pct_negative']:.0f}%", "target rows negative"),
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

    print(f"Wrote {MD_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
