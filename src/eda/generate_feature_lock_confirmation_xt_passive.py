"""CLI entrypoint: post-lock confirmation + pattern findings for
`target_xt_delta_passive` -- PASSIVE LEG. Computes the JSON and renders its
HTML in one pass.

STEP-0 DISPOSITION: genuine passive build.
`reports/analysis/xg_target/FEATURE_LOCK_CONFIRMATION_XG.json` covers BOTH
legs in ONE file -- confirmed by reading it: its `confirmed_counts` block
carries `active: 34` and `passive: 38`, and its `correlation_diff` block
carries both. This portal's `FEATURE_LOCK_CONFIRMATION_XT.json` is Prompt
67's ACTIVE-ONLY output. Merging a passive section in would mean editing
active-leg report content, which this pass does not do.

Structural point carried over exactly, not re-derived: the
correlation/redundancy machinery is TARGET-AGNOSTIC (feature vs feature,
never feature vs target). Re-running it against a different target would
produce byte-identical pairs and tiers. So `correlation_diff` is READ
DIRECTLY from `reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json`'s
PASSIVE block and labelled identical-by-construction, the same discipline
`generate_feature_lock_confirmation_xg.py` applies.

The renderer is written here rather than reused from
`generate_feature_lock_report_xt.py`, and the reason is structural rather
than cosmetic: that renderer's leakage section is hardcoded to the active
audit's Parts B'/C'/D' (`has_previous_event`, the three
`action_*_possession` flags, the `action_x` coupling) and its pattern
section is hardcoded to the Clearance artefact and the possession flags.
None of those columns or findings exists on this leg. Relabelling could not
have produced an honest passive document, so the sections it does not share
are written out. The shared helpers (`_count_card`, `_table`) and the whole
visual system are imported from it unchanged.

Usage:
    python -m src.eda.generate_feature_lock_confirmation_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import render, xt_common as xc
from src.eda import generate_feature_lock_report_xt as fr1
from src.eda.feature_config import PASSIVE
from src.eda.render import esc, finding_card

BINARY_LOCK_PATH = xc.REPO_ROOT / "reports" / "analysis" / "shot_target" / "FEATURE_LOCK_CONFIRMATION.json"
OUT_DIR = xc.OUT_DIR
JSON_PATH = OUT_DIR / "PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.json"
HTML_PATH = OUT_DIR / "PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.html"

# From reports/analysis/xg_target/FEATURE_LOCK_CONFIRMATION_XG.json's own
# `confirmed_counts.passive_expected` -- read, not guessed.
EXPECTED_PASSIVE_COUNT = 38


def _load(name: str) -> dict:
    return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))


def build_leakage_confirmation_passive() -> dict:
    la = _load("PASSIVE_LEAKAGE_AUDIT.json")
    b = la["part_b_screened_option_was_avoided"]
    c = la["part_c_has_screened_outcome"]
    d = la["part_d_has_option_2_3"]
    e = la["part_e_prime_construction_coupling"]

    return {
        "source": "reports/analysis/xt_target/PASSIVE_LEAKAGE_AUDIT.json",
        "scope_difference_from_the_active_confirmation": (
            "The ACTIVE half of this portal folds in its leakage audit's Parts B'/C'/D', which are "
            "substitutes Prompt 65 had to invent because no active-side leakage audit exists in this repo for "
            "any target. This half needs no substitutes: leakage auditing in this project IS a passive "
            "methodology, so Parts A/B/C/D below are the established checks on their own columns, mirrored "
            "literally from the passive xg audit. Part E' is new to this target family."
        ),
        "part_a_verdict": la["part_a_known_leaky_scan"]["verdict"],
        "part_b_screened_option_was_avoided": {
            "delta": b["delta"], "zero_share_gap_pp": b["zero_share_gap_pp"], "verdict": b["verdict"],
            "implication_for_the_lock": (
                "screened_option_was_avoided stays EXCLUDED, and this target changes nothing about that. Its "
                "exclusion is conceptual/structural, so no empirical result on any target can move it. "
                "Recorded here only so the confirmation covers every part of the audit."
            ),
        },
        "part_c_has_screened_outcome": {
            "delta": c["delta"], "zero_share_gap_pp": c["zero_share_gap_pp"], "p_value": c["p_value"],
            "verdict": c["verdict"],
            "implication_for_the_lock": (
                "has_screened_outcome stays EXCLUDED and this document does not change that. The exclusion "
                "rests on the censoring MECHANISM -- the column is False exactly when the forward window was "
                "truncated -- which is a property of how it is built, not of which target it is measured "
                f"against. Against this target the mean difference is {c['delta']:+.6f} and the exactly-zero "
                f"share differs by {c['zero_share_gap_pp']:+.1f}pp. Read the zero-share gap, not only the "
                "mean: this target's effects can live in its 26.4% exactly-zero mass, which is the same "
                "caveat Prompt 65 recorded for the active leg and which applies to MORE rows here (26.4% vs "
                "20.2%). Do not cite this target's number as fresh independent evidence for the exclusion -- "
                "it is the same structural argument, measured a third time."
            ),
        },
        "part_d_has_option_2_3": {
            "columns": list(d.keys()),
            "verdicts": {k: v["verdict"] for k, v in d.items()},
            "deltas": {k: v["delta"] for k, v in d.items()},
            "implication_for_the_lock": (
                "has_option_2 and has_option_3 stay IN the candidate list, the same disposition the binary and "
                "xg audits reached. They describe the current freeze frame rather than the computability of "
                "the forward window, so they are not censoring proxies. Both were flagged for confound testing "
                "on the earlier targets and this portal's own passive Confound report runs exactly that test "
                "against this target -- see its Part D mirrors."
            ),
        },
        "part_e_prime_construction_coupling": {
            "strongest_vs_target": e["strongest_vs_target"],
            "strongest_vs_xt_before": e["strongest_vs_xt_before"],
            "verdict": e["verdict"],
            "is_new_part": True,
            "implication_for_the_lock": (
                "NO FEATURE IS DROPPED ON THIS EVIDENCE AND NONE IS PROPOSED FOR DROPPING -- the locked set "
                "stays exactly as feature_config.py has it, and this document does not edit it. What is "
                "recorded is a reading instruction with no counterpart in the binary or xg confirmations, and "
                "a DIFFERENT one from the active half's. On this leg xt_before = xT(ball_x, ball_y), and both "
                "ball_x and ball_y are LOCKED candidate features, so one of the target's two terms is a "
                f"deterministic function of two locked columns. Measured: the strongest locked-feature "
                f"correlation with xt_before is {e['strongest_vs_xt_before']['feature']} at "
                f"r={e['strongest_vs_xt_before']['r_vs_xt_before']:+.4f}, which is the definition of the term "
                f"rather than a finding; what survives the subtraction is "
                f"{e['strongest_vs_target']['feature']} at r={e['strongest_vs_target']['r_vs_target']:+.4f} "
                "against the delta itself. For scale: Prompt 65 measured the active leg's equivalent at "
                "+0.5551 under v1 and Prompt 67 at +0.1431 under the corrected v2. This leg sits between "
                "them, through a different mechanism, and the caveat is therefore LIVE here in a way the "
                "active half's is no longer."
            ),
        },
        "verdict": (
            "The locked PASSIVE feature set is internally unchanged for xT-delta work -- correlation and "
            "redundancy are target-agnostic, so nothing about the feature list moves, and every Type-1/3/4 "
            "review verdict matches the binary and xg passive reviews by construction. Two target-specific "
            "caveats are RECORDED, not resolved: (1) construction coupling running through xt_before, which is "
            "a grid lookup at two locked features on this leg -- live here, unlike on the corrected active "
            "leg; and (2) the fact that this target's effects can sit in its exactly-zero mass, where a "
            "mean-difference test will miss them, which applies to a larger share of rows here (26.4%) than on "
            "the active leg (20.2%)."
        ),
    }


def build_pattern_findings_passive() -> dict:
    atlas = _load("passive_numerical_target_atlas.json")
    cat = _load("passive_category_atlas.json")
    flag = _load("passive_flag_ledger.json")
    confound = _load("PASSIVE_CONFOUND_ANALYSIS.json")
    tourn = _load("PASSIVE_TOURNAMENT_STABILITY_CHECK.json")
    v1 = _load("PASSIVE_SLICE_STRATIFICATION.json")
    v2 = _load("PASSIVE_SLICE_STRATIFICATION_V2.json")
    inter = _load("PASSIVE_FEATURE_INTERACTION_ANALYSIS.json")
    review = _load("PASSIVE_REVIEW_ANALYSIS.json")
    dist = _load("passive_distribution_atlas.json")

    et = cat["columns"]["on_ball_event_type"]["categories"]
    flag_ranked = sorted(
        ({"column": k, **v["lift_stats"]} for k, v in flag["columns"].items()),
        key=lambda e: abs(e["lift"]), reverse=True,
    )

    return {
        "source_note": (
            "Every number in this section is read from this portal's own PASSIVE JSON outputs, not re-typed by "
            "hand, not carried over from the xg or binary portals, and not taken from this portal's active "
            "half. Where a finding has an xg, binary or active-leg counterpart, the comparison is made in the "
            "individual report rather than restated here."
        ),
        "target_shape": dist["target_distribution"],
        "grain_comparison": dist["target_distribution"]["grain_comparison"],
        "numerical_vs_target_rankings": {
            "passive_top": [
                {"feature": f["feature"], "spearman_rho": f["spearman_rho"], "shape": f["shape"],
                 "binned_curve_crosses_zero": f.get("binned_curve_crosses_zero")}
                for f in atlas["features"][:5]
            ],
            "n_curves_crossing_zero": atlas["n_features_whose_binned_curve_crosses_zero"],
            "n_features": atlas["n_features_analyzed"],
            "n_train_test_inconsistent": atlas["n_features_flagged_inconsistent"],
            "n_shape_flips_under_robust_margin": atlas["n_features_classifying_differently_under_robust_margin"],
            "note": (
                "Top 5 of the reconstructed PASSIVE pool by |Spearman rho|, read from the atlas's own sort "
                "order. Ranking is by ABSOLUTE rho in both directions -- on this target a strong negative rho "
                "is as much a finding as a strong positive one, and on this leg negative deltas are the "
                "long-tailed direction. Correlations are computed over defender-slot rows sharing one target "
                "value per event, so rank and magnitude are the readable parts and significance is not."
            ),
        },
        "category_atlas": {
            "column": "on_ball_event_type",
            "n_categories": len(et),
            "most_threat_reducing": {"category": et[0]["category"], "mean_xt": et[0]["mean_xt"],
                                     "n": et[0]["n"], "pct_positive": et[0]["pct_positive"]},
            "most_threat_increasing": {"category": et[-1]["category"], "mean_xt": et[-1]["mean_xt"],
                                       "n": et[-1]["n"], "pct_negative": et[-1]["pct_negative"]},
            "clearance_artefact_not_applicable": (
                "The ACTIVE half of this portal's headline Category Atlas finding is the Clearance reversal -- "
                "Clearance moving from 8th of 8 event_type categories to 1st of 8 under Prompt 66's corrected "
                "xt_after. THAT FINDING HAS NO PASSIVE COUNTERPART AND IS NOT FORCED INTO ONE. `event_type` "
                "does not exist in passive_defense.parquet, checked against the live schema. The nearest "
                "relative, `on_ball_event_type`, describes the ATTACKING action the defensive snapshot is "
                "anchored to, not a defensive action, so a Clearance row over there and a Pass row here are "
                "not comparable objects. What this column does carry is reported on its own terms above."
            ),
            "finding": (
                f"Across {len(et)} on_ball_event_type categories, `{et[0]['category']}` is the most "
                f"threat-REDUCING at {et[0]['mean_xt']:+.6f} (n={et[0]['n']:,}) and `{et[-1]['category']}` the "
                f"most threat-INCREASING at {et[-1]['mean_xt']:+.6f} (n={et[-1]['n']:,}). Read both as "
                "descriptions of what happens around a defensive snapshot rather than as verdicts on "
                "defending: the target is a property of the EVENT, identical across every defender slot at "
                "that event, so no part of either number is attributable to an individual defender."
            ),
        },
        "flag_ledger": {
            "top_by_abs_lift": [
                {"column": e["column"], "lift": e["lift"], "mean_xt_true": e["mean_xt_true"],
                 "mean_xt_false": e["mean_xt_false"], "n_true": e["n_true"]}
                for e in flag_ranked[:5]
            ],
            "possession_flags_not_applicable": flag["active_only_possession_flags_absent_here"]["note"],
        },
        "confound_tests": {
            "n_tests": len(confound["tests"]),
            "n_new_tests": 0,
            "tests": [{"name": t["name"], "title": t["title"], "verdict": t["verdict"]["verdict"],
                       "verdict_given_nonzero_delta": t["given_nonzero_delta"]["verdict"]["verdict"],
                       "is_new_test": bool(t.get("is_new_test"))}
                      for t in confound["tests"]],
            "note": confound["step_0_scope_note"],
            "threshold_note": confound["threshold_note"],
        },
        "tournament_stability": {
            "n_features": len(tourn["features"]),
            "features": [{"feature": f["feature"], "verdict": f["verdict"],
                          "verdict_flips_under_robust_margin":
                              f["robust_margin_recheck"]["verdict_flips_under_robust_margin"]}
                         for f in tourn["features"]],
            "n_genuine": sum(1 for f in tourn["features"]
                             if f["verdict"] == "genuine tournament-level difference"),
            "n_verdicts_flipping_under_robust_margin": sum(
                1 for f in tourn["features"]
                if f["robust_margin_recheck"]["verdict_flips_under_robust_margin"]),
        },
        "slice_stratification": {
            "v1": {"n_cells": v1["n_cells"],
                   "n_genuine_divergences": v1["cross_dataset_summary"]["n_genuine_divergences"],
                   "n_genuine_divergences_given_nonzero_delta":
                       v1["cross_dataset_summary"]["n_genuine_divergences_given_nonzero_delta"],
                   "n_conditioning_disagreements":
                       v1["cross_dataset_summary"]["n_conditioning_disagreements"]},
            "v2": {"n_cells": v2["n_cells"],
                   "n_genuine_divergences": v2["cross_dataset_summary"]["n_genuine_divergences"],
                   "n_genuine_divergences_given_nonzero_delta":
                       v2["cross_dataset_summary"]["n_genuine_divergences_given_nonzero_delta"],
                   "archetype_comparison":
                       v2["cross_dataset_summary"]["closing_note_archetype_vs_boolean"]},
        },
        "numeric_interaction": {
            "n_pairs": inter["n_pairs_total"],
            "n_agree": inter["n_pairs_where_unconditional_and_conditional_agree"],
            "n_sign_flipping": inter["n_pairs_with_sign_flipping_stratum_deltas"],
            "pairs": [{"feature_a": p["feature_a"], "feature_b": p["feature_b"],
                       "classification": p["classification"],
                       "within_stratum_deltas_change_sign": p["within_stratum_deltas_change_sign"]}
                      for p in inter["summary_table"]],
        },
        "review_tier": {
            "n_review_pairs": review["datasets"]["passive"]["n_review_pairs"],
            "n_needs_human_call": len(review["datasets"]["passive"]["needs_human_call"]),
            "n_type2_opposite_sign": review["datasets"]["passive"]["n_type2_opposite_sign_pairs"],
        },
        "cross_referenced_target_independent_sections": {
            "correlation_atlas_v1_v2_v3": (
                "Not rebuilt, and a PASSIVE half already exists. Correlation clustering is feature-vs-feature "
                "and never references a target -- reports/analysis/xg_target/CORRELATION_ATLAS*.html and the "
                "underlying CORRELATION_ANALYSIS*.json carry a `datasets.passive` block, checked directly. "
                "Re-running it against target_xt_delta_passive would produce byte-identical pairs and tiers."
            ),
            "vif": (
                "Not rebuilt, and a PASSIVE half already exists -- VIF_ANALYSIS.json carries "
                "`datasets.passive`. Multicollinearity is a property of the feature matrix alone."
            ),
            "slicer_redundancy": (
                "Not rebuilt, and a PASSIVE half already exists -- SLICER_REDUNDANCY.json carries "
                "`datasets.passive`. Slicer-vs-slicer redundancy, no target term."
            ),
            "player_level_validity": (
                "Not rebuilt, and NO passive half exists anywhere in this project -- "
                "reports/analysis/shot_target/PLAYER_LEVEL_VALIDITY_CHECK.json carries "
                "\"dataset\": \"active\" and its own scope_limitation field. This is a genuine leg asymmetry "
                "rather than an omission: the check is about row concentration per player_id and train/test "
                "player overlap, and the passive leg is not player-indexed the same way. It is also "
                "target-independent, so building one would be outside this pass's remit either way."
            ),
            "player_grouped_split_check": (
                "Not rebuilt, and NO passive half exists -- its own `scope` field says \"Active leg only "
                "(player_id / position are active-dataset concepts; the passive leg is not player-indexed the "
                "same way)\", read directly rather than inferred from the title. Same genuine leg asymmetry as "
                "the Player-Level Validity Check, and likewise target-independent."
            ),
        },
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _leakage_section_passive(lc: dict) -> str:
    b, c, d, e = (lc["part_b_screened_option_was_avoided"], lc["part_c_has_screened_outcome"],
                  lc["part_d_has_option_2_3"], lc["part_e_prime_construction_coupling"])
    d_rows = "".join(f"<li><code>{esc(k)}</code>: <b>{esc(v)}</b> (delta {d['deltas'][k]:+.6f})</li>"
                     for k, v in d["verdicts"].items())
    s, sb = e["strongest_vs_target"], e["strongest_vs_xt_before"]
    return f"""
<h2 style="margin-top:40px;">Leakage confirmation (xT delta, passive)</h2>
<p class="section-note">Source: <code>{esc(lc['source'])}</code></p>
{finding_card("scope differs from the ACTIVE confirmation", "said so, not glossed",
              esc(lc['scope_difference_from_the_active_confirmation']))}

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part A &mdash; schema / naming scan</h5>
  <p class="stratum-delta">Verdict: <b>{esc(lc['part_a_verdict'])}</b> (method imported unchanged).</p>
</div>

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part B &mdash; <code>screened_option_was_avoided</code></h5>
  <p class="stratum-delta">Delta <b>{b['delta']:+.6f}</b>, exactly-zero share gap
  <b>{b['zero_share_gap_pp']:+.1f}pp</b>. Verdict: <b>{esc(b['verdict'])}</b>.</p>
  <p class="stratum-delta">{esc(b['implication_for_the_lock'])}</p>
</div>

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part C &mdash; <code>has_screened_outcome</code>, the censoring proxy</h5>
  <p class="stratum-delta">Delta <b>{c['delta']:+.6f}</b>, exactly-zero share gap
  <b>{c['zero_share_gap_pp']:+.1f}pp</b>, Welch p={c['p_value']:.4g}. Verdict: <b>{esc(c['verdict'])}</b>.</p>
  <p class="stratum-delta">{esc(c['implication_for_the_lock'])}</p>
</div>

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part D &mdash; <code>has_option_2</code> / <code>has_option_3</code></h5>
  <ul style="font-size:12.5px;margin:8px 0 8px 18px;">{d_rows}</ul>
  <p class="stratum-delta">{esc(d['implication_for_the_lock'])}</p>
</div>

<div class="stratum-card">
  <h5>Part E&prime; &mdash; construction coupling (new to this target family, and specific to this leg)</h5>
  <p class="stratum-delta">Strongest against <code>xt_before</code>: <code>{esc(sb['feature'])}</code> at
  r={sb['r_vs_xt_before']:+.4f}. Strongest against the delta itself: <code>{esc(s['feature'])}</code> at
  r={s['r_vs_target']:+.4f}. Verdict: <b>{esc(e['verdict'])}</b>.</p>
  <p class="stratum-delta">{esc(e['implication_for_the_lock'])}</p>
</div>

<div class="verdict-banner v-no" style="margin-top:20px;"><b>{esc(lc['verdict'])}</b></div>"""


def _pattern_section_passive(pf: dict) -> str:
    ts, nr, ca = pf["target_shape"], pf["numerical_vs_target_rankings"], pf["category_atlas"]
    fl, ct, tr = pf["flag_ledger"], pf["confound_tests"], pf["tournament_stability"]
    ss, ni, rv = pf["slice_stratification"], pf["numeric_interaction"], pf["review_tier"]
    cr = pf["cross_referenced_target_independent_sections"]
    g = pf["grain_comparison"]["prompt_68_unique_event"]

    return f"""
<h2 style="margin-top:48px;padding-top:24px;border-top:2px solid var(--border);">Pattern-analysis findings
(xT delta, passive)</h2>
<p class="section-note">{esc(pf['source_note'])}</p>

<h3 style="margin-top:28px;">The target's shape</h3>
<p class="section-note">At the defender-slot grain this suite measures at: mean <b>{ts['mean']:+.6f}</b>, std
<b>{ts['std']:.6f}</b>, skew <b>{ts['skew']:+.3f}</b>, excess kurtosis <b>{ts['excess_kurtosis']:.2f}</b>,
<b>{ts['pct_negative']:.1f}%</b> negative / <b>{ts['pct_positive']:.1f}%</b> positive /
<b>{ts['pct_zero']:.1f}%</b> exactly zero, on {ts['n_defined']:,} rows with a defined delta
({ts['n_nan']:,} NaN, not zero-filled). At Prompt 68's unique-event grain: mean {g['mean']:+.5f}, skew
{g['skew']:+.3f}, excess kurtosis {g['excess_kurtosis']:.2f} over {g['n_defined']:,} events. Same target, two
grains -- see the Distribution Atlas for why the row-level one is the more extreme of the two.</p>
{finding_card("carried forward from Prompt 68 -- CONFIRMED, more extreme at this grain",
              "left-skewed, very heavy-tailed, negative-capable target",
              esc(pc.PROMPT_68_SHAPE_FINDING), flag=True)}
{finding_card("carried forward from Prompt 68 -- the property every n on this page depends on",
              "one target value per event, repeated across that event's defender slots",
              esc(pc.PROMPT_68_SHARED_TARGET_FINDING), flag=True)}
{finding_card("carried forward from Prompt 68 -- no v1 pass to supersede",
              "this leg got the corrected definition on the first pass",
              esc(pc.PROMPT_68_NO_V1_DETOUR_FINDING))}

<h3 style="margin-top:28px;">Numerical features vs the target</h3>
<p class="section-note">{esc(nr['note'])} {nr['n_curves_crossing_zero']}/{nr['n_features']} features' binned
curves cross zero; {nr['n_train_test_inconsistent']} feature(s) flagged train/test-inconsistent;
<b>{nr['n_shape_flips_under_robust_margin']}/{nr['n_features']}</b> classify differently under the MAD-derived
robust flat margin (the comparable active-leg figure was 24 of 31 on v2 and 15 of 31 on v1).</p>
{fr1._table(nr['passive_top'], [('feature', 'feature'), ('spearman_rho', 'rho'), ('shape', 'shape'),
                                ('binned_curve_crosses_zero', 'crosses zero')])}

<h3 style="margin-top:28px;">Category Atlas &mdash; <code>on_ball_event_type</code></h3>
{finding_card("the active leg's Clearance finding does not transfer",
              "and is not forced into a passive counterpart",
              esc(ca['clearance_artefact_not_applicable']), flag=True)}
<p class="section-note">{esc(ca['finding'])}</p>

<h3 style="margin-top:28px;">Flag Ledger &mdash; top 5 by |lift|</h3>
{fr1._table(fl['top_by_abs_lift'], [('column', 'column'), ('lift', 'lift'), ('mean_xt_true', 'mean True'),
                                    ('mean_xt_false', 'mean False'), ('n_true', 'n True')])}
{finding_card("the active ledger's possession-flag section has no counterpart here",
              "those three columns do not exist in passive_defense.parquet",
              esc(fl['possession_flags_not_applicable']), flag=True)}

<h3 style="margin-top:28px;">Confound (reversal) tests</h3>
<p class="section-note">{esc(ct['note'])}</p>
{fr1._table(ct['tests'], [('title', 'test'), ('verdict', 'unconditional'),
                          ('verdict_given_nonzero_delta', 'non-zero delta'), ('is_new_test', 'new test')])}
{finding_card("no threshold in the confound report needed recomputing", "a checked result, not an omission",
              esc(ct['threshold_note']))}

<h3 style="margin-top:28px;">Tournament stability</h3>
<p class="section-note">{tr['n_genuine']}/{tr['n_features']} passive features show a genuine tournament-level
difference rather than arbitrary train/test noise;
{tr['n_verdicts_flipping_under_robust_margin']}/{tr['n_features']} verdicts would change under the MAD-derived
robust flat margin.</p>
{fr1._table(tr['features'], [('feature', 'feature'), ('verdict', 'verdict'),
                             ('verdict_flips_under_robust_margin', 'flips under robust margin')])}

<h3 style="margin-top:28px;">Slice stratification</h3>
<p class="section-note">V1: {ss['v1']['n_cells']} cells, {ss['v1']['n_genuine_divergences']} unconditional
genuine divergences, {ss['v1']['n_genuine_divergences_given_nonzero_delta']} on the non-zero-delta subset,
{ss['v1']['n_conditioning_disagreements']} categories where removing the zero mass flips the conclusion.
V2: {ss['v2']['n_cells']} cells, {ss['v2']['n_genuine_divergences']} unconditional /
{ss['v2']['n_genuine_divergences_given_nonzero_delta']} conditional.</p>
{finding_card("archetype-vs-boolean comparison", "APPLICABLE on this leg, unlike the active half",
              esc(ss['v2']['archetype_comparison']))}

<h3 style="margin-top:28px;">Numeric x numeric interaction</h3>
<p class="section-note">{ni['n_agree']}/{ni['n_pairs']} pairs classify the same way unconditionally and on the
non-zero-delta subset. {ni['n_sign_flipping']}/{ni['n_pairs']} have within-stratum deltas that change SIGN --
a direction reversal that cannot arise on the xG target. All five pairs involve this leg's own screening,
option or engagement columns and could not have been run on the active half.</p>
{fr1._table(ni['pairs'], [('feature_a', 'feature A'), ('feature_b', 'feature B'),
                          ('classification', 'classification'),
                          ('within_stratum_deltas_change_sign', 'sign flip')])}

<h3 style="margin-top:28px;">Review tier</h3>
<p class="section-note">{rv['n_review_pairs']} PASSIVE review pairs, {rv['n_needs_human_call']} still
needs_human_call, {rv['n_type2_opposite_sign']} Type-2 pair(s) resolved on the opposite-sign rule.</p>

<h3 style="margin-top:28px;">Target-independent reports &mdash; linked, not rebuilt, and which have a passive
half</h3>
{"".join(finding_card(k.replace('_', ' '), "checked directly in reports/analysis/, not assumed", v)
         for k, v in cr.items())}
"""


def build_report_passive(data: dict) -> str:
    counts = data["confirmed_counts"]
    d = data["correlation_diff"]["passive"]
    verdict_class = "v-no" if not data["any_newly_risky_pairs_found"] else "v-yes"

    body = f"""
<div class="verdict-banner {verdict_class}" style="margin-bottom:28px;">
<b>{'LOCKED (xT delta, passive) -- no new tier crossings' if not data['any_newly_risky_pairs_found'] else 'ATTENTION -- new tier crossing(s) found'}</b>
&mdash; {esc(data['verdict'])}
</div>

<h2>Confirmed candidate feature count</h2>
<p class="section-note">Read directly from feature_config.py's PASSIVE dict, not assumed. Same count as the
binary and xG confirmations' passive halves -- feature_config.py is not target-specific.</p>
<div class="before-after">
{fr1._count_card('Passive', counts['passive'], counts['passive_expected'], counts['passive_matches_expected'])}
</div>

<h2 style="margin-top:40px;">Correlation diff vs. the last confirmed run</h2>
{finding_card("correlation diff -- identical by construction", "cross-referenced, not re-derived",
              data["correlation_diff_note"])}
<div class="vif-dataset-block">
  <h2 class="dataset-title">Passive</h2>
  <p class="dataset-substat">{d['n_pairs_before']} DROP/COLLAPSE/REVIEW pairs before &rarr;
  {d['n_pairs_after']} after &middot; {d['n_added']} added &middot; {d['n_removed']} removed &middot;
  {d['n_tier_changed']} tier changes</p>
  <div class="corr-ledger"><div class="corr-row"><span class="corr-pair">
  {'No changes -- diff is empty.' if data['diff_is_empty'] else 'Non-empty, see JSON.'}
  </span></div></div>
</div>

{_leakage_section_passive(data['leakage_confirmation_xt_passive'])}
{_pattern_section_passive(data['pattern_analysis_findings'])}
"""

    return render.render_article(
        eyebrow="FEATURE LOCK CONFIRMATION -- XT DELTA &middot; PASSIVE LEG",
        title="Post-Lock Confirmation + Pattern Findings (xT delta, passive)",
        dek=(
            "Correlation/redundancy is target-agnostic and is reused directly rather than re-derived; the "
            "leakage side and every pattern finding are rebuilt from this portal's own PASSIVE outputs. "
            "Combined into one document, matching the xG portal's own confirmation-plus-findings layout. "
            "Passive leg; this portal's active half is in FEATURE_LOCK_CONFIRMATION_XT.html."
        ),
        stats=[
            (str(counts["passive"]), "passive features"),
            ("0" if data["diff_is_empty"] else "!", "tier changes found"),
            ("2", "target-specific caveats recorded"),
            ("0", "features added, dropped or re-tiered"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + render.VIF_CSS + render.CONFOUND_CSS,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary_lock = json.loads(BINARY_LOCK_PATH.read_text(encoding="utf-8"))
    assert "passive" in binary_lock["correlation_diff"], (
        "reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json has no passive correlation_diff block -- "
        "this confirmation reuses it by construction rather than recomputing it, so it cannot proceed without."
    )

    passive_count = sum(len(PASSIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))
    cd = binary_lock["correlation_diff"]["passive"]
    lc = build_leakage_confirmation_passive()

    output = {
        "leg": "passive (this portal also covers the active-binary leg -- see FEATURE_LOCK_CONFIRMATION_XT.json)",
        "target": xc.TARGET,
        "supersedes": (
            "nothing -- unlike the active leg, this leg has no v1 pass to supersede. "
            + pc.PROMPT_68_NO_V1_DETOUR_FINDING
        ),
        "step_0_scope_note": (
            "reports/analysis/xg_target/FEATURE_LOCK_CONFIRMATION_XG.json covers BOTH legs in ONE file -- "
            "confirmed by reading it: confirmed_counts carries active 34 and passive 38, and correlation_diff "
            "carries both. That convention is not mirrored here because this portal's "
            "FEATURE_LOCK_CONFIRMATION_XT.json is Prompt 67's ACTIVE-ONLY output and merging a passive section "
            "into it would mean editing active-leg report content that this pass leaves frozen."
        ),
        "confirmed_counts": {
            "passive": passive_count,
            "passive_expected": EXPECTED_PASSIVE_COUNT,
            "passive_matches_expected": passive_count == EXPECTED_PASSIVE_COUNT,
        },
        "changes_since_last_full_run": binary_lock["changes_since_last_full_run"],
        "correlation_diff": {"passive": cd},
        "correlation_diff_note": (
            "IDENTICAL to reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json's PASSIVE "
            "correlation_diff, by construction, not independently re-derived -- correlation clustering "
            "(feature vs feature) never references a target column, so re-running it against "
            "target_xt_delta_passive would produce the exact same pairs and tiers. This is also precisely why "
            "the target-independent reports are linked rather than rebuilt: nothing about this target's "
            "definition can move a feature-vs-feature statistic. Same discipline "
            "generate_feature_lock_confirmation_xg.py applies."
        ),
        "diff_is_empty": cd["n_added"] == 0 and cd["n_removed"] == 0 and cd["n_tier_changed"] == 0,
        "any_newly_risky_pairs_found": bool(cd["newly_risky_pairs"]),
        "leakage_confirmation_xt_passive": lc,
        "pattern_analysis_findings": build_pattern_findings_passive(),
        "prompt_68_cross_references": {
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "possession_ending_collapse": pc.PROMPT_68_POSSESSION_COLLAPSE_FINDING,
            "correlation_with_old_targets": pc.PROMPT_68_CORRELATION_FINDING,
            "no_v1_detour": pc.PROMPT_68_NO_V1_DETOUR_FINDING,
        },
        "verdict": (
            f"The locked PASSIVE feature set ({passive_count} features) is confirmed internally consistent for "
            "xT-delta work -- the same count as the binary and xG confirmations' passive halves, since "
            "feature_config.py is not target-specific and correlation/redundancy is target-agnostic. NO "
            "FEATURE IS ADDED, DROPPED OR RE-TIERED by this document, and no model artefact was touched. Two "
            "target-specific caveats are RECORDED, not resolved: construction coupling running through "
            "xt_before (a grid lookup at ball_x/ball_y, both locked features -- live on this leg, unlike the "
            "corrected active leg where Prompt 66 removed the equivalent mechanism), and the fact that this "
            "target's effects can sit in its exactly-zero mass, which covers 26.4% of rows here against 20.2% "
            "on the active leg. One threshold does NOT transfer cleanly and is reported rather than forced: "
            "the std-derived flat_margin spans 62.1% of this leg's rows, worse than the active leg's own "
            "53.4%, and it was recomputed from this leg's own std rather than copied -- see "
            "`target_scale.threshold_transfer_warning` in this half of the portal's other JSONs."
        ),
    }

    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    HTML_PATH.write_text(build_report_passive(output), encoding="utf-8")

    print(f"passive: {passive_count} (expected {EXPECTED_PASSIVE_COUNT}) "
          f"{'OK' if passive_count == EXPECTED_PASSIVE_COUNT else 'MISMATCH'}")
    print(f"correlation diff empty: {output['diff_is_empty']} (reused from the binary confirmation)")
    e = lc["part_e_prime_construction_coupling"]
    print(f"construction coupling: {e['strongest_vs_xt_before']['feature']} "
          f"r={e['strongest_vs_xt_before']['r_vs_xt_before']:+.4f} vs xt_before, "
          f"{e['strongest_vs_target']['feature']} r={e['strongest_vs_target']['r_vs_target']:+.4f} vs the delta")
    print(f"\nWrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
