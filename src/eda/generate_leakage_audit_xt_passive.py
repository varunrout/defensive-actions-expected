"""CLI entrypoint: leakage audit for `target_xt_delta_passive` -- PASSIVE
LEG. Computes the JSON and renders its HTML in one pass.

STEP-0 DISPOSITION: genuine passive build, and the one report type where
the leg asymmetry runs the OTHER way from everywhere else in this portal.

Checked directly rather than assumed: `reports/analysis/xg_target/
LEAKAGE_AUDIT.json` carries `"dataset": "passive"` and its Parts B/C/D test
three passive-only columns (`screened_option_was_avoided`,
`has_screened_outcome`, `has_option_2`/`has_option_3`). The same is true of
`reports/analysis/shot_target/LEAKAGE_AUDIT.json`. **The established
leakage-audit methodology in this project IS the passive one** -- it is the
ACTIVE leg that has no native counterpart, which is exactly what Prompt 65
recorded as a scope gap and substituted Parts B'/C'/D' for.

So this report is not an adaptation of the active xT audit. It is a
LITERAL MIRROR of the passive xg audit's own Parts A/B/C/D against
`target_xt_delta_passive`, with two method changes stated in the output:

  - the xg audit's plain groupby-mean + Welch's t-test becomes
    `generate_leakage_audit_xt._split_stats`, the same test with this
    target's negative / zero / positive shares reported alongside every
    mean. That is Prompt 65's Adaptation 3, and it matters here for the
    same reason it mattered there: an effect on this target can sit
    entirely in the exactly-zero mass and a mean-difference test alone
    would report nothing.
  - every p-value carries the row-grain caveat. The target is one value
    per `event_id` repeated across that event's defender slots, so Welch's
    t-test's nominal n overstates the independent information behind it by
    roughly a factor of eight. p-values are reported, never used to gate a
    verdict.

Part E' is NEW and has no counterpart in any existing audit, on either leg
or any target. It is this leg's construction-coupling check, and it is the
finding this report exists to surface: `xt_before = xT(ball_x, ball_y)`,
and `ball_x` / `ball_y` are LOCKED PASSIVE FEATURES. That is a structurally
different situation from the active leg, where Prompt 66 removed the
coupling entirely by never looking up the acting row's own coordinates.

Usage:
    python -m src.eda.generate_leakage_audit_xt_passive
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import render, xt_common as xc
from src.eda import generate_leakage_audit_xt as l1
from src.eda import generate_leakage_report_xt as lr1
from src.eda.feature_config import PASSIVE
from src.eda.generate_leakage_audit import _candidate_columns, part_a
from src.eda.render import esc, finding_card

JSON_PATH = xc.OUT_DIR / "PASSIVE_LEAKAGE_AUDIT.json"
HTML_PATH = xc.OUT_DIR / "PASSIVE_LEAKAGE_AUDIT.html"
TARGET = xc.TARGET

# xt_before is a grid lookup at exactly these two columns, and both are in
# PASSIVE["continuous"]. Checked at runtime in part_e_prime, not assumed.
XT_BEFORE_SOURCE_COLUMNS = ["ball_x", "ball_y"]

P_VALUE_CAVEAT = (
    "ROW-GRAIN CAVEAT ON EVERY P-VALUE IN THIS REPORT, stated once and applying throughout. "
    "target_xt_delta_passive is computed once per event_id and left-joined onto every visible defender-slot row "
    "of that event (mean ~8.03 rows per event), exactly as target_future_shot_10s and target_future_xg_10s "
    "already are on this leg. Welch's t-test assumes independent observations; these rows are not independent "
    "within an event, so the nominal n is roughly eight times the effective one and every p-value below is "
    "correspondingly optimistic. The existing passive xg audit states the row grain but carries no such caveat "
    "-- checked directly in reports/analysis/xg_target/LEAKAGE_AUDIT.json, not assumed -- so this is an "
    "addition rather than a mirrored convention. No verdict in this report is gated on a p-value: every one "
    "turns on an effect size, a share, or a structural argument about how a column is constructed."
)


def part_b_passive(df: pd.DataFrame) -> dict:
    """LITERAL MIRROR of generate_leakage_audit_xg.part_b_xg -- same column,
    same question, this target."""
    r = l1._split_stats(df, "screened_option_was_avoided")
    tr, fa = r["by_value"].get("True", {}), r["by_value"].get("False", {})
    r.update({
        "mirrors": "Part B of the passive xg audit (generate_leakage_audit_xg.part_b_xg), same column",
        "method": (
            "groupby mean target + Welch's t-test, the xg audit's own method, with this target's "
            "negative/zero/positive shares added (Adaptation 3)"
        ),
        "verdict": "kept-excluded-conceptually; empirical result is informational only",
        "explanation": (
            f"screened_option_was_avoided is excluded from the candidate list on conceptual/structural grounds "
            f"regardless of any empirical result, and this audit does not change that. Measured against "
            f"{TARGET} on this run: True n={tr.get('n', 0):,} mean {tr.get('mean_xt', float('nan')):+.6f} "
            f"({tr.get('pct_zero', float('nan')):.1f}% exactly zero), False n={fa.get('n', 0):,} mean "
            f"{fa.get('mean_xt', float('nan')):+.6f} ({fa.get('pct_zero', float('nan')):.1f}% exactly zero), "
            f"delta {r['delta']:+.6f}, zero-share gap {r['zero_share_gap_pp']:+.1f}pp. The xg audit recorded a "
            "near-flat difference here and read it as confirming low leakage risk for this specific pair; the "
            "same reading applies or does not on this target's own numbers above, and the exclusion holds "
            "either way because it was never empirical."
        ),
    })
    return r


def part_c_passive(df: pd.DataFrame) -> dict:
    """LITERAL MIRROR of generate_leakage_audit_xg.part_c_xg. This is the
    censoring-mechanism proxy, and the one place where this target's zero
    mass changes what the test can see."""
    r = l1._split_stats(df, "has_screened_outcome")
    tr, fa = r["by_value"].get("True", {}), r["by_value"].get("False", {})
    r.update({
        "mirrors": "Part C of the passive xg audit (generate_leakage_audit_xg.part_c_xg), same column",
        "method": (
            "groupby mean target + Welch's t-test, the xg audit's own method, with this target's "
            "negative/zero/positive shares added (Adaptation 3)"
        ),
        "feature_config_exclusion_reason": PASSIVE["excluded"].get("has_screened_outcome"),
        "verdict": "excluded -- censoring-mechanism proxy; the STRUCTURAL exclusion holds and does not depend on this target",
        "explanation": (
            "has_screened_outcome is False only when screened_option_was_avoided could not be computed -- end "
            "of period or match, no next event to compare against. feature_config.py excludes it as a "
            "censoring-mechanism proxy for target_future_shot_10s's own 10-second window being truncated "
            "(chi2=136.5, p=1.54e-31, a ~11.9x shot-rate gap). The binary audit measured it at ~12x, the xg "
            f"audit at only 1.3x and barely significant. Against {TARGET} on this run: False n={fa.get('n', 0):,} "
            f"mean {fa.get('mean_xt', float('nan')):+.6f} versus True n={tr.get('n', 0):,} mean "
            f"{tr.get('mean_xt', float('nan')):+.6f}, delta {r['delta']:+.6f}, and the exactly-zero share "
            f"differs by {r['zero_share_gap_pp']:+.1f}pp "
            f"({fa.get('pct_zero', float('nan')):.1f}% when False vs {tr.get('pct_zero', float('nan')):.1f}% "
            "when True). READ THE ZERO-SHARE GAP, NOT ONLY THE MEAN: this target's effects can sit entirely in "
            "its 26.4% exactly-zero mass, which is exactly the caveat Prompt 65 recorded as its Caveat 2 for "
            "the active leg and which applies to a LARGER share of rows here than there (26.4% vs 20.2%). "
            "CRUCIALLY, the exclusion does not rest on any of these numbers: the censoring mechanism is a "
            "property of how the column is constructed, not of which target it is measured against, so it holds "
            "identically whatever this target says. Do not cite this target's figure as independent evidence "
            "for the exclusion -- it is the same structural argument, measured a third time."
        ),
    })
    return r


def part_d_passive(df: pd.DataFrame) -> dict:
    """LITERAL MIRROR of generate_leakage_audit_xg.part_d_xg -- has_option_2
    and has_option_3, both kept as candidate features."""
    results = {}
    for col in ("has_option_2", "has_option_3"):
        r = l1._split_stats(df, col)
        tr, fa = r["by_value"].get("True", {}), r["by_value"].get("False", {})
        r.update({
            "mirrors": f"Part D of the passive xg audit (generate_leakage_audit_xg.part_d_xg), same column {col}",
            "method": "groupby mean target + Welch's t-test with direction shares (Adaptation 3)",
            "in_candidate_list": col in _candidate_columns(PASSIVE),
            "verdict": "flagged-for-follow-up -- kept as a candidate feature, same disposition as the xg audit",
            "explanation": (
                f"{col} describes the CURRENT freeze frame (whether a second/third passing option was "
                f"identifiable at all), not whether the forward window was computable, so it is not a censoring "
                f"proxy and stays in the candidate list. Against {TARGET} on this run: False n={fa.get('n', 0):,} "
                f"mean {fa.get('mean_xt', float('nan')):+.6f}, True n={tr.get('n', 0):,} mean "
                f"{tr.get('mean_xt', float('nan')):+.6f}, delta {r['delta']:+.6f}, zero-share gap "
                f"{r['zero_share_gap_pp']:+.1f}pp. The binary and xg audits both found False rows carrying "
                "MORE danger than True rows and flagged the pair for confound testing rather than exclusion; "
                "this portal's own passive Confound report runs exactly that test against this target. The "
                "direction shares are the part to read on this target: a difference in mean can be produced "
                "either by a shift in which way the threat moves or by a shift in how many rows move at all, "
                "and only the shares distinguish those."
            ),
        })
        results[col] = r
    return results


def part_e_prime_passive(df: pd.DataFrame) -> dict:
    """NEW. No counterpart in any existing audit on any leg or target.

    This leg's construction-coupling question, and it has a genuinely
    different answer from the active leg's. On active, Prompt 66 removed
    the coupling by never looking up the acting row's own coordinates, and
    Prompt 67 measured the strongest locked-feature correlation falling 74%
    to r=+0.1431. On THIS leg, `xt_before = xT(ball_x, ball_y)` and BOTH
    `ball_x` and `ball_y` are locked candidate features.
    """
    candidate_cols = _candidate_columns(PASSIVE)
    locked = sorted(set(PASSIVE["continuous"]) | set(PASSIVE["discrete"]))
    sub = df.dropna(subset=[TARGET])

    rows = []
    for col in locked:
        if col not in sub.columns or not pd.api.types.is_numeric_dtype(sub[col]):
            continue
        rows.append({
            "feature": col,
            "in_candidate_list": col in candidate_cols,
            "is_an_xt_before_source_column": col in XT_BEFORE_SOURCE_COLUMNS,
            "r_vs_xt_after": round(float(sub[col].corr(sub["xt_after"])), 4),
            "r_vs_xt_before": round(float(sub[col].corr(sub["xt_before"])), 4),
            "r_vs_target": round(float(sub[col].corr(sub[TARGET])), 4),
        })
    rows.sort(key=lambda r: abs(r["r_vs_target"]), reverse=True)
    strongest = rows[0] if rows else None
    strongest_before = max(rows, key=lambda r: abs(r["r_vs_xt_before"])) if rows else None

    source_status = {
        c: {
            "in_candidate_list": c in candidate_cols,
            "r_vs_xt_before": next((r["r_vs_xt_before"] for r in rows if r["feature"] == c), None),
            "r_vs_xt_after": next((r["r_vs_xt_after"] for r in rows if r["feature"] == c), None),
            "r_vs_target": next((r["r_vs_target"] for r in rows if r["feature"] == c), None),
        }
        for c in XT_BEFORE_SOURCE_COLUMNS
    }

    return {
        "is_new_part": True,
        "method": (
            "Pearson r of every locked numerical PASSIVE feature against xt_after, xt_before and the delta "
            "itself. The same plain instrument Prompt 65 used for the active leg's Part D', applied here for "
            "the first time -- there is no established audit to mirror, on either leg or any target, and it is "
            "labelled as new rather than presented as established."
        ),
        "why_this_part_exists": (
            "target_xt_delta_passive = xt_before - xt_after, and on THIS leg xt_before = xT(ball_x, ball_y) at "
            "the snapshot itself -- a deterministic 8x12 grid lookup at two columns that are both in "
            "PASSIVE['continuous'] and therefore both LOCKED CANDIDATE FEATURES. That is a structurally "
            "different situation from the active leg, where xt_before is a lookup at the PREVIOUS event's "
            "location (not a column in the active feature table at all) and where Prompt 66 removed the "
            "xt_after coupling entirely. Here, one of the target's two terms is a pure function of two locked "
            "features, so it must be measured rather than assumed harmless."
        ),
        "xt_before_source_columns": source_status,
        "locked_feature_correlations": rows,
        "strongest_vs_target": strongest,
        "strongest_vs_xt_before": strongest_before,
        "active_leg_comparison": {
            "active_v1_strongest_r_vs_target": 0.5551,
            "active_v1_strongest_feature": "distance_to_attacking_box",
            "active_v2_strongest_r_vs_target": 0.1431,
            "active_v2_strongest_feature": "distance_to_attacking_box",
            "note": (
                "Prompt 65 recorded construction coupling as the single most important caveat the v1 active "
                "portal carried (strongest locked feature r=+0.5551 against the target); Prompt 67 measured it "
                "falling 74% to +0.1431 under the corrected xt_after and called the caveat substantially "
                "retired. Those numbers are quoted for scale only -- the mechanism on this leg is a different "
                "one and the active figures do not transfer."
            ),
        },
        "verdict": (
            "real but BOUNDED, and bounded by subtraction rather than by exclusion"
            if strongest and abs(strongest["r_vs_target"]) < 0.5 else
            "STRONG coupling present -- see locked_feature_correlations"
        ),
        "explanation": (
            f"Measured on this run: the strongest locked-feature correlation with xt_before is "
            f"{strongest_before['feature']} at r={strongest_before['r_vs_xt_before']:+.4f}, which is exactly "
            "what a grid lookup at ball_x/ball_y should produce and is NOT itself a finding about leakage -- "
            "it is the definition of the term. The question that matters is what survives the SUBTRACTION: the "
            f"strongest locked-feature correlation with the DELTA is {strongest['feature']} at "
            f"r={strongest['r_vs_target']:+.4f}. The delta is a difference of two grid values at two nearby "
            "locations, so most of the location information cancels, and that cancellation is visible in the "
            "gap between those two columns in the table. NO FEATURE IS DROPPED OR PROPOSED FOR DROPPING ON "
            "THIS EVIDENCE, and feature_config.py is unchanged by this report. What is recorded is a reading "
            "instruction: on this leg, unlike the corrected active leg, a location-derived feature's "
            "relationship to this target should be read with the knowledge that it shares a definitional term "
            "with half of it. That is the same caveat Prompt 65 attached to the active v1 Numerical Target "
            "Atlas, arriving here through a different mechanism and at a different magnitude."
        ) if strongest and strongest_before else "no numeric locked features found",
        "not_a_modelling_recommendation": (
            "Out of scope for an EDA suite and explicitly not decided here: whether a model on this leg should "
            "use ball_x/ball_y at all when predicting this target, and whether xt_before should be offered as "
            "a feature in its own right rather than left implicit. Both are modelling-ladder questions. No "
            "model artefact was touched."
        ),
    }


# --------------------------------------------------------------------------
# Rendering. Part A's section, the direction table and the whole visual
# system are imported unchanged from the active xT renderer.
# --------------------------------------------------------------------------

def _mirrored_part_section(title: str, r: dict, banner_class: str = "v-partially") -> str:
    extra = ""
    if r.get("feature_config_exclusion_reason"):
        extra = (f'<p class="qc-note"><b>feature_config.py exclusion reason:</b> '
                 f'{esc(str(r["feature_config_exclusion_reason"]))}</p>')
    return f"""
<div class="test-block">
  <h2 class="test-title">{title}</h2>
  <p class="test-subnote">Mirrors: {esc(r['mirrors'])}. Method: {esc(r['method'])}.</p>
  <div class="verdict-banner {banner_class}"><b>Verdict: {esc(r['verdict'])}</b></div>
  <div class="qchart-card">
    <h4>mean {esc(TARGET)} by <code>{esc(r['column'])}</code></h4>
    <p class="qc-note">Delta {r['delta']:+.6f} &middot; exactly-zero share gap
    {r['zero_share_gap_pp']:+.1f}pp &middot; Welch's t={r['t_stat']:.3f}, p={r['p_value']:.3e}
    (see the row-grain caveat above &mdash; this p-value's nominal n is roughly eight times its effective one).
    Bars diverge from zero; green = xT fell (threat reduced), red = xT rose.</p>
    {lr1._dir_table(r['by_value'])}
  </div>
  {extra}
  <div class="closing-note">{esc(r['explanation'])}</div>
</div>"""


def _part_e_section(e: dict) -> str:
    rows = "".join(
        f"<tr><td><code>{esc(r['feature'])}</code>"
        + (' <b>(xt_before source)</b>' if r["is_an_xt_before_source_column"] else "")
        + f"</td><td>{r['r_vs_xt_after']:+.4f}</td><td>{r['r_vs_xt_before']:+.4f}</td>"
        f"<td>{r['r_vs_target']:+.4f}</td></tr>"
        for r in e["locked_feature_correlations"]
    )
    src = "".join(
        f"<tr><td><code>{esc(k)}</code></td>"
        f"<td>{'IN CANDIDATE LIST' if v['in_candidate_list'] else 'excluded'}</td>"
        f"<td>{v['r_vs_xt_before']:+.4f}</td><td>{v['r_vs_xt_after']:+.4f}</td>"
        f"<td>{v['r_vs_target']:+.4f}</td></tr>"
        for k, v in e["xt_before_source_columns"].items()
    )
    s, sb = e["strongest_vs_target"], e["strongest_vs_xt_before"]
    ac = e["active_leg_comparison"]
    return f"""
<div class="test-block">
  <h2 class="test-title">Part E&prime; &mdash; construction coupling on this leg
  <span class="nf-badge type" style="margin-left:8px;">NEW PART</span></h2>
  <div class="verdict-banner v-partially"><b>Verdict: {esc(e['verdict'])}</b></div>
  {finding_card("this part is new", "no counterpart in any existing audit, on either leg or any target",
                esc(e['why_this_part_exists']), flag=True)}
  <p class="test-subnote">{esc(e['method'])}</p>

  <h3 style="margin-top:22px;">The two columns <code>xt_before</code> is computed from</h3>
  <p class="test-subnote">Both are locked passive candidate features. This is the structural difference from
  the corrected active leg, where <code>xt_before</code> is a lookup at the PREVIOUS event's location and is
  not a column in the active feature table at all.</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Column</th><th>Candidate status</th><th>r vs xt_before</th><th>r vs xt_after</th><th>r vs target</th></tr>
  {src}
  </table></div>

  <h3 style="margin-top:22px;">Every locked numerical feature vs the target's two terms</h3>
  <p class="test-subnote">Ranked by |r vs target|. Strongest against the DELTA:
  <code>{esc(s['feature'])}</code> at <b>{s['r_vs_target']:+.4f}</b>. Strongest against
  <code>xt_before</code> specifically: <code>{esc(sb['feature'])}</code> at
  <b>{sb['r_vs_xt_before']:+.4f}</b>. The gap between those two columns is the subtraction doing its work.</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Locked feature</th><th>r vs xt_after</th><th>r vs xt_before</th><th>r vs target</th></tr>
  {rows}
  </table></div>

  <div class="closing-note" style="margin-top:18px;">{esc(e['explanation'])}</div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">how this compares with the active leg &mdash; for scale only</span>
  <p>{esc(ac['note'])}</p>
  </div>
  <div class="closing-note">{esc(e['not_a_modelling_recommendation'])}</div>
</div>"""


def build_report(data: dict) -> str:
    body = f"""
<div class="rule-banner"><b>Scope, confirmed by reading the source.</b> {esc(data['step_0_scope_note'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Method, and what changed from the xg mirror.</b>
{esc(data['methodology'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>{esc(data['p_value_caveat'])}</b></div>
"""
    body += lr1._part_a_section(data["part_a_known_leaky_scan"])
    body += _mirrored_part_section(
        "Part B &mdash; <code>screened_option_was_avoided</code>",
        data["part_b_screened_option_was_avoided"], "v-no")
    body += _mirrored_part_section(
        "Part C &mdash; <code>has_screened_outcome</code>, the censoring proxy",
        data["part_c_has_screened_outcome"], "v-partially")
    for col, r in data["part_d_has_option_2_3"].items():
        body += _mirrored_part_section(f"Part D &mdash; <code>{esc(col)}</code>", r, "v-partially")
    body += _part_e_section(data["part_e_prime_construction_coupling"])

    e = data["part_e_prime_construction_coupling"]
    return render.render_article(
        eyebrow="LEAKAGE AUDIT -- XT DELTA &middot; PASSIVE LEG",
        title="Passive Dataset Leakage Audit (xT delta)",
        dek=(
            "A literal mirror of the passive xg audit's Parts A/B/C/D against target_xt_delta_passive -- this "
            "is the leg the established leakage-audit methodology was written for, so nothing had to be "
            "substituted. Two method changes are stated: direction shares beside every mean, and a row-grain "
            "caveat on every p-value. Part E' is new and is this leg's own construction-coupling check."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (passive)"),
            (f"{data['n_unique_events']:,}", "unique events"),
            (str(data["part_a_known_leaky_scan"]["n_columns_scanned_total"]), "columns scanned"),
            ("4", "xg parts mirrored literally"),
            (f"{e['strongest_vs_target']['r_vs_target']:+.3f}", "strongest coupling vs target"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.VIF_CSS + render.CORR_CSS,
    )


def main() -> None:
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    candidate_cols = _candidate_columns(PASSIVE)

    a = part_a(df, candidate_cols)
    b = part_b_passive(df)
    c = part_c_passive(df)
    d = part_d_passive(df)
    e = part_e_prime_passive(df)

    output = {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg -- see LEAKAGE_AUDIT.json)",
        "parquet_path": PASSIVE["parquet_path"],
        "target_parquet_path": "outputs/prototypes/passive_xt_delta.parquet",
        "n_rows": len(df),
        "n_unique_events": int(df["event_id"].nunique()),
        "target_col": TARGET,
        "target_scale": xc.target_scale(df),
        "step_0_scope_note": (
            "CONFIRMED BY READING THE SOURCE, and it points the OPPOSITE way from every other scope note in "
            "this portal: reports/analysis/xg_target/LEAKAGE_AUDIT.json carries \"dataset\": \"passive\" and "
            "its Parts B/C/D test three passive-only columns; reports/analysis/shot_target/LEAKAGE_AUDIT.json "
            "is the same. The established leakage-audit methodology in this project IS the passive one, and it "
            "is the ACTIVE leg that has no native counterpart -- which is exactly the scope gap Prompt 65 "
            "recorded and substituted its Parts B'/C'/D' for. This report therefore needs no substitution at "
            "all: Parts A/B/C/D are literal mirrors of the passive xg audit's own parts, same columns, same "
            "questions. Part E' is new."
        ),
        "methodology": (
            "Part A (target- and dataset-independent schema/naming scan) imported unchanged from "
            "generate_leakage_audit.part_a. Parts B/C/D use generate_leakage_audit_xt._split_stats -- the xg "
            "audit's own groupby-mean + Welch's t-test, with this target's negative/zero/positive shares added "
            "(Prompt 65's Adaptation 3), because an effect on this target can sit entirely in its 26.4% "
            "exactly-zero mass and a mean-difference test alone would report nothing. Part E' is a "
            "construction-coupling check with no counterpart anywhere in this project, added because this "
            "leg's xt_before term is a deterministic grid lookup at two LOCKED features."
        ),
        "p_value_caveat": P_VALUE_CAVEAT,
        "no_v1_supersession": pc.PROMPT_68_NO_V1_DETOUR_FINDING,
        "part_a_known_leaky_scan": a,
        "part_b_screened_option_was_avoided": b,
        "part_c_has_screened_outcome": c,
        "part_d_has_option_2_3": d,
        "part_e_prime_construction_coupling": e,
        "prompt_68_cross_references": {
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "possession_ending_collapse": pc.PROMPT_68_POSSESSION_COLLAPSE_FINDING,
            "correlation_with_old_targets": pc.PROMPT_68_CORRELATION_FINDING,
        },
    }

    JSON_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    HTML_PATH.write_text(build_report(output), encoding="utf-8")

    print("=== Part A ===", "clean" if a["verdict"] == "clean" else "ISSUES FOUND")
    print(f"Part B screened_option_was_avoided: delta {b['delta']:+.6f}, zero gap {b['zero_share_gap_pp']:+.1f}pp")
    print(f"Part C has_screened_outcome: delta {c['delta']:+.6f}, zero gap {c['zero_share_gap_pp']:+.1f}pp, "
          f"p={c['p_value']:.3e}")
    for col, r in d.items():
        print(f"Part D {col}: delta {r['delta']:+.6f}, zero gap {r['zero_share_gap_pp']:+.1f}pp")
    e_s, e_sb = e["strongest_vs_target"], e["strongest_vs_xt_before"]
    print(f"Part E' strongest vs xt_before: {e_sb['feature']} r={e_sb['r_vs_xt_before']:+.4f}; "
          f"strongest vs delta: {e_s['feature']} r={e_s['r_vs_target']:+.4f}")
    print(f"\nWrote {JSON_PATH} and {HTML_PATH}")


if __name__ == "__main__":
    main()
