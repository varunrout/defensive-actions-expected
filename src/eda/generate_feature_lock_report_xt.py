"""CLI entrypoint: render
reports/analysis/xt_target/FEATURE_LOCK_CONFIRMATION_XT.json as a
self-contained HTML report -- the xT sibling of
generate_feature_lock_report_xg.py, which renders the combined
confirmation-plus-pattern-findings document in the xg portal.

Usage:
    python -m src.eda.generate_feature_lock_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc, finding_card

INPUT_PATH = xc.OUT_DIR / "FEATURE_LOCK_CONFIRMATION_XT.json"
OUTPUT_PATH = xc.OUT_DIR / "FEATURE_LOCK_CONFIRMATION_XT.html"


def _count_card(label: str, actual: int, expected: int, matches: bool) -> str:
    return f"""
<div class="ba-card {'after' if matches else 'before'}">
  <h4>{esc(label)}</h4>
  <p style="font-size:28px; font-family:'Archivo',sans-serif; font-weight:800; margin:4px 0;">{actual}</p>
  <p style="font-size:12px; color:var(--text-muted); margin:0 0 8px;">expected {expected}</p>
  <span class="verdict-chip {'distinct' if matches else 'drop'}">{'CONFIRMED' if matches else 'MISMATCH'}</span>
</div>"""


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    header = "".join(f"<th style='padding:5px 8px;'>{esc(lbl)}</th>" for _, lbl in cols)
    body = "".join(
        "<tr>" + "".join(
            f"<td style='padding:5px 8px;border-top:1px solid var(--border);'>{esc(str(r.get(k, '')))}</td>"
            for k, _ in cols) + "</tr>"
        for r in rows
    )
    return (f'<table style="width:100%;border-collapse:collapse;font-size:12.5px;">'
            f'<thead><tr style="text-align:left;color:var(--text-muted);">{header}</tr></thead>'
            f"<tbody>{body}</tbody></table>")


def _leakage_section(lc: dict) -> str:
    b, c, d = (lc["part_b_prime_has_previous_event"], lc["part_c_prime_action_possession_flags"],
               lc["part_d_prime_construction_coupling"])
    s = d["strongest"]
    verdict_rows = "".join(f"<li><code>{esc(k)}</code>: <b>{esc(v)}</b></li>" for k, v in c["verdicts"].items())

    return f"""
<h2 style="margin-top:40px;">Leakage confirmation (xT delta)</h2>
<p class="section-note">Source: <code>{esc(lc['source'])}</code></p>
{finding_card("scope differs from the xG confirmation", "said so, not glossed", esc(lc['scope_difference_from_xg']))}

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part A &mdash; schema / naming scan</h5>
  <p class="stratum-delta">Verdict: <b>{esc(lc['part_a_verdict'])}</b> (method imported unchanged).</p>
</div>

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part B&prime; &mdash; has_previous_event</h5>
  <p class="stratum-delta">Exactly-zero share gap <b>{b['zero_share_gap_pp']:+.1f}pp</b>; Welch's t-test on the
  MEAN gives p={b['p_value_on_the_mean']:.4g}. Verdict: <b>{esc(b['verdict'])}</b>.</p>
  <p class="stratum-delta">{esc(b['implication_for_the_lock'])}</p>
</div>

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part C&prime; &mdash; the three action_*_possession flags</h5>
  <ul style="font-size:12.5px;margin:8px 0 8px 18px;">{verdict_rows}</ul>
  <p class="stratum-delta">{esc(c['implication_for_the_lock'])}</p>
</div>

<div class="stratum-card">
  <h5>Part D&prime; &mdash; construction coupling (new to this target)</h5>
  <p class="stratum-delta">Strongest: <code>{esc(s['feature'])}</code> at r={s['r_vs_target']:+.4f} against the
  target and r={s['r_vs_xt_after']:+.4f} against <code>xt_after</code>. Verdict: <b>{esc(d['verdict'])}</b>.</p>
  <p class="stratum-delta">{esc(d['implication_for_the_lock'])}</p>
</div>

<div class="verdict-banner v-no" style="margin-top:20px;"><b>{esc(lc['verdict'])}</b></div>"""


def _pattern_section(pf: dict) -> str:
    ts = pf["target_shape"]
    nr = pf["numerical_vs_target_rankings"]
    ca = pf["category_atlas"]
    fl = pf["flag_ledger"]
    ct = pf["confound_tests"]
    tr = pf["tournament_stability"]
    ss = pf["slice_stratification"]
    ni = pf["numeric_interaction"]
    rv = pf["review_tier"]
    cr = pf["cross_referenced_target_independent_sections"]

    poss_rows = "".join(
        f"<li><code>{esc(k)}</code>: True-group mean <b>{v['mean_xt_true']:+.6f}</b> over n={v['n_true']:,}, "
        f"only {v['pct_zero_true']:.1f}% exactly zero</li>"
        for k, v in fl["possession_flags_no_longer_degenerate"].items()
    )

    return f"""
<h2 style="margin-top:48px;padding-top:24px;border-top:2px solid var(--border);">Pattern-analysis findings (xT delta)</h2>
<p class="section-note">{esc(pf['source_note'])}</p>

<h3 style="margin-top:28px;">The target's shape</h3>
<p class="section-note">mean <b>{ts['mean']:+.6f}</b>, std <b>{ts['std']:.6f}</b>, skew <b>{ts['skew']:+.3f}</b>,
excess kurtosis <b>{ts['excess_kurtosis']:.2f}</b>, <b>{ts['pct_negative']:.1f}%</b> negative /
<b>{ts['pct_positive']:.1f}%</b> positive / <b>{ts['pct_zero']:.1f}%</b> exactly zero, on
{ts['n_defined']:,} rows with a defined delta ({ts['n_nan']} NaN, explained in Prompt 64 section 2).</p>
{finding_card("carried forward from Prompt 64 -- CONFIRMED", "symmetric, negative-capable target", esc(xc.PROMPT_64_SHAPE_FINDING), flag=True)}

<h3 style="margin-top:28px;">Numerical features vs the target</h3>
<p class="section-note">{esc(nr['note'])} {nr['n_curves_crossing_zero']}/{nr['n_features']} features' binned
curves cross zero; {nr['n_train_test_inconsistent']} feature(s) flagged train/test-inconsistent.</p>
{_table(nr['active_top'], [('feature', 'feature'), ('spearman_rho', 'rho'), ('shape', 'shape'), ('binned_curve_crosses_zero', 'crosses zero')])}

<h3 style="margin-top:28px;">Category Atlas &mdash; the Clearance artefact</h3>
{finding_card("carried forward from Prompt 64 -- RESURFACED", f"Clearance ranks {esc(ca['clearance_rank'])}", esc(ca['finding']), flag=True)}

<h3 style="margin-top:28px;">Flag Ledger &mdash; top 5 by |lift|</h3>
{_table(fl['top_by_abs_lift'], [('column', 'column'), ('lift', 'lift'), ('mean_xt_true', 'mean True'), ('mean_xt_false', 'mean False'), ('n_true', 'n True')])}
<div class="finding flag" style="margin:16px 0;">
<span class="tag">carried forward from Prompt 64 -- the motivating problem, confirmed fixed (off-pool)</span>
<p>The three <code>action_*_possession</code> flags are 100% identical zeros against
<code>target_future_shot_10s</code>. Against this target:</p>
<ul>{poss_rows}</ul>
<p>{esc(fl['possession_flags_off_pool_note'])}</p>
</div>

<h3 style="margin-top:28px;">Confound (reversal) tests</h3>
<p class="section-note">{esc(ct['note'])}</p>
{_table(ct['tests'], [('title', 'test'), ('verdict', 'unconditional'), ('verdict_given_nonzero_delta', 'non-zero delta'), ('is_new_test', 'new test')])}

<h3 style="margin-top:28px;">Tournament stability</h3>
<p class="section-note">{tr['n_genuine']}/{tr['n_features']} active features show a genuine tournament-level
difference rather than arbitrary train/test noise.</p>
{_table(tr['features'], [('feature', 'feature'), ('verdict', 'verdict')])}

<h3 style="margin-top:28px;">Slice stratification</h3>
<p class="section-note">V1: {ss['v1']['n_cells']} cells, {ss['v1']['n_genuine_divergences']} unconditional
genuine divergences, {ss['v1']['n_genuine_divergences_given_nonzero_delta']} on the non-zero-delta subset,
{ss['v1']['n_conditioning_disagreements']} categories where removing the zero mass flips the conclusion.
V2: {ss['v2']['n_cells']} cells, {ss['v2']['n_genuine_divergences']} unconditional /
{ss['v2']['n_genuine_divergences_given_nonzero_delta']} conditional.</p>
{finding_card("archetype-vs-boolean comparison", "not reproducible on the active leg", esc(ss['v2']['archetype_comparison']))}

<h3 style="margin-top:28px;">Numeric x numeric interaction</h3>
<p class="section-note">{ni['n_agree']}/{ni['n_pairs']} pairs classify the same way unconditionally and on the
non-zero-delta subset. {ni['n_sign_flipping']}/{ni['n_pairs']} have within-stratum deltas that change SIGN --
a direction reversal that cannot arise on the xG target.</p>
{_table(ni['pairs'], [('feature_a', 'feature A'), ('feature_b', 'feature B'), ('classification', 'classification'), ('within_stratum_deltas_change_sign', 'sign flip')])}

<h3 style="margin-top:28px;">Review tier</h3>
<p class="section-note">{rv['n_review_pairs']} ACTIVE review pairs, {rv['n_needs_human_call']} still
needs_human_call, {rv['n_type2_opposite_sign']} Type-2 pair(s) resolved on the opposite-sign rule.</p>

<h3 style="margin-top:28px;">Target-independent reports &mdash; linked, not rebuilt</h3>
{"".join(finding_card(k.replace('_', ' '), "linked from this portal's index", v) for k, v in cr.items())}
"""


def build_report(data: dict) -> str:
    counts = data["confirmed_counts"]
    d = data["correlation_diff"]["active"]
    verdict_class = "v-no" if not data["any_newly_risky_pairs_found"] else "v-yes"

    body = f"""
<div class="verdict-banner {verdict_class}" style="margin-bottom:28px;">
<b>{'LOCKED (xT delta) -- no new tier crossings' if not data['any_newly_risky_pairs_found'] else 'ATTENTION -- new tier crossing(s) found'}</b>
&mdash; {esc(data['verdict'])}
</div>

<h2>Confirmed candidate feature count</h2>
<p class="section-note">Read directly from feature_config.py's ACTIVE dict, not assumed. Same count as the
binary and xG confirmations -- feature_config.py is not target-specific.</p>
<div class="before-after">
{_count_card('Active', counts['active'], counts['active_expected'], counts['active_matches_expected'])}
</div>

<h2 style="margin-top:40px;">Correlation diff vs. the last confirmed run</h2>
{finding_card("correlation diff -- identical by construction", "cross-referenced, not re-derived", data["correlation_diff_note"])}
<div class="vif-dataset-block">
  <h2 class="dataset-title">Active</h2>
  <p class="dataset-substat">{d['n_pairs_before']} DROP/COLLAPSE/REVIEW pairs before &rarr; {d['n_pairs_after']} after
  &middot; {d['n_added']} added &middot; {d['n_removed']} removed &middot; {d['n_tier_changed']} tier changes</p>
  <div class="corr-ledger"><div class="corr-row"><span class="corr-pair">
  {'No changes -- diff is empty.' if data['diff_is_empty'] else 'Non-empty, see JSON.'}
  </span></div></div>
</div>

{_leakage_section(data['leakage_confirmation_xt'])}
{_pattern_section(data['pattern_analysis_findings'])}
"""

    return render.render_article(
        eyebrow="FEATURE LOCK CONFIRMATION -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="Post-Lock Confirmation + Pattern Findings (xT delta)",
        dek=(
            "Correlation/redundancy is target-agnostic and is reused directly rather than re-derived; the "
            "leakage side and every pattern finding are rebuilt from this portal's own outputs. Combined into "
            "one document, matching the xG portal's own confirmation-plus-findings layout. Active-binary leg "
            "only."
        ),
        stats=[
            (str(counts["active"]), "active features"),
            ("0" if data["diff_is_empty"] else "!", "tier changes found"),
            ("2", "target-specific caveats recorded"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + render.VIF_CSS + render.CONFOUND_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
