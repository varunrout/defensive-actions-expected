"""CLI entrypoint: render reports/eda_xg/FEATURE_LOCK_CONFIRMATION_XG.json
as a self-contained HTML report -- the continuous-target counterpart to
FEATURE_LOCK_CONFIRMATION.html.

Usage:
    python -m src.eda.generate_feature_lock_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "FEATURE_LOCK_CONFIRMATION_XG.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "FEATURE_LOCK_CONFIRMATION_XG.html"


def _count_card(label: str, actual: int, expected: int, matches: bool) -> str:
    cls = "distinct" if matches else "drop"
    status = "CONFIRMED" if matches else "MISMATCH"
    return f"""
<div class="ba-card {'after' if matches else 'before'}">
  <h4>{esc(label)}</h4>
  <p style="font-size:28px; font-family:'Archivo',sans-serif; font-weight:800; margin:4px 0;">{actual}</p>
  <p style="font-size:12px; color:var(--text-muted); margin:0 0 8px;">expected {expected}</p>
  <span class="verdict-chip {cls}">{status}</span>
</div>"""


def _diff_section(ds_key: str, d: dict) -> str:
    is_empty = d["n_added"] == 0 and d["n_removed"] == 0 and d["n_tier_changed"] == 0
    body = '<div class="corr-row"><span class="corr-pair">No changes -- diff is empty for this dataset.</span></div>' if is_empty else "(non-empty, see JSON)"
    return f"""
<div class="vif-dataset-block">
  <h2 class="dataset-title">{esc(ds_key.title())}</h2>
  <p class="dataset-substat">{d['n_pairs_before']} DROP/COLLAPSE/REVIEW pairs before &rarr; {d['n_pairs_after']} after &middot;
  {d['n_added']} added &middot; {d['n_removed']} removed &middot; {d['n_tier_changed']} tier changes</p>
  <div class="corr-ledger">{body}</div>
</div>"""


def _leakage_section(lc: dict) -> str:
    pc = lc["part_c_has_screened_outcome"]
    pd_ = lc["part_d_has_option_2_3"]
    followup = pd_["followup_status"]

    followup_html = ""
    if isinstance(followup, dict):
        verdict_rows = "".join(f"<li><code>{esc(k)}</code>: <b>{esc(v)}</b></li>" for k, v in followup["verdicts"].items())
        followup_html = f"""
<div class="finding" style="margin-top:12px;">
<span class="tag">follow-up located: {esc(followup['closed_by'])}</span>
<p>{followup['n_tests']} confound test(s) found closing this item:</p>
<ul>{verdict_rows}</ul>
</div>"""
    else:
        followup_html = finding_card("follow-up NOT located", "do not assume it landed", esc(str(followup)))

    return f"""
<h2 style="margin-top:40px;">Leakage confirmation (continuous target)</h2>
<p class="section-note">Source: <code>{esc(lc['source'])}</code></p>

<div class="stratum-card" style="margin-bottom:16px;">
  <h5>Part C -- has_screened_outcome</h5>
  <p class="stratum-delta">ratio (True:False mean xG) = <b>{pc['ratio_true_to_false']}x</b>,
  t={pc['t_stat']:.3f}, p={pc['p_value']:.4g} &mdash; verdict: <b>{esc(pc['verdict'])}</b></p>
  <p class="stratum-delta">{esc(pc['comparison_to_binary'])}</p>
</div>

<div class="stratum-card">
  <h5>Part D -- has_option_2 / has_option_3</h5>
  <p class="stratum-delta">LEAKAGE_AUDIT.json verdicts: {esc(str(pd_['leakage_audit_verdicts']))}</p>
  <p class="stratum-delta">{esc(pd_['leakage_audit_note'])}</p>
  {followup_html}
</div>

<div class="verdict-banner v-no" style="margin-top:20px;">
<b>{esc(lc['verdict'])}</b>
</div>"""


def _kv_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    header = "".join(f"<th style='padding:5px 8px;'>{esc(label)}</th>" for _, label in cols)
    body = "".join(
        "<tr>" + "".join(f"<td style='padding:5px 8px;border-top:1px solid var(--border);'>{esc(str(r.get(key, '')))}</td>" for key, _ in cols) + "</tr>"
        for r in rows
    )
    return f'<table style="width:100%;border-collapse:collapse;font-size:12.5px;"><thead><tr style="text-align:left;color:var(--text-muted);">{header}</tr></thead><tbody>{body}</tbody></table>'


def _pattern_findings_section_xg(pf: dict) -> str:
    corrections_html = "".join(f"<li>{esc(c)}</li>" for c in pf["corrections_to_prompt_34s_premise"])
    nr = pf["numerical_vs_target_rankings_xg"]
    sz = pf["structural_zero_check"]
    ct = pf["confound_tests_xg"]
    ss = pf["slice_stratification_xg"]
    ts = pf["tournament_stability_xg"]
    ni = pf["numeric_interaction_xg"]
    cr = pf["cross_referenced_target_agnostic_sections"]
    ex = ss["occurrence_vs_quality_worked_example"]

    return f"""
<h2 style="margin-top:48px;padding-top:24px;border-top:2px solid var(--border);">Pattern-analysis findings (continuous target)</h2>
<p class="section-note">{esc(pf['source_note'])}</p>

{finding_card("premise corrections", "verified before writing anything", "<ul>" + corrections_html + "</ul>")}

<h3 style="margin-top:28px;">Numerical-vs-target rankings (xG)</h3>
<div class="before-after">
<div class="ba-card"><h4>Active (top 5)</h4>{_kv_table(nr['active_top'], [('feature','feature'),('spearman_rho','rho'),('shape','shape')])}</div>
<div class="ba-card"><h4>Passive (top 5)</h4>{_kv_table(nr['passive_top'], [('feature','feature'),('spearman_rho','rho'),('shape','shape')])}</div>
</div>

<h3 style="margin-top:28px;">Structural-zero check</h3>
<p class="section-note">Verdict: <b>{esc(sz['overall_verdict'])}</b>. {esc(sz['note'])}</p>

<h3 style="margin-top:28px;">xG confound-test reproductions</h3>
<p class="section-note">{esc(ct['note'])}</p>
{_kv_table(ct['tests'], [('name','test'),('verdict','verdict')])}

<h3 style="margin-top:28px;">Slice stratification (xG)</h3>
<p class="section-note">Unconditional genuine divergences: {ss['n_genuine_divergences_unconditional']}. Given-shot:
{ss['n_genuine_divergences_given_shot']}. Conditioning disagreements: {ss['n_conditioning_disagreements']}.</p>
{finding_card(f"occurrence-vs-quality worked example -- {ex['feature']} x {ex['slicer']} / {ex['category']}", "diverges unconditionally, flat given-shot", esc(ex['takeaway']))}

<h3 style="margin-top:28px;">Tournament stability (xG)</h3>
<p class="section-note">{esc(ts['target_agnosticism_note'])}</p>
{_kv_table(ts['features'], [('feature','feature'),('dataset','dataset'),('verdict','verdict')])}

<h3 style="margin-top:28px;">Numeric x numeric interaction (xG)</h3>
<p class="section-note">{esc(ni['target_agnosticism_note'])} Agreement between unconditional and given-shot:
{ni['n_pairs_where_unconditional_and_given_shot_agree']}/10 pairs.</p>
<p class="section-note">Unconditional classification counts: {esc(str(ni['unconditional']['classification_counts']))}.
Given-shot: {esc(str(ni['given_shot']['classification_counts']))}.</p>
{_kv_table(ni['unconditional']['pairs'], [('feature_a','feature A'),('feature_b','feature B'),('dataset','dataset'),('classification','unconditional')])}

<h3 style="margin-top:28px;">Target-agnostic sections -- cross-referenced, not duplicated</h3>
{finding_card("slicer redundancy", "see the binary file", cr['slicer_redundancy'])}
{finding_card("player-level validity", "see the binary file", cr['player_level_validity'])}
"""


def build_report(data: dict) -> str:
    counts = data["confirmed_counts"]
    verdict_class = "v-no" if not data["any_newly_risky_pairs_found"] else "v-yes"

    cross_ref_card = finding_card(
        "correlation diff -- identical by construction",
        "cross-referenced, not re-derived",
        data["correlation_diff_note"],
    )

    body = f"""
<div class="verdict-banner {verdict_class}" style="margin-bottom:28px;">
<b>{'LOCKED (xG) -- no new tier crossings' if not data['any_newly_risky_pairs_found'] else 'ATTENTION -- new tier crossing(s) found'}</b>
&mdash; {esc(data['verdict'])}
</div>

<h2>Confirmed candidate feature counts</h2>
<p class="section-note">Read directly from feature_config.py's ACTIVE/PASSIVE dicts, not assumed. Same counts as
the binary confirmation -- feature_config.py is not target-specific.</p>
<div class="before-after">
{_count_card('Active', counts['active'], counts['active_expected'], counts['active_matches_expected'])}
{_count_card('Passive', counts['passive'], counts['passive_expected'], counts['passive_matches_expected'])}
</div>

<h2 style="margin-top:40px;">Correlation diff vs. the last confirmed run</h2>
{cross_ref_card}
{_diff_section('active', data['correlation_diff']['active'])}
{_diff_section('passive', data['correlation_diff']['passive'])}

{_leakage_section(data['leakage_confirmation_xg'])}
{_pattern_findings_section_xg(data['pattern_analysis_findings']) if 'pattern_analysis_findings' in data else ''}
"""

    return render.render_article(
        eyebrow="FEATURE LOCK CONFIRMATION -- XG",
        title="Post-Lock Correlation Confirmation (Continuous Target)",
        dek=(
            "reports/eda/FEATURE_LOCK_CONFIRMATION.html only ever confirmed the lock against the binary target's "
            "leakage check. Correlation/redundancy is target-agnostic (reused directly, not re-derived), but the "
            "leakage side is target-specific -- this folds the continuous-target leakage checks "
            "(LEAKAGE_AUDIT.json Parts C/D) into a formal confirmation the way prompt 14 did for binary. Extended "
            "(prompt 34) with a pattern-analysis-findings section, xG-specific -- this is now the single "
            "confirmation-plus-findings reference for the continuous target."
        ),
        stats=[
            (str(counts["active"]), "active features"),
            (str(counts["passive"]), "passive features"),
            ("0" if data["diff_is_empty"] else "!", "tier changes found"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + render.VIF_CSS + render.CONFOUND_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
