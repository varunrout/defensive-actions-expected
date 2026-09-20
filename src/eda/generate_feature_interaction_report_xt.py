"""CLI entrypoint: render
reports/analysis/xt_target/FEATURE_INTERACTION_ANALYSIS.json as a
self-contained HTML report -- the xT sibling of
generate_feature_interaction_report_xg.py.

The heatmap is the one deliberate visual change: the xg renderer maps
lo..hi onto a single-hue alpha ramp, which on a signed target would colour
a strongly threat-REDUCING cell and a mildly threat-increasing one almost
identically. Here the ramp is diverging around zero -- green for positive
(threat reduced), red for negative -- so the sign is legible at a glance.

Usage:
    python -m src.eda.generate_feature_interaction_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc, finding_card

INPUT_PATH = xc.OUT_DIR / "FEATURE_INTERACTION_ANALYSIS.json"
OUTPUT_PATH = xc.OUT_DIR / "FEATURE_INTERACTION_ANALYSIS.html"
XG_INPUT_PATH = xc.REPO_ROOT / "reports" / "analysis" / "xg_target" / "FEATURE_INTERACTION_ANALYSIS.json"

CLASS_BADGE = {
    "interactive": ("v-yes", "INTERACTIVE"),
    "additive": ("v-no", "ADDITIVE"),
    "substitutive": ("v-partially", "SUBSTITUTIVE"),
    "inconclusive": ("v-partially", "INCONCLUSIVE"),
}

EXTRA_CSS = """
.hm-table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.hm-table th { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); padding: 6px 8px; text-align: center; }
.hm-cell { text-align: center; padding: 10px 8px; border: 1px solid var(--border); font-size: 12px; border-radius: 4px; }
.hm-cell.hm-empty { background: var(--plane); color: var(--text-muted); }
.hm-n { font-size: 10px; color: var(--text-muted); }
.hm-key { font-size: 11px; color: var(--text-muted); margin: 4px 0 0; }
.sr-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0 20px; }
.sr-table th { text-align: left; color: var(--text-muted); padding: 5px 8px; }
.sr-table td { padding: 5px 8px; border-top: 1px solid var(--border); }
"""


def _heatmap(grid: list[dict], a_labels, b_labels, value_key: str) -> str:
    """Diverging colour ramp centred on zero -- see module docstring."""
    values = [c[value_key] for c in grid if c[value_key] is not None]
    max_abs = max([abs(v) for v in values] + [1e-12])
    n_key = "n" if value_key == "mean_xt" else "n_given_nonzero_delta"

    def cell(a, b):
        c = next((c for c in grid if c["a_bin"] == a and c["b_bin"] == b), None)
        if c is None or c[value_key] is None:
            return '<td class="hm-cell hm-empty">n/a</td>'
        v = c[value_key]
        t = abs(v) / max_abs
        rgb = "67,160,71" if v >= 0 else "227,73,72"
        bg = f"rgba({rgb},{0.06 + 0.60 * t:.2f})"
        return (f'<td class="hm-cell" style="background:{bg};"><b>{v:+.6f}</b><br>'
                f'<span class="hm-n">n={c[n_key]:,}</span></td>')

    header = "".join(f"<th>{esc(a)}</th>" for a in a_labels)
    rows = "".join(f"<tr><th>{esc(b)}</th>{''.join(cell(a, b) for a in a_labels)}</tr>" for b in b_labels)
    return (f'<table class="hm-table"><thead><tr><th></th>{header}</tr></thead><tbody>{rows}</tbody></table>'
            '<p class="hm-key">Green = positive mean delta (xT fell, threat reduced). Red = negative (xT rose). '
            'Intensity is |value| relative to the strongest cell in this grid.</p>')


def _gradient_table(gradient: list[dict]) -> str:
    rows = "".join(
        f"<tr><td>{esc(g['b_bin'])}</td><td>{esc(str(g['a_delta_within_stratum']))}</td>"
        f"<td>{esc(str(g['a_bin_compared_low']))} &rarr; {esc(str(g['a_bin_compared_high']))}</td>"
        f"<td>{g['n_rows']:,}</td></tr>"
        for g in gradient
    )
    return ('<table class="sr-table"><thead><tr><th>B stratum</th><th>A delta within stratum</th>'
            f'<th>A bins compared</th><th>n</th></tr></thead><tbody>{rows}</tbody></table>')


def _pair_section(p: dict, xg_cls: str | None) -> str:
    c_cls, c_lbl = CLASS_BADGE[p["classification"]]
    n_cls, n_lbl = CLASS_BADGE[p["classification_given_nonzero_delta"]]

    agree_note = ""
    if p["classification"] != p["classification_given_nonzero_delta"]:
        agree_note = f"""
<div class="finding flag" style="margin:12px 0;">
<span class="tag">unconditional and non-zero-delta disagree</span>
<p>Unconditional: <b>{esc(p['classification'])}</b>. Given a non-zero delta:
<b>{esc(p['classification_given_nonzero_delta'])}</b>. Don't assume the unconditional conclusion survives
removing this target's structural-zero mass.</p>
</div>"""

    xg_note = ""
    if xg_cls is not None:
        if xg_cls != p["classification"]:
            xg_note = f"""
<div class="finding flag" style="margin:12px 0;">
<span class="tag">diverges from the xG version of this same pair</span>
<p>xG target classification: <b>{esc(xg_cls)}</b>. xT-delta: <b>{esc(p['classification'])}</b>. Same pair, same
4x4 method, same ratio cutoffs -- the two targets genuinely disagree.</p>
</div>"""
        else:
            xg_note = f"""
<div class="finding" style="margin:12px 0;">
<span class="tag">agrees with the xG version of this same pair</span>
<p>Both targets classify this pair as <b>{esc(xg_cls)}</b>.</p>
</div>"""

    sign_note = ""
    if p["within_stratum_deltas_change_sign"]:
        sign_note = """
<div class="finding flag" style="margin:12px 0;">
<span class="tag">within-stratum deltas change sign</span>
<p>Feature A's gradient points one way in some strata of B and the opposite way in others -- on this target that
means A's effect <b>reverses direction</b> (threat-reducing at some levels of B, threat-increasing at others),
not merely that it crosses a base rate. This distinction cannot arise on the xG target, whose binned means are
non-negative by construction.</p>
</div>"""

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(p['feature_a'])} &times; {esc(p['feature_b'])}
  <span style="font-weight:400;font-size:15px;color:var(--text-muted);">(active, xT delta)</span></h2>
  {finding_card("football rationale", f"{p['feature_a']} x {p['feature_b']}", p["football_rationale"])}
  {xg_note}
  {sign_note}
  <div class="verdict-banner {c_cls}"><b>Unconditional: {esc(c_lbl)}</b><br>{esc(p['classification_reason'])}</div>
  <div class="verdict-banner {n_cls}"><b>Given a non-zero delta: {esc(n_lbl)}</b><br>{esc(p['classification_reason_given_nonzero_delta'])}</div>
  {agree_note}
  <p class="test-subnote">n={p['n_rows_used']:,} rows unconditional, {p['n_rows_used_given_nonzero_delta']:,} with a
  non-zero delta. Marginal A delta: {esc(str(p['marginal_a_delta']))} unconditional,
  {esc(str(p['marginal_a_delta_given_nonzero_delta']))} conditional.</p>
  <h4>Unconditional 4&times;4 grid (mean {esc(xc.TARGET)})</h4>
  {_heatmap(p['grid'], p['a_labels'], p['b_labels'], 'mean_xt')}
  <h4 style="margin-top:16px;">Non-zero-delta 4&times;4 grid</h4>
  {_heatmap(p['grid'], p['a_labels'], p['b_labels'], 'mean_xt_given_nonzero_delta')}
  <h4 style="margin-top:20px;">{esc(p['feature_a'])}'s unconditional gradient by {esc(p['feature_b'])} stratum</h4>
  {_gradient_table(p['a_gradient_by_b_stratum'])}
  <h4>{esc(p['feature_a'])}'s non-zero-delta gradient by {esc(p['feature_b'])} stratum</h4>
  {_gradient_table(p['a_gradient_by_b_stratum_given_nonzero_delta'])}
</div>"""


def _summary_table(rows: list[dict]) -> str:
    body = "".join(
        f"<tr><td><code>{esc(r['feature_a'])}</code></td><td><code>{esc(r['feature_b'])}</code></td>"
        f'<td><span class="verdict-banner {CLASS_BADGE[r["classification"]][0]}" style="display:inline-block;padding:2px 10px;margin:0;font-size:11px;">{esc(r["classification"])}</span></td>'
        f'<td><span class="verdict-banner {CLASS_BADGE[r["classification_given_nonzero_delta"]][0]}" style="display:inline-block;padding:2px 10px;margin:0;font-size:11px;">{esc(r["classification_given_nonzero_delta"])}</span></td>'
        f"<td>{'yes' if r['within_stratum_deltas_change_sign'] else 'no'}</td></tr>"
        for r in rows
    )
    return f"""
<div class="qchart-card" style="overflow-x:auto;">
<table class="sr-table">
<thead><tr><th>feature A</th><th>feature B</th><th>unconditional</th><th>non-zero delta</th>
<th>stratum deltas change sign</th></tr></thead>
<tbody>{body}</tbody></table>
</div>"""


def build_report(data: dict, xg: dict | None) -> str:
    xg_cls = {}
    if xg:
        for p in xg["pairs"]:
            if p["dataset"] == "active":
                xg_cls[(p["feature_a"], p["feature_b"])] = p["classification"]

    n_by_class = {}
    for r in data["summary_table"]:
        n_by_class[r["classification"]] = n_by_class.get(r["classification"], 0) + 1

    body = f"""
<div class="rule-banner"><b>Scope.</b> {esc(data['step_0_scope_note'])}</div>
{finding_card("candidate selection", "same ACTIVE pairs as the binary and xG files", data["candidate_selection"])}
{finding_card("thresholds -- what was and was not adapted", "stated explicitly", esc(data["thresholds"]["note"]))}
<div class="rule-banner" style="margin-top:16px;"><b>Reading the grids.</b>
{esc(data['target_scale']['direction_note'])}</div>
<h2 class="test-title" style="margin-top:32px;">Summary -- interactive pairs first (unconditional)</h2>
{_summary_table(data['summary_table'])}
{"".join(_pair_section(p, xg_cls.get((p['feature_a'], p['feature_b']))) for p in data['pairs'])}
"""

    return render.render_article(
        eyebrow="FEATURE INTERACTION ANALYSIS -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="Numeric x Numeric Interaction Analysis (xT delta)",
        dek=(
            "The same ACTIVE pairs and the same 4x4 quartile-grid method as the binary and xG versions, with an "
            "unconditional and a non-zero-delta classification for each. Only the minimum-marginal-delta "
            "threshold needed adapting -- the substitutive/additive cutoffs are ratios, and carry over "
            "unchanged. Active-binary leg only."
        ),
        stats=[
            (str(len(data["pairs"])), "pairs tested"),
            (str(n_by_class.get("interactive", 0)), "interactive unconditionally"),
            (str(data["n_pairs_where_unconditional_and_conditional_agree"]), "pairs where both panels agree"),
            (str(data["n_pairs_with_sign_flipping_stratum_deltas"]), "with sign-flipping stratum deltas"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + render.CORR_CSS + EXTRA_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    xg = json.loads(XG_INPUT_PATH.read_text(encoding="utf-8")) if XG_INPUT_PATH.exists() else None
    OUTPUT_PATH.write_text(build_report(data, xg), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
