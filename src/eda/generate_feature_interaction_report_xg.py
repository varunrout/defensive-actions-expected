"""CLI entrypoint: render reports/eda_xg/FEATURE_INTERACTION_ANALYSIS.json
as a self-contained HTML report -- the xG counterpart to
FEATURE_INTERACTION_ANALYSIS.html.

Usage:
    python -m src.eda.generate_feature_interaction_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "FEATURE_INTERACTION_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "FEATURE_INTERACTION_ANALYSIS.html"

CLASS_BADGE = {
    "interactive": ("v-yes", "INTERACTIVE"),
    "additive": ("v-no", "ADDITIVE"),
    "substitutive": ("v-partially", "SUBSTITUTIVE"),
    "inconclusive": ("v-partially", "INCONCLUSIVE"),
}


def _heatmap(grid: list[dict], a_labels: list[str], b_labels: list[str], value_key: str) -> str:
    values = [c[value_key] for c in grid if c[value_key] is not None]
    lo, hi = (min(values), max(values)) if values else (0, 1)
    span = (hi - lo) or 1.0
    n_key = "n" if value_key == "mean_xg" else "n_given_shot"

    def cell(a_label: str, b_label: str) -> str:
        c = next((c for c in grid if c["a_bin"] == a_label and c["b_bin"] == b_label), None)
        if c is None or c[value_key] is None:
            return '<td class="hm-cell hm-empty">n/a</td>'
        t = (c[value_key] - lo) / span
        bg = f"rgba(227,73,72,{0.12 + 0.55 * t:.2f})"
        return f'<td class="hm-cell" style="background:{bg};"><b>{c[value_key]:.5f}</b><br><span class="hm-n">n={c[n_key]:,}</span></td>'

    header = "".join(f"<th>{esc(a)}</th>" for a in a_labels)
    rows = "".join(f"<tr><th>{esc(b)}</th>{''.join(cell(a, b) for a in a_labels)}</tr>" for b in b_labels)
    return f'<table class="hm-table"><thead><tr><th></th>{header}</tr></thead><tbody>{rows}</tbody></table>'


def _gradient_table(gradient: list[dict]) -> str:
    rows = "".join(
        f"""<tr>
  <td>{esc(g['b_bin'])}</td>
  <td>{esc(str(g['a_delta_within_stratum']))}</td>
  <td>{esc(str(g['a_bin_compared_low']))} &rarr; {esc(str(g['a_bin_compared_high']))}</td>
  <td>{g['n_rows']:,}</td>
</tr>"""
        for g in gradient
    )
    return f'<table class="sr-table"><thead><tr><th>B stratum</th><th>A delta within stratum</th><th>A bins compared</th><th>n</th></tr></thead><tbody>{rows}</tbody></table>'


def _pair_section(p: dict) -> str:
    cls_class, cls_label = CLASS_BADGE[p["classification"]]
    cls_class_gs, cls_label_gs = CLASS_BADGE[p["classification_given_shot"]]
    rationale_card = finding_card("football rationale", f"{p['feature_a']} x {p['feature_b']}", p["football_rationale"])

    agree_note = ""
    if p["classification"] != p["classification_given_shot"]:
        agree_note = f"""
<div class="finding flag" style="margin:12px 0;">
<span class="tag">unconditional and given-shot disagree</span>
<p>Unconditional: <b>{esc(p['classification'])}</b>. Given-shot: <b>{esc(p['classification_given_shot'])}</b>. Don't
assume the unconditional conclusion carries over to chance quality.</p>
</div>"""

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(p['feature_a'])} &times; {esc(p['feature_b'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">({esc(p['dataset'])}, xG)</span></h2>
  {rationale_card}
  <div class="verdict-banner {cls_class}">
    <b>Unconditional: {esc(cls_label)}</b><br>{esc(p['classification_reason'])}
  </div>
  <div class="verdict-banner {cls_class_gs}">
    <b>Given-shot: {esc(cls_label_gs)}</b><br>{esc(p['classification_reason_given_shot'])}
  </div>
  {agree_note}
  <p class="test-subnote">n={p['n_rows_used']:,} rows unconditional, {p['n_rows_used_given_shot']:,} given-shot.</p>
  <h4>Unconditional 4&times;4 grid (mean xG)</h4>
  {_heatmap(p['grid'], p['a_labels'], p['b_labels'], 'mean_xg')}
  <h4 style="margin-top:16px;">Given-shot 4&times;4 grid (mean xG, target_future_shot_10s==1 only)</h4>
  {_heatmap(p['grid'], p['a_labels'], p['b_labels'], 'mean_xg_given_shot')}
  <h4 style="margin-top:20px;">{esc(p['feature_a'])}'s unconditional gradient by {esc(p['feature_b'])} stratum</h4>
  {_gradient_table(p['a_gradient_by_b_stratum'])}
  <h4>{esc(p['feature_a'])}'s given-shot gradient by {esc(p['feature_b'])} stratum</h4>
  {_gradient_table(p['a_gradient_by_b_stratum_given_shot'])}
</div>"""


def _summary_table(rows: list[dict]) -> str:
    body = "".join(
        f"""<tr>
  <td>{esc(r['dataset'])}</td><td><code>{esc(r['feature_a'])}</code></td><td><code>{esc(r['feature_b'])}</code></td>
  <td><span class="verdict-banner {CLASS_BADGE[r['classification']][0]}" style="display:inline-block;padding:2px 10px;margin:0;font-size:11px;">{esc(r['classification'])}</span></td>
  <td><span class="verdict-banner {CLASS_BADGE[r['classification_given_shot']][0]}" style="display:inline-block;padding:2px 10px;margin:0;font-size:11px;">{esc(r['classification_given_shot'])}</span></td>
</tr>"""
        for r in rows
    )
    return f"""
<div class="qchart-card" style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="text-align:left;color:var(--text-muted);">
  <th style="padding:6px 10px;">dataset</th><th style="padding:6px 10px;">feature A</th>
  <th style="padding:6px 10px;">feature B</th><th style="padding:6px 10px;">unconditional</th>
  <th style="padding:6px 10px;">given-shot</th>
</tr></thead>
<tbody>{body}</tbody>
</table>
</div>"""


def build_report(data: dict) -> str:
    thresholds_card = finding_card("classification thresholds", "stated explicitly", esc(data["thresholds"]["note"]))
    selection_card = finding_card("candidate selection", "same 10 pairs as the binary-target file", data["candidate_selection"])

    n_by_class = {}
    for r in data["summary_table"]:
        n_by_class[r["classification"]] = n_by_class.get(r["classification"], 0) + 1

    body = f"""
{selection_card}
{thresholds_card}
<h2 class="test-title" style="margin-top:32px;">Summary -- interactive pairs first (unconditional)</h2>
{_summary_table(data['summary_table'])}
{"".join(_pair_section(p) for p in data['pairs'])}
"""

    return render.render_article(
        eyebrow="FEATURE INTERACTION ANALYSIS -- XG",
        title="Numeric x Numeric Interaction Analysis (xG)",
        dek=(
            "The continuous-target counterpart to FEATURE_INTERACTION_ANALYSIS.html -- same 10 pairs, same 4x4 "
            "grid method, with an unconditional and a shot-conditional classification for each (xg is "
            "structurally zero wherever no shot occurs, so the two can genuinely disagree)."
        ),
        stats=[
            (str(len(data["pairs"])), "pairs tested"),
            (str(n_by_class.get("interactive", 0)), "interactive unconditionally"),
            (str(data["n_pairs_where_unconditional_and_given_shot_agree"]), "pairs where both panels agree"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
.hm-table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.hm-table th { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); padding: 6px 8px; text-align: center; }
.hm-cell { text-align: center; padding: 10px 8px; border: 1px solid var(--border); font-size: 12px; border-radius: 4px; }
.hm-cell.hm-empty { background: var(--plane); color: var(--text-muted); }
.hm-n { font-size: 10px; color: var(--text-muted); }
.sr-table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0 20px; }
.sr-table th { text-align: left; color: var(--text-muted); padding: 5px 8px; }
.sr-table td { padding: 5px 8px; border-top: 1px solid var(--border); }
""",
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
