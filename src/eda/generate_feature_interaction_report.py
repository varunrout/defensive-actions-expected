"""CLI entrypoint: render reports/analysis/shot_target/FEATURE_INTERACTION_ANALYSIS.json as a
self-contained HTML report, in the same shared design system as the other
EDA reports.

Usage:
    python -m src.eda.generate_feature_interaction_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "FEATURE_INTERACTION_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "FEATURE_INTERACTION_ANALYSIS.html"

CLASS_BADGE = {
    "interactive": ("v-yes", "INTERACTIVE -- genuine interaction-term candidate"),
    "additive": ("v-no", "ADDITIVE -- safe to model independently"),
    "substitutive": ("v-partially", "SUBSTITUTIVE -- mostly redundant with B"),
    "inconclusive": ("v-partially", "INCONCLUSIVE"),
}


def _heatmap(grid: list[dict], a_labels: list[str], b_labels: list[str]) -> str:
    rates = [c["rate"] for c in grid if c["rate"] is not None]
    lo, hi = (min(rates), max(rates)) if rates else (0, 1)
    span = (hi - lo) or 1.0

    def cell(a_label: str, b_label: str) -> str:
        c = next((c for c in grid if c["a_bin"] == a_label and c["b_bin"] == b_label), None)
        if c is None or c["rate"] is None:
            return '<td class="hm-cell hm-empty">n/a</td>'
        t = (c["rate"] - lo) / span
        bg = f"rgba(227,73,72,{0.12 + 0.55 * t:.2f})"
        return f'<td class="hm-cell" style="background:{bg};"><b>{c["rate"]:.2f}%</b><br><span class="hm-n">n={c["n"]:,}</span></td>'

    header = "".join(f"<th>{esc(a)}</th>" for a in a_labels)
    rows = "".join(
        f"<tr><th>{esc(b)}</th>{''.join(cell(a, b) for a in a_labels)}</tr>"
        for b in b_labels
    )
    return f"""
<table class="hm-table">
<thead><tr><th></th>{header}</tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _gradient_table(gradient: list[dict]) -> str:
    rows = "".join(
        f"""<tr>
  <td>{esc(g['b_bin'])}</td>
  <td>{esc(str(g['a_delta_within_stratum']))}pp</td>
  <td>{esc(str(g['a_bin_compared_low']))} &rarr; {esc(str(g['a_bin_compared_high']))}</td>
  <td>{g['n_rows']:,}</td>
</tr>"""
        for g in gradient
    )
    return f"""
<table class="sr-table">
<thead><tr><th>B stratum</th><th>A delta within stratum</th><th>A bins compared</th><th>n</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _pair_section(p: dict) -> str:
    cls_class, cls_label = CLASS_BADGE[p["classification"]]
    rationale_card = finding_card("football rationale", f"{p['feature_a']} x {p['feature_b']}", p["football_rationale"])

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(p['feature_a'])} &times; {esc(p['feature_b'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">({esc(p['dataset'])})</span></h2>
  {rationale_card}
  <div class="verdict-banner {cls_class}">
    <b>{esc(cls_label)}</b><br>{esc(p['classification_reason'])}
  </div>
  <p class="test-subnote">Pooled marginal delta for {esc(p['feature_a'])} (ignoring B): <b>{esc(str(p['marginal_a_delta_pp']))}pp</b>
  (n={p['n_rows_used']:,} rows used).</p>
  <h4>4&times;4 grid -- shot rate by ({esc(p['feature_a'])} column, {esc(p['feature_b'])} row)</h4>
  {_heatmap(p['grid'], p['a_labels'], p['b_labels'])}
  <h4 style="margin-top:20px;">{esc(p['feature_a'])}'s gradient within each {esc(p['feature_b'])} stratum</h4>
  {_gradient_table(p['a_gradient_by_b_stratum'])}
</div>"""


def _summary_table(rows: list[dict]) -> str:
    body = "".join(
        f"""<tr>
  <td>{esc(r['dataset'])}</td><td><code>{esc(r['feature_a'])}</code></td><td><code>{esc(r['feature_b'])}</code></td>
  <td><span class="verdict-banner {CLASS_BADGE[r['classification']][0]}" style="display:inline-block;padding:2px 10px;margin:0;font-size:11px;">{esc(r['classification'])}</span></td>
</tr>"""
        for r in rows
    )
    return f"""
<div class="qchart-card" style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="text-align:left;color:var(--text-muted);">
  <th style="padding:6px 10px;">dataset</th><th style="padding:6px 10px;">feature A</th>
  <th style="padding:6px 10px;">feature B</th><th style="padding:6px 10px;">classification</th>
</tr></thead>
<tbody>{body}</tbody>
</table>
</div>"""


def build_report(data: dict) -> str:
    thresholds_card = finding_card(
        "classification thresholds",
        "stated explicitly",
        f"{esc(data['thresholds']['note'])} substitutive_ratio_max={data['thresholds']['substitutive_ratio_max']}, "
        f"additive_magnitude_ratio_max={data['thresholds']['additive_magnitude_ratio_max']}, "
        f"min_marginal_delta_pp={data['thresholds']['min_marginal_delta_pp']}, "
        f"min_cell_n_for_delta={data['thresholds']['min_cell_n_for_delta']}.",
    )
    selection_card = finding_card("candidate selection", "how the 10 pairs were chosen", data["candidate_selection"])

    n_by_class = {}
    for r in data["summary_table"]:
        n_by_class[r["classification"]] = n_by_class.get(r["classification"], 0) + 1

    body = f"""
{selection_card}
{thresholds_card}
<h2 class="test-title" style="margin-top:32px;">Summary -- interactive pairs first</h2>
{_summary_table(data['summary_table'])}
{"".join(_pair_section(p) for p in data['pairs'])}
"""

    return render.render_article(
        eyebrow="FEATURE INTERACTION ANALYSIS",
        title="Numeric x Numeric Interaction Analysis",
        dek=(
            "Everything tested so far pairs one numerical feature against one categorical slicer, or a "
            "numeric-vs-numeric confound with 1D quartile stratification. This is the first 2D joint analysis: "
            "the 10 least-stable features (by prompt 27's redundancy-collapsed slice-divergence ranking) each "
            "paired with another locked feature on football-domain grounds, binned into a 4x4 quartile grid."
        ),
        stats=[
            (str(len(data["pairs"])), "pairs tested"),
            (str(n_by_class.get("interactive", 0)), "interactive (interaction-term candidates)"),
            (str(n_by_class.get("substitutive", 0)), "substitutive (redundancy footnote)"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
.hm-table { border-collapse: collapse; width: 100%; margin: 12px 0; }
.hm-table th { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); padding: 6px 8px; text-align: center; }
.hm-cell { text-align: center; padding: 10px 8px; border: 1px solid var(--border); font-size: 12.5px; border-radius: 4px; }
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
