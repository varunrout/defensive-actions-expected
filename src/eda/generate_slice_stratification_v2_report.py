"""CLI entrypoint: render reports/analysis/shot_target/SLICE_STRATIFICATION_V2.json as a
self-contained HTML report -- archetype + boolean-flag slicers (prompt 28),
in the same shared design system as SLICE_STRATIFICATION.html (V1).

Usage:
    python -m src.eda.generate_slice_stratification_v2_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SLICE_STRATIFICATION_V2.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SLICE_STRATIFICATION_V2.html"


def _sparkline_svg(bins: list[dict]) -> str:
    values = [b["shot_rate_pct"] for b in bins if b.get("shot_rate_pct") is not None]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    w, h, pad = 100, 20, 2
    step = w / (len(values) - 1)
    points = " ".join(f"{i * step:.0f},{pad + (1 - (v - lo) / span) * (h - 2 * pad):.0f}" for i, v in enumerate(values))
    return f'<svg viewBox="0 0 {w} {h}" class="spark"><polyline points="{points}"/></svg>'


def _category_row(cat_label: str, cat: dict) -> str:
    row_class = ""
    caveat = ""
    if cat["structural_caution"]:
        row_class = " sr-structural"
        caveat = f' title="{esc(cat["structural_caution"])}"'
    elif cat["small_n"]:
        row_class = " sr-smalln"
    diverge_mark = "!" if cat["diverges_from_overall"] and not cat["small_n"] and not cat.get("structural_caution") else ""

    return (
        f'<tr class="sr{row_class}"{caveat}>'
        f"<td>{esc(cat_label)}</td>"
        f'<td class="sr-num">{cat["n_rows"]:,}</td>'
        f"<td>{esc(cat['shape'])}{diverge_mark}</td>"
        f"<td>{_sparkline_svg(cat['bins'])}</td>"
        f'<td class="sr-num">{esc(str(cat["rate_range_pp"]))}</td>'
        f"</tr>"
    )


def _cell_section(cell: dict) -> str:
    is_boolean = cell["slicer_type"] == "boolean"
    scoping_note = (
        ' <span class="nf-badge type">boolean slicer -- diverges = True vs False, not vs pooled overall</span>'
        if is_boolean else ""
    )
    cat_rows = "".join(_category_row(label, cat) for label, cat in cell["categories"].items())

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(cell['feature'])} &times; {esc(cell['slicer'])}{scoping_note}</h2>
  <p class="test-subnote">Overall shape (pooled, informational for boolean slicers): <b>{esc(cell['overall_shape'])}</b>
  (range {esc(str(cell['overall_rate_range_pp']))}pp) &mdash; {cell['n_categories_diverging']}/{cell['n_categories']}
  categories flagged diverging. "!" marks a genuine (non-small-n, non-structural) divergence.</p>
  <div class="qchart-card sr-wrap">
  <table class="sr-table">
  <thead><tr><th>category</th><th>n</th><th>shape</th><th>curve</th><th>range (pp)</th></tr></thead>
  <tbody>{cat_rows}</tbody>
  </table>
  </div>
</div>"""


def _summary_section(summary: dict) -> str:
    by_feature_rows = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(v['feature'])} <span style="font-weight:400;color:var(--text-muted);">({esc(v['dataset'])})</span></h5>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0 0 4px;">
  {'<b style="color:var(--pos);">needs interaction</b>' if v['needs_interaction_term_signal'] else '<b style="color:var(--good);">stable</b>'}
  ({len(v['diverging_slicers'])}/{v['n_slicers_tested']} slicers diverge)</p>
  <p style="font-size:11px;color:var(--text-muted);margin:0;">diverges under: {esc(', '.join(v['diverging_slicers']) or '(none)')}</p>
</div>"""
        for v in sorted(summary["by_feature"].values(), key=lambda v: (v["dataset"], v["feature"]))
    )

    divergence_rows = "".join(
        f"""
<tr>
  <td>{esc(d['dataset'])}</td><td><code>{esc(d['feature'])}</code></td><td><code>{esc(d['slicer'])}</code></td>
  <td>{esc(d['slicer_type'])}</td><td>{esc(d['category'])}</td><td>{esc(d['category_shape'])}</td><td>{esc(d['overall_shape'])}</td>
</tr>"""
        for d in summary["genuine_divergences"]
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">Cross-dataset summary (V2 -- kept separate from V1's counts)</h2>
  <div class="numsplit-grid" style="grid-template-columns:repeat(3,1fr);">{by_feature_rows}</div>

  <h3 style="margin-top:32px;">Genuine divergences ({summary['n_genuine_divergences']})</h3>
  <div class="qchart-card" style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
  <thead><tr style="text-align:left;color:var(--text-muted);">
    <th style="padding:6px 10px;">dataset</th><th style="padding:6px 10px;">feature</th><th style="padding:6px 10px;">slicer</th>
    <th style="padding:6px 10px;">slicer type</th><th style="padding:6px 10px;">category</th>
    <th style="padding:6px 10px;">category shape</th><th style="padding:6px 10px;">overall shape</th>
  </tr></thead>
  <tbody>{divergence_rows}</tbody>
  </table>
  </div>

  <div class="finding" style="margin-top:20px;">
  <span class="tag">closing comparison</span>
  <p>{esc(summary['closing_note_archetype_vs_boolean'])}</p>
  </div>
</div>"""


def build_report(data: dict) -> str:
    sections = [_cell_section(cell) for ds in data["results"].values() for slicers in ds.values() for cell in slicers.values()]
    summary = data["cross_dataset_summary"]

    v1_note_html = finding_card("V1 cross-reference", "why there are two files", data["v1_cross_reference"])

    body = f"""
<p class="test-subnote" style="margin-bottom:16px;">{esc(data['methodology'])}</p>
{v1_note_html}
{_summary_section(summary)}
{"".join(sections)}
"""

    n_cells = sum(len(s) for f in data["results"].values() for s in f.values())
    return render.render_article(
        eyebrow="SLICE STRATIFICATION V2",
        title="Archetype + Boolean-Flag Slicers",
        dek=(
            "V1 (SLICE_STRATIFICATION.html) tested every locked numerical feature against every locked categorical "
            "slicer. This file tests a disjoint slicer set never tried before: defender_archetype_name (the "
            "clustering label, passive only) and every locked boolean flag on both datasets, against the same "
            "feature pools -- same method, different slicers."
        ),
        stats=[
            (str(n_cells), "feature x slicer cells"),
            (str(summary["n_genuine_divergences"]), "genuine divergences"),
            (str(summary["n_features_with_at_least_one_diverging_slicer"]), "features needing an interaction term"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
.sr-wrap { overflow-x: auto; }
.sr-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.sr-table th { text-align: left; color: var(--text-muted); padding: 5px 8px; }
.sr-table td { padding: 5px 8px; }
.sr-table td.sr-num { text-align: right; }
tr.sr.sr-structural { background: var(--amber-wash); }
tr.sr.sr-smalln { opacity: 0.7; }
.spark { display: block; width: 90px; height: 20px; }
.spark polyline { fill: none; stroke: var(--neg); stroke-width: 1.5; }
""",
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
