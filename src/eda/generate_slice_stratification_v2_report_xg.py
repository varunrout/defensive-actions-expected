"""CLI entrypoint: render reports/analysis/xg_target/SLICE_STRATIFICATION_V2.json as a
self-contained HTML report -- the xG counterpart to SLICE_STRATIFICATION_V2.html.

Usage:
    python -m src.eda.generate_slice_stratification_v2_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "SLICE_STRATIFICATION_V2.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "SLICE_STRATIFICATION_V2.html"

OCCURRENCE_QUALITY_CLASS = {"occurrence-only": "v-yes", "occurrence+quality": "v-no", "inconclusive": "v-partially"}


def _sparkline_svg(bins: list[dict]) -> str:
    values = [b["mean_xg"] for b in bins if b.get("mean_xg") is not None]
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
    oq = cat["occurrence_vs_quality"]
    oq_badge = f'<span class="oq-badge {OCCURRENCE_QUALITY_CLASS[oq]}">{esc(oq)}</span>'
    diverge_gs_mark = "!" if cat["diverges_from_overall_given_shot"] and oq != "inconclusive" else ""

    return (
        f'<tr class="sr{row_class}"{caveat}>'
        f"<td>{esc(cat_label)}</td>"
        f'<td class="sr-num">{cat["n_rows"]:,}</td>'
        f"<td>{esc(cat['shape'])}{diverge_mark}</td>"
        f"<td>{_sparkline_svg(cat['bins'])}</td>"
        f'<td class="sr-num">{cat["n_given_shot"]:,}</td>'
        f"<td>{esc(cat['shape_given_shot'])}{diverge_gs_mark}</td>"
        f"<td>{oq_badge}</td>"
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
  <h2 class="test-title">{esc(cell['feature'])} &times; {esc(cell['slicer'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">(xG)</span>{scoping_note}</h2>
  <p class="test-subnote">Overall shape: <b>{esc(cell['overall_shape'])}</b> (mean-xG range {esc(str(cell['overall_mean_xg_range']))})
  &mdash; {cell['n_categories_diverging']}/{cell['n_categories']} categories flagged diverging. Given-shot overall shape:
  <b>{esc(cell['overall_shape_given_shot'])}</b>.</p>
  <div class="qchart-card sr-wrap">
  <table class="sr-table">
  <thead><tr>
    <th>category</th><th>n</th><th>shape</th><th>curve</th><th>n (given shot)</th>
    <th>shape (given shot)</th><th>occurrence vs quality</th>
  </tr></thead>
  <tbody>{cat_rows}</tbody>
  </table>
  </div>
</div>"""


def _summary_section(summary: dict) -> str:
    by_feature_rows = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(v['feature'])} <span style="font-weight:400;color:var(--text-muted);">({esc(v['dataset'])})</span></h5>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0 0 2px;">unconditional:
  {'<b style="color:var(--pos);">needs interaction</b>' if v['needs_interaction_term_signal'] else '<b style="color:var(--good);">stable</b>'}
  ({len(v['diverging_slicers'])}/{v['n_slicers_tested']})</p>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0;">given-shot:
  {'<b style="color:var(--pos);">needs interaction</b>' if v['needs_interaction_term_signal_given_shot'] else '<b style="color:var(--good);">stable</b>'}
  ({len(v['diverging_slicers_given_shot'])}/{v['n_slicers_tested']})</p>
</div>"""
        for v in sorted(summary["by_feature"].values(), key=lambda v: (v["dataset"], v["feature"]))
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">Cross-dataset summary (V2 -- kept separate from V1's counts)</h2>
  <div class="numsplit-grid" style="grid-template-columns:repeat(3,1fr);">{by_feature_rows}</div>
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
        eyebrow="SLICE STRATIFICATION V2 -- XG",
        title="Archetype + Boolean-Flag Slicers (xG)",
        dek=(
            "The continuous-target counterpart to SLICE_STRATIFICATION_V2.html -- same disjoint slicer set "
            "(defender_archetype_name + every locked boolean flag), with shot-conditional fields built in from "
            "the start rather than a later retrofit."
        ),
        stats=[
            (str(n_cells), "feature x slicer cells"),
            (str(summary["n_genuine_divergences_given_shot"]), "given-shot divergences"),
            (str(summary["n_features_with_at_least_one_diverging_slicer_given_shot"]), "features needing an interaction term (given-shot)"),
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
.oq-badge { display: inline-block; padding: 1px 6px; font-size: 10px; border-radius: 4px; white-space: nowrap; }
.oq-badge.v-yes { background: var(--good-wash); color: var(--good); }
.oq-badge.v-no { background: var(--amber-wash); color: var(--amber); }
.oq-badge.v-partially { background: rgba(42,120,214,0.10); color: var(--accent); }
""",
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
