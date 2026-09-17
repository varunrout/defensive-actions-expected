"""CLI entrypoint: render reports/eda/SLICE_STRATIFICATION.json as a
self-contained HTML report, in the same shared design system as the other
EDA reports.

Usage:
    python -m src.eda.generate_slice_stratification_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "SLICE_STRATIFICATION.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "SLICE_STRATIFICATION.html"


def _sparkline_svg(bins: list[dict]) -> str:
    """Compact inline-SVG sparkline (one <polyline>, not one element per
    bin, integer coords, styling factored into a shared CSS class rather
    than repeated inline) -- cheap enough to put on all ~950 category
    cards. The preview browser this report is checked in has a hard
    ~512KB file-size cutoff (confirmed empirically); a per-bin bar chart
    per category blew past that at ~2MB, and even a naive per-point <span>
    sparkline landed at ~600KB. Full bin tables still live in the JSON for
    anyone who needs exact values."""
    rates = [b["shot_rate_pct"] for b in bins if b.get("shot_rate_pct") is not None]
    if len(rates) < 2:
        return ""
    lo, hi = min(rates), max(rates)
    span = (hi - lo) or 1.0
    w, h, pad = 100, 24, 2
    step = w / (len(rates) - 1)
    points = " ".join(
        f"{i * step:.0f},{pad + (1 - (r - lo) / span) * (h - 2 * pad):.0f}"
        for i, r in enumerate(rates)
    )
    return f'<svg viewBox="0 0 {w} {h}" class="spark"><polyline points="{points}"/></svg>'


def _category_card(cat_label: str, cat: dict) -> str:
    """Shape + range + a compact sparkline -- full per-bin bar charts (one
    per category, like prompt 24's report) don't scale to a 118-cell x
    up-to-23-category report (that rendering alone produced a ~2MB HTML
    page); an inline-SVG sparkline gives the same shape-at-a-glance for a
    fraction of the markup."""
    classes = ["numsplit-card"]
    caveat = ""
    if cat["structural_caution"]:
        classes.append("cat-structural")
        caveat = f'<p class="cat-caveat">{esc(cat["structural_caution"])}</p>'
    elif cat["small_n"]:
        classes.append("cat-smalln")
        caveat = '<p class="cat-caveat">Small n (&lt; threshold) -- treat this category\'s shape with caution.</p>'

    diverge_badge = (
        '<span class="nf-badge unreliable" style="margin-left:8px;">DIVERGES</span>'
        if cat["diverges_from_overall"] and not cat["small_n"] and not cat["structural_caution"]
        else ""
    )

    return f"""
<div class="{' '.join(classes)}">
  <h5>{esc(cat_label)} (n={cat['n_rows']:,}){diverge_badge}</h5>
  <p style="font-size:12.5px;color:var(--text-secondary);margin:0 0 8px;">shape: <b>{esc(cat['shape'])}</b>
  &mdash; range {esc(str(cat['rate_range_pp']))}pp</p>
  {caveat}
  {_sparkline_svg(cat['bins'])}
</div>"""


def _cell_section(cell: dict) -> str:
    driven = cell["driven_by_single_category"]
    driven_html = ""
    if driven:
        driven_html = finding_card(
            "driven by one category",
            f"{cell['feature']} x {cell['slicer']}",
            f"Only <b>{esc(driven)}</b> reproduces the overall pattern ({esc(cell['overall_shape'])}) among the "
            "non-small-n, non-structural categories -- the rest diverge. Don't present the overall shape as if it "
            "held broadly.",
            flag=True,
        )

    cat_cards = "".join(_category_card(label, cat) for label, cat in cell["categories"].items())

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(cell['feature'])} &times; {esc(cell['slicer'])}</h2>
  <p class="test-subnote">Overall shape: <b>{esc(cell['overall_shape'])}</b> (range {esc(str(cell['overall_rate_range_pp']))}pp)
  &mdash; {cell['n_categories_diverging']}/{cell['n_categories']} categories diverge from it.</p>
  {driven_html}
  <div class="numsplit-grid slice-grid">{cat_cards}</div>
</div>"""


def _summary_section(summary: dict) -> str:
    by_feature_rows = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(v['feature'])} <span style="font-weight:400;color:var(--text-muted);">({esc(v['dataset'])})</span></h5>
  <p style="font-size:12px;color:var(--text-secondary);margin:0 0 4px;">
  {'<b style="color:var(--pos);">needs interaction term signal</b>' if v['needs_interaction_term_signal'] else '<b style="color:var(--good);">fully stable</b>'}
  &mdash; {len(v['diverging_slicers'])}/{v['n_slicers_tested']} slicers diverge</p>
  <p style="font-size:11px;color:var(--text-muted);margin:0;">diverges under: {esc(', '.join(v['diverging_slicers']) or '(none)')}</p>
  <p style="font-size:11px;color:var(--text-muted);margin:2px 0 0;">stable under: {esc(', '.join(v['stable_slicers']) or '(none)')}</p>
</div>"""
        for v in sorted(summary["by_feature"].values(), key=lambda v: (v["dataset"], v["feature"]))
    )

    divergence_rows = "".join(
        f"""
<tr>
  <td>{esc(d['dataset'])}</td><td><code>{esc(d['feature'])}</code></td><td><code>{esc(d['slicer'])}</code></td>
  <td>{esc(d['category'])}</td><td>{esc(d['category_shape'])}</td><td>{esc(d['overall_shape'])}</td>
</tr>"""
        for d in summary["genuine_divergences"]
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">Cross-dataset summary</h2>
  <p class="test-subnote">Per feature, which slicers it diverges under vs. is stable under -- the practical
  "does this feature need an interaction term" signal for the modelling stage.</p>
  <div class="numsplit-grid" style="grid-template-columns:repeat(3,1fr);">{by_feature_rows}</div>

  <h3 style="margin-top:32px;">Genuine divergences ({summary['n_genuine_divergences']})</h3>
  <p class="test-subnote">Every (feature, slicer, category) triple flagged <code>diverges_from_overall: true</code>
  with <code>small_n</code> and <code>structural_caution</code> both false -- filtered of noise, the list a human
  should actually read.</p>
  <div class="qchart-card" style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:12px;">
  <thead><tr style="text-align:left;color:var(--text-muted);">
    <th style="padding:6px 10px;">dataset</th><th style="padding:6px 10px;">feature</th><th style="padding:6px 10px;">slicer</th>
    <th style="padding:6px 10px;">category</th><th style="padding:6px 10px;">category shape</th><th style="padding:6px 10px;">overall shape</th>
  </tr></thead>
  <tbody>{divergence_rows}</tbody>
  </table>
  </div>
</div>"""


def build_report(data: dict) -> str:
    all_cells = [cell for ds in data["results"].values() for cell in ds.values()]
    flat_cells = [cell for feature_slicers in all_cells for cell in feature_slicers.values()]
    n_driven = sum(1 for c in flat_cells if c["driven_by_single_category"])
    summary = data.get("cross_dataset_summary")

    sections = []
    for dataset_key, features in data["results"].items():
        for feature, slicers in features.items():
            for slicer, cell in slicers.items():
                sections.append(_cell_section(cell))

    summary_html = _summary_section(summary) if summary else ""
    extension_note_html = (
        finding_card("prompt 25 extension", "104 new cells + corrections", data["prompt_25_extension_note"])
        if data.get("prompt_25_extension_note") else ""
    )

    body = f"""
<p class="test-subnote" style="margin-bottom:24px;">{esc(data['methodology'])}</p>
{extension_note_html}
{summary_html}
{"".join(sections)}
"""

    return render.render_article(
        eyebrow="SLICE STRATIFICATION",
        title="Categorical Stratification of Established Patterns",
        dek=(
            "For features with an already-established shape classification, does the shape hold inside every "
            "category of a categorical column, or is it driven by one slice of the data? Full explicit "
            "feature x slicer cross-product (prompt 25) on top of prompt 24's original 7 features -- "
            f"{len(flat_cells)} cells total."
        ),
        stats=[
            (str(len(flat_cells)), "feature x slicer cells"),
            (str(summary["n_genuine_divergences"]) if summary else "0", "genuine divergences"),
            (str(n_driven), "cells driven by one category"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
.cat-structural { border-left: 4px solid var(--amber); background: var(--amber-wash); }
.cat-smalln { border-left: 4px solid var(--text-muted); opacity: 0.85; }
.cat-caveat { font-size: 11.5px; color: var(--amber); margin: 0 0 8px; }
.slice-grid { grid-template-columns: repeat(3, 1fr); }
@media (max-width: 900px) { .slice-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 600px) { .slice-grid { grid-template-columns: 1fr; } }
.spark { display: block; width: 90px; height: 22px; margin-top: 6px; }
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
