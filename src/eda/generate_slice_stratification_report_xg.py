"""CLI entrypoint: render reports/eda_xg/SLICE_STRATIFICATION.json as a
self-contained HTML report -- the xG counterpart to SLICE_STRATIFICATION.html.

Usage:
    python -m src.eda.generate_slice_stratification_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "SLICE_STRATIFICATION.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "SLICE_STRATIFICATION.html"


OCCURRENCE_QUALITY_CLASS = {
    "occurrence-only": "v-yes",
    "occurrence+quality": "v-no",
    "inconclusive": "v-partially",
}


def _sparkline_svg(bins: list[dict], color: str = "var(--neg)") -> str:
    """Compact inline-SVG sparkline (one <polyline>) -- a per-bin bar chart
    on ~920 categories x 2 (unconditional + given-shot) blew well past the
    ~512KB file-size cutoff this report is checked against (confirmed
    empirically on the binary-target report). Full bin tables live in the
    JSON."""
    values = [b["mean_xg"] for b in bins if b.get("mean_xg") is not None]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    w, h, pad = 100, 20, 2
    step = w / (len(values) - 1)
    points = " ".join(
        f"{i * step:.0f},{pad + (1 - (v - lo) / span) * (h - 2 * pad):.0f}"
        for i, v in enumerate(values)
    )
    style = f' style="stroke:{color};"' if color != "var(--neg)" else ""
    return f'<svg viewBox="0 0 {w} {h}" class="spark"><polyline points="{points}"{style}/></svg>'


def _category_row(cat_label: str, cat: dict) -> str:
    """One <tr> per category -- a card-per-category layout (like the
    binary-target report) roughly doubled in cost here because every
    category now also carries a shot-conditional panel, landing at ~800KB+
    and blowing the ~512KB cutoff this report is checked against. A table
    row with one sparkline column keeps both the unconditional and
    given-shot numbers visible per category at a fraction of the markup."""
    row_class = ""
    caveat = ""
    if cat["structural_caution"]:
        row_class = " sr-structural"
        caveat = f' title="{esc(cat["structural_caution"])}"'
    elif cat["small_n"]:
        row_class = " sr-smalln"

    diverge_mark = "!" if cat["diverges_from_overall"] and not cat["small_n"] and not cat["structural_caution"] else ""
    oq = cat["occurrence_vs_quality"]
    oq_badge = f'<span class="oq-badge {OCCURRENCE_QUALITY_CLASS[oq]}">{esc(oq)}</span>'
    diverge_given_shot_mark = "!" if cat["diverges_from_overall_given_shot"] and oq != "inconclusive" else ""

    return (
        f'<tr class="sr{row_class}"{caveat}>'
        f"<td>{esc(cat_label)}</td>"
        f'<td class="sr-num">{cat["n_rows"]:,}</td>'
        f"<td>{esc(cat['shape'])}{diverge_mark}</td>"
        f"<td>{_sparkline_svg(cat['bins'])}</td>"
        f'<td class="sr-num">{cat["n_given_shot"]:,}</td>'
        f"<td>{esc(cat['shape_given_shot'])}{diverge_given_shot_mark}</td>"
        f"<td>{oq_badge}</td>"
        f"</tr>"
    )


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

    cat_rows = "".join(_category_row(label, cat) for label, cat in cell["categories"].items())

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(cell['feature'])} &times; {esc(cell['slicer'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">(xG)</span></h2>
  <p class="test-subnote">Overall shape: <b>{esc(cell['overall_shape'])}</b> (mean-xG range {esc(str(cell['overall_mean_xg_range']))})
  &mdash; {cell['n_categories_diverging']}/{cell['n_categories']} categories diverge from it. Given-shot overall shape:
  <b>{esc(cell.get('overall_shape_given_shot', 'n/a'))}</b>. "!" marks a genuine (non-small-n, non-structural) divergence.</p>
  {driven_html}
  <div class="qchart-card sr-wrap">
  <table class="sr-table">
  <thead><tr>
    <th>category</th><th>n</th><th>shape</th>
    <th>curve</th><th>n (given shot)</th>
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
  ({len(v['diverging_slicers'])}/{v['n_slicers_tested']} diverge)</p>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0 0 6px;">given-shot:
  {'<b style="color:var(--pos);">needs interaction</b>' if v['needs_interaction_term_signal_given_shot'] else '<b style="color:var(--good);">stable</b>'}
  ({len(v['diverging_slicers_given_shot'])}/{v['n_slicers_tested']} diverge)</p>
</div>"""
        for v in sorted(summary["by_feature"].values(), key=lambda v: (v["dataset"], v["feature"]))
    )

    agreement_rows = "".join(
        f"""
<tr>
  <td>{esc(r['dataset'])}</td><td><code>{esc(r['feature'])}</code></td><td><code>{esc(r['slicer'])}</code></td>
  <td>{esc(r['category'])}</td><td>{r['diverges_unconditional']}</td><td>{r['diverges_given_shot']}</td>
  <td>{esc(r['note'])}</td>
</tr>"""
        for r in summary["conditioning_agreement"] if not r["agrees"]
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">Cross-dataset, cross-conditioning summary</h2>
  <p class="test-subnote">Per feature, unconditional vs. given-shot slicer stability -- the same "does this need an
  interaction term" signal as the binary report, split by whether it's about shot occurrence or shot quality.</p>
  <div class="numsplit-grid" style="grid-template-columns:repeat(3,1fr);">{by_feature_rows}</div>

  <h3 style="margin-top:32px;">Where conditioning changes the conclusion ({summary['n_conditioning_disagreements']})</h3>
  <p class="test-subnote">Categories where "diverges from overall" flips once you condition on a shot happening --
  either an unconditional divergence turns out to be occurrence-only (flattens once conditioned), or a category
  that looked stable unconditionally shows a real quality-level divergence once conditioned.</p>
  <div class="qchart-card" style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:11.5px;">
  <thead><tr style="text-align:left;color:var(--text-muted);">
    <th style="padding:6px 10px;">dataset</th><th style="padding:6px 10px;">feature</th><th style="padding:6px 10px;">slicer</th>
    <th style="padding:6px 10px;">category</th><th style="padding:6px 10px;">diverges unconditional</th>
    <th style="padding:6px 10px;">diverges given-shot</th><th style="padding:6px 10px;">note</th>
  </tr></thead>
  <tbody>{agreement_rows}</tbody>
  </table>
  </div>
</div>"""


def build_report(data: dict) -> str:
    flat_cells = [cell for ds in data["results"].values() for slicers in ds.values() for cell in slicers.values()]
    n_driven = sum(1 for c in flat_cells if c["driven_by_single_category"])
    summary = data.get("cross_dataset_summary")

    sections = []
    for dataset_key, features in data["results"].items():
        for feature, slicers in features.items():
            for slicer, cell in slicers.items():
                sections.append(_cell_section(cell))

    summary_html = _summary_section(summary) if summary else ""
    extension_note_html = (
        finding_card("prompt 26 extension", "104 new cells + shot-conditional retrofit", data["prompt_26_extension_note"])
        if data.get("prompt_26_extension_note") else ""
    )

    body = f"""
<p class="test-subnote" style="margin-bottom:24px;">{esc(data['methodology'])}</p>
{extension_note_html}
{summary_html}
{"".join(sections)}
"""

    n_cells_label = str(len(flat_cells))
    return render.render_article(
        eyebrow="SLICE STRATIFICATION -- XG",
        title="Categorical Stratification of Established Patterns (xG)",
        dek=(
            "The continuous-target counterpart to SLICE_STRATIFICATION.html -- full feature x slicer cross-product "
            f"({n_cells_label} cells), with every category carrying both an unconditional mean-xG curve and a "
            "shot-conditional one (target_future_shot_10s == 1 only) -- xg is structurally zero wherever no shot "
            "happens, so the unconditional curve mostly re-derives occurrence; the shot-conditional curve is where "
            "genuine chance-quality signal shows up."
        ),
        stats=[
            (n_cells_label, "feature x slicer cells"),
            (str(summary["n_genuine_divergences_given_shot"]) if summary else "0", "given-shot divergences"),
            (str(summary["n_conditioning_disagreements"]) if summary else "0", "conditioning flips"),
            (str(n_driven), "cells driven by one category"),
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
