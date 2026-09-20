"""CLI entrypoint: render reports/analysis/xt_target/SLICE_STRATIFICATION.json
(V1) and SLICE_STRATIFICATION_V2.json as self-contained HTML reports -- the
xT sibling of generate_slice_stratification_report_xg.py and
generate_slice_stratification_v2_report_xg.py.

Both xg renderers share essentially the same body (one table row per
category, one inline-SVG sparkline per row, for the same file-size reason
their docstrings give: a card-per-category layout blew past the ~512KB
cutoff). They are kept as one module here since the two reports differ only
in their header text and summary block.

Sparklines are drawn with a zero baseline, which the xg version has no need
for: its values are non-negative, so min-max scaling alone conveys the
shape. Here a curve entirely below zero and one entirely above it would
look identical without the baseline.

Usage:
    python -m src.eda.generate_slice_stratification_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc, finding_card

V1_IN = xc.OUT_DIR / "SLICE_STRATIFICATION.json"
V1_OUT = xc.OUT_DIR / "SLICE_STRATIFICATION.html"
V2_IN = xc.OUT_DIR / "SLICE_STRATIFICATION_V2.json"
V2_OUT = xc.OUT_DIR / "SLICE_STRATIFICATION_V2.html"

ZMM_CLASS = {"zero-mass-driven": "v-yes", "magnitude-driven": "v-no", "inconclusive": "v-partially"}

EXTRA_CSS = """
.sr-wrap { overflow-x: auto; }
.sr-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.sr-table th { text-align: left; color: var(--text-muted); padding: 5px 8px; }
.sr-table td { padding: 5px 8px; }
.sr-table td.sr-num { text-align: right; }
tr.sr.sr-structural { background: var(--amber-wash); }
tr.sr.sr-smalln { opacity: 0.7; }
.spark { display: block; width: 96px; height: 22px; }
.spark polyline { fill: none; stroke: var(--neg); stroke-width: 1.5; }
.spark line.base { stroke: var(--text-muted); stroke-width: 0.6; stroke-dasharray: 2 2; }
.zmm-badge { display: inline-block; padding: 1px 6px; font-size: 10px; border-radius: 4px; white-space: nowrap; }
.zmm-badge.v-yes { background: var(--amber-wash); color: var(--amber); }
.zmm-badge.v-no { background: var(--good-wash); color: var(--good); }
.zmm-badge.v-partially { background: rgba(42,120,214,0.10); color: var(--accent); }
.xc-badge { display: inline-block; padding: 1px 6px; font-size: 10px; border-radius: 4px;
  background: rgba(42,120,214,0.10); color: var(--accent); }
"""


def _sparkline(bins: list[dict]) -> str:
    values = [b["mean_xt"] for b in bins if b.get("mean_xt") is not None]
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    lo, hi = min(lo, 0.0), max(hi, 0.0)   # always include zero, so sign is visible
    span = (hi - lo) or 1.0
    w, h, pad = 100, 22, 2
    step = w / (len(values) - 1)
    y0 = pad + (1 - (0.0 - lo) / span) * (h - 2 * pad)
    pts = " ".join(f"{i * step:.0f},{pad + (1 - (v - lo) / span) * (h - 2 * pad):.1f}" for i, v in enumerate(values))
    return (f'<svg viewBox="0 0 {w} {h}" class="spark">'
            f'<line class="base" x1="0" y1="{y0:.1f}" x2="{w}" y2="{y0:.1f}"/>'
            f'<polyline points="{pts}"/></svg>')


def _category_row(label: str, cat: dict) -> str:
    row_class, caveat = "", ""
    if cat["structural_caution"]:
        row_class, caveat = " sr-structural", f' title="{esc(cat["structural_caution"])}"'
    elif cat["small_n"]:
        row_class = " sr-smalln"

    div = "!" if cat["diverges_from_overall"] and not cat["small_n"] and not cat["structural_caution"] else ""
    zmm = cat["zero_mass_vs_magnitude"]
    div_nz = "!" if cat["diverges_from_overall_given_nonzero_delta"] and zmm != "inconclusive" else ""
    cross = '<span class="xc-badge">crosses 0</span>' if cat.get("curve_crosses_zero") else ""

    return (
        f'<tr class="sr{row_class}"{caveat}>'
        f"<td>{esc(label)}</td>"
        f'<td class="sr-num">{cat["n_rows"]:,}</td>'
        f"<td>{esc(cat['shape'])}{div} {cross}</td>"
        f"<td>{_sparkline(cat['bins'])}</td>"
        f'<td class="sr-num">{cat["n_given_nonzero_delta"]:,}</td>'
        f"<td>{esc(cat['shape_given_nonzero_delta'])}{div_nz}</td>"
        f'<td><span class="zmm-badge {ZMM_CLASS[zmm]}">{esc(zmm)}</span></td>'
        f"</tr>"
    )


def _cell_section(cell: dict) -> str:
    driven_html = ""
    if cell.get("driven_by_single_category"):
        driven_html = finding_card(
            "driven by one category",
            f"{cell['feature']} x {cell['slicer']}",
            f"Only <b>{esc(cell['driven_by_single_category'])}</b> reproduces the overall pattern "
            f"({esc(cell['overall_shape'])}) among the non-small-n, non-structural categories -- the rest "
            "diverge. Don't present the overall shape as if it held broadly.",
            flag=True,
        )
    scoping = ('<span class="nf-badge type" style="margin-left:8px;">boolean slicer -- diverges = True vs False, '
               'not vs pooled overall</span>') if cell.get("slicer_type") == "boolean" else ""

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(cell['feature'])} &times; {esc(cell['slicer'])}
  <span style="font-weight:400;font-size:15px;color:var(--text-muted);">(active, xT delta)</span>{scoping}</h2>
  <p class="test-subnote">Overall shape: <b>{esc(cell['overall_shape'])}</b> (mean-delta range
  {esc(str(cell['overall_mean_xt_range']))}) &mdash; {cell['n_categories_diverging']}/{cell['n_categories']}
  categories diverge. Overall shape given a non-zero delta:
  <b>{esc(str(cell.get('overall_shape_given_nonzero_delta', 'n/a')))}</b>. "!" marks a genuine (non-small-n,
  non-structural) divergence; "crosses 0" marks a category whose binned curve spans both threat-reducing and
  threat-increasing means.</p>
  {driven_html}
  <div class="qchart-card sr-wrap">
  <table class="sr-table">
  <thead><tr>
    <th>category</th><th>n</th><th>shape</th><th>curve (dashed line = zero)</th>
    <th>n (non-zero delta)</th><th>shape (non-zero delta)</th><th>zero mass vs magnitude</th>
  </tr></thead>
  <tbody>{"".join(_category_row(l, c) for l, c in cell["categories"].items())}</tbody>
  </table>
  </div>
</div>"""


def _summary_section(summary: dict, show_flips: bool) -> str:
    cards = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(v['feature'])}</h5>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0 0 2px;">unconditional:
  {'<b style="color:var(--pos);">needs interaction</b>' if v['needs_interaction_term_signal'] else '<b style="color:var(--good);">stable</b>'}
  ({len(v['diverging_slicers'])}/{v['n_slicers_tested']})</p>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0;">non-zero delta:
  {'<b style="color:var(--pos);">needs interaction</b>' if v['needs_interaction_term_signal_given_nonzero_delta'] else '<b style="color:var(--good);">stable</b>'}
  ({len(v['diverging_slicers_given_nonzero_delta'])}/{v['n_slicers_tested']})</p>
</div>"""
        for v in sorted(summary["by_feature"].values(), key=lambda v: v["feature"])
    )

    flips = ""
    if show_flips:
        rows = "".join(
            f"<tr><td><code>{esc(r['feature'])}</code></td><td><code>{esc(r['slicer'])}</code></td>"
            f"<td>{esc(r['category'])}</td><td>{r['diverges_unconditional']}</td>"
            f"<td>{r['diverges_given_nonzero_delta']}</td><td>{esc(r['note'])}</td></tr>"
            for r in summary.get("conditioning_agreement", []) if not r["agrees"]
        )
        flips = f"""
<h3 style="margin-top:32px;">Where removing the zero mass changes the conclusion ({summary.get('n_conditioning_disagreements', 0)})</h3>
<p class="test-subnote">Categories where "diverges from overall" flips once rows with an exactly-zero delta are
removed -- either an unconditional divergence turns out to be carried by the zero mass, or a category that
looked stable shows a real magnitude-level divergence once that mass is gone.</p>
<div class="qchart-card" style="overflow-x:auto;">
<table class="sr-table">
<thead><tr><th>feature</th><th>slicer</th><th>category</th><th>diverges unconditional</th>
<th>diverges non-zero</th><th>note</th></tr></thead>
<tbody>{rows}</tbody></table>
</div>"""

    return f"""
<div class="test-block">
  <h2 class="test-title">Per-feature summary</h2>
  <p class="test-subnote">The same "does this need an interaction term" signal as the binary and xg reports,
  split by whether it survives removing this target's structural-zero mass.</p>
  <div class="numsplit-grid" style="grid-template-columns:repeat(3,1fr);">{cards}</div>
  {flips}
</div>"""


def build_report(data: dict, *, version: str) -> str:
    summary = data["cross_dataset_summary"]
    cells = [c for slicers in data["results"]["active"].values() for c in slicers.values()]
    n_driven = sum(1 for c in cells if c.get("driven_by_single_category"))

    notes = f"""
<div class="rule-banner"><b>Cell provenance.</b> {esc(data.get('cell_provenance', data.get('v1_cross_reference', '')))}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Method, and what changed.</b> {esc(data['methodology'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Reading the curves.</b>
{esc(data['target_scale']['direction_note'])} Every sparkline carries a dashed zero baseline, which the xg
report has no need for -- its values are non-negative, so min-max scaling alone conveys the shape; here a curve
entirely below zero and one entirely above it would otherwise look identical. flat_margin on this run:
<code>{data['flat_margin']:.6f}</code> unconditional, <code>{data['flat_margin_given_nonzero_delta']:.6f}</code>
on the non-zero-delta subset.</div>"""

    if version == "V2":
        notes += f"""
<div class="rule-banner" style="margin-top:16px;"><b>Not applicable here, said so rather than substituted.</b>
{esc(data['archetype_slicer_not_applicable'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Boolean-slicer scoping.</b>
{esc(data['boolean_slicer_scoping_note'])}</div>"""

    body = notes + _summary_section(summary, show_flips=(version == "V1")) + "".join(_cell_section(c) for c in cells)

    if version == "V1":
        eyebrow = "SLICE STRATIFICATION V1 -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY"
        title = "Categorical Stratification of Established Patterns (xT delta)"
        dek = (
            "Every ACTIVE locked numerical feature crossed with every locked categorical slicer, against mean "
            "xT delta. Each category carries an unconditional curve and one restricted to rows with a non-zero "
            "delta. Active-binary leg only."
        )
        stats = [
            (str(len(cells)), "feature x slicer cells"),
            (str(summary["n_genuine_divergences"]), "unconditional divergences"),
            (str(summary["n_genuine_divergences_given_nonzero_delta"]), "non-zero-delta divergences"),
            (str(summary["n_conditioning_disagreements"]), "conditioning flips"),
            (str(n_driven), "cells driven by one category"),
        ]
    else:
        eyebrow = "SLICE STRATIFICATION V2 -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY"
        title = "Boolean-Flag Slicers (xT delta)"
        dek = (
            "V2's disjoint slicer set, active half: every ACTIVE locked numerical feature crossed with every "
            "locked boolean flag. The archetype slicer the passive half uses does not exist in the active "
            "parquet, so neither does its closing comparison. Active-binary leg only."
        )
        stats = [
            (str(len(cells)), "feature x slicer cells"),
            (str(summary["n_genuine_divergences"]), "unconditional divergences"),
            (str(summary["n_genuine_divergences_given_nonzero_delta"]), "non-zero-delta divergences"),
            (str(summary["n_features_with_at_least_one_diverging_slicer"]), "features needing an interaction term"),
        ]

    return render.render_article(
        eyebrow=eyebrow, title=title, dek=dek, stats=stats, body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + render.CORR_CSS + EXTRA_CSS,
    )


def main() -> None:
    for in_path, out_path, version in ((V1_IN, V1_OUT, "V1"), (V2_IN, V2_OUT, "V2")):
        data = json.loads(in_path.read_text(encoding="utf-8"))
        out_path.write_text(build_report(data, version=version), encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"Wrote {out_path} ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
