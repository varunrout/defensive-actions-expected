"""CLI entrypoint: render reports/analysis/shot_target/SLICER_REDUNDANCY.json as a
self-contained HTML report, in the same shared design system as the other
EDA reports.

Usage:
    python -m src.eda.generate_slicer_redundancy_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SLICER_REDUNDANCY.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SLICER_REDUNDANCY.html"

VERDICT_CLASS = {"independent": "good", "partially redundant": "warn", "highly redundant": "severe"}


def _pair_row(p: dict) -> str:
    v_cls = VERDICT_CLASS[p["verdict"]]
    named = f' <span class="nf-badge type">NAMED CANDIDATE</span>' if "named_candidate_reason" in p else ""
    disagree = ' <span class="nf-badge unreliable">METRICS DISAGREE</span>' if p["metrics_disagree"] else ""
    ct = p["contingency_table_shape"]
    note = f'<p class="numfeat-note">{esc(p["named_candidate_reason"])}</p>' if "named_candidate_reason" in p else ""

    return f"""
<div class="vif-row" style="grid-template-columns: 260px 1fr 90px;">
  <div class="vif-label">{esc(p['slicer_a'])} &times; {esc(p['slicer_b'])}{named}{disagree}</div>
  <div class="vif-track">
    <div class="vif-fill {v_cls}" style="width:{min(100, p['cramers_v']*100):.1f}%;"></div>
    <div class="vif-threshold" style="left:10%;"></div>
    <div class="vif-threshold" style="left:30%;"></div>
  </div>
  <div class="vif-value">V={p['cramers_v']}</div>
</div>
<p style="font-size:11px;color:var(--text-muted);margin:2px 0 10px 0;">NMI={p['normalized_mutual_info']} (verdict from NMI: {esc(p['verdict_from_nmi'])})
&mdash; contingency table {ct[0]}&times;{ct[1]} &mdash; n={p['n_rows_used']:,} (dropped {p['n_rows_dropped_na']} with a null)
&mdash; verdict: <b>{esc(p['verdict'])}</b></p>
{note}"""


def _cluster_view(title: str, view: dict, note: str, kind: str) -> str:
    clusters_html = "".join(
        f'<span class="nf-badge unreliable" style="margin:2px 6px 2px 0;">{esc(" + ".join(c))}</span>'
        for c in view["clusters"]
    ) or '<span style="color:var(--text-muted);">none</span>'
    independent_html = "".join(
        f'<span class="nf-badge locked" style="margin:2px 6px 2px 0;">{esc(s)}</span>'
        for s in view["independent_slicers"]
    ) or '<span style="color:var(--text-muted);">none</span>'

    return f"""
<div class="ba-card {kind}">
  <h4>{esc(title)}</h4>
  <p style="font-size:11.5px;color:var(--text-secondary);margin:0 0 10px;">{esc(note)}</p>
  <p style="font-size:12px;margin:0 0 6px;"><b>Redundant clusters:</b><br>{clusters_html}</p>
  <p style="font-size:12px;margin:0;"><b>Independent of everything else:</b><br>{independent_html}</p>
</div>"""


def _dataset_section(ds: dict) -> str:
    pairs_html = "".join(_pair_row(p) for p in ds["pairs"])
    c = ds["clusters"]
    clusters_html = f"""
<div class="before-after" style="grid-template-columns: repeat(3, 1fr);">
  {_cluster_view("From raw Cramer's V (over-clusters)", c["from_cramers_v"], "Edges where the Cramer's-V verdict != independent -- inflated by sparse contingency-table cells in this data.", "before")}
  {_cluster_view("From NMI (conservative)", c["from_nmi"], "Edges where the NMI verdict != independent.", "before")}
  {_cluster_view("Consensus -- trust this", c["consensus_both_metrics_agree"], "Edges where BOTH metrics agree it's not independent (not metrics_disagree). This is the view to act on.", "after")}
</div>"""

    return f"""
<div class="vif-dataset-block">
  <h2 class="test-title">{esc(ds['dataset'])} ({ds['n_pairs']} pairs)</h2>
  <div class="vif-legend">
    <span><span class="dot" style="background:var(--good);"></span>independent (V &lt; 0.1)</span>
    <span><span class="dot" style="background:var(--amber);"></span>partially redundant (0.1 &ndash; 0.3)</span>
    <span><span class="dot" style="background:var(--pos);"></span>highly redundant (V &gt; 0.3)</span>
  </div>
  {pairs_html}
  <h3 style="margin-top:28px;">Cluster views</h3>
  <p class="test-subnote">{esc(ds['cluster_note'])}</p>
  {clusters_html}
</div>"""


def build_report(data: dict) -> str:
    thresholds = data["verdict_thresholds"]
    threshold_html = finding_card(
        "verdict thresholds",
        "stated explicitly, not implicit",
        f"{esc(thresholds['independent'])} &nbsp;|&nbsp; {esc(thresholds['partially redundant'])} &nbsp;|&nbsp; "
        f"{esc(thresholds['highly redundant'])}. {esc(thresholds['note'])}",
    )
    downstream_html = finding_card("downstream connection", "interaction-term shortlist (not acted on here)", esc(data["downstream_connection"]))

    body = f"""
<p class="test-subnote" style="margin-bottom:16px;">{esc(data['methodology'])}</p>
{threshold_html}
{downstream_html}
{"".join(_dataset_section(ds) for ds in data["datasets"].values())}
"""

    n_pairs = sum(ds["n_pairs"] for ds in data["datasets"].values())
    n_highly_redundant = sum(1 for ds in data["datasets"].values() for p in ds["pairs"] if p["verdict"] == "highly redundant")
    n_disagree = sum(1 for ds in data["datasets"].values() for p in ds["pairs"] if p["metrics_disagree"])

    return render.render_article(
        eyebrow="SLICER REDUNDANCY",
        title="Are the Slice-Stratification Slicers Independent?",
        dek=(
            "Prompt 25/26's slice-stratification grid tested every locked numerical feature against every locked "
            "categorical slicer. Before that becomes an interaction-term shortlist for modelling: are the slicers "
            "themselves independent, or does testing two redundant slicers just double-count one finding?"
        ),
        stats=[
            (str(n_pairs), "slicer pairs tested"),
            (str(n_highly_redundant), "highly redundant by Cramer's V"),
            (str(n_disagree), "pairs where V and NMI disagree"),
        ],
        body_html=body,
        extra_css=render.VIF_CSS + render.NUMERICAL_ATLAS_CSS + """
.vif-fill.good { background: var(--good); }
""",
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
