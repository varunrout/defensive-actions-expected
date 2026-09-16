"""CLI entrypoint: render reports/eda_xg/REVIEW_ANALYSIS.json as a
self-contained HTML report -- the xG counterpart to REVIEW_METHODOLOGY.html.

Usage:
    python -m src.eda.generate_review_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = REPO_ROOT / "reports" / "eda_xg" / "REVIEW_ANALYSIS.json"
OUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "REVIEW_METHODOLOGY.html"


def _evidence_row(r: dict) -> str:
    ev = r.get("evidence", {})
    return f"""
<tr>
  <td>{esc(r['feature_a'])} &harr; {esc(r['feature_b'])}</td>
  <td>{ev.get('lift_a', '—')}</td>
  <td>{ev.get('lift_b', '—')}</td>
  <td>{ev.get('lift_diff', '—')}</td>
  <td>{esc(r['verdict'])}</td>
</tr>"""


def build_report(active: dict, passive: dict) -> str:
    n_hcall_active = len(active["needs_human_call"])
    n_hcall_passive = len(passive["needs_human_call"])

    all_hcalls = [{**r, "_dataset": "active"} for r in active["needs_human_call"]] + \
                 [{**r, "_dataset": "passive"} for r in passive["needs_human_call"]]
    hcall_html = "".join(
        f'<div class="hcall-item"><div class="hp-pair">[{esc(item["_dataset"])}] {esc(item["feature_a"])} &harr; {esc(item["feature_b"])} '
        f'<span style="color:var(--text-muted); font-weight:400;">({esc(item.get("method",""))}, r={item.get("value")})</span></div>'
        f'<div class="hp-reason">{esc(item["reason"])}</div></div>'
        for item in all_hcalls
    )

    type2_rows_active = "".join(_evidence_row(r) for r in active["resolved_pairs"] if r.get("relationship_type") == 2)
    type2_rows_passive = "".join(_evidence_row(r) for r in passive["resolved_pairs"] if r.get("relationship_type") == 2)

    body = f"""
<div class="rule-banner"><b>xG lift, not shot-rate lift.</b> Every Type-2 (boolean vs boolean) verdict on this
page traces to a mean-xG difference (True group mean xG minus False group mean xG), not the correlation r-value
alone and not shot-rate lift. Types 1, 3 and 4 are target-independent (structural correlation threshold,
conceptual eta threshold, name-pattern check) and reused unchanged from
<a href="../eda/REVIEW_METHODOLOGY.html" style="color:var(--neg);">REVIEW_METHODOLOGY.html</a> (the binary-target
version) -- only Type 2 can disagree between the two targets.</div>

<h2 style="margin-top:36px;">Thresholds used (relative to each dataset's own mean xG)</h2>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Dataset</th><th>Overall mean xG</th><th>Near-identical threshold (0.5x)</th><th>Distinct-signal threshold (1.5x)</th></tr>
<tr><td>Active</td><td>{active['overall_mean_xg']}</td><td>{active['near_identical_threshold']}</td><td>{active['distinct_signal_threshold']}</td></tr>
<tr><td>Passive</td><td>{passive['overall_mean_xg']}</td><td>{passive['near_identical_threshold']}</td><td>{passive['distinct_signal_threshold']}</td></tr>
</table></div>

<div class="hcall-panel">
  <p class="hp-title">needs_human_call -- {n_hcall_active + n_hcall_passive} pairs across both datasets ({n_hcall_active} active, {n_hcall_passive} passive)</p>
  <p class="hp-note">These pairs did not resolve automatically -- same backlog composition as the binary-target
  review (Types 1/3/4 are identical; the backlog is dominated by continuous&harr;continuous pairs, which no
  relationship type covers regardless of target).</p>
  {hcall_html}
</div>

<h2>Type 2 evidence ledger -- boolean vs boolean, xG lift</h2>
<h3>Active</h3>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Pair</th><th>Lift A</th><th>Lift B</th><th>Diff</th><th>Verdict</th></tr>
{type2_rows_active}
</table></div>

<h3 style="margin-top:28px;">Passive</h3>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Pair</th><th>Lift A</th><th>Lift B</th><th>Diff</th><th>Verdict</th></tr>
{type2_rows_passive}
</table></div>
"""

    return render.render_article(
        eyebrow="REVIEW METHODOLOGY -- XG",
        title="Resolving the Review Tier (xG)",
        dek=(
            "How every REVIEW-tier pair resolves using xG lift instead of shot-rate lift as evidence -- the "
            "continuous-target counterpart to REVIEW_METHODOLOGY.html, on the same reconstructed 51/44-era pool."
        ),
        stats=[
            (str(active["n_review_pairs"]), "active review pairs"),
            (str(passive["n_review_pairs"]), "passive review pairs"),
            (str(n_hcall_active + n_hcall_passive), "needs human call"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS,
    )


def main() -> None:
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    html = build_report(data["datasets"]["active"], data["datasets"]["passive"])
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
