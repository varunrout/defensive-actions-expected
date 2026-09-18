"""CLI entrypoint: render CONFOUND_ANALYSIS.json as a self-contained HTML
report, in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.generate_confound_reports
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CONFOUND_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CONFOUND_ANALYSIS.html"

VERDICT_CLASS = {"no": "v-no", "yes": "v-yes", "partially": "v-partially", "inconclusive": "v-partially"}
VERDICT_LABEL = {
    "no": "Reversal survives -- confound does not explain it away",
    "yes": "Reversal disappears -- consistent with the confound explaining it away",
    "partially": "Mixed -- survives in some strata, flattens in others",
    "inconclusive": "Inconclusive",
}


def _marginal_chart(table: list[dict], col_name: str, is_boolean: bool = False) -> str:
    max_rate = max(b["rate"] for b in table) * 1.15 or 1.0
    bars = "".join(
        f"""
<div class="qbar-col">
  <span class="qbar-val">{b['rate']:.2f}%</span>
  <div class="qbar-fill" style="height:{max(2, b['rate']/max_rate*100):.1f}%"></div>
  <span class="qbar-label">{esc(b['bin'])}</span>
  <span class="qbar-n">n={b['n']:,}</span>
</div>"""
        for b in table
    )
    grouping = "True/False value" if is_boolean else "quartile"
    return f"""
<div class="qchart-card">
  <h4>Marginal shot rate by {esc(col_name)} {grouping}</h4>
  <p class="qc-note">Unconditioned -- the raw reversal being tested.</p>
  <div class="qbar-row">{bars}</div>
</div>"""


def _stratified_chart(strata: list[dict], confound_col: str) -> str:
    all_rates = [b["rate"] for s in strata for b in s["bins"] if b["rate"] is not None]
    max_rate = (max(all_rates) * 1.15) if all_rates else 1.0

    cards = []
    for s in strata:
        bars = "".join(
            f"""
<div class="qbar-col">
  <span class="qbar-val">{b['rate']:.2f}%</span>
  <div class="qbar-fill" style="height:{max(2, b['rate']/max_rate*100):.1f}%"></div>
  <span class="qbar-label">{esc(b['bin'])}</span>
  <span class="qbar-n">n={b['n']:,}</span>
</div>""" if b["rate"] is not None else '<div class="qbar-col"><span class="qbar-label">n/a</span></div>'
            for b in s["bins"]
        )
        delta = s["first_to_last_delta_pp"]
        delta_str = f"{delta:+.2f}pp (first quartile to last)" if delta is not None else "n/a"
        cards.append(f"""
<div class="stratum-card">
  <h5>{esc(s['stratum'])}</h5>
  <p class="stratum-delta">delta: <b>{delta_str}</b></p>
  <div class="qbar-row small">{bars}</div>
</div>""")

    return f"""
<div class="qchart-card">
  <h4>Stratified by {esc(confound_col)} quartile</h4>
  <p class="qc-note">Same marginal breakdown, repeated within each quartile of the proposed confound.</p>
  <div class="stratum-grid">{''.join(cards)}</div>
</div>"""


def _extra_notes(test: dict) -> str:
    notes = []
    if "cluster_5_connection" in test:
        notes.append(("Cluster 5 connection (not decided here)", test["cluster_5_connection"]))
    if "confound_selection_note" in test:
        notes.append(("Confound selection", test["confound_selection_note"]))
    if "reliability_note" in test:
        notes.append(("Reliability caveat", test["reliability_note"]))
    if not notes:
        return ""
    cards = "".join(
        f"""
<div class="finding flag" style="margin:12px 0;">
<span class="tag">{esc(title)}</span>
<p>{esc(body)}</p>
</div>"""
        for title, body in notes
    )
    return cards


def _test_section(test: dict) -> str:
    v = test["verdict"]
    verdict_class = VERDICT_CLASS.get(v["verdict"], "v-partially")
    verdict_label = VERDICT_LABEL.get(v["verdict"], v["verdict"])
    confound_type_note = (
        f" ({esc(test['confound_type'])})" if "confound_type" in test else ""
    )
    marginal_type_note = (
        f" ({esc(test['marginal_type'])})" if "marginal_type" in test else ""
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(test['title'])}</h2>
  <p class="test-subnote"><code>{esc(test['marginal_column'])}</code>{marginal_type_note} vs proposed confound <code>{esc(test['confound_column'])}</code>{confound_type_note}
  &mdash; proposed reasoning: &ldquo;{esc(test['proposed_confound_reason'])}&rdquo;</p>

  {_extra_notes(test)}

  <div class="verdict-banner {verdict_class}">
    <b>Verdict: {esc(v['verdict'].upper())}</b> &mdash; {esc(verdict_label)}.
    {esc(v['verdict_meaning'])}
    ({v['n_strata_reversal_survives']}/{v['n_strata_total']} strata show the reversal surviving.)
  </div>

  {_marginal_chart(test['marginal_table'], test['marginal_column'], is_boolean="marginal_type" in test)}
  {_stratified_chart(test['stratified_table'], test['confound_column'])}
</div>"""


def build_confound_report(data: dict) -> str:
    body = "".join(_test_section(t) for t in data["tests"])

    body += """
<div class="caveat-box">
<h3>What this doesn't tell us</h3>
<p>Each test here checks exactly one candidate confound at a time. A reversal surviving stratification on
<em>this</em> confound does not rule out others. Open questions this analysis does not resolve:</p>
<ul>
<li>Attacker count or quality in the frame (more/better attackers could drive both tighter marking and a higher
eventual shot rate independently)</li>
<li><code>overload_score</code> (defensive overload could correlate with both marking tightness and outcome
through a path this test never conditions on)</li>
<li>Whether the 10s outcome window catches a shot that already had no time to be prevented, regardless of
marking or screening quality at the anchor frame</li>
</ul>
<p>None of these are resolved as "ruled out" -- they are open, not tested here.</p>
</div>"""

    return render.render_article(
        eyebrow="CONFOUND ANALYSIS",
        title="Reversal (Simpson's-Paradox-Style) Checks",
        dek=(
            "A marginal correlation can reverse direction once you condition on a third variable -- a classic "
            "Simpson's-paradox pattern. The first 2 tests below check marginal reversals in the passive dataset "
            "previously asserted as “probably a selection effect” without being tested. The 5 tests added after "
            "them (prompt 24 Part A) extend the same method to the threat-score U-shapes, both other lane-screening "
            "options, and defenders_within_10m -- the last of those carries an explicit reliability caveat, see its "
            "card below. The final 3 (prompt 29) close LEAKAGE_AUDIT.json's has_option_2/3 follow-up item, adapting "
            "the method for a boolean marginal (split True/False directly, not quartiled)."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (passive)"),
            (str(len(data["tests"])), "tests run"),
            (sum(1 for t in data["tests"] if t["verdict"]["verdict"] == "no"), "reversals surviving"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_confound_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
