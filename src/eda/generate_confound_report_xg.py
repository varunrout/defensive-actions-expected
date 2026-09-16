"""CLI entrypoint: render reports/eda_xg/CONFOUND_ANALYSIS.json as a
self-contained HTML report -- the xG counterpart to CONFOUND_ANALYSIS.html.

Usage:
    python -m src.eda.generate_confound_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "CONFOUND_ANALYSIS.json"
BINARY_INPUT_PATH = REPO_ROOT / "reports" / "eda" / "CONFOUND_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "CONFOUND_ANALYSIS.html"


def _qbar_row(b: dict, max_val: float, small: bool = False) -> str:
    pct = min(100.0, (b["rate"] / max_val) * 100) if b["rate"] is not None and max_val else 0.0
    return f"""
<div class="qbar-col">
  <span class="qbar-val">{b['rate']:.5f}</span>
  <div class="qbar-fill" style="height:{max(2, pct)}%"></div>
  <span class="qbar-label">{esc(b['bin'])}</span>
  <span class="qbar-n">n={b['n']:,}</span>
</div>"""


def _marginal_chart(table: list[dict]) -> str:
    max_val = max((b["rate"] for b in table if b["rate"] is not None), default=1.0) or 1.0
    return f'<div class="qbar-row">{"".join(_qbar_row(b, max_val) for b in table)}</div>'


def _stratum_card(stratum: dict) -> str:
    max_val = max((b["rate"] for b in stratum["bins"] if b["rate"] is not None), default=1.0) or 1.0
    delta = stratum["first_to_last_delta_pp"]
    delta_str = f"{delta:+.6f}" if delta is not None else "n/a"
    return f"""
<div class="stratum-card">
  <h5>{esc(stratum['stratum'])}</h5>
  <p class="stratum-delta">first&rarr;last mean-xG delta: <b>{delta_str}</b></p>
  <div class="qbar-row small">{"".join(_qbar_row(b, max_val, small=True) for b in stratum['bins'])}</div>
</div>"""


def _test_section(test: dict, binary_verdict: str | None) -> str:
    v = test["verdict"]
    verdict_class = {"no": "v-no", "yes": "v-yes", "partially": "v-partially", "inconclusive": "v-partially"}[v["verdict"]]

    divergence_html = ""
    if binary_verdict is not None and binary_verdict != v["verdict"]:
        divergence_html = f"""
<div class="finding flag" style="margin:16px 0;">
<span class="tag">diverges from the binary-target audit</span>
<p>Binary target verdict: <b>{esc(binary_verdict)}</b>. xG verdict: <b>{esc(v['verdict'])}</b>. Same
stratification, same features -- the two targets genuinely disagree here. Don't assume the binary-target
conclusion carries over.</p>
</div>"""

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(test['title'])} (xG)</h2>
  <p class="test-subnote">Proposed confound: <code>{esc(test['confound_column'])}</code> -- {esc(test['proposed_confound_reason'])}</p>
  {divergence_html}
  <div class="verdict-banner {verdict_class}">
  <b>Verdict: {esc(v['verdict'].upper())}</b> &mdash; {esc(v['verdict_meaning'])}
  ({v['n_strata_reversal_survives']}/{v['n_strata_total']} strata)
  </div>
  <div class="qchart-card">
    <h4>Marginal mean xG by {esc(test['marginal_column'])} quartile</h4>
    <p class="qc-note">Unconditioned.</p>
    {_marginal_chart(test['marginal_table'])}
  </div>
  <div class="stratum-grid">{"".join(_stratum_card(s) for s in test['stratified_table'])}</div>
</div>"""


def build_report(data: dict, binary_data: dict | None) -> str:
    binary_verdicts = (
        {t["name"]: t["verdict"]["verdict"] for t in binary_data["tests"]} if binary_data else {}
    )
    body = "".join(_test_section(t, binary_verdicts.get(t["name"])) for t in data["tests"])

    return render.render_article(
        eyebrow="CONFOUND ANALYSIS -- XG",
        title="Passive Dataset Confound (Reversal) Testing (xG)",
        dek=(
            "The continuous-target counterpart to CONFOUND_ANALYSIS.html -- same two reversal tests, same "
            "quartile-stratification method, against mean xG instead of shot-rate percentage."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (passive)"),
            (str(len(data["tests"])), "reversals tested"),
            (str(sum(1 for t in data['tests'] if binary_verdicts.get(t['name']) and binary_verdicts[t['name']] != t['verdict']['verdict'])), "diverge from binary-target verdict"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    binary_data = json.loads(BINARY_INPUT_PATH.read_text(encoding="utf-8")) if BINARY_INPUT_PATH.exists() else None
    html = build_report(data, binary_data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
