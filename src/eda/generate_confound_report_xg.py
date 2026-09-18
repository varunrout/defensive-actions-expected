"""CLI entrypoint: render reports/analysis/xg_target/CONFOUND_ANALYSIS.json as a
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
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "CONFOUND_ANALYSIS.json"
BINARY_INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CONFOUND_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "CONFOUND_ANALYSIS.html"


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


def _given_shot_section(test: dict) -> str:
    gs = test["given_shot"]
    v, gv = test["verdict"]["verdict"], gs["verdict"]["verdict"]
    verdict_class = {"no": "v-no", "yes": "v-yes", "partially": "v-partially", "inconclusive": "v-partially"}[gv]

    rates = [b["rate"] for b in gs["marginal_table"] if b["rate"] is not None]
    rng = (max(rates) - min(rates)) if rates else 0.0
    mean_rate = sum(rates) / len(rates) if rates else 1.0
    is_flat = rng < 0.05 * mean_rate

    occurrence_vs_quality_html = ""
    if is_flat:
        occurrence_vs_quality_html = f"""
<div class="finding" style="margin:16px 0;">
<span class="tag">occurrence, not quality</span>
<p>The unconditional reversal (which reflects whether a shot happens) is essentially <b>flat</b> once conditioned
on a shot happening (range {rng:.6f}, ~{rng / mean_rate * 100:.1f}% of the mean) -- this feature relates to
whether a chance occurs, not to how good the chance is once it does. A real, useful finding on its own:
occurrence and quality are separate things here.</p>
</div>"""
    elif v != gv:
        occurrence_vs_quality_html = f"""
<div class="finding flag" style="margin:16px 0;">
<span class="tag">unconditional vs shot-conditional verdict differs</span>
<p>Unconditional verdict: <b>{esc(v)}</b>. Given-shot verdict: <b>{esc(gv)}</b>. The reversal's relationship to
the confound changes once restricted to rows where a shot happened.</p>
</div>"""
    else:
        occurrence_vs_quality_html = f"""
<div class="finding" style="margin:16px 0;">
<span class="tag">holds for quality too</span>
<p>The marginal pattern survives conditioning on a shot happening (range {rng:.6f}) -- this feature relates to
chance quality, not just chance occurrence. Same verdict as unconditional: <b>{esc(gv)}</b>.</p>
</div>"""

    return f"""
<h3 style="margin-top:24px;">Given a shot happened (n={gs['n_rows_used']:,})</h3>
<p class="test-subnote">Same marginal-quartile + stratified-by-confound structure, restricted to
target_future_shot_10s==1 -- the real test of whether {esc(test['marginal_column'])} says anything about chance
QUALITY, not just chance occurrence.</p>
{occurrence_vs_quality_html}
<div class="verdict-banner {verdict_class}">
<b>Given-shot verdict: {esc(gv.upper())}</b> &mdash; {esc(gs['verdict']['verdict_meaning'])}
({gs['verdict']['n_strata_reversal_survives']}/{gs['verdict']['n_strata_total']} strata)
</div>
<div class="qchart-card">
  <h4>Marginal mean xG by {esc(test['marginal_column'])} {"True/False value" if "marginal_type" in test else "quartile"}, shot-only subset</h4>
  {_marginal_chart(gs['marginal_table'])}
</div>
<div class="stratum-grid">{"".join(_stratum_card(s) for s in gs['stratified_table'])}</div>"""


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
  <p class="test-subnote">Marginal: <code>{esc(test['marginal_column'])}</code>{f' ({esc(test["marginal_type"])})' if "marginal_type" in test else ""}
  vs proposed confound: <code>{esc(test['confound_column'])}</code> -- {esc(test['proposed_confound_reason'])}</p>
  {divergence_html}
  <div class="verdict-banner {verdict_class}">
  <b>Unconditional verdict: {esc(v['verdict'].upper())}</b> &mdash; {esc(v['verdict_meaning'])}
  ({v['n_strata_reversal_survives']}/{v['n_strata_total']} strata)
  </div>
  <div class="qchart-card">
    <h4>Marginal mean xG by {esc(test['marginal_column'])} {"True/False value" if "marginal_type" in test else "quartile"}</h4>
    <p class="qc-note">Unconditioned.</p>
    {_marginal_chart(test['marginal_table'])}
  </div>
  <div class="stratum-grid">{"".join(_stratum_card(s) for s in test['stratified_table'])}</div>
  {_given_shot_section(test)}
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
            "quartile-stratification method, against mean xG instead of shot-rate percentage. Each test now also "
            "carries a shot-conditional version (target_future_shot_10s==1 only) -- the real test of whether the "
            "reversal is about chance quality, not just chance occurrence."
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
