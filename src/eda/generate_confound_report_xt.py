"""CLI entrypoint: render reports/analysis/xt_target/CONFOUND_ANALYSIS.json as
a self-contained HTML report -- the xT sibling of
generate_confound_report_xg.py (the confirmed renderer of the xg portal's
CONFOUND_ANALYSIS.html). Same `test-block` / `qbar-*` / `stratum-card` /
`verdict-banner` markup and the same render.CONFOUND_CSS.

Bars diverge from zero rather than growing from a baseline -- the xg
renderer's `height: rate/max_val` would render every negative mean delta as
a zero-height bar.

Usage:
    python -m src.eda.generate_confound_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc, finding_card

INPUT_PATH = xc.OUT_DIR / "CONFOUND_ANALYSIS.json"
BINARY_INPUT_PATH = xc.REPO_ROOT / "reports" / "analysis" / "shot_target" / "CONFOUND_ANALYSIS.json"
OUTPUT_PATH = xc.OUT_DIR / "CONFOUND_ANALYSIS.html"

EXTRA_CSS = """
.xtq-row { display: flex; align-items: stretch; gap: 10px; margin: 14px 0 6px; }
.xtq-col { flex: 1; display: flex; flex-direction: column; align-items: center; }
.xtq-track { position: relative; width: 100%; height: 130px; background: var(--plane); border-radius: 4px; }
.xtq-track .mid { position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: var(--text-primary); opacity: 0.35; }
.xtq-bar { position: absolute; left: 22%; width: 56%; border-radius: 3px; }
.xtq-bar.pos { bottom: 50%; background: var(--good); }
.xtq-bar.neg { top: 50%; background: var(--neg); }
.xtq-val { font-family: "JetBrains Mono", monospace; font-size: 11px; font-weight: 700; margin-bottom: 4px; }
.xtq-label { font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--text-secondary); margin-top: 5px; text-align: center; }
.xtq-n { font-family: "JetBrains Mono", monospace; font-size: 9.5px; color: var(--text-muted); }
.xtq-row.small .xtq-track { height: 78px; }
"""


def _bars(table: list[dict], small: bool = False) -> str:
    vals = [b["rate"] for b in table if b["rate"] is not None]
    max_abs = max([abs(v) for v in vals] + [1e-12])
    cols = ""
    for b in table:
        v = b["rate"]
        if v is None:
            cols += f'<div class="xtq-col"><span class="xtq-val">n/a</span><div class="xtq-track"><span class="mid"></span></div><span class="xtq-label">{esc(b["bin"])}</span></div>'
            continue
        h = max(1.5, abs(v) / max_abs * 50)
        side = "pos" if v >= 0 else "neg"
        shares = (f'{b["pct_negative"]:.0f}% neg / {b["pct_zero"]:.0f}% zero / {b["pct_positive"]:.0f}% pos'
                  if b.get("pct_negative") is not None else "")
        cols += f"""
<div class="xtq-col">
  <span class="xtq-val">{v:+.6f}</span>
  <div class="xtq-track" title="{esc(shares)}"><span class="mid"></span><div class="xtq-bar {side}" style="height:{h:.1f}%"></div></div>
  <span class="xtq-label">{esc(b['bin'])}</span>
  <span class="xtq-n">n={b['n']:,}</span>
</div>"""
    return f'<div class="xtq-row{" small" if small else ""}">{cols}</div>'


def _stratum_card(s: dict) -> str:
    d = s["first_to_last_delta_pp"]
    return f"""
<div class="stratum-card">
  <h5>{esc(s['stratum'])}</h5>
  <p class="stratum-delta">first&rarr;last mean-delta change: <b>{f"{d:+.6f}" if d is not None else "n/a"}</b></p>
  {_bars(s['bins'], small=True)}
</div>"""


def _conditional_section(test: dict) -> str:
    g = test["given_nonzero_delta"]
    v, gv = test["verdict"]["verdict"], g["verdict"]["verdict"]
    cls = {"no": "v-no", "yes": "v-yes", "partially": "v-partially", "inconclusive": "v-partially"}[gv]

    rates = [b["rate"] for b in g["marginal_table"] if b["rate"] is not None]
    rng = (max(rates) - min(rates)) if rates else 0.0

    if v != gv:
        note = f"""
<div class="finding flag" style="margin:16px 0;">
<span class="tag">unconditional and non-zero-delta verdicts differ</span>
<p>Unconditional verdict: <b>{esc(v)}</b>. Given a non-zero delta: <b>{esc(gv)}</b>. Removing this target's
structural-zero mass changes the conclusion -- the pattern was partly carried by which rows sit exactly at zero
rather than by the size of the swing.</p>
</div>"""
    else:
        note = f"""
<div class="finding" style="margin:16px 0;">
<span class="tag">holds once the zero mass is removed</span>
<p>The pattern survives restricting to rows with a non-zero delta (marginal range {rng:.6f}) -- it is about the
SIZE and DIRECTION of the threat swing, not merely about which rows registered any swing at all. Same verdict as
unconditional: <b>{esc(gv)}</b>.</p>
</div>"""

    return f"""
<h3 style="margin-top:24px;">Given a non-zero delta (n={g['n_rows_used']:,})</h3>
<p class="test-subnote">Same marginal + stratified-by-confound structure, restricted to
<code>{esc(xc.TARGET)} != 0</code> -- this target's structural-zero analogue of the xg report's
"given a shot happened" panel (Adaptation 2). It asks whether the pattern is about the size of the threat swing
rather than about whether the action crossed an xT grid-cell boundary at all.</p>
{note}
<div class="verdict-banner {cls}">
<b>Non-zero-delta verdict: {esc(gv.upper())}</b> &mdash; {esc(g['verdict']['verdict_meaning'])}
({g['verdict']['n_strata_reversal_survives']}/{g['verdict']['n_strata_total']} strata)
</div>
<div class="qchart-card"><h4>Marginal mean delta, non-zero-delta subset</h4>{_bars(g['marginal_table'])}</div>
<div class="stratum-grid">{"".join(_stratum_card(s) for s in g['stratified_table'])}</div>"""


def _test_section(test: dict, binary_verdict: str | None) -> str:
    v = test["verdict"]
    cls = {"no": "v-no", "yes": "v-yes", "partially": "v-partially", "inconclusive": "v-partially"}[v["verdict"]]

    divergence = ""
    if binary_verdict is not None and binary_verdict != v["verdict"]:
        divergence = f"""
<div class="finding flag" style="margin:16px 0;">
<span class="tag">diverges from the binary-target version of this same test</span>
<p>Binary target (<code>target_future_shot_10s</code>) verdict: <b>{esc(binary_verdict)}</b>. xT-delta verdict:
<b>{esc(v['verdict'])}</b>. Same stratification, same columns, same <code>_verdict()</code> function -- the two
targets genuinely disagree here. Don't assume the binary-target conclusion carries over.</p>
</div>"""
    elif binary_verdict is not None:
        divergence = f"""
<div class="finding" style="margin:16px 0;">
<span class="tag">agrees with the binary-target version of this same test</span>
<p>Both the binary target and xT delta return <b>{esc(v['verdict'])}</b> on this test.</p>
</div>"""

    new_badge = ('<span class="nf-badge type" style="margin-left:8px;">NEW TEST</span>'
                 if test.get("is_new_test") else "")
    extra = ""
    if test.get("confound_selection_note"):
        extra += finding_card("confound selection", "both candidates checked on this run",
                              esc(test["confound_selection_note"]))
    if test.get("reliability_note"):
        extra += finding_card("reliability caveat", "read the numbers as pooled-population description",
                              esc(test["reliability_note"]), flag=True)
    if test.get("prompt_64_link"):
        extra += finding_card("carried forward from Prompt 64", "Clearance / action_x artefact",
                              esc(test["prompt_64_link"]), flag=True)

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(test['title'])} (xT delta){new_badge}</h2>
  <p class="test-subnote">Marginal: <code>{esc(test['marginal_column'])}</code> ({esc(test['marginal_type'])})
  vs proposed confound: <code>{esc(test['confound_column'])}</code> ({esc(test['confound_type'])}) --
  {esc(test['proposed_confound_reason'])}</p>
  {finding_card("provenance", "mirrored or new -- said which", esc(test["provenance"]))}
  {extra}
  {divergence}
  <div class="verdict-banner {cls}">
  <b>Unconditional verdict: {esc(v['verdict'].upper())}</b> &mdash; {esc(v['verdict_meaning'])}
  ({v['n_strata_reversal_survives']}/{v['n_strata_total']} strata)
  </div>
  <div class="qchart-card">
    <h4>Marginal mean {esc(xc.TARGET)} by {esc(test['marginal_column'])}</h4>
    <p class="qc-note">Unconditioned. Bars diverge from the centre line: green up = xT fell (threat reduced),
    red down = xT rose. Hover a bar for its negative/zero/positive row shares.</p>
    {_bars(test['marginal_table'])}
  </div>
  <div class="stratum-grid">{"".join(_stratum_card(s) for s in test['stratified_table'])}</div>
  {_conditional_section(test)}
</div>"""


def build_report(data: dict, binary_data: dict | None) -> str:
    binary_verdicts = {t["name"]: t["verdict"]["verdict"] for t in binary_data["tests"]} if binary_data else {}
    s = data["target_scale"]

    body = f"""
<div class="rule-banner"><b>Scope, confirmed by reading the source.</b> {esc(data['step_0_scope_note'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Method, and what did and did not need adapting.</b>
{esc(data['methodology'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Reading the numbers.</b> {esc(s['direction_note'])}</div>
"""
    body += "".join(_test_section(t, binary_verdicts.get(t["name"])) for t in data["tests"])

    n_div = sum(1 for t in data["tests"]
                if binary_verdicts.get(t["name"]) and binary_verdicts[t["name"]] != t["verdict"]["verdict"])
    return render.render_article(
        eyebrow="CONFOUND ANALYSIS -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="Active Dataset Confound (Reversal) Testing (xT delta)",
        dek=(
            "One test mirrored from the only active confound test that exists in this repo, one new test "
            "motivated by Prompt 64's Clearance/action_x finding. The verdict function is reused byte-for-byte "
            "(sign-only, already target-scale-agnostic); each test carries a non-zero-delta panel in place of "
            "the xg version's shot-conditional one. Active-binary leg only."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (active)"),
            (str(len(data["tests"])), "tests"),
            ("1", "mirrored / 1 new"),
            (str(n_div), "diverge from binary-target verdict"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.CORR_CSS + EXTRA_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    binary = json.loads(BINARY_INPUT_PATH.read_text(encoding="utf-8")) if BINARY_INPUT_PATH.exists() else None
    OUTPUT_PATH.write_text(build_report(data, binary), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
