"""CLI entrypoint: render reports/analysis/xt_target/REVIEW_ANALYSIS.json as a
self-contained HTML report -- the xT sibling of generate_review_report_xg.py
(the confirmed renderer of xg_target/REVIEW_METHODOLOGY.html).

Usage:
    python -m src.eda.generate_review_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc

IN_PATH = xc.OUT_DIR / "REVIEW_ANALYSIS.json"
OUT_PATH = xc.OUT_DIR / "REVIEW_METHODOLOGY.html"


def _evidence_row(r: dict) -> str:
    ev = r.get("evidence", {})
    same_sign = ev.get("same_sign")
    sign_cell = "&mdash;" if same_sign is None else ("same sign" if same_sign else "<b>OPPOSITE SIGN</b>")
    return f"""
<tr>
  <td>{esc(r['feature_a'])} &harr; {esc(r['feature_b'])}</td>
  <td>{ev.get('lift_a', '—')}</td>
  <td>{ev.get('lift_b', '—')}</td>
  <td>{ev.get('lift_diff', '—')}</td>
  <td>{sign_cell}</td>
  <td>{esc(r['verdict'])}</td>
</tr>"""


def build_report(active: dict, meta: dict) -> str:
    s = active["target_scale"]
    hcalls = active["needs_human_call"]
    hcall_html = "".join(
        f'<div class="hcall-item"><div class="hp-pair">{esc(i["feature_a"])} &harr; {esc(i["feature_b"])} '
        f'<span style="color:var(--text-muted); font-weight:400;">({esc(i.get("method",""))}, r={i.get("value")})</span></div>'
        f'<div class="hp-reason">{esc(i["reason"])}</div></div>'
        for i in hcalls
    )
    type2_rows = "".join(_evidence_row(r) for r in active["resolved_pairs"] if r.get("relationship_type") == 2)
    counts = "".join(f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in sorted(active["verdict_counts"].items()))

    body = f"""
<div class="rule-banner"><b>xT-delta lift, not shot-rate lift and not xG lift.</b> Every Type-2 (boolean vs
boolean) verdict on this page traces to a mean-<code>{esc(xc.TARGET)}</code> difference (True-group mean minus
False-group mean), never to the correlation r-value alone. Types 1, 3 and 4 are target-independent (structural
correlation threshold, conceptual eta threshold, name-pattern persistence check) and their resolvers are
<b>imported and reused unchanged</b> from the binary-target review -- only Type 2 can disagree between targets.
Active-binary leg only; the passive half of the xg version's file is out of scope for this target.</div>

<div class="rule-banner" style="margin-top:16px;">
<b>Threshold adaptation, stated not assumed.</b> {esc(s['scale_note'])}
</div>

<div class="rule-banner" style="margin-top:16px;">
<b>Why the opposite-sign rule matters more here.</b> {esc(s['direction_note'])} A Type-2 pair with opposite-sign
lift means one flag marks rows where xT FELL across the action and the other marks rows where it ROSE -- a
disagreement about direction, not degree. On <code>target_future_xg_10s</code> (strictly non-negative) the same
rule could only ever express "one flag is above the base rate, the other below it".
<b>{active['n_type2_opposite_sign_pairs']}</b> Type-2 pair(s) resolve on that rule in this run.</div>

<h2 style="margin-top:36px;">Thresholds used (translated from the xg suite, computed at runtime)</h2>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Quantity</th><th>Value</th><th>How it was derived</th></tr>
<tr><td>mean {esc(xc.TARGET)}</td><td>{s['mean']:+.6f}</td><td>measured on the {s['n_defined']:,} rows with a defined delta</td></tr>
<tr><td>std {esc(xc.TARGET)}</td><td>{s['std']:.6f}</td><td>the scale-setting statistic, replacing the mean</td></tr>
<tr><td>xg flat margin on the same rows</td><td>{s['xg_flat_margin_absolute']:.6f}</td><td>0.5 &times; mean(target_future_xg_10s), the xg suite's own convention</td></tr>
<tr><td>as a fraction of xg's own std</td><td>{s['xg_flat_margin_as_fraction_of_xg_std']:.6f}</td><td>the translated ratio</td></tr>
<tr><td>near-identical threshold (0.5&times; equivalent)</td><td>{active['near_identical_threshold']:.6f}</td><td>that fraction &times; std({esc(xc.TARGET)})</td></tr>
<tr><td>distinct-signal threshold (1.5&times; equivalent)</td><td>{active['distinct_signal_threshold']:.6f}</td><td>3 &times; the near-identical threshold, preserving the xg suite's own 0.5:1.5 ratio</td></tr>
</table></div>

<h2 style="margin-top:36px;">Verdict counts (active)</h2>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Verdict</th><th>Pairs</th></tr>
{counts}
</table></div>

<div class="hcall-panel">
  <p class="hp-title">needs_human_call -- {len(hcalls)} pair(s), active</p>
  <p class="hp-note">Same backlog composition as the binary- and xg-target reviews: dominated by
  continuous&harr;continuous pairs, which no relationship type covers regardless of target.</p>
  {hcall_html}
</div>

<h2>Type 2 evidence ledger -- boolean vs boolean, xT-delta lift (active)</h2>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Pair</th><th>Lift A</th><th>Lift B</th><th>Diff</th><th>Sign</th><th>Verdict</th></tr>
{type2_rows}
</table></div>

<div class="closing-note" style="margin-top:28px;">{esc(meta['note'])}</div>
"""

    return render.render_article(
        eyebrow="REVIEW METHODOLOGY -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="Resolving the Review Tier (xT delta)",
        dek=(
            "How every ACTIVE REVIEW-tier pair resolves using xT-delta lift as evidence -- the xT-target "
            "counterpart to the xg portal's REVIEW_METHODOLOGY.html, on the same reconstructed 51-era pool. "
            "Active-binary leg only."
        ),
        stats=[
            (str(active["n_review_pairs"]), "active review pairs"),
            (str(len(hcalls)), "needs human call"),
            (str(active["n_type2_opposite_sign_pairs"]), "Type-2 opposite-sign"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS,
    )


def main() -> None:
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    OUT_PATH.write_text(build_report(data["datasets"]["active"], data), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
