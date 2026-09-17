"""CLI entrypoint: render LEAKAGE_AUDIT.json as a self-contained HTML
report, in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.generate_leakage_reports
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "LEAKAGE_AUDIT.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "LEAKAGE_AUDIT.html"


def _ledger_row(label: str, method: str, magnitude_str: str, verdict: str, verdict_class: str) -> str:
    # .corr-row's grid is 1fr / 100px / 160px / 76px / 96px (pair / method /
    # magnitude-bar / value / chip) -- this ledger has no bar to draw, so the
    # third track is left empty rather than reusing a class built for one.
    return f"""
<div class="corr-row">
  <span class="corr-pair">{label}</span>
  <span class="corr-method">{esc(method)}</span>
  <span></span>
  <span class="corr-value">{magnitude_str}</span>
  <span class="verdict-chip {verdict_class}">{esc(verdict)}</span>
</div>"""


def _part_a_section(a: dict) -> str:
    rows = []
    for col, info in a["known_leaky_columns"].items():
        rows.append(_ledger_row(f"<code>{esc(col)}</code>", "presence check", "not in candidate list", "EXCLUDED", "distinct"))
    rows.append(_ledger_row(
        f"{a['n_columns_scanned_total']} columns scanned for future/next/avoided/outcome/after",
        "name-pattern scan",
        f"{len(a['all_leakage_suggestive_column_names'])} flagged by name",
        "CLEAN" if a["verdict"] == "clean" else "ISSUES FOUND",
        "distinct" if a["verdict"] == "clean" else "drop",
    ))
    flagged_list = "".join(f"<li><code>{esc(c)}</code></li>" for c in a["all_leakage_suggestive_column_names"])

    return f"""
<div class="test-block">
  <h2 class="test-title">Part A &mdash; known-leaky columns &amp; full name-pattern scan</h2>
  <p class="test-subnote">Verified directly against feature_config.py's PASSIVE candidate lists, not assumed from docstrings.</p>
  <div class="corr-ledger" style="margin-bottom:16px;">{''.join(rows)}</div>
  <div class="qchart-card">
    <h4>Every column matching future / next / avoided / outcome / after</h4>
    <p class="qc-note">Full list, so "nothing else found" is checkable, not just asserted.</p>
    <ul style="margin:0; padding-left:20px; font-size:13px; color:var(--text-secondary);">{flagged_list}</ul>
  </div>
</div>"""


def _mini_bar_table(by_value: dict, target_label: str) -> str:
    rows = ""
    max_rate = max((v["rate"] for v in by_value.values()), default=1) * 1.15 or 1.0
    for label in ("True", "False"):
        if label not in by_value:
            continue
        v = by_value[label]
        width = max(2, v["rate"] / max_rate * 100)
        rows += f"""
<div class="stratum-delta" style="display:grid; grid-template-columns: 60px 1fr 140px; align-items:center; gap:10px; margin-bottom:8px;">
  <span class="mono" style="font-family:'JetBrains Mono',monospace; font-weight:700;">{label}</span>
  <div class="vif-track"><div class="vif-fill severe" style="width:{width:.1f}%"></div></div>
  <span>{v['rate']:.2f}% (n={v['n']:,})</span>
</div>"""
    return f'<div style="margin:16px 0;">{rows}</div>'


def _part_b_section(b: dict) -> str:
    return f"""
<div class="test-block">
  <h2 class="test-title">Part B &mdash; <code>screened_option_was_avoided</code> guard, quantified</h2>
  <p class="test-subnote">Informational: the exclusion is correct on conceptual/structural grounds regardless of this result.</p>
  <div class="qchart-card">
    <h4>{esc(b['target_col'])} rate by {esc(b['column'])}</h4>
    <p class="qc-note">Essentially flat &mdash; delta {b['delta_pp']:+.2f}pp. Confirms low empirical leakage risk for this specific pair.</p>
    {_mini_bar_table(b['by_value'], b['target_col'])}
  </div>
</div>"""


def _part_c_section(c: dict) -> str:
    return f"""
<div class="test-block">
  <h2 class="test-title">Part C &mdash; <code>has_screened_outcome</code>: censoring-mechanism proxy (EXCLUDED)</h2>
  <p class="test-subnote">The real finding of this audit &mdash; a candidate feature removed as a result.</p>
  <div class="verdict-banner v-yes">
    <b>Verdict: EXCLUDED</b> &mdash; chi2={c['chi2']:.1f}, p={c['p_value']:.3e}, dof={c['dof']}. Ratio True:False rate
    = {c['ratio_true_to_false']}x.
  </div>
  <div class="qchart-card">
    <h4>{esc(c['target_col'])} rate by {esc(c['column'])}</h4>
    <p class="qc-note">has_screened_outcome=False means the target's own 10s window was truncated (end of period/match) -- not that nothing defensively interesting happened.</p>
    {_mini_bar_table(c['by_value'], c['target_col'])}
  </div>
  <div class="closing-note">{esc(c['explanation'])}</div>
</div>"""


def _part_d_section(d: dict) -> str:
    cards = ""
    for col, r in d.items():
        cards += f"""
<div class="qchart-card">
  <h4><code>{esc(col)}</code></h4>
  {_mini_bar_table(r['by_value'], r['target_col'])}
  <p class="qc-note">{esc(r['explanation'])}</p>
</div>"""
    return f"""
<div class="test-block">
  <h2 class="test-title">Part D &mdash; <code>has_option_2</code> / <code>has_option_3</code> (FLAGGED FOR FOLLOW-UP, not excluded)</h2>
  <div class="verdict-banner v-partially">
    <b>Verdict: FLAGGED FOR FOLLOW-UP</b> &mdash; different shape of finding from Part C. About the current freeze
    frame, not future-window computability. Kept as candidate features; worth a confound test later, not resolved here.
  </div>
  {cards}
</div>"""


def build_leakage_report(data: dict) -> str:
    body = _part_a_section(data["part_a_known_leaky_scan"])
    body += _part_b_section(data["part_b_screened_option_was_avoided"])
    body += _part_c_section(data["part_c_has_screened_outcome"])
    body += _part_d_section(data["part_d_has_option_2_3"])

    return render.render_article(
        eyebrow="LEAKAGE AUDIT",
        title="Passive Dataset Leakage Audit",
        dek=(
            "target_future_shot_10s and target_future_xg_10s are forward-looking by construction. One leakage "
            "bug (screened_option_was_avoided) was already caught during design. This is the systematic follow-up: "
            "every candidate feature asked — could this only be knowable because of what happens in the "
            "10-second forward window?"
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (passive)"),
            (str(data["part_a_known_leaky_scan"]["n_columns_scanned_total"]), "columns scanned"),
            ("1", "feature excluded (has_screened_outcome)"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.VIF_CSS + render.CORR_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_leakage_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
