"""CLI entrypoint: render reports/analysis/xg_target/LEAKAGE_AUDIT.json as a
self-contained HTML report -- the xG counterpart to LEAKAGE_AUDIT.html.

Usage:
    python -m src.eda.generate_leakage_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "LEAKAGE_AUDIT.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "LEAKAGE_AUDIT.html"


def _ledger_row(label: str, method: str, magnitude_str: str, verdict: str, verdict_class: str) -> str:
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
    return f"""
<div class="test-block">
  <h2 class="test-title">Part A &mdash; known-leaky columns &amp; full name-pattern scan</h2>
  <p class="test-subnote">Target-independent -- reused unchanged from the binary-target audit (never groups by target).</p>
  <div class="corr-ledger" style="margin-bottom:16px;">{''.join(rows)}</div>
</div>"""


def _mini_bar_table(by_value: dict) -> str:
    rows = ""
    max_val = max((v["mean_xg"] for v in by_value.values()), default=1) * 1.15 or 1.0
    for label in ("True", "False"):
        if label not in by_value:
            continue
        v = by_value[label]
        width = max(2, v["mean_xg"] / max_val * 100)
        rows += f"""
<div class="stratum-delta" style="display:grid; grid-template-columns: 60px 1fr 140px; align-items:center; gap:10px; margin-bottom:8px;">
  <span class="mono" style="font-family:'JetBrains Mono',monospace; font-weight:700;">{label}</span>
  <div class="vif-track"><div class="vif-fill severe" style="width:{width:.1f}%"></div></div>
  <span>{v['mean_xg']:.5f} (n={v['n']:,})</span>
</div>"""
    return f'<div style="margin:16px 0;">{rows}</div>'


def _part_b_section(b: dict) -> str:
    return f"""
<div class="test-block">
  <h2 class="test-title">Part B &mdash; <code>screened_option_was_avoided</code> guard, quantified (xG)</h2>
  <p class="test-subnote">Informational: the exclusion is correct on conceptual/structural grounds regardless of this result.</p>
  <div class="qchart-card">
    <h4>mean {esc(b['target_col'])} by {esc(b['column'])}</h4>
    <p class="qc-note">Delta {b['delta']:+.6f}, Welch's t-test p={b['p_value']:.3e}.</p>
    {_mini_bar_table(b['by_value'])}
  </div>
</div>"""


def _part_c_section(c: dict) -> str:
    return f"""
<div class="test-block">
  <h2 class="test-title">Part C &mdash; <code>has_screened_outcome</code>: censoring-mechanism proxy (EXCLUDED, xG)</h2>
  <p class="test-subnote">Structural exclusion holds -- but the empirical strength differs sharply from the binary-target audit. See below.</p>
  <div class="verdict-banner v-yes">
    <b>Verdict: EXCLUDED</b> &mdash; Welch's t-test t={c['t_stat']:.2f}, p={c['p_value']:.3e}. Ratio True:False mean xG
    = {c['ratio_true_to_false']}x.
  </div>
  <div class="finding flag" style="margin:16px 0;">
    <span class="tag">weaker than the binary-target version</span>
    <p>Binary target: ~12x ratio, p=1.5e-31. xG: only <b>{c['ratio_true_to_false']}x</b>, p={c['p_value']:.3e}
    (barely significant). The censoring mechanism doesn't depend on which target is used, so the structural case
    for exclusion still holds -- but don't cite this xG number as equally strong evidence. Likely explanation: xG
    is heavily right-skewed and this group is small (n={c['by_value'].get('False', {}).get('n', '?')}), so the
    mean is noisier and less separated than the binary rate was.</p>
  </div>
  <div class="qchart-card">
    <h4>mean {esc(c['target_col'])} by {esc(c['column'])}</h4>
    {_mini_bar_table(c['by_value'])}
  </div>
  <div class="closing-note">{esc(c['explanation'])}</div>
</div>"""


def _part_d_section(d: dict) -> str:
    cards = ""
    for col, r in d.items():
        cards += f"""
<div class="qchart-card">
  <h4><code>{esc(col)}</code></h4>
  {_mini_bar_table(r['by_value'])}
  <p class="qc-note">{esc(r['explanation'])}</p>
</div>"""
    return f"""
<div class="test-block">
  <h2 class="test-title">Part D &mdash; <code>has_option_2</code> / <code>has_option_3</code> (FLAGGED FOR FOLLOW-UP, xG)</h2>
  <div class="verdict-banner v-partially">
    <b>Verdict: FLAGGED FOR FOLLOW-UP</b> &mdash; same direction as the binary-target audit. Kept as candidate
    features; worth a confound test later, not resolved here.
  </div>
  {cards}
</div>"""


def build_leakage_report_xg(data: dict) -> str:
    body = _part_a_section(data["part_a_known_leaky_scan"])
    body += _part_b_section(data["part_b_screened_option_was_avoided"])
    body += _part_c_section(data["part_c_has_screened_outcome"])
    body += _part_d_section(data["part_d_has_option_2_3"])

    return render.render_article(
        eyebrow="LEAKAGE AUDIT -- XG",
        title="Passive Dataset Leakage Audit (xG)",
        dek=(
            "The continuous-target counterpart to LEAKAGE_AUDIT.html -- same structural checks (Part A reused "
            "unchanged), Parts B-D redone against target_future_xg_10s with a Welch's t-test in place of chi-square."
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
    html = build_leakage_report_xg(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
