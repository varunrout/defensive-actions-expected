"""CLI entrypoint: render reports/analysis/xt_target/LEAKAGE_AUDIT.json as a
self-contained HTML report -- the xT sibling of generate_leakage_report_xg.py.

Same visual system (render.CONFOUND_CSS + VIF_CSS + CORR_CSS, the same
`test-block` / `qchart-card` / `verdict-banner` markup). The scope
difference from its xg counterpart -- active instead of passive, and three
of the xg audit's four parts having no active counterpart -- is stated in
the report body, not hidden.

Usage:
    python -m src.eda.generate_leakage_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc, finding_card

INPUT_PATH = xc.OUT_DIR / "LEAKAGE_AUDIT.json"
OUTPUT_PATH = xc.OUT_DIR / "LEAKAGE_AUDIT.html"


def _ledger_row(label: str, method: str, magnitude: str, verdict: str, cls: str) -> str:
    return f"""
<div class="corr-row">
  <span class="corr-pair">{label}</span>
  <span class="corr-method">{esc(method)}</span>
  <span></span>
  <span class="corr-value">{magnitude}</span>
  <span class="verdict-chip {cls}">{esc(verdict)}</span>
</div>"""


def _part_a_section(a: dict) -> str:
    rows = [
        _ledger_row(f"<code>{esc(col)}</code>", "presence check",
                    "not in candidate list" if not info["in_candidate_list"] else "IN CANDIDATE LIST",
                    info["verdict"].upper(), "distinct" if not info["in_candidate_list"] else "drop")
        for col, info in a["known_leaky_columns"].items()
    ]
    rows.append(_ledger_row(
        f"{a['n_columns_scanned_total']} columns scanned for future/next/avoided/outcome/after",
        "name-pattern scan",
        f"{len(a['all_leakage_suggestive_column_names'])} flagged by name",
        "CLEAN" if a["verdict"] == "clean" else "ISSUES FOUND",
        "distinct" if a["verdict"] == "clean" else "drop",
    ))
    return f"""
<div class="test-block">
  <h2 class="test-title">Part A &mdash; known-leaky columns &amp; full name-pattern scan (ACTIVE)</h2>
  <p class="test-subnote">Target- and dataset-independent -- <code>generate_leakage_audit.part_a</code> is
  imported and reused <b>unchanged</b>, pointed at the active parquet and the active candidate list. It never
  groups by a target, so nothing about this target changes it.</p>
  <div class="corr-ledger" style="margin-bottom:16px;">{''.join(rows)}</div>
</div>"""


def _dir_table(by_value: dict) -> str:
    rows = ""
    max_abs = max((abs(v["mean_xt"]) for v in by_value.values()), default=1e-9) or 1e-9
    for label in ("True", "False"):
        v = by_value.get(label)
        if not v:
            continue
        w = min(50.0, abs(v["mean_xt"]) / max_abs * 50)
        side = "pos" if v["mean_xt"] >= 0 else "neg"
        rows += f"""
<div style="display:grid; grid-template-columns: 60px 1fr 260px; align-items:center; gap:10px; margin-bottom:8px;">
  <span style="font-family:'JetBrains Mono',monospace; font-weight:700;">{label}</span>
  <div class="vif-track" style="position:relative;">
    <span style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--text-primary);opacity:0.4;"></span>
    <div class="vif-fill severe" style="width:{w:.1f}%; position:absolute; {'left' if side == 'pos' else 'right'}:50%;"></div>
  </div>
  <span style="font-family:'JetBrains Mono',monospace;font-size:11.5px;">{v['mean_xt']:+.6f} (n={v['n']:,})
  &middot; {v['pct_zero']:.1f}% zero &middot; {v['pct_negative']:.1f}% neg &middot; {v['pct_positive']:.1f}% pos</span>
</div>"""
    return f'<div style="margin:16px 0;">{rows}</div>'


def _missing_parts_section(d: dict) -> str:
    rows = "".join(f"<li><code>{esc(k)}</code> &mdash; {esc(v)}</li>" for k, v in d.items())
    return f"""
<div class="test-block">
  <h2 class="test-title">Parts B, C and D of the xg audit &mdash; NO ACTIVE COUNTERPART</h2>
  <div class="verdict-banner v-partially">
  <b>Stated plainly rather than substituted for silently.</b> The xg portal's leakage audit is
  <b>passive-dataset-only</b>. Its three quantified parts test columns that do not exist in the active parquet:
  </div>
  <ul style="font-size:13px;line-height:1.7;">{rows}</ul>
  <p class="test-subnote">No active-side leakage audit exists anywhere in this repo, for any target -- checked
  directly across <code>reports/analysis/shot_target/</code> and <code>reports/analysis/xg_target/</code>, each
  of which holds a single passive-only LEAKAGE_AUDIT. The parts below apply the <b>same method</b> (groupby mean
  target + Welch's t-test) to the active dataset's own analogous columns, and are numbered B&prime;/C&prime;/D&prime;
  to make clear they are substitutes, not the original parts.</p>
</div>"""


def _part_b_prime(b: dict) -> str:
    return f"""
<div class="test-block">
  <h2 class="test-title">Part B&prime; &mdash; <code>has_previous_event</code>, the active censoring proxy</h2>
  <p class="test-subnote">Substitutes for: {esc(b['substitutes_for'])}. Method: {esc(b['method'])}.</p>
  <div class="verdict-banner v-partially"><b>Verdict: {esc(b['verdict'].upper())}</b></div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">the effect is in the zero mass, not the mean</span>
  <p>Welch's t-test on the MEAN gives p={b['p_value']:.3e} &mdash; not significant at any conventional level.
  But the exactly-zero share differs by <b>{b['zero_share_gap_pp']:+.1f}pp</b>
  ({b['by_value']['False']['pct_zero']:.1f}% when False vs {b['by_value']['True']['pct_zero']:.1f}% when True).
  <b>The xg audit's method, applied literally, would have missed this.</b> That is itself a finding about
  methodology transfer: a mean-difference test is the right instrument for a strictly non-negative,
  zero-inflated target whose signal lives in its centre, and the wrong one for a symmetric target whose
  structural mass sits exactly at zero.</p>
  </div>
  <div class="qchart-card">
    <h4>mean {esc(xc.TARGET)} by <code>{esc(b['column'])}</code></h4>
    <p class="qc-note">Delta {b['delta']:+.6f}, Welch's t={b['t_stat']:.3f}, p={b['p_value']:.3e}. Bars diverge
    from zero; green = xT fell (threat reduced), red = xT rose.</p>
    {_dir_table(b['by_value'])}
  </div>
  <div class="closing-note">{esc(b['explanation'])}</div>
</div>"""


def _part_c_prime(c: dict) -> str:
    cards = ""
    for col, r in c.items():
        cards += f"""
<div class="qchart-card">
  <h4><code>{esc(col)}</code></h4>
  <p class="qc-note"><b>feature_config.py exclusion reason:</b> {esc(str(r['feature_config_exclusion_reason']))}</p>
  {_dir_table(r['by_value'])}
  <p class="qc-note">{esc(r['explanation'])}</p>
</div>"""
    return f"""
<div class="test-block">
  <h2 class="test-title">Part C&prime; &mdash; the three <code>action_*_possession</code> flags</h2>
  <p class="test-subnote">Substitutes for: Part D of the passive xg audit. Same method.</p>
  <div class="verdict-banner v-partially">
  <b>Verdict: the "structural zero" exclusion justification does NOT transfer to this target</b> &mdash; flagged
  as an observation, not reopened. These three columns stay excluded in feature_config.py; this report does not
  change it.
  </div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">carried forward from Prompt 64</span>
  <p>This reproduces Prompt 64's central result on the full active dataset rather than only inside the two
  slices it examined. Against <code>target_future_shot_10s</code> these rows are 100% identical zeros; against
  <code>{esc(xc.TARGET)}</code> roughly 77-79% of them are non-zero with real row-level spread. The
  degenerate-zero collapse the xT target was built to fix is, on this evidence, fixed.</p>
  </div>
  {cards}
</div>"""


def _part_d_prime(d: dict) -> str:
    rows = "".join(
        f"<tr><td><code>{esc(r['feature'])}</code></td><td>{r['r_vs_xt_after']:+.4f}</td>"
        f"<td>{r['r_vs_xt_before']:+.4f}</td><td>{r['r_vs_target']:+.4f}</td></tr>"
        for r in d["locked_feature_correlations"]
    )
    constr = "".join(
        f"<tr><td><code>{esc(k)}</code></td>"
        f"<td>{'IN CANDIDATE LIST' if v['in_candidate_list'] else 'excluded'}</td>"
        f"<td>{v['r_vs_xt_after']:+.4f}</td><td>{esc(str(v['exclusion_reason']))}</td></tr>"
        for k, v in d["target_construction_columns"].items()
    )
    s = d["strongest"]
    return f"""
<div class="test-block">
  <h2 class="test-title">Part D&prime; &mdash; construction coupling (NEW &mdash; no xg counterpart)</h2>
  <div class="verdict-banner v-yes"><b>Verdict: {esc(d['verdict'].upper())}</b></div>
  {finding_card("method is new, and labelled as new", "not presented as the established audit", esc(d['method']))}
  <p class="test-subnote">{esc(d['why_this_part_exists'])}</p>

  <h3 style="margin-top:22px;">The target's own construction columns</h3>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Column</th><th>Candidate status</th><th>r vs xt_after</th><th>Exclusion reason (feature_config.py)</th></tr>
  {constr}
  </table></div>

  <h3 style="margin-top:22px;">Locked numerical features vs the target's two terms</h3>
  <p class="test-subnote">Ranked by |r vs target|. Strongest: <code>{esc(s['feature'])}</code> at
  <b>{s['r_vs_target']:+.4f}</b> against the target and <b>{s['r_vs_xt_after']:+.4f}</b> against
  <code>xt_after</code> specifically.</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Locked feature</th><th>r vs xt_after</th><th>r vs xt_before</th><th>r vs target</th></tr>
  {rows}
  </table></div>

  <div class="closing-note" style="margin-top:18px;">{esc(d['explanation'])}</div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">carried forward from Prompt 64 -- Clearance / action_x</span>
  <p>{esc(d['clearance_link'])}</p>
  <p style="margin-top:8px;">{esc(d['prompt_64_cross_reference'])}</p>
  </div>
</div>"""


def build_report(data: dict) -> str:
    body = f"""
<div class="rule-banner"><b>Scope, confirmed by reading the source.</b> {esc(data['step_0_scope_note'])}</div>
"""
    body += _part_a_section(data["part_a_known_leaky_scan"])
    body += _missing_parts_section(data["parts_with_no_active_counterpart"])
    body += _part_b_prime(data["part_b_prime_has_previous_event"])
    body += _part_c_prime(data["part_c_prime_action_possession_flags"])
    body += _part_d_prime(data["part_d_prime_construction_coupling"])

    return render.render_article(
        eyebrow="LEAKAGE AUDIT -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="Active Dataset Leakage Audit (xT delta)",
        dek=(
            "Part A's method reused unchanged; the xg audit's Parts B/C/D have no active counterpart and that is "
            "said so rather than substituted for silently; Parts B'/C' apply the same method to the active "
            "dataset's own analogous columns; Part D' is a new construction-coupling check specific to this "
            "target. Active-binary leg only."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (active)"),
            (str(data["part_a_known_leaky_scan"]["n_columns_scanned_total"]), "columns scanned"),
            ("3", "xg parts with no active counterpart"),
            ("1", "new part (construction coupling)"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.VIF_CSS + render.CORR_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
