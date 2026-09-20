"""CLI entrypoint: render the v2 LEAKAGE_AUDIT.json as HTML.

STEP-0 DISPOSITION: REWRITE (renderer half of the Part D' rewrite).

`generate_leakage_report_xt.py` cannot render the v2 JSON at all: Part D'
was rewritten, so its key set changed (`target_construction_columns` ->
`v1_construction_columns_action_x_y` plus the new
`v2_construction_coupling_moved_to_flags` and `v1_comparison`;
`why_this_part_exists` -> `why_this_part_changed`;
`prompt_64_cross_reference` -> `prompt_66_cross_reference`). Its Part C'
block also hardcodes "roughly 77-79% of them are non-zero", a v1 figure,
and its Part B' block describes the target as "symmetric", which v2 is not.

Part A, the missing-parts section, the shared row/table helpers and the
whole visual system are imported and reused unchanged from the v1 renderer.

Usage:
    python -m src.eda.generate_leakage_report_xt_v2
"""

from __future__ import annotations

import json

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import render, xt_common as xc
from src.eda import generate_leakage_report_xt as lr1
from src.eda.render import esc, finding_card

INPUT_PATH = xc.OUT_DIR / "LEAKAGE_AUDIT.json"
OUTPUT_PATH = xc.OUT_DIR / "LEAKAGE_AUDIT.html"


def _part_b_prime_v2(b: dict) -> str:
    return f"""
<div class="test-block">
  <h2 class="test-title">Part B&prime; &mdash; <code>has_previous_event</code>, the active censoring proxy</h2>
  <p class="test-subnote">Substitutes for: {esc(b['substitutes_for'])}. Method: {esc(b['method'])}.</p>
  <div class="verdict-banner v-partially"><b>Verdict: {esc(b['verdict'].upper())}</b></div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">the zero-mass caveat survives from v1 to v2 intact</span>
  <p>The exactly-zero share differs by <b>{b['zero_share_gap_pp']:+.1f}pp</b>
  ({b['by_value']['False']['pct_zero']:.1f}% when False vs {b['by_value']['True']['pct_zero']:.1f}% when True),
  alongside a mean difference of {b['delta']:+.6f} (Welch p={b['p_value']:.3e}). The v1 edition of this report
  recorded this as its Caveat 2: on v1 the mean test alone returned p=0.0645 and would have reported nothing at
  all, while the zero share differed by +27.3pp. <b>That caveat is not retired by the v2 correction.</b> A
  mean-difference test is the right instrument for a strictly non-negative, zero-inflated target whose signal
  lives in its centre, and the wrong one for a target with {esc(f"{b['by_value']['True']['pct_zero']:.1f}")}%
  of one group's mass sitting exactly at zero. v2's exactly-zero share is <i>larger</i> than v1's (20.2% vs
  11.3% overall), so if anything the instrument matters more here.</p>
  </div>
  <div class="qchart-card">
    <h4>mean {esc(xc.TARGET)} by <code>{esc(b['column'])}</code></h4>
    <p class="qc-note">Delta {b['delta']:+.6f}, Welch's t={b['t_stat']:.3f}, p={b['p_value']:.3e}. Bars diverge
    from zero; green = xT fell (threat reduced), red = xT rose.</p>
    {lr1._dir_table(b['by_value'])}
  </div>
  <div class="closing-note">{esc(b['explanation'])}</div>
</div>"""


def _part_c_prime_v2(c: dict) -> str:
    cards = ""
    for col, r in c.items():
        collapse = ""
        if r.get("collapses_to_xt_before"):
            collapse = (
                f'<p class="qc-note"><b>Construction identity, verified on this run.</b> Every row in this '
                f'flag\'s True group has <code>{esc(xc.TARGET)}</code> exactly equal to its own '
                f'<code>xt_before</code> (mean {r["xt_before_mean_same_slice"]:+.6f}, <code>np.allclose</code>) '
                f'&mdash; because this flag is the deterministic trigger for <code>xt_after = 0</code>.</p>'
            )
        cards += f"""
<div class="qchart-card">
  <h4><code>{esc(col)}</code></h4>
  <p class="qc-note"><b>feature_config.py exclusion reason:</b> {esc(str(r['feature_config_exclusion_reason']))}</p>
  {lr1._dir_table(r['by_value'])}
  {collapse}
  <p class="qc-note">{esc(r['explanation'])}</p>
</div>"""
    return f"""
<div class="test-block">
  <h2 class="test-title">Part C&prime; &mdash; the three <code>action_*_possession</code> flags</h2>
  <p class="test-subnote">Substitutes for: Part D of the passive xg audit. Same method.</p>
  <div class="verdict-banner v-partially">
  <b>Verdict: the "structural zero" exclusion justification still does NOT transfer &mdash; but on v2 a stronger,
  different reason to exclude appears.</b> These three columns stay excluded in feature_config.py; this report
  does not change it.
  </div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">carried forward from Prompt 66 &mdash; CHARACTER CHANGED</span>
  <p>The v1 edition of this report read these flags as vindication: against
  <code>target_future_shot_10s</code> they are 100% identical zeros, while against v1's xT target "roughly
  77-79% of them are non-zero with real spread", so the degenerate-zero collapse the xT target was built to fix
  looked fixed. <b>That reading does not carry over, and the v1 figure is v1's.</b> Under the corrected target,
  <code>action_ended_possession</code> is the flag that TRIGGERS <code>xt_after = 0</code>, so its True group's
  delta equals <code>xt_before</code> by construction &mdash; a deterministic identity, not a measurement.</p>
  <p style="margin-top:8px;">{esc(v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING)}</p>
  <p style="margin-top:8px;">The practical consequence points the same way as before but for a firmer reason:
  these columns should stay out of the candidate list. A column that deterministically sets one term of the
  target is a construction input, and feeding it to a model would be circular.</p>
  </div>
  {cards}
</div>"""


def _part_d_prime_v2(d: dict) -> str:
    rows = "".join(
        f"<tr><td><code>{esc(r['feature'])}</code></td><td>{r['r_vs_xt_after']:+.4f}</td>"
        f"<td>{r['r_vs_xt_before']:+.4f}</td><td>{r['r_vs_target']:+.4f}</td></tr>"
        for r in d["locked_feature_correlations"]
    )
    old = "".join(
        f"<tr><td><code>{esc(k)}</code></td>"
        f"<td>{'IN CANDIDATE LIST' if v['in_candidate_list'] else 'excluded'}</td>"
        f"<td>{v['r_vs_xt_after']:+.4f}</td><td>{esc(str(v['exclusion_reason']))}</td></tr>"
        for k, v in d["v1_construction_columns_action_x_y"].items()
    )
    flags = "".join(
        f"<tr><td><code>{esc(k)}</code></td>"
        f"<td>{'IN CANDIDATE LIST' if v['in_candidate_list'] else 'excluded'}</td>"
        f"<td>{v['r_vs_xt_after']:+.4f}</td><td>{v['r_vs_target']:+.4f}</td></tr>"
        for k, v in d["v2_construction_coupling_moved_to_flags"]["columns"].items()
    )
    s = d["strongest"]
    vc = d["v1_comparison"]
    return f"""
<div class="test-block">
  <h2 class="test-title">Part D&prime; &mdash; construction coupling, RE-MEASURED for v2</h2>
  <div class="verdict-banner v-yes"><b>Verdict: {esc(d['verdict'].upper())}</b></div>
  {finding_card("this part was rewritten, not re-pointed",
                "its v1 premise is false under the corrected target",
                esc(d['why_this_part_changed']), flag=True)}
  <p class="test-subnote">{esc(d['method'])}</p>

  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">v1 &rarr; v2: Prompt 65's most important caveat, substantially retired</span>
  <p>Strongest locked-feature correlation with the target: <code>{esc(vc['v1_strongest_feature'])}</code> at
  <b>{vc['v1_r_vs_target']:+.4f}</b> under v1 &rarr; <code>{esc(vc['v2_strongest_feature'])}</code> at
  <b>{vc['v2_r_vs_target']:+.4f}</b> under v2, a <b>{vc['abs_reduction_pct']:.0f}% reduction</b> in absolute
  magnitude.</p>
  <p style="margin-top:8px;">{esc(vc['note'])}</p>
  </div>

  <h3 style="margin-top:22px;">v1's construction columns &mdash; no longer part of the construction</h3>
  <p class="test-subnote">Under v1 these two <i>were</i> the <code>xt_after</code> lookup. Under v2 they are not
  used in the target's construction at all. Both remain out of the candidate list regardless &mdash; confirmed
  here, not assumed.</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Column</th><th>Candidate status</th><th>r vs xt_after</th><th>Exclusion reason (feature_config.py)</th></tr>
  {old}
  </table></div>

  <h3 style="margin-top:22px;">Where the coupling moved: from a location term to a flag</h3>
  <p class="test-subnote">{esc(d['v2_construction_coupling_moved_to_flags']['note'])}</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Column</th><th>Candidate status</th><th>r vs xt_after</th><th>r vs target</th></tr>
  {flags}
  </table></div>

  <h3 style="margin-top:22px;">Locked numerical features vs the target's two terms</h3>
  <p class="test-subnote">Ranked by |r vs target|. Strongest: <code>{esc(s['feature'])}</code> at
  <b>{s['r_vs_target']:+.4f}</b> against the target and <b>{s['r_vs_xt_after']:+.4f}</b> against
  <code>xt_after</code> specifically. Note how much higher the same feature sits against
  <code>xt_before</code> ({s['r_vs_xt_before']:+.4f}) &mdash; that term is unchanged from v1 and is where the
  residual, expected location relationship now lives.</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Locked feature</th><th>r vs xt_after</th><th>r vs xt_before</th><th>r vs target</th></tr>
  {rows}
  </table></div>

  <div class="closing-note" style="margin-top:18px;">{esc(d['explanation'])}</div>
  <div class="finding flag" style="margin:16px 0;">
  <span class="tag">carried forward from Prompt 66 &mdash; Clearance reversal, same root cause</span>
  <p>{esc(d['clearance_link'])}</p>
  <p style="margin-top:8px;">{esc(d['prompt_66_cross_reference'])}</p>
  </div>
</div>"""


def build_report(data: dict) -> str:
    d = data["part_d_prime_construction_coupling"]
    body = f"""
<div class="rule-banner"><b>This report supersedes a prior pass.</b> {esc(data['v2_rebuild_note'])}</div>
<div class="rule-banner"><b>Scope, confirmed by reading the source.</b> {esc(data['step_0_scope_note'])}</div>
"""
    body += lr1._part_a_section(data["part_a_known_leaky_scan"])
    body += lr1._missing_parts_section(data["parts_with_no_active_counterpart"])
    body += _part_b_prime_v2(data["part_b_prime_has_previous_event"])
    body += _part_c_prime_v2(data["part_c_prime_action_possession_flags"])
    body += _part_d_prime_v2(d)

    return render.render_article(
        eyebrow="LEAKAGE AUDIT -- XT DELTA V2 &middot; ACTIVE-BINARY LEG ONLY",
        title="Active Dataset Leakage Audit (xT delta v2)",
        dek=(
            "Part A's method reused unchanged; the xg audit's Parts B/C/D have no active counterpart and that is "
            "said so rather than substituted for silently; Parts B'/C' apply the same method to the active "
            "dataset's own analogous columns. Part D' was REWRITTEN for v2 -- its v1 premise (xt_after looked up "
            "from action_x/action_y) is false under the corrected target. Active-binary leg only."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows (active)"),
            (str(data["part_a_known_leaky_scan"]["n_columns_scanned_total"]), "columns scanned"),
            ("3", "xg parts with no active counterpart"),
            (f"{d['v1_comparison']['abs_reduction_pct']:.0f}%", "construction coupling reduced vs v1"),
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
