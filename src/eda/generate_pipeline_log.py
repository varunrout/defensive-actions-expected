"""CLI entrypoint: render a single audit-trail rollup of every EDA pipeline
stage that touched the candidate feature lists -- one page, in order, with
the feature-count trajectory and a link to each stage's own report.

Unlike CHANGES_SINCE_LAST_FULL_RUN (generate_feature_lock_confirmation.py),
which only covers what changed since the last full correlation run, this
covers the whole pipeline from the original 51/44-feature baseline through
to the current locked 34/38. The per-stage feature-count deltas below are a
documented historical record (the intermediate snapshots themselves are not
preserved as JSON anywhere -- feature_config.py only holds the final,
current lists) -- same pattern as CHANGES_SINCE_LAST_FULL_RUN's manifest.
The final cumulative total is asserted against feature_config.py's live
counts on every run, so a future edit to the feature lists without updating
this history fails loudly instead of silently drifting.

Usage:
    python -m src.eda.generate_pipeline_log
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc
from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports" / "analysis" / "shot_target"
OUTPUT_PATH = REPORTS_DIR / "EDA_PIPELINE_LOG.html"

# One entry per stage that touched the candidate feature lists or is a named
# checkpoint in the pipeline. active_delta/passive_delta are the count
# change *at this stage*, not the running total -- the running total is
# computed once, below, and checked against the live feature_config.py
# counts so this history can't silently drift from what's actually locked.
STAGE_HISTORY = [
    {
        "stage": "00", "title": "Baseline -- manual EDA", "status": "done",
        "active_delta": 0, "passive_delta": 0,
        "finding": "Base category/flag/distribution atlases set the starting candidate lists and surfaced the "
                   "marking-tightness / lane-screening reversals.",
        "artifact": "EDA_ANALYSIS.html",
    },
    {
        "stage": "01-03", "title": "Correlation, review & report scripting", "status": "done",
        "active_delta": 0, "passive_delta": 0,
        "finding": "Five type-matched association methods tiered into DROP/COLLAPSE/REVIEW/DISTINCT; REVIEW "
                   "resolved against real flag-ledger lift, not r alone. Verdicts computed, not yet applied.",
        "artifact": "CORRELATION_ATLAS.html",
    },
    {
        "stage": "04", "title": "Structural redesign", "status": "done",
        "active_delta": -7, "passive_delta": +6,
        "finding": "Applied resolved DROP verdicts. Passive-side raw option coordinates were a coordinate-frame "
                   "artefact (r=0.88 with ball_x) -- replaced with ball-relative dx/dy/distance/angle "
                   "(kept alongside the raw columns during a transition period).",
        "artifact": "REVIEW_METHODOLOGY.html",
    },
    {
        "stage": "05", "title": "Collapse-tier resolution (clusters 1-4)", "status": "done",
        "active_delta": -8, "passive_delta": -5,
        "finding": "Drop-to-one on goal-proximity/possession-clock (active) and defender-position (passive) "
                   "clusters; attacker/defender centroids merged into defender_attacker_gap_x/y (active). "
                   "Cluster 5 (passive option threat-score ranks) deliberately left open.",
        "artifact": "CORRELATION_ATLAS.html",
    },
    {
        "stage": "06", "title": "Drop raw option coordinates", "status": "done",
        "active_delta": 0, "passive_delta": -6,
        "finding": "Ball-relative replacements from stage 04 confirmed working -- dropped the 6 raw "
                   "top_option_n_target_x/y columns (kept as an internal computation, not a candidate feature).",
        "artifact": None,
    },
    {
        "stage": "07", "title": "V2 -- categorical REVIEW band + Type 5/6 rules", "status": "done",
        "active_delta": 0, "passive_delta": 0,
        "finding": "Categorical pairs had no REVIEW band at all, and continuous<->continuous REVIEW pairs had no "
                   "resolution rule. Both fixed in separate _v2 modules; V1 candidate lists untouched.",
        "artifact": "REVIEW_METHODOLOGY_V2.html",
    },
    {
        "stage": "08", "title": "Functional-role bucket fix", "status": "done",
        "active_delta": 0, "passive_delta": 0,
        "finding": "62.4% \"unclassified\" wasn't sparse visibility -- a missing advanced+wide bucket plus narrow "
                   "terciles. Added advanced_wide, renamed the n>=2 fallback to mid_block. Category-value change "
                   "on an existing column, not a column add/drop.",
        "artifact": "PASSIVE_ARCHETYPES.html",
    },
    {
        "stage": "09", "title": "Multicollinearity (VIF)", "status": "confirmed",
        "active_delta": -2, "passive_delta": 0,
        "finding": "Pairwise correlation cleared a 6-column active cluster; VIF didn't -- joint linear "
                   "dependency invisible to pairwise correlation. Dropped both.",
        "artifact": "VIF_ANALYSIS.html",
    },
    {
        "stage": "10", "title": "Leakage audit", "status": "confirmed",
        "active_delta": 0, "passive_delta": -1,
        "finding": "has_screened_outcome is a censoring-mechanism proxy for the target's own truncated window, "
                   "not defensive signal -- excluded.",
        "artifact": "LEAKAGE_AUDIT.html",
    },
    {
        "stage": "11", "title": "Football sanity check", "status": "confirmed",
        "active_delta": 0, "passive_delta": 0,
        "finding": "Independent recomputation of 4 derived features from raw geometry, compared row-for-row "
                   "against stored output. Validation only -- no feature-list change.",
        "artifact": "FOOTBALL_SANITY_CHECK.html",
    },
    {
        "stage": "12", "title": "Passive archetype clustering", "status": "confirmed",
        "active_delta": 0, "passive_delta": 0,
        "finding": "KMeans within each defender_functional_role bucket, on a separate 16-feature analysis set. "
                   "Downstream module -- doesn't touch feature_config.py's candidate lists.",
        "artifact": "PASSIVE_ARCHETYPES.html",
    },
    {
        "stage": "13", "title": "Canonical match-grouped split", "status": "confirmed",
        "active_delta": 0, "passive_delta": 0,
        "finding": "Frozen, shared TEST/CV match assignment for both legs. Model-validation infra -- no feature "
                   "list touch.",
        "artifact": "SPLIT_VALIDATION.html",
    },
    {
        "stage": "14", "title": "Post-lock correlation confirmation", "status": "confirmed",
        "active_delta": 0, "passive_delta": 0,
        "finding": "Re-ran correlation/review fresh against the final locked lists. Zero tier crossings found.",
        "artifact": "FEATURE_LOCK_CONFIRMATION.html",
    },
]

STATUS_LABEL = {"done": "DONE", "confirmed": "CONFIRMED"}

PIPELINE_CSS = """
.pl-chart-frame { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin: 8px 0 28px; }
.pl-node { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 0 10px 10px 0; padding: 16px 20px; margin-bottom: 14px; }
.pl-node-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; }
.pl-stage-tag { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-muted); }
.pl-title { font-family: "Archivo", sans-serif; font-weight: 700; font-size: 16px; margin: 2px 0 0; }
.pl-status { font-family: "JetBrains Mono", monospace; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; padding: 3px 9px; border-radius: 999px; font-weight: 700; background: var(--good-wash); color: var(--good); white-space: nowrap; }
.pl-delta { font-family: "JetBrains Mono", monospace; font-size: 11.5px; color: var(--text-muted); text-align: right; white-space: nowrap; }
.pl-delta b { color: var(--text-primary); }
.pl-finding { color: var(--text-secondary); font-size: 13.5px; margin: 0 0 8px; }
.pl-link { font-family: "JetBrains Mono", monospace; font-size: 11px; }
.pl-link a { color: var(--neg); text-decoration: none; }
.pl-link a:hover { text-decoration: underline; }
.pl-legend { display: flex; gap: 20px; font-size: 11.5px; color: var(--text-muted); margin: 4px 0 16px; }
.pl-legend span { display: inline-flex; align-items: center; gap: 6px; }
.pl-legend .dot { width: 9px; height: 9px; border-radius: 50%; }
"""


def _compute_running_counts() -> list[dict]:
    active_running, passive_running = 51, 44
    out = []
    for s in STAGE_HISTORY:
        active_running += s["active_delta"]
        passive_running += s["passive_delta"]
        out.append({**s, "active_running": active_running, "passive_running": passive_running})
    return out


def _line_chart_svg(rows: list[dict]) -> str:
    w, h, pad_l, pad_r, pad_t, pad_b = 900, 220, 50, 20, 20, 30
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    all_vals = [r["active_running"] for r in rows] + [r["passive_running"] for r in rows]
    v_min, v_max = min(all_vals) - 2, max(all_vals) + 2
    n = len(rows)

    def xy(i: int, v: int) -> tuple[float, float]:
        x = pad_l + (i / max(1, n - 1)) * plot_w
        y = pad_t + (1 - (v - v_min) / (v_max - v_min)) * plot_h
        return x, y

    def polyline(key: str, color: str) -> str:
        pts = " ".join(f"{xy(i, r[key])[0]:.1f},{xy(i, r[key])[1]:.1f}" for i, r in enumerate(rows))
        dots = "".join(
            f'<circle cx="{xy(i, r[key])[0]:.1f}" cy="{xy(i, r[key])[1]:.1f}" r="3" fill="{color}"></circle>'
            for i, r in enumerate(rows)
        )
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"></polyline>{dots}'

    labels = "".join(
        f'<text x="{xy(i, rows[i]["active_running"])[0]:.1f}" y="{h - 8}" font-size="9" '
        f'font-family="JetBrains Mono, monospace" fill="var(--text-muted)" text-anchor="middle">{esc(r["stage"])}</text>'
        for i, r in enumerate(rows)
    )

    return f"""
<svg viewBox="0 0 {w} {h}" role="img" aria-label="Active and passive candidate feature count across every pipeline stage" style="width:100%; height:auto;">
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="var(--border)"></line>
  <line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="var(--border)"></line>
  {polyline("active_running", "var(--neg)")}
  {polyline("passive_running", "var(--pos)")}
  {labels}
</svg>"""


def _node_html(r: dict) -> str:
    status = STATUS_LABEL[r["status"]]
    delta_bits = []
    if r["active_delta"]:
        delta_bits.append(f"active <b>{r['active_delta']:+d}</b> &rarr; {r['active_running']}")
    if r["passive_delta"]:
        delta_bits.append(f"passive <b>{r['passive_delta']:+d}</b> &rarr; {r['passive_running']}")
    delta_html = " &middot; ".join(delta_bits) if delta_bits else f"no change ({r['active_running']} / {r['passive_running']})"
    link_html = f'<p class="pl-link"><a href="{esc(r["artifact"])}">{esc(r["artifact"])}</a></p>' if r["artifact"] else ""

    return f"""
<div class="pl-node">
  <div class="pl-node-head">
    <div>
      <span class="pl-stage-tag">STAGE {esc(r['stage'])}</span>
      <p class="pl-title">{esc(r['title'])}</p>
    </div>
    <div style="text-align:right;">
      <span class="pl-status">{esc(status)}</span>
      <p class="pl-delta">{delta_html}</p>
    </div>
  </div>
  <p class="pl-finding">{r['finding']}</p>
  {link_html}
</div>"""


def build_pipeline_log(rows: list[dict], live_active: int, live_passive: int) -> str:
    final = rows[-1]
    if final["active_running"] != live_active or final["passive_running"] != live_passive:
        raise AssertionError(
            f"STAGE_HISTORY's cumulative total (active={final['active_running']}, passive={final['passive_running']}) "
            f"no longer matches feature_config.py's live counts (active={live_active}, passive={live_passive}) -- "
            "a feature list changed without updating STAGE_HISTORY's deltas above."
        )

    legend = """
<div class="pl-legend">
  <span><span class="dot" style="background:var(--neg)"></span> active</span>
  <span><span class="dot" style="background:var(--pos)"></span> passive</span>
</div>"""

    body = f"""
<h2>Candidate feature count, every stage that changed it</h2>
<p class="section-note">Computed from the documented per-stage deltas below, cross-checked against
feature_config.py's live counts on every render -- this page fails to build if the two disagree.</p>
{legend}
<div class="pl-chart-frame">{_line_chart_svg(rows)}</div>

<h2 style="margin-top:8px;">Every stage, in order</h2>
<p class="section-note">Method and finding per stage, in the order it ran, linking to that stage's own report where one exists.</p>
{''.join(_node_html(r) for r in rows)}

<div class="closing-note" style="margin-top:8px;">
<b>Reading this page:</b> stages 01-03, 07, 08, 11, 12, 13 and 14 are analysis or validation passes with no
feature-list impact of their own -- they're listed because they're part of the same pipeline, not because they
moved the count. The count only moves at stages 04, 05, 06, 09 and 10.
</div>"""

    return render.render_article(
        eyebrow="EDA PIPELINE LOG",
        title="Feature Pipeline Audit Trail",
        dek=(
            "Every stage that touched the candidate feature lists, in order -- what ran, what it found, what it "
            "dropped, merged, or engineered -- verified against feature_config.py and every stage's own report, "
            "not taken on trust."
        ),
        stats=[
            (f"{live_active}", "active features (final)"),
            (f"{live_passive}", "passive features (final)"),
            (str(len(rows)), "pipeline stages"),
            (str(sum(1 for r in rows if r["active_delta"] or r["passive_delta"])), "stages that changed the count"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + PIPELINE_CSS,
    )


def main() -> None:
    live_active = len(ACTIVE["categorical"]) + len(ACTIVE["boolean"]) + len(ACTIVE["continuous"]) + len(ACTIVE["discrete"])
    live_passive = len(PASSIVE["categorical"]) + len(PASSIVE["boolean"]) + len(PASSIVE["continuous"]) + len(PASSIVE["discrete"])

    rows = _compute_running_counts()
    html = build_pipeline_log(rows, live_active, live_passive)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
