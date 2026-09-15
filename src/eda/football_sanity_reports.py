"""CLI entrypoint: render FOOTBALL_SANITY_CHECK.json as a self-contained
HTML report, in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.football_sanity_reports
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "FOOTBALL_SANITY_CHECK.json"
PARQUET_PATH = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "FOOTBALL_SANITY_CHECK.html"

# Role swatches -- deliberately siblings of the canonical palette, not reuses
# of --accent/--pos/--neg/--amber/--good, since those are reserved for
# pass/fail semantics elsewhere on the page.
ROLE_COLORS = {
    "last_line": "#5563c1",
    "wide_cover": "#1f9e93",
    "central_screen": "#7c5cbf",
    "advanced_wide": "#b98a3d",
    "mid_block": "#6b7280",
    "unclassified": "#c14f8f",
}

FOOTBALL_CSS = """
.results-table-wrap { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 8px; }
.results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.results-table thead th {
  font-family: "JetBrains Mono", monospace; text-transform: uppercase; font-size: 10px;
  letter-spacing: 0.04em; color: var(--text-muted); text-align: left; padding: 9px 14px;
  background: var(--plane); border-bottom: 1px solid var(--text-primary);
}
.results-table td { padding: 10px 14px; border-bottom: 1px solid var(--border); color: var(--text-secondary); vertical-align: top; background: var(--surface); }
.results-table tr:last-child td { border-bottom: none; }
.results-table td:first-child { display: flex; align-items: center; gap: 9px; color: var(--text-primary); font-weight: 600; }
.check-row-icon { width: 14px; height: 14px; border-radius: 50%; background: var(--good); flex: none; display: inline-block; }
.check-row-icon.bad { background: var(--pos); }
.results-table td.result { font-family: "JetBrains Mono", monospace; font-weight: 700; text-align: right; }
.results-table td.result.ok { color: var(--good); }
.results-table td.result.bad { color: var(--pos); }
.results-table code { font-family: "JetBrains Mono", monospace; font-size: 0.92em; background: var(--plane); border: 1px solid var(--border); border-radius: 4px; padding: 1px 5px; }

.fig-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 16px 0 8px; }
@media (max-width: 760px) { .fig-grid { grid-template-columns: 1fr; } }
.fig-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.fig-card.wide { grid-column: 1 / -1; }
.pitch-svg { background: var(--plane); border-radius: 6px; display: block; width: 100%; height: auto; }
figcaption { border-top: 1px dashed var(--gridline); margin-top: 12px; padding-top: 10px; font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-muted); text-align: left; line-height: 1.6; }
figcaption b { color: var(--text-primary); font-family: "Public Sans", sans-serif; font-weight: 600; }

.role-legend { display: flex; flex-wrap: wrap; gap: 6px 4px; margin: 14px 0 22px; }
.lg-chip { display: inline-flex; align-items: center; gap: 6px; font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-secondary); background: var(--plane); border: 1px solid var(--border); border-radius: 999px; padding: 4px 11px 4px 8px; }
.lg-dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
"""

# Pitch geometry: 0-120 x 0-80, defending goal at x=0.
PITCH_W, PITCH_H = 120.0, 80.0
PAD = 8.0
SVG_W = 640
SVG_H = SVG_W * ((PITCH_H + 2 * PAD) / (PITCH_W + 2 * PAD))


def _px(x: float) -> float:
    return (x + PAD) / (PITCH_W + 2 * PAD) * SVG_W


def _py(y: float) -> float:
    return (y + PAD) / (PITCH_H + 2 * PAD) * SVG_H


def _pitch_background() -> str:
    x0, y0 = _px(0), _py(0)
    x1, y1 = _px(PITCH_W), _py(PITCH_H)
    mid_x = _px(PITCH_W / 2)
    return f"""
<rect x="{_px(-PAD):.1f}" y="{_py(-PAD):.1f}" width="{SVG_W}" height="{SVG_H}" fill="var(--plane)" />
<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" fill="none" stroke="var(--text-muted)" stroke-width="1.5" />
<line x1="{mid_x:.1f}" y1="{y0:.1f}" x2="{mid_x:.1f}" y2="{y1:.1f}" stroke="var(--text-muted)" stroke-width="1" />
<circle cx="{mid_x:.1f}" cy="{_py(PITCH_H / 2):.1f}" r="{(_px(9.15) - _px(0)):.1f}" fill="none" stroke="var(--text-muted)" stroke-width="1" />
<rect x="{_px(0):.1f}" y="{_py(18):.1f}" width="{(_px(18) - _px(0)):.1f}" height="{(_py(62) - _py(18)):.1f}" fill="none" stroke="var(--text-muted)" stroke-width="1" />
<rect x="{(_px(120) - (_px(18) - _px(0))):.1f}" y="{_py(18):.1f}" width="{(_px(18) - _px(0)):.1f}" height="{(_py(62) - _py(18)):.1f}" fill="none" stroke="var(--text-muted)" stroke-width="1" />
"""


def _pitch_figure(rows: pd.DataFrame, ball_x: float, ball_y: float, event_id: str, title: str, caption: str, lane_rank: int | None = 1, wide: bool = False) -> str:
    elements = [_pitch_background()]

    for rank in (1, 2, 3):
        tx_col, ty_col = f"top_option_{rank}_target_x", f"top_option_{rank}_target_y"
        if tx_col not in rows.columns:
            continue
        tx, ty = rows.iloc[0].get(tx_col), rows.iloc[0].get(ty_col)
        if pd.isna(tx) or pd.isna(ty):
            continue
        if rank == lane_rank:
            elements.append(f'<line x1="{_px(ball_x):.1f}" y1="{_py(ball_y):.1f}" x2="{_px(tx):.1f}" y2="{_py(ty):.1f}" stroke="var(--text-primary)" stroke-width="2" stroke-dasharray="4 3" opacity="0.85" />')
        elements.append(f'<circle cx="{_px(tx):.1f}" cy="{_py(ty):.1f}" r="5" fill="none" stroke="var(--text-secondary)" stroke-width="2" />')
        elements.append(f'<text x="{_px(tx):.1f}" y="{_py(ty) - 9:.1f}" font-size="11" text-anchor="middle" fill="var(--text-secondary)" font-family="JetBrains Mono, monospace">O{rank}</text>')

    elements.append(f'<circle cx="{_px(ball_x):.1f}" cy="{_py(ball_y):.1f}" r="4" fill="var(--text-primary)" />')
    elements.append(f'<text x="{_px(ball_x):.1f}" y="{_py(ball_y) - 8:.1f}" font-size="10" text-anchor="middle" fill="var(--text-primary)" font-family="JetBrains Mono, monospace">ball</text>')

    for _, r in rows.iterrows():
        color = ROLE_COLORS.get(r["defender_functional_role"], ROLE_COLORS["unclassified"])
        cx, cy = _px(r["defender_x"]), _py(r["defender_y"])
        stroke = "var(--text-primary)" if r.get("screens_top_option") else "none"
        elements.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{color}" stroke="{stroke}" stroke-width="2" />')
        elements.append(f'<text x="{cx:.1f}" y="{cy + 3.5:.1f}" font-size="9" text-anchor="middle" fill="#ffffff" font-family="JetBrains Mono, monospace" font-weight="700">{int(r["defender_slot_index"])}</text>')

    svg = f"""<svg class="pitch-svg" viewBox="0 0 {SVG_W:.0f} {SVG_H:.0f}" role="img" aria-label="{esc(title)}">
{''.join(elements)}
</svg>"""

    return f"""
<div class="fig-card{' wide' if wide else ''}">
<figure style="margin:0;">
{svg}
<figcaption>
<b>{esc(title)}</b><br>{caption}
<br>event_id: {esc(event_id)}
</figcaption>
</figure>
</div>"""


def _role_legend() -> str:
    items = "".join(
        f'<span class="lg-chip"><span class="lg-dot" style="background:{color}"></span>{esc(role)}</span>'
        for role, color in ROLE_COLORS.items()
    )
    return f'<div class="role-legend">{items}</div>'


def _checks_table(checks: list[dict]) -> str:
    rows = ""
    for c in checks:
        ok = c["matches_known_result"]
        status = "OK" if ok else "DISCREPANCY"
        rows += f"""
<tr>
  <td><span class="check-row-icon{'' if ok else ' bad'}"></span><code>{esc(c['check'])}</code></td>
  <td>{esc(c['method'])}</td>
  <td>{esc(c['known_result'])}</td>
  <td class="result {'ok' if ok else 'bad'}">{status}</td>
</tr>"""
    return f"""
<div class="results-table-wrap">
<table class="results-table">
<thead><tr><th>Check</th><th>Method</th><th>Known result</th><th style="text-align:right;">Verdict</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""


def build_report(data: dict, df: pd.DataFrame) -> str:
    n_matched = sum(1 for c in data["checks"] if c["matches_known_result"])
    n_total = len(data["checks"])

    body = f"""
<h2 class="section-title">Four recomputation checks</h2>
<p class="section-note">{n_matched}/{n_total} matched known result.</p>
{_checks_table(data['checks'])}

<div class="caveat-box" style="margin-top:24px;">
<h3>Coverage gap</h3>
<p style="margin:0; color:var(--text-secondary); font-size:13.5px;">{esc(data['coverage_gap'])}</p>
</div>

<h2 class="section-title" style="margin-top:48px;">Three real frames, plotted to scale</h2>
<p class="section-note">Role legend below.</p>
{_role_legend()}
<div class="fig-grid">
"""

    ex = data["example_frames"]
    for key, spec in (
        ("clean_screen", {"title": "Near-perfect lane screen", "caption": "highest lane_screening_score_option_1 among screens_top_option==True rows; defender highlighted with a dark ring, ball&rarr;O1 lane drawn emphasized."}),
        ("overload", {"title": "overload_score == 3 frame", "caption": "an isolated defender nearest to all three ranked options."}),
        ("last_line_rich", {"title": "8-defender frame, last_line-rich", "caption": "at least 3 slots classified last_line, showing the tercile split working with room to separate cleanly.", "wide": True}),
    ):
        info = ex.get(key, {})
        event_id = info.get("event_id")
        if not event_id:
            body += f'<div class="fig-card"><p class="section-note">No example available for {esc(key)}.</p></div>'
            continue
        rows = df[df["event_id"] == event_id]
        if rows.empty:
            body += f'<div class="fig-card"><p class="section-note">Event {esc(event_id)} not found in the current parquet.</p></div>'
            continue
        ball_x, ball_y = rows.iloc[0]["ball_x"], rows.iloc[0]["ball_y"]
        source_note = "" if info.get("source", "").startswith("cited") else " (replacement example -- the originally cited event_id no longer fit the description)"
        body += _pitch_figure(rows, ball_x, ball_y, event_id, spec["title"], spec["caption"] + source_note, wide=spec.get("wide", False))

    body += """
</div>

<div class="closing-note" style="margin-top:32px;">
<b>What this replaces, partially:</b> This replaces the "check against video" step of Phase 7 only partially.
Every row's stored value was verified against an independent recomputation from raw geometry, and the three
frames above are drawn to scale from the same stored coordinates -- but no frame here has been checked against
the actual match footage. Real video validation of specific flagged moments (e.g. the clean-screen and overload
examples above) is still open.
</div>"""

    return render.render_article(
        eyebrow="FOOTBALL SANITY CHECK · PHASE 7",
        title="Passive Dataset Football Sanity Check",
        dek=(
            "Phase 7 calls for checking feature values against match video, which isn't available outside this "
            "repo's own tooling. This reimplements four documented formulas independently from raw geometry and "
            "compares every row against the real stored output on the rebuilt parquet."
        ),
        stats=[
            (f"{data['n_rows']:,}", "rows checked"),
            (str(n_total), "recomputation checks"),
            (str(n_matched), "matched known result"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + render.CONFOUND_CSS + FOOTBALL_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    df = pd.read_parquet(PARQUET_PATH)
    html = build_report(data, df)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
