"""CLI entrypoint: render reports/eda/PLAYER_LEVEL_VALIDITY_CHECK.json as a
self-contained HTML report, in the same shared design system as the other
EDA reports.

Usage:
    python -m src.eda.generate_player_level_validity_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "PLAYER_LEVEL_VALIDITY_CHECK.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "PLAYER_LEVEL_VALIDITY_CHECK.html"

RISK_CLASS = {"high": "v-yes", "moderate": "v-partially", "low": "v-no"}


def _lorenz_chart(lorenz: list[dict]) -> str:
    bars = "".join(
        f"""
<div class="qbar-col">
  <span class="qbar-val">{p['cumulative_row_share']*100:.1f}%</span>
  <div class="qbar-fill" style="height:{max(2, p['cumulative_row_share']*100):.1f}%"></div>
  <span class="qbar-label">top {p['pct_of_players']}%<br>({p['n_players']} players)</span>
</div>"""
        for p in lorenz
    )
    return f'<div class="qbar-row">{bars}</div>'


def _icc_table(rows: list[dict], threshold: float) -> str:
    body = "".join(
        f"""<tr{' class="hi-row"' if r['player_identity_flavoured'] else ''}>
  <td>{r['rank']}</td><td><code>{esc(r['feature'])}</code></td><td>{esc(r['feature_type'])}</td>
  <td>{esc(str(r['value']))}</td><td style="font-size:11px;color:var(--text-muted);">{esc(r['method'])}</td>
  <td>{'FLAGGED &gt; ' + str(threshold) if r['player_identity_flavoured'] else ''}</td>
</tr>"""
        for r in rows
    )
    return f"""
<div class="qchart-card" style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:12.5px;">
<thead><tr style="text-align:left;color:var(--text-muted);">
  <th style="padding:6px 10px;">rank</th><th style="padding:6px 10px;">feature</th><th style="padding:6px 10px;">type</th>
  <th style="padding:6px 10px;">ICC / Cramer's V</th><th style="padding:6px 10px;">method</th><th style="padding:6px 10px;"></th>
</tr></thead>
<tbody>{body}</tbody>
</table>
</div>"""


def build_report(data: dict) -> str:
    scope_card = finding_card("scope limitation -- passive dataset excluded", "not an oversight", data["scope_limitation"])
    target_card = finding_card("target-independent", "same for binary and xG", data["target_independence_note"])

    rc = data["row_concentration"]
    overlap = data["train_test_player_overlap"]

    risk_html = ""
    if data["risk_pairs"]:
        risk_rows = "".join(
            f"""
<div class="verdict-banner {RISK_CLASS[rp['risk_level']]}" style="margin-bottom:10px;">
<b>{esc(rp['feature'])} -- {esc(rp['risk_level'].upper())} RISK</b><br>{esc(rp['reason'])}
</div>"""
            for rp in data["risk_pairs"]
        )
        risk_html = f"""
<h2 class="test-title" style="margin-top:32px;">Risk pairs -- high ICC AND high train/test player overlap</h2>
<p class="test-subnote">Diagnostic only -- no correction applied here. Consider a player-grouped CV fold in
addition to the match-grouped one, or drop/de-weight the feature, at the modelling stage.</p>
{risk_rows}"""
    else:
        risk_html = '<h2 class="test-title" style="margin-top:32px;">Risk pairs</h2><p class="test-subnote">None found.</p>'

    body = f"""
{scope_card}
{target_card}

<h2 class="test-title">Step 1 -- row concentration</h2>
<p class="test-subnote">{rc['n_distinct_players']} distinct players across {rc['n_rows']:,} rows. Top 10 players:
<b>{rc['top_10_players_row_share']*100:.2f}%</b> of rows. Top 25: <b>{rc['top_25_players_row_share']*100:.2f}%</b>.
Top 50: <b>{rc['top_50_players_row_share']*100:.2f}%</b>. Gini coefficient: <b>{rc['gini_coefficient']}</b>
(0 = perfectly even, 1 = fully concentrated in one player). Max rows for a single player: {rc['max_rows_single_player']}
(median {rc['median_rows_per_player']}).</p>
<div class="qchart-card">
  <h4>Lorenz curve -- cumulative row share by player rank</h4>
  {_lorenz_chart(rc['lorenz_curve_by_player_rank'])}
</div>

<h2 class="test-title" style="margin-top:32px;">Step 2 -- per-feature ICC / player-identity association (all 34 locked active features)</h2>
<p class="test-subnote">{esc(data['icc_methodology'])} Threshold: <b>{data['icc_threshold']}</b> (same convention as
prompt 27's Cramer's V thresholds).</p>
{_icc_table(data['feature_icc_ranking'], data['icc_threshold'])}

<h2 class="test-title" style="margin-top:32px;">Step 3 -- train/test player overlap</h2>
<p class="test-subnote">{overlap['n_test_matches']} test matches ({overlap['n_test_players']} distinct players) vs
{overlap['n_trainval_matches']} train+val matches ({overlap['n_trainval_players']} distinct players).
<b>{overlap['overlap_pct_of_test_players']}%</b> of test-set players ({overlap['n_overlap_players']} of
{overlap['n_test_players']}) also appear in train+val. {esc(overlap['note'])}</p>

{risk_html}
"""

    return render.render_article(
        eyebrow="PLAYER-LEVEL VALIDITY CHECK",
        title="Player-Level Repeated-Measures / Split-Validity Check",
        dek=(
            "StratifiedGroupKFold groups by match, not by player. This checks whether that matters: how "
            "concentrated is the active dataset in a small number of players, which locked features are more "
            "'who is defending' than 'what is the defensive situation', and how much train/test player overlap "
            "exists. Diagnostic only -- no causal claims, no corrections applied here."
        ),
        stats=[
            (f"{data['row_concentration']['n_distinct_players']}", "distinct players"),
            (f"{data['train_test_player_overlap']['overlap_pct_of_test_players']}%", "test players also in train+val"),
            (str(len(data["risk_pairs"])), "risk pair(s) flagged"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + """
tr.hi-row { background: var(--amber-wash); }
""",
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
