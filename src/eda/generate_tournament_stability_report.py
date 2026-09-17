"""CLI entrypoint: render reports/eda/TOURNAMENT_STABILITY_CHECK.json as a
self-contained HTML report, in the same shared design system as the other
EDA reports.

Usage:
    python -m src.eda.generate_tournament_stability_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "TOURNAMENT_STABILITY_CHECK.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "TOURNAMENT_STABILITY_CHECK.html"

VERDICT_CLASS = {"arbitrary train/test noise": "v-no", "genuine tournament-level difference": "v-yes"}


def _bin_bars(bins: list[dict]) -> str:
    rates = [b["shot_rate_pct"] for b in bins if b.get("shot_rate_pct") is not None]
    max_rate = (max(rates) * 1.15) if rates else 1.0
    bars = "".join(
        f"""
<div class="qbar-col">
  <span class="qbar-val">{b['shot_rate_pct']:.2f}%</span>
  <div class="qbar-fill" style="height:{max(2, b['shot_rate_pct']/max_rate*100):.1f}%"></div>
  <span class="qbar-label">{esc(b['bin'])}</span>
  <span class="qbar-n">n={b['n']:,}</span>
</div>""" if b.get("shot_rate_pct") is not None else '<div class="qbar-col"><span class="qbar-label">n/a</span></div>'
        for b in bins
    )
    return f'<div class="qbar-row small">{bars}</div>'


def _feature_section(entry: dict) -> str:
    tt = entry["train_test_finding"]
    verdict_class = VERDICT_CLASS.get(entry["verdict"], "v-partially")

    tt_html = f"""
<p class="test-subnote">Prompt 21 train/test finding: shape=<b>{esc(tt['overall_shape'])}</b>,
spearman_rho={tt['spearman_rho']}, train/val shape=<b>{esc(str(tt['train_val_shape']))}</b>,
test shape=<b>{esc(str(tt['test_shape']))}</b>, flagged consistent={tt['train_test_consistent']}.</p>"""

    tournament_cards = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(label)} (n={d['n_rows']:,} rows, {d['n_matches']} matches)</h5>
  <p style="font-size:12.5px;color:var(--text-secondary);margin:0 0 8px;">shape: <b>{esc(d['shape'])}</b>
  &mdash; range {d['rate_range_pp']}pp</p>
  {_bin_bars(d['bins'])}
</div>"""
        for label, d in entry["per_tournament"].items()
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(entry['feature'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">({esc(entry['dataset'])})</span></h2>
  {tt_html}
  <div class="verdict-banner {verdict_class}">
    <b>Verdict: {esc(entry['verdict'].upper())}</b> &mdash; {esc(entry['verdict_reason'])}
  </div>
  <div class="numsplit-grid">{tournament_cards}</div>
</div>"""


def build_report(data: dict) -> str:
    n_genuine = sum(1 for f in data["features"] if f["verdict"] == "genuine tournament-level difference")
    n_noise = sum(1 for f in data["features"] if f["verdict"] == "arbitrary train/test noise")

    excluded_html = "".join(
        finding_card("excluded", feat, reason)
        for feat, reason in data["excluded_from_investigation"].items()
    )

    body = f"""
<p class="test-subnote" style="margin-bottom:24px;">{esc(data['methodology'])}</p>
{excluded_html}
{"".join(_feature_section(f) for f in data["features"])}
"""

    return render.render_article(
        eyebrow="TOURNAMENT STABILITY CHECK",
        title="Train/Test-Inconsistent Features, Split by Tournament",
        dek=(
            "Prompt 21 flagged 11 features as train/test-inconsistent. Instead of an arbitrary match split, this "
            "splits by tournament (WC2022 vs Euro2024) to check a different hypothesis: is the instability "
            "arbitrary noise, or a genuine population-mixing artefact (rules, pitch dimensions, squad quality "
            "differ between a World Cup and a Euros)? nearest_defender_distance is excluded -- a separate, "
            "already-known self-reference bug."
        ),
        stats=[
            (f"{len(data['features'])}", "features investigated"),
            (str(n_genuine), "genuine tournament-level difference"),
            (str(n_noise), "arbitrary train/test noise"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
