"""CLI entrypoint: render reports/eda_xg/TOURNAMENT_STABILITY_CHECK.json as a
self-contained HTML report -- the xG counterpart to TOURNAMENT_STABILITY_CHECK.html.

Usage:
    python -m src.eda.generate_tournament_stability_report_xg
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc, finding_card

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "TOURNAMENT_STABILITY_CHECK.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "TOURNAMENT_STABILITY_CHECK.html"
BINARY_INPUT_PATH = REPO_ROOT / "reports" / "eda" / "TOURNAMENT_STABILITY_CHECK.json"

VERDICT_CLASS = {"arbitrary train/test noise": "v-no", "genuine tournament-level difference": "v-yes"}


def _bin_bars(bins: list[dict]) -> str:
    values = [b["mean_xg"] for b in bins if b.get("mean_xg") is not None]
    max_val = (max(values) * 1.15) if values else 1.0
    bars = "".join(
        f"""
<div class="qbar-col">
  <span class="qbar-val">{b['mean_xg']:.5f}</span>
  <div class="qbar-fill" style="height:{max(2, b['mean_xg']/max_val*100):.1f}%"></div>
  <span class="qbar-label">{esc(b['bin'])}</span>
  <span class="qbar-n">n={b['n']:,}</span>
</div>""" if b.get("mean_xg") is not None else '<div class="qbar-col"><span class="qbar-label">n/a</span></div>'
        for b in bins
    )
    return f'<div class="qbar-row small">{bars}</div>'


def _feature_section(entry: dict, binary_verdict: str | None) -> str:
    tt = entry["train_test_finding"]
    verdict_class = VERDICT_CLASS.get(entry["verdict"], "v-partially")

    divergence_html = ""
    if binary_verdict is not None and binary_verdict != entry["verdict"]:
        divergence_html = f"""
<div class="finding flag" style="margin:12px 0;">
<span class="tag">diverges from the binary-target check</span>
<p>Binary target verdict: <b>{esc(binary_verdict)}</b>. xG verdict: <b>{esc(entry['verdict'])}</b>. Same feature,
same tournament split -- the two targets disagree here.</p>
</div>"""

    tt_html = f"""
<p class="test-subnote">xG atlas train/test finding: shape=<b>{esc(tt['overall_shape'])}</b>,
spearman_rho={tt['spearman_rho']}, train/val shape=<b>{esc(str(tt['train_val_shape']))}</b>,
test shape=<b>{esc(str(tt['test_shape']))}</b>, flagged consistent={tt['train_test_consistent']}.</p>"""

    tournament_cards = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(label)} (n={d['n_rows']:,} rows, {d['n_matches']} matches)</h5>
  <p style="font-size:12.5px;color:var(--text-secondary);margin:0 0 8px;">shape: <b>{esc(d['shape'])}</b>
  &mdash; mean-xG range {d['mean_xg_range']}</p>
  {_bin_bars(d['bins'])}
</div>"""
        for label, d in entry["per_tournament"].items()
    )

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(entry['feature'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">({esc(entry['dataset'])}, xG)</span></h2>
  {tt_html}
  {divergence_html}
  <div class="verdict-banner {verdict_class}">
    <b>Verdict: {esc(entry['verdict'].upper())}</b> &mdash; {esc(entry['verdict_reason'])}
  </div>
  <div class="numsplit-grid">{tournament_cards}</div>
</div>"""


def build_report(data: dict, binary_data: dict | None) -> str:
    binary_verdicts = (
        {f["feature"]: f["verdict"] for f in binary_data["features"]} if binary_data else {}
    )
    n_genuine = sum(1 for f in data["features"] if f["verdict"] == "genuine tournament-level difference")
    n_diverging = sum(1 for f in data["features"] if binary_verdicts.get(f["feature"]) and binary_verdicts[f["feature"]] != f["verdict"])

    excluded_html = "".join(
        finding_card("excluded", feat, reason)
        for feat, reason in data["excluded_from_investigation"].items()
    )

    body = f"""
<p class="test-subnote" style="margin-bottom:24px;">{esc(data['methodology'])}</p>
{excluded_html}
{"".join(_feature_section(f, binary_verdicts.get(f['feature'])) for f in data["features"])}
"""

    return render.render_article(
        eyebrow="TOURNAMENT STABILITY CHECK -- XG",
        title="Train/Test-Inconsistent Features, Split by Tournament (xG)",
        dek=(
            "The continuous-target counterpart to TOURNAMENT_STABILITY_CHECK.html -- same 10 features, same "
            "tournament split (WC2022 vs Euro2024), same method, against mean xG instead of shot-rate percentage. "
            "flat_margin is relative to each dataset's own overall mean xG, not a fixed percentage-point margin."
        ),
        stats=[
            (f"{len(data['features'])}", "features investigated"),
            (str(n_genuine), "genuine tournament-level difference"),
            (str(n_diverging), "diverge from binary-target verdict"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    binary_data = json.loads(BINARY_INPUT_PATH.read_text(encoding="utf-8")) if BINARY_INPUT_PATH.exists() else None
    html = build_report(data, binary_data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
