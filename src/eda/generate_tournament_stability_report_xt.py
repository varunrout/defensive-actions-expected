"""CLI entrypoint: render reports/analysis/xt_target/TOURNAMENT_STABILITY_CHECK.json
as a self-contained HTML report -- the xT sibling of
generate_tournament_stability_report_xg.py.

Usage:
    python -m src.eda.generate_tournament_stability_report_xt
"""

from __future__ import annotations

import json

from src.eda import render, xt_common as xc
from src.eda.render import esc, finding_card

INPUT_PATH = xc.OUT_DIR / "TOURNAMENT_STABILITY_CHECK.json"
OUTPUT_PATH = xc.OUT_DIR / "TOURNAMENT_STABILITY_CHECK.html"
BINARY_INPUT_PATH = xc.REPO_ROOT / "reports" / "analysis" / "shot_target" / "TOURNAMENT_STABILITY_CHECK.json"
XG_INPUT_PATH = xc.REPO_ROOT / "reports" / "analysis" / "xg_target" / "TOURNAMENT_STABILITY_CHECK.json"

VERDICT_CLASS = {"arbitrary train/test noise": "v-no", "genuine tournament-level difference": "v-yes"}

EXTRA_CSS = """
.xtb-row { display: flex; align-items: stretch; gap: 3px; margin: 10px 0 4px; }
.xtb-col { flex: 1; display: flex; flex-direction: column; align-items: center; }
.xtb-track { position: relative; width: 100%; height: 72px; background: var(--plane); border-radius: 3px; }
.xtb-track .mid { position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: var(--text-primary); opacity: 0.35; }
.xtb-bar { position: absolute; left: 18%; width: 64%; border-radius: 2px; }
.xtb-bar.pos { bottom: 50%; background: var(--good); }
.xtb-bar.neg { top: 50%; background: var(--neg); }
.xtb-n { font-family: "JetBrains Mono", monospace; font-size: 9px; color: var(--text-muted); margin-top: 3px; }
"""


def _bin_bars(bins: list[dict]) -> str:
    vals = [b["mean_xt"] for b in bins if b.get("mean_xt") is not None]
    max_abs = max([abs(v) for v in vals] + [1e-12])
    cols = ""
    for b in bins:
        v = b.get("mean_xt")
        if v is None:
            cols += '<div class="xtb-col"><div class="xtb-track"><span class="mid"></span></div></div>'
            continue
        h = max(1.5, abs(v) / max_abs * 50)
        side = "pos" if v >= 0 else "neg"
        cols += (f'<div class="xtb-col"><div class="xtb-track" title="{esc(b["bin"])}: {v:+.6f} (n={b["n"]:,})">'
                 f'<span class="mid"></span><div class="xtb-bar {side}" style="height:{h:.1f}%"></div></div>'
                 f'<span class="xtb-n">{b["n"]:,}</span></div>')
    return f'<div class="xtb-row">{cols}</div>'


def _feature_section(entry: dict, binary_verdict: str | None, xg_verdict: str | None) -> str:
    tt = entry["train_test_finding"]
    cls = VERDICT_CLASS.get(entry["verdict"], "v-partially")

    comparisons = ""
    for label, other in (("binary target (target_future_shot_10s)", binary_verdict),
                         ("xG target (target_future_xg_10s)", xg_verdict)):
        if other is None:
            continue
        if other != entry["verdict"]:
            comparisons += f"""
<div class="finding flag" style="margin:12px 0;">
<span class="tag">diverges from the {esc(label.split(' (')[0])} check</span>
<p>{esc(label)} verdict: <b>{esc(other)}</b>. xT-delta verdict: <b>{esc(entry['verdict'])}</b>. Same feature,
same tournament split, same method -- the targets disagree here.</p>
</div>"""
        else:
            comparisons += f"""
<div class="finding" style="margin:12px 0;">
<span class="tag">agrees with the {esc(label.split(' (')[0])} check</span>
<p>Both return <b>{esc(other)}</b> for this feature.</p>
</div>"""

    if tt.get("in_atlas"):
        tt_html = (f'<p class="test-subnote">This portal\'s own atlas finding: shape=<b>{esc(str(tt["overall_shape"]))}</b>, '
                   f'Spearman &rho;={tt["spearman_rho"]}, train/val shape=<b>{esc(str(tt["train_val_shape"]))}</b>, '
                   f'test shape=<b>{esc(str(tt["test_shape"]))}</b>, flagged consistent={tt["train_test_consistent"]}.</p>')
    else:
        tt_html = f'<p class="test-subnote">{esc(str(tt.get("note", "")))}</p>'

    cards = "".join(
        f"""
<div class="numsplit-card">
  <h5>{esc(label)} (n={d['n_rows']:,} rows, {d['n_matches']} matches)</h5>
  <p style="font-size:12.5px;color:var(--text-secondary);margin:0 0 4px;">shape: <b>{esc(d['shape'])}</b>
  &mdash; bin means {d['bin_mean_min']:+.6f} to {d['bin_mean_max']:+.6f} (range {d['mean_xt_range']:.6f}),
  curve {"crosses zero" if d['curve_crosses_zero'] else "stays on one side of zero"}</p>
  {_bin_bars(d['bins'])}
</div>"""
        for label, d in entry["per_tournament"].items()
    )

    zc = ""
    if entry["zero_crossing_differs_between_tournaments"]:
        zc = """
<div class="finding flag" style="margin:12px 0;">
<span class="tag">direction differs between tournaments</span>
<p>The binned curve crosses zero in one tournament but not the other -- the feature separates threat-reducing
from threat-increasing actions in one population and only varies in magnitude in the other. This question cannot
be asked of the xG target at all, whose binned means are non-negative by construction.</p>
</div>"""

    return f"""
<div class="test-block">
  <h2 class="test-title">{esc(entry['feature'])} <span style="font-weight:400;font-size:15px;color:var(--text-muted);">(active, xT delta)</span></h2>
  {tt_html}
  {comparisons}
  {zc}
  <div class="verdict-banner {cls}">
    <b>Verdict: {esc(entry['verdict'].upper())}</b> &mdash; {esc(entry['verdict_reason'])}
  </div>
  <div class="numsplit-grid">{cards}</div>
</div>"""


def build_report(data: dict, binary: dict | None, xg: dict | None) -> str:
    bv = {f["feature"]: f["verdict"] for f in binary["features"]} if binary else {}
    xv = {f["feature"]: f["verdict"] for f in xg["features"]} if xg else {}
    n_genuine = sum(1 for f in data["features"] if f["verdict"] == "genuine tournament-level difference")
    n_div_b = sum(1 for f in data["features"] if bv.get(f["feature"]) and bv[f["feature"]] != f["verdict"])
    n_div_x = sum(1 for f in data["features"] if xv.get(f["feature"]) and xv[f["feature"]] != f["verdict"])

    excluded = "".join(finding_card("excluded", feat, reason)
                       for feat, reason in data["excluded_from_investigation"].items())

    body = f"""
<div class="rule-banner"><b>Scope.</b> {esc(data['step_0_scope_note'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Method.</b> {esc(data['methodology'])}</div>
<div class="rule-banner" style="margin-top:16px;"><b>Reading the bars.</b>
{esc(data['target_scale']['direction_note'])} Bars diverge from a centre line: green up = xT fell (threat
reduced), red down = xT rose. flat_margin for the shape classification on this run is
<code>{data['flat_margin']:.6f}</code>.</div>
{excluded}
{"".join(_feature_section(f, bv.get(f['feature']), xv.get(f['feature'])) for f in data['features'])}
"""

    return render.render_article(
        eyebrow="TOURNAMENT STABILITY CHECK -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
        title="Train/Test-Inconsistent Features, Split by Tournament (xT delta)",
        dek=(
            "The same 6 ACTIVE features, the same tournament split (WC2022 vs Euro2024) and the same method as "
            "the binary and xG checks, against mean xT delta. Only flat_margin was adapted. Active-binary leg "
            "only -- the 4 passive features in the shared INVESTIGATE list are out of scope."
        ),
        stats=[
            (str(len(data["features"])), "features investigated"),
            (str(n_genuine), "genuine tournament-level difference"),
            (str(n_div_b), "diverge from binary verdict"),
            (str(n_div_x), "diverge from xG verdict"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.NUMERICAL_ATLAS_CSS + EXTRA_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    binary = json.loads(BINARY_INPUT_PATH.read_text(encoding="utf-8")) if BINARY_INPUT_PATH.exists() else None
    xg = json.loads(XG_INPUT_PATH.read_text(encoding="utf-8")) if XG_INPUT_PATH.exists() else None
    OUTPUT_PATH.write_text(build_report(data, binary, xg), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
