"""CLI entrypoint: render
reports/analysis/xt_target/active_numerical_target_atlas.json as a
self-contained HTML report -- the xT sibling of
generate_numerical_xg_target_reports.py (the confirmed renderer of the xg
portal's equivalent file).

Same visual system: the same NUM_TARGET_CSS block the xg renderer imports
from generate_numerical_target_reports, the same `nt-card` / `nt-bin-row` /
`nt-badge` markup, the same html_shell. The bin bars are the one deliberate
visual change -- they diverge from zero instead of filling from the left,
because a left-anchored fill renders every negative bin as zero width.

Usage:
    python -m src.eda.generate_numerical_xt_target_reports
"""

from __future__ import annotations

import json

from src.eda import xt_common as xc
from src.eda.generate_numerical_target_reports import NUM_TARGET_CSS
from src.eda.render import esc, finding_card, findings_grid, html_shell

OUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET

XT_NUM_CSS = """
.nt-div-row { display: grid; grid-template-columns: 200px 1fr 92px 74px; align-items: center; gap: 8px;
  font-size: 11.5px; margin-bottom: 3px; }
.nt-div-label { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-secondary);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nt-div-track { position: relative; height: 13px; background: var(--plane); border-radius: 3px; }
.nt-div-track .zero { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px;
  background: var(--text-primary); opacity: 0.4; }
.nt-div-fill { position: absolute; top: 2px; height: 9px; border-radius: 2px; }
.nt-div-fill.neg { background: var(--neg); right: 50%; }
.nt-div-fill.pos { background: var(--good); left: 50%; }
.nt-div-val { font-family: "JetBrains Mono", monospace; text-align: right; font-weight: 600; }
.nt-div-n { font-family: "JetBrains Mono", monospace; color: var(--text-muted); text-align: right; }
.xt-adapt { border-left: 4px solid var(--accent); background: var(--accent-wash); padding: 12px 14px;
  border-radius: 8px; margin: 14px 0 24px; font-size: 12.5px; line-height: 1.55; }
.xt-adapt h3 { margin: 0 0 8px; font-size: 15px; }
.xt-adapt ul { margin: 8px 0 0 18px; padding: 0; }
.xt-adapt li { margin-bottom: 6px; }
"""


def _bin_rows_html(bins: list[dict]) -> str:
    if not bins:
        return '<p style="font-size:11.5px; color:var(--text-muted);">no bins</p>'
    max_abs = max([abs(b["mean_xt"]) for b in bins] + [1e-12])
    rows = []
    for b in bins:
        v = b["mean_xt"]
        w = min(50.0, abs(v) / max_abs * 50)
        side = "pos" if v >= 0 else "neg"
        badge = '<span class="small-n-badge">small n</span>' if b.get("small_n") else ""
        rows.append(f"""
<div class="nt-div-row" title="{esc(b['bin'])}: n={b['n']:,}, mean {v:+.6f}, {b['pct_negative']:.1f}% neg / {b['pct_positive']:.1f}% pos">
  <span class="nt-div-label">{esc(b['bin'])}{badge}</span>
  <div class="nt-div-track"><span class="zero"></span><div class="nt-div-fill {side}" style="width:{w:.2f}%"></div></div>
  <span class="nt-div-val">{v:+.6f}</span>
  <span class="nt-div-n">n={b['n']:,}</span>
</div>""")
    return "".join(rows)


def _nonzero_html(f: dict) -> str:
    n = f.get("n_nonzero_delta") or 0
    if not f.get("bins_nonzero_delta"):
        return f'<p style="font-size:11.5px; color:var(--text-muted);">Insufficient non-zero-delta data (n={n:,}).</p>'
    r, rho = f.get("pearson_r_nonzero_delta"), f.get("spearman_rho_nonzero_delta")
    n_small = sum(1 for b in f["bins_nonzero_delta"] if b.get("small_n"))
    small_note = f' &middot; {n_small} bin(s) flagged small-n (&lt;{xc.SMALL_N_CONDITIONAL_THRESHOLD})' if n_small else ""
    shape = f.get("shape_nonzero_delta")
    shape_note = f' &middot; shape <b>{esc(str(shape))}</b>' if shape else ""
    return f"""
<p><b>Correlation given a non-zero delta:</b> Pearson r={r:+.4f} &middot; Spearman &rho;={rho:+.4f} &middot;
n={n:,}{small_note}{shape_note}</p>
<p><b>Binned mean {esc(TARGET)}, non-zero-delta subset</b> (fresh deciles on this subset's own distribution):</p>
{_bin_rows_html(f['bins_nonzero_delta'])}"""


def _consistency_html(cc: dict) -> str:
    if not cc.get("checked"):
        return f'<p style="font-size:11.5px; color:var(--text-muted);">Not checked -- {esc(cc.get("reason", ""))}</p>'
    consistent = cc["consistent"]
    verdict = (
        f"Consistent -- both splits classify as <b>{esc(cc['train_val_shape'])}</b>." if consistent else
        f'<b>Inconsistent</b> -- train+val shows <b>{esc(cc["train_val_shape"])}</b>, test shows '
        f'<b>{esc(cc["test_shape"])}</b>. Treat this pattern as noise, not a confirmed finding.'
    )
    note_html = (
        f'<p style="font-size:11.5px; color:var(--text-secondary);">{verdict}</p>' if consistent
        else f'<div class="nt-inconsistent-note">{verdict}</div>'
    )
    return f"""
{note_html}
<div class="nt-split-grid">
  <div><p class="nt-split-title">Train+Val</p>{_bin_rows_html(cc['train_val_bins'])}</div>
  <div><p class="nt-split-title">Test</p>{_bin_rows_html(cc['test_bins'])}</div>
</div>"""


def _feature_card(rank: int, f: dict, max_abs_rho: float) -> str:
    rho = f["spearman_rho"]
    is_dropped = f["status"] == "dropped"
    is_unreliable = bool(f.get("unreliable_note"))
    is_inconsistent = f["consistency_check"].get("checked") and not f["consistency_check"].get("consistent")
    card_cls = "unreliable" if is_unreliable else ("dropped" if is_dropped else "locked")

    if rho is None:
        rho_html = '<span style="font-size:10.5px; color:var(--text-muted);">n/a</span>'
    else:
        pct = min(50.0, abs(rho) / max_abs_rho * 50) if max_abs_rho else 0.0
        side = "pos" if rho > 0 else "neg"
        rho_html = f"""
<div>
  <div class="nt-rho-track"><span class="zero"></span><div class="nt-rho-fill {side}" style="width:{pct:.1f}%"></div></div>
  <div class="nt-rho-val">&rho;={rho:+.3f}</div>
</div>"""

    badges = [f'<span class="nt-badge {"dropped" if is_dropped else "locked"}">{"DROPPED" if is_dropped else "LOCKED"}</span>']
    if is_unreliable:
        badges.append('<span class="nt-badge unreliable">UNRELIABLE</span>')
    if is_inconsistent:
        badges.append('<span class="nt-badge inconsistent">TRAIN/TEST MISMATCH</span>')

    reason_html = f'<div class="nt-reason"><b>Why dropped:</b> {esc(f["reason"])}</div>' if is_dropped and f.get("reason") else ""
    unreliable_html = f'<div class="nt-unreliable-note"><b>Unreliable:</b> {f["unreliable_note"]}</div>' if is_unreliable else ""

    if f.get("bins"):
        crosses = f.get("binned_curve_crosses_zero")
        cross_html = (
            f'<p style="font-size:11.5px;color:var(--text-secondary);"><b>Direction:</b> the binned curve '
            f'{"crosses zero" if crosses else "stays on one side of zero"} '
            f'(bin means from {f["bin_mean_min"]:+.6f} to {f["bin_mean_max"]:+.6f}) -- '
            + ("this feature separates threat-reducing from threat-increasing actions, not just magnitude."
               if crosses else
               "this feature only varies in magnitude within one direction.")
            + "</p>"
        )
        body = f"""
<p><b>Correlation:</b> Pearson r={f['pearson_r']:+.4f} &middot; Spearman &rho;={rho:+.4f} &middot;
binning: {esc(f['binning_method'])} &middot; n={f['n_rows_used']:,}</p>
<p><b>Binned mean {esc(TARGET)}</b> (bin range {f['bin_range']:.6f}):</p>
{_bin_rows_html(f['bins'])}
{cross_html}
<p><b>Train/test consistency check:</b></p>
{_consistency_html(f['consistency_check'])}
"""
    else:
        body = f'<p style="color:var(--text-muted);">Insufficient data to analyze ({f["consistency_check"].get("reason", "")}).</p>'

    body += f"""
<details style="margin-top:10px;">
<summary style="cursor:pointer; font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--neg);">
  given a non-zero delta (n={f.get('n_nonzero_delta', 0):,}) -- click to expand
</summary>
<div style="margin-top:8px;">
<p style="font-size:11.5px; color:var(--text-secondary);">Conditional on <code>{esc(TARGET)} != 0</code> -- this
target's structural-zero analogue of the xg atlas's "given a shot" panel. It removes the rows where the action
never crossed an xT grid-cell boundary. Binning is recomputed fresh on this subset.</p>
{_nonzero_html(f)}
</div>
</details>"""

    return f"""
<details class="nt-card {card_cls}">
  <summary class="nt-summary">
    <span class="nt-rank">#{rank}</span>
    <span class="nt-name">{esc(f['feature'])}<span class="mono-sub">{esc(f['type'])}</span></span>
    {rho_html}
    <span class="nt-shape">{esc(f['shape'])}</span>
    <span></span>
    <span class="nt-badges">{''.join(badges)}</span>
  </summary>
  <div class="nt-body">
    {reason_html}
    {unreliable_html}
    {body}
  </div>
</details>"""


def _adaptation_section(data: dict) -> str:
    s = data["target_scale"]
    return f"""
<div class="xt-adapt">
<h3>What this target is, and exactly what methodology changed because of it</h3>
<p><b>{esc(TARGET)} can be negative and is roughly symmetric.</b> On these same {s['n_defined']:,} rows:
mean <b>{s['mean']:+.6f}</b>, std <b>{s['std']:.6f}</b>, skew <b>{s['skew']:+.3f}</b>, excess kurtosis
<b>{s['excess_kurtosis']:.2f}</b>, <b>{s['pct_negative']:.1f}%</b> negative / <b>{s['pct_positive']:.1f}%</b>
positive / <b>{s['pct_zero']:.1f}%</b> exactly zero. A POSITIVE value means xT fell across the defensive action
(threat reduced -- good defending); a NEGATIVE value means xT rose. This is nothing like
<code>target_future_xg_10s</code>, which is strictly non-negative, zero-inflated and right-skewed, and none of
that target's one-sided framing survives unmodified.</p>
<p><b>What was adapted, and what was not:</b></p>
<ul>
<li><b>Threshold scale-setting statistic -- CHANGED.</b> {esc(s['scale_note'])}
  Concretely on this run: flat_margin = <b>{s['flat_margin']:.6f}</b>, consistency range-trigger =
  <b>{s['range_trigger']:.6f}</b>.</li>
<li><b>Conditional panel -- CHANGED.</b> {esc(s['conditional_panel_note'])}</li>
<li><b>Correlation / lift framing -- CHANGED to both directions.</b> {esc(s['direction_note'])}
  Every card states whether its binned curve crosses zero;
  <b>{data['n_features_whose_binned_curve_crosses_zero']}</b> of
  {data['n_features_analyzed']} features' curves do.</li>
<li><b>Log-transform -- NOTHING TO CHANGE.</b> {esc(s['log_transform_note'])}</li>
<li><b>Carried over completely unchanged</b> (all scale-free): the reconstructed pre-drop numerical pool
  (<code>build_pool</code>, unmodified), {data['thresholds_carried_over_unchanged']['n_quantile_bins']} quantile
  deciles, the discrete-cardinality cutoff of
  {data['thresholds_carried_over_unchanged']['discrete_cardinality_threshold']}, the
  |Spearman &rho;| &ge; {data['thresholds_carried_over_unchanged']['rho_threshold']} consistency-check trigger,
  <code>classify_shape()</code>'s U / inverse-U / monotonic / flat heuristic, and the canonical match-grouped
  train/test split.</li>
</ul>
</div>"""


def build_report(data: dict) -> str:
    features = data["features"]
    max_abs_rho = max((abs(f["spearman_rho"]) for f in features if f["spearman_rho"] is not None), default=1.0) or 1.0
    pool = data["pool_construction"]
    s = data["target_scale"]

    findings = [
        finding_card(
            "pool",
            f"{pool['n_total']} features analyzed",
            f"{pool['n_locked']} locked + {pool['n_dropped']} dropped-but-included -- the same reconstructed "
            "pre-drop numerical candidate pool the xg and binary atlases use, run against the xT-delta target.",
        ),
        finding_card(
            "scale",
            f"mean {esc(TARGET)} = {s['mean']:+.6f}, std {s['std']:.6f}",
            f"roughly symmetric, {s['pct_negative']:.1f}% negative -- NOT a rate and NOT non-negative. Thresholds "
            f"are translated from the xg suite via xg's own std-fraction ({s['xg_flat_margin_as_fraction_of_xg_std']:.6f}): "
            f"flat margin {s['flat_margin']:.6f}, range trigger {s['range_trigger']:.6f}.",
        ),
        finding_card(
            "conditional",
            "given a non-zero delta",
            f"each card expands to a panel on the {s['n_nonzero']:,} rows whose delta is not exactly zero -- "
            "this target's analogue of the xg atlas's shot-conditional panel.",
        ),
        finding_card(
            "direction",
            f"{data['n_features_whose_binned_curve_crosses_zero']}/{data['n_features_analyzed']} binned curves cross zero",
            "reported explicitly because a feature that separates threat-reducing from threat-increasing actions "
            "is doing something different from one that only varies in magnitude on one side.",
        ),
    ]
    if data["n_features_flagged_inconsistent"]:
        findings.append(finding_card(
            "caution",
            f"{data['n_features_flagged_inconsistent']} feature(s) flagged train/test-inconsistent",
            "shape classification differs between train+val and test matches -- treat as noise, not a confirmed "
            "pattern, until re-examined.",
            flag=True,
        ))
    unreliable = [f["feature"] for f in features if f.get("unreliable_note")]
    if unreliable:
        findings.append(finding_card(
            "data issue", ", ".join(unreliable),
            "flagged unreliable pending a known self-reference bug -- see the Distribution Atlas.", flag=True,
        ))
    findings.append(finding_card(
        "carried forward from Prompt 64",
        "Clearance / action_x artefact",
        "this atlas covers NUMERICAL features only, so the artefact does not surface here directly -- "
        "<code>event_type</code> is categorical. It is surfaced in the Category Atlas and cross-referenced "
        "in the closing note below, since it bears on how any location-derived feature's relationship to this "
        "target should be read.",
        flag=True,
    ))

    cards_html = "".join(_feature_card(i + 1, f, max_abs_rho) for i, f in enumerate(features))

    body = _adaptation_section(data) + findings_grid(findings) + f"""
<h2 class="section-title">Ranked by |Spearman &rho;| vs {esc(TARGET)}</h2>
<p class="section-note">Ranked by ABSOLUTE rho -- both directions. A strong negative rho (higher feature value
coincides with threat rising) is exactly as much a finding as a strong positive one on this target. Click a row to
expand its binned mean-delta curve; bars diverge from zero, green right = threat reduced, red left = threat
increased. Green left border = locked, amber = dropped (still included, reason inside), red = flagged
unreliable.</p>
<div class="nt-list">{cards_html}</div>
"""

    return html_shell(
        eyebrow="NUMERICAL XT TARGET ATLAS · ACTIVE-BINARY LEG ONLY",
        title=f"Active Defensive Actions: Numerical Features vs {TARGET}",
        dek=(
            f"Every numerical feature's relationship to the xT-delta target, on the same reconstructed pre-drop "
            f"candidate pool ({pool['n_total']} features) the xg and binary atlases use, one row per actual "
            "defensive action. Active-binary leg only -- the passive leg is out of scope for this target."
        ),
        stats=[
            (str(pool["n_total"]), "features analyzed"),
            (str(pool["n_locked"]), "locked"),
            (str(pool["n_dropped"]), "dropped, still shown"),
            (str(data["n_features_flagged_inconsistent"]), "train/test mismatch"),
            (str(data["n_features_whose_binned_curve_crosses_zero"]), "curves crossing zero"),
        ],
        body=body,
        footer=f"""
<p>Values are the <b>mean {esc(TARGET)}</b> per bin. This target is continuous, roughly SYMMETRIC and
CAN BE NEGATIVE (mean {s['mean']:+.6f}, skew {s['skew']:+.3f}, {s['pct_negative']:.1f}% negative,
{s['pct_zero']:.1f}% exactly zero) -- it is not a shot-rate percentage and not the zero-inflated, right-skewed
shape <code>target_future_xg_10s</code> has. <b>Shape classification</b> uses the unmodified
<code>classify_shape()</code> heuristic, but its flat margin ({s['flat_margin']:.6f}) is translated from the xg
suite's "0.5x overall mean" convention via xg's own standard-deviation fraction
({s['xg_flat_margin_as_fraction_of_xg_std']:.6f}), because a multiple of this target's own near-zero, negative
mean would be meaningless. <b>Train/test consistency</b> reuses the canonical match-grouped split
(outputs/models/splits/match_assignment.json), checked whenever |rho| &ge; {data['thresholds_carried_over_unchanged']['rho_threshold']}
or the bin range exceeds {s['range_trigger']:.6f}.</p>
<p><b>Prompt 64 cross-reference.</b> {esc(xc.PROMPT_64_CLEARANCE_FINDING)} No numerical feature in this pool is a
direct proxy for that artefact, but any feature derived from the action's own pitch location inherits the same
caveat: for clearances specifically, the location this target scores is where the clearance was taken from, not
where the ball went.</p>
<p>Separate output set from <code>reports/analysis/xg_target/</code> and
<code>reports/analysis/shot_target/</code> -- different target, different shape, different thresholds. Does not
change feature_config.py's locked candidate list.</p>""",
    )


def main() -> None:
    in_path = OUT_DIR / "active_numerical_target_atlas.json"
    out_path = OUT_DIR / "active_numerical_target_atlas.html"
    data = json.loads(in_path.read_text(encoding="utf-8"))
    html = build_report(data)
    html = html.replace("</style>", NUM_TARGET_CSS + XT_NUM_CSS + "</style>", 1)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
