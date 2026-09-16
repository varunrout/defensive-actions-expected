"""CLI entrypoint: render reports/eda_xg/{active,passive}_numerical_target_atlas.json
as self-contained HTML reports. Visually the same system as the binary-target
version (generate_numerical_target_reports.py, reused CSS/markup shape), but
values are "mean xG" not "shot rate %" -- the binary version's report script
is not imported here beyond that shared CSS block, since the bin values,
units, and thresholds differ throughout.

Usage:
    python -m src.eda.generate_numerical_xg_target_reports
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda.render import esc, finding_card, findings_grid, html_shell
from src.eda.feature_config import DATASETS
from src.eda.generate_numerical_target_reports import NUM_TARGET_CSS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "eda_xg"


def _bin_rows_html(bins: list[dict]) -> str:
    if not bins:
        return '<p style="font-size:11.5px; color:var(--text-muted);">no bins</p>'
    max_val = max((b["mean_xg"] for b in bins), default=1.0) or 1.0
    rows = []
    for b in bins:
        pct = min(100.0, (b["mean_xg"] / max_val) * 100) if max_val else 0.0
        rows.append(f"""
<div class="nt-bin-row" title="{esc(b['bin'])}: n={b['n']}, mean xG={b['mean_xg']}">
  <span class="nt-bin-label">{esc(b['bin'])}</span>
  <div class="nt-bin-track"><div class="nt-bin-fill" style="width:{pct:.1f}%"></div></div>
  <span class="nt-bin-rate">{b['mean_xg']:.4f}</span>
  <span class="nt-bin-n">n={b['n']:,}</span>
</div>""")
    return "".join(rows)


def _consistency_html(cc: dict) -> str:
    if not cc.get("checked"):
        return f'<p style="font-size:11.5px; color:var(--text-muted);">Not checked -- {esc(cc.get("reason", ""))}</p>'
    consistent = cc["consistent"]
    note_cls = "nt-inconsistent-note" if not consistent else ""
    verdict = (
        f"Consistent -- both splits classify as <b>{esc(cc['train_val_shape'])}</b>." if consistent else
        f'<b>Inconsistent</b> -- train+val shows <b>{esc(cc["train_val_shape"])}</b>, test shows <b>{esc(cc["test_shape"])}</b>. '
        "Treat this pattern as noise, not a confirmed finding."
    )
    note_html = f'<div class="{note_cls}">{verdict}</div>' if not consistent else f'<p style="font-size:11.5px; color:var(--text-secondary);">{verdict}</p>'
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
        body = f"""
<p><b>Correlation:</b> Pearson r={f['pearson_r']:+.4f} &middot; Spearman &rho;={rho:+.4f} &middot;
binning: {esc(f['binning_method'])} &middot; n={f['n_rows_used']:,}</p>
<p><b>Binned mean xG</b> (bin range {f['bin_range']:.5f}):</p>
{_bin_rows_html(f['bins'])}
<p><b>Train/test consistency check:</b></p>
{_consistency_html(f['consistency_check'])}
"""
    else:
        body = f'<p style="color:var(--text-muted);">Insufficient data to analyze ({f["consistency_check"].get("reason", "")}).</p>'

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


def build_report(data: dict) -> str:
    dataset_cfg = DATASETS[data["dataset"]]
    features = data["features"]
    max_abs_rho = max((abs(f["spearman_rho"]) for f in features if f["spearman_rho"] is not None), default=1.0) or 1.0

    pool = data["pool_construction"]
    findings = [
        finding_card(
            "pool",
            f"{pool['n_total']} features analyzed",
            f"{pool['n_locked']} locked + {pool['n_dropped']} dropped-but-included -- same reconstructed pre-drop "
            "numerical candidate pool as the binary-target atlas, run against the continuous target instead.",
        ),
        finding_card(
            "scale",
            f"overall mean {esc(data['target'])} = {data['overall_mean_target']}",
            f"heavily zero-inflated, nothing like a 0-100% rate -- shape-classification flat margin is "
            f"{data['flat_margin_ratio']}x this mean ({data['flat_margin']}), and the consistency-check range "
            f"trigger is {data['range_trigger_ratio']}x this mean ({data['range_trigger']}), not a fixed "
            "percentage-point margin.",
        ),
    ]
    if data["n_features_flagged_inconsistent"]:
        findings.append(
            finding_card(
                "caution",
                f"{data['n_features_flagged_inconsistent']} feature(s) flagged train/test-inconsistent",
                "the shape classification differs between train+val and test matches -- treat as noise, "
                "not a confirmed pattern, until re-examined.",
                flag=True,
            )
        )
    unreliable_features = [f["feature"] for f in features if f.get("unreliable_note")]
    if unreliable_features:
        findings.append(
            finding_card(
                "data issue",
                ", ".join(unreliable_features),
                "flagged unreliable pending a known self-reference bug -- see the Distribution Atlas.",
                flag=True,
            )
        )

    cards_html = "".join(_feature_card(i + 1, f, max_abs_rho) for i, f in enumerate(features))

    body = findings_grid(findings) + f"""
<h2 class="section-title">Ranked by |Spearman &rho;| vs {esc(data['target'])}</h2>
<p class="section-note">Click a row to expand its binned mean-xG curve. Green left border = locked (current
34/38), amber = dropped (still included -- see reason inside), red = flagged unreliable. Amber "TRAIN/TEST
MISMATCH" badge = shape classification disagrees between train+val and test matches.</p>
<div class="nt-list">{cards_html}</div>
"""

    return html_shell(
        eyebrow=f"NUMERICAL XG TARGET ATLAS · {dataset_cfg['label'].upper()}",
        title=f"{dataset_cfg['label']}: Numerical Features vs {data['target']}",
        dek=(
            f"Every numerical feature's relationship to the CONTINUOUS xG target, on the same reconstructed "
            f"pre-drop candidate pool ({pool['n_total']} features) as the binary-target atlas, "
            f"{esc(dataset_cfg['row_description'])}."
        ),
        stats=[
            (str(pool["n_total"]), "features analyzed"),
            (str(pool["n_locked"]), "locked"),
            (str(pool["n_dropped"]), "dropped, still shown"),
            (str(data["n_features_flagged_inconsistent"]), "train/test mismatch"),
        ],
        body=body,
        footer=f"""
<p>Values are <b>mean {esc(data['target'])}</b> per bin, not a shot-rate percentage -- this target is continuous
and heavily zero-inflated (overall mean {data['overall_mean_target']}). <b>Shape classification</b>: flat if the
bin-range is under {data['flat_margin_ratio']}x the overall mean; U-shaped/inverse-U if both ends average that
much above/below the middle third; monotonic increasing/decreasing if |Spearman rho(bin index, bin value)| >= 0.7;
otherwise no-clear-pattern. <b>Train/test consistency</b> reuses the canonical match-grouped split
(outputs/models/splits/match_assignment.json) -- checked whenever |rho| >= {data['rho_threshold']} or the bin
range exceeds {data['range_trigger_ratio']}x the overall mean. This is a separate output set from the
binary-target atlas (reports/eda/) -- different target, different scale, different thresholds. Does not change
feature_config.py's locked candidate list.</p>""",
    )


def main() -> None:
    for dataset_key in ("active", "passive"):
        in_path = OUT_DIR / f"{dataset_key}_numerical_target_atlas.json"
        out_path = OUT_DIR / f"{dataset_key}_numerical_target_atlas.html"
        data = json.loads(in_path.read_text(encoding="utf-8"))
        html = build_report(data)
        html = html.replace("</style>", NUM_TARGET_CSS + "</style>", 1)
        out_path.write_text(html, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
