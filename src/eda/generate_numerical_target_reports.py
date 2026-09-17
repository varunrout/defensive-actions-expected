"""CLI entrypoint: render {active,passive}_numerical_target_atlas.json as
self-contained HTML reports, in the same shared design system as the other
reports/eda/*.html files (reuses render.html_shell + the finding/card
vocabulary already established by the Category Atlas / Flag Ledger).

Usage:
    python -m src.eda.generate_numerical_target_reports
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda.render import esc, finding_card, findings_grid, html_shell
from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_TARGET_CSS = """
.nt-list { display: flex; flex-direction: column; gap: 8px; }
.nt-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.nt-card.dropped { border-left: 4px solid var(--amber); }
.nt-card.locked { border-left: 4px solid var(--good); }
.nt-card.unreliable { border-left: 4px solid var(--pos); }
.nt-summary { display: grid; grid-template-columns: 28px 1fr 90px 160px 130px 100px; align-items: center; gap: 12px;
  padding: 10px 14px; cursor: pointer; list-style: none; font-size: 13px; }
.nt-summary::-webkit-details-marker { display: none; }
.nt-rank { font-family: "JetBrains Mono", monospace; color: var(--text-muted); font-size: 11px; }
.nt-name { font-weight: 600; color: var(--text-primary); }
.nt-name .mono-sub { display: block; font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--text-muted); font-weight: 400; }
.nt-rho-track { position: relative; height: 8px; background: var(--gridline); border-radius: 4px; }
.nt-rho-track .zero { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--text-muted); opacity: 0.5; }
.nt-rho-fill { position: absolute; top: 0; height: 100%; border-radius: 4px; }
.nt-rho-fill.pos { left: 50%; background: var(--neg); }
.nt-rho-fill.neg { right: 50%; background: var(--pos); }
.nt-rho-val { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-secondary); text-align: right; margin-top: 2px; }
.nt-shape { font-family: "JetBrains Mono", monospace; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.02em; color: var(--text-secondary); text-align: center; }
.nt-badges { display: flex; gap: 4px; justify-content: flex-end; flex-wrap: wrap; }
.nt-badge { font-family: "JetBrains Mono", monospace; font-size: 9.5px; text-transform: uppercase; padding: 2px 6px; border-radius: 999px; font-weight: 700; white-space: nowrap; }
.nt-badge.locked { background: var(--good-wash); color: var(--good); }
.nt-badge.dropped { background: var(--amber-wash); color: var(--amber); }
.nt-badge.unreliable { background: rgba(227,73,72,0.14); color: var(--pos); }
.nt-badge.inconsistent { background: rgba(237,161,0,0.18); color: var(--amber); }
.nt-body { padding: 4px 16px 16px; border-top: 1px dashed var(--border); }
.nt-body p { font-size: 12.5px; color: var(--text-secondary); margin: 10px 0 6px; }
.nt-bin-row { display: grid; grid-template-columns: 140px 1fr 70px 50px; align-items: center; gap: 8px; font-size: 11.5px; margin-bottom: 3px; }
.nt-bin-label { font-family: "JetBrains Mono", monospace; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nt-bin-track { position: relative; height: 8px; background: var(--gridline); border-radius: 4px; }
.nt-bin-fill { position: absolute; top: 0; left: 0; height: 100%; background: var(--neg); border-radius: 4px; }
.nt-bin-rate { font-family: "JetBrains Mono", monospace; font-weight: 600; text-align: right; }
.nt-bin-n { font-family: "JetBrains Mono", monospace; color: var(--text-muted); text-align: right; font-size: 10.5px; }
.nt-split-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 8px; }
@media (max-width: 700px) { .nt-split-grid { grid-template-columns: 1fr; } .nt-summary { grid-template-columns: 24px 1fr 70px; } .nt-summary .nt-shape, .nt-summary .nt-badges { display: none; } }
.nt-split-title { font-family: "JetBrains Mono", monospace; font-size: 10.5px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
.nt-reason { background: var(--amber-wash); border: 1px solid rgba(237,161,0,0.35); border-radius: 6px; padding: 8px 10px; font-size: 11.5px; color: var(--text-secondary); margin: 6px 0; }
.nt-unreliable-note { background: rgba(227,73,72,0.10); border: 1px solid rgba(227,73,72,0.35); border-radius: 6px; padding: 8px 10px; font-size: 11.5px; color: var(--text-secondary); margin: 6px 0; }
.nt-inconsistent-note { background: var(--amber-wash); border: 1px solid rgba(237,161,0,0.35); border-radius: 6px; padding: 8px 10px; font-size: 11.5px; color: var(--text-secondary); margin: 6px 0; }
"""


def _bin_rows_html(bins: list[dict]) -> str:
    if not bins:
        return '<p style="font-size:11.5px; color:var(--text-muted);">no bins</p>'
    max_rate = max((b["shot_rate_pct"] for b in bins), default=1.0) or 1.0
    rows = []
    for b in bins:
        pct = min(100.0, (b["shot_rate_pct"] / max_rate) * 100) if max_rate else 0.0
        rows.append(f"""
<div class="nt-bin-row" title="{esc(b['bin'])}: n={b['n']}, rate={b['shot_rate_pct']}%">
  <span class="nt-bin-label">{esc(b['bin'])}</span>
  <div class="nt-bin-track"><div class="nt-bin-fill" style="width:{pct:.1f}%"></div></div>
  <span class="nt-bin-rate">{b['shot_rate_pct']}%</span>
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

    body = ""
    if f.get("bins"):
        body = f"""
<p><b>Correlation:</b> Pearson r={f['pearson_r']:+.4f} &middot; Spearman &rho;={rho:+.4f} &middot;
binning: {esc(f['binning_method'])} &middot; n={f['n_rows_used']:,}</p>
<p><b>Binned shot rate</b> (bin range {f['bin_range_pp']}pp):</p>
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
            f"{pool['n_locked']} locked + {pool['n_dropped']} dropped-but-included -- reconstructed pre-drop numerical "
            "candidate pool, not the current locked list. Redundancy/VIF drops never looked at the target.",
        ),
    ]
    if pool["excluded_as_coordinate_duplicate"]:
        findings.append(
            finding_card(
                "excluded",
                f"{len(pool['excluded_as_coordinate_duplicate'])} raw coordinate columns",
                "excluded as coordinate-frame duplicates of already-included ball-relative replacements -- "
                f"{', '.join(pool['excluded_as_coordinate_duplicate'])}.",
                flag=True,
            )
        )
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
<p class="section-note">Click a row to expand its binned shot-rate curve. Green left border = locked (current
34/38), amber = dropped (still included -- see reason inside), red = flagged unreliable. Amber "TRAIN/TEST
MISMATCH" badge = shape classification disagrees between train+val and test matches.</p>
<div class="nt-list">{cards_html}</div>
"""

    return html_shell(
        eyebrow=f"NUMERICAL TARGET ATLAS · {dataset_cfg['label'].upper()}",
        title=f"{dataset_cfg['label']}: Numerical Features vs {data['target']}",
        dek=(
            f"Every numerical feature's relationship to the binary target, on the reconstructed pre-drop "
            f"candidate pool ({pool['n_total']} features), {esc(dataset_cfg['row_description'])}."
        ),
        stats=[
            (str(pool["n_total"]), "features analyzed"),
            (str(pool["n_locked"]), "locked"),
            (str(pool["n_dropped"]), "dropped, still shown"),
            (str(data["n_features_flagged_inconsistent"]), "train/test mismatch"),
        ],
        body=body,
        footer="""
<p><b>Shape classification</b> is a stated heuristic, not read off a chart: flat if the bin-range is under 1.5pp;
U-shaped/inverse-U if both ends average >=1.5pp above/below the middle third; monotonic increasing/decreasing if
|Spearman rho(bin index, bin rate)| >= 0.7; otherwise no-clear-pattern. <b>Train/test consistency</b> reuses the
canonical match-grouped split (outputs/models/splits/match_assignment.json) -- checked whenever |rho| >= """
        + f"{data['rho_threshold']} or the bin range exceeds {data['range_threshold_pp']}pp."
        + """ This pass does not change feature_config.py's locked candidate list -- dropped features are shown
for visibility only.</p>""",
    )


def main() -> None:
    for dataset_key in ("active", "passive"):
        in_path = REPO_ROOT / "reports" / "eda" / f"{dataset_key}_numerical_target_atlas.json"
        out_path = REPO_ROOT / "reports" / "eda" / f"{dataset_key}_numerical_target_atlas.html"
        data = json.loads(in_path.read_text(encoding="utf-8"))
        html = build_report(data)
        html = html.replace("</style>", NUM_TARGET_CSS + "</style>", 1)
        out_path.write_text(html, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
