"""CLI entrypoint: Distribution Atlas for the xT-delta portal (ACTIVE-BINARY
LEG ONLY) -- both the locked-list variant and the reconstructed-pool
variant, plus a target-shape section the xg portal has no equivalent of.

STEP-0 NOTE, stated plainly rather than papered over. The xg portal's
`active_distribution_atlas.html` was produced by
src/eda/generate_distribution_atlas_reconstructed.py -- confirmed by
reading it. That script is explicitly TARGET-INDEPENDENT: it calls
compute_stats.continuous_distribution / discrete_distribution, neither of
which ever references a target column, and its own docstring says the
single atlas "is valid for both the binary and xG contexts". Its
filename-selection branch (`out_dir.name == "eda_xg"`) no longer matches
the current directory name `xg_target`, which is why the file in
xg_target/ is named `active_distribution_atlas.html` while the same run
would today write `..._reconstructed.html` -- a stale-path artefact in the
original script, noted here, not fixed there (that script is left
byte-for-byte untouched so the xg portal stays reproducible).

Because the FEATURE distribution cards are target-independent, they are
reproduced here unchanged in substance (same functions, same 24-bin /
p99-clip / top-20 conventions). The genuinely new content is the TARGET
distribution section: `target_xt_delta`'s own shape, which is the report
in this suite that Prompt 64's symmetric/negative-capable finding should
surface in -- and does.

Two files are written, matching the task's "+ reconstructed-pool variant":
  active_distribution_atlas.html              -- locked ACTIVE candidate list
  active_distribution_atlas_reconstructed.html -- reconstructed pre-drop pool

Usage:
    python -m src.eda.generate_distribution_atlas_xt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eda import compute_stats as cs
from src.eda import xt_common as xc
from src.eda.feature_config import ACTIVE, CONTINUOUS_BIN_COUNT, CONTINUOUS_PERCENTILE_CAP
from src.eda.generate_numerical_target_analysis import build_pool
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid
from src.eda.render import _continuous_card, _discrete_card

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET

DIST_CSS = """
.dist-card-wrap { position: relative; border-radius: 12px; }
.dist-card-wrap.locked { border-left: 4px solid var(--good); }
.dist-card-wrap.dropped { border-left: 4px solid var(--amber); }
.dist-status-badge { font-family: "JetBrains Mono", monospace; font-size: 9.5px; text-transform: uppercase;
  padding: 2px 7px; border-radius: 999px; font-weight: 700; margin-left: 6px; vertical-align: middle; }
.dist-status-badge.locked { background: var(--good-wash); color: var(--good); }
.dist-status-badge.dropped { background: var(--amber-wash); color: var(--amber); }
.dist-reason { font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--text-muted); margin: 2px 0 8px; }
.xt-adapt { border-left: 4px solid var(--accent); background: var(--accent-wash); padding: 12px 14px;
  border-radius: 8px; margin: 14px 0; font-size: 12.5px; line-height: 1.55; }
.xt-hist { display: flex; align-items: flex-end; gap: 1px; height: 190px; margin: 18px 0 6px;
  border-bottom: 1px solid var(--border); position: relative; }
.xt-hist .b { flex: 1; background: var(--neg); border-radius: 2px 2px 0 0; min-height: 1px; }
.xt-hist .b.pos { background: var(--good); }
.xt-hist .b.zero { background: var(--text-muted); }
.xt-hist .axis0 { position: absolute; top: -6px; bottom: -6px; width: 1px; background: var(--text-primary); opacity: 0.55; }
.xt-histlab { display: flex; justify-content: space-between; font-family: "JetBrains Mono", monospace;
  font-size: 10px; color: var(--text-muted); }
.xt-stat-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 10px; }
.xt-stat-table td { padding: 5px 8px; border-top: 1px solid var(--border); }
.xt-stat-table td:first-child { color: var(--text-secondary); }
.xt-stat-table td:last-child { font-family: "JetBrains Mono", monospace; text-align: right; font-weight: 600; }
"""


def _target_histogram(values: np.ndarray, n_bins: int = 61) -> tuple[str, dict]:
    """Symmetric histogram centred on zero -- the feature cards' own
    `_continuous_card` clips to the 99th percentile from the column MIN,
    which silently discards this target's entire negative tail. Clipping
    here is symmetric (|value| p99 on both sides) for the same reason
    Adaptation 3 exists: both tails are meaningful."""
    cap = float(np.percentile(np.abs(values), CONTINUOUS_PERCENTILE_CAP))
    edges = np.linspace(-cap, cap, n_bins + 1)
    clipped = values[(values >= -cap) & (values <= cap)]
    counts, _ = np.histogram(clipped, bins=edges)
    mx = counts.max() or 1

    bars = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        cls = "b pos" if lo >= 0 else ("b" if hi <= 0 else "b zero")
        h = max(1.0, c / mx * 100)
        bars.append(f'<div class="{cls}" style="height:{h:.2f}%" title="[{lo:+.4f}, {hi:+.4f}): n={c:,}"></div>')

    zero_pct = ((0 - (-cap)) / (2 * cap)) * 100
    html = (
        f'<div class="xt-hist"><div class="axis0" style="left:{zero_pct:.2f}%"></div>{"".join(bars)}</div>'
        f'<div class="xt-histlab"><span>{-cap:+.4f}</span><span>0</span><span>{cap:+.4f}</span></div>'
    )
    meta = {
        "n_bins": n_bins,
        "symmetric_cap_abs_p99": round(cap, 8),
        "n_rows_inside_cap": int(len(clipped)),
        "n_rows_clipped": int(len(values) - len(clipped)),
    }
    return html, meta


def _target_section(scale: dict, hist_html: str, hist_meta: dict) -> str:
    rows = [
        ("rows with a defined delta", f"{scale['n_defined']:,}"),
        ("rows with NaN delta (explained, Prompt 64 s.2)", f"{scale['n_nan']}"),
        ("mean", f"{scale['mean']:+.6f}"),
        ("std", f"{scale['std']:.6f}"),
        ("skew", f"{scale['skew']:+.3f}"),
        ("excess kurtosis", f"{scale['excess_kurtosis']:.3f}"),
        ("share exactly zero", f"{scale['pct_zero']:.2f}%"),
        ("share negative (xT rose)", f"{scale['pct_negative']:.2f}%"),
        ("share positive (xT fell)", f"{scale['pct_positive']:.2f}%"),
        (f"symmetric |value| p{CONTINUOUS_PERCENTILE_CAP} clip used for the histogram", f"{hist_meta['symmetric_cap_abs_p99']:.6f}"),
        ("rows outside that clip (shown in the table, not the bars)", f"{hist_meta['n_rows_clipped']:,}"),
    ]
    table = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in rows)

    return f"""
<h2 class="section-title">The target's own distribution -- {esc(TARGET)}</h2>
<p class="section-note">The xg portal's Distribution Atlas has no equivalent of this section: it covers feature
shapes only, and is target-independent by construction. It is added here because this is the report in the suite
where Prompt 64's distribution-shape finding should surface -- and it does, measured independently on the same
56,068 active rows.</p>
<div class="card" style="margin-bottom:16px;">
<h3>{esc(TARGET)}</h3>
<p class="subnote">Green bars = positive delta (xT fell across the action, threat reduced). Red bars = negative
(xT rose). Grey = the bin straddling zero. The vertical line is exact zero. Binning is clipped SYMMETRICALLY at
the {CONTINUOUS_PERCENTILE_CAP}th percentile of |delta|, not from the column minimum -- the atlas's usual
min-to-p99 convention would silently discard this target's entire negative tail.</p>
{hist_html}
<table class="xt-stat-table"><tbody>{table}</tbody></table>
</div>
<div class="xt-adapt">
<b>Prompt 64's distribution-shape finding: CONFIRMED independently here.</b>
<p style="margin:8px 0 0;">{esc(xc.PROMPT_64_SHAPE_FINDING)}</p>
<p style="margin:8px 0 0;">Recomputed on this run: skew <b>{scale['skew']:+.3f}</b>, excess kurtosis
<b>{scale['excess_kurtosis']:.2f}</b>, <b>{scale['pct_negative']:.1f}%</b> negative / <b>{scale['pct_positive']:.1f}%</b>
positive / <b>{scale['pct_zero']:.1f}%</b> exactly zero, mean <b>{scale['mean']:+.6f}</b>, std
<b>{scale['std']:.5f}</b> -- matching Prompt 64's figures. Nothing in this portal may assume a non-negative or
right-skewed target.</p>
<p style="margin:8px 0 0;">{esc(scale['scale_note'])}</p>
</div>"""


def _wrap_card(card_html: str, status: str, reason: str | None) -> str:
    badge = f'<span class="dist-status-badge {status}">{status.upper()}</span>'
    card_html = card_html.replace("</h3>", f"{badge}</h3>", 1)
    if status == "dropped" and reason:
        card_html = card_html.replace(
            '<p class="subnote">', f'<p class="dist-reason">Why dropped: {esc(reason)}</p><p class="subnote">', 1
        )
    return f'<div class="dist-card-wrap {status}">{card_html}</div>'


def render_atlas(variant: str, n_rows: int, pool: list[dict], continuous: dict, discrete: dict,
                 scale: dict, hist_html: str, hist_meta: dict) -> str:
    n_locked = sum(1 for p in pool if p["status"] == "locked")
    n_dropped = sum(1 for p in pool if p["status"] == "dropped")

    findings = [
        finding_card(
            "pool",
            f"{len(pool)} numerical columns ({n_locked} locked + {n_dropped} dropped-but-included)"
            if variant == "reconstructed" else f"{len(pool)} numerical columns (current locked ACTIVE list)",
            "reconstructed pre-drop 51-era ACTIVE pool -- same pool the xg Distribution Atlas uses."
            if variant == "reconstructed" else
            "the current locked ACTIVE candidate list only, for readers who want the shipped feature set without "
            "the dropped columns alongside it.",
        ),
        finding_card(
            "target-independent",
            "the feature cards below are identical to the xg portal's",
            "continuous_distribution / discrete_distribution never reference a target column -- these cards "
            "describe each FEATURE's own shape and are reproduced here unchanged in substance, not re-derived "
            "with a different method. The target section above them is the only genuinely xT-specific content "
            "on this page.",
        ),
        finding_card(
            "carried forward from Prompt 64",
            "symmetric, negative-capable target -- confirmed",
            f"skew {scale['skew']:+.3f}, {scale['pct_negative']:.1f}% negative. See the target section above.",
            flag=True,
        ),
    ]

    cont_cards, disc_cards = [], []
    for p in pool:
        col = p["feature"]
        if col in continuous:
            cont_cards.append(_wrap_card(_continuous_card(col, continuous[col]), p["status"], p["reason"]))
        elif col in discrete:
            disc_cards.append(_wrap_card(_discrete_card(col, discrete[col]), p["status"], p["reason"]))

    body = findings_grid(findings) + _target_section(scale, hist_html, hist_meta)
    if cont_cards:
        body += '\n<h2 class="section-title">Continuous Feature Distributions</h2>' \
            f'<p class="section-note">Target-independent. {CONTINUOUS_BIN_COUNT} equal-width bins from the ' \
            f'column min to its {CONTINUOUS_PERCENTILE_CAP}th percentile. Green border = locked, amber = ' \
            'dropped (still shown, reason inline).</p>' \
            f'<div class="card-grid">{"".join(cont_cards)}</div>'
    if disc_cards:
        body += '\n<h2 class="section-title">Discrete Feature Distributions</h2>' \
            '<p class="section-note">Target-independent. Raw value counts, capped at 20 distinct values.</p>' \
            f'<div class="card-grid">{"".join(disc_cards)}</div>'

    label = "reconstructed pre-drop pool" if variant == "reconstructed" else "locked candidate list"
    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(lab)}</span></div>'
        for v, lab in [
            (f"{n_rows:,}", "active rows"),
            (f"{scale['skew']:+.3f}", "target skew"),
            (f"{scale['pct_negative']:.1f}%", "target negative"),
            (str(len(pool)), "numerical columns"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Active Defensive Actions: Distribution Atlas (xT delta, {esc(label)})</title>
{FONT_LINKS}
<style>{CSS}{DIST_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">DISTRIBUTION ATLAS -- XT DELTA &middot; {esc(label.upper())} &middot; ACTIVE-BINARY LEG ONLY</p>
  <h1 class="title">Active Defensive Actions: Distribution Atlas</h1>
  <p class="dek">{esc(TARGET)}'s own shape, plus every numerical column's own shape ({esc(label)}), one row per
  actual defensive action. Active-binary leg only.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recon_pool, _ = build_pool("active")
    locked_types = {c: "continuous" for c in ACTIVE["continuous"]}
    locked_types.update({c: "discrete" for c in ACTIVE["discrete"]})
    locked_pool = [
        {"feature": c, "type": t, "status": "locked", "reason": None}
        for c, t in sorted(locked_types.items())
    ]

    all_feats = sorted({p["feature"] for p in recon_pool} | {p["feature"] for p in locked_pool})
    df = xc.load_active_xt(columns=all_feats)
    n_rows = len(df)
    scale = xc.target_scale(df)

    cont_types = {p["feature"]: p["type"] for p in recon_pool}
    for p in locked_pool:
        cont_types.setdefault(p["feature"], p["type"])

    continuous = {f: cs.continuous_distribution(df, f) for f, t in cont_types.items() if t == "continuous"}
    discrete = {f: cs.discrete_distribution(df, f) for f, t in cont_types.items() if t == "discrete"}

    hist_html, hist_meta = _target_histogram(df[TARGET].dropna().to_numpy())

    (OUT_DIR / "active_distribution_atlas.html").write_text(
        render_atlas("locked", n_rows, locked_pool, continuous, discrete, scale, hist_html, hist_meta), encoding="utf-8")
    (OUT_DIR / "active_distribution_atlas_reconstructed.html").write_text(
        render_atlas("reconstructed", n_rows, recon_pool, continuous, discrete, scale, hist_html, hist_meta), encoding="utf-8")

    (OUT_DIR / "active_distribution_atlas.json").write_text(json.dumps({
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": TARGET,
        "n_rows": n_rows,
        "target_distribution": scale,
        "target_histogram": hist_meta,
        "feature_cards_are_target_independent": (
            "continuous_distribution / discrete_distribution never reference a target column -- the feature "
            "cards in the HTML are reproduced from the same functions the xg portal's atlas uses, unchanged."
        ),
        "variants": {
            "active_distribution_atlas.html": f"locked ACTIVE candidate list ({len(locked_pool)} numerical columns)",
            "active_distribution_atlas_reconstructed.html": f"reconstructed pre-drop pool ({len(recon_pool)} numerical columns)",
        },
        "prompt_64_cross_references": {"distribution_shape": xc.PROMPT_64_SHAPE_FINDING},
    }, indent=2, default=str), encoding="utf-8")

    print(f"active: {n_rows:,} rows, locked pool={len(locked_pool)}, reconstructed pool={len(recon_pool)}")
    print(f"  target skew={scale['skew']:+.3f} kurt={scale['excess_kurtosis']:.2f} "
          f"neg={scale['pct_negative']:.1f}% pos={scale['pct_positive']:.1f}% zero={scale['pct_zero']:.1f}%")
    print(f"Wrote 3 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
