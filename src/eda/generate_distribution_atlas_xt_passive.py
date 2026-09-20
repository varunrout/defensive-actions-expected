"""CLI entrypoint: Distribution Atlas for the xT-delta portal -- PASSIVE LEG.
Both the locked-list variant and the reconstructed-pool variant, plus the
target-shape section the xg portal has no equivalent of.

STEP-0 DISPOSITION: genuine passive build (inherently per-leg). The xg
portal ships `active_distribution_atlas.html` and
`passive_distribution_atlas.html` as a literal pair; the `shot_target`
portal additionally ships `passive_distribution_atlas_reconstructed.html`,
so the two-variant convention DOES apply to the passive leg -- checked by
listing both directories rather than assumed from the active xT portal.
Both variants are therefore written here, matching what this portal's own
active half already does.

The FEATURE cards are target-independent by construction:
`compute_stats.continuous_distribution` / `discrete_distribution` never
reference a target column, and are imported and used unchanged. The
genuinely new content is the TARGET section -- `target_xt_delta_passive`'s
own shape, which is the report in this suite where Prompt 68's
distribution finding has to surface, and where the row-grain-vs-event-grain
difference has to be stated rather than papered over.

Usage:
    python -m src.eda.generate_distribution_atlas_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator import
from src.eda import compute_stats as cs
from src.eda import xt_common as xc
from src.eda import generate_distribution_atlas_xt as d1
from src.eda.feature_config import PASSIVE, CONTINUOUS_BIN_COUNT, CONTINUOUS_PERCENTILE_CAP
from src.eda.generate_numerical_target_analysis import build_pool
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid

OUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET


def _target_section_passive(scale: dict, hist_html: str, hist_meta: dict, n_events: int) -> str:
    g = scale["grain_comparison"]
    p68 = g["prompt_68_unique_event"]
    rows = [
        ("defender-slot rows with a defined delta", f"{scale['n_defined']:,}"),
        ("unique events behind those rows", f"{n_events:,}"),
        ("defender-slot rows with a NaN delta", f"{scale['n_nan']:,}"),
        ("mean", f"{scale['mean']:+.6f}"),
        ("std", f"{scale['std']:.6f}"),
        ("skew", f"{scale['skew']:+.3f}"),
        ("excess kurtosis", f"{scale['excess_kurtosis']:.3f}"),
        ("share exactly zero", f"{scale['pct_zero']:.2f}%"),
        ("share negative (xT rose)", f"{scale['pct_negative']:.2f}%"),
        ("share positive (xT fell)", f"{scale['pct_positive']:.2f}%"),
        ("IQR", f"{scale['robust_scale']['iqr']:.6f}"),
        ("MAD", f"{scale['robust_scale']['mad']:.6f}"),
        (f"symmetric |value| p{CONTINUOUS_PERCENTILE_CAP} clip used for the histogram",
         f"{hist_meta['symmetric_cap_abs_p99']:.6f}"),
        ("rows outside that clip (in the table, not the bars)", f"{hist_meta['n_rows_clipped']:,}"),
    ]
    table = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in rows)

    grain_rows = [
        ("mean", f"{p68['mean']:+.5f}", f"{scale['mean']:+.5f}"),
        ("std", f"{p68['std']:.5f}", f"{scale['std']:.5f}"),
        ("skew", f"{p68['skew']:+.3f}", f"{scale['skew']:+.3f}"),
        ("excess kurtosis", f"{p68['excess_kurtosis']:.2f}", f"{scale['excess_kurtosis']:.2f}"),
        ("% exactly zero", f"{p68['pct_zero']:.1f}%", f"{scale['pct_zero']:.1f}%"),
        ("% negative", f"{p68['pct_negative']:.1f}%", f"{scale['pct_negative']:.1f}%"),
        ("% positive", f"{p68['pct_positive']:.1f}%", f"{scale['pct_positive']:.1f}%"),
        ("n defined", f"{p68['n_defined']:,}", f"{scale['n_defined']:,}"),
    ]
    grain_table = "".join(
        f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in grain_rows
    )

    return f"""
<h2 class="section-title">The target's own distribution -- {esc(TARGET)}</h2>
<p class="section-note">The xg portal's Distribution Atlas has no equivalent of this section: it covers feature
shapes only and is target-independent by construction. It is added here for the same reason the active half of
this portal adds it -- this is the report in the suite where the distribution-shape finding has to surface --
and it carries one extra thing the active half does not need: this target lives at a different grain from the
rows it is measured on.</p>
<div class="card" style="margin-bottom:16px;">
<h3>{esc(TARGET)}</h3>
<p class="subnote">Green bars = positive delta (xT fell across the event, threat reduced). Red bars = negative
(xT rose). Grey = the bin straddling zero. The vertical line is exact zero. Binning is clipped SYMMETRICALLY at
the {CONTINUOUS_PERCENTILE_CAP}th percentile of |delta|, not from the column minimum -- the atlas's usual
min-to-p99 convention would silently discard this target's entire negative tail, which on this leg is the LONGER
of the two.</p>
{hist_html}
<table class="xt-stat-table"><tbody>{table}</tbody></table>
</div>
<div class="xt-adapt">
<b>Prompt 68's distribution finding: CONFIRMED in direction, and MORE extreme at the grain this suite measures
at.</b>
<p style="margin:8px 0 0;">{esc(pc.PROMPT_68_SHAPE_FINDING)}</p>
<p style="margin:12px 0 0;">Recomputed on this run at the DEFENDER-SLOT grain -- the grain every other report in
this half of the portal uses, because it is the grain the features live at: skew <b>{scale['skew']:+.3f}</b>,
excess kurtosis <b>{scale['excess_kurtosis']:.2f}</b>, <b>{scale['pct_negative']:.1f}%</b> negative /
<b>{scale['pct_positive']:.1f}%</b> positive / <b>{scale['pct_zero']:.1f}%</b> exactly zero, mean
<b>{scale['mean']:+.6f}</b>, std <b>{scale['std']:.5f}</b>. Prompt 68 reported its figures at the UNIQUE-EVENT
grain. The two are the same target measured two ways, not a disagreement:</p>
<table class="xt-stat-table" style="margin-top:10px;"><tbody>
<tr><td><b>statistic</b></td><td><b>unique event (Prompt 68)</b></td><td><b>defender slot (this suite)</b></td></tr>
{grain_table}
</tbody></table>
<p style="margin:10px 0 0;">{esc(g['note'])}</p>
<p style="margin:10px 0 0;">{esc(scale['scale_note'])}</p>
<p style="margin:10px 0 0;"><b>Threshold consequence, recomputed rather than inherited.</b>
{esc(scale['threshold_transfer_warning'])}</p>
</div>"""


def render_atlas_passive(variant: str, n_rows: int, n_events: int, pool: list[dict], continuous: dict,
                         discrete: dict, scale: dict, hist_html: str, hist_meta: dict) -> str:
    n_locked = sum(1 for p in pool if p["status"] == "locked")
    n_dropped = sum(1 for p in pool if p["status"] == "dropped")
    rs = scale["robust_scale"]

    findings = [
        finding_card(
            "pool",
            f"{len(pool)} numerical columns ({n_locked} locked + {n_dropped} dropped-but-included)"
            if variant == "reconstructed" else f"{len(pool)} numerical columns (current locked PASSIVE list)",
            "reconstructed pre-drop 44-era PASSIVE pool -- the same pool the passive xg Distribution Atlas "
            "uses, including this leg's own screening and lane columns, which have no active counterpart."
            if variant == "reconstructed" else
            "the current locked PASSIVE candidate list only, for readers who want the shipped feature set "
            "without the dropped columns alongside it.",
        ),
        finding_card(
            "target-independent",
            "the feature cards below are identical to the passive xg portal's",
            "continuous_distribution / discrete_distribution never reference a target column -- these cards "
            "describe each FEATURE's own shape and are reproduced here unchanged in substance, not re-derived "
            "with a different method. The target section above them is the only genuinely xT-specific content "
            "on this page.",
        ),
        finding_card(
            "carried forward from Prompt 68 -- CONFIRMED, and more extreme at this grain",
            "left-skewed, very heavy-tailed, negative-capable target",
            f"skew {scale['skew']:+.3f} and excess kurtosis {scale['excess_kurtosis']:.1f} at the "
            "defender-slot grain, against -1.779 / 31.02 at Prompt 68's unique-event grain and -0.306 / 18.33 "
            "for the active leg's own target. See the target section above.",
            flag=True,
        ),
        finding_card(
            "threshold caveat",
            f"the std-derived flat margin spans {rs['pct_rows_inside_flat_margin']:.1f}% of rows",
            f"{rs['flat_margin_over_mad']:.2f}x this leg's own MAD, recomputed from the passive parquet rather "
            "than transferred from active's 1.28x / 53.4%. Every shape label in this half of the portal is "
            "indicative rather than decided.",
            flag=True,
        ),
    ]

    cont_cards, disc_cards = [], []
    for p in pool:
        col = p["feature"]
        if col in continuous:
            cont_cards.append(d1._wrap_card(d1._continuous_card(col, continuous[col]), p["status"], p["reason"]))
        elif col in discrete:
            disc_cards.append(d1._wrap_card(d1._discrete_card(col, discrete[col]), p["status"], p["reason"]))

    body = findings_grid(findings) + _target_section_passive(scale, hist_html, hist_meta, n_events)
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
            (f"{n_rows:,}", "passive rows"),
            (f"{n_events:,}", "unique events"),
            (f"{scale['skew']:+.3f}", "target skew"),
            (f"{scale['excess_kurtosis']:.1f}", "excess kurtosis"),
            (str(len(pool)), "numerical columns"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Passive / Off-Ball Defensive Positioning: Distribution Atlas (xT delta, {esc(label)})</title>
{FONT_LINKS}
<style>{CSS}{d1.DIST_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">DISTRIBUTION ATLAS -- XT DELTA &middot; {esc(label.upper())} &middot; PASSIVE LEG</p>
  <h1 class="title">Passive / Off-Ball Defensive Positioning: Distribution Atlas</h1>
  <p class="dek">{esc(TARGET)}'s own shape, plus every numerical column's own shape ({esc(label)}), one row per
  visible defender-slot per on-ball attacking event.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recon_pool, _ = build_pool("passive")
    locked_types = {c: "continuous" for c in PASSIVE["continuous"]}
    locked_types.update({c: "discrete" for c in PASSIVE["discrete"]})
    locked_pool = [
        {"feature": c, "type": t, "status": "locked", "reason": None}
        for c, t in sorted(locked_types.items())
    ]

    all_feats = sorted({p["feature"] for p in recon_pool} | {p["feature"] for p in locked_pool})
    df = xc.load_active_xt(columns=all_feats)
    n_rows = len(df)
    n_events = int(df["event_id"].nunique())
    scale = xc.target_scale(df)

    types = {p["feature"]: p["type"] for p in recon_pool}
    for p in locked_pool:
        types.setdefault(p["feature"], p["type"])

    continuous = {f: cs.continuous_distribution(df, f) for f, t in types.items() if t == "continuous"}
    discrete = {f: cs.discrete_distribution(df, f) for f, t in types.items() if t == "discrete"}

    hist_html, hist_meta = d1._target_histogram(df[TARGET].dropna().to_numpy())

    (OUT_DIR / "passive_distribution_atlas.html").write_text(
        render_atlas_passive("locked", n_rows, n_events, locked_pool, continuous, discrete, scale,
                             hist_html, hist_meta), encoding="utf-8")
    (OUT_DIR / "passive_distribution_atlas_reconstructed.html").write_text(
        render_atlas_passive("reconstructed", n_rows, n_events, recon_pool, continuous, discrete, scale,
                             hist_html, hist_meta), encoding="utf-8")

    (OUT_DIR / "passive_distribution_atlas.json").write_text(json.dumps({
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg)",
        "target": TARGET,
        "n_rows": n_rows,
        "n_unique_events": n_events,
        "target_distribution": scale,
        "target_histogram": hist_meta,
        "feature_cards_are_target_independent": (
            "continuous_distribution / discrete_distribution never reference a target column -- the feature "
            "cards in the HTML are reproduced from the same functions the passive xg portal's atlas uses, "
            "unchanged."
        ),
        "two_variant_convention_check": (
            "CONFIRMED by listing the directories rather than assumed: reports/analysis/shot_target/ ships "
            "passive_distribution_atlas.html AND passive_distribution_atlas_reconstructed.html, so the "
            "two-variant convention does apply to the passive leg. reports/analysis/xg_target/ ships only the "
            "single passive_distribution_atlas.html, which is an artefact of the stale filename branch in "
            "generate_distribution_atlas_reconstructed.py that Prompt 65 already recorded for the active side "
            "-- not a decision that the passive leg gets one variant. Both are written here, matching this "
            "portal's own active half."
        ),
        "prompt_68_cross_references": {
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "no_v1_detour": pc.PROMPT_68_NO_V1_DETOUR_FINDING,
        },
        "variants": {
            "passive_distribution_atlas.html": f"locked PASSIVE candidate list ({len(locked_pool)} numerical columns)",
            "passive_distribution_atlas_reconstructed.html": f"reconstructed pre-drop pool ({len(recon_pool)} numerical columns)",
        },
    }, indent=2, default=str), encoding="utf-8")

    print(f"passive: {n_rows:,} rows over {n_events:,} events, locked pool={len(locked_pool)}, "
          f"reconstructed pool={len(recon_pool)}")
    print(f"  target skew={scale['skew']:+.3f} kurt={scale['excess_kurtosis']:.2f} "
          f"neg={scale['pct_negative']:.1f}% pos={scale['pct_positive']:.1f}% zero={scale['pct_zero']:.1f}%")
    print(f"Wrote 3 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
