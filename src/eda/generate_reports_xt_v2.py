"""CLI entrypoint: Category Atlas + Flag Ledger against `target_xt_delta_v2`
(ACTIVE-BINARY LEG ONLY).

STEP-0 DISPOSITION: REWRITE, not a plain re-point.

`generate_reports_xt.py` is target-value-agnostic in all of its COMPUTATION
-- pool construction, `categorical_xt_table`, `boolean_xt_lift`, the
diverging-bar geometry and the JSON key set all come through `xt_common`
and need nothing but the re-point. Three pieces of its PROSE, however, bake
in v1-specific findings that v2 disproves, and re-pointing alone would have
published false statements:

  1. `_clearance_note_for()` asserts "Clearance / action_x artefact --
     confirmed present in this atlas" and instructs the reader not to read
     the row as "clearances are bad defending". Under v2 Clearance is the
     BEST event type, not the worst (Prompt 66 s.2.2: 8th of 8 at -0.0325
     -> 1st of 8 at +0.0247). The note is rewritten to report the REVERSAL,
     which is the finding this atlas actually carries now.

  2. `_possession_flag_note()` states the three `action_*_possession` flags
     are "roughly 77-79% non-zero" against the target. That figure is v1's.
     Under v2 two of those flags are the very trigger for `xt_after = 0`,
     so their True groups collapse onto `xt_before` exactly (Prompt 66
     s.2.1) -- a construction identity, not a measurement. Every number in
     the rewritten note is computed at runtime, and the identity is checked
     on this run rather than quoted.

  3. `render_category_atlas_xt()`'s scale finding-card says "roughly
     symmetric". v2's skew is -0.306, not +0.093.

Everything else is reused by direct import from the v1 module, so the two
portals stay methodologically identical wherever v2 did not force a change.
`generate_reports_xt.py` itself is NOT edited -- it still regenerates the
preserved v1 portal exactly.

Usage:
    python -m src.eda.generate_reports_xt_v2
"""

from __future__ import annotations

import json

import numpy as np

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import xt_common as xc
from src.eda import generate_reports_xt as r1
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid

OUTPUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET


def _clearance_note_for_v2(col: str, rows: list[dict]) -> str:
    """REWRITTEN for v2. The v1 version announced the Clearance artefact as
    "confirmed present"; on v2 the finding this atlas carries is that the
    artefact has REVERSED, so the note reports the reversal and cites
    Prompt 66's own numbers rather than re-deriving them."""
    if col != "event_type":
        return ""
    cats = [r["category"] for r in rows]
    if xc.CLEARANCE_EVENT_TYPE not in cats:
        return ""
    row = rows[cats.index(xc.CLEARANCE_EVENT_TYPE)]
    rank = cats.index(xc.CLEARANCE_EVENT_TYPE) + 1
    worst = rows[-1]
    reversed_ok = rank == 1 and row["mean_xt"] > 0
    headline = (
        "Clearance artefact -- REVERSED under the corrected <code>xt_after</code>, exactly as Prompt 66 predicted."
        if reversed_ok else
        "Clearance artefact -- measured on this run; the reversal Prompt 66 reported is NOT reproduced here."
    )
    return f"""
<div class="xt-clearance-note">
<b>{headline}</b>
<p style="margin:8px 0;">On this run <code>Clearance</code> ranks <b>{rank} of {len(rows)}</b>
<code>event_type</code> categories, mean <b>{row['mean_xt']:+.6f}</b> over n={row['n']:,} rows
({row['pct_negative']:.1f}% negative, {row['pct_positive']:.1f}% positive). The v1 portal
(<a href="../{v2.V1_PORTAL_DIRNAME}/active_category_atlas.html">preserved here</a>) measured the same category at
<b>8 of 8, -0.032482</b> -- the single worst-looking event type -- and this atlas's v1 edition carried a warning
not to read that as "clearances are bad defending".</p>
<p style="margin:8px 0;">{esc(v2.PROMPT_66_CLEARANCE_FINDING)}</p>
<p style="margin:8px 0 0;">The lowest-mean event type on this run is <code>{esc(worst['category'])}</code> at
<b>{worst['mean_xt']:+.6f}</b> (n={worst['n']:,}) -- Prompt 66 recorded the same handover and explicitly left it
unchased as outside its scope. It is surfaced here, not investigated: this portal takes no position on whether
<code>{esc(worst['category'])}</code>'s negative mean is itself an artefact.</p>
</div>"""


def _possession_flag_note_v2(pool_columns: set[str], off_pool_lifts: dict[str, dict]) -> str:
    """REWRITTEN for v2. Same pool-construction gap as v1 (these three
    columns predate the reconstructed 51-era pool, so the ledger structurally
    cannot show them), but a completely different finding behind it: under
    v2 two of the three ARE the trigger for `xt_after = 0`, so their True
    groups equal `xt_before` by construction."""
    ranked = sorted(off_pool_lifts.values(), key=lambda r: abs(r["lift"]), reverse=True)
    rows = "".join(
        f"<li><code>{esc(r['column'])}</code>: True-group mean <b>{r['mean_xt_true']:+.6f}</b> over "
        f"n={r['n_true']:,} ({r['true_direction']['pct_zero']:.1f}% exactly zero, "
        f"{r['true_direction']['pct_negative']:.1f}% negative, "
        f"{r['true_direction']['pct_positive']:.1f}% positive) &mdash; lift <b>{r['lift']:+.6f}</b>"
        + (f" &middot; <b>identical to this slice's own <code>xt_before</code> mean</b> "
           f"({r['xt_before_mean_same_slice']:+.6f}, np.allclose confirmed on this run)"
           if r.get("collapses_to_xt_before") else "")
        + "</li>"
        for r in ranked
    )
    return f"""
<div class="xt-clearance-note">
<b>A finding this ledger's own methodology does NOT surface -- stated rather than omitted.</b>
<p style="margin:8px 0;">None of the three <code>action_*_possession</code> flags appears as a row below, for the
same structural reason as in the v1 edition of this report: the ledger is built from the reconstructed pre-drop
51-era pool (<code>feature_config_v1_historical.ACTIVE_V1_HISTORICAL</code>), and all three were already out of
the candidate list before that snapshot was taken. The pool construction is mirrored unchanged from
<code>generate_reports_xg.py</code>, so it structurally cannot show them. The same
<code>boolean_xt_lift</code> function used for every row below was therefore run on them OFF-POOL:</p>
<ul style="margin:8px 0 0 18px;">{rows}</ul>
<p style="margin:8px 0;"><b>What changed from v1, and it is not a measurement.</b> Under the corrected target,
<code>action_ended_possession</code> is the flag that TRIGGERS <code>xt_after = 0</code>. Its True group's delta
is therefore <code>xt_before</code> exactly, by construction -- not a discovered effect, and checked on this run
with <code>np.allclose</code> rather than quoted. The v1 edition of this note reported these flags as "roughly
77-79% non-zero with real spread" against v1's target; that figure described v1's target and does not carry
over.</p>
<p style="margin:8px 0;">{esc(v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING)}</p>
<p style="margin:8px 0 0;">Still reported as an observation about the target, <b>not</b> a proposal to unlock
these columns -- and on v2 the case for leaving them excluded is <i>stronger</i>, not weaker: a column that
deterministically sets one term of the target to zero is a construction input, and feeding it to a model would
be circular. They stay excluded in <code>feature_config.py</code>; this portal changes nothing there. See also
this portal's Leakage Audit, Part C&prime;.</p>
</div>
<div class="xt-adapt">
<b>The Clearance reversal does NOT appear in this report, and cannot &mdash; stated rather than left silent.</b>
<p style="margin:8px 0;">Prompt 66's headline result is that Clearance flips from 8th of 8 event types
(-0.0325) to 1st of 8 (+0.0247) under the corrected <code>xt_after</code>. A reader arriving from the portal
index might reasonably look for it here. It is not below, and that is structural rather than a gap in the
finding: <code>Clearance</code> is a value of the categorical column <code>event_type</code>, and this ledger
covers <b>boolean columns only</b>. <code>event_type</code> is not, and could not be, one of its rows.</p>
<p style="margin:8px 0 0;">The reversal is reported where this suite's methodology does surface it &mdash; the
<a href="active_category_atlas.html">Category Atlas</a>'s <code>event_type</code> card, with Prompt 66's own
numbers cited inline &mdash; and it is tested twice in the
<a href="CONFOUND_ANALYSIS.html">Confound report</a>, including against the possession-ending construction rule
described just above, which is the mechanism that could have made it self-fulfilling.</p>
</div>"""


def render_category_atlas_xt_v2(scale, base_xt, base_xt_nz, n_rows, pool, tables, tables_nz) -> str:
    """Copy of the v1 renderer with the scale finding-card's shape wording
    corrected (v1 said "roughly symmetric"; v2's skew is -0.306) and the
    carried-forward card re-pointed at Prompt 66."""
    cards = [
        r1._category_card(p["column"], p["status"], p["reason"], tables[p["column"]], base_xt,
                          tables_nz[p["column"]], base_xt_nz)
        for p in pool
    ]
    n_locked = sum(1 for p in pool if p["status"] == "locked")
    n_dropped = sum(1 for p in pool if p["status"] == "dropped")

    findings = [
        finding_card(
            "pool",
            f"{len(pool)} categorical columns ({n_locked} locked + {n_dropped} dropped-but-included)",
            "reconstructed pre-drop 51-era ACTIVE pool, same discipline and same source lists as the xg Category "
            "Atlas and as this report's v1 edition -- a dropped-but-strong column stands out rather than "
            "disappearing.",
        ),
        finding_card(
            "scale",
            f"mean {esc(TARGET)} = {base_xt:+.6f}",
            f"left-skewed (skew {scale['skew']}) and heavy-tailed (excess kurtosis "
            f"{scale['excess_kurtosis']}), {scale['pct_negative']:.1f}% negative / {scale['pct_positive']:.1f}% "
            f"positive / {scale['pct_zero']:.1f}% exactly zero, std {scale['std']:.5f}. Not a rate, not a "
            "percentage, NOT non-negative and -- unlike v1 -- no longer close to symmetric.",
        ),
        finding_card(
            "conditional",
            f"mean given a non-zero delta = {base_xt_nz:+.6f}",
            "each card expands to a panel restricted to rows whose delta is not exactly zero -- this target's "
            "structural-zero analogue of the xg suite's shot-conditional panel.",
        ),
        finding_card(
            "carried forward from Prompt 66 -- REVERSED",
            "Clearance is now the BEST event type, not the worst",
            "the v1 edition of this atlas ranked Clearance 8 of 8 and flagged it as an <code>action_x</code> "
            "measurement artefact. Under the corrected <code>xt_after</code> it ranks 1 of 8. Cross-referenced "
            "inline on the <code>event_type</code> card.",
            flag=True,
        ),
    ]

    body = r1._adaptation_banner(scale) + findings_grid(findings) \
        + "\n<h2 class=\"section-title\">Category Atlas -- xT delta v2</h2>" \
        "<p class=\"section-note\">One card per categorical column, reconstructed pre-drop ACTIVE pool. Green " \
        "border = locked, amber = dropped (still shown, reason inline). Bars diverge from zero: green right = " \
        "threat reduced, red left = threat increased. Expand a card for the non-zero-delta panel.</p>" \
        + "".join(cards)

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "active rows"),
            (f"{base_xt:+.5f}", "mean xT delta v2"),
            (f"{scale['pct_negative']:.1f}%", "negative"),
            (f"{scale['pct_positive']:.1f}%", "positive"),
            (str(len(pool)), "categorical columns"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Active Defensive Actions: Category Atlas (xT delta v2)</title>
{FONT_LINKS}
<style>{CSS}{r1.XT_ATLAS_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">CATEGORY ATLAS -- XT DELTA V2 &middot; ACTIVE-BINARY LEG ONLY</p>
  <h1 class="title">Active Defensive Actions: Category Atlas (xT delta v2)</h1>
  <p class="dek">Mean {esc(TARGET)} by category, reconstructed pre-drop ACTIVE pool, one row per actual defensive
  action. Supersedes the v1 edition preserved at
  <code>reports/analysis/{v2.V1_PORTAL_DIRNAME}/</code>. Active-binary leg only.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def _augment_possession_lifts(df, off_pool_lifts: dict[str, dict]) -> dict[str, dict]:
    """Check Prompt 66's construction identity on this run rather than quoting it."""
    for col, r in off_pool_lifts.items():
        sub = df.loc[(df[col] == True) & df[TARGET].notna()]  # noqa: E712
        if len(sub) == 0:
            continue
        r["xt_before_mean_same_slice"] = round(float(sub["xt_before"].mean()), 8)
        r["collapses_to_xt_before"] = bool(
            np.allclose(sub[TARGET].to_numpy(), sub["xt_before"].to_numpy(), equal_nan=True)
        )
        r["collapse_note"] = (
            "target == xt_before exactly for every row in this slice (np.allclose, checked on this run): "
            "xt_after is forced to 0.0 by construction for possession-ending actions. Construction identity, "
            "not a discovered effect."
        ) if r["collapses_to_xt_before"] else (
            "does NOT collapse onto xt_before -- this flag is not the xt_after=0 trigger."
        )
    return off_pool_lifts


def main() -> None:
    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    n_rows = len(df)
    base_xt = xc.base_xt(df)
    nz = xc.nonzero_subset(df)
    base_xt_nz = xc.base_xt(nz)

    cat_pool = r1._column_pool("categorical")
    bool_pool = r1._column_pool("boolean")

    cat_tables = {p["column"]: xc.categorical_xt_table(df, p["column"]) for p in cat_pool}
    cat_tables_nz = {p["column"]: xc.categorical_xt_table_nonzero(df, p["column"]) for p in cat_pool}
    bool_lifts = [
        {**p, "lift_stats": xc.boolean_xt_lift(df, p["column"]),
         "lift_stats_nonzero": xc.boolean_xt_lift_nonzero(df, p["column"])}
        for p in bool_pool
    ]
    off_pool_lifts = _augment_possession_lifts(df, {
        c: xc.boolean_xt_lift(df, c)
        for c in r1.POSSESSION_FLAGS
        if c in df.columns and c not in {p["column"] for p in bool_pool}
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": TARGET,
        "target_source": scale["target_source"],
        "supersedes": scale["supersedes"],
        "n_rows": n_rows,
        "n_rows_nonzero": int(len(nz)),
        "base_xt": base_xt,
        "base_xt_nonzero": base_xt_nz,
        "target_scale": scale,
        "prompt_66_cross_references": {
            "clearance_reversal": v2.PROMPT_66_CLEARANCE_FINDING,
            "distribution_shape": v2.PROMPT_66_SHAPE_FINDING,
            "possession_ending_collapse": v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING,
        },
    }

    cat_json = {**common, "columns": {
        p["column"]: {
            "status": p["status"], "reason": p["reason"],
            "categories": cat_tables[p["column"]],
            "categories_given_nonzero_delta": cat_tables_nz[p["column"]],
        } for p in cat_pool
    }}
    flag_json = {**common, "columns": {
        e["column"]: {
            "status": e["status"], "reason": e["reason"],
            "lift_stats": e["lift_stats"],
            "lift_stats_given_nonzero_delta": e["lift_stats_nonzero"],
        } for e in bool_lifts
    }, "off_pool_possession_flags": {
        "note": (
            "These columns are NOT rows in `columns` above, and that is not an oversight -- the ledger is built "
            "from the reconstructed pre-drop 51-era pool and all three were already out of the candidate list "
            "before that snapshot. Same structural gap as the v1 edition of this report. What differs on v2: "
            "action_ended_possession is the flag that TRIGGERS xt_after = 0, so its True group's delta equals "
            "xt_before exactly by construction (verified with np.allclose on this run, see "
            "`collapses_to_xt_before`). That is Prompt 66 section 2.1's result, and it is a construction "
            "identity rather than a measurement. Observation only -- these columns remain excluded in "
            "feature_config.py, and on v2 the case for keeping them excluded is stronger, not weaker."
        ),
        "lifts": off_pool_lifts,
    }}

    (OUTPUT_DIR / "active_category_atlas.json").write_text(json.dumps(cat_json, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "active_flag_ledger.json").write_text(json.dumps(flag_json, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "active_category_atlas.html").write_text(
        render_category_atlas_xt_v2(scale, base_xt, base_xt_nz, n_rows, cat_pool, cat_tables, cat_tables_nz),
        encoding="utf-8")
    # The flag-ledger renderer needed no logic change at all -- only its two
    # masthead strings name the target version, and they are literals in the
    # v1 module. Retitled here rather than duplicating 80 lines of markup.
    ledger_html = (
        r1.render_flag_ledger_xt(scale, base_xt, base_xt_nz, n_rows, bool_lifts, off_pool_lifts)
        .replace("Flag Ledger (xT delta)", "Flag Ledger (xT delta v2)")
        .replace("FLAG LEDGER -- XT DELTA &middot;", "FLAG LEDGER -- XT DELTA V2 &middot;")
    )
    (OUTPUT_DIR / "active_flag_ledger.html").write_text(ledger_html, encoding="utf-8")

    et = cat_tables.get("event_type", [])
    cats = [r["category"] for r in et]
    clr = et[cats.index("Clearance")] if "Clearance" in cats else None
    print(f"active: {n_rows:,} rows, mean xT delta v2={base_xt:+.6f}, cat={len(cat_pool)}, bool={len(bool_pool)}")
    if clr:
        print(f"  Clearance (event_type): mean={clr['mean_xt']:+.6f} n={clr['n']:,} "
              f"-- rank {cats.index('Clearance') + 1}/{len(et)} (v1 portal: 8/8 at -0.032482)")
    print(f"  lowest-mean event type: {et[-1]['category']} at {et[-1]['mean_xt']:+.6f}")
    for c, r in off_pool_lifts.items():
        print(f"  off-pool {c}: collapses_to_xt_before={r.get('collapses_to_xt_before')}")
    print(f"Wrote 4 files to {OUTPUT_DIR}")


# Module-level overrides so the reused v1 renderers pick up the v2 prose.
# Python resolves module globals at CALL time, so rebinding the attribute on
# the v1 module reaches the nested calls inside `_category_card` and
# `render_flag_ledger_xt` without editing that file -- the same mechanism
# `xt_v2_common` uses for `xt_common`.
r1._possession_flag_note = _possession_flag_note_v2
r1._clearance_note_for = _clearance_note_for_v2

if __name__ == "__main__":
    main()
