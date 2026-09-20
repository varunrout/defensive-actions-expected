"""CLI entrypoint: Category Atlas + Flag Ledger against
`target_xt_delta_passive` -- PASSIVE LEG.

STEP-0 DISPOSITION: genuine passive build (this report type is inherently
per-leg). Confirmed by reading `reports/analysis/xg_target/INDEX.html` and
its files: the xg portal ships literal `active_category_atlas.html` /
`passive_category_atlas.html` and `active_flag_ledger.html` /
`passive_flag_ledger.html` pairs, one per leg, because a category atlas is
a walk over ONE dataset's own categorical columns and the two datasets
share almost none of them.

What is reused, by direct import from `generate_reports_xt.py` (Prompt 65's
active-leg generator, NOT edited): the diverging-bar geometry, the
direction bar, the card and ledger markup, the adaptation banner, the
CSS, and the whole JSON key set. What is passive-specific and written here:

  1. The POOL. `generate_reports_xt._column_pool` reads `ACTIVE` /
     `ACTIVE_V1_HISTORICAL` only. The passive pool is built the same way
     from `PASSIVE` / `PASSIVE_V1_HISTORICAL`, exactly as
     `generate_reports_xg._column_pool` does for its own passive half.
  2. The CLEARANCE NOTE has no passive counterpart and is replaced rather
     than forced. `event_type` does not exist in `passive_defense.parquet`;
     its nearest relative is `on_ball_event_type`, which describes the
     ATTACKING action the snapshot is anchored to, not a defensive action.
     Prompt 66/67's Clearance reversal is an active-leg finding about
     defensive actions and simply is not a statement about this dataset.
  3. The POSSESSION-FLAG NOTE has no passive counterpart either: none of
     the three `action_*_possession` columns exists in the passive parquet
     (checked directly, not assumed). What replaces it is this leg's own
     structural note -- the shared-target row grain, and the passive-only
     screening columns the xg leakage audit flags.

Usage:
    python -m src.eda.generate_reports_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator import
from src.eda import xt_common as xc
from src.eda import generate_reports_xt as r1
from src.eda.feature_config import PASSIVE
from src.eda.feature_config_v1_historical import PASSIVE_V1_HISTORICAL
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid

OUTPUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET

# Checked against the live schema at runtime (see `main`), not assumed.
ACTIVE_ONLY_POSSESSION_FLAGS = r1.POSSESSION_FLAGS


def _passive_column_pool(kind: str) -> list[dict]:
    """Reconstructed pre-drop PASSIVE pool for one column kind.

    Identical logic to `generate_reports_xt._column_pool` and to
    `generate_reports_xg._column_pool`'s passive branch -- locked columns
    from `feature_config.PASSIVE`, plus everything the 44-era historical
    snapshot held that has since been dropped, tagged with its own recorded
    drop reason so a dropped-but-strong column stands out rather than
    disappearing.
    """
    locked_set = set(PASSIVE[kind])
    dropped_set = set(PASSIVE_V1_HISTORICAL[kind]) - locked_set

    pool = [{"column": c, "status": "locked", "reason": None} for c in sorted(locked_set)]
    for c in sorted(dropped_set):
        reason = PASSIVE["excluded"].get(c, "Dropped (reason not found in feature_config.py's excluded dict)")
        pool.append({"column": c, "status": "dropped", "reason": reason})
    pool.sort(key=lambda p: p["column"])
    return pool


def _event_type_note_passive(col: str, rows: list[dict]) -> str:
    """Replaces the active atlas's Clearance note. Two separate points, and
    neither is a restatement of the active finding."""
    if col != "on_ball_event_type":
        return ""
    if not rows:
        return ""
    best, worst = rows[0], rows[-1]
    return f"""
<div class="xt-clearance-note">
<b>The active portal's Clearance finding does NOT transfer to this column, and is not forced to.</b>
<p style="margin:8px 0;">Prompt 66/67's headline active-leg result is that <code>Clearance</code> flips from 8th
of 8 <code>event_type</code> categories to 1st of 8 under the corrected <code>xt_after</code>. A reader arriving
from the <a href="active_category_atlas.html">active Category Atlas</a> might reasonably look for the same row
here. <b>It is not below, and it could not be.</b> <code>event_type</code> does not exist in
<code>passive_defense.parquet</code> -- checked directly against the live schema, not assumed. This leg's
nearest relative, <code>on_ball_event_type</code>, describes the ATTACKING action the defensive snapshot is
anchored to (Pass, Carry, Ball Receipt*, Shot, ...), not a defensive action taken by the row's own defender.
A Clearance row in the active dataset and a Pass row here are not the same kind of thing, and comparing their
mean deltas would read like a finding while meaning nothing.</p>
<p style="margin:8px 0 0;">What this card does carry, on its own terms: <code>{esc(best['category'])}</code>
is the most threat-REDUCING on-ball event type at <b>{best['mean_xt']:+.6f}</b> (n={best['n']:,},
{best['pct_positive']:.1f}% positive), and <code>{esc(worst['category'])}</code> the most threat-INCREASING at
<b>{worst['mean_xt']:+.6f}</b> (n={worst['n']:,}, {worst['pct_negative']:.1f}% negative). Read both as
descriptions of what happens around a defensive snapshot, not as verdicts on defending: the row's target is a
property of the EVENT, shared identically across every defender slot at that event.</p>
</div>"""


def _structural_note_passive(pool_columns: set[str], off_pool_lifts: dict[str, dict]) -> str:
    """Replaces the active ledger's possession-flag note. The active note
    exists to surface three columns the pool cannot show; on this leg those
    columns do not exist at all, so a different, real structural point is
    made instead."""
    missing = ", ".join(f"<code>{esc(c)}</code>" for c in ACTIVE_ONLY_POSSESSION_FLAGS)
    return f"""
<div class="xt-clearance-note">
<b>The active ledger's off-pool possession-flag section has no counterpart here &mdash; stated, not silently
dropped.</b>
<p style="margin:8px 0;">The <a href="active_flag_ledger.html">active Flag Ledger</a> closes with an off-pool
section on {missing}, because those three columns trigger <code>xt_after = 0</code> in the active target's
construction and the reconstructed pre-drop pool structurally cannot show them. <b>None of the three exists in
<code>passive_defense.parquet</code></b> &mdash; verified against the live schema on this run, not assumed &mdash;
so there is nothing off-pool to compute and no construction-input flag on this leg's own candidate list.
<code>action_ended_possession</code> is still what sets <code>xt_after = 0</code> for this target, but it is read
from <code>events_with_targets.parquet</code> during Prompt 68's build and never lands in the passive feature
table, so no passive feature carries that coupling. This portal's
<a href="PASSIVE_LEAKAGE_AUDIT.html">passive Leakage Audit</a> confirms it directly rather than asserting it.</p>
<p style="margin:8px 0 0;"><b>The structural point that IS this leg's own.</b>
{esc(pc.PROMPT_68_SHARED_TARGET_FINDING)}</p>
</div>
<div class="xt-adapt">
<b>Reading a lift on this leg: what the number is a lift in.</b>
<p style="margin:8px 0 0;">Each row below compares the mean <code>{esc(TARGET)}</code> of the defender-slot rows
where the flag is True against those where it is False. Because the target is one value per EVENT repeated
across that event's defender slots, a lift here says "events at which at least this defender was positioned
this way tend to see the ball's threat move this way" &mdash; a statement about the situations a positioning
pattern occurs in, not about that defender's individual contribution. No per-defender attribution scheme exists
for this target and building one is explicitly out of scope, the same call Prompt 68 made. The two old passive
targets <code>target_future_shot_10s</code> and <code>target_future_xg_10s</code> have exactly this property
already, so every existing passive lift in this project reads the same way.</p>
</div>"""


def render_category_atlas_xt_passive(scale, base_xt, base_xt_nz, n_rows, pool, tables, tables_nz) -> str:
    """Copy of the active renderer's structure with passive-specific findings
    cards and masthead. The card bodies themselves come from the imported
    `_category_card`, so both legs' atlases render identically."""
    cards = [
        r1._category_card(p["column"], p["status"], p["reason"], tables[p["column"]], base_xt,
                          tables_nz[p["column"]], base_xt_nz)
        for p in pool
    ]
    n_locked = sum(1 for p in pool if p["status"] == "locked")
    n_dropped = sum(1 for p in pool if p["status"] == "dropped")
    rs = scale["robust_scale"]

    findings = [
        finding_card(
            "pool",
            f"{len(pool)} categorical columns ({n_locked} locked + {n_dropped} dropped-but-included)",
            "reconstructed pre-drop 44-era PASSIVE pool, the same source lists and the same discipline the "
            "passive xg Category Atlas uses -- not the active pool, which shares only <code>period</code> and "
            "<code>phase_label</code> with it.",
        ),
        finding_card(
            "scale",
            f"mean {esc(TARGET)} = {base_xt:+.6f}",
            f"strongly LEFT-skewed (skew {scale['skew']}) and very heavy-tailed (excess kurtosis "
            f"{scale['excess_kurtosis']}), {scale['pct_negative']:.1f}% negative / {scale['pct_positive']:.1f}% "
            f"positive / {scale['pct_zero']:.1f}% exactly zero, std {scale['std']:.5f}. Not a rate, not a "
            "percentage, NOT non-negative, and heavier-tailed than the active leg's own target.",
        ),
        finding_card(
            "conditional",
            f"mean given a non-zero delta = {base_xt_nz:+.6f}",
            f"each card expands to a panel restricted to the {scale['n_nonzero']:,} rows whose delta is not "
            "exactly zero -- this target's structural-zero analogue of the xg suite's shot-conditional panel.",
        ),
        finding_card(
            "row grain -- read every n with this in hand",
            "one value per event, repeated across that event's defender slots",
            "the target is computed once per <code>event_id</code> and left-joined onto every visible "
            "defender-slot row, exactly as <code>target_future_shot_10s</code> and "
            "<code>target_future_xg_10s</code> already are on this leg. Category n's are row counts, not "
            "independent observations.",
            flag=True,
        ),
        finding_card(
            "threshold caveat -- recomputed for this leg, not inherited",
            f"the std-derived flat margin spans {rs['pct_rows_inside_flat_margin']:.1f}% of rows",
            f"{rs['flat_margin_over_mad']:.2f}x the MAD on this leg, against 1.28x and 53.4% on the active "
            "leg's own v2 target and 0.53x / 35.4% on v1. Shape labels anywhere in this half of the portal are "
            "indicative, not decided.",
            flag=True,
        ),
    ]

    body = r1._adaptation_banner(scale) + findings_grid(findings) \
        + "\n<h2 class=\"section-title\">Category Atlas -- xT delta (passive)</h2>" \
        "<p class=\"section-note\">One card per categorical column, reconstructed pre-drop PASSIVE pool. Green " \
        "border = locked, amber = dropped (still shown, reason inline). Bars diverge from zero: green right = " \
        "threat reduced across the event, red left = threat increased. Expand a card for the non-zero-delta " \
        "panel.</p>" \
        + "".join(cards)

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "passive rows"),
            (f"{scale['n_unique_events']:,}", "unique events"),
            (f"{base_xt:+.5f}", "mean xT delta"),
            (f"{scale['pct_negative']:.1f}%", "negative"),
            (f"{scale['pct_positive']:.1f}%", "positive"),
            (str(len(pool)), "categorical columns"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Passive / Off-Ball Defensive Positioning: Category Atlas (xT delta)</title>
{FONT_LINKS}
<style>{CSS}{r1.XT_ATLAS_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">CATEGORY ATLAS -- XT DELTA &middot; PASSIVE LEG</p>
  <h1 class="title">Passive / Off-Ball Defensive Positioning: Category Atlas (xT delta)</h1>
  <p class="dek">Mean {esc(TARGET)} by category, reconstructed pre-drop PASSIVE pool, one row per visible
  defender-slot per on-ball attacking event. Target computed once per event and joined on; no v1 pass to
  supersede on this leg.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def main() -> None:
    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    scale["n_unique_events"] = int(df["event_id"].nunique())
    n_rows = len(df)
    base_xt = xc.base_xt(df)
    nz = xc.nonzero_subset(df)
    base_xt_nz = xc.base_xt(nz)

    # Confirm, rather than assume, that the active leg's construction-input
    # flags genuinely have no passive counterpart.
    absent_possession_flags = [c for c in ACTIVE_ONLY_POSSESSION_FLAGS if c not in df.columns]
    assert absent_possession_flags == ACTIVE_ONLY_POSSESSION_FLAGS, (
        "Expected none of the three action_*_possession flags to exist in passive_defense.parquet. "
        f"Present: {[c for c in ACTIVE_ONLY_POSSESSION_FLAGS if c in df.columns]}. The passive Flag Ledger's "
        "structural note asserts they are absent and must be rewritten if that changes."
    )

    cat_pool = _passive_column_pool("categorical")
    bool_pool = _passive_column_pool("boolean")

    cat_tables = {p["column"]: xc.categorical_xt_table(df, p["column"]) for p in cat_pool}
    cat_tables_nz = {p["column"]: xc.categorical_xt_table_nonzero(df, p["column"]) for p in cat_pool}
    bool_lifts = [
        {**p, "lift_stats": xc.boolean_xt_lift(df, p["column"]),
         "lift_stats_nonzero": xc.boolean_xt_lift_nonzero(df, p["column"])}
        for p in bool_pool
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common = {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg -- see active_category_atlas.json)",
        "target": TARGET,
        "target_source": scale["target_source"],
        "supersedes": scale["supersedes"],
        "n_rows": n_rows,
        "n_unique_events": scale["n_unique_events"],
        "n_rows_nonzero": int(len(nz)),
        "base_xt": base_xt,
        "base_xt_nonzero": base_xt_nz,
        "target_scale": scale,
        "active_only_possession_flags_absent_here": {
            "columns": ACTIVE_ONLY_POSSESSION_FLAGS,
            "verified_absent_on_this_run": True,
            "note": (
                "The active Flag Ledger carries an off-pool section on these three columns because they trigger "
                "xt_after = 0 in the active target's construction. None of them exists in "
                "passive_defense.parquet -- asserted against the live schema on this run rather than assumed -- "
                "so this leg has no construction-input flag on or off its candidate list. Reported as a real "
                "structural difference between the legs, not as a missing section."
            ),
        },
        "prompt_68_cross_references": {
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "possession_ending_collapse": pc.PROMPT_68_POSSESSION_COLLAPSE_FINDING,
            "correlation_with_old_targets": pc.PROMPT_68_CORRELATION_FINDING,
            "no_v1_detour": pc.PROMPT_68_NO_V1_DETOUR_FINDING,
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
    }}

    (OUTPUT_DIR / "passive_category_atlas.json").write_text(
        json.dumps(cat_json, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "passive_flag_ledger.json").write_text(
        json.dumps(flag_json, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "passive_category_atlas.html").write_text(
        render_category_atlas_xt_passive(scale, base_xt, base_xt_nz, n_rows, cat_pool, cat_tables, cat_tables_nz),
        encoding="utf-8")

    # The flag-ledger renderer needed no logic change -- only its two masthead
    # strings and its "active rows" stat label name the leg, and they are
    # literals in the active module. Retitled here by declared substitution
    # rather than duplicating 80 lines of markup, the same trade
    # generate_feature_lock_report_xt_v2.py makes. Each substitution is
    # asserted, so a change to the active renderer fails loudly.
    ledger_html = r1.render_flag_ledger_xt(scale, base_xt, base_xt_nz, n_rows, bool_lifts, {})
    for needle, replacement in [
        ("Active Defensive Actions: Flag Ledger (xT delta)",
         "Passive / Off-Ball Defensive Positioning: Flag Ledger (xT delta)"),
        ("FLAG LEDGER -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY",
         "FLAG LEDGER -- XT DELTA &middot; PASSIVE LEG"),
        ("reconstructed pre-drop ACTIVE pool, one row\n  per actual defensive action. Active-binary leg only.",
         "reconstructed pre-drop PASSIVE pool, one row\n  per visible defender-slot per on-ball attacking event."),
        ("reconstructed pre-drop 51-era ACTIVE pool, same source lists as the xg Flag Ledger.",
         "reconstructed pre-drop 44-era PASSIVE pool, same source lists as the passive xg Flag Ledger."),
        (">active rows<", ">passive rows<"),
    ]:
        assert needle in ledger_html, (
            f"Expected {needle!r} in generate_reports_xt.render_flag_ledger_xt's output so it could be "
            "relabelled for the passive leg. It is not there -- that module must have changed. Refusing to "
            "publish a passive report carrying active-leg labels."
        )
        ledger_html = ledger_html.replace(needle, replacement)
    (OUTPUT_DIR / "passive_flag_ledger.html").write_text(ledger_html, encoding="utf-8")

    et = cat_tables.get("on_ball_event_type", [])
    print(f"passive: {n_rows:,} rows over {scale['n_unique_events']:,} unique events, "
          f"mean xT delta={base_xt:+.6f}, cat={len(cat_pool)}, bool={len(bool_pool)}")
    if et:
        print(f"  on_ball_event_type best: {et[0]['category']} {et[0]['mean_xt']:+.6f} (n={et[0]['n']:,})")
        print(f"  on_ball_event_type worst: {et[-1]['category']} {et[-1]['mean_xt']:+.6f} (n={et[-1]['n']:,})")
    print(f"  flat_margin spans {scale['robust_scale']['pct_rows_inside_flat_margin']:.1f}% of rows "
          f"({scale['robust_scale']['flat_margin_over_mad']:.2f}x MAD)")
    print(f"Wrote 4 files to {OUTPUT_DIR}")


# Module-level overrides so the reused active renderers pick up passive prose.
# Python resolves module globals at CALL time, so rebinding the attribute on
# the active module reaches the nested calls inside `_category_card` and
# `render_flag_ledger_xt` without editing that file -- the same mechanism
# `xt_v2_common` uses for `xt_common`.
r1._clearance_note_for = _event_type_note_passive
r1._possession_flag_note = _structural_note_passive

if __name__ == "__main__":
    main()
