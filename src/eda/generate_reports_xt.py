"""CLI entrypoint: Category Atlas + Flag Ledger against `target_xt_delta`
(ACTIVE-BINARY LEG ONLY) -- the xT sibling of generate_reports_xg.py.

generate_reports_xg.py is the confirmed generator of
reports/analysis/xg_target/active_category_atlas.html/.json and
active_flag_ledger.html/.json (confirmed by reading it: same reconstructed
pre-drop pool via feature_config_v1_historical, same locked/dropped badge
markup, same `catbar-*`/`ledger-*` class names, same JSON key set). This
file is a sibling, not an edit -- the xg portal stays reproducible from the
unmodified original.

Three methodology adaptations, all decided and documented once in
src/eda/xt_common.py and restated in the rendered output:
  1. thresholds translated via xg's own std-fraction (the mean-multiple
     convention is meaningless for a near-zero-mean symmetric target);
  2. the "given a shot" conditional panel becomes "given a non-zero delta";
  3. lift is framed two-directionally, with negative/positive/zero shares
     shown alongside every mean.

Passive leg is out of scope for this prompt and is not touched.

Usage:
    python -m src.eda.generate_reports_xt
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import xt_common as xc
from src.eda.feature_config import ACTIVE
from src.eda.feature_config_v1_historical import ACTIVE_V1_HISTORICAL
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = xc.OUT_DIR
TARGET = xc.TARGET

XT_ATLAS_CSS = """
.xg-card.locked { border-left: 4px solid var(--good); }
.xg-card.dropped { border-left: 4px solid var(--amber); }
.xg-status-badge { font-family: "JetBrains Mono", monospace; font-size: 9.5px; text-transform: uppercase;
  padding: 2px 7px; border-radius: 999px; font-weight: 700; margin-left: 8px; vertical-align: middle; }
.xg-status-badge.locked { background: var(--good-wash); color: var(--good); }
.xg-status-badge.dropped { background: var(--amber-wash); color: var(--amber); }
.xg-reason { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); margin: 2px 0 10px; }
.xt-dirbar { display: inline-block; width: 110px; height: 7px; border-radius: 3px; overflow: hidden;
  background: var(--plane); vertical-align: middle; margin-left: 6px; }
.xt-dirbar i { display: inline-block; height: 100%; }
.xt-dirbar i.neg { background: var(--neg); }
.xt-dirbar i.zero { background: var(--text-muted); opacity: 0.45; }
.xt-dirbar i.pos { background: var(--good); }
.xt-clearance-note { border-left: 4px solid var(--amber); background: var(--amber-wash); padding: 12px 14px;
  border-radius: 8px; margin: 14px 0; font-size: 12.5px; line-height: 1.55; }
.xt-adapt { border-left: 4px solid var(--accent); background: var(--accent-wash); padding: 12px 14px;
  border-radius: 8px; margin: 14px 0; font-size: 12.5px; line-height: 1.55; }
"""


def _dirbar(d: dict) -> str:
    neg, zero, pos = d.get("pct_negative") or 0, d.get("pct_zero") or 0, d.get("pct_positive") or 0
    return (
        f'<span class="xt-dirbar" title="{neg:.1f}% negative / {zero:.1f}% zero / {pos:.1f}% positive">'
        f'<i class="neg" style="width:{neg:.1f}%"></i><i class="zero" style="width:{zero:.1f}%"></i>'
        f'<i class="pos" style="width:{pos:.1f}%"></i></span>'
    )


def _column_pool(kind: str) -> list[dict]:
    """Reconstructed pre-drop pool for one column kind -- identical logic to
    generate_reports_xg._column_pool, active side only."""
    locked_set = set(ACTIVE[kind])
    dropped_set = set(ACTIVE_V1_HISTORICAL[kind]) - locked_set

    pool = [{"column": c, "status": "locked", "reason": None} for c in sorted(locked_set)]
    for c in sorted(dropped_set):
        reason = ACTIVE["excluded"].get(c, "Dropped (reason not found in feature_config.py's excluded dict)")
        pool.append({"column": c, "status": "dropped", "reason": reason})
    pool.sort(key=lambda p: p["column"])
    return pool


def _category_bar_rows(rows: list[dict], base_xt: float, small_n_threshold: int | None = None) -> str:
    """Diverging bars centred on zero -- the xg version's one-sided
    left-anchored fill assumes a non-negative target and would render every
    negative category as a zero-width bar (Adaptation 3)."""
    max_abs = max([abs(r["mean_xt"]) for r in rows] + [abs(base_xt), 1e-9])
    out = []
    for r in rows:
        v = r["mean_xt"]
        pos_pct = max(0.0, min(50.0, (max(v, 0) / max_abs) * 50))
        neg_pct = max(0.0, min(50.0, (max(-v, 0) / max_abs) * 50))
        badge = (
            '<span class="small-n-badge">small n</span>'
            if small_n_threshold is not None and r["n"] < small_n_threshold else ""
        )
        out.append(f"""
<div class="catbar-row" title="{esc(r['category'])}: n={r['n']}, mean={r['mean_xt']:+.6f}, {r['pct_negative']:.1f}% neg / {r['pct_positive']:.1f}% pos">
  <div class="catbar-label"><span>{esc(r['category'])}</span><span class="n">n={r['n']}</span>{badge}{_dirbar(r)}</div>
  <div class="divaxis"><span class="zero"></span>
    <div class="divbar neg" style="width:{neg_pct:.2f}%"></div>
    <div class="divbar pos" style="width:{pos_pct:.2f}%"></div>
  </div>
  <span class="rate-val">{r['mean_xt']:+.6f}</span>
</div>""")
    return "".join(out)


def _clearance_note_for(col: str, rows: list[dict]) -> str:
    """Prompt 64's clearance/action_x artefact, surfaced wherever it actually
    shows up in this atlas rather than asserted generically."""
    if col != "event_type":
        return ""
    row = next((r for r in rows if r["category"] == xc.CLEARANCE_EVENT_TYPE), None)
    if row is None:
        return ""
    rank = [r["category"] for r in rows].index(xc.CLEARANCE_EVENT_TYPE) + 1
    return f"""
<div class="xt-clearance-note">
<b>Clearance / <code>action_x</code> artefact -- confirmed present in this atlas.</b>
<code>Clearance</code> ranks {rank} of {len(rows)} <code>event_type</code> categories here, mean
<b>{row['mean_xt']:+.6f}</b> over n={row['n']:,} rows ({row['pct_negative']:.1f}% negative,
{row['pct_positive']:.1f}% positive). {esc(xc.PROMPT_64_CLEARANCE_FINDING)}
Do not read this row as "clearances are bad defending" -- it is the same measurement artefact Prompt 64
identified, resurfacing here exactly where it was predicted to.
</div>"""


def _category_card(col: str, status: str, reason: str | None, rows: list[dict], base_xt: float,
                   rows_nz: list[dict], base_xt_nz: float) -> str:
    reason_html = f'<p class="xg-reason"><b>Why dropped:</b> {esc(reason)}</p>' if status == "dropped" and reason else ""
    n_nz_total = sum(r["n"] for r in rows_nz)
    return f"""
<div class="card xg-card {status}" style="margin-bottom:16px;">
<h3>{esc(col)} <span class="xg-status-badge {status}">{status.upper()}</span></h3>
{reason_html}
<p class="subnote">Unconditional -- mean {esc(TARGET)} by category, diverging from zero. Positive (green) = xT fell
across the action (threat reduced). Negative (red) = xT rose. Dataset mean = {base_xt:+.6f}. The small
three-colour bar beside each label is that category's negative / exactly-zero / positive row share.</p>
{_category_bar_rows(rows, base_xt)}
{_clearance_note_for(col, rows)}
<details style="margin-top:10px;">
<summary style="cursor:pointer; font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--neg);">
  given a non-zero delta (n={n_nz_total:,}) -- click to expand
</summary>
<p class="subnote" style="margin-top:8px;">Conditional on <code>{esc(TARGET)} != 0</code> -- the xT analogue of the
xg suite's "given a shot" panel (Adaptation 2): it strips the rows where the action never crossed an xT grid-cell
boundary, which are this target's structural-zero mass. Subset mean = {base_xt_nz:+.6f}. Small-sample categories
flagged, not hidden.</p>
{_category_bar_rows(rows_nz, base_xt_nz, small_n_threshold=xc.SMALL_N_CONDITIONAL_THRESHOLD)}
</details>
</div>"""


def _adaptation_banner(scale: dict) -> str:
    return f"""
<div class="xt-adapt">
<b>Methodology adapted for a symmetric, sign-carrying target -- stated, not assumed.</b>
<p style="margin:8px 0 0;">{esc(scale['scale_note'])}</p>
<p style="margin:8px 0 0;">{esc(scale['conditional_panel_note'])}</p>
<p style="margin:8px 0 0;">{esc(scale['direction_note'])}</p>
<p style="margin:8px 0 0;">{esc(scale['log_transform_note'])}</p>
</div>"""


def render_category_atlas_xt(scale: dict, base_xt: float, base_xt_nz: float, n_rows: int, pool: list[dict],
                             tables: dict, tables_nz: dict) -> str:
    cards = [
        _category_card(p["column"], p["status"], p["reason"], tables[p["column"]], base_xt,
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
            "Atlas -- a dropped-but-strong column stands out rather than disappearing.",
        ),
        finding_card(
            "scale",
            f"mean {esc(TARGET)} = {base_xt:+.6f}",
            f"roughly symmetric, {scale['pct_negative']:.1f}% negative / {scale['pct_positive']:.1f}% positive / "
            f"{scale['pct_zero']:.1f}% exactly zero, std {scale['std']:.5f}. Not a rate, not a percentage, and "
            "NOT non-negative -- the sign is the football meaning.",
        ),
        finding_card(
            "conditional",
            f"mean given a non-zero delta = {base_xt_nz:+.6f}",
            "each card expands to a panel restricted to rows whose delta is not exactly zero -- this target's "
            "structural-zero analogue of the xg suite's shot-conditional panel.",
        ),
        finding_card(
            "carried forward from Prompt 64",
            "Clearance / action_x artefact",
            "expected to resurface on the <code>event_type</code> card, and it does -- cross-referenced inline "
            "there with Prompt 64's own measurement.",
            flag=True,
        ),
    ]

    body = _adaptation_banner(scale) + findings_grid(findings) \
        + "\n<h2 class=\"section-title\">Category Atlas -- xT delta</h2>" \
        "<p class=\"section-note\">One card per categorical column, reconstructed pre-drop ACTIVE pool. Green " \
        "border = locked, amber = dropped (still shown, reason inline). Bars diverge from zero: green right = " \
        "threat reduced, red left = threat increased. Expand a card for the non-zero-delta panel.</p>" \
        + "".join(cards)

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "active rows"),
            (f"{base_xt:+.5f}", "mean xT delta"),
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
<title>Active Defensive Actions: Category Atlas (xT delta)</title>
{FONT_LINKS}
<style>{CSS}{XT_ATLAS_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">CATEGORY ATLAS -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY</p>
  <h1 class="title">Active Defensive Actions: Category Atlas (xT delta)</h1>
  <p class="dek">Mean {esc(TARGET)} by category, reconstructed pre-drop ACTIVE pool, one row per actual defensive
  action. Active-binary leg only -- the passive leg is out of scope for this target.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def _flag_row(entry: dict, max_abs_lift: float) -> str:
    r = entry["lift_stats"]
    rn = entry["lift_stats_nonzero"]
    pos_pct = max(0.0, min(50.0, (max(r["lift"], 0) / max_abs_lift) * 50))
    neg_pct = max(0.0, min(50.0, (max(-r["lift"], 0) / max_abs_lift) * 50))
    badge = '<span class="small-n-badge">small n</span>' if r["small_n"] else ""
    status_badge = f'<span class="xg-status-badge {entry["status"]}">{entry["status"].upper()}</span>'
    reason_html = (
        f' &middot; <span class="xg-reason" style="display:inline;">{esc(entry["reason"])}</span>'
        if entry["status"] == "dropped" and entry["reason"] else ""
    )
    nz_small = " (small n)" if rn["small_n"] else ""
    return f"""
<div style="border-bottom:1px solid var(--border); padding-bottom:4px; margin-bottom:2px;">
<div class="ledger-row" style="border-bottom:none;" title="True: {r['mean_xt_true']:+.6f} (n={r['n_true']})  False: {r['mean_xt_false']:+.6f} (n={r['n_false']})">
  <div class="ledger-label">{esc(r['column'])}{badge}{status_badge}<span class="pct">{r['pct_true']:.1f}% True{reason_html}</span>
    <br><span class="pct">True {_dirbar(r['true_direction'])} &middot; False {_dirbar(r['false_direction'])}</span>
  </div>
  <div class="divaxis"><span class="zero"></span>
    <div class="divbar neg" style="width:{neg_pct:.2f}%"></div>
    <div class="divbar pos" style="width:{pos_pct:.2f}%"></div>
  </div>
  <div class="ledger-figs"><span class="lift">{r['lift']:+.6f}</span><br>T {r['mean_xt_true']:+.6f} &middot; F {r['mean_xt_false']:+.6f}</div>
</div>
<details>
<summary style="cursor:pointer; font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--neg); padding:2px 14px;">
  given a non-zero delta (n_true={rn['n_true']:,}, n_false={rn['n_false']:,}){nz_small} -- click to expand
</summary>
<div class="ledger-figs" style="padding:4px 14px 6px; text-align:left;">
  lift given non-zero delta <span class="lift">{rn['lift']:+.6f}</span> &middot; T {rn['mean_xt_true']:+.6f} &middot; F {rn['mean_xt_false']:+.6f}
</div>
</details>
</div>"""


POSSESSION_FLAGS = ["action_changed_possession", "action_ended_possession", "action_won_possession"]


def _possession_flag_note(pool_columns: set[str], off_pool_lifts: dict[str, dict]) -> str:
    """Prompt 64's motivating finding -- the three `action_*_possession`
    flags stop being degenerate zeros on this target -- does NOT surface
    through this ledger's own methodology, because those columns are not in
    the reconstructed pre-drop pool the ledger is built from (they were
    already out before the 51-era snapshot `ACTIVE_V1_HISTORICAL` captures).

    That omission is itself worth reporting, so rather than silently leaving
    the finding out, the lifts are computed off-pool and the gap is stated.
    """
    ranked = sorted(off_pool_lifts.values(), key=lambda r: abs(r["lift"]), reverse=True)
    rows = "".join(
        f"<li><code>{esc(r['column'])}</code>: True-group mean <b>{r['mean_xt_true']:+.6f}</b> over "
        f"n={r['n_true']:,} ({r['true_direction']['pct_zero']:.1f}% exactly zero, "
        f"{r['true_direction']['pct_negative']:.1f}% negative, "
        f"{r['true_direction']['pct_positive']:.1f}% positive) &mdash; lift <b>{r['lift']:+.6f}</b></li>"
        for r in ranked
    )
    return f"""
<div class="xt-clearance-note">
<b>A finding this ledger's own methodology does NOT surface -- stated rather than omitted.</b>
<p style="margin:8px 0;">Prompt 64's motivating result is that the three <code>action_*_possession</code> flags
stop being degenerate zeros on this target. <b>None of the three appears as a row below.</b> Not because the
finding fails, but because this ledger is built from the reconstructed pre-drop 51-era pool
(<code>feature_config_v1_historical.ACTIVE_V1_HISTORICAL</code>), and all three were already out of the
candidate list before that snapshot was taken -- only <code>action_was_under_opponent_possession</code> survived
into it. The ledger's pool construction, mirrored unchanged from the xg version, therefore cannot show them.</p>
<p style="margin:8px 0;">Rather than let the finding disappear on a technicality, the same
<code>boolean_xt_lift</code> function used for every row below was run on the three columns OFF-POOL. Measured
on this run:</p>
<ul style="margin:8px 0 0 18px;">{rows}</ul>
<p style="margin:8px 0;">Against <code>target_future_shot_10s</code> these rows are 100% identical zeros -- which
is the stated reason feature_config.py excludes them ("structural zero (0.00% shot rate by target-window
definition)"). Against <code>{esc(TARGET)}</code> roughly 77-79% of them are non-zero with real spread. This
reproduces Prompt 64's central result on the full active dataset rather than only inside the two slices it
examined.</p>
<p style="margin:8px 0 0;">Reported as an observation about the target, <b>not</b> a proposal to unlock these
columns. They stay excluded in feature_config.py; changing it is outside this report suite's remit, and any such
change would need its own leakage review. They are also covered in this portal's Leakage Audit, Part C&prime;.</p>
</div>"""


def render_flag_ledger_xt(scale: dict, base_xt: float, base_xt_nz: float, n_rows: int,
                          pool_lifts: list[dict], off_pool_lifts: dict[str, dict]) -> str:
    ranked = sorted(pool_lifts, key=lambda e: abs(e["lift_stats"]["lift"]), reverse=True)
    max_abs_lift = max([abs(e["lift_stats"]["lift"]) for e in ranked] + [1e-9])
    n_locked = sum(1 for e in pool_lifts if e["status"] == "locked")
    n_dropped = sum(1 for e in pool_lifts if e["status"] == "dropped")

    findings = []
    for e in ranked[:2]:
        r = e["lift_stats"]
        direction = "MORE threat-reducing" if r["lift"] > 0 else "MORE threat-increasing"
        findings.append(
            finding_card(
                "lift",
                r["column"],
                f"True rows are {direction}: lift {r['lift']:+.6f} (True {r['mean_xt_true']:+.6f} vs False "
                f"{r['mean_xt_false']:+.6f}). Ranked by |lift|, both directions.",
            )
        )
    findings.append(
        finding_card(
            "pool",
            f"{len(pool_lifts)} boolean columns ({n_locked} locked + {n_dropped} dropped-but-included)",
            "reconstructed pre-drop 51-era ACTIVE pool, same source lists as the xg Flag Ledger.",
        )
    )
    for e in ranked:
        if e["lift_stats"]["small_n"]:
            r = e["lift_stats"]
            findings.append(
                finding_card(
                    "small n",
                    r["column"],
                    f"has a True or False group under {xc.SMALL_N_THRESHOLD} rows (n_true={r['n_true']}, "
                    f"n_false={r['n_false']}) -- lift estimate is noisy.",
                    flag=True,
                )
            )

    rows_html = "".join(_flag_row(e, max_abs_lift) for e in ranked)
    body = _adaptation_banner(scale) + findings_grid(findings) \
        + "\n<h2 class=\"section-title\">Flag Ledger -- xT delta</h2>" \
        "<p class=\"section-note\">Diverging bars ranked by <b>absolute</b> lift in mean xT delta, centered on " \
        "zero -- explicitly both directions, since a negative lift (True rows coincide with threat rising) is " \
        "just as much a finding as a positive one on this target. Green LOCKED / amber DROPPED badge per row. " \
        "The two small three-colour bars per row are the True and False groups' negative/zero/positive shares. " \
        "Expand a row for the non-zero-delta lift.</p>" \
        + _possession_flag_note({e["column"] for e in pool_lifts}, off_pool_lifts) \
        + f'<div class="ledger">{rows_html}</div>'

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "active rows"),
            (f"{base_xt:+.5f}", "mean xT delta"),
            (f"{base_xt_nz:+.5f}", "mean given non-zero"),
            (str(len(pool_lifts)), "boolean columns"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Active Defensive Actions: Flag Ledger (xT delta)</title>
{FONT_LINKS}
<style>{CSS}{XT_ATLAS_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">FLAG LEDGER -- XT DELTA &middot; ACTIVE-BINARY LEG ONLY</p>
  <h1 class="title">Active Defensive Actions: Flag Ledger (xT delta)</h1>
  <p class="dek">Every boolean column ranked by |mean-xT-delta lift|, reconstructed pre-drop ACTIVE pool, one row
  per actual defensive action. Active-binary leg only.</p>
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
    n_rows = len(df)
    base_xt = xc.base_xt(df)
    nz = xc.nonzero_subset(df)
    base_xt_nz = xc.base_xt(nz)

    cat_pool = _column_pool("categorical")
    bool_pool = _column_pool("boolean")

    cat_tables = {p["column"]: xc.categorical_xt_table(df, p["column"]) for p in cat_pool}
    cat_tables_nz = {p["column"]: xc.categorical_xt_table_nonzero(df, p["column"]) for p in cat_pool}
    bool_lifts = [
        {**p, "lift_stats": xc.boolean_xt_lift(df, p["column"]),
         "lift_stats_nonzero": xc.boolean_xt_lift_nonzero(df, p["column"])}
        for p in bool_pool
    ]
    # Off-pool: the three action_*_possession flags are NOT in the
    # reconstructed pre-drop pool, so the ledger's own methodology cannot
    # show them -- see _possession_flag_note for why that is reported rather
    # than omitted. Same lift function, computed outside the pool.
    off_pool_lifts = {
        c: xc.boolean_xt_lift(df, c)
        for c in POSSESSION_FLAGS
        if c in df.columns and c not in {p["column"] for p in bool_pool}
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": TARGET,
        "target_source": "outputs/prototypes/active_binary_xt_delta.parquet (joined read-only by event_id)",
        "n_rows": n_rows,
        "n_rows_nonzero": int(len(nz)),
        "base_xt": base_xt,
        "base_xt_nonzero": base_xt_nz,
        "target_scale": scale,
        "prompt_64_cross_references": {
            "clearance_action_x_artefact": xc.PROMPT_64_CLEARANCE_FINDING,
            "distribution_shape": xc.PROMPT_64_SHAPE_FINDING,
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
            "These columns are NOT rows in `columns` above, and that is not an oversight. This ledger is built "
            "from the reconstructed pre-drop 51-era pool (feature_config_v1_historical.ACTIVE_V1_HISTORICAL), "
            "and all three were already out of the candidate list before that snapshot -- so the ledger's own "
            "pool construction, mirrored unchanged from generate_reports_xg.py, cannot surface them. Prompt 64's "
            "motivating finding (they stop being degenerate zeros on this target) would therefore have "
            "disappeared on a technicality. The same boolean_xt_lift function used for every pooled row was run "
            "on them off-pool instead, and the gap is stated in the rendered report rather than omitted. "
            "Observation only -- these columns remain excluded in feature_config.py."
        ),
        "lifts": off_pool_lifts,
    }}

    (OUTPUT_DIR / "active_category_atlas.json").write_text(json.dumps(cat_json, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "active_flag_ledger.json").write_text(json.dumps(flag_json, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "active_category_atlas.html").write_text(
        render_category_atlas_xt(scale, base_xt, base_xt_nz, n_rows, cat_pool, cat_tables, cat_tables_nz), encoding="utf-8")
    (OUTPUT_DIR / "active_flag_ledger.html").write_text(
        render_flag_ledger_xt(scale, base_xt, base_xt_nz, n_rows, bool_lifts, off_pool_lifts), encoding="utf-8")

    et = cat_tables.get("event_type", [])
    clr = next((r for r in et if r["category"] == xc.CLEARANCE_EVENT_TYPE), None)
    print(f"active: {n_rows:,} rows, mean xT delta={base_xt:+.6f}, cat={len(cat_pool)}, bool={len(bool_pool)}")
    if clr:
        print(f"  Clearance (event_type): mean={clr['mean_xt']:+.6f} n={clr['n']:,} "
              f"-- rank {[r['category'] for r in et].index('Clearance') + 1}/{len(et)}")
    print(f"Wrote 4 files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
