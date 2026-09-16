"""CLI entrypoint: Category Atlas + Flag Ledger against the CONTINUOUS xG
target (target_future_xg_10s) -- the categorical/boolean counterpart to
generate_numerical_xg_target_analysis.py. Together the three atlases
(category, flag, numerical) replicate the full binary-target base-EDA stage
for the continuous target.

Feature lists start from the reconstructed pre-drop 51/44 pool (same
ACTIVE_V1_HISTORICAL/PASSIVE_V1_HISTORICAL lists the numerical xG atlas
uses), not the current locked candidate lists -- every categorical/boolean
column is tagged locked or dropped (+ reason), same discipline as the
numerical atlas, so a dropped-but-strong column stands out rather than
disappearing.

Distribution Atlas is NOT replicated here: compute_stats.continuous_distribution/
discrete_distribution never reference target_col at all (they describe a
feature's own shape) -- the existing reports/eda/*_distribution_atlas.html
already applies to both target contexts as-is, nothing to duplicate.

Usage:
    python -m src.eda.generate_reports_xg
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import compute_stats_xg as cs
from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.feature_config_v1_historical import ACTIVE_V1_HISTORICAL, PASSIVE_V1_HISTORICAL
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "reports" / "eda_xg"
TARGET_XG = "target_future_xg_10s"

XG_ATLAS_CSS = """
.xg-card.locked { border-left: 4px solid var(--good); }
.xg-card.dropped { border-left: 4px solid var(--amber); }
.xg-status-badge { font-family: "JetBrains Mono", monospace; font-size: 9.5px; text-transform: uppercase;
  padding: 2px 7px; border-radius: 999px; font-weight: 700; margin-left: 8px; vertical-align: middle; }
.xg-status-badge.locked { background: var(--good-wash); color: var(--good); }
.xg-status-badge.dropped { background: var(--amber-wash); color: var(--amber); }
.xg-reason { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); margin: 2px 0 10px; }
"""


def _column_pool(dataset_key: str, kind: str) -> list[dict]:
    """Reconstructed pre-drop pool for one column kind ('categorical' or
    'boolean'), tagged locked/dropped -- mirrors
    generate_numerical_target_analysis.build_pool's approach, generalized to
    non-numerical column kinds (which need no coordinate-duplicate exclusion)."""
    locked_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    hist_cfg = {"active": ACTIVE_V1_HISTORICAL, "passive": PASSIVE_V1_HISTORICAL}[dataset_key]

    locked_set = set(locked_cfg[kind])
    dropped_set = set(hist_cfg[kind]) - locked_set

    pool = [{"column": c, "status": "locked", "reason": None} for c in sorted(locked_set)]
    for c in sorted(dropped_set):
        reason = locked_cfg["excluded"].get(c, "Dropped (reason not found in feature_config.py's excluded dict)")
        pool.append({"column": c, "status": "dropped", "reason": reason})
    pool.sort(key=lambda p: p["column"])
    return pool


def _category_card(col: str, status: str, reason: str | None, rows: list[dict], base_xg: float) -> str:
    max_val = max([r["mean_xg"] for r in rows] + [base_xg]) * 1.15 or 1.0
    bar_rows = []
    for r in rows:
        fill_pct = max(0.0, min(100.0, (r["mean_xg"] / max_val) * 100))
        ref_pct = max(0.0, min(100.0, (base_xg / max_val) * 100))
        bar_rows.append(f"""
<div class="catbar-row" title="{esc(r['category'])}: n={r['n']}, xg_sum={r['xg_sum']:.3f}, mean_xg={r['mean_xg']:.5f}">
  <div class="catbar-label"><span>{esc(r['category'])}</span><span class="n">n={r['n']}</span></div>
  <div class="catbar-track">
    <div class="catbar-fill" style="width:{fill_pct:.2f}%"></div>
    <div class="catbar-refline" style="left:{ref_pct:.2f}%"></div>
  </div>
  <span class="rate-val">{r['mean_xg']:.5f}</span>
</div>""")
    reason_html = f'<p class="xg-reason"><b>Why dropped:</b> {esc(reason)}</p>' if status == "dropped" and reason else ""
    return f"""
<div class="card xg-card {status}" style="margin-bottom:16px;">
<h3>{esc(col)} <span class="xg-status-badge {status}">{status.upper()}</span></h3>
<p class="subnote">mean xG by category &middot; dashed marker = dataset mean xG ({base_xg:.5f})</p>
{reason_html}
{''.join(bar_rows)}
</div>"""


def render_category_atlas_xg(dataset_cfg: dict, base_xg: float, n_rows: int, pool: list[dict], tables: dict[str, list[dict]]) -> str:
    cards = [_category_card(p["column"], p["status"], p["reason"], tables[p["column"]], base_xg) for p in pool]

    n_locked = sum(1 for p in pool if p["status"] == "locked")
    n_dropped = sum(1 for p in pool if p["status"] == "dropped")

    findings = [
        finding_card(
            "pool",
            f"{len(pool)} categorical columns ({n_locked} locked + {n_dropped} dropped-but-included)",
            "reconstructed pre-drop 51/44-era pool, not the current locked list -- same discipline as the "
            "numerical xG atlas.",
        ),
        finding_card(
            "scale",
            f"mean {esc(TARGET_XG)} = {base_xg:.5f}",
            "heavily zero-inflated continuous target -- values here are mean xG per category, not a percentage.",
        ),
    ]

    body = findings_grid(findings) + "\n<h2 class=\"section-title\">Category Atlas -- xG</h2>" \
        "<p class=\"section-note\">One card per categorical column, reconstructed pre-drop pool. Green border = " \
        "locked (current candidate list), amber = dropped (still shown, reason inline). Bars = groupby-mean xG " \
        "against the full row population per category.</p>" \
        + "".join(cards)

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "rows"),
            (f"{base_xg:.5f}", "mean xG"),
            (str(len(pool)), "categorical columns"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(dataset_cfg['label'])}: Category Atlas (xG)</title>
{FONT_LINKS}
<style>{CSS}{XG_ATLAS_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">CATEGORY ATLAS -- XG &middot; {esc(dataset_cfg['label'].upper())}</p>
  <h1 class="title">{esc(dataset_cfg['label'])}: Category Atlas (xG)</h1>
  <p class="dek">Mean {esc(TARGET_XG)} by category, reconstructed pre-drop pool, {esc(dataset_cfg['row_description'])}.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def _flag_row(entry: dict, max_abs_lift: float) -> str:
    r = entry["lift_stats"]
    pos_pct = max(0.0, min(50.0, (max(r["lift"], 0) / max_abs_lift) * 50))
    neg_pct = max(0.0, min(50.0, (max(-r["lift"], 0) / max_abs_lift) * 50))
    badge = '<span class="small-n-badge">small n</span>' if r["small_n"] else ""
    status_badge = f'<span class="xg-status-badge {entry["status"]}">{entry["status"].upper()}</span>'
    reason_html = f' &middot; <span class="xg-reason" style="display:inline;">{esc(entry["reason"])}</span>' if entry["status"] == "dropped" and entry["reason"] else ""
    return f"""
<div class="ledger-row" title="True: {r['mean_xg_true']:.5f} (n={r['n_true']})  False: {r['mean_xg_false']:.5f} (n={r['n_false']})">
  <div class="ledger-label">{esc(r['column'])}{badge}{status_badge}<span class="pct">{r['pct_true']:.1f}% True{reason_html}</span></div>
  <div class="divaxis"><span class="zero"></span>
    <div class="divbar neg" style="width:{neg_pct:.2f}%"></div>
    <div class="divbar pos" style="width:{pos_pct:.2f}%"></div>
  </div>
  <div class="ledger-figs"><span class="lift">{r['lift']:+.5f}</span><br>T {r['mean_xg_true']:.5f} &middot; F {r['mean_xg_false']:.5f}</div>
</div>"""


def render_flag_ledger_xg(dataset_cfg: dict, base_xg: float, n_rows: int, pool_lifts: list[dict]) -> str:
    ranked = sorted(pool_lifts, key=lambda e: abs(e["lift_stats"]["lift"]), reverse=True)
    max_abs_lift = max([abs(e["lift_stats"]["lift"]) for e in ranked] + [1e-9])

    n_locked = sum(1 for e in pool_lifts if e["status"] == "locked")
    n_dropped = sum(1 for e in pool_lifts if e["status"] == "dropped")

    findings = []
    for e in ranked[:2]:
        r = e["lift_stats"]
        findings.append(
            finding_card(
                "lift",
                r["column"],
                f"is associated with a {r['lift']:+.5f} shift in mean xG (True {r['mean_xg_true']:.5f} vs False {r['mean_xg_false']:.5f}).",
            )
        )
    findings.append(
        finding_card(
            "pool",
            f"{len(pool_lifts)} boolean columns ({n_locked} locked + {n_dropped} dropped-but-included)",
            "reconstructed pre-drop 51/44-era pool, not the current locked list.",
        )
    )
    for e in ranked:
        if e["lift_stats"]["small_n"]:
            r = e["lift_stats"]
            findings.append(
                finding_card(
                    "small n",
                    r["column"],
                    f"has a True or False group under 500 rows (n_true={r['n_true']}, n_false={r['n_false']}) -- lift estimate is noisy.",
                    flag=True,
                )
            )

    rows_html = "".join(_flag_row(e, max_abs_lift) for e in ranked)
    body = findings_grid(findings) + "\n<h2 class=\"section-title\">Flag Ledger -- xG</h2>" \
        "<p class=\"section-note\">Diverging bars ranked by absolute lift in mean xG, centered on zero. " \
        "Green LOCKED / amber DROPPED badge per row, reconstructed pre-drop pool.</p>" \
        f'<div class="ledger">{rows_html}</div>'

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "rows"),
            (f"{base_xg:.5f}", "mean xG"),
            (str(len(pool_lifts)), "boolean columns"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(dataset_cfg['label'])}: Flag Ledger (xG)</title>
{FONT_LINKS}
<style>{CSS}{XG_ATLAS_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">FLAG LEDGER -- XG &middot; {esc(dataset_cfg['label'].upper())}</p>
  <h1 class="title">{esc(dataset_cfg['label'])}: Flag Ledger (xG)</h1>
  <p class="dek">Every boolean column ranked by mean-xG lift, reconstructed pre-drop pool, {esc(dataset_cfg['row_description'])}.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def generate_dataset_reports_xg(dataset_key: str) -> dict:
    dataset_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"])
    n_rows = len(df)
    base_xg = cs.base_xg_rate(df, TARGET_XG)

    cat_pool = _column_pool(dataset_key, "categorical")
    bool_pool = _column_pool(dataset_key, "boolean")

    cat_tables = {p["column"]: cs.categorical_xg_table(df, p["column"], TARGET_XG) for p in cat_pool}
    bool_lifts = [
        {**p, "lift_stats": cs.boolean_xg_lift(df, p["column"], TARGET_XG)}
        for p in bool_pool
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"{dataset_key}_category_atlas.html": render_category_atlas_xg(dataset_cfg, base_xg, n_rows, cat_pool, cat_tables),
        f"{dataset_key}_flag_ledger.html": render_flag_ledger_xg(dataset_cfg, base_xg, n_rows, bool_lifts),
    }
    for filename, html_content in outputs.items():
        (OUTPUT_DIR / filename).write_text(html_content, encoding="utf-8")

    return {
        "dataset": dataset_key,
        "n_rows": n_rows,
        "base_xg": base_xg,
        "n_categorical": len(cat_pool),
        "n_boolean": len(bool_pool),
        "files": list(outputs.keys()),
    }


def main() -> None:
    summaries = []
    for key in ("active", "passive"):
        print(f"Generating {key} xG reports...")
        summaries.append(generate_dataset_reports_xg(key))

    for s in summaries:
        print(f"  {s['dataset']}: {s['n_rows']:,} rows, mean xG={s['base_xg']:.5f}, "
              f"cat={s['n_categorical']}, bool={s['n_boolean']}")
    print(f"\nWrote {sum(len(s['files']) for s in summaries)} HTML files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
