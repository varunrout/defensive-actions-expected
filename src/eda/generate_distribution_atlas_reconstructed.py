"""CLI entrypoint: Distribution Atlas over the RECONSTRUCTED pre-drop
51/44-era numerical pool (31 active / 34 passive), not the current locked
19/28 -- the existing reports/eda/*_distribution_atlas.html only ever
covered feature_config.py's locked continuous+discrete lists, so the 12
active / 6 passive dropped numerical columns never got a distribution card
anywhere. Target-independent (continuous_distribution/discrete_distribution
never reference a target column), so this single atlas is valid for both
the binary and xG contexts -- written to reports/eda_xg/ since that's where
the gap was noticed, using the same locked/dropped tagging as every other
atlas in this pool-reconstruction effort.

Usage:
    python -m src.eda.generate_distribution_atlas_reconstructed
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda import compute_stats as cs
from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import build_pool
from src.eda.render import FONT_LINKS, CSS, esc, finding_card, findings_grid
from src.eda.render import _continuous_card, _discrete_card

REPO_ROOT = Path(__file__).resolve().parents[2]
# Target-independent content -- written to both portals so the gap (dropped
# numerical columns having no distribution card anywhere) is closed in both
# places, not just the one where it was first noticed.
OUTPUT_DIRS = [REPO_ROOT / "reports" / "eda_xg", REPO_ROOT / "reports" / "eda"]

DIST_CSS = """
.dist-card-wrap { position: relative; border-radius: 12px; }
.dist-card-wrap.locked { border-left: 4px solid var(--good); }
.dist-card-wrap.dropped { border-left: 4px solid var(--amber); }
.dist-status-badge { font-family: "JetBrains Mono", monospace; font-size: 9.5px; text-transform: uppercase;
  padding: 2px 7px; border-radius: 999px; font-weight: 700; margin-left: 6px; vertical-align: middle; }
.dist-status-badge.locked { background: var(--good-wash); color: var(--good); }
.dist-status-badge.dropped { background: var(--amber-wash); color: var(--amber); }
.dist-reason { font-family: "JetBrains Mono", monospace; font-size: 10px; color: var(--text-muted); margin: 2px 0 8px; }
"""


def _wrap_card(card_html: str, status: str, reason: str | None) -> str:
    badge = f'<span class="dist-status-badge {status}">{status.upper()}</span>'
    # Inject the badge right after the <h3>column-name</h3> opening, and a
    # reason line right after the subnote, without needing to rebuild the
    # whole card markup -- _continuous_card/_discrete_card already close
    # </h3> right after the column name.
    card_html = card_html.replace("</h3>", f"{badge}</h3>", 1)
    if status == "dropped" and reason:
        card_html = card_html.replace(
            '<p class="subnote">', f'<p class="dist-reason">Why dropped: {esc(reason)}</p><p class="subnote">', 1
        )
    return f'<div class="dist-card-wrap {status}">{card_html}</div>'


def render_distribution_atlas_reconstructed(dataset_cfg: dict, n_rows: int, pool: list[dict], continuous: dict, discrete: dict) -> str:
    n_locked = sum(1 for p in pool if p["status"] == "locked")
    n_dropped = sum(1 for p in pool if p["status"] == "dropped")

    findings = [
        finding_card(
            "pool",
            f"{len(pool)} numerical columns ({n_locked} locked + {n_dropped} dropped-but-included)",
            "reconstructed pre-drop 51/44-era pool -- the existing Distribution Atlas only ever covered the "
            "current locked list, so the dropped columns never had a shape card anywhere until now.",
        ),
    ]

    cont_cards = []
    disc_cards = []
    for p in pool:
        col = p["feature"]
        if col in continuous:
            cont_cards.append(_wrap_card(_continuous_card(col, continuous[col]), p["status"], p["reason"]))
        elif col in discrete:
            disc_cards.append(_wrap_card(_discrete_card(col, discrete[col]), p["status"], p["reason"]))

    body = findings_grid(findings)
    if cont_cards:
        body += '\n<h2 class="section-title">Continuous Distributions</h2>' \
            '<p class="section-note">Green border = locked (current candidate list), amber = dropped ' \
            '(still shown, reason inline). 24 equal-width bins from the column min to its 99th percentile.</p>' \
            f'<div class="card-grid">{"".join(cont_cards)}</div>'
    if disc_cards:
        body += '\n<h2 class="section-title">Discrete Distributions</h2>' \
            '<p class="section-note">Raw value counts, capped at 20 distinct values.</p>' \
            f'<div class="card-grid">{"".join(disc_cards)}</div>'

    stat_html = "".join(
        f'<div class="stat"><b>{esc(v)}</b><span>{esc(label)}</span></div>'
        for v, label in [
            (f"{n_rows:,}", "rows"),
            (str(len(pool)), "numerical columns"),
            (str(n_locked), "locked"),
            (str(n_dropped), "dropped, still shown"),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(dataset_cfg['label'])}: Distribution Atlas (reconstructed pool)</title>
{FONT_LINKS}
<style>{CSS}{DIST_CSS}</style>
</head>
<body>
<div class="masthead"><div class="masthead-inner">
  <p class="eyebrow">DISTRIBUTION ATLAS -- RECONSTRUCTED POOL &middot; {esc(dataset_cfg['label'].upper())}</p>
  <h1 class="title">{esc(dataset_cfg['label'])}: Distribution Atlas</h1>
  <p class="dek">Every numerical column's own shape (mean/median/skew), reconstructed pre-drop 51/44-era pool,
  {esc(dataset_cfg['row_description'])}. Target-independent -- valid for both the binary and xG contexts.</p>
  <div class="statbar">{stat_html}</div>
</div></div>
<div class="wrap">
{body}
</div>
</body>
</html>"""


def generate(dataset_key: str) -> dict:
    dataset_cfg = {"active": ACTIVE, "passive": PASSIVE}[dataset_key]
    pool, _pool_meta = build_pool(dataset_key)
    df = pd.read_parquet(REPO_ROOT / dataset_cfg["parquet_path"], columns=[p["feature"] for p in pool])
    n_rows = len(df)

    continuous = {p["feature"]: cs.continuous_distribution(df, p["feature"]) for p in pool if p["type"] == "continuous"}
    discrete = {p["feature"]: cs.discrete_distribution(df, p["feature"]) for p in pool if p["type"] == "discrete"}
    html = render_distribution_atlas_reconstructed(dataset_cfg, n_rows, pool, continuous, discrete)

    paths = []
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        # reports/eda/ already has a locked-only *_distribution_atlas.html
        # (generate_reports.py) -- never overwrite that; this reconstructed
        # version gets a distinct name there. reports/eda_xg/ has no such
        # file yet, so the plain name is fine.
        filename = (
            f"{dataset_key}_distribution_atlas.html" if out_dir.name == "eda_xg"
            else f"{dataset_key}_distribution_atlas_reconstructed.html"
        )
        out_path = out_dir / filename
        out_path.write_text(html, encoding="utf-8")
        paths.append(str(out_path))

    return {"dataset": dataset_key, "n_rows": n_rows, "n_pool": len(pool), "paths": paths}


def main() -> None:
    for dataset_key in ("active", "passive"):
        result = generate(dataset_key)
        print(f"{result['dataset']}: {result['n_rows']:,} rows, {result['n_pool']} numerical columns")
        for p in result["paths"]:
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
