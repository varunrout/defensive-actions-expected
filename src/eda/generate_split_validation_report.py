"""CLI entrypoint: render SPLIT_VALIDATION.json as a self-contained HTML
report, in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.generate_split_validation_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SPLIT_VALIDATION.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SPLIT_VALIDATION.html"

SPLIT_LABELS = {
    "test": "TEST (held out)",
    "fold0": "Fold 0",
    "fold1": "Fold 1",
    "fold2": "Fold 2",
    "fold3": "Fold 3",
    "fold4": "Fold 4",
}

# Per-fold colours, siblings of the shared palette (--f0..--f4) so the grid
# doesn't compete with the --pos/--neg/--amber/--good vocabulary used
# elsewhere; TEST reuses --amber since every other table already marks TEST
# rows with the amber wash.
FOLD_COLOR_VARS = ["var(--f0)", "var(--f1)", "var(--f2)", "var(--f3)", "var(--f4)"]

SPLIT_CSS = """
:root { --f0: #3b6e8f; --f1: #4f8f6e; --f2: #a3843a; --f3: #7a5a9e; --f4: #a1516f; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --f0: #6fa3c4; --f1: #7dc19c; --f2: #d1ab5f; --f3: #a98cd1; --f4: #d0839e; }
}
:root[data-theme="dark"] { --f0: #6fa3c4; --f1: #7dc19c; --f2: #d1ab5f; --f3: #a98cd1; --f4: #d0839e; }

.match-grid-frame { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin: 8px 0 8px; }
.match-grid-caption { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-muted); text-align: center; padding-top: 12px; border-top: 1px dashed var(--gridline); margin-top: 10px; }
.match-legend { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 28px; }
.match-legend-chip { display: flex; align-items: center; gap: 7px; font-family: "JetBrains Mono", monospace; font-size: 11px; padding: 5px 10px 5px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); }
.match-legend-dot { width: 10px; height: 10px; border-radius: 3px; flex: none; }
"""


def _fold_color(label: str) -> str:
    if label == "test":
        return "var(--amber)"
    return FOLD_COLOR_VARS[int(label.replace("fold", "")) % len(FOLD_COLOR_VARS)]


def _match_grid_svg(match_assignment: dict[str, str]) -> str:
    """Every one of the 115 matches as one tile, coloured by split
    assignment -- generated straight from match_assignment.json, not
    illustrative. Static SVG (no client-side JS), matching every other
    report's server-rendered-only approach."""
    match_ids = sorted(match_assignment.keys(), key=int)
    cols, cell, gap, ox, oy = 15, 60, 8, 20, 20
    rows = -(-len(match_ids) // cols)

    tiles = []
    for idx, mid in enumerate(match_ids):
        col, row = idx % cols, idx // cols
        x, y = ox + col * cell, oy + row * cell
        color = _fold_color(match_assignment[mid])
        opacity = "1" if match_assignment[mid] == "test" else "0.85"
        tiles.append(
            f'<rect x="{x}" y="{y}" width="{cell - gap}" height="{cell - gap}" rx="4" '
            f'fill="{color}" opacity="{opacity}"></rect>'
            f'<text x="{x + (cell - gap) / 2}" y="{y + (cell - gap) / 2 + 4}" '
            f'text-anchor="middle" font-size="10" font-family="JetBrains Mono, monospace" '
            f'fill="#fff" opacity="0.9">{esc(mid[-3:])}</text>'
        )

    legend = [("test", "Test (held out, 23 matches)")] + [(f"fold{i}", f"Fold {i} (CV)") for i in range(5)]
    legend_html = "".join(
        f'<div class="match-legend-chip"><span class="match-legend-dot" '
        f'style="background:{_fold_color(key)}"></span>{esc(label)}</div>'
        for key, label in legend
    )

    return f"""
<div class="match-grid-frame">
  <svg viewBox="0 0 {ox * 2 + cols * cell} {oy * 2 + rows * cell + 20}" role="img"
       aria-label="Grid of all 115 matches, each one tile coloured by its split assignment.">
    {''.join(tiles)}
  </svg>
  <p class="match-grid-caption">115 matches, one tile each &mdash; coloured by split assignment, read straight from match_assignment.json.</p>
</div>
<div class="match-legend">{legend_html}</div>"""


def _leg_table(leg_key: str, leg: dict) -> str:
    max_rate = max((r["shot_rate_pct"] or 0) for r in leg["by_split"]) * 1.15 or 1.0
    rows = ""
    for r in leg["by_split"]:
        is_test = r["split"] == "test"
        pct = min(100.0, (r["shot_rate_pct"] or 0) / max_rate * 100)
        rows += f"""
<tr{' style="background:var(--amber-wash);"' if is_test else ''}>
  <td>{esc(SPLIT_LABELS.get(r['split'], r['split']))}</td>
  <td>{r['n_matches']}</td>
  <td>{r['n_rows']:,}</td>
  <td>
    <div class="vif-track" style="display:inline-block; width:140px; vertical-align:middle; margin-right:8px;">
      <div class="vif-fill {'severe' if is_test else ''}" style="width:{pct:.1f}%; background:{'var(--amber)' if is_test else 'var(--neg)'};"></div>
    </div>
    {r['shot_rate_pct']}%
  </td>
</tr>"""

    return f"""
<div class="vif-dataset-block">
  <h2 class="dataset-title">{esc(leg_key.title())}</h2>
  <p class="dataset-substat">{esc(leg['parquet_path'])} &middot; {leg['n_rows_total']:,} rows &middot; {leg['n_matches_total']} matches &middot; overall shot rate {leg['overall_shot_rate_pct']}%</p>
  <div class="table-scroll"><table class="evidence-ledger">
  <tr><th>Split</th><th>Matches</th><th>Rows</th><th>Shot rate</th></tr>
  {rows}
  </table></div>
</div>"""


def _integrity_ledger(checks: list[dict]) -> str:
    rows = ""
    for c in checks:
        cls = "distinct" if c["passed"] else "drop"
        status = "PASS" if c["passed"] else "FAIL"
        rows += f"""
<div class="corr-row">
  <span class="corr-pair">{esc(c['name'].replace('_', ' '))}</span>
  <span class="corr-method">{esc(c['description'])}</span>
  <span></span>
  <span class="corr-value" style="font-size:11px;">{esc(c['detail'])}</span>
  <span class="verdict-chip {cls}">{status}</span>
</div>"""
    return f'<div class="corr-ledger">{rows}</div>'


def build_report(data: dict) -> str:
    all_passed = data["all_integrity_checks_passed"]

    body = f"""
<div class="finding" style="margin-bottom:24px;">
<span class="tag">method</span>
<p>{esc(data['method_note'])}</p>
</div>

<div class="verdict-banner {'v-no' if all_passed else 'v-yes'}" style="margin-bottom:32px;">
<b>{'All integrity checks passed' if all_passed else 'INTEGRITY CHECKS FAILED'}</b> &mdash;
{data['n_matches_total']} matches total, {data['n_test_matches']} TEST, {data['n_train_val_matches']} TRAIN+VAL
across {data['n_folds']} folds.
</div>

<h2>Every match, one tile</h2>
<p class="section-note">Same split, drawn instead of tabulated -- read straight from the frozen match_assignment.json.</p>
{_match_grid_svg(data['match_assignment'])}

<h2>Integrity checks</h2>
<p class="section-note">Mirrors tests/test_canonical_split.py -- these run as real regression tests, not one-off prints; this report surfaces the same checks for readability.</p>
{_integrity_ledger(data['integrity_checks'])}

<h2 style="margin-top:40px;">Per-leg, per-split composition</h2>
<p class="section-note">TEST (amber) is held out entirely from cross-validation -- never used for model selection, only for a final, single evaluation once. Both legs share byte-identical match membership per split.</p>
{_leg_table('active', data['legs']['active'])}
{_leg_table('passive', data['legs']['passive'])}

<div class="closing-note" style="margin-top:32px;">
This split is frozen at <code>{esc(data['canonical_split_path'])}</code> and loaded via
<code>dax.models.splits.load_canonical_split()</code> -- never regenerated per run. Regenerating it would
silently break comparability between the active and passive legs and between past and future training runs.
</div>"""

    return render.render_article(
        eyebrow="SPLIT VALIDATION",
        title="Canonical Match-Grouped Split",
        dek=(
            "The frozen, shared TEST/CV split both the active and passive model legs read from -- "
            "match-grouped because target_future_shot_10s is heavily clustered within possessions "
            "(ICC 0.27-0.49) but barely differs between matches (ICC 0.006-0.008)."
        ),
        stats=[
            (str(data["n_matches_total"]), "matches total"),
            (str(data["n_test_matches"]), "TEST matches"),
            (str(data["n_train_val_matches"]), "TRAIN+VAL matches"),
            (f"{sum(1 for c in data['integrity_checks'] if c['passed'])}/{len(data['integrity_checks'])}", "integrity checks passed"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + render.VIF_CSS + SPLIT_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
