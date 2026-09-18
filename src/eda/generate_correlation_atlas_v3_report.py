"""CLI entrypoint: render CORRELATION_ANALYSIS_V3.json as a self-contained
HTML report, reusing V2's tier/method-strip rendering (correlation_v2's
tiering, including the categorical REVIEW band) since V3 runs the same
methodology, just on TRAIN+VAL rows only.

Compared against CORRELATION_ANALYSIS.json (not CORRELATION_ANALYSIS_V2.json)
-- that's the file generate_feature_lock_confirmation.py's stage-14 pass
scores on the same final 34/38 locked list V3 uses, just on the full
population (TEST included). V2 is no longer on that list -- it's stage 07's
historical 36/39 snapshot -- so it isn't a valid comparison partner here.

Usage:
    python -m src.eda.generate_correlation_atlas_v3_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.generate_correlation_reports_v2 import _dataset_block, _method_strip

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V3.json"
LOCKED_FULL_POPULATION_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ATLAS_V3.html"


def _tier_diff_banner(locked_data: dict, v3_data: dict) -> str:
    """Computed live from both JSONs -- does removing TEST-match rows change
    any tier verdict on the SAME final 34/38 list? A changed collapse/review
    count means yes; matching counts mean full-population scoring was never
    a problem for this feature set."""
    rows = []
    any_diff = False
    for ds_key in ("active", "passive"):
        locked_tc = locked_data["datasets"][ds_key]["tier_counts"]
        v3_tc = v3_data["datasets"][ds_key]["tier_counts"]
        diff_here = locked_tc["collapse"] != v3_tc["collapse"] or locked_tc["review"] != v3_tc["review"]
        any_diff = any_diff or diff_here
        rows.append(
            f'<tr><td>{ds_key.title()}</td>'
            f'<td>{locked_tc["collapse"]}</td><td>{v3_tc["collapse"]}</td>'
            f'<td>{locked_tc["review"]}</td><td>{v3_tc["review"]}</td>'
            f'<td>{locked_tc["distinct_total"]}</td><td>{v3_tc["distinct_total"]}</td></tr>'
        )
    verdict = (
        "At least one tier count shifted once TEST matches were excluded -- see the tables below for which "
        "pairs moved." if any_diff else
        "Every tier count matches the full-population run exactly. Including TEST matches never changed a "
        "redundancy verdict here -- expected, since dropping a duplicate/collinear column doesn't leak target "
        "information the way fitting a scaler or encoder would."
    )
    return f"""
<div class="finding {'flag' if any_diff else ''}" style="margin-bottom:24px;">
<span class="tag">V3 vs the full-population run on the same list -- does excluding TEST change anything?</span>
<p>{verdict} Both this page and CORRELATION_ANALYSIS.json score the identical final 34/38 list -- the only
difference is TEST-match rows in vs out.</p>
<div class="table-scroll"><table class="evidence-ledger" style="margin-top:10px;">
<tr><th>Dataset</th><th>Collapse (full pop.)</th><th>Collapse (V3)</th><th>Review (full pop.)</th><th>Review (V3)</th><th>Distinct (full pop.)</th><th>Distinct (V3)</th></tr>
{''.join(rows)}
</table></div>
</div>"""


def build_report(data: dict, locked_data: dict | None) -> str:
    datasets = data["datasets"]

    body = f"""
<div class="finding" style="margin-bottom:24px;">
<span class="tag">method</span>
<p>The final locked feature lists (34 active / 38 passive) -- neither
<a href="CORRELATION_ATLAS.html" style="color:var(--neg);">V1</a> (51/44, stage 01) nor
<a href="CORRELATION_ATLAS_V2.html" style="color:var(--neg);">V2</a> (36/39, stage 07) is on this list; both are
historical snapshots. V3 excludes every one of the canonical split's {data['n_matches_test_excluded']} TEST
matches before computing anything, scoring only the {data['n_matches_train_val']} TRAIN+VAL matches --
FEATURE_LOCK_CONFIRMATION.html's underlying data scores the same 34/38 list on the full 115-match population.</p>
</div>
"""
    if locked_data is not None:
        body += _tier_diff_banner(locked_data, data)
    body += _method_strip()
    for ds_key in ("active", "passive"):
        body += _dataset_block(ds_key, datasets[ds_key])

    return render.render_article(
        eyebrow="CORRELATION ATLAS — V3 (TRAIN+VAL ONLY)",
        title="Feature Correlation Atlas (V3, Train+Val Only)",
        dek=(
            "The final locked feature lists, scored with every canonical-split TEST match excluded -- a check "
            "on whether V1/V2's full-population scoring ever leaked into a redundancy decision."
        ),
        stats=[
            (datasets["active"]["n_rows_train_val"], "active rows (train+val)"),
            (datasets["passive"]["n_rows_train_val"], "passive rows (train+val, before sampling)"),
            (data["n_matches_train_val"], "train+val matches"),
            (data["n_matches_test_excluded"], "test matches excluded"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    locked_data = json.loads(LOCKED_FULL_POPULATION_PATH.read_text(encoding="utf-8")) if LOCKED_FULL_POPULATION_PATH.exists() else None
    html = build_report(data, locked_data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
