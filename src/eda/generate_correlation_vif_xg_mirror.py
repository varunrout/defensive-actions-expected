"""CLI entrypoint: mirror the target-independent arm of the binary-target
pipeline (Correlation Atlas V1/V2/V3, VIF Analysis, Slicer Redundancy Check,
Player-Level Validity Check) into reports/analysis/xg_target/, so the xG portal is a
complete, self-contained arm -- not missing the pieces that don't happen to
change with the target.

These are NOT recomputed differently for xG: correlation (feature vs
feature), VIF (a feature's own multicollinearity against every other
feature), slicer redundancy (association between two categorical slicer
columns), and the player-level validity check (row concentration,
per-feature ICC/Cramer's V against player_id, train/test player overlap --
all computed from player_id and match_assignment.json, never a target
column) never reference a target column at all -- the numbers are
identical to the binary-target versions by construction. This script reuses
the EXISTING build functions from generate_correlation_reports.py,
generate_correlation_reports_v2.py, generate_correlation_atlas_v3_report.py,
generate_vif_reports.py, generate_slicer_redundancy_report.py and
generate_player_level_validity_report.py directly, reading the same
already-computed JSON (no re-analysis), and injects one small banner into
each explaining why the content matches the binary-target page exactly.

Usage:
    python -m src.eda.generate_correlation_vif_xg_mirror
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda.generate_correlation_reports import build_correlation_atlas as build_v1
from src.eda.generate_correlation_reports_v2 import build_correlation_atlas as build_v2
from src.eda.generate_correlation_atlas_v3_report import build_report as build_v3
from src.eda.generate_vif_reports import build_vif_report as build_vif
from src.eda.generate_slicer_redundancy_report import build_report as build_slicer_redundancy
from src.eda.generate_player_level_validity_report import build_report as build_player_level_validity

REPO_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = REPO_ROOT / "reports" / "analysis" / "shot_target"
EDA_XG_DIR = REPO_ROOT / "reports" / "analysis" / "xg_target"

TARGET_INDEPENDENT_BANNER = """
<div class="finding" style="margin-bottom:24px;">
<span class="tag">target-independent -- mirrored, not recomputed</span>
<p>Correlation (feature vs feature), VIF (a feature's own multicollinearity against every other feature), slicer
redundancy (association between two categorical slicer columns), and the player-level validity check (row
concentration, per-feature ICC/Cramer's V against player_id, train/test player overlap) never reference a target
column at all -- this page's numbers are <b>identical</b> to
<a href="../eda/{binary_filename}" style="color:var(--neg);">the binary-target version</a> by construction, not
independently re-derived. Mirrored here so the xG portal is a complete, self-contained arm.</p>
</div>
"""


def _load(name: str) -> dict:
    return json.loads((EDA_DIR / name).read_text(encoding="utf-8"))


def _inject_banner(html: str, binary_filename: str) -> str:
    banner = TARGET_INDEPENDENT_BANNER.format(binary_filename=binary_filename)
    marker = '<div class="wrap">'
    idx = html.find(marker)
    if idx == -1:
        raise AssertionError(f"Expected marker {marker!r} not found -- html_shell/render_article's structure changed.")
    insert_at = idx + len(marker)
    return html[:insert_at] + banner + html[insert_at:]


def main() -> None:
    EDA_XG_DIR.mkdir(parents=True, exist_ok=True)

    v1_data = _load("CORRELATION_ANALYSIS_V1_HISTORICAL.json")
    v2_data = _load("CORRELATION_ANALYSIS_V2.json")
    v3_data = _load("CORRELATION_ANALYSIS_V3.json")
    locked_data = _load("CORRELATION_ANALYSIS.json")
    vif_data = _load("VIF_ANALYSIS.json")
    slicer_redundancy_data = _load("SLICER_REDUNDANCY.json")
    player_validity_data = _load("PLAYER_LEVEL_VALIDITY_CHECK.json")

    outputs = {
        "CORRELATION_ATLAS.html": (build_v1(v1_data), "CORRELATION_ATLAS.html"),
        "CORRELATION_ATLAS_V2.html": (build_v2(v2_data, v1_data), "CORRELATION_ATLAS_V2.html"),
        "CORRELATION_ATLAS_V3.html": (build_v3(v3_data, locked_data), "CORRELATION_ATLAS_V3.html"),
        "VIF_ANALYSIS.html": (build_vif(vif_data), "VIF_ANALYSIS.html"),
        "SLICER_REDUNDANCY.html": (build_slicer_redundancy(slicer_redundancy_data), "SLICER_REDUNDANCY.html"),
        "PLAYER_LEVEL_VALIDITY_CHECK.html": (build_player_level_validity(player_validity_data), "PLAYER_LEVEL_VALIDITY_CHECK.html"),
    }

    for filename, (html, binary_filename) in outputs.items():
        html = _inject_banner(html, binary_filename)
        out_path = EDA_XG_DIR / filename
        out_path.write_text(html, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
