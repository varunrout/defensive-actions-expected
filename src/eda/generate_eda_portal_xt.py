"""CLI entrypoint: build the single-tab portal shell
(reports/analysis/xt_target/INDEX.html) for the xT-delta report collection.

The CSS/JS/shell structure is copied from generate_eda_portal_xg.py --
imported directly, not duplicated -- so this portal is visually identical
to the xg one. Only the title, subtitle, group structure, default report
and footer note differ.

Group structure, per the task's own specification:
  - "Overview": this portal's own Master Findings.
  - "Target-independent (mirrored)": relative links OUT to the existing
    xg_target files for the five reports that are properties of the
    features rather than of the target. They are NOT rebuilt.
  - Target-dependent groups: the reports rebuilt against target_xt_delta.

Usage:
    python -m src.eda.generate_eda_portal_xt
"""

from __future__ import annotations

from src.eda import xt_common as xc
from src.eda.generate_eda_portal_xg import PORTAL_CSS, PORTAL_JS
from src.eda.render import FONT_LINKS, esc

REPORTS_DIR = xc.OUT_DIR
OUTPUT_PATH = REPORTS_DIR / "INDEX.html"

DEFAULT_REPORT = "MASTER_FINDINGS.html"

REPORT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Overview", [
        ("MASTER_FINDINGS.html", "Master Findings -- start here (xT delta, active-binary leg only)"),
    ]),
    ("Target-independent (mirrored)", [
        ("../xg_target/CORRELATION_ATLAS.html", "Correlation Atlas -- V1 (51/44, stage 01)"),
        ("../xg_target/CORRELATION_ATLAS_V2.html", "Correlation Atlas -- V2 (36/39, stage 07)"),
        ("../xg_target/CORRELATION_ATLAS_V3.html", "Correlation Atlas -- V3 (34/38, train+val only)"),
        ("../xg_target/VIF_ANALYSIS.html", "Multicollinearity (VIF)"),
        ("../xg_target/SLICER_REDUNDANCY.html", "Slicer Redundancy Check"),
        ("../xg_target/PLAYER_LEVEL_VALIDITY_CHECK.html", "Player-Level Validity Check (active only)"),
        ("../shot_target/PLAYER_GROUPED_SPLIT_CHECK.html", "Player-Grouped Split Check (identity-leakage stress test)"),
    ]),
    ("Base EDA vs xT delta (reconstructed 51-era pool, active)", [
        ("active_category_atlas.html", "Active -- Category Atlas (xT)"),
        ("active_flag_ledger.html", "Active -- Flag Ledger (xT)"),
        ("active_distribution_atlas.html", "Active -- Distribution Atlas (locked list)"),
        ("active_distribution_atlas_reconstructed.html", "Active -- Distribution Atlas (reconstructed pool)"),
        ("active_numerical_target_atlas.html", "Active -- Numerical XT Target Atlas"),
    ]),
    ("Redundancy / leakage / confound vs xT delta (active)", [
        ("REVIEW_METHODOLOGY.html", "Review Methodology (xT lift)"),
        ("LEAKAGE_AUDIT.html", "Leakage Audit (xT)"),
        ("CONFOUND_ANALYSIS.html", "Confound (Reversal) Testing (xT)"),
        ("TOURNAMENT_STABILITY_CHECK.html", "Tournament Stability Check (xT)"),
        ("SLICE_STRATIFICATION.html", "Slice Stratification (xT, V1 -- categorical slicers)"),
        ("SLICE_STRATIFICATION_V2.html", "Slice Stratification V2 (xT) -- boolean-flag slicers"),
        ("FEATURE_INTERACTION_ANALYSIS.html", "Feature Interaction Analysis (numeric x numeric, xT)"),
    ]),
    ("Confirmation", [
        ("FEATURE_LOCK_CONFIRMATION_XT.html", "Feature Lock Confirmation + Pattern Findings (xT)"),
    ]),
]

FOOTER_NOTE = (
    "Separate from BOTH the binary-target portal (reports/analysis/shot_target/INDEX.html, "
    "target_future_shot_10s) and the continuous-xG portal (reports/analysis/xg_target/INDEX.html, "
    "target_future_xg_10s) -- different target (target_xt_delta), different distribution shape, different "
    "thresholds. Scoped to the ACTIVE-BINARY LEG ONLY: the passive leg is out of scope for this target and no "
    "passive-side xT report exists. The seven links under \"Target-independent (mirrored)\" point OUT to the "
    "existing files in the other portals -- those reports are properties of the locked features, not of the "
    "target, so they are reused rather than rebuilt."
)


def _known_files() -> set[str]:
    return {f for _, entries in REPORT_GROUPS for f, _ in entries}


def _discover_unlisted() -> list[tuple[str, str]]:
    known = _known_files()
    on_disk = sorted(p.name for p in REPORTS_DIR.glob("*.html") if p.name != OUTPUT_PATH.name)
    return [(f, f) for f in on_disk if f not in known]


def build_portal() -> str:
    groups = list(REPORT_GROUPS)
    unlisted = _discover_unlisted()
    if unlisted:
        groups.append(("Other reports", unlisted))

    nav_html = ""
    for group_title, entries in groups:
        links = "".join(
            f'<a class="sb-link" href="{esc(fname)}" target="viewer">{esc(label)}'
            f'<span class="sb-file">{esc(fname)}</span></a>'
            for fname, label in entries
        )
        nav_html += f'<div class="sb-group"><div class="sb-group-title">{esc(group_title)}</div>{links}</div>'

    n_reports = sum(len(entries) for _, entries in groups)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA Report Portal -- xT Delta (Active-Binary Leg)</title>
{FONT_LINKS}
<style>{PORTAL_CSS}</style>
</head>
<body>
<div class="shell">
  <div class="sidebar">
    <p class="sb-title">EDA Report Portal -- xT Delta</p>
    <p class="sb-subtitle">{n_reports} report(s) &middot; active-binary leg only</p>
    {nav_html}
    <p class="sb-note">{esc(FOOTER_NOTE)}</p>
  </div>
  <div class="viewer-pane">
    <iframe id="viewer" name="viewer" src="{esc(DEFAULT_REPORT)}" title="Selected xT EDA report"></iframe>
  </div>
</div>
<script>{PORTAL_JS}</script>
</body>
</html>"""


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_portal(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    unlisted = _discover_unlisted()
    if unlisted:
        print(f"  NOTE: {len(unlisted)} unlisted HTML file(s) appended under 'Other reports': "
              f"{', '.join(f for f, _ in unlisted)}")


if __name__ == "__main__":
    main()
