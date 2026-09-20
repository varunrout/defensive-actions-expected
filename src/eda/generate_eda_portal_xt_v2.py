"""CLI entrypoint: build the whole `target_xt_delta_v2` report portal --
driver + portal shell for reports/analysis/xt_target/.

STEP-0 DISPOSITION: REWRITE (of the portal shell) + DRIVER (new role).

This file does two jobs.

1. DRIVER. It runs the thirteen target-dependent reports in dependency
   order, and -- critically -- it imports `xt_v2_common` FIRST. Every one of
   Prompt 65's generators binds `TARGET = xc.TARGET` at module-import time,
   so the re-point must land before any of them is imported. Running a
   generator directly (`python -m src.eda.generate_distribution_atlas_xt`)
   still produces the v1 report against the v1 parquet; it is only through
   this driver, or through an explicit `import src.eda.xt_v2_common` first,
   that the suite points at v2. That asymmetry is deliberate: it is what
   keeps Prompt 65's 21 scripts exactly reproducible.

2. PORTAL SHELL. `generate_eda_portal_xt.py`'s CSS/JS/structure is imported
   and reused unchanged (it in turn imports the xg portal's PORTAL_CSS /
   PORTAL_JS), so this portal is visually identical to the v1 and xg ones.
   Only the sidebar title/subtitle, the target name and the footer note
   differ -- the footer now points back at the superseded v1 portal instead
   of presenting v2 as if it had always existed.

Usage:
    python -m src.eda.generate_eda_portal_xt_v2            # portal shell only
    python -m src.eda.generate_eda_portal_xt_v2 --all      # full rebuild
"""

from __future__ import annotations

import sys

from src.eda import xt_v2_common as v2  # MUST be first -- re-points xt_common for every import below
from src.eda import xt_common as xc
from src.eda.generate_eda_portal_xg import PORTAL_CSS, PORTAL_JS
from src.eda.render import FONT_LINKS, esc

REPORTS_DIR = xc.OUT_DIR
OUTPUT_PATH = REPORTS_DIR / "INDEX.html"
V1_PORTAL = f"../{v2.V1_PORTAL_DIRNAME}/INDEX.html"

DEFAULT_REPORT = "MASTER_FINDINGS.html"

REPORT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Overview", [
        ("MASTER_FINDINGS.html", "Master Findings -- start here (xT delta v2, active-binary leg only)"),
    ]),
    ("Target-independent (mirrored, not rebuilt)", [
        ("../xg_target/CORRELATION_ATLAS.html", "Correlation Atlas -- V1 (51/44, stage 01)"),
        ("../xg_target/CORRELATION_ATLAS_V2.html", "Correlation Atlas -- V2 (36/39, stage 07)"),
        ("../xg_target/CORRELATION_ATLAS_V3.html", "Correlation Atlas -- V3 (34/38, train+val only)"),
        ("../xg_target/VIF_ANALYSIS.html", "Multicollinearity (VIF)"),
        ("../xg_target/SLICER_REDUNDANCY.html", "Slicer Redundancy Check"),
        ("../xg_target/PLAYER_LEVEL_VALIDITY_CHECK.html", "Player-Level Validity Check (active only)"),
        ("../shot_target/PLAYER_GROUPED_SPLIT_CHECK.html", "Player-Grouped Split Check (identity-leakage stress test)"),
    ]),
    ("Base EDA vs xT delta v2 (reconstructed 51-era pool, active)", [
        ("active_category_atlas.html", "Active -- Category Atlas (xT v2)"),
        ("active_flag_ledger.html", "Active -- Flag Ledger (xT v2)"),
        ("active_distribution_atlas.html", "Active -- Distribution Atlas (locked list)"),
        ("active_distribution_atlas_reconstructed.html", "Active -- Distribution Atlas (reconstructed pool)"),
        ("active_numerical_target_atlas.html", "Active -- Numerical XT Target Atlas (v2)"),
    ]),
    ("Redundancy / leakage / confound vs xT delta v2 (active)", [
        ("REVIEW_METHODOLOGY.html", "Review Methodology (xT v2 lift)"),
        ("LEAKAGE_AUDIT.html", "Leakage Audit (xT v2)"),
        ("CONFOUND_ANALYSIS.html", "Confound (Reversal) Testing (xT v2)"),
        ("TOURNAMENT_STABILITY_CHECK.html", "Tournament Stability Check (xT v2)"),
        ("SLICE_STRATIFICATION.html", "Slice Stratification (xT v2, V1 -- categorical slicers)"),
        ("SLICE_STRATIFICATION_V2.html", "Slice Stratification V2 (xT v2) -- boolean-flag slicers"),
        ("FEATURE_INTERACTION_ANALYSIS.html", "Feature Interaction Analysis (numeric x numeric, xT v2)"),
    ]),
    ("Confirmation", [
        ("FEATURE_LOCK_CONFIRMATION_XT.html", "Feature Lock Confirmation + Pattern Findings (xT v2)"),
    ]),
    ("Superseded", [
        (V1_PORTAL, "v1 portal (target_xt_delta, Prompt 65) -- preserved, not deleted"),
    ]),
]

FOOTER_NOTE = (
    "Target: target_xt_delta_v2, from outputs/prototypes/active_binary_xt_delta_v2.parquet. THIS PORTAL "
    "SUPERSEDES an earlier pass. Prompt 65 built the same thirteen reports against target_xt_delta (v1), whose "
    "xt_after term was xT(action_x, action_y) -- the defensive action's own recorded location. Prompt 66 found "
    "that definition wrong (for a Clearance it scores where the ball WAS, not where it went) and replaced it "
    "with a possession-outcome definition. The v1 portal is preserved UNCHANGED at "
    "reports/analysis/xt_target_v1_superseded/ -- it is the accurate historical record of what Prompt 65 found "
    "against v1, not something to correct. Separate from BOTH the binary-target portal "
    "(reports/analysis/shot_target/, target_future_shot_10s) and the continuous-xG portal "
    "(reports/analysis/xg_target/, target_future_xg_10s). Scoped to the ACTIVE-BINARY LEG ONLY: the passive leg "
    "is out of scope for this target and no passive-side xT report exists. The seven links under "
    "\"Target-independent (mirrored)\" point OUT to the existing files in the other portals -- those reports are "
    "properties of the locked features, not of the target, so a target correction cannot move them and they are "
    "reused rather than rebuilt."
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

    n_reports = sum(len(entries) for title, entries in groups if title != "Superseded")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA Report Portal -- xT Delta v2 (Active-Binary Leg)</title>
{FONT_LINKS}
<style>{PORTAL_CSS}</style>
</head>
<body>
<div class="shell">
  <div class="sidebar">
    <p class="sb-title">EDA Report Portal -- xT Delta v2</p>
    <p class="sb-subtitle">{n_reports} report(s) &middot; <code>target_xt_delta_v2</code> &middot; active-binary leg only</p>
    {nav_html}
    <p class="sb-note">{esc(FOOTER_NOTE)}</p>
  </div>
  <div class="viewer-pane">
    <iframe id="viewer" name="viewer" src="{esc(DEFAULT_REPORT)}" title="Selected xT v2 EDA report"></iframe>
  </div>
</div>
<script>{PORTAL_JS}</script>
</body>
</html>"""


# Dependency-ordered build. Modules marked "(v2)" are new siblings written
# because a re-point alone would have published a v1-specific claim; the
# rest are Prompt 65's own files, unedited, running against the re-pointed
# xt_common.
BUILD_STEPS: list[tuple[str, str]] = [
    ("src.eda.generate_reports_xt_v2", "Category Atlas + Flag Ledger (v2)"),
    ("src.eda.generate_distribution_atlas_xt", "Distribution Atlas (+ reconstructed variant)"),
    ("src.eda.generate_numerical_xt_target_analysis", "Numerical Target Atlas -- compute"),
    ("src.eda.generate_numerical_xt_target_reports", "Numerical Target Atlas -- render"),
    ("src.eda.generate_review_analysis_xt", "Review Methodology -- compute"),
    ("src.eda.generate_review_report_xt", "Review Methodology -- render"),
    ("src.eda.generate_leakage_audit_xt_v2", "Leakage Audit -- compute (v2)"),
    ("src.eda.generate_leakage_report_xt_v2", "Leakage Audit -- render (v2)"),
    ("src.eda.generate_confound_analysis_xt_v2", "Confound Testing -- compute (v2)"),
    ("src.eda.generate_confound_report_xt", "Confound Testing -- render"),
    ("src.eda.generate_tournament_stability_check_xt", "Tournament Stability -- compute"),
    ("src.eda.generate_tournament_stability_report_xt", "Tournament Stability -- render"),
    ("src.eda.generate_slice_stratification_xt", "Slice Stratification V1 -- compute"),
    ("src.eda.generate_slice_stratification_v2_xt", "Slice Stratification V2 -- compute"),
    ("src.eda.generate_slice_stratification_report_xt", "Slice Stratification V1+V2 -- render"),
    ("src.eda.generate_feature_interaction_analysis_xt", "Feature Interaction -- compute"),
    ("src.eda.generate_feature_interaction_report_xt", "Feature Interaction -- render"),
    ("src.eda.generate_feature_lock_confirmation_xt_v2", "Feature Lock Confirmation -- compute (v2)"),
    ("src.eda.generate_feature_lock_report_xt_v2", "Feature Lock Confirmation -- render (v2)"),
    ("src.eda.xt_v2_prose_fixups", "v2 prose corrections to re-pointed Prompt-65 outputs"),
    ("src.eda.generate_master_findings_xt_v2", "Master Findings (v2)"),
]


def build_all() -> None:
    import importlib

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (mod_name, label) in enumerate(BUILD_STEPS, 1):
        print(f"\n--- [{i}/{len(BUILD_STEPS)}] {label}  ({mod_name}) ---")
        mod = importlib.import_module(mod_name)
        if getattr(mod, "TARGET", v2.TARGET_V2) != v2.TARGET_V2 and hasattr(mod, "TARGET"):
            raise RuntimeError(
                f"{mod_name} bound TARGET={mod.TARGET!r}, expected {v2.TARGET_V2!r} -- the re-point did not "
                "land before this import. Refusing to write a v1 report into the v2 portal."
            )
        mod.main()


def main() -> None:
    if "--all" in sys.argv:
        build_all()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_portal(), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")
    unlisted = _discover_unlisted()
    if unlisted:
        print(f"  NOTE: {len(unlisted)} unlisted HTML file(s) appended under 'Other reports': "
              f"{', '.join(f for f, _ in unlisted)}")


if __name__ == "__main__":
    main()
