"""CLI entrypoint: build the PASSIVE half of the xT-delta report portal, and
rebuild the shared portal index covering BOTH legs.

STEP-0 DISPOSITION: new sibling driver + a rebuilt portal shell.

This file does two jobs, mirroring what `generate_eda_portal_xt_v2.py` does
for the active leg.

1. DRIVER. It runs the twelve passive target-dependent reports in
   dependency order and -- critically -- imports `xt_passive_common` FIRST.
   Every generator in this suite binds `TARGET = xc.TARGET` at module-import
   time, so the re-point must land before any of them is imported. Running a
   passive generator directly still works because each one imports
   `xt_passive_common` itself, first, for exactly this reason.

   **This driver and `generate_eda_portal_xt_v2` must never run in the same
   process.** Both re-point the same `xt_common` module attributes and
   whichever imports last would win. Each is its own process; the active
   portal is not rebuilt here and its files are not touched.

2. PORTAL SHELL, rebuilt for BOTH legs. `generate_eda_portal_xt.py`'s
   CSS/JS/structure is reused (it in turn imports the xg portal's
   PORTAL_CSS / PORTAL_JS), so this portal stays visually identical to the
   xg and shot portals. The group structure now mirrors
   `xg_target/INDEX.html`'s own Active-then-Passive interleaving: the base
   EDA group lists the four active atlases followed by the four passive
   ones, and the redundancy/leakage/confound group lists each report's
   active file followed by its passive one.

   The ACTIVE entries are asserted against
   `generate_eda_portal_xt_v2.REPORT_GROUPS` at run time so that this file
   cannot silently drop or rename one of Prompt 67's reports. That import is
   deliberately LAZY and happens only inside `build_portal()`, which runs
   after every passive report has already been written -- importing it
   re-points `xt_common` back onto v2, which is harmless once no data work
   remains but would not be if it happened at module import time.

Usage:
    python -m src.eda.generate_eda_portal_xt_passive            # portal shell only
    python -m src.eda.generate_eda_portal_xt_passive --all      # full passive rebuild
"""

from __future__ import annotations

import sys

from src.eda import xt_passive_common as pc  # MUST be first -- re-points xt_common for every import below
from src.eda import xt_common as xc
from src.eda.generate_eda_portal_xg import PORTAL_CSS, PORTAL_JS
from src.eda.render import FONT_LINKS, esc

REPORTS_DIR = xc.OUT_DIR
OUTPUT_PATH = REPORTS_DIR / "INDEX.html"
V1_PORTAL = "../xt_target_v1_superseded/INDEX.html"

DEFAULT_REPORT = "MASTER_FINDINGS.html"

REPORT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Overview", [
        ("MASTER_FINDINGS.html", "Master Findings -- start here (covers BOTH legs)"),
    ]),
    ("Target-independent (mirrored, not rebuilt)", [
        ("../xg_target/CORRELATION_ATLAS.html", "Correlation Atlas -- V1 (51/44, stage 01)"),
        ("../xg_target/CORRELATION_ATLAS_V2.html", "Correlation Atlas -- V2 (36/39, stage 07)"),
        ("../xg_target/CORRELATION_ATLAS_V3.html", "Correlation Atlas -- V3 (34/38, train+val only)"),
        ("../xg_target/VIF_ANALYSIS.html", "Multicollinearity (VIF)"),
        ("../xg_target/SLICER_REDUNDANCY.html", "Slicer Redundancy Check"),
        ("../xg_target/PLAYER_LEVEL_VALIDITY_CHECK.html", "Player-Level Validity Check (active only)"),
        ("../shot_target/PLAYER_GROUPED_SPLIT_CHECK.html",
         "Player-Grouped Split Check (active only -- identity-leakage stress test)"),
    ]),
    ("Base EDA vs xT delta (reconstructed pre-drop pools)", [
        ("active_category_atlas.html", "Active -- Category Atlas (xT v2)"),
        ("active_flag_ledger.html", "Active -- Flag Ledger (xT v2)"),
        ("active_distribution_atlas.html", "Active -- Distribution Atlas (locked list)"),
        ("active_distribution_atlas_reconstructed.html", "Active -- Distribution Atlas (reconstructed pool)"),
        ("active_numerical_target_atlas.html", "Active -- Numerical XT Target Atlas (v2)"),
        ("passive_category_atlas.html", "Passive -- Category Atlas (xT passive)"),
        ("passive_flag_ledger.html", "Passive -- Flag Ledger (xT passive)"),
        ("passive_distribution_atlas.html", "Passive -- Distribution Atlas (locked list)"),
        ("passive_distribution_atlas_reconstructed.html", "Passive -- Distribution Atlas (reconstructed pool)"),
        ("passive_numerical_target_atlas.html", "Passive -- Numerical XT Target Atlas"),
    ]),
    ("Redundancy / leakage / confound vs xT delta -- ACTIVE", [
        ("REVIEW_METHODOLOGY.html", "Active -- Review Methodology (xT v2 lift)"),
        ("LEAKAGE_AUDIT.html", "Active -- Leakage Audit (xT v2)"),
        ("CONFOUND_ANALYSIS.html", "Active -- Confound (Reversal) Testing (xT v2)"),
        ("TOURNAMENT_STABILITY_CHECK.html", "Active -- Tournament Stability Check (xT v2)"),
        ("SLICE_STRATIFICATION.html", "Active -- Slice Stratification (V1, categorical slicers)"),
        ("SLICE_STRATIFICATION_V2.html", "Active -- Slice Stratification V2 (boolean-flag slicers)"),
        ("FEATURE_INTERACTION_ANALYSIS.html", "Active -- Feature Interaction Analysis (numeric x numeric)"),
    ]),
    ("Redundancy / leakage / confound vs xT delta -- PASSIVE", [
        ("PASSIVE_REVIEW_METHODOLOGY.html", "Passive -- Review Methodology (xT passive lift)"),
        ("PASSIVE_LEAKAGE_AUDIT.html", "Passive -- Leakage Audit (xT passive)"),
        ("PASSIVE_CONFOUND_ANALYSIS.html", "Passive -- Confound (Reversal) Testing (xT passive)"),
        ("PASSIVE_TOURNAMENT_STABILITY_CHECK.html", "Passive -- Tournament Stability Check (xT passive)"),
        ("PASSIVE_SLICE_STRATIFICATION.html", "Passive -- Slice Stratification (V1, categorical slicers)"),
        ("PASSIVE_SLICE_STRATIFICATION_V2.html",
         "Passive -- Slice Stratification V2 (archetype + boolean slicers)"),
        ("PASSIVE_FEATURE_INTERACTION_ANALYSIS.html",
         "Passive -- Feature Interaction Analysis (numeric x numeric)"),
    ]),
    ("Confirmation", [
        ("FEATURE_LOCK_CONFIRMATION_XT.html", "Active -- Feature Lock Confirmation + Pattern Findings"),
        ("PASSIVE_FEATURE_LOCK_CONFIRMATION_XT.html",
         "Passive -- Feature Lock Confirmation + Pattern Findings"),
    ]),
    ("Superseded", [
        (V1_PORTAL, "v1 ACTIVE portal (target_xt_delta, Prompt 65) -- preserved, not deleted"),
    ]),
]

FOOTER_NOTE = (
    "Targets: target_xt_delta_v2 (ACTIVE-BINARY leg, from "
    "outputs/prototypes/active_binary_xt_delta_v2.parquet) and target_xt_delta_passive (PASSIVE leg, from "
    "outputs/prototypes/passive_xt_delta.parquet). BOTH LEGS ARE COVERED -- any earlier statement in this "
    "portal that the passive leg is out of scope is superseded. The two legs share one xt_after definition "
    "(0.0 if the event ends its possession; else the next same-possession event's grid value, or that event's "
    "own shot xG if it is a Shot) but differ in what a row is, where xt_before is looked up, and what grain "
    "the target is computed at: the passive target is ONE VALUE PER EVENT, left-joined onto every visible "
    "defender-slot row of that event, exactly as target_future_shot_10s and target_future_xg_10s already are "
    "on that leg. THE ACTIVE HALF SUPERSEDES AN EARLIER PASS: Prompt 65 built its thirteen reports against "
    "target_xt_delta (v1), whose xt_after scored the defensive action's own recorded location; Prompt 66 found "
    "that wrong and Prompt 67 rebuilt against the correction. The v1 portal is preserved UNCHANGED at "
    "reports/analysis/xt_target_v1_superseded/. The passive half has no superseded version -- Prompt 68 "
    "implemented the corrected definition on the first pass. FILE-NAMING NOTE: the xg portal keeps both legs' "
    "sections inside one file for the eight shared-name reports (Leakage Audit, Confound, Tournament "
    "Stability, both Slice Stratifications, Feature Interaction, Feature Lock Confirmation, Review "
    "Methodology). That convention is NOT mirrored here, deliberately: this portal's files of those names are "
    "Prompt 67's active-only output and are left byte-for-byte as they were, so the passive halves are "
    "separate PASSIVE_* files. Separate from BOTH the binary-target portal (reports/analysis/shot_target/) and "
    "the continuous-xG portal (reports/analysis/xg_target/). The seven links under \"Target-independent "
    "(mirrored)\" point OUT to existing files in the other portals -- those reports are properties of the "
    "locked features, not of the target, so no target change can move them; three of them already carry both "
    "legs, and the two Player-* checks are ACTIVE-ONLY project-wide because the passive leg is not "
    "player-indexed the same way."
)


def _known_files() -> set[str]:
    return {f for _, entries in REPORT_GROUPS for f, _ in entries}


def _discover_unlisted() -> list[tuple[str, str]]:
    known = _known_files()
    on_disk = sorted(p.name for p in REPORTS_DIR.glob("*.html") if p.name != OUTPUT_PATH.name)
    return [(f, f) for f in on_disk if f not in known]


def _assert_active_entries_preserved() -> None:
    """Every active-leg entry Prompt 67's portal listed must still be listed
    here, under some group. LAZY import: `generate_eda_portal_xt_v2` imports
    `xt_v2_common`, which re-points `xt_common` back onto the active target.
    That is harmless at this point -- the portal shell is written last, after
    every passive report already exists -- but it must not happen at module
    import time, which is why this is not a top-level import."""
    from src.eda import generate_eda_portal_xt_v2 as v2portal

    listed = _known_files()
    v2_files = {f for _, entries in v2portal.REPORT_GROUPS for f, _ in entries}
    # The v1-portal link is relative to this directory in both files, but the
    # v2 portal builds its path from V1_PORTAL_DIRNAME; compare on basenames.
    missing = {f for f in v2_files if f not in listed and not f.endswith("xt_target_v1_superseded/INDEX.html")}
    assert not missing, (
        "The rebuilt portal index drops report(s) that Prompt 67's index listed: "
        f"{sorted(missing)}. Refusing to publish an index that hides existing active-leg reports."
    )


def build_portal() -> str:
    _assert_active_entries_preserved()

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
<title>EDA Report Portal -- xT Delta (Active + Passive Legs)</title>
{FONT_LINKS}
<style>{PORTAL_CSS}</style>
</head>
<body>
<div class="shell">
  <div class="sidebar">
    <p class="sb-title">EDA Report Portal -- xT Delta</p>
    <p class="sb-subtitle">{n_reports} report(s) &middot; <code>target_xt_delta_v2</code> (active) +
    <code>target_xt_delta_passive</code> (passive) &middot; both legs covered</p>
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


# Dependency-ordered build of the PASSIVE half. The active half is NOT
# rebuilt here and none of its files is touched.
BUILD_STEPS: list[tuple[str, str]] = [
    ("src.eda.generate_reports_xt_passive", "Passive Category Atlas + Flag Ledger"),
    ("src.eda.generate_distribution_atlas_xt_passive", "Passive Distribution Atlas (+ reconstructed variant)"),
    ("src.eda.generate_numerical_xt_target_passive", "Passive Numerical Target Atlas"),
    ("src.eda.generate_review_analysis_xt_passive", "Passive Review Methodology"),
    ("src.eda.generate_leakage_audit_xt_passive", "Passive Leakage Audit"),
    ("src.eda.generate_confound_analysis_xt_passive", "Passive Confound Testing"),
    ("src.eda.generate_tournament_stability_check_xt_passive", "Passive Tournament Stability"),
    ("src.eda.generate_slice_stratification_xt_passive", "Passive Slice Stratification V1 + V2"),
    ("src.eda.generate_feature_interaction_analysis_xt_passive", "Passive Feature Interaction"),
    ("src.eda.generate_feature_lock_confirmation_xt_passive", "Passive Feature Lock Confirmation"),
    ("src.eda.generate_master_findings_xt_passive", "Master Findings (both legs)"),
]


def build_all() -> None:
    import importlib

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (mod_name, label) in enumerate(BUILD_STEPS, 1):
        print(f"\n--- [{i}/{len(BUILD_STEPS)}] {label}  ({mod_name}) ---")
        mod = importlib.import_module(mod_name)
        bound = getattr(mod, "TARGET", pc.TARGET_PASSIVE)
        if bound != pc.TARGET_PASSIVE:
            raise RuntimeError(
                f"{mod_name} bound TARGET={bound!r}, expected {pc.TARGET_PASSIVE!r} -- the passive re-point "
                "did not land before this import. Refusing to write an active-leg report into the passive "
                "half of the portal."
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
