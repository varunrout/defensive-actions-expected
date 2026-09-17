"""CLI entrypoint: build a single-tab portal shell (reports/eda/INDEX.html)
that lists every formal EDA report in one sidebar, grouped by pipeline
stage, each opening in an embedded pane -- so reading through the whole
EDA trail never requires more than one browser tab.

Pure HTML navigation (<a target="viewer">, no JavaScript required) so it
works even with scripts disabled; a small amount of JS only handles
highlighting the active link. The report list is generated from what's
actually on disk in reports/eda/, not a hand-maintained list that can drift
-- any file present but not in REPORT_GROUPS below is still listed, under
"Other reports", so nothing silently disappears from the portal.

Usage:
    python -m src.eda.generate_eda_portal
"""

from __future__ import annotations

from pathlib import Path

from src.eda.render import FONT_LINKS, esc

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports" / "eda"
OUTPUT_PATH = REPORTS_DIR / "INDEX.html"

DEFAULT_REPORT = "MASTER_FINDINGS.html"

# (group title, [(filename, label), ...]) -- order here is the order shown
# in the sidebar. A file present on disk but not listed here still appears,
# under "Other reports", so nothing is silently hidden.
REPORT_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Overview", [
        ("MASTER_FINDINGS.html", "Master Findings -- start here"),
        ("EDA_PIPELINE_LOG.html", "Pipeline Audit Trail"),
    ]),
    ("Base EDA (stage 00)", [
        ("EDA_ANALYSIS.html", "Narrative Summary"),
        ("active_category_atlas.html", "Active -- Category Atlas"),
        ("active_flag_ledger.html", "Active -- Flag Ledger"),
        ("active_distribution_atlas.html", "Active -- Distribution Atlas"),
        ("active_distribution_atlas_reconstructed.html", "Active -- Distribution Atlas (reconstructed pool)"),
        ("active_numerical_target_atlas.html", "Active -- Numerical Target Atlas"),
        ("passive_category_atlas.html", "Passive -- Category Atlas"),
        ("passive_flag_ledger.html", "Passive -- Flag Ledger"),
        ("passive_distribution_atlas.html", "Passive -- Distribution Atlas"),
        ("passive_distribution_atlas_reconstructed.html", "Passive -- Distribution Atlas (reconstructed pool)"),
        ("passive_numerical_target_atlas.html", "Passive -- Numerical Target Atlas"),
    ]),
    ("Correlation & Review", [
        ("CORRELATION_ATLAS.html", "Correlation Atlas -- V1 (51/44, stage 01)"),
        ("REVIEW_METHODOLOGY.html", "Review Methodology -- V1"),
        ("CORRELATION_ATLAS_V2.html", "Correlation Atlas -- V2 (36/39, stage 07)"),
        ("REVIEW_METHODOLOGY_V2.html", "Review Methodology -- V2"),
        ("CORRELATION_ATLAS_V3.html", "Correlation Atlas -- V3 (34/38, train+val only)"),
    ]),
    ("Redundancy checks", [
        ("VIF_ANALYSIS.html", "Multicollinearity (VIF)"),
        ("CONFOUND_ANALYSIS.html", "Confound (Reversal) Testing"),
        ("LEAKAGE_AUDIT.html", "Leakage Audit"),
        ("TOURNAMENT_STABILITY_CHECK.html", "Tournament Stability Check"),
        ("SLICE_STRATIFICATION.html", "Slice Stratification (V1 -- categorical slicers)"),
        ("SLICE_STRATIFICATION_V2.html", "Slice Stratification V2 -- archetype + boolean slicers"),
        ("SLICER_REDUNDANCY.html", "Slicer Redundancy Check"),
        ("FEATURE_INTERACTION_ANALYSIS.html", "Feature Interaction Analysis (numeric x numeric)"),
    ]),
    ("Validation", [
        ("FOOTBALL_SANITY_CHECK.html", "Football Sanity Check"),
    ]),
    ("Modelling infrastructure", [
        ("PASSIVE_ARCHETYPES.html", "Passive Archetype Clustering"),
        ("SPLIT_VALIDATION.html", "Canonical Match-Grouped Split"),
        ("PLAYER_LEVEL_VALIDITY_CHECK.html", "Player-Level Validity Check (active only)"),
    ]),
    ("Confirmation", [
        ("FEATURE_LOCK_CONFIRMATION.html", "Feature Lock Confirmation + Pattern Findings"),
    ]),
]

PORTAL_CSS = """
:root {
  --bg: #f4f5f1; --plane: #eceee8; --surface: #ffffff;
  --text-primary: #10151c; --text-secondary: #4c525b; --text-muted: #868d96;
  --border: rgba(16,21,28,0.09); --accent: #2a78d6; --accent-wash: rgba(42,120,214,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d1117; --plane: #090c10; --surface: #161b22;
    --text-primary: #f3f5f7; --text-secondary: #b6bdc7; --text-muted: #7c8591;
    --border: rgba(255,255,255,0.09); --accent: #3987e5; --accent-wash: rgba(57,135,229,0.14);
  }
}
:root[data-theme="dark"] {
  --bg: #0d1117; --plane: #090c10; --surface: #161b22;
  --text-primary: #f3f5f7; --text-secondary: #b6bdc7; --text-muted: #7c8591;
  --border: rgba(255,255,255,0.09); --accent: #3987e5; --accent-wash: rgba(57,135,229,0.14);
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text-primary);
  font-family: "Public Sans", -apple-system, sans-serif; font-size: 14px; }
.shell { display: grid; grid-template-columns: 300px 1fr; height: 100vh; }
@media (max-width: 760px) { .shell { grid-template-columns: 1fr; height: auto; } .viewer-pane { height: 80vh; } }
.sidebar {
  background: var(--plane); border-right: 1px solid var(--border); overflow-y: auto; padding: 18px 14px 40px;
  scrollbar-width: thin; scrollbar-color: var(--text-muted) transparent;
}
.sidebar::-webkit-scrollbar { width: 7px; }
.sidebar::-webkit-scrollbar-track { background: transparent; }
.sidebar::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 4px; }
.sidebar::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
.sb-title { font-family: "Archivo", sans-serif; font-weight: 800; font-size: 16px; margin: 4px 6px 2px; }
.sb-subtitle { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.04em; margin: 0 6px 18px; }
.sb-group { margin-bottom: 6px; }
.sb-group-title { font-family: "JetBrains Mono", monospace; font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--text-muted); padding: 12px 10px 6px; }
.sb-link { display: block; padding: 8px 10px; border-radius: 6px; color: var(--text-secondary);
  text-decoration: none; font-size: 13px; line-height: 1.35; margin-bottom: 1px; }
.sb-link:hover { background: var(--surface); color: var(--text-primary); }
.sb-link.active { background: var(--accent-wash); color: var(--accent); font-weight: 600; }
.sb-file { display: block; font-family: "JetBrains Mono", monospace; font-size: 9.5px; color: var(--text-muted); margin-top: 1px; }
.viewer-pane { background: var(--surface); }
.viewer-pane iframe { width: 100%; height: 100%; border: none; display: block; }
"""

PORTAL_JS = """
(function() {
  var frame = document.getElementById('viewer');
  var links = document.querySelectorAll('.sb-link');
  function setActive(href) {
    links.forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === href);
    });
  }
  links.forEach(function(a) {
    a.addEventListener('click', function() { setActive(a.getAttribute('href')); });
  });
  setActive(frame.getAttribute('src'));
})();
"""


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
<title>EDA Report Portal</title>
{FONT_LINKS}
<style>{PORTAL_CSS}</style>
</head>
<body>
<div class="shell">
  <div class="sidebar">
    <p class="sb-title">EDA Report Portal</p>
    <p class="sb-subtitle">{n_reports} reports, one tab</p>
    {nav_html}
  </div>
  <div class="viewer-pane">
    <iframe id="viewer" name="viewer" src="{esc(DEFAULT_REPORT)}" title="Selected EDA report"></iframe>
  </div>
</div>
<script>{PORTAL_JS}</script>
</body>
</html>"""


def main() -> None:
    html = build_portal()
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
