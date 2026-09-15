"""CLI entrypoint: render FEATURE_LOCK_CONFIRMATION.json as a self-contained
HTML report, in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.generate_feature_lock_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "FEATURE_LOCK_CONFIRMATION.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "FEATURE_LOCK_CONFIRMATION.html"


def _count_card(label: str, actual: int, expected: int, matches: bool) -> str:
    cls = "distinct" if matches else "drop"
    status = "CONFIRMED" if matches else "MISMATCH"
    return f"""
<div class="ba-card {'after' if matches else 'before'}">
  <h4>{esc(label)}</h4>
  <p style="font-size:28px; font-family:'Archivo',sans-serif; font-weight:800; margin:4px 0;">{actual}</p>
  <p style="font-size:12px; color:var(--text-muted); margin:0 0 8px;">expected {expected}</p>
  <span class="verdict-chip {cls}">{status}</span>
</div>"""


def _diff_section(ds_key: str, d: dict) -> str:
    def _pair_row(p: dict, prefix: str) -> str:
        return f"""
<div class="corr-row">
  <span class="corr-pair">{esc(p['feature_a'])}<span class="arrow">↔</span>{esc(p['feature_b'])}</span>
  <span class="corr-method">{esc(prefix)}</span>
  <span></span>
  <span class="corr-value">{p.get('value', p.get('after_value', ''))}</span>
  <span class="verdict-chip drop">{esc(p.get('tier', p.get('after_tier', '')).upper())}</span>
</div>"""

    rows = ""
    for p in d["added_pairs"]:
        rows += _pair_row(p, "newly appeared")
    for p in d["removed_pairs"]:
        rows += _pair_row(p, "no longer present")
    for p in d["tier_changed_pairs"]:
        rows += f"""
<div class="corr-row">
  <span class="corr-pair">{esc(p['feature_a'])}<span class="arrow">↔</span>{esc(p['feature_b'])}</span>
  <span class="corr-method">tier changed</span>
  <span></span>
  <span class="corr-value">{p['before_value']} → {p['after_value']}</span>
  <span class="verdict-chip drop">{esc(p['before_tier'].upper())} → {esc(p['after_tier'].upper())}</span>
</div>"""

    is_empty = d["n_added"] == 0 and d["n_removed"] == 0 and d["n_tier_changed"] == 0
    body = rows if not is_empty else '<div class="corr-row"><span class="corr-pair">No changes -- diff is empty for this dataset.</span></div>'

    return f"""
<div class="vif-dataset-block">
  <h2 class="dataset-title">{esc(ds_key.title())}</h2>
  <p class="dataset-substat">{d['n_pairs_before']} DROP/COLLAPSE/REVIEW pairs before &rarr; {d['n_pairs_after']} after &middot;
  {d['n_added']} added &middot; {d['n_removed']} removed &middot; {d['n_tier_changed']} tier changes</p>
  <div class="corr-ledger">{body}</div>
</div>"""


def _manifest(changes: list[dict]) -> str:
    rows = "".join(
        f"""
<div class="stratum-card" style="margin-bottom:10px;">
  <h5>{esc(c['change'])}</h5>
  <p class="stratum-delta" style="margin-bottom:4px;">{esc(c['detail'])}</p>
  <p class="stratum-delta"><b>Feature-count impact:</b> {esc(c['feature_count_impact'])}</p>
</div>"""
        for c in changes
    )
    return f'<div class="stratum-grid" style="grid-template-columns:1fr;">{rows}</div>'


def build_report(data: dict) -> str:
    counts = data["confirmed_counts"]
    verdict_class = "v-no" if not data["any_newly_risky_pairs_found"] else "v-yes"

    body = f"""
<div class="verdict-banner {verdict_class}" style="margin-bottom:28px;">
<b>{'LOCKED -- no new tier crossings' if not data['any_newly_risky_pairs_found'] else 'ATTENTION -- new tier crossing(s) found'}</b>
&mdash; {esc(data['verdict'])}
</div>

<h2>Confirmed candidate feature counts</h2>
<p class="section-note">Read directly from feature_config.py's ACTIVE/PASSIVE dicts, not assumed.</p>
<div class="before-after">
{_count_card('Active', counts['active'], counts['active_expected'], counts['active_matches_expected'])}
{_count_card('Passive', counts['passive'], counts['passive_expected'], counts['passive_matches_expected'])}
</div>

<h2 style="margin-top:40px;">Correlation diff vs. the last confirmed run</h2>
<p class="section-note">DROP/COLLAPSE/REVIEW tiers only (exhaustive lists) -- the DISTINCT tier's top-10-by-magnitude
sample is excluded from this diff since sampling noise there isn't a real tier change.
{'Diff is empty, as expected.' if data['diff_is_empty'] else 'Diff is NOT empty -- see below.'}</p>
{_diff_section('active', data['correlation_diff']['active'])}
{_diff_section('passive', data['correlation_diff']['passive'])}

<h2 style="margin-top:40px;">Manifest: every change since the last full correlation run</h2>
<p class="section-note">So this report is legible on its own -- proof the locked feature set is still internally
consistent, without needing the full session history.</p>
{_manifest(data['changes_since_last_full_run'])}
"""

    return render.render_article(
        eyebrow="FEATURE LOCK CONFIRMATION",
        title="Post-Lock Correlation Confirmation",
        dek=(
            "Not another COLLAPSE/REVIEW redundancy round -- that work is closed. A confirmation pass: after "
            "every landed feature-list change this session, does the locked candidate set still hold up under "
            "a fresh correlation re-run?"
        ),
        stats=[
            (str(counts["active"]), "active features"),
            (str(counts["passive"]), "passive features"),
            ("0" if data["diff_is_empty"] else "!", "tier changes found"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS + render.VIF_CSS + render.CONFOUND_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
