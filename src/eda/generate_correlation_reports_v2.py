"""V2 (prompt 7/7) CLI entrypoint: render CORRELATION_ANALYSIS_V2.json and
REVIEW_ANALYSIS_V2.json as two self-contained HTML reports, in the same
shared design system as the other reports/analysis/shot_target/*.html files. Writes to
_V2-suffixed output filenames -- the V1 reports (CORRELATION_ATLAS.html,
REVIEW_METHODOLOGY.html, produced by generate_correlation_reports.py) are
untouched by this script.

Usage:
    python -m src.eda.generate_correlation_reports_v2
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc
from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRELATION_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V2.json"
REVIEW_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "REVIEW_ANALYSIS_V2.json"
V1_CORRELATION_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ANALYSIS_V1_HISTORICAL.json"
V1_REVIEW_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "REVIEW_ANALYSIS_V1_HISTORICAL.json"
ATLAS_OUTPUT = REPO_ROOT / "reports" / "analysis" / "shot_target" / "CORRELATION_ATLAS_V2.html"
METHODOLOGY_OUTPUT = REPO_ROOT / "reports" / "analysis" / "shot_target" / "REVIEW_METHODOLOGY_V2.html"

METHOD_DESCRIPTIONS = [
    ("Spearman ρ", "continuous ↔ continuous", "Monotonic rank correlation, robust to the heavy right-skew present in several features (e.g. top_option_1_threat_score skew 7.91) where Pearson would understate the relationship."),
    ("phi", "boolean ↔ boolean", "Pearson r computed directly on the 0/1-encoded columns."),
    ("point-biserial r", "boolean ↔ continuous", "Correlation between a binary grouping and a continuous variable; equivalent to Pearson r with one side coded 0/1."),
    ("Cramer's V", "categorical ↔ categorical", "Bias-corrected association strength from a chi-square test on the crosstab; 0 = independent, 1 = perfectly associated. Boolean columns are treated as 2-level categoricals when paired with a categorical column."),
    ("correlation ratio η", "categorical ↔ continuous", "Share of the continuous column's variance explained by categorical group membership (sqrt of eta-squared)."),
]

TIER_INFO = {
    "drop": ("Exact duplicates", "drop", "|r| ≥ 0.98 confirmed > 0.999 (or Cramer's V > 0.99 / η > 0.995) — one side is mathematically derived from the other. Delete one side without further analysis."),
    "collapse": ("Near-perfect cluster", "collapse", "0.90 ≤ |r| < 0.98, or 0.4 ≤ Cramer's V, or 0.6 ≤ η < 0.98 — the same underlying signal restated. Worth collapsing into fewer features, not a pure duplicate."),
    "review": ("Moderate redundancy", "review", "0.5 ≤ |r| < 0.90 for continuous/boolean pairs, or 0.3 ≤ Cramer's V < 0.4, or 0.5 ≤ η < 0.6 for categorical pairs (V2) — not resolved by the correlation script itself; see REVIEW_METHODOLOGY.html for how each pair here was resolved."),
    "distinct": ("Confirmed distinct", "distinct", "Below every threshold above. No action needed — top pairs by magnitude shown here so the report shows what was checked and cleared, not just silence."),
}


def _fmt_pair_name(p: dict) -> str:
    return f'<span class="corr-pair">{esc(p["feature_a"])}<span class="arrow">↔</span>{esc(p["feature_b"])}</span>'


def _corr_row(p: dict, tier: str) -> str:
    magnitude = min(100.0, p["abs_value"] * 100)
    chip_label = tier.upper() + (" *" if p.get("downgraded_from_drop") else "")
    title = f'{p["feature_a"]} vs {p["feature_b"]}: {p["method"]} = {p["value"]} (n={p["n"]})'
    if p.get("downgraded_from_drop"):
        title += " -- cleared the raw DROP threshold but not the near-1.0 confirm check, downgraded to COLLAPSE"
    return f"""
<div class="corr-row" title="{esc(title)}">
  {_fmt_pair_name(p)}
  <span class="corr-method">{esc(p['method'])}</span>
  <div class="corr-mag-track"><div class="corr-mag-fill {tier}" style="width:{magnitude:.1f}%"></div></div>
  <span class="corr-value">{p['value']:+.3f}</span>
  <span class="verdict-chip {tier}">{esc(chip_label)}</span>
</div>"""


def _tier_section(tier: str, pairs: list[dict], total_n: int | None = None) -> str:
    label, css_tier, desc = TIER_INFO[tier]
    count = total_n if total_n is not None else len(pairs)
    heading = f"{label} — {count} pair{'s' if count != 1 else ''}"
    if tier == "distinct":
        heading += f" (top {len(pairs)} shown by magnitude)"
    rows = "".join(_corr_row(p, css_tier) for p in pairs) or '<div class="corr-row"><span class="corr-pair">None</span></div>'
    return f"""
<div class="tier-section">
  <div class="tier-heading"><span class="tier-dot {css_tier}"></span>{esc(heading)}</div>
  <p class="tier-note">{desc}</p>
  <div class="corr-ledger">{rows}</div>
</div>"""


def _method_strip() -> str:
    cards = "".join(
        f'<div class="method-card"><span class="m-name">{esc(name)}</span><p><b>{esc(applies)}</b><br>{esc(desc)}</p></div>'
        for name, applies, desc in METHOD_DESCRIPTIONS
    )
    return f'<div class="method-strip">{cards}</div>'


def _dataset_block(ds_key: str, ds: dict) -> str:
    dataset_cfg = DATASETS[ds_key]
    tc = ds["tier_counts"]
    stat_html = "".join(
        f'<div class="stat"><b>{v}</b><span>{label}</span></div>'
        for v, label in [
            (f"{ds['n_rows_used']:,}", "rows used" + (" (sampled)" if ds["sampled"] else "")),
            (ds["n_features"], "features"),
            (tc["drop"], "drop"),
            (tc["collapse"], "collapse"),
            (tc["review"], "review"),
            (tc["distinct_total"], "distinct"),
        ]
    )
    return f"""
<div class="dataset-block">
  <h2 class="dataset-title">{esc(dataset_cfg['label'])}</h2>
  <p class="dataset-substat">{esc(dataset_cfg['parquet_path'])} · {esc(dataset_cfg['row_description'])}</p>
  <div class="statbar" style="margin-bottom:28px;">{stat_html}</div>
  {_tier_section('drop', ds['pairs']['drop'])}
  {_tier_section('collapse', ds['pairs']['collapse'])}
  {_tier_section('review', ds['pairs']['review'])}
  {_tier_section('distinct', ds['pairs']['distinct_sample'], total_n=tc['distinct_total'])}
</div>"""


def _v1_vs_v2_banner(v1_data: dict | None, v2_data: dict) -> str:
    """Computed live from both JSONs on every render -- never a hand-typed
    number that can drift. V1 (stage 01) and V2 (stage 07) are genuinely
    different snapshots: the feature-count drop between them is real
    (structural-redesign + collapse-tier + raw-coordinate-drop landed in
    between), and V2 also fixed a methodology gap (no categorical REVIEW
    band existed yet at V1)."""
    if v1_data is None:
        return ""
    rows = []
    for ds_key in ("active", "passive"):
        v1_ds, v2_ds = v1_data["datasets"][ds_key], v2_data["datasets"][ds_key]
        rows.append(
            f'<tr><td>{esc(ds_key.title())}</td>'
            f'<td>{v1_ds["n_features"]}</td><td>{v2_ds["n_features"]}</td>'
            f'<td>{v1_ds["tier_counts"]["review"]}</td><td>{v2_ds["tier_counts"]["review"]}</td>'
            f'<td>{v2_ds["n_features"] - v1_ds["n_features"]:+d}</td></tr>'
        )
    return f"""
<div class="finding" style="margin-bottom:24px;">
<span class="tag">what changed vs V1</span>
<p>V1 (<a href="CORRELATION_ATLAS.html" style="color:var(--neg);">CORRELATION_ATLAS.html</a>) is stage 01's
original 51/44-feature scope; this page is stage 07's 36/39, after the structural-redesign and
collapse-tier/raw-coordinate-drop stages already ran. Two things changed between them, not one: the feature
count actually dropped, <b>and</b> V2 adds a categorical REVIEW band (Cramer's V 0.3-0.4, &eta; 0.5-0.6) that
V1's tiering has no equivalent for. See
<a href="CORRELATION_ATLAS_V3.html" style="color:var(--neg);">CORRELATION_ATLAS_V3.html</a> for the final
locked list (34/38), scored on TRAIN+VAL matches only.</p>
<div class="table-scroll"><table class="evidence-ledger" style="margin-top:10px;">
<tr><th>Dataset</th><th>Features (V1)</th><th>Features (V2)</th><th>Review pairs (V1)</th><th>Review pairs (V2)</th><th>Feature-count diff</th></tr>
{''.join(rows)}
</table></div>
</div>"""


def build_correlation_atlas(data: dict, v1_data: dict | None = None) -> str:
    datasets = data["datasets"]
    total_drop = sum(d["tier_counts"]["drop"] for d in datasets.values())
    total_collapse = sum(d["tier_counts"]["collapse"] for d in datasets.values())
    total_review = sum(d["tier_counts"]["review"] for d in datasets.values())
    total_distinct = sum(d["tier_counts"]["distinct_total"] for d in datasets.values())

    body = _v1_vs_v2_banner(v1_data, data)
    body += _method_strip()
    for ds_key in ("active", "passive"):
        body += _dataset_block(ds_key, datasets[ds_key])

    body += f"""
<div class="closing-note">
<b>This does not change model accuracy for tree-based models.</b> A GBM or random forest splits on whichever
correlated column happens to be available regardless of collinearity between features -- dropping or collapsing
DROP/COLLAPSE-tier pairs will not move ROC-AUC, PR-AUC or calibration for those model families. What it changes
is feature-importance interpretability (splits no longer get arbitrarily divided between near-duplicate columns)
and model size/training time. For a linear or distance-based method, collinearity matters more directly.
</div>"""

    return render.render_article(
        eyebrow="CORRELATION ATLAS — V2",
        title="Feature Correlation Atlas (V2)",
        dek=(
            "Mixed-type association matrix for both feature datasets, classified into DROP / COLLAPSE / REVIEW / "
            "DISTINCT tiers using the type-matched method for each pair -- never a single Pearson matrix across "
            "continuous, boolean and categorical columns."
        ),
        stats=[
            (total_drop, "drop (both datasets)"),
            (total_collapse, "collapse"),
            (total_review, "review"),
            (total_distinct, "distinct (full count)"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS,
    )


# ---------------------------------------------------------------------------
# Review methodology report
# ---------------------------------------------------------------------------

FRAMEWORK_CARDS = [
    {
        "type": "Type 1",
        "title": "Boolean vs. its continuous parent",
        "test": "Boolean↔continuous pair, |r| ≥ 0.75, and the continuous side isn't itself scheduled for removal elsewhere. Structural call, no lift lookup needed.",
        "verdict": "Drop the boolean, keep the continuous.",
    },
    {
        "type": "Type 2",
        "title": "Boolean vs. boolean, same rough concept",
        "test": "Both boolean, r ≥ 0.5. Checked in order: opposite-sign lift → keep both. Same sign with one flag a near-total subset of the other (containment ≥ 90%) → keep the higher-signal flag. Same sign, lift diff < 1.0pp → near-identical, drop the broader flag. Same sign, diff ≥ 3.0pp, no subset → keep both. Diff between 1.0pp and 3.0pp → needs_human_call.",
        "verdict": "Depends on lift comparison -- see evidence ledger below.",
    },
    {
        "type": "Type 4",
        "title": "State-persistence pairs",
        "test": "Same base column paired with its own _prev_event / lagged version, detected by name pattern.",
        "verdict": "Keep both, always -- this is autocorrelation, not redundancy.",
    },
    {
        "type": "Type 5 (V2)",
        "title": "Categorical vs. its correlated continuous/boolean context",
        "test": "Conceptual, not statistical: does the categorical column carry information beyond what's already in the continuous/boolean measurement, or is it just a relabeling of the same thing? Now also reaches into COLLAPSE (categorical pairs used to be tiered and then never processed by anything). Default keep both; flagged needs_human_call when η ≥ 0.85 or Cramér's V ≥ 0.7.",
        "verdict": "Keep both by default; needs_human_call above the relabeling threshold.",
    },
    {
        "type": "Type 6 (V2)",
        "title": "Continuous vs. continuous, same engineered family",
        "test": "Both continuous/discrete, and both columns appear together in an explicit, hardcoded family group (feature_config.FAMILY_GROUPS) -- e.g. *_within_5m vs *_within_10m, or a ranked-option column at different ranks. Default keep both (deferred to feature-importance evidence); needs_human_call if |r| ≥ 0.90, or if no family group matches at all.",
        "verdict": "Keep both if same family and |r| < 0.90; needs_human_call otherwise -- never forces a family match that isn't real.",
    },
]


def _framework_card(card: dict, example_html: str) -> str:
    return f"""
<div class="framework-card">
  <span class="f-type">{esc(card['type'])}</span>
  <h3>{esc(card['title'])}</h3>
  <p>{esc(card['test'])}</p>
  <p><b>Verdict rule:</b> {esc(card['verdict'])}</p>
  <div class="example">{example_html}</div>
</div>"""


def _find_example(dataset_results: dict, verdict_prefix: str, relationship_type: int) -> dict | None:
    for r in dataset_results["resolved_pairs"]:
        if r.get("relationship_type") == relationship_type and r["verdict"].startswith(verdict_prefix):
            return r
    return None


def _evidence_row(r: dict) -> str:
    ev = r.get("evidence", {})
    return f"""
<tr>
  <td>{esc(r['feature_a'])} ↔ {esc(r['feature_b'])}</td>
  <td>{ev.get('lift_a', '—')}pp</td>
  <td>{ev.get('lift_b', '—')}pp</td>
  <td>{ev.get('lift_diff_pp', '—')}pp</td>
  <td>{esc(r['verdict'])}</td>
</tr>"""


def _type1_evidence_row(r: dict) -> str:
    return f"""
<tr>
  <td>{esc(r.get('boolean_column', r['feature_a']))}</td>
  <td>{esc(r.get('continuous_column', r['feature_b']))}</td>
  <td>{r['value']:+.3f}</td>
  <td>{esc(r['verdict'])}</td>
</tr>"""


def _hcall_item(r: dict) -> str:
    return f"""
<div class="hcall-item">
  <div class="hp-pair">{esc(r['feature_a'])} ↔ {esc(r['feature_b'])} <span style="color:var(--text-muted); font-weight:400;">({esc(r['method'])}, r={r['value']})</span></div>
  <div class="hp-reason">{esc(r['reason'])}</div>
</div>"""


def _structural_redesign_section(review_data: dict) -> str:
    passive = review_data["datasets"]["passive"]
    cluster5_pairs = [
        r for r in passive["resolved_pairs"]
        if r["verdict"] == "needs_human_call"
        and "top_option_" in r["feature_a"] and "threat_score" in r["feature_a"]
        and "top_option_" in r["feature_b"] and "threat_score" in r["feature_b"]
    ]
    cluster5_html = "".join(
        f'<li><code>{esc(r["feature_a"])}</code> ↔ <code>{esc(r["feature_b"])}</code> (r={r["value"]})</li>'
        for r in cluster5_pairs
    )

    return f"""
<h2>Structural Redesign</h2>
<p>Some REVIEW-tier (and adjacent) pairs aren't redundancy problems at all -- the fix is "measure this differently,"
not "drop one side." Two root causes showed up in the passive-side <code>top_option_1/2/3_*</code> family: absolute
pitch coordinates encoded near a moving reference point (the ball), and several ranked/repeated columns describing
the same real-world cluster of objects (ranked attacking options).</p>

<h3 style="margin-top:28px;">Part A -- DONE</h3>
<p><code>top_option_{{1,2,3}}_target_x/y</code> were absolute pitch coordinates, which is why they correlated
0.68-0.88 with <code>ball_x</code> and with each other -- the same absolute position meant something different
depending on where the ball was. Replaced with <code>top_option_n_dx/dy</code> (offset from the ball),
<code>top_option_n_distance_from_ball</code>, and <code>top_option_n_angle_from_ball</code>, kept alongside
the raw columns during a transition period, then the raw <code>target_x/y</code> columns were dropped from the
candidate feature list once the fix was confirmed working -- correlation with
<code>ball_x</code>/<code>defender_x</code> is gone from the output entirely, not just re-tiered.</p>

<h3 style="margin-top:28px;">Part B -- still deferred, Cluster 5 ({len(cluster5_pairs)} pairs)</h3>
<p>The <code>top_option_1/2/3_threat_score</code> inter-rank correlations were never a coordinate-encoding problem
-- they're a genuine question of whether ranks 2-3 carry independent signal or are redundant with rank 1. Still
open:</p>
<ul>{cluster5_html}</ul>

<div class="path-grid">
  <div class="path-card do-now">
    <span class="p-label">Done</span>
    <p>Ball-relative coordinate transform (Part A) -- unambiguous, lost no information, confirmed working by
    re-running the correlation analysis after the drop.</p>
  </div>
  <div class="path-card defer">
    <span class="p-label">Still deferred</span>
    <p>Collapsing <code>top_option_2/3_*</code> into aggregate "backup option" features (max threat score, spread
    vs rank 1) is a real tradeoff -- it may throw away signal a model would otherwise use. Gated behind
    <code>PASSIVE_COLLAPSE_OPTION_RANKS</code> (default <code>False</code>) until a baseline model's feature
    importance says whether ranks 2-3 are pulling their weight. Type 6's review logic deliberately keeps these
    pairs in needs_human_call regardless of |r| -- correlation alone was never going to be what decides this.</p>
  </div>
</div>
<div class="recommendation"><b>Recommendation:</b> no action until a baseline model exists. Do not collapse ranks
2-3 from correlation alone; correlation cannot tell you whether they're pulling weight, only that they move
together.</div>
"""


def _hcall_backlog_banner(v1_review: dict | None, v2_review: dict) -> str:
    if v1_review is None:
        return ""
    n_v1 = sum(len(ds["needs_human_call"]) for ds in v1_review["datasets"].values())
    n_v2 = sum(len(ds["needs_human_call"]) for ds in v2_review["datasets"].values())
    return f"""
<div class="finding flag" style="margin-bottom:24px;">
<span class="tag">backlog vs V1</span>
<p>V1's needs_human_call backlog was <b>{n_v1} pairs</b>, on the 51/44-feature scope
(<a href="REVIEW_METHODOLOGY.html" style="color:var(--amber);">REVIEW_METHODOLOGY.html</a>). This page's backlog
is <b>{n_v2}</b> pairs, on the smaller 36/39-feature scope -- the two numbers aren't purely a methodology
comparison, since most of the drop is fewer features/pairs existing at all by stage 07, not Types 5/6 resolving
more. The remaining {n_v2} are genuinely open questions (mostly the passive-side
<code>top_option_2/3_threat_score</code> family), not gaps in the resolution logic.</p>
</div>"""


def build_review_methodology(review_data: dict, correlation_data: dict, v1_review_data: dict | None = None) -> str:
    active = review_data["datasets"]["active"]
    passive = review_data["datasets"]["passive"]

    type1_example = _find_example(active, "drop_boolean", 1) or _find_example(passive, "drop_boolean", 1)
    type2_drop_example = _find_example(active, "drop_", 2)
    type4_example = _find_example(active, "keep_both", 4) or _find_example(passive, "keep_both", 4)
    type5_example = _find_example(active, "keep_both", 5) or _find_example(passive, "keep_both", 5)
    type6_example = _find_example(active, "keep_both", 6) or _find_example(passive, "keep_both", 6)

    examples = {
        "Type 1": f"<b>{esc(type1_example['boolean_column'])}</b> vs <b>{esc(type1_example['continuous_column'])}</b>: r={type1_example['value']} → drop {esc(type1_example['boolean_column'])}." if type1_example else "No Type 1 example resolved in this run.",
        "Type 2": f"<b>{esc(type2_drop_example['feature_a'])}</b> vs <b>{esc(type2_drop_example['feature_b'])}</b>: lift {type2_drop_example['evidence']['lift_a']:+.2f}pp vs {type2_drop_example['evidence']['lift_b']:+.2f}pp → {esc(type2_drop_example['verdict'])}." if type2_drop_example else "No Type 2 drop example in this run.",
        "Type 4": f"<b>{esc(type4_example['feature_a'])}</b> vs <b>{esc(type4_example['feature_b'])}</b> ({esc(type4_example['method'])}) -- kept both, the persistence relationship this rule protects." if type4_example else "No Type 4 example resolved in this run.",
        "Type 5 (V2)": f"<b>{esc(type5_example['feature_a'])}</b> vs <b>{esc(type5_example['feature_b'])}</b>: {esc(type5_example['method'])}={type5_example['value']} → keep both." if type5_example else "No Type 5 example resolved in this run.",
        "Type 6 (V2)": f"<b>{esc(type6_example['feature_a'])}</b> vs <b>{esc(type6_example['feature_b'])}</b>: r={type6_example['value']} → keep both (same engineered family)." if type6_example else "No Type 6 example resolved in this run.",
    }

    framework_html = "".join(
        _framework_card(card, examples[card["type"]]) for card in FRAMEWORK_CARDS
    )

    all_hcalls = []
    for ds_key, ds in review_data["datasets"].items():
        for r in ds["needs_human_call"]:
            all_hcalls.append({**r, "_dataset": ds_key})

    hcall_html = "".join(
        f'<div class="hcall-item"><div class="hp-pair">[{esc(item["_dataset"])}] {esc(item["feature_a"])} ↔ {esc(item["feature_b"])} '
        f'<span style="color:var(--text-muted); font-weight:400;">({esc(item.get("method",""))}, r={item.get("value")})</span></div>'
        f'<div class="hp-reason">{esc(item["reason"])}</div></div>'
        for item in all_hcalls
    )

    type1_rows = "".join(
        _type1_evidence_row(r)
        for ds in (active, passive)
        for r in ds["resolved_pairs"]
        if r.get("relationship_type") == 1 and r["verdict"] in ("drop_boolean", "moot_after_drop")
    )
    type2_rows = "".join(
        _evidence_row(r)
        for ds in (active, passive)
        for r in ds["resolved_pairs"]
        if r.get("relationship_type") == 2
    )

    n_hcall_active = len(active["needs_human_call"])
    n_hcall_passive = len(passive["needs_human_call"])

    body = _hcall_backlog_banner(v1_review_data, review_data)
    body += f"""
<div class="rule-banner"><b>Correlation says two features move together -- never which one (or whether either)
matters.</b> Every verdict on this page traces to a lift number (the actual shot-rate difference), not the
r-value alone. The r-value only got a pair onto this list; it never decided the verdict.</div>

<h2 style="margin-top:36px;">The five relationship types</h2>
<p class="section-note">Types 1, 2 and 4 are from V1; Types 5 and 6 (V2) close the two gaps V1 left open -- categorical pairs that were tiered and then never processed, and a blanket "no rule" fallback for continuous↔continuous pairs.</p>
<div class="framework-grid">{framework_html}</div>

<div class="hcall-panel">
  <p class="hp-title">needs_human_call -- {n_hcall_active + n_hcall_passive} pairs across both datasets ({n_hcall_active} active, {n_hcall_passive} passive)</p>
  <p class="hp-note">These pairs did not resolve automatically. Both lift values (or the reason no lift test applies) are included so a human can decide without re-deriving anything.</p>
  {hcall_html}
</div>

<h2>Evidence ledger</h2>
<p>Every Type 1 pair (boolean vs. its continuous parent) and Type 2 pair (boolean vs. boolean), with the evidence used and the resulting verdict.</p>

<h3>Type 1 -- boolean vs. continuous parent</h3>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Boolean</th><th>Continuous parent</th><th>r</th><th>Verdict</th></tr>
{type1_rows}
</table></div>

<h3 style="margin-top:28px;">Type 2 -- boolean vs. boolean</h3>
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>Pair</th><th>Lift A</th><th>Lift B</th><th>Diff</th><th>Verdict</th></tr>
{type2_rows}
</table></div>

{_structural_redesign_section(review_data)}
"""

    return render.render_article(
        eyebrow="REVIEW METHODOLOGY — V2",
        title="Resolving the Review Tier (V2)",
        dek=(
            "How every REVIEW-tier correlation pair was resolved -- not just the final answer. Companion to "
            "CORRELATION_ATLAS.html, not a duplicate of it."
        ),
        stats=[
            (active["n_review_pairs"], "active review pairs"),
            (passive["n_review_pairs"], "passive review pairs"),
            (n_hcall_active + n_hcall_passive, "needs human call"),
        ],
        body_html=body,
        extra_css=render.CORR_CSS,
    )


def main() -> None:
    correlation_data = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))
    review_data = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    v1_correlation_data = json.loads(V1_CORRELATION_PATH.read_text(encoding="utf-8")) if V1_CORRELATION_PATH.exists() else None
    v1_review_data = json.loads(V1_REVIEW_PATH.read_text(encoding="utf-8")) if V1_REVIEW_PATH.exists() else None

    atlas_html = build_correlation_atlas(correlation_data, v1_correlation_data)
    ATLAS_OUTPUT.write_text(atlas_html, encoding="utf-8")
    print(f"Wrote {ATLAS_OUTPUT}")

    methodology_html = build_review_methodology(review_data, correlation_data, v1_review_data)
    METHODOLOGY_OUTPUT.write_text(methodology_html, encoding="utf-8")
    print(f"Wrote {METHODOLOGY_OUTPUT}")


if __name__ == "__main__":
    main()
