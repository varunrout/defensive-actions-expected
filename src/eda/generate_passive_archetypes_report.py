"""CLI entrypoint: render PASSIVE_ARCHETYPES.json as a self-contained HTML
report, in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.generate_passive_archetypes_report
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "PASSIVE_ARCHETYPES.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "PASSIVE_ARCHETYPES.html"

BUCKET_LABELS = {
    "last_line": "Last Line",
    "wide_cover": "Wide Cover",
    "central_screen": "Central Screen",
    "advanced_wide": "Advanced Wide",
    "mid_block": "Mid Block",
}

# One two-colour pair for cluster 0 / cluster 1, reused across every bucket --
# tints of the canonical --neg (blue) and --amber, siblings of the palette
# rather than a new independent hue pair.
CLUSTER_COLOR_VARS = ["var(--neg)", "var(--amber)"]

PASSIVE_CSS = """
.bucket-section { margin-bottom: 48px; }
.bucket-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; border-top: 2px solid var(--border); padding-top: 14px; margin-bottom: 18px; }
.bucket-title { font-family: "Archivo", sans-serif; font-weight: 800; font-size: 22px; margin: 0 0 4px; }
.bucket-desc { color: var(--text-muted); font-size: 13px; margin: 0; max-width: 60ch; }
.bucket-stats { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-secondary); background: var(--plane); border: 1px solid var(--border); border-radius: 8px; padding: 8px 14px; white-space: nowrap; }
.bucket-stats div { margin-bottom: 2px; }
.bucket-stats div:last-child { margin-bottom: 0; }

.cluster-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px; }
@media (max-width: 760px) { .cluster-grid { grid-template-columns: 1fr; } }
.cluster-card { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--cluster-color, var(--accent)); border-radius: 0 10px 10px 0; padding: 16px 18px; }
.cc-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; margin-bottom: 14px; }
.cc-archetype { font-family: "Archivo", sans-serif; font-weight: 700; font-size: 15px; color: var(--cluster-color, var(--text-primary)); margin: 0 0 3px; }
.cc-n { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--text-muted); }
.cc-rate { text-align: right; white-space: nowrap; }
.cc-rate b { font-family: "Archivo", sans-serif; font-weight: 700; font-size: 18px; display: block; color: var(--text-primary); }
.cc-rate .delta { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); }

.bar-row { display: grid; grid-template-columns: 120px 1fr 56px; align-items: center; gap: 10px; padding: 5px 0; }
.bar-label { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { position: relative; height: 8px; background: var(--gridline); border-radius: 4px; }
.bar-track::before { content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: var(--text-muted); opacity: 0.5; }
.bar-fill { position: absolute; top: 0; height: 100%; border-radius: 4px; }
.bar-fill.pos { left: 50%; }
.bar-fill.neg { right: 50%; }
.bar-value { font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: var(--text-muted); text-align: right; }
"""


def _cluster_color(idx: int) -> str:
    return CLUSTER_COLOR_VARS[idx % len(CLUSTER_COLOR_VARS)]


def _bar_row(feature: str, diff: float, max_abs: float, own_color: str, other_color: str) -> str:
    pct = min(50.0, abs(diff) / max_abs * 50) if max_abs else 0.0
    side = "pos" if diff > 0 else "neg"
    color = own_color if side == "pos" else other_color
    label = feature.replace("_", " ")
    style = f"width:{pct:.2f}%; background:{color};"
    return f"""
<div class="bar-row">
  <div class="bar-label" title="{esc(label)}">{esc(label)}</div>
  <div class="bar-track"><div class="bar-fill {side}" style="{style}"></div></div>
  <div class="bar-value">{diff:+.2f}</div>
</div>"""


def _cluster_card(cluster: dict, idx: int, n_clusters: int) -> str:
    own_color = _cluster_color(idx)
    other_color = _cluster_color(idx + 1) if n_clusters > 1 else own_color
    max_abs = max((abs(d["standardised_diff"]) for d in cluster["top_distinguishing_features"]), default=1.0) or 1.0
    rows = "".join(_bar_row(d["feature"], d["standardised_diff"], max_abs, own_color, other_color) for d in cluster["top_distinguishing_features"])
    delta = cluster["shot_rate_delta_vs_bucket_pp"]
    delta_str = f"{delta:+.2f}pp vs bucket" if delta else "at bucket average"

    return f"""
<div class="cluster-card" style="--cluster-color:{own_color};">
  <div class="cc-head">
    <div>
      <p class="cc-archetype">{esc(cluster['name'])}</p>
      <p class="cc-n">n={cluster['n']:,} &middot; {cluster['share_of_bucket_sample']}% of sample</p>
    </div>
    <div class="cc-rate">
      <b>{cluster['shot_rate_pct']}%</b>
      <span class="delta">{esc(delta_str)}</span>
    </div>
  </div>
  <div class="bars">{rows}</div>
</div>"""


def _k_eval_table(evaluation: list[dict], selected_k: int) -> str:
    rows = ""
    for row in evaluation:
        is_selected = int(row["k"]) == selected_k
        rows += f"""
<tr{' style="background:var(--good-wash);"' if is_selected else ''}>
  <td>{int(row['k'])}{' &larr; selected' if is_selected else ''}</td>
  <td>{row['silhouette']:.3f}</td>
  <td>{row['calinski_harabasz']:.1f}</td>
  <td>{row['davies_bouldin']:.3f}</td>
  <td>{row['size_balance']:.3f}</td>
  <td>{row['selection_score']:.3f}</td>
</tr>"""
    return f"""
<div class="table-scroll"><table class="evidence-ledger">
<tr><th>k</th><th>Silhouette</th><th>Calinski-Harabasz</th><th>Davies-Bouldin</th><th>Size balance</th><th>Selection score</th></tr>
{rows}
</table></div>"""


def _bucket_section(bucket_key: str, b: dict) -> str:
    label = BUCKET_LABELS.get(bucket_key, bucket_key)
    n_clusters = len(b["clusters"])
    cluster_cards = "".join(_cluster_card(c, i, n_clusters) for i, c in enumerate(b["clusters"]))

    stats = f"""
<div>n = {b['n_rows_total']:,} in bucket</div>
<div>sampled {b['n_rows_sampled']:,} (seed 42)</div>
<div>bucket mean shot rate {b['bucket_mean_shot_rate_pct']}%</div>
<div>selected k = {b['selected_k']}</div>"""

    return f"""
<div class="bucket-section">
  <div class="bucket-head">
    <div>
      <h3 class="bucket-title">{esc(label)}</h3>
      <p class="bucket-desc">Archetypes clustered within this functional-role bucket -- standardised difference from the bucket population mean.</p>
    </div>
    <div class="bucket-stats">{stats}</div>
  </div>
  <h4 style="font-family:'Archivo',sans-serif; font-size:14px; margin:0 0 8px;">k selection (silhouette / Calinski-Harabasz / inverse Davies-Bouldin / size balance, percentile-rank average)</h4>
  {_k_eval_table(b['k_evaluation'], b['selected_k'])}
  <div class="cluster-grid">{cluster_cards}</div>
</div>"""


def build_report(data: dict) -> str:
    body = f"""
<div class="finding flag" style="margin-bottom:24px;">
<span class="tag">sampling limitation</span>
<p>{esc(data['sampling_limitation'])}</p>
</div>
<div class="finding" style="margin-bottom:32px;">
<span class="tag">method</span>
<p>{esc(data['method_note'])}</p>
</div>
"""
    for bucket_key in data["buckets_clustered"]:
        body += _bucket_section(bucket_key, data["buckets"][bucket_key])

    body += f"""
<div class="closing-note" style="margin-top:40px;">
<b>Non-causal framing:</b> {esc(data['non_causal_note'])}
</div>"""

    return render.render_article(
        eyebrow="PASSIVE ARCHETYPES",
        title="Passive-Defence Archetype Clustering (Phase 8)",
        dek=(
            "Clustering within each defender_functional_role bucket -- one row per defender-slot, no player "
            "identity, ever, by design. Clustering asks what varies within a role, not the role itself."
        ),
        stats=[
            (f"{data['n_rows_total']:,}", "rows (passive)"),
            (str(len(data["buckets_clustered"])), "buckets clustered"),
            (str(len(data["feature_columns"])), "clustering features"),
        ],
        body_html=body,
        extra_css=render.CONFOUND_CSS + render.CORR_CSS + PASSIVE_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
