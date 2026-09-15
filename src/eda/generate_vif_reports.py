"""CLI entrypoint: render VIF_ANALYSIS.json as a self-contained HTML report,
in the same shared design system as CORRELATION_ATLAS.html.

Usage:
    python -m src.eda.generate_vif_reports
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eda import render
from src.eda.render import esc
from src.eda.feature_config import DATASETS

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "reports" / "eda" / "VIF_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "VIF_ANALYSIS.html"

THRESHOLD_LOW = 5
THRESHOLD_HIGH = 10


def _vif_severity(v: float | None) -> str:
    if v is None:
        return "severe"
    if v >= THRESHOLD_HIGH:
        return "severe"
    if v >= THRESHOLD_LOW:
        return "warn"
    return ""


def _vif_chart(vif_list: list[dict], max_display: float | None = None, show_legend: bool = True) -> str:
    finite_vals = [v["vif"] for v in vif_list if v["vif"] is not None]
    max_val = max_display or (max(finite_vals + [THRESHOLD_HIGH]) * 1.1 if finite_vals else THRESHOLD_HIGH * 1.5)

    def pct(v: float) -> float:
        return min(100.0, v / max_val * 100)

    rows = []
    for v in vif_list:
        sev = _vif_severity(v["vif"])
        cls = f"vif-fill {sev}".strip()
        val_str = f"{v['vif']:.2f}" if v["vif"] is not None else "∞"
        width = pct(v["vif"]) if v["vif"] is not None else 100.0
        rows.append(f"""
<div class="vif-row">
  <span class="vif-label" title="{esc(v['feature'])}">{esc(v['feature'])}</span>
  <div class="vif-track">
    <div class="{cls}" style="width:{width:.1f}%"></div>
  </div>
  <span class="vif-value">{val_str}</span>
</div>""")

    low_pct = pct(THRESHOLD_LOW)
    high_pct = pct(THRESHOLD_HIGH)
    legend = """
<div class="vif-legend">
  <span><span class="dot" style="background:var(--neg)"></span> &lt; 5 (fine)</span>
  <span><span class="dot" style="background:var(--amber)"></span> 5–10 (moderate)</span>
  <span><span class="dot" style="background:var(--pos)"></span> ≥ 10 (severe, or unbounded)</span>
</div>""" if show_legend else ""
    return f"""{legend}
<div class="vif-chart">
  <div class="vif-threshold" style="left:calc(190px + {low_pct:.1f}% * (100% - 260px) / 100)"></div>
  <div class="vif-threshold" style="left:calc(190px + {high_pct:.1f}% * (100% - 260px) / 100)"></div>
  {''.join(rows)}
</div>"""


def _dataset_section(ds_key: str, ds: dict) -> str:
    dataset_cfg = DATASETS[ds_key]
    cond = ds["condition_number"]
    cond_str = "∞ (SVD did not converge)" if not ds["svd_converged"] else f"{cond:.3e}"

    constant_note = ""
    if ds["constant_columns_excluded"]:
        items = "".join(f"<li><code>{esc(c['feature'])}</code> — {esc(c['reason'])}</li>" for c in ds["constant_columns_excluded"])
        constant_note = f"""
<div class="finding flag" style="margin-bottom:20px;">
<span class="tag">data artefact</span>
<p><b>{len(ds['constant_columns_excluded'])} column(s) excluded as constant</b> after listwise deletion (VIF
requires complete cases across every candidate column at once) — not truly constant in the full dataset:</p>
<ul>{items}</ul>
</div>"""

    return f"""
<div class="vif-dataset-block">
  <h2 class="dataset-title">{esc(dataset_cfg['label'])}</h2>
  <p class="dataset-substat">{esc(dataset_cfg['parquet_path'])}</p>
  <div class="statbar" style="margin-bottom:20px;">
    <div class="stat"><b>{ds['n_rows_used']:,}</b><span>complete-case rows</span></div>
    <div class="stat"><b>{ds['n_features_used_in_vif']}</b><span>features in VIF</span></div>
    <div class="stat"><b>{cond_str}</b><span>correlation matrix condition #</span></div>
  </div>
  <div class="finding" style="margin-bottom:20px;">
  <span class="tag">limitation</span>
  <p>{esc(ds['categorical_excluded_caveat'])}</p>
  </div>
  {constant_note}
  {_vif_chart(ds['vif'])}
</div>"""


def _before_after_section(active_ds: dict) -> str:
    ba = active_ds["before_after"]
    n_unbounded_before = sum(1 for v in ba["before"]["vif"] if v["vif"] is None)
    before_max = (
        f"∞ ({n_unbounded_before} columns unbounded)" if n_unbounded_before
        else f"{ba['before']['max_vif']:,.2f}"
    )
    after_max = f"{ba['after']['max_vif']:.2f}"

    # Sort so unbounded (None -> infinite) VIFs surface first -- they are
    # the worst case, not the best, and must not be hidden by a naive
    # descending-finite sort.
    before_top = sorted(ba["before"]["vif"], key=lambda v: (v["vif"] is not None, -(v["vif"] or 0)))[:8]
    after_top = sorted(ba["after"]["vif"], key=lambda v: (v["vif"] is not None, -(v["vif"] or 0)))[:8]

    return f"""
<h2 class="dataset-title" style="margin-top:56px;">Active dataset: before / after the drop</h2>
<p class="section-note">The only place this analysis changed the feature list. Dropped:
{', '.join(f'<code>{esc(c)}</code>' for c in ba['dropped_columns'])}.</p>
<div class="before-after">
  <div class="ba-card before">
    <h4>Before &mdash; max VIF {before_max}</h4>
    {_vif_chart(before_top, max_display=10, show_legend=False)}
  </div>
  <div class="ba-card after">
    <h4>After &mdash; max VIF {after_max}</h4>
    {_vif_chart(after_top, max_display=10, show_legend=False)}
  </div>
</div>
<div class="closing-note">
<b>Root cause:</b> {esc(ba['root_cause'])}
</div>"""


def build_vif_report(data: dict) -> str:
    datasets = data["datasets"]
    body = """
<div class="finding" style="margin-bottom:28px;">
<span class="tag">what VIF catches</span>
<p>A feature can look perfectly fine against every other feature pairwise (correlation well under any collapse
threshold) and still be redundant <b>jointly</b> — a linear combination of two or more other kept columns.
Pairwise correlation cannot see this; VIF, which measures how well each column is predicted by <em>all the others
at once</em>, can. VIF = 1 / (1 - R²) where R² comes from regressing that column on every other candidate
feature. VIF ≥ 10 (R² ≥ 0.9) is conventionally severe; this report also marks 5–10 as moderate.</p>
</div>
"""
    for ds_key in ("active", "passive"):
        body += _dataset_section(ds_key, datasets[ds_key])

    if "before_after" in datasets["active"]:
        body += _before_after_section(datasets["active"])

    return render.render_article(
        eyebrow="VIF ANALYSIS",
        title="Multicollinearity (VIF) Analysis",
        dek=(
            "Variance inflation factor per dataset, over continuous + discrete + boolean candidate features. "
            "Computed via the standard R²-based definition (statsmodels variance_inflation_factor / "
            "np.linalg.inv) — never np.linalg.pinv, which silently suppresses near-singular directions and "
            "reports deceptively low VIFs for a genuinely collinear cluster."
        ),
        stats=[
            (datasets["active"]["n_features_used_in_vif"], "active features (VIF)"),
            (datasets["passive"]["n_features_used_in_vif"], "passive features (VIF)"),
        ],
        body_html=body,
        extra_css=render.VIF_CSS,
    )


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    html = build_vif_report(data)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
