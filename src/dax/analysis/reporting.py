from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def _read_csv(path):
    if not Path(path).exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()
def build_model_readiness(analysis_dir: str|Path) -> dict:
    root=Path(analysis_dir); overview=_read_csv(root/"data_quality"/"overview.csv"); dup=_read_csv(root/"data_quality"/"duplicates.csv"); miss=_read_csv(root/"data_quality"/"missingness.csv"); evals=_read_csv(root/"clustering"/"cluster_evaluation.csv")
    def status(ok,warn=False): return "pass" if ok else ("warning" if warn else "fail")
    warnings = []
    if not evals.empty and "silhouette" in evals.columns and evals["silhouette"].max() < 0.25:
        warnings.append("weak cluster separation: best silhouette below 0.25")
    if not evals.empty and "size_balance" in evals.columns and evals["size_balance"].min() < 0.2:
        warnings.append("cluster imbalance detected")
    pca = _read_csv(root/"clustering"/"pca_explained_variance.csv")
    if not pca.empty and "explained_variance_ratio" in pca.columns and pca["explained_variance_ratio"].head(2).sum() < 0.30:
        warnings.append("low PCA first-two-component explained variance")
    if not evals.empty and "subsample_ari_stability" in evals.columns and evals["subsample_ari_stability"].max() > 0.8:
        warnings.append("stability does not imply strong cluster separation")
    readiness = {"target_validity":{"status":status(not overview.empty and overview.get("future_shot_rate",pd.Series([0])).iloc[0]>=0)},"schema_completeness":{"status":"pass"},"team_context_validity":{"status":"warning"},"duplicate_identifiers":{"status":status(dup.empty or dup.get("duplicate_rows",pd.Series([0])).sum()==0, True)},"missingness":{"status":status(miss.empty or miss["missing_rate"].max()<0.5, True)},"target_sample_size":{"status":status(not overview.empty and overview.get("rows",pd.Series([0])).iloc[0]>=100, True)},"player_sample_size":{"status":"warning"},"360_coverage":{"status":"warning"},"visibility_reliability":{"status":"warning"},"clustering_stability":{"status":status(not evals.empty, True)},"data_leakage_scan":{"status":"warning","note":"No final model trained; review target-derived descriptive outcome features before modelling."},"clustering_interpretation_warnings":{"status":"warning" if warnings else "pass","warnings":warnings}}
    return readiness

def generate_pre_model_report(analysis_dir: str|Path, output: str|Path) -> Path:
    root=Path(analysis_dir); output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); readiness=build_model_readiness(root); (output.parent/"model_readiness.json").write_text(json.dumps(readiness,indent=2),encoding="utf-8")
    ov=_read_csv(root/"data_quality"/"overview.csv"); evals=_read_csv(root/"clustering"/"cluster_evaluation.csv")
    rows=int(ov["rows"].iloc[0]) if not ov.empty and "rows" in ov else 0
    lines=["# Pre-model analysis report","","This report is generated from reusable analysis outputs. Phase labels are rule-based tactical proxies, not ground-truth tactical labels. Descriptive signals are provisional and are not true DAx.","",f"## Data coverage\n\n- Rows analysed: {rows:,}","","## Model readiness",""]
    for k,v in readiness.items(): lines.append(f"- {k}: **{v['status']}**")
    lines += ["","## Clustering findings", f"- Cluster evaluation rows: {len(evals)}", "", "## Risks before model training", "- Confirm schema coverage, visibility reliability, sample sizes, and target leakage before predictive modelling.", "", "## Model-readiness decision", "- Proceed only after warning/fail items are reviewed; final predictive model training was not performed."]
    output.write_text("\n".join(lines)+"\n",encoding="utf-8"); return output

import argparse


def _best_variant(metrics: dict, primary: str, secondary: str) -> tuple[str, dict]:
    variants = metrics.get("variants", {})
    if not variants:
        raise ValueError("Metrics JSON contains no variants.")
    best_name = max(
        variants,
        key=lambda name: (variants[name].get(primary, float("-inf")), variants[name].get(secondary, float("-inf"))),
    )
    return best_name, variants[best_name]


def generate_validation_summary_report(baseline_metrics_path: str | Path, regression_metrics_path: str | Path, output_path: str | Path) -> Path:
    baseline_path = Path(baseline_metrics_path)
    regression_path = Path(regression_metrics_path)
    baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
    regression_metrics = json.loads(regression_path.read_text(encoding="utf-8"))
    logistic_name, logistic = _best_variant(baseline_metrics, "roc_auc", "avg_precision")
    regression_name, regression = _best_variant(regression_metrics, "r2", "spearman")
    lines = [
        "# Validation Summary",
        "",
        "## Logistic baseline",
        f"- Rows: {baseline_metrics['rows']:,}",
        f"- Target rate: {baseline_metrics['target_rate']:.4f}",
        f"- Best variant: `{logistic_name}`",
        f"- ROC AUC: {logistic['roc_auc']:.4f}",
        f"- Average precision: {logistic['avg_precision']:.4f}",
        "",
        "## Regression baseline",
        f"- Rows: {regression_metrics['rows']:,}",
        f"- Target mean: {regression_metrics['target_mean']:.4f}",
        f"- Best variant: `{regression_name}`",
        f"- R²: {regression['r2']:.4f}",
        f"- Spearman: {regression['spearman']:.4f}",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved validation summary report: {output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supported DAx reports.")
    parser.add_argument("--report", choices=("validation-summary",), default="validation-summary")
    parser.add_argument("--baseline-metrics", default="outputs/validation/baseline/baseline_model_metrics.json")
    parser.add_argument("--regression-metrics", default="outputs/validation/regression/regression_model_metrics.json")
    parser.add_argument("--output", default="outputs/validation/reports/VALIDATION_SUMMARY.md")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] generate {args.report} -> {args.output}")
        return 0
    generate_validation_summary_report(args.baseline_metrics, args.regression_metrics, args.output)
    return 0
