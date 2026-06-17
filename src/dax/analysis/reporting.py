from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supported DAx validation reports.")
    parser.add_argument(
        "--report",
        choices=("validation-summary",),
        default="validation-summary",
        help="Currently supported canonical report kind.",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "baseline" / "baseline_model_metrics.json"),
    )
    parser.add_argument(
        "--regression-metrics",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "regression" / "regression_model_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "reports" / "VALIDATION_SUMMARY.md"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the report path without writing files.")
    return parser.parse_args(argv)


def _best_variant(metrics: dict, primary: str, secondary: str) -> tuple[str, dict]:
    variants = metrics.get("variants", {})
    if not variants:
        raise ValueError("Metrics JSON contains no variants.")
    best_name = max(variants, key=lambda name: (variants[name].get(primary, float("-inf")), variants[name].get(secondary, float("-inf"))))
    return best_name, variants[best_name]


def generate_validation_summary_report(
    baseline_metrics_path: str | Path,
    regression_metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    baseline_path = Path(baseline_metrics_path)
    regression_path = Path(regression_metrics_path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline metrics: {baseline_path}")
    if not regression_path.exists():
        raise FileNotFoundError(f"Missing regression metrics: {regression_path}")

    baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
    regression_metrics = json.loads(regression_path.read_text(encoding="utf-8"))
    logistic_name, logistic = _best_variant(baseline_metrics, "roc_auc", "avg_precision")
    regression_name, regression = _best_variant(regression_metrics, "r2", "spearman")

    lines = [
        "# Validation Summary",
        "",
        "## Canonical report coverage",
        "",
        "- `scripts/generate_reports.py` currently supports the validation summary report.",
        "- Specialized portfolio and defensibility reports remain available via the active `scripts/analysis/` entry points until consolidation reaches parity.",
        "",
        "## Logistic baseline",
        "",
        f"- Rows: {baseline_metrics['rows']:,}",
        f"- Matches: {baseline_metrics['matches']:,}",
        f"- Target rate: {baseline_metrics['target_rate']:.4f}",
        f"- Best variant: `{logistic_name}`",
        f"- ROC AUC: {logistic['roc_auc']:.4f}",
        f"- Average precision: {logistic['avg_precision']:.4f}",
        "",
        "## Regression baseline",
        "",
        f"- Rows: {regression_metrics['rows']:,}",
        f"- Matches: {regression_metrics['matches']:,}",
        f"- Target mean: {regression_metrics['target_mean']:.4f}",
        f"- Best variant: `{regression_name}`",
        f"- R²: {regression['r2']:.4f}",
        f"- Spearman: {regression['spearman']:.4f}",
        "",
        "## Source artifacts",
        "",
        f"- `{baseline_path}`",
        f"- `{regression_path}`",
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved validation summary report: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] generate {args.report} -> {args.output}")
        return 0
    generate_validation_summary_report(args.baseline_metrics, args.regression_metrics, args.output)
    return 0
