"""Evaluate DAx baseline logistic models and generate plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline logistic model artifacts")
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "validation" / "baseline"),
        help="Validation artifacts directory.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="baseline_model_metrics.json",
        help="Summary metrics json filename.",
    )
    parser.add_argument(
        "--oof-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "oof" / "baseline"),
        help="Out-of-fold artifacts directory.",
    )
    return parser.parse_args()


def _plot_coefficients(validation_dir: Path, variant: str, out_path: Path) -> bool:
    """Generate coefficient plot for a single variant."""
    coef_path = validation_dir / f"baseline_{variant}_coefficients.csv"
    if not coef_path.exists():
        return False

    coef_df = pd.read_csv(coef_path)
    if coef_df.empty or "abs_coef" not in coef_df.columns:
        return False

    top = coef_df.nlargest(20, "abs_coef").sort_values("abs_coef", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["coef"], color="steelblue")
    ax.set_title(f"Top 20 Logistic Coefficients ({variant})")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return True


def _plot_all_coefficients_comparison(validation_dir: Path) -> list[tuple[str, bool]]:
    """Generate coefficient plots for v2-v8."""
    results: list[tuple[str, bool]] = []
    for variant in [
        "v2_full_baseline",
        "v3_context_enhanced",
        "v4_freeze_geometry",
        "v5_interpretable_clustered",
        "v6_balanced_clustered",
        "v7_interpretable_ridge",
        "v8_balanced_ridge",
    ]:
        out_path = validation_dir / f"baseline_{variant}_feature_importance.png"
        success = _plot_coefficients(validation_dir, variant, out_path)
        results.append((str(out_path), success))
    return results


def _plot_metrics_comparison(validation_dir: Path, metrics_df: pd.DataFrame) -> Path:
    """Generate metrics comparison chart for AUC and AP across variants."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sort by AUC for consistency
    order = metrics_df.sort_values("roc_auc", ascending=False)
    
    axes[0].bar(order["variant"], order["roc_auc"], color="steelblue")
    axes[0].set_title("ROC AUC by Variant")
    axes[0].set_ylabel("ROC AUC")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylim([0, 1])
    
    axes[1].bar(order["variant"], order["avg_precision"], color="coral")
    axes[1].set_title("Average Precision by Variant")
    axes[1].set_ylabel("AP")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim([0, 1])
    
    fig.suptitle("Baseline Logistic Metrics by Variant", fontsize=12)
    fig.tight_layout()
    
    metrics_plot = validation_dir / "baseline_metrics_by_variant.png"
    fig.savefig(metrics_plot, dpi=150)
    return metrics_plot


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.validation_dir)
    summary_path = validation_dir / args.summary
    oof_dir = Path(args.oof_dir)
    oof_path = oof_dir / "baseline_oof_predictions.parquet"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions parquet: {oof_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    oof_df = pd.read_parquet(oof_path)
    if oof_df.empty:
        raise ValueError("OOF predictions are empty.")

    metrics_rows: list[dict[str, float | str]] = []

    # ROC curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    # PR curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for variant, vmeta in summary.get("variants", {}).items():
        part = oof_df[oof_df["variant"] == variant].dropna(subset=["y_score"])
        if part.empty:
            continue

        y_true = part["y_true"].astype(int)
        y_score = part["y_score"].astype(float)

        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))

        metrics_rows.append({"variant": variant, "roc_auc": auc, "avg_precision": ap})

        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax_roc, name=f"{variant} (AUC={auc:.3f})")
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax_pr, name=f"{variant} (AP={ap:.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax_roc.set_title("Baseline Logistic ROC Curves")
    ax_pr.set_title("Baseline Logistic Precision-Recall Curves")

    roc_path = validation_dir / "baseline_roc_curves.png"
    pr_path = validation_dir / "baseline_pr_curves.png"
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    fig_pr.savefig(pr_path, dpi=150)

    # Metrics dataframe for later use
    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    metrics_csv = validation_dir / "baseline_model_metrics_table.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    # Generate metrics comparison bar chart
    metrics_plot_path = _plot_metrics_comparison(validation_dir, metrics_df)
    
    # Generate coefficient plots for v2, v3, and v4
    coef_results = _plot_all_coefficients_comparison(validation_dir)

    print("=" * 72)
    print("EVALUATE BASELINE LOGISTIC")
    print("=" * 72)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    print("-" * 72)
    print(f"Saved: {roc_path}")
    print(f"Saved: {pr_path}")
    print(f"Saved: {metrics_plot_path}")
    for path, success in coef_results:
        if success:
            print(f"Saved: {path}")
        else:
            print(f"Skipped: {path} (missing coefficients)")
    print(f"Saved: {metrics_csv}")


if __name__ == "__main__":
    main()
