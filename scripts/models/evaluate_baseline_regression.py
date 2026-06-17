"""Evaluate DAx baseline regression models and generate plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline regression model artifacts")
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "validation" / "regression"),
        help="Validation artifacts directory.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="regression_model_metrics.json",
        help="Summary metrics json filename.",
    )
    parser.add_argument(
        "--oof-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "oof" / "regression"),
        help="Out-of-fold artifacts directory.",
    )
    parser.add_argument(
        "--variant-for-coef",
        type=str,
        default="v2_full_baseline",
        help="Variant used for coefficient plot.",
    )
    return parser.parse_args()


def _metrics_table(summary: dict) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, meta in summary.get("variants", {}).items():
        rows.append(
            {
                "variant": variant,
                "r2": float(meta.get("r2", 0.0)),
                "rmse": float(meta.get("rmse", 0.0)),
                "mae": float(meta.get("mae", 0.0)),
                "mape": float(meta.get("mape", 0.0)),
                "spearman": float(meta.get("spearman", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    specs = [
        ("r2", "R2 (higher better)"),
        ("rmse", "RMSE (lower better)"),
        ("mae", "MAE (lower better)"),
        ("spearman", "Spearman (higher better)"),
    ]

    for idx, (metric, label) in enumerate(specs):
        ax = axes[idx]
        order = metrics_df.sort_values(metric, ascending=(metric in {"rmse", "mae"}))
        ax.bar(order["variant"], order[metric], color="steelblue")
        ax.set_title(label)
        ax.set_xlabel("Variant")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Baseline Regression Metrics by Variant", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _plot_pred_vs_actual(oof_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    variants = sorted(oof_df["variant"].unique())

    for variant in variants:
        part = oof_df[oof_df["variant"] == variant].dropna(subset=["y_true", "y_pred"])
        if part.empty:
            continue
        sample = part.sample(n=min(6000, len(part)), random_state=42) if len(part) > 6000 else part
        ax.scatter(sample["y_true"], sample["y_pred"], s=6, alpha=0.25, label=variant)

    lo = float(min(oof_df["y_true"].min(), oof_df["y_pred"].min()))
    hi = float(max(oof_df["y_true"].max(), oof_df["y_pred"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")
    ax.set_title("Predicted vs Actual Future xG")
    ax.set_xlabel("Actual future xG")
    ax.set_ylabel("Predicted future xG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _plot_residuals(oof_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    variants = sorted(oof_df["variant"].unique())

    for variant in variants:
        part = oof_df[oof_df["variant"] == variant].dropna(subset=["y_true", "y_pred"])
        if part.empty:
            continue
        residual = part["y_true"] - part["y_pred"]
        sample = residual.sample(n=min(8000, len(residual)), random_state=42) if len(residual) > 8000 else residual
        axes[0].hist(sample, bins=40, alpha=0.4, density=True, label=variant)

        scat = part
        if len(scat) > 6000:
            scat = scat.sample(n=6000, random_state=42)
        axes[1].scatter(scat["y_pred"], scat["y_true"] - scat["y_pred"], s=6, alpha=0.2, label=variant)

    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("y_true - y_pred")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].axhline(0.0, linestyle="--", linewidth=1, color="gray")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted future xG")
    axes[1].set_ylabel("Residual")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _plot_coefficients(validation_dir: Path, variant: str, out_path: Path) -> bool:
    coef_path = validation_dir / f"regression_{variant}_coefficients.csv"
    if not coef_path.exists():
        return False

    coef_df = pd.read_csv(coef_path)
    if coef_df.empty or "abs_coef" not in coef_df.columns:
        return False

    top = coef_df.nlargest(20, "abs_coef").sort_values("abs_coef", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["coef"], color="steelblue")
    ax.set_title(f"Top 20 Regression Coefficients ({variant})")
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
        out_path = validation_dir / f"regression_{variant}_feature_importance.png"
        success = _plot_coefficients(validation_dir, variant, out_path)
        results.append((str(out_path), success))
    return results


def main() -> None:
    args = parse_args()
    validation_dir = Path(args.validation_dir)
    summary_path = validation_dir / args.summary
    oof_dir = Path(args.oof_dir)
    oof_path = oof_dir / "regression_oof_predictions.parquet"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions parquet: {oof_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    oof_df = pd.read_parquet(oof_path)
    if oof_df.empty:
        raise ValueError("OOF predictions are empty.")

    metrics_df = _metrics_table(summary)
    if metrics_df.empty:
        raise ValueError("No regression variants found in summary JSON.")

    metrics_csv = validation_dir / "regression_model_metrics_table.csv"
    metrics_df.sort_values("r2", ascending=False).to_csv(metrics_csv, index=False)

    metrics_plot = validation_dir / "regression_metrics_by_variant.png"
    pred_plot = validation_dir / "regression_predicted_vs_actual.png"
    residual_plot = validation_dir / "regression_residuals.png"

    _plot_metrics(metrics_df, metrics_plot)
    _plot_pred_vs_actual(oof_df, pred_plot)
    _plot_residuals(oof_df, residual_plot)
    
    # Generate coefficient plots for v2, v3, and v4
    coef_results = _plot_all_coefficients_comparison(validation_dir)

    print("=" * 72)
    print("EVALUATE BASELINE REGRESSION")
    print("=" * 72)
    print(metrics_df.sort_values("r2", ascending=False).to_string(index=False))
    print("-" * 72)
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {metrics_plot}")
    print(f"Saved: {pred_plot}")
    print(f"Saved: {residual_plot}")
    for path, success in coef_results:
        if success:
            print(f"Saved: {path}")
        else:
            print(f"Skipped: {path} (missing coefficients)")


if __name__ == "__main__":
    main()

