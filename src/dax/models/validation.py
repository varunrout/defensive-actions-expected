from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate supported DAx baseline model artifacts.")
    parser.add_argument("--task", choices=("all", "logistic", "regression"), default="all")
    parser.add_argument("--validation-dir", type=str, default=str(REPO_ROOT / "outputs" / "validation"))
    parser.add_argument("--oof-dir", type=str, default=str(REPO_ROOT / "outputs" / "oof"))
    parser.add_argument("--dry-run", action="store_true", help="Print the planned validation task without running it.")
    return parser.parse_args(argv)


def _plot_logistic_coefficients(validation_dir: Path, variant: str, out_path: Path) -> bool:
    coefficients_path = validation_dir / f"baseline_{variant}_coefficients.csv"
    if not coefficients_path.exists():
        return False
    coefficients = pd.read_csv(coefficients_path)
    if coefficients.empty or "abs_coef" not in coefficients.columns:
        return False
    top = coefficients.nlargest(20, "abs_coef").sort_values("abs_coef", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["coef"], color="steelblue")
    ax.set_title(f"Top 20 Logistic Coefficients ({variant})")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def evaluate_logistic_models(validation_dir: str | Path, oof_dir: str | Path, summary_name: str = "baseline_model_metrics.json") -> Path:
    validation_path = Path(validation_dir)
    summary_path = validation_path / summary_name
    oof_path = Path(oof_dir) / "baseline_oof_predictions.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions parquet: {oof_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    oof = pd.read_parquet(oof_path)
    if oof.empty:
        raise ValueError("OOF predictions are empty.")

    metrics_rows: list[dict[str, float | str]] = []
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for variant in summary.get("variants", {}):
        part = oof[oof["variant"] == variant].dropna(subset=["y_score"])
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

    roc_path = validation_path / "baseline_roc_curves.png"
    pr_path = validation_path / "baseline_pr_curves.png"
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    fig_pr.savefig(pr_path, dpi=150)
    plt.close(fig_roc)
    plt.close(fig_pr)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    metrics_csv = validation_path / "baseline_model_metrics_table.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    for variant in [
        "v2_full_baseline",
        "v3_context_enhanced",
        "v4_freeze_geometry",
        "v5_interpretable_clustered",
        "v6_balanced_clustered",
        "v7_interpretable_ridge",
        "v8_balanced_ridge",
    ]:
        _plot_logistic_coefficients(validation_path, variant, validation_path / f"baseline_{variant}_feature_importance.png")

    print("=" * 72)
    print("EVALUATE BASELINE LOGISTIC")
    print("=" * 72)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    print(f"Saved: {metrics_csv}")
    return metrics_csv


def _metrics_table(summary: dict) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, meta in summary.get("variants", {}).items():
        rows.append(
            {
                "variant": variant,
                "r2": float(meta.get("r2", 0.0)),
                "rmse": float(meta.get("rmse", 0.0)),
                "mae": float(meta.get("mae", 0.0)),
                "spearman": float(meta.get("spearman", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _plot_regression_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    specs = [("r2", "R2 (higher better)"), ("rmse", "RMSE (lower better)"), ("mae", "MAE (lower better)"), ("spearman", "Spearman (higher better)")]
    for index, (metric, label) in enumerate(specs):
        ax = axes[index]
        order = metrics_df.sort_values(metric, ascending=(metric in {"rmse", "mae"}))
        ax.bar(order["variant"], order[metric], color="steelblue")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Baseline Regression Metrics by Variant", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_regression_pred_vs_actual(oof_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for variant in sorted(oof_df["variant"].unique()):
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
    plt.close(fig)


def _plot_regression_residuals(oof_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for variant in sorted(oof_df["variant"].unique()):
        part = oof_df[oof_df["variant"] == variant].dropna(subset=["y_true", "y_pred"])
        if part.empty:
            continue
        residual = part["y_true"] - part["y_pred"]
        sample = residual.sample(n=min(8000, len(residual)), random_state=42) if len(residual) > 8000 else residual
        axes[0].hist(sample, bins=40, alpha=0.4, density=True, label=variant)
        scat = part.sample(n=6000, random_state=42) if len(part) > 6000 else part
        axes[1].scatter(scat["y_pred"], scat["y_true"] - scat["y_pred"], s=6, alpha=0.2, label=variant)
    axes[0].set_title("Residual Distribution")
    axes[0].legend()
    axes[1].axhline(0.0, linestyle="--", linewidth=1, color="gray")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_regression_coefficients(validation_dir: Path, variant: str, out_path: Path) -> bool:
    coefficients_path = validation_dir / f"regression_{variant}_coefficients.csv"
    if not coefficients_path.exists():
        return False
    coefficients = pd.read_csv(coefficients_path)
    if coefficients.empty or "abs_coef" not in coefficients.columns:
        return False
    top = coefficients.nlargest(20, "abs_coef").sort_values("abs_coef", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["coef"], color="steelblue")
    ax.set_title(f"Top 20 Regression Coefficients ({variant})")
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def evaluate_regression_models(validation_dir: str | Path, oof_dir: str | Path, summary_name: str = "regression_model_metrics.json") -> Path:
    validation_path = Path(validation_dir)
    summary_path = validation_path / summary_name
    oof_path = Path(oof_dir) / "regression_oof_predictions.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_path}")
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions parquet: {oof_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    oof = pd.read_parquet(oof_path)
    if oof.empty:
        raise ValueError("OOF predictions are empty.")

    metrics_df = _metrics_table(summary)
    if metrics_df.empty:
        raise ValueError("No regression variants found in summary JSON.")

    metrics_csv = validation_path / "regression_model_metrics_table.csv"
    metrics_df.sort_values("r2", ascending=False).to_csv(metrics_csv, index=False)
    _plot_regression_metrics(metrics_df, validation_path / "regression_metrics_by_variant.png")
    _plot_regression_pred_vs_actual(oof, validation_path / "regression_predicted_vs_actual.png")
    _plot_regression_residuals(oof, validation_path / "regression_residuals.png")

    for variant in [
        "v2_full_baseline",
        "v3_context_enhanced",
        "v4_freeze_geometry",
        "v5_interpretable_clustered",
        "v6_balanced_clustered",
        "v7_interpretable_ridge",
        "v8_balanced_ridge",
    ]:
        _plot_regression_coefficients(validation_path, variant, validation_path / f"regression_{variant}_feature_importance.png")

    print("=" * 72)
    print("EVALUATE BASELINE REGRESSION")
    print("=" * 72)
    print(metrics_df.sort_values("r2", ascending=False).to_string(index=False))
    print(f"Saved: {metrics_csv}")
    return metrics_csv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] validate task={args.task}")
        return 0
    if args.task in {"all", "logistic"}:
        evaluate_logistic_models(Path(args.validation_dir) / "baseline", Path(args.oof_dir) / "baseline")
    if args.task in {"all", "regression"}:
        evaluate_regression_models(Path(args.validation_dir) / "regression", Path(args.oof_dir) / "regression")
    return 0
