"""Diagnostic tables and charts for model validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def decile_table(df: pd.DataFrame, prediction_col: str, target_col: str) -> pd.DataFrame:
    out = df[[prediction_col, target_col]].copy()
    out["decile"] = pd.qcut(out[prediction_col].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    return (
        out.groupby("decile", dropna=False)
        .agg(rows=(target_col, "size"), mean_prediction=(prediction_col, "mean"), mean_observed=(target_col, "mean"))
        .reset_index()
    )


def subgroup_metrics(df: pd.DataFrame, prediction_col: str, target_col: str, group_cols: Iterable[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for col in group_cols:
        if col not in df.columns:
            continue
        table = (
            df.groupby(col, dropna=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "rows": len(g),
                        "mean_prediction": g[prediction_col].mean(),
                        "mean_observed": g[target_col].mean(),
                        "bias": (g[prediction_col] - g[target_col]).mean(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
            .rename(columns={col: "group_value"})
        )
        table.insert(0, "group", col)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_classification_charts(df: pd.DataFrame, chart_dir: str | Path, *, target_col: str = "y_true", score_col: str = "y_score") -> list[Path]:
    directory = ensure_dir(chart_dir)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(df[target_col], df[score_col], ax=ax)
    ax.set_title("Precision-recall curve")
    paths.append(directory / "precision_recall_curve.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    if df[target_col].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        RocCurveDisplay.from_predictions(df[target_col], df[score_col], ax=ax)
        ax.set_title("ROC curve")
        paths.append(directory / "roc_curve.png")
        fig.savefig(paths[-1], bbox_inches="tight")
        plt.close(fig)

    prob_true, prob_pred = calibration_curve(df[target_col], df[score_col], n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration curve")
    paths.append(directory / "calibration_curve.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[score_col], bins=20)
    ax.set_xlabel("Predicted probability")
    ax.set_title("Prediction distribution")
    paths.append(directory / "prediction_distribution.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    return paths


def save_regression_charts(df: pd.DataFrame, chart_dir: str | Path, *, target_col: str = "y_true", prediction_col: str = "y_pred") -> list[Path]:
    directory = ensure_dir(chart_dir)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[prediction_col], df[target_col], alpha=0.5)
    ax.set_xlabel("Predicted future xG")
    ax.set_ylabel("Observed future xG")
    ax.set_title("Predicted versus observed")
    paths.append(directory / "predicted_vs_observed.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    residual = df[prediction_col] - df[target_col]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=20)
    ax.set_xlabel("Residual")
    ax.set_title("Residual distribution")
    paths.append(directory / "residual_distribution.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[prediction_col], residual, alpha=0.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted future xG")
    ax.set_ylabel("Residual")
    ax.set_title("Residual versus prediction")
    paths.append(directory / "residual_vs_prediction.png")
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    return paths
