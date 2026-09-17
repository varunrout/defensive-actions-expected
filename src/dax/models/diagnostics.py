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

# Shared visual system for every diagnostic chart this module produces: fixed
# figure size and DPI (so PNGs are pixel-identical in size regardless of
# legend/label length), one accent color, light gridlines, no top/right spines.
_FIGSIZE = (6.4, 4.6)
_DPI = 140
_ACCENT = "#2563eb"
_ACCENT_DARK = "#1e3a8a"
_MUTED = "#94a3b8"
_GRID_COLOR = "#e2e8f0"
_FACE_COLOR = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": _GRID_COLOR,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "figure.facecolor": _FACE_COLOR,
        "axes.facecolor": _FACE_COLOR,
        "savefig.facecolor": _FACE_COLOR,
    }
)


def _new_axes():
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def _save_fixed(fig, path: Path) -> None:
    # No bbox_inches="tight": that crops to content and makes PNG dimensions
    # vary per-plot (legend width, tick labels, ...). A fixed layout keeps
    # every chart in a variant folder -- and across variants -- the same size.
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


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

    fig, ax = _new_axes()
    PrecisionRecallDisplay.from_predictions(
        df[target_col], df[score_col], ax=ax, curve_kwargs={"color": _ACCENT, "linewidth": 2.2}
    )
    ax.set_title("Precision-recall curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    paths.append(directory / "precision_recall_curve.png")
    _save_fixed(fig, paths[-1])

    if df[target_col].nunique() > 1:
        fig, ax = _new_axes()
        RocCurveDisplay.from_predictions(
            df[target_col], df[score_col], ax=ax, curve_kwargs={"color": _ACCENT, "linewidth": 2.2}
        )
        ax.plot([0, 1], [0, 1], color=_MUTED, linewidth=1.2, linestyle="--", label="Chance")
        ax.legend(loc="lower right")
        ax.set_title("ROC curve")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        paths.append(directory / "roc_curve.png")
        _save_fixed(fig, paths[-1])

    prob_true, prob_pred = calibration_curve(df[target_col], df[score_col], n_bins=10, strategy="quantile")
    fig, ax = _new_axes()
    ax.plot([0, 1], [0, 1], color=_MUTED, linewidth=1.2, linestyle="--", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", markersize=5, color=_ACCENT, linewidth=2.2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    paths.append(directory / "calibration_curve.png")
    _save_fixed(fig, paths[-1])

    fig, ax = _new_axes()
    ax.hist(df[score_col], bins=20, color=_ACCENT, edgecolor=_ACCENT_DARK, linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction distribution")
    ax.set_xlim(-0.02, 1.02)
    paths.append(directory / "prediction_distribution.png")
    _save_fixed(fig, paths[-1])

    return paths


def save_regression_charts(df: pd.DataFrame, chart_dir: str | Path, *, target_col: str = "y_true", prediction_col: str = "y_pred") -> list[Path]:
    directory = ensure_dir(chart_dir)
    paths: list[Path] = []

    fig, ax = _new_axes()
    ax.scatter(df[prediction_col], df[target_col], alpha=0.45, s=16, color=_ACCENT, edgecolor="none")
    lo = float(min(df[prediction_col].min(), df[target_col].min()))
    hi = float(max(df[prediction_col].max(), df[target_col].max()))
    ax.plot([lo, hi], [lo, hi], color=_MUTED, linewidth=1.2, linestyle="--", label="y = x")
    ax.legend(loc="upper left")
    ax.set_xlabel("Predicted future xG")
    ax.set_ylabel("Observed future xG")
    ax.set_title("Predicted versus observed")
    paths.append(directory / "predicted_vs_observed.png")
    _save_fixed(fig, paths[-1])

    residual = df[prediction_col] - df[target_col]
    fig, ax = _new_axes()
    ax.hist(residual, bins=20, color=_ACCENT, edgecolor=_ACCENT_DARK, linewidth=0.6, alpha=0.85)
    ax.axvline(0, color=_MUTED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    paths.append(directory / "residual_distribution.png")
    _save_fixed(fig, paths[-1])

    fig, ax = _new_axes()
    ax.scatter(df[prediction_col], residual, alpha=0.45, s=16, color=_ACCENT, edgecolor="none")
    ax.axhline(0, color=_MUTED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Predicted future xG")
    ax.set_ylabel("Residual")
    ax.set_title("Residual versus prediction")
    paths.append(directory / "residual_vs_prediction.png")
    _save_fixed(fig, paths[-1])

    return paths
