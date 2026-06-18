"""Evaluation metrics for classification and regression models."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Compute expected calibration error over fixed-width probability bins."""

    y = np.asarray(y_true)
    p = np.asarray(y_score)
    bins = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (p >= lower) & (p < (upper if upper < 1 else upper + 1e-12))
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def calibration_slope_intercept(y_true, y_score) -> tuple[float, float]:
    """Fit logistic calibration slope/intercept on predictions."""

    y = np.asarray(y_true)
    p = np.clip(np.asarray(y_score), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p))
    model = LogisticRegression().fit(logit.reshape(-1, 1), y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def classification_metrics(y_true, y_score) -> dict[str, float]:
    """Return primary classification metrics; accuracy is intentionally omitted."""

    y = np.asarray(y_true)
    p = np.clip(np.asarray(y_score), 1e-6, 1 - 1e-6)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "average_precision": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "expected_calibration_error": expected_calibration_error(y, p),
        "positive_rate": float(np.mean(y)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return primary regression metrics, including zero/non-zero xG slices."""

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    nonzero = y > 0
    zero = ~nonzero
    spearman = spearmanr(y, p).statistic if len(y) > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "spearman": float(spearman) if not np.isnan(spearman) else float("nan"),
        "mean_prediction": float(p.mean()),
        "mean_observed": float(y.mean()),
        "prediction_bias": float((p - y).mean()),
        "zero_target_mae": float(mean_absolute_error(y[zero], p[zero])) if zero.any() else float("nan"),
        "nonzero_target_mae": float(mean_absolute_error(y[nonzero], p[nonzero])) if nonzero.any() else float("nan"),
        "nonzero_target_rmse": float(mean_squared_error(y[nonzero], p[nonzero]) ** 0.5) if nonzero.any() else float("nan"),
    }
