"""Leakage-safe calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def calibration_bin_table(y_true, y_score, *, n_bins: int = 10) -> pd.DataFrame:
    """Return calibration bins for OOF predictions."""

    y = np.asarray(y_true)
    p = np.asarray(y_score)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        mask = (p >= lower) & (p < (upper if upper < 1 else upper + 1e-12))
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "rows": int(mask.sum()),
                "mean_prediction": float(p[mask].mean()) if mask.any() else np.nan,
                "observed_rate": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fit_calibrated_classifier(base_estimator, x_train, y_train, *, method: str, seed: int = 42):
    """Fit fold-internal Platt or isotonic calibration on training data only."""

    if method == "uncalibrated":
        model = base_estimator.fit(x_train, y_train)
        return model
    if method not in {"platt", "isotonic"}:
        raise ValueError(f"Unsupported calibration method: {method}")
    positives = int(np.sum(y_train))
    negatives = int(len(y_train) - positives)
    if positives < 3 or negatives < 3:
        raise ValueError(f"Insufficient training support for {method} calibration: positives={positives}, negatives={negatives}")
    inner_splits = min(3, positives, negatives)
    cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    sklearn_method = "sigmoid" if method == "platt" else "isotonic"
    return CalibratedClassifierCV(base_estimator, method=sklearn_method, cv=cv).fit(x_train, y_train)
