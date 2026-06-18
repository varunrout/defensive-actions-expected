"""Simple sklearn-compatible reference baseline estimators."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class ConstantClassifier(BaseEstimator, ClassifierMixin):
    """Predict the training-set positive class rate for every row."""

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        self.p_ = float(y_array.mean()) if len(y_array) else 0.0
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "p_")
        n_rows = len(X)
        return np.column_stack([np.full(n_rows, 1.0 - self.p_), np.full(n_rows, self.p_)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """Predict the training-set mean or median for every row."""

    def __init__(self, stat: str = "mean"):
        self.stat = stat

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float)
        if self.stat not in {"mean", "median"}:
            raise ValueError("stat must be 'mean' or 'median'")
        self.value_ = float(np.median(y_array) if self.stat == "median" else np.mean(y_array)) if len(y_array) else 0.0
        return self

    def predict(self, X):
        check_is_fitted(self, "value_")
        return np.full(len(X), self.value_)
