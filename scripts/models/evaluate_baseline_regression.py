"""Compatibility wrapper for regression baseline validation."""

from __future__ import annotations

from pathlib import Path

from dax.models.validation import evaluate_regression_models

REPO_ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    evaluate_regression_models(REPO_ROOT / "outputs" / "validation" / "regression", REPO_ROOT / "outputs" / "oof" / "regression")
