"""Compatibility wrapper for regression baseline validation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dax.models.validation import evaluate_regression_models


if __name__ == "__main__":
    evaluate_regression_models(REPO_ROOT / "outputs" / "validation" / "regression", REPO_ROOT / "outputs" / "oof" / "regression")
