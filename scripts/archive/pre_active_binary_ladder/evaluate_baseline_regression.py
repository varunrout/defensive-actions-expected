"""Compatibility wrapper for regression baseline validation.

Archived (prompt 48): superseded by the active-binary model ladder
(scripts/models/train_active_binary_baseline.py and the v1b-v1e rung
scripts). Nothing else in the repo imports or invokes this file
(confirmed before archiving). Kept, not deleted, per standing project
policy.
"""

from __future__ import annotations

from pathlib import Path

from dax.models.validation import evaluate_regression_models

# One directory level deeper than before the prompt-48 archive move
# (scripts/models/ -> scripts/archive/pre_active_binary_ladder/), so
# parents[2] -> parents[3] to keep resolving to the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    evaluate_regression_models(REPO_ROOT / "outputs" / "validation" / "regression", REPO_ROOT / "outputs" / "oof" / "regression")
