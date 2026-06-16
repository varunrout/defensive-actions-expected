"""Backward-compatible imports for legacy callers.

Use analysis.player_features.io for new code.
"""

from .io import CATEGORICAL_CANDIDATES, NUMERIC_CANDIDATES, REQUIRED_COLUMNS, load_player_features

__all__ = [
    "REQUIRED_COLUMNS",
    "NUMERIC_CANDIDATES",
    "CATEGORICAL_CANDIDATES",
    "load_player_features",
]


