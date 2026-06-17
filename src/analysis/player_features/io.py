"""Data loading and schema checks for player-feature analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "match_id",
    "event_id",
    "player_id",
    "player",
    "position_group",
    "phase_label",
    "action_family",
    "target_future_shot_10s",
}

NUMERIC_CANDIDATES = [
    "distance_to_attacking_goal",
    "distance_to_center_line",
    "local_numerical_balance_5m",
    "local_numerical_balance_10m",
    "attackers_within_5m",
    "attackers_within_10m",
    "defenders_within_5m",
    "defenders_within_10m",
    "nearest_attacker_distance",
    "nearest_defender_distance",
    "attacker_spread",
    "defender_spread",
    "attacker_defender_ratio",
    "possession_elapsed_seconds",
    "phase_transitions_observed_so_far",
    "action_x",
    "action_y",
]

CATEGORICAL_CANDIDATES = [
    "phase_label",
    "position_group",
    "action_family",
    "event_type",
    "action_zone",
    "play_pattern",
]


def load_player_features(path: Path) -> pd.DataFrame:
    """Load and normalize player feature data."""
    df = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["target_future_shot_10s"] = pd.to_numeric(df["target_future_shot_10s"], errors="coerce").fillna(0).astype(int)
    df["target_future_shot_10s"] = (df["target_future_shot_10s"] > 0).astype(int)

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

