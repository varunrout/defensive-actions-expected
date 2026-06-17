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
    "target_shot_in_10s",
}

NUMERIC_CANDIDATES = [
    "nearest_goal_distance",
    "distance_to_center_line",
    "local_numerical_balance_5m",
    "local_numerical_balance_10m",
    "attackers_within_5m",
    "attackers_within_10m",
    "freeze_teammate_nearest_distance",
    "freeze_opponent_nearest_distance",
    "freeze_teammate_spread",
    "freeze_opponent_spread",
    "teammate_count",
    "opponent_count",
    "teammate_opponent_ratio",
    "possession_progress_ratio",
    "phase_transition_count_so_far",
    "seconds_since_possession_start",
    "possession_duration_total",
    "possession_event_count_total",
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

    df["target_shot_in_10s"] = pd.to_numeric(df["target_shot_in_10s"], errors="coerce").fillna(0).astype(int)
    df["target_shot_in_10s"] = (df["target_shot_in_10s"] > 0).astype(int)

    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

