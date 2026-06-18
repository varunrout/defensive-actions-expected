"""Spatial summaries using StatsBomb-normalised 120x80 pitch coordinates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import coordinate_columns

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def add_pitch_zones(df: pd.DataFrame, bins_x: int = 6, bins_y: int = 4) -> pd.DataFrame:
    """Add grid pitch-zone columns using action coordinates.

    Coordinates use the project convention where the attacking goal is at ``(120, 40)``.
    """
    x_column, y_column = coordinate_columns(df)
    output = df.copy()
    if not x_column or not y_column:
        output["x_bin"] = pd.NA
        output["y_bin"] = pd.NA
        output["pitch_zone"] = "unknown"
        return output
    output["x_bin"] = pd.cut(output[x_column], np.linspace(0, PITCH_LENGTH, bins_x + 1), include_lowest=True, labels=False)
    output["y_bin"] = pd.cut(output[y_column], np.linspace(0, PITCH_WIDTH, bins_y + 1), include_lowest=True, labels=False)
    output["pitch_zone"] = output["x_bin"].astype("Int64").astype(str) + "_" + output["y_bin"].astype("Int64").astype(str)
    return output


def zone_summary(df: pd.DataFrame, bins_x: int = 6, bins_y: int = 4) -> pd.DataFrame:
    """Summarise action density and descriptive outcomes by pitch grid zone."""
    data = add_pitch_zones(df, bins_x=bins_x, bins_y=bins_y)
    aggregations = {
        "rows": ("event_id", "size"),
        "future_shot_rate": ("target_future_shot_10s", "mean"),
        "future_xg_mean": ("target_future_xg_10s", "mean"),
    }
    if "action_won_possession" in data.columns:
        aggregations["possession_win_rate"] = ("action_won_possession", "mean")
    return data.groupby(["x_bin", "y_bin", "pitch_zone"], dropna=False).agg(**aggregations).reset_index()


def player_spatial_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Return player-team action-location profiles."""
    x_column, y_column = coordinate_columns(df)
    if not x_column or not y_column:
        return pd.DataFrame()
    return (
        df.groupby(["player_id", "player", "team"], dropna=False)
        .agg(
            total_actions=("event_id", "size"),
            mean_action_x=(x_column, "mean"),
            median_action_x=(x_column, "median"),
            mean_action_y=(y_column, "mean"),
            median_action_y=(y_column, "median"),
            action_width_std=(y_column, "std"),
        )
        .reset_index()
        .rename(columns={"player": "player_name"})
    )
