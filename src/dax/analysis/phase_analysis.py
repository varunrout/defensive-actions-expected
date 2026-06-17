"""Rule-based phase-proxy analysis."""
from __future__ import annotations

import pandas as pd


def phase_tables(df: pd.DataFrame, min_count: int = 20) -> dict[str, pd.DataFrame]:
    """Return phase frequency, player exposure, and descriptive outcome summaries.

    Phase labels are rule-based tactical proxies, not ground-truth tactical labels.
    """
    frequency = df["phase_label"].value_counts(dropna=False).rename_axis("phase_label").reset_index(name="rows")
    player_column = "player_name" if "player_name" in df.columns else "player"
    if {"player_id", player_column, "phase_label"}.issubset(df.columns):
        by_player = (
            df.groupby(["player_id", player_column, "phase_label"], dropna=False)
            .size()
            .reset_index(name="actions")
            .rename(columns={player_column: "player_name"})
        )
    else:
        by_player = pd.DataFrame()
    outcomes = (
        df.groupby("phase_label", dropna=False)
        .agg(
            rows=("phase_label", "size"),
            future_shot_rate=("target_future_shot_10s", "mean"),
            future_xg_mean=("target_future_xg_10s", "mean"),
        )
        .reset_index()
    )
    if "action_won_possession" in df.columns:
        win_rates = df.groupby("phase_label", dropna=False)["action_won_possession"].mean().reset_index(name="possession_win_rate")
        outcomes = outcomes.merge(win_rates, on="phase_label", how="left")
    outcomes["minimum_sample_warning"] = outcomes["rows"] < min_count
    return {"phase_frequency": frequency, "phase_exposure_by_player": by_player, "phase_outcomes": outcomes}
