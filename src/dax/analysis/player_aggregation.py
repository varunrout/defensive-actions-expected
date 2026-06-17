"""Player-level aggregation for canonical defensive-action features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .spatial_analysis import add_pitch_zones

IDENTITY_COLUMNS = ["player_id", "player", "team"]



def _add_categorical_counts(summary: pd.DataFrame, df: pd.DataFrame, keys: list[str], column: str, prefix: str) -> pd.DataFrame:
    table = pd.crosstab([df[key] for key in keys], df[column], dropna=False).reset_index()
    table.columns = keys + [f"{prefix}_{str(value)}_count" for value in table.columns[len(keys):]]
    out = summary.merge(table, on=keys, how="left")
    count_columns = [column for column in out.columns if column.startswith(f"{prefix}_") and column.endswith("_count")]
    for count_column in count_columns:
        share_column = count_column.replace("_count", "_share")
        out[share_column] = out[count_column] / out["total_actions"].replace(0, np.nan)
    return out


def _add_target_rates_by_category(summary: pd.DataFrame, df: pd.DataFrame, keys: list[str], column: str, prefix: str) -> pd.DataFrame:
    grouped = df.groupby(keys + [column], dropna=False).agg(
        denominator=("event_id", "size"),
        future_shot_count=("target_future_shot_10s", "sum"),
        future_xg_total=("target_future_xg_10s", "sum"),
    )
    grouped["future_shot_rate"] = grouped["future_shot_count"] / grouped["denominator"].replace(0, np.nan)
    grouped["future_xg_mean"] = grouped["future_xg_total"] / grouped["denominator"].replace(0, np.nan)
    wide = grouped.reset_index().pivot(index=keys, columns=column)
    wide.columns = [f"{prefix}_{category}_{metric}" for metric, category in wide.columns]
    return summary.merge(wide.reset_index(), on=keys, how="left")


def build_player_summary(df: pd.DataFrame, min_actions: int = 30, *, grid_dimensions: tuple[int, int] = (6, 4)) -> pd.DataFrame:
    """Aggregate canonical player defensive-action rows to one row per player-team.

    Every rate has an explicit denominator column. Visibility reliability uses local-region visibility,
    not ``has_360``. Role-known actions use ``freeze_frame_roles_known``.
    """
    bins_x, bins_y = grid_dimensions
    data = add_pitch_zones(df, bins_x=bins_x, bins_y=bins_y).copy()
    data["is_defensive_box_action"] = (
        data["action_x"].between(0.0, 18.0, inclusive="both")
        & data["action_y"].between(18.0, 62.0, inclusive="both")
    )
    keys = IDENTITY_COLUMNS
    grouped = data.groupby(keys, dropna=False)

    summary = grouped.agg(
        matches=("match_id", "nunique"),
        total_actions=("event_id", "size"),
        future_shot_count=("target_future_shot_10s", "sum"),
        future_shot_rate=("target_future_shot_10s", "mean"),
        future_xg_total=("target_future_xg_10s", "sum"),
        future_xg_mean=("target_future_xg_10s", "mean"),
        mean_action_x=("action_x", "mean"),
        mean_action_y=("action_y", "mean"),
        median_action_x=("action_x", "median"),
        median_action_y=("action_y", "median"),
        action_width_std=("action_y", "std"),
        box_defence_actions=("is_defensive_box_action", "sum"),
    ).reset_index()
    summary = summary.rename(columns={"player": "player_name"})
    bool_metrics = {
        "action_won_possession": "possession_wins",
        "action_ended_possession": "opponent_possessions_ended",
        "action_retained_defensive_team_control": "retained_control_actions",
        "action_was_under_opponent_possession": "actions_under_opponent_possession",
        "freeze_frame_roles_known": "role_known_actions",
        "visibility_limited": "visibility_limited_actions",
    }
    reliable_visibility_mask = data["local_5m_region_fully_visible"].fillna(False).astype(bool) & data[
        "local_10m_region_fully_visible"
    ].fillna(False).astype(bool)
    data["analysis_reliable_visibility"] = reliable_visibility_mask
    bool_metrics["analysis_reliable_visibility"] = "reliable_visibility_actions"

    bool_counts = []
    for source, output in bool_metrics.items():
        counts = grouped[source].apply(lambda values: int(values.fillna(False).astype(bool).sum())).reset_index(name=output)
        bool_counts.append(counts.rename(columns={"player": "player_name"}))
    for counts in bool_counts:
        summary = summary.merge(counts, on=["player_id", "player_name", "team"], how="left")

    summary["minimum_sample_flag"] = summary["total_actions"] < min_actions
    summary["future_shot_denominator"] = summary["total_actions"]
    summary["future_xg_denominator"] = summary["total_actions"]
    summary["actions_per_match"] = summary["total_actions"] / summary["matches"].replace(0, np.nan)
    summary["box_defence_denominator"] = summary["total_actions"]
    summary["box_defence_share"] = summary["box_defence_actions"] / summary["box_defence_denominator"].replace(0, np.nan)

    rate_specs = {
        "possession_wins": "possession_win_rate",
        "opponent_possessions_ended": "opponent_possession_end_rate",
        "retained_control_actions": "retained_control_rate",
        "actions_under_opponent_possession": "actions_under_opponent_possession_rate",
        "role_known_actions": "role_known_share",
        "reliable_visibility_actions": "reliable_visibility_share",
        "visibility_limited_actions": "visibility_limited_share",
    }
    for count_column, rate_column in rate_specs.items():
        summary[f"{rate_column}_denominator"] = summary["total_actions"]
        summary[rate_column] = summary[count_column] / summary["total_actions"].replace(0, np.nan)

    numeric_optional = {
        "local_numerical_balance_5m": "mean_local_numerical_balance_5m",
        "local_numerical_balance_10m": "mean_local_numerical_balance_10m",
        "visible_attacker_count": "mean_visible_attacker_count",
        "visible_defender_count": "mean_visible_defender_count",
        "nearest_attacker_distance": "mean_nearest_attacker_distance",
        "nearest_defender_distance": "mean_nearest_defender_distance",
        "defenders_between_ball_and_attacking_goal": "mean_defenders_between_ball_and_attacking_goal",
    }
    for source, output in numeric_optional.items():
        if source in data.columns:
            values = grouped[source].mean().reset_index(name=output).rename(columns={"player": "player_name"})
            summary = summary.merge(values, on=["player_id", "player_name", "team"], how="left")

    for balance_column, suffix in (("local_numerical_balance_5m", "5m"), ("local_numerical_balance_10m", "10m")):
        if balance_column in data.columns:
            valid = data[balance_column].notna()
            data[f"numerical_disadvantage_{suffix}"] = valid & (data[balance_column] < 0)
            denom = grouped[balance_column].count().reset_index(name=f"numerical_disadvantage_{suffix}_denominator")
            count = grouped[f"numerical_disadvantage_{suffix}"].sum().reset_index(name=f"numerical_disadvantage_{suffix}_count")
            denom = denom.rename(columns={"player": "player_name"})
            count = count.rename(columns={"player": "player_name"})
            summary = summary.merge(denom, on=["player_id", "player_name", "team"], how="left")
            summary = summary.merge(count, on=["player_id", "player_name", "team"], how="left")
            summary[f"numerical_disadvantage_{suffix}_share"] = summary[f"numerical_disadvantage_{suffix}_count"] / summary[
                f"numerical_disadvantage_{suffix}_denominator"
            ].replace(0, np.nan)

    for column, prefix in (("action_family", "action_family"), ("phase_label", "phase"), ("pitch_zone", "zone"), ("visibility_quality_band", "visibility_quality")):
        summary = _add_categorical_counts(summary, data.rename(columns={"player": "player_name"}), ["player_id", "player_name", "team"], column, prefix)

    renamed = data.rename(columns={"player": "player_name"})
    summary = _add_target_rates_by_category(summary, renamed, ["player_id", "player_name", "team"], "action_family", "action_family")
    summary = _add_target_rates_by_category(summary, renamed, ["player_id", "player_name", "team"], "phase_label", "phase")

    return summary
