"""Canonical feature schemas for generated analysis notebooks."""

from __future__ import annotations

TEAM_NOTEBOOK_AGGREGATION_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("matches", "match_id", "nunique"),
    ("actions", "event_id", "size"),
    ("players", "player_id", "nunique"),
    ("shot_rate", "target_future_shot_10s", "mean"),
    ("mean_future_xg", "target_future_xg_10s", "mean"),
    ("has_360_share", "has_360", "mean"),
    ("counterpress_share", "counterpress", "mean"),
    ("central_lane_share", "is_central_lane", "mean"),
    ("wide_lane_share", "is_wide_lane", "mean"),
    ("deep_zone_share", "is_deep_zone", "mean"),
    ("high_zone_share", "is_high_zone", "mean"),
    ("avg_goal_distance", "distance_to_attacking_goal", "mean"),
    ("avg_support_balance_10m", "local_numerical_balance_10m", "mean"),
    ("avg_support_ratio_10m", "attackers_within_10m", "mean"),
    ("avg_nearest_attacker_distance", "nearest_attacker_distance", "mean"),
)

PLAYER_NOTEBOOK_AGGREGATION_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("player", "player", "mode_or_unknown"),
    ("primary_team", "team", "mode_or_unknown"),
    ("primary_position", "position", "mode_or_unknown"),
    ("position_group", "position_group", "mode_or_unknown"),
    ("matches", "match_id", "nunique"),
    ("actions", "event_id", "size"),
    ("shot_rate", "target_future_shot_10s", "mean"),
    ("mean_future_xg", "target_future_xg_10s", "mean"),
    ("has_360_share", "has_360", "mean"),
    ("counterpress_share", "counterpress", "mean"),
    ("central_lane_share", "is_central_lane", "mean"),
    ("wide_lane_share", "is_wide_lane", "mean"),
    ("deep_zone_share", "is_deep_zone", "mean"),
    ("high_zone_share", "is_high_zone", "mean"),
    ("avg_goal_distance", "distance_to_attacking_goal", "mean"),
    ("avg_support_balance_10m", "local_numerical_balance_10m", "mean"),
    ("avg_support_ratio_10m", "attackers_within_10m", "mean"),
    ("avg_nearest_attacker_distance", "nearest_attacker_distance", "mean"),
)
