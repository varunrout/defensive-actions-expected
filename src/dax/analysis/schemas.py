"""Strict input contracts for pre-modelling analysis datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemaResult:
    """Schema validation result."""

    name: str
    required: tuple[str, ...]
    optional_present: tuple[str, ...]
    missing_required: tuple[str, ...]
    rows: int
    columns: int

    @property
    def ok(self) -> bool:
        """Return ``True`` when no required columns are missing."""
        return not self.missing_required

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        data = asdict(self)
        data["ok"] = self.ok
        return data


EVENT_REQUIRED: tuple[str, ...] = (
    "match_id",
    "period",
    "index",
    "possession",
    "event_type",
    "phase_label",
    "has_360",
    "target_future_shot_10s",
    "target_future_xg_10s",
    "attacking_team_before_action",
    "defending_team_before_action",
)
EVENT_ALTERNATIVES: tuple[tuple[str, ...], ...] = (("event_id", "id"), ("timestamp", "minute"), ("team", "team_id"), ("player", "player_id"))

PLAYER_REQUIRED: tuple[str, ...] = (
    "match_id",
    "period",
    "possession",
    "event_id",
    "event_index",
    "player_id",
    "player",
    "team",
    "actor_team",
    "attacking_team",
    "defending_team",
    "event_type",
    "action_family",
    "phase_label",
    "action_x",
    "action_y",
    "action_zone",
    "has_360",
    "freeze_frame_roles_known",
    "visibility_quality_band",
    "visibility_limited",
    "local_5m_region_fully_visible",
    "local_10m_region_fully_visible",
    "action_won_possession",
    "action_ended_possession",
    "action_retained_defensive_team_control",
    "action_was_under_opponent_possession",
    "target_future_shot_10s",
    "target_future_xg_10s",
)
PLAYER_OPTIONAL: tuple[str, ...] = (
    "visible_attacker_count",
    "visible_defender_count",
    "attackers_within_5m",
    "defenders_within_5m",
    "attackers_within_10m",
    "defenders_within_10m",
    "nearest_attacker_distance",
    "nearest_defender_distance",
    "defenders_between_ball_and_attacking_goal",
    "local_numerical_balance_5m",
    "local_numerical_balance_10m",
    "distance_to_attacking_goal",
    "distance_to_attacking_box",
    "is_central_lane",
    "is_wide_lane",
    "is_deep_zone",
    "is_high_zone",
    "counterpress",
    "position",
    "position_group",
)


def _missing_with_alternatives(required: tuple[str, ...], alternatives: tuple[tuple[str, ...], ...], columns: pd.Index) -> tuple[str, ...]:
    missing = [column for column in required if column not in columns]
    for group in alternatives:
        if not any(column in columns for column in group):
            missing.append(" or ".join(group))
    return tuple(missing)


def validate_processed_events(df: pd.DataFrame, *, strict: bool = True) -> SchemaResult:
    """Validate the processed event table used before feature construction.

    Raises
    ------
    ValueError
        If ``strict`` is true and a required column is missing.
    """
    result = SchemaResult(
        name="processed_events",
        required=EVENT_REQUIRED,
        optional_present=tuple(column for group in EVENT_ALTERNATIVES for column in group if column in df.columns),
        missing_required=_missing_with_alternatives(EVENT_REQUIRED, EVENT_ALTERNATIVES, df.columns),
        rows=len(df),
        columns=len(df.columns),
    )
    if strict and not result.ok:
        raise ValueError(f"Processed event schema missing required columns: {list(result.missing_required)}")
    return result


def validate_player_actions(df: pd.DataFrame, *, strict: bool = True) -> SchemaResult:
    """Validate the canonical player defensive-action table produced by feature code."""
    missing = tuple(column for column in PLAYER_REQUIRED if column not in df.columns)
    result = SchemaResult(
        name="player_defensive_actions",
        required=PLAYER_REQUIRED,
        optional_present=tuple(column for column in PLAYER_OPTIONAL if column in df.columns),
        missing_required=missing,
        rows=len(df),
        columns=len(df.columns),
    )
    if strict and missing:
        raise ValueError(
            "Player defensive-action schema missing required canonical columns: "
            f"{list(missing)}. Rebuild with src/dax/features/player_defense.py."
        )
    return result


def coordinate_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return canonical action coordinate columns when present."""
    if {"action_x", "action_y"}.issubset(df.columns):
        return "action_x", "action_y"
    return None, None
