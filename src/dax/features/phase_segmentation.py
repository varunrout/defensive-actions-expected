"""Rule-based defensive phase segmentation for DAx MVP."""

from __future__ import annotations

from typing import Any


def _event_seconds(row: dict[str, Any]) -> int:
    minute = int(row.get("minute") or 0)
    second = int(row.get("second") or 0)
    return (minute * 60) + second


def _event_type(row: dict[str, Any]) -> Any:
    return row.get("event_type") or row.get("type")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    import math
    if math.isnan(out):
        return None
    return out


def label_defensive_phases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    previous_team_in_possession: str | None = None
    previous_time: int | None = None
    previous_match_id: Any = None
    previous_period: Any = None

    for row in rows:
        match_id = row.get("match_id")
        period = row.get("period")
        if previous_match_id is not None and (
            match_id != previous_match_id or period != previous_period
        ):
            # Reset turn-over state at every match/period boundary.
            previous_team_in_possession = None
            previous_time = None

        team_in_possession = row.get("team_in_possession")
        ball_x = _as_float(row.get("ball_x"))
        ball_y = _as_float(row.get("ball_y"))
        event_type = _event_type(row)
        now = _event_seconds(row)

        turnover = previous_team_in_possession is not None and team_in_possession != previous_team_in_possession
        seconds_since_turnover = 999
        if turnover and previous_time is not None:
            seconds_since_turnover = max(0, now - previous_time)

        phase = "settled_mid_block"

        # Unidirectional convention: attacking team progresses left -> right.
        # Defensive low block / box defence therefore happens at high x values.
        if event_type in {"Clearance", "Block"} and ball_x is not None and ball_x >= 95:
            phase = "box_defence"
        elif turnover and seconds_since_turnover <= 5:
            phase = "counterpress_after_loss"
        elif turnover and seconds_since_turnover <= 10:
            phase = "transition_defence"
        elif event_type in {"Duel", "50/50"}:
            phase = "second_ball"
        elif ball_x is not None and ball_x >= 95:
            phase = "box_defence"
        elif ball_x is not None and ball_x >= 75:
            phase = "settled_low_block"
        elif ball_x is not None and ball_x <= 30:
            phase = "high_press"
        elif ball_y is not None and (ball_y <= 12 or ball_y >= 68):
            phase = "wide_defending"
        elif ball_y is not None and 24 <= ball_y <= 56:
            phase = "central_progression_defence"

        labeled_row = dict(row)
        labeled_row["phase_label"] = phase
        labeled_row["seconds_since_turnover"] = seconds_since_turnover if turnover else None
        labeled.append(labeled_row)

        previous_team_in_possession = team_in_possession
        previous_time = now
        previous_match_id = match_id
        previous_period = period

    return labeled

