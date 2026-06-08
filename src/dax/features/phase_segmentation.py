"""Rule-based defensive phase segmentation for DAx MVP."""

from __future__ import annotations

from typing import Any


def _event_seconds(row: dict[str, Any]) -> int:
    minute = int(row.get("minute") or 0)
    second = int(row.get("second") or 0)
    return (minute * 60) + second


def label_defensive_phases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled: list[dict[str, Any]] = []
    previous_team_in_possession: str | None = None
    previous_time: int | None = None

    for row in rows:
        team_in_possession = row.get("team_in_possession")
        ball_x = row.get("ball_x")
        event_type = row.get("event_type")
        now = _event_seconds(row)

        turnover = previous_team_in_possession is not None and team_in_possession != previous_team_in_possession
        seconds_since_turnover = 999
        if turnover and previous_time is not None:
            seconds_since_turnover = max(0, now - previous_time)

        phase = "settled_mid_block"

        if event_type in {"Clearance", "Block"} and ball_x is not None and ball_x <= 25:
            phase = "box_defence"
        elif turnover and seconds_since_turnover <= 5:
            phase = "counterpress_after_loss"
        elif turnover and seconds_since_turnover <= 10:
            phase = "transition_defence"
        elif ball_x is not None and ball_x <= 30:
            phase = "settled_low_block"
        elif ball_x is not None and ball_x >= 75:
            phase = "high_press"
        elif event_type in {"Duel", "50/50"}:
            phase = "second_ball"
        elif ball_x is not None and ball_x >= 95:
            phase = "box_defence"
        elif row.get("ball_y") is not None and (row["ball_y"] <= 12 or row["ball_y"] >= 68):
            phase = "wide_defending"
        elif row.get("ball_y") is not None and 24 <= row["ball_y"] <= 56:
            phase = "central_progression_defence"

        labeled_row = dict(row)
        labeled_row["phase_label"] = phase
        labeled_row["seconds_since_turnover"] = seconds_since_turnover if turnover else None
        labeled.append(labeled_row)

        previous_team_in_possession = team_in_possession
        previous_time = now

    return labeled

