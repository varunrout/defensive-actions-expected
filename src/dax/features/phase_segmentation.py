"""Rule-based defensive phase proxy segmentation.

Labels are proxies, not confirmed tactical truth. State is maintained within
match/period so counterpress and transition windows persist across events.
"""
from __future__ import annotations
from typing import Any


def _event_seconds(row: dict[str, Any]) -> int | None:
    if row.get("minute") is None and row.get("second") is None:
        return None
    return int(row.get("minute") or 0) * 60 + int(row.get("second") or 0)


def _event_type(row: dict[str, Any]) -> Any:
    return row.get("event_type") or row.get("type")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def label_defensive_phases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled=[]
    prev_match=prev_period=None; current_possession=None; current_team=None
    lost_team=None; won_team=None; turnover_time=None
    for row in rows:
        match=row.get('match_id'); period=row.get('period')
        if match != prev_match or period != prev_period:
            current_possession=None; current_team=None; lost_team=None; won_team=None; turnover_time=None
        poss=row.get('possession'); team=row.get('team_in_possession') or row.get('possession_team')
        now=_event_seconds(row)
        valid_change = team and current_team and team != current_team and (current_possession is None or poss is None or poss != current_possession)
        if valid_change:
            lost_team=current_team; won_team=team; turnover_time=now
        elapsed = None if turnover_time is None or now is None else max(0, now-turnover_time)
        defending = row.get('defending_team_before_action') or row.get('defending_team') or row.get('team')
        ball_x=_as_float(row.get('ball_x')); ball_y=_as_float(row.get('ball_y')); event_type=_event_type(row)
        phase='settled_mid_block_proxy'; conf=0.45; rule='settled_mid_default'
        if elapsed is None and now is None:
            phase='unknown_time_proxy'; conf=0.0; rule='missing_timestamp'
        elif lost_team and elapsed is not None and elapsed <= 5 and (not defending or defending == lost_team):
            phase='counterpress_after_loss'; conf=0.7; rule='turnover_0_5s'
        elif lost_team and elapsed is not None and 5 < elapsed <= 10 and (not defending or defending == lost_team):
            phase='transition_defence'; conf=0.65; rule='turnover_5_10s'
        elif event_type in {'Duel','50/50'} and row.get('preceded_by_loose_ball_evidence'):
            phase='second_ball'; conf=0.3; rule='contest_with_loose_ball_evidence'
        elif event_type in {'Clearance','Block'} and ball_x is not None and ball_x >= 95:
            phase='box_defence'; conf=0.55; rule='clearance_or_block_near_defensive_box'
        elif ball_x is not None and ball_x >= 95:
            phase='box_defence'; conf=0.5; rule='ball_near_attacking_goal'
        elif ball_x is not None and ball_x >= 75:
            phase='settled_low_block_proxy'; conf=0.45; rule='advanced_ball_proxy'
        elif ball_x is not None and ball_x <= 30:
            phase='high_press_proxy'; conf=0.4; rule='deep_ball_proxy'
        elif ball_y is not None and (ball_y <= 12 or ball_y >= 68):
            phase='wide_defending_proxy'; conf=0.35; rule='wide_ball_proxy'
        labeled_row=dict(row); labeled_row.update({'phase_label':phase,'phase_source':'rule_based_proxy','phase_confidence':conf,'phase_rule_id':rule,'seconds_since_turnover':elapsed,'turnover_lost_team':lost_team,'turnover_won_team':won_team})
        labeled.append(labeled_row)
        if poss is not None:
            current_possession = poss
        if team:
            current_team = team
        prev_match = match
        prev_period = period
    return labeled
