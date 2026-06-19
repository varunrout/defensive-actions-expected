from __future__ import annotations
import pandas as pd

def _event_time_seconds(frame: pd.DataFrame) -> pd.Series:
    if 'event_time_seconds' in frame.columns:
        return frame['event_time_seconds'].astype(float)
    minute = frame['minute'].fillna(0).astype(float) if 'minute' in frame.columns else 0.0
    second = frame['second'].fillna(0).astype(float) if 'second' in frame.columns else 0.0
    return minute * 60.0 + second


def _next_rows(events: pd.DataFrame, action: pd.Series) -> pd.DataFrame:
    same_group = (
        events['match_id'].eq(action['match_id'])
        & events['period'].eq(action['period'])
        & events['event_index'].gt(action['event_index'])
    )
    return events.loc[same_group].sort_values('event_index')


def construct_sequences(
    actions: pd.DataFrame,
    events: pd.DataFrame | None = None,
    *,
    repeated_window_seconds: float = 10.0,
    recycle_window_seconds: float = 8.0,
) -> pd.DataFrame:
    if actions.empty:
        return actions.copy()
    order = [column for column in ('match_id', 'period', 'event_index') if column in actions.columns]
    out = actions.sort_values(order).copy()
    out['coach_action_time_seconds'] = _event_time_seconds(out)

    if {'match_id', 'period', 'team', 'possession'}.issubset(out.columns):
        delta = out.groupby(['match_id', 'period', 'team', 'possession'], dropna=False)['coach_action_time_seconds'].diff()
        out['coach_is_repeated_defensive_action'] = delta.le(repeated_window_seconds).fillna(False)
    else:
        out['coach_is_repeated_defensive_action'] = False

    retained = out['action_retained_defensive_team_control'] if 'action_retained_defensive_team_control' in out.columns else pd.Series(False, index=out.index)
    changed = out['action_changed_possession'] if 'action_changed_possession' in out.columns else pd.Series(False, index=out.index)
    out['coach_possession_retained'] = retained.fillna(False).eq(True)
    out['coach_possession_changed'] = changed.fillna(False).eq(True)

    if events is None or events.empty:
        return out
    event_required = {'match_id', 'period', 'event_index', 'team', 'type', 'possession'}
    missing = sorted(event_required.difference(events.columns))
    if missing:
        raise ValueError(f"events_enriched missing required sequence columns: {missing}")
    events_local = events.copy()
    events_local['event_time_seconds'] = _event_time_seconds(events_local)
    events_local['event_type'] = events_local['type'].astype(str)

    next_event_type = []
    next_opposition_type = []
    second_opposition_type = []
    attack_recycled = []
    shot_followed = []
    clearance_opp_recovery = []
    block_rebound = []
    recovery_turnover = []
    pressure_progression = []

    for _, row in out.iterrows():
        candidates = _next_rows(events_local, row)
        opposition_team = row.get('attacking_team', None)
        if candidates.empty:
            next_event_type.append(pd.NA)
            next_opposition_type.append(pd.NA)
            second_opposition_type.append(pd.NA)
            attack_recycled.append(False)
            shot_followed.append(False)
            clearance_opp_recovery.append(False)
            block_rebound.append(False)
            recovery_turnover.append(False)
            pressure_progression.append(False)
            continue

        first_event = candidates.iloc[0]
        next_event_type.append(first_event['event_type'])

        if opposition_team is not None:
            opposition_events = candidates[candidates['team'].eq(opposition_team)]
        else:
            opposition_events = candidates[candidates['team'].ne(row.get('team', None))]

        if opposition_events.empty:
            next_opposition_type.append(pd.NA)
            second_opposition_type.append(pd.NA)
            opp_within_window = opposition_events
        else:
            next_opposition_type.append(opposition_events.iloc[0]['event_type'])
            second_opposition_type.append(opposition_events.iloc[1]['event_type'] if len(opposition_events) > 1 else pd.NA)
            opp_within_window = opposition_events[
                opposition_events['event_time_seconds'].sub(row['coach_action_time_seconds']).le(recycle_window_seconds)
            ]

        attack_recycled.append(not opp_within_window.empty)
        shot_followed.append(bool(candidates['event_type'].str.lower().eq('shot').head(3).any()))

        current_type = str(row.get('event_type', '')).lower()
        current_family = str(row.get('action_family', '')).lower()
        clearance_opp_recovery.append(
            ('clearance' in current_type)
            and bool(opposition_events['event_type'].str.lower().str.contains('recovery|interception').head(2).any())
        )
        block_rebound.append(
            ('block' in current_type)
            and bool(opposition_events['event_type'].str.lower().str.contains('shot|recovery').head(2).any())
        )
        recovery_turnover.append(
            bool(('recovery' in current_family or 'interception' in current_family) and not opposition_events.head(1).empty)
        )
        pressure_progression.append(
            ('pressure' in current_family)
            and bool(opposition_events['event_type'].str.lower().str.contains('pass|carry|dribble').head(1).any())
        )

    out['coach_next_event_type'] = next_event_type
    out['coach_next_opposition_event_type'] = next_opposition_type
    out['coach_second_opposition_event_type'] = second_opposition_type
    out['coach_attack_recycled'] = pd.Series(attack_recycled, index=out.index)
    out['coach_shot_followed'] = pd.Series(shot_followed, index=out.index)
    out['coach_clearance_followed_by_opposition_recovery'] = pd.Series(clearance_opp_recovery, index=out.index)
    out['coach_block_followed_by_rebound'] = pd.Series(block_rebound, index=out.index)
    out['coach_recovery_followed_by_immediate_turnover'] = pd.Series(recovery_turnover, index=out.index)
    out['coach_pressure_followed_by_progression'] = pd.Series(pressure_progression, index=out.index)
    return out
