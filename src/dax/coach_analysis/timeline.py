from __future__ import annotations

import pandas as pd

CHRONOLOGICAL_FIELDS = ['period', 'minute', 'second', 'timestamp', 'event_index', 'index']
EVENT_KEYS = ['match_id', 'event_id']


def normalise_processed_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if 'event_id' not in out.columns and 'id' in out.columns:
        out = out.rename(columns={'id': 'event_id'})
    return out


def event_seconds(df: pd.DataFrame) -> pd.Series:
    if {'minute', 'second'}.issubset(df.columns):
        return pd.to_numeric(df['minute'], errors='coerce') * 60 + pd.to_numeric(df['second'], errors='coerce')
    if 'timestamp' in df.columns:
        ts = pd.to_timedelta(df['timestamp'].astype(str), errors='coerce')
        return ts.dt.total_seconds()
    return pd.Series(pd.NA, index=df.index, dtype='Float64')


def validate_processed_timeline(events: pd.DataFrame) -> dict:
    events = normalise_processed_events(events)
    if events.empty:
        return {'valid': False, 'rows': 0, 'matches': 0, 'event_ids': 0, 'periods': [], 'chronological_fields': [], 'duplicate_event_ids': None, 'next_event_sequence_possible': False}
    chronological = [c for c in CHRONOLOGICAL_FIELDS if c in events.columns]
    has_keys = set(EVENT_KEYS).issubset(events.columns)
    duplicate_event_ids = int(events.duplicated(EVENT_KEYS).sum()) if has_keys else None
    matches = int(events['match_id'].nunique()) if 'match_id' in events.columns else 0
    event_ids = int(events['event_id'].nunique()) if 'event_id' in events.columns else 0
    periods = sorted(events['period'].dropna().unique().tolist()) if 'period' in events.columns else []
    return {'valid': bool(has_keys and chronological and duplicate_event_ids == 0), 'rows': int(len(events)), 'matches': matches, 'event_ids': event_ids, 'periods': periods, 'chronological_fields': chronological, 'duplicate_event_ids': duplicate_event_ids, 'next_event_sequence_possible': bool(has_keys and chronological and duplicate_event_ids == 0 and 'period' in events.columns)}


def add_next_events(actions: pd.DataFrame, events: pd.DataFrame, window_seconds: float = 10.0) -> pd.DataFrame:
    events = normalise_processed_events(events)
    if actions.empty or events.empty or not set(EVENT_KEYS).issubset(actions.columns) or not set(EVENT_KEYS).issubset(events.columns):
        return actions.copy()
    order = [c for c in ['match_id', 'period', 'minute', 'second', 'timestamp', 'event_index', 'index'] if c in events.columns]
    ev = events.sort_values(order or EVENT_KEYS).copy()
    ev['coach_event_seconds'] = event_seconds(ev)
    group = ['match_id', 'period'] if 'period' in ev.columns else ['match_id']
    for col in ['event_id', 'team', 'possession', 'possession_team', 'type', 'event_type', 'x', 'y', 'location_x', 'location_y', 'coach_event_seconds']:
        if col in ev.columns:
            ev[f'next_{col}'] = ev.groupby(group, dropna=False)[col].shift(-1)
            ev[f'second_next_{col}'] = ev.groupby(group, dropna=False)[col].shift(-2)
    if 'next_coach_event_seconds' in ev.columns:
        ev['next_event_within_window'] = (ev['next_coach_event_seconds'] - ev['coach_event_seconds']).le(window_seconds)
        ev.loc[ev['next_coach_event_seconds'].isna(), 'next_event_within_window'] = False
    if 'second_next_coach_event_seconds' in ev.columns:
        ev['second_next_event_within_window'] = (ev['second_next_coach_event_seconds'] - ev['coach_event_seconds']).le(window_seconds)
        ev.loc[ev['second_next_coach_event_seconds'].isna(), 'second_next_event_within_window'] = False
    next_cols = [c for c in ev.columns if c.startswith('next_') or c.startswith('second_next_')]
    keep = EVENT_KEYS + [c for c in ['coach_event_seconds', 'next_event_within_window', 'second_next_event_within_window'] if c in ev.columns] + next_cols
    keep = list(dict.fromkeys(keep))
    return actions.merge(ev[keep], on=EVENT_KEYS, how='left', validate='m:1')
