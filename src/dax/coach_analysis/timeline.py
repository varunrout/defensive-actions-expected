from __future__ import annotations

import pandas as pd

CHRONOLOGICAL_FIELDS = ['period', 'minute', 'second', 'timestamp', 'event_index', 'index']
EVENT_KEYS = ['match_id', 'event_id']


def validate_processed_timeline(events: pd.DataFrame) -> dict:
    if events.empty:
        return {'valid': False, 'rows': 0, 'matches': 0, 'event_ids': 0, 'periods': [], 'chronological_fields': [], 'duplicate_event_ids': None, 'next_event_sequence_possible': False}
    chronological = [c for c in CHRONOLOGICAL_FIELDS if c in events.columns]
    has_keys = set(EVENT_KEYS).issubset(events.columns)
    duplicate_event_ids = int(events.duplicated(EVENT_KEYS).sum()) if has_keys else None
    matches = int(events['match_id'].nunique()) if 'match_id' in events.columns else 0
    event_ids = int(events['event_id'].nunique()) if 'event_id' in events.columns else 0
    periods = sorted(events['period'].dropna().unique().tolist()) if 'period' in events.columns else []
    return {
        'valid': bool(has_keys and chronological and duplicate_event_ids == 0),
        'rows': int(len(events)),
        'matches': matches,
        'event_ids': event_ids,
        'periods': periods,
        'chronological_fields': chronological,
        'duplicate_event_ids': duplicate_event_ids,
        'next_event_sequence_possible': bool(has_keys and chronological and duplicate_event_ids == 0),
    }


def add_next_events(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if actions.empty or events.empty or not set(EVENT_KEYS).issubset(actions.columns) or not set(EVENT_KEYS).issubset(events.columns):
        return actions.copy()
    order = [c for c in ['match_id', 'period', 'minute', 'second', 'timestamp', 'event_index', 'index'] if c in events.columns]
    ev = events.sort_values(order or EVENT_KEYS).copy()
    group = ['match_id']
    for col in ['event_id', 'team', 'possession', 'type', 'event_type', 'x', 'y', 'location_x', 'location_y']:
        if col in ev.columns:
            ev[f'next_{col}'] = ev.groupby(group)[col].shift(-1)
            ev[f'second_next_{col}'] = ev.groupby(group)[col].shift(-2)
    next_cols = [c for c in ev.columns if c.startswith('next_') or c.startswith('second_next_')]
    return actions.merge(ev[EVENT_KEYS + next_cols], on=EVENT_KEYS, how='left', validate='m:1')
