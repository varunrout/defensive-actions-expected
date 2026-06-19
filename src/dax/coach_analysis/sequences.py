from __future__ import annotations

import numpy as np
import pandas as pd

def _event_time_seconds(frame: pd.DataFrame) -> pd.Series:
    if 'event_time_seconds' in frame.columns:
        return frame['event_time_seconds'].astype(float)
    minute = frame['minute'].fillna(0).astype(float) if 'minute' in frame.columns else 0.0
    second = frame['second'].fillna(0).astype(float) if 'second' in frame.columns else 0.0
    return minute * 60.0 + second


def _first_non_matching_team(team_values: np.ndarray, start: int, own_team: object) -> int | None:
    for pos in range(start, len(team_values)):
        if team_values[pos] != own_team:
            return pos
    return None


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
    event_required = {'match_id', 'period', 'team', 'type', 'possession'}
    if 'event_index' not in events.columns and 'index' not in events.columns:
        raise ValueError("events_enriched missing required sequence column: event_index or index")
    missing = sorted(event_required.difference(events.columns))
    if missing:
        raise ValueError(f"events_enriched missing required sequence columns: {missing}")
    events_local = events.copy()
    if 'event_index' not in events_local.columns and 'index' in events_local.columns:
        events_local['event_index'] = events_local['index']
    events_local['event_time_seconds'] = _event_time_seconds(events_local)
    events_local['event_type'] = events_local['type'].astype(str)

    out['coach_next_event_type'] = pd.NA
    out['coach_next_opposition_event_type'] = pd.NA
    out['coach_second_opposition_event_type'] = pd.NA
    out['coach_attack_recycled'] = False
    out['coach_shot_followed'] = False
    out['coach_clearance_followed_by_opposition_recovery'] = False
    out['coach_block_followed_by_rebound'] = False
    out['coach_recovery_followed_by_immediate_turnover'] = False
    out['coach_pressure_followed_by_progression'] = False

    events_local = events_local.sort_values(['match_id', 'period', 'event_index']).copy()
    out_groups = out.groupby(['match_id', 'period'], dropna=False)

    for (match_id, period), action_group in out_groups:
        evg = events_local[(events_local['match_id'] == match_id) & (events_local['period'] == period)]
        if evg.empty:
            continue
        ev_idx = evg['event_index'].to_numpy(dtype=float)
        ev_type = evg['event_type'].astype(str).to_numpy()
        ev_type_lower = np.char.lower(ev_type.astype(str))
        ev_team = evg['team'].to_numpy()
        ev_time = evg['event_time_seconds'].to_numpy(dtype=float)

        team_to_positions: dict[object, np.ndarray] = {}
        for team_value in pd.Series(ev_team).dropna().unique().tolist():
            team_to_positions[team_value] = np.flatnonzero(ev_team == team_value)

        for row in action_group.itertuples():
            pos = int(np.searchsorted(ev_idx, float(row.event_index), side='right'))
            if pos >= len(ev_idx):
                continue

            out.at[row.Index, 'coach_next_event_type'] = ev_type[pos]
            out.at[row.Index, 'coach_shot_followed'] = bool(np.any(ev_type_lower[pos:min(pos + 3, len(ev_type_lower))] == 'shot'))

            opposition_team = getattr(row, 'attacking_team', None)
            next_opp_pos: int | None = None
            second_opp_pos: int | None = None
            if opposition_team in team_to_positions:
                team_positions = team_to_positions[opposition_team]
                lookup = int(np.searchsorted(ev_idx[team_positions], float(row.event_index), side='right'))
                if lookup < len(team_positions):
                    next_opp_pos = int(team_positions[lookup])
                if lookup + 1 < len(team_positions):
                    second_opp_pos = int(team_positions[lookup + 1])
            else:
                own_team = getattr(row, 'team', None)
                next_opp_pos = _first_non_matching_team(ev_team, pos, own_team)
                if next_opp_pos is not None:
                    second_opp_pos = _first_non_matching_team(ev_team, next_opp_pos + 1, own_team)

            if next_opp_pos is not None:
                out.at[row.Index, 'coach_next_opposition_event_type'] = ev_type[next_opp_pos]
                out.at[row.Index, 'coach_attack_recycled'] = bool(ev_time[next_opp_pos] - float(row.coach_action_time_seconds) <= recycle_window_seconds)
            if second_opp_pos is not None:
                out.at[row.Index, 'coach_second_opposition_event_type'] = ev_type[second_opp_pos]

            current_type = str(getattr(row, 'event_type', '')).lower()
            current_family = str(getattr(row, 'action_family', '')).lower()
            opp_window_positions = []
            if next_opp_pos is not None:
                opp_window_positions.append(next_opp_pos)
            if second_opp_pos is not None:
                opp_window_positions.append(second_opp_pos)
            opp_window_types = [ev_type_lower[p] for p in opp_window_positions]

            out.at[row.Index, 'coach_clearance_followed_by_opposition_recovery'] = (
                ('clearance' in current_type)
                and any(('recovery' in typ) or ('interception' in typ) for typ in opp_window_types)
            )
            out.at[row.Index, 'coach_block_followed_by_rebound'] = (
                ('block' in current_type)
                and any(('shot' in typ) or ('recovery' in typ) for typ in opp_window_types)
            )
            out.at[row.Index, 'coach_recovery_followed_by_immediate_turnover'] = (
                (('recovery' in current_family) or ('interception' in current_family)) and (next_opp_pos is not None)
            )
            out.at[row.Index, 'coach_pressure_followed_by_progression'] = (
                ('pressure' in current_family)
                and (next_opp_pos is not None)
                and any(token in str(ev_type_lower[next_opp_pos]) for token in ('pass', 'carry', 'dribble'))
            )
    return out
