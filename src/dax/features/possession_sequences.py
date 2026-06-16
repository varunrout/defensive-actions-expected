"""
Possession sequence enrichment for DAx.

Extracts possession-level features from event sequences:
  - Phase trajectory: How defensive phases evolve during the possession
  - Pressure dynamics: How 360-based opposition pressure changes
  - Zone progression: Ball movement through pitch zones
  
A possession is defined as consecutive events by the same team until turnover.
Only possessions with at least one 360 frame are enhanced.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import warnings

from dax.constants import PITCH_X_MAX, PITCH_Y_MAX


def _event_seconds(row: dict[str, Any]) -> int:
    """Convert minute + second to total seconds in period."""
    minute = int(row.get("minute") or 0)
    second = int(row.get("second") or 0)
    return (minute * 60) + second


def _as_float(value: Any) -> float | None:
    """Safely convert to float, handling NaN."""
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


def _get_zone(ball_x: float | None, ball_y: float | None) -> str | None:
    """Classify ball location into pitch zones."""
    if ball_x is None or ball_y is None:
        return None
    
    # X zones (field thirds)
    if ball_x < 40:
        x_zone = "defensive_third"
    elif ball_x < 80:
        x_zone = "middle_third"
    else:
        x_zone = "attacking_third"
    
    # Y zones (flanks vs center)
    if ball_y < 20:
        y_zone = "left_flank"
    elif ball_y > 60:
        y_zone = "right_flank"
    else:
        y_zone = "center"
    
    return f"{x_zone}_{y_zone}"


@dataclass
class PossessionSequence:
    """Represents a single possession sequence."""
    
    possession_id: str
    match_id: int
    period: int
    team_in_possession: str
    defending_team: str
    
    # Event-level data
    event_count: int
    event_indices: list[int]
    
    # Temporal data
    start_time: int  # seconds in period
    end_time: int
    duration: int
    
    # Phase trajectory
    phases: list[str]  # Defensive phases during this possession
    phase_transitions: list[tuple[str, str]]  # (from_phase, to_phase)
    phase_unique_count: int  # How many different phases
    
    # Ball progression
    start_ball_x: float | None
    start_ball_y: float | None
    end_ball_x: float | None
    end_ball_y: float | None
    start_zone: str | None
    end_zone: str | None
    zones_visited: list[str]  # Sequential zones
    
    # 360 pressure dynamics
    events_with_360: int  # How many events have 360 data
    opponent_count_start: int | None  # At first 360-event
    opponent_count_end: int | None    # At last 360-event
    opponent_count_avg: float
    opponent_count_max: int
    opponent_count_min: int
    opponent_count_decay: float  # (start - end) / start, if both exist
    
    teammate_count_avg: float
    teammate_count_max: int
    
    # Outcome
    has_shot_in_10s: int  # Binary target
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return asdict(self)


def _extract_possession_id(
    match_id: int,
    period: int,
    team_in_possession: str,
    sequence_index: int,
) -> str:
    """Generate unique possession ID."""
    return f"{match_id}_{period}_{team_in_possession}_{sequence_index}"


def build_possession_sequences(
    events: list[dict[str, Any]],
    only_with_360: bool = True,
    verbose: bool = False,
) -> list[PossessionSequence]:
    """
    Build possession sequences from event list.
    
    Parameters
    ----------
    events : list[dict]
        List of event dictionaries (from events_with_targets.parquet)
    only_with_360 : bool
        If True, only return possessions with at least one 360 frame.
        If False, return all possessions.
    verbose : bool
        Print progress.
    
    Returns
    -------
    list[PossessionSequence]
        Enriched possession-level features.
    """
    
    if not events:
        return []
    
    possessions: list[PossessionSequence] = []
    current_possession_events: list[tuple[int, dict[str, Any]]] = []
    
    previous_team: str | None = None
    previous_match_id: Any = None
    previous_period: Any = None
    possession_count = 0
    
    for event_idx, event in enumerate(events):
        match_id = event.get("match_id")
        period = event.get("period")
        team_in_possession = event.get("team_in_possession")
        
        # Reset at match/period boundary
        if (
            previous_match_id is not None
            and (match_id != previous_match_id or period != previous_period)
        ):
            if current_possession_events:
                poss = _build_single_possession(
                    current_possession_events,
                    possession_count,
                    previous_match_id,
                    previous_period,
                )
                if poss is not None:
                    possessions.append(poss)
                    possession_count += 1
            current_possession_events = []
            previous_team = None
        
        # Reset at turnover
        if previous_team is not None and team_in_possession != previous_team:
            if current_possession_events:
                poss = _build_single_possession(
                    current_possession_events,
                    possession_count,
                    match_id,
                    period,
                )
                if poss is not None:
                    possessions.append(poss)
                    possession_count += 1
            current_possession_events = []
        
        # Accumulate event
        current_possession_events.append((event_idx, event))
        previous_team = team_in_possession
        previous_match_id = match_id
        previous_period = period
    
    # Don't forget last possession
    if current_possession_events:
        poss = _build_single_possession(
            current_possession_events,
            possession_count,
            previous_match_id,
            previous_period,
        )
        if poss is not None:
            possessions.append(poss)
    
    # Filter by 360 availability if requested
    if only_with_360:
        possessions = [p for p in possessions if p.events_with_360 > 0]
    
    if verbose:
        total_with_360 = sum(1 for p in possessions if p.events_with_360 > 0)
        print(f"[Possession Sequencer] Built {len(possessions)} possessions")
        print(f"  With 360 data: {total_with_360}")
    
    return possessions


def _build_single_possession(
    events_with_indices: list[tuple[int, dict[str, Any]]],
    possession_index: int,
    match_id: int,
    period: int,
) -> PossessionSequence | None:
    """Build a single possession sequence."""
    
    if not events_with_indices:
        return None
    
    first_idx, first_event = events_with_indices[0]
    last_idx, last_event = events_with_indices[-1]
    
    team_in_possession = first_event.get("team_in_possession")
    defending_team = first_event.get("defending_team")
    
    if not team_in_possession:
        return None
    
    possession_id = _extract_possession_id(
        match_id,
        period,
        team_in_possession,
        possession_index,
    )
    
    # Event indices
    event_indices = [idx for idx, _ in events_with_indices]
    
    # Temporal
    start_time = _event_seconds(first_event)
    end_time = _event_seconds(last_event)
    duration = max(0, end_time - start_time)
    
    # Phases
    phases = [e.get("phase_label") for _, e in events_with_indices]
    phases = [p for p in phases if p is not None]
    
    phase_transitions = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_transitions.append((phases[i - 1], phases[i]))
    
    unique_phases = set(phases)
    
    # Ball progression
    start_ball_x = _as_float(first_event.get("ball_x"))
    start_ball_y = _as_float(first_event.get("ball_y"))
    end_ball_x = _as_float(last_event.get("ball_x"))
    end_ball_y = _as_float(last_event.get("ball_y"))
    
    start_zone = _get_zone(start_ball_x, start_ball_y)
    end_zone = _get_zone(end_ball_x, end_ball_y)
    
    # Zone progression
    zones_visited = []
    for _, event in events_with_indices:
        zone = _get_zone(
            _as_float(event.get("ball_x")),
            _as_float(event.get("ball_y")),
        )
        if zone and (not zones_visited or zones_visited[-1] != zone):
            zones_visited.append(zone)
    
    # 360 pressure dynamics
    opponent_counts = []
    teammate_counts = []
    
    for _, event in events_with_indices:
        has_360 = event.get("has_360", False)
        if has_360:
            opp = event.get("opponent_count")
            if opp is not None:
                opponent_counts.append(int(opp))
            
            tm = event.get("teammate_count")
            if tm is not None:
                teammate_counts.append(int(tm))
    
    events_with_360 = len(opponent_counts)
    
    opponent_count_start = opponent_counts[0] if opponent_counts else None
    opponent_count_end = opponent_counts[-1] if opponent_counts else None
    opponent_count_avg = sum(opponent_counts) / len(opponent_counts) if opponent_counts else 0.0
    opponent_count_max = max(opponent_counts) if opponent_counts else 0
    opponent_count_min = min(opponent_counts) if opponent_counts else 0
    
    opponent_count_decay = 0.0
    if (
        opponent_count_start is not None
        and opponent_count_end is not None
        and opponent_count_start > 0
    ):
        opponent_count_decay = (opponent_count_start - opponent_count_end) / opponent_count_start
    
    teammate_count_avg = sum(teammate_counts) / len(teammate_counts) if teammate_counts else 0.0
    teammate_count_max = max(teammate_counts) if teammate_counts else 0
    
    # Outcome
    has_shot_in_10s = max(
        (event.get("target_shot_in_10s") or 0) for _, event in events_with_indices
    )
    
    return PossessionSequence(
        possession_id=possession_id,
        match_id=match_id,
        period=period,
        team_in_possession=team_in_possession,
        defending_team=defending_team,
        event_count=len(event_indices),
        event_indices=event_indices,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        phases=phases,
        phase_transitions=phase_transitions,
        phase_unique_count=len(unique_phases),
        start_ball_x=start_ball_x,
        start_ball_y=start_ball_y,
        end_ball_x=end_ball_x,
        end_ball_y=end_ball_y,
        start_zone=start_zone,
        end_zone=end_zone,
        zones_visited=zones_visited,
        events_with_360=events_with_360,
        opponent_count_start=opponent_count_start,
        opponent_count_end=opponent_count_end,
        opponent_count_avg=opponent_count_avg,
        opponent_count_max=opponent_count_max,
        opponent_count_min=opponent_count_min,
        opponent_count_decay=opponent_count_decay,
        teammate_count_avg=teammate_count_avg,
        teammate_count_max=teammate_count_max,
        has_shot_in_10s=int(has_shot_in_10s),
    )

