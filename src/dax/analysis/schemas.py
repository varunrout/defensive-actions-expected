from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class SchemaResult:
    name: str
    required: tuple[str, ...]
    optional_present: tuple[str, ...]
    missing_required: tuple[str, ...]
    rows: int
    columns: int
    @property
    def ok(self) -> bool: return not self.missing_required
    def to_dict(self) -> dict: return self.__dict__ | {"ok": self.ok}

EVENT_REQUIRED=("match_id","period","index","possession","event_type","phase_label","target_future_shot_10s","target_future_xg_10s","attacking_team_before_action","defending_team_before_action")
EVENT_ALTERNATIVES=(("event_id","id"),("timestamp","minute"),("team","team_id"),("player","player_id"))
PLAYER_REQUIRED=("match_id","event_id","player_id","player_name","team","actor_team","attacking_team","defending_team","event_type","action_family","phase_label","target_future_shot_10s","target_future_xg_10s")
PLAYER_OPTIONAL=("x","y","location_x","location_y","action_x","action_y","has_360","visible_attacker_count","visible_defender_count","local_numerical_balance","nearest_attacker_distance","nearest_defender_distance","goal_side_defenders","visibility_quality","possession_won","ends_opponent_possession","retained_control","under_opponent_possession","distance_to_attacking_goal","is_central_lane","is_box_defence","is_transition_defence","is_counterpress","is_low_block","high_threat_context")

def _with_alternatives(required, alternatives, cols):
    missing=[c for c in required if c not in cols]
    for group in alternatives:
        if not any(c in cols for c in group): missing.append(" or ".join(group))
    return tuple(missing)

def validate_processed_events(df: pd.DataFrame, *, strict: bool=True) -> SchemaResult:
    res=SchemaResult("processed_events", EVENT_REQUIRED, tuple(c for c in df.columns if c in sum(EVENT_ALTERNATIVES, ())), _with_alternatives(EVENT_REQUIRED, EVENT_ALTERNATIVES, df.columns), len(df), len(df.columns))
    if strict and not res.ok: raise ValueError(f"Processed event schema missing required columns: {res.missing_required}")
    return res

def validate_player_actions(df: pd.DataFrame, *, strict: bool=True) -> SchemaResult:
    missing=tuple(c for c in PLAYER_REQUIRED if c not in df.columns)
    res=SchemaResult("player_defensive_actions", PLAYER_REQUIRED, tuple(c for c in PLAYER_OPTIONAL if c in df.columns), missing, len(df), len(df.columns))
    if strict and missing: raise ValueError(f"Player action schema missing required columns: {missing}")
    return res

def coordinate_columns(df: pd.DataFrame) -> tuple[str|None,str|None]:
    for x,y in (("x","y"),("location_x","location_y"),("action_x","action_y")):
        if x in df.columns and y in df.columns: return x,y
    return None,None
