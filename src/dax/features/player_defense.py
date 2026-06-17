"""Player defensive feature engineering for DAx.
Builds a 360-only, player-action table for defensive events.
A row represents one defensive action by an identifiable player,
with phase, possession, and freeze-frame support context.
"""
from __future__ import annotations
from itertools import groupby
from math import sqrt
from typing import Any
DEFENSIVE_ACTION_TYPES = {"Pressure", "Ball Recovery", "Duel", "Clearance", "Block", "Interception", "Foul Committed", "50/50"}
ACTION_FAMILIES = {"Pressure": "pressure", "Ball Recovery": "recovery", "Duel": "contest", "50/50": "contest", "Clearance": "intervention", "Block": "intervention", "Interception": "intervention", "Foul Committed": "discipline"}
def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out
def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            out = value.tolist()
            return out if isinstance(out, list) else [out]
        except Exception:
            return []
    return []
def _event_type(row: dict[str, Any]) -> str | None:
    value = row.get("type") or row.get("event_type")
    return str(value) if value is not None else None
def _is_defensive_action(event_type: str | None) -> bool:
    return bool(event_type and event_type in DEFENSIVE_ACTION_TYPES)
def _action_family(event_type: str | None) -> str:
    return ACTION_FAMILIES.get(event_type or "", "other")
def _position_group(position: Any) -> str:
    text = str(position or "").lower()
    if not text:
        return "unknown"
    if "goalkeeper" in text or "keeper" in text:
        return "goalkeeper"
    if any(t in text for t in ["centre back", "center back", "cb", "sweeper"]):
        return "centre_back"
    if any(t in text for t in ["full back", "wing back", "left back", "right back", "fb", "wb"]):
        return "fullback_wingback"
    if any(t in text for t in ["defensive midfield", "holding midfield", "anchor", "dm"]):
        return "defensive_midfielder"
    if "midfield" in text:
        return "midfielder"
    if any(t in text for t in ["winger", "wide midfielder"]):
        return "winger"
    if any(t in text for t in ["forward", "striker", "attacker"]):
        return "forward"
    return "other"
def _location(row: dict[str, Any]) -> tuple[float, float] | None:
    loc = row.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        x = _as_float(loc[0])
        y = _as_float(loc[1])
        if x is not None and y is not None:
            return x, y
    if hasattr(loc, "tolist"):
        try:
            loc_list = loc.tolist()
            if isinstance(loc_list, list) and len(loc_list) >= 2:
                x = _as_float(loc_list[0])
                y = _as_float(loc_list[1])
                if x is not None and y is not None:
                    return x, y
        except Exception:
            return None
    return None
def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)
def _pitch_zone(x: float | None, y: float | None) -> str | None:
    if x is None or y is None:
        return None
    x_zone = "defensive_third" if x < 40 else "middle_third" if x < 80 else "attacking_third"
    y_zone = "left_flank" if y < 20 else "right_flank" if y > 60 else "center"
    return f"{x_zone}_{y_zone}"
def _goal_metrics(x: float, y: float) -> dict[str, Any]:
    """Goal geometry after normalising attack left-to-right.

    Attacking goal is always (120, 40); defending goal is always (0, 40).
    """
    import math
    distance_to_attacking_goal = _distance(x, y, 120.0, 40.0)
    distance_to_defending_goal = _distance(x, y, 0.0, 40.0)
    angle_to_attacking_goal = abs(math.atan2(y - 40.0, 120.0 - x))
    distance_to_attacking_box = max(0.0, 102.0 - x) if 18.0 <= y <= 62.0 else _distance(x, y, min(max(x, 102.0), 120.0), min(max(y, 18.0), 62.0))
    distance_to_defending_box = max(0.0, x - 18.0) if 18.0 <= y <= 62.0 else _distance(x, y, min(max(x, 0.0), 18.0), min(max(y, 18.0), 62.0))
    return {
        "distance_to_attacking_goal": distance_to_attacking_goal,
        "distance_to_defending_goal": distance_to_defending_goal,
        "angle_to_attacking_goal": angle_to_attacking_goal,
        "distance_to_attacking_box": distance_to_attacking_box,
        "distance_to_defending_box": distance_to_defending_box,
        "attacking_goal_centrality": max(0.0, 1.0 - abs(y - 40.0) / 40.0),
        "is_in_attacking_box": int(x >= 102.0 and 18.0 <= y <= 62.0),
        "is_in_defending_box": int(x <= 18.0 and 18.0 <= y <= 62.0),
        "distance_to_center_line": abs(y - 40.0),
        "is_central_lane": 1 if 24.0 <= y <= 56.0 else 0,
        "is_wide_lane": 1 if y <= 12.0 or y >= 68.0 else 0,
        "is_deep_zone": 1 if x < 30.0 else 0,
        "is_high_zone": 1 if x > 90.0 else 0,
    }
def _freeze_frame_points(frame: Any, actor_is_attacking: bool | None = None) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    attackers: list[tuple[float, float]] = []
    defenders: list[tuple[float, float]] = []
    for player in _safe_list(frame):
        if not isinstance(player, dict):
            continue
        loc = player.get("location")
        if not isinstance(loc, (list, tuple)):
            if hasattr(loc, "tolist"):
                try:
                    loc = loc.tolist()
                except Exception:
                    loc = None
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            continue
        x = _as_float(loc[0])
        y = _as_float(loc[1])
        if x is None or y is None:
            continue
        if player.get("teammate") is True:
            if actor_is_attacking is False:
                defenders.append((x, y))
            else:
                attackers.append((x, y))
        elif player.get("teammate") is False:
            if actor_is_attacking is False:
                attackers.append((x, y))
            else:
                defenders.append((x, y))
    return attackers, defenders
def _centroid(points: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if not points:
        return None, None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)
def _spread(points: list[tuple[float, float]], centroid: tuple[float | None, float | None]) -> float | None:
    if not points or centroid[0] is None or centroid[1] is None:
        return None
    cx, cy = centroid
    return sum(_distance(x, y, cx, cy) for x, y in points) / len(points)
def _density(points: list[tuple[float, float]], x: float, y: float, radius: float) -> int:
    return sum(1 for px, py in points if _distance(px, py, x, y) <= radius)
def _polygon_points(area: Any) -> list[tuple[float, float]]:
    vals = [_as_float(v) for v in _safe_list(area)]
    nums = [v for v in vals if v is not None]
    return list(zip(nums[0::2], nums[1::2], strict=False)) if len(nums) >= 6 else []

def _polygon_area(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=False))) / 2

def _point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool | None:
    if len(points) < 3:
        return None
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside

def _visibility_features(area: Any, x: float, y: float) -> dict[str, Any]:
    pts = _polygon_points(area)
    poly_area = _polygon_area(pts)
    frac = None if poly_area is None else min(1.0, poly_area / (120.0 * 80.0))
    action_inside = _point_in_polygon(x, y, pts)
    band = "missing" if frac is None else "high" if frac >= 0.75 else "medium" if frac >= 0.4 else "low"
    return {
        "visible_area_polygon_area": poly_area,
        "visible_area_fraction_of_pitch": frac,
        "ball_inside_visible_area": action_inside,
        "action_inside_visible_area": action_inside,
        "local_5m_region_fully_visible": bool(action_inside and frac is not None and frac >= 0.4),
        "local_10m_region_fully_visible": bool(action_inside and frac is not None and frac >= 0.75),
        "visibility_quality_band": band,
        "visibility_limited": band in {"missing", "low"},
    }

def _support_features(frame: Any, x: float, y: float, actor_is_attacking: bool | None, visibility: dict[str, Any]) -> dict[str, Any]:
    attackers, defenders = _freeze_frame_points(frame, actor_is_attacking)
    a_centroid = _centroid(attackers)
    d_centroid = _centroid(defenders)
    a5 = _density(attackers, x, y, 5.0)
    d5 = _density(defenders, x, y, 5.0)
    a10 = _density(attackers, x, y, 10.0)
    d10 = _density(defenders, x, y, 10.0)
    local_ok_5 = visibility.get("local_5m_region_fully_visible")
    local_ok_10 = visibility.get("local_10m_region_fully_visible")
    if not local_ok_5:
        a5 = d5 = None
    if not local_ok_10:
        a10 = d10 = None
    return {
        "visible_attacker_count": len(attackers),
        "visible_defender_count": len(defenders),
        "attackers_within_5m": a5,
        "defenders_within_5m": d5,
        "attackers_within_10m": a10,
        "defenders_within_10m": d10,
        "nearest_attacker_distance": min((_distance(px, py, x, y) for px, py in attackers), default=None),
        "nearest_defender_distance": min((_distance(px, py, x, y) for px, py in defenders), default=None),
        "attacker_centroid_x": a_centroid[0],
        "attacker_centroid_y": a_centroid[1],
        "defender_centroid_x": d_centroid[0],
        "defender_centroid_y": d_centroid[1],
        "attacker_spread": _spread(attackers, a_centroid),
        "defender_spread": _spread(defenders, d_centroid),
        "defenders_goal_side_count": sum(1 for px, _ in defenders if px <= x),
        "local_numerical_balance_5m": None if a5 is None or d5 is None else a5 - d5,
        "local_numerical_balance_10m": None if a10 is None or d10 is None else a10 - d10,
        "attacker_defender_ratio": len(attackers) / max(1, len(defenders)),
    }
def build_player_defensive_actions(events: list[dict[str, Any]], only_with_360: bool = True, defensive_only: bool = True, verbose: bool = False) -> list[dict[str, Any]]:
    """Return one row per defensive action by an identifiable player."""
    if not events:
        return []
    indexed_events = list(enumerate(events))
    indexed_events.sort(key=lambda pair: (int(pair[1].get("match_id") or -1), int(pair[1].get("period") or -1), int(pair[1].get("index") if pair[1].get("index") is not None else pair[0])))
    rows: list[dict[str, Any]] = []
    for (match_id, period, possession), group_iter in groupby(indexed_events, key=lambda pair: (pair[1].get("match_id"), pair[1].get("period"), pair[1].get("possession"))):
        group = list(group_iter)
        if not group:
            continue
        group_start = (_as_float(group[0][1].get("minute") or 0) or 0) * 60 + (_as_float(group[0][1].get("second") or 0) or 0)
        prev_phase: str | None = None
        phase_changes = 0
        for order_in_possession, (_, row) in enumerate(group):
            phase_label = row.get("phase_label")
            if prev_phase is not None and phase_label != prev_phase:
                phase_changes += 1
            event_type = _event_type(row)
            prev_phase = phase_label
            if only_with_360 and not row.get("has_360"):
                continue
            if defensive_only and not _is_defensive_action(event_type):
                continue
            loc = _location(row)
            if loc is None or row.get("player_id") is None:
                continue
            action_x, action_y = loc
            actor_team = row.get("actor_team") or row.get("team")
            attacking_team = row.get("attacking_team_before_action") or row.get("possession_team") or row.get("team_in_possession")
            defending_team = row.get("defending_team_before_action") or row.get("defending_team")
            if attacking_team is not None and defending_team is not None and attacking_team == defending_team:
                continue
            actor_is_attacking = None if actor_team is None or attacking_team is None else actor_team == attacking_team
            visibility = _visibility_features(row.get("visible_area"), action_x, action_y)
            support = _support_features(row.get("freeze_frame"), action_x, action_y, actor_is_attacking, visibility)
            goal_metrics = _goal_metrics(action_x, action_y)
            rows.append({
                "match_id": match_id,
                "period": period,
                "possession": possession,
                "possession_id": f"{match_id}_{period}_{possession}",
                "event_id": row.get("id"),
                "event_index": row.get("index"),
                "event_order_in_possession": order_in_possession,
                "match_time_seconds": (_as_float(row.get("minute") or 0) or 0) * 60 + (_as_float(row.get("second") or 0) or 0),
                "possession_elapsed_seconds": max(0.0, ((_as_float(row.get("minute") or 0) or 0) * 60 + (_as_float(row.get("second") or 0) or 0)) - group_start),
                "events_elapsed_in_possession": order_in_possession + 1,
                "phase_transitions_observed_so_far": phase_changes,
                "phase_changed_since_prev_event": int(order_in_possession > 0 and phase_label != group[order_in_possession - 1][1].get("phase_label")),
                "phase_label": phase_label,
                "target_future_shot_10s": int(row.get("target_future_shot_10s") or 0),
                "target_future_xg_10s": float(row.get("target_future_xg_10s") or 0.0),
                "has_360": bool(row.get("has_360")),
                "team": row.get("team"),
                "team_id": row.get("team_id"),
                "player": row.get("player"),
                "player_id": row.get("player_id"),
                "position": row.get("position"),
                "position_group": _position_group(row.get("position")),
                "actor_team": actor_team,
                "attacking_team": attacking_team,
                "defending_team": defending_team,
                "event_type": event_type,
                "action_family": _action_family(event_type),
                "play_pattern": row.get("play_pattern"),
                "counterpress": bool(row.get("counterpress")),
                "action_changed_possession": bool(row.get("action_changed_possession")),
                "action_ended_possession": bool(row.get("action_ended_possession", row.get("action_changed_possession"))),
                "action_won_possession": bool(row.get("action_won_possession")),
                "action_retained_defensive_team_control": bool(row.get("action_retained_defensive_team_control")),
                "action_was_under_opponent_possession": bool(row.get("action_was_under_opponent_possession")),
                "action_x": action_x,
                "action_y": action_y,
                "action_zone": _pitch_zone(action_x, action_y),
                **goal_metrics,
                "ball_x": row.get("ball_x"),
                "ball_y": row.get("ball_y"),
                "freeze_frame_count": row.get("freeze_frame_count"),
                **visibility,
                **support,
                "phase_label_prev_event": group[order_in_possession - 1][1].get("phase_label") if order_in_possession > 0 else None,
            })
    if verbose:
        print(f"[Player Defense] Built {len(rows)} defensive-action rows")
    return rows
