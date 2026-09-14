"""Passive/off-ball defensive feature engineering for DAx.

Builds a 360-only table of anonymous defender-slots observed in the freeze
frame of attacking on-ball events (Pass, Carry, Dribble, Shot). This is a
descriptive, observational signal about defensive positioning -- it makes no
causal claim about what a defender's positioning "caused". Player identity is
deliberately out of scope here (defender-slots are anonymous within a frame);
that is a later phase.
"""
from __future__ import annotations
from itertools import groupby
from typing import Any

from dax.features.player_defense import (
    _as_float,
    _distance,
    _distance_point_to_segment,
    _freeze_frame_points,
    _goal_metrics,
    _location,
    _none_if_missing,
    _pitch_zone,
    _visibility_features,
)


ON_BALL_EVENT_TYPES = {"Pass", "Carry", "Dribble", "Shot"}
TOP_K_OPTIONS = 3
OPTION_DENSITY_RADIUS_M = 10.0
LANE_SCREENING_DISTANCE_THRESHOLD_M = 2.0


def _event_type(row: dict[str, Any]) -> str | None:
    value = row.get("type") or row.get("event_type")
    return str(value) if value is not None else None


def _is_on_ball_event(event_type: str | None) -> bool:
    return bool(event_type and event_type in ON_BALL_EVENT_TYPES)


def _possession_group_key(row: dict[str, Any]) -> Any:
    sequence_id = row.get("possession_sequence_id")
    return sequence_id if sequence_id is not None else row.get("possession")


def _marking_features(defender_x: float, defender_y: float, attackers: list[tuple[float, float]]) -> dict[str, Any]:
    """Distance/side features relative to the nearest visible attacker.

    is_goal_side_of_nearest_attacker compares x-position only, consistent with
    the pitch normalisation used throughout this module (defending goal is
    always at x=0, attacking goal always at x=120).
    """
    if not attackers:
        return {"marking_tightness": None, "is_goal_side_of_nearest_attacker": None}
    nearest_x, nearest_y = min(attackers, key=lambda p: _distance(p[0], p[1], defender_x, defender_y))
    marking_tightness = _distance(defender_x, defender_y, nearest_x, nearest_y)
    return {
        "marking_tightness": marking_tightness,
        "is_goal_side_of_nearest_attacker": defender_x < nearest_x,
    }


def _zone_defensive_value(defender_x: float, defender_y: float) -> dict[str, Any]:
    """Pitch zone plus a simple inverse-distance-to-defending-goal weighting.

    zone_defensive_value = 1 / (1 + distance_to_defending_goal). This is a
    deliberately simple monotonic decay (closer to the defending goal scores
    higher), not a learned weighting -- it is not intended as a model of
    positional value.
    """
    distance_to_defending_goal = _goal_metrics(defender_x, defender_y)["distance_to_defending_goal"]
    return {
        "defender_zone": _pitch_zone(defender_x, defender_y),
        "zone_defensive_value": 1.0 / (1.0 + distance_to_defending_goal),
    }


def _rank_option_candidates(attackers: list[tuple[float, float]], defenders: list[tuple[float, float]], k: int = TOP_K_OPTIONS) -> list[dict[str, Any]]:
    """Rank freeze-frame attackers as plausible pass-target options.

    threat_score = goal_proximity * openness, both in (0, 1]:
      - goal_proximity = 1 / (1 + distance_to_attacking_goal) via _goal_metrics
      - openness = 1 / (1 + count of defenders within OPTION_DENSITY_RADIUS_M)
    A deterministic, documented formula -- not a fitted model. Higher score
    means a candidate that is both closer to the attacking goal and less
    locally marked by defenders; it is a descriptive proxy, not a prediction
    of what the on-ball player will actually do.
    """
    candidates = []
    for x, y in attackers:
        distance_to_attacking_goal = _goal_metrics(x, y)["distance_to_attacking_goal"]
        goal_proximity = 1.0 / (1.0 + distance_to_attacking_goal)
        nearby_defenders = sum(1 for dx, dy in defenders if _distance(dx, dy, x, y) <= OPTION_DENSITY_RADIUS_M)
        openness = 1.0 / (1.0 + nearby_defenders)
        candidates.append({"x": x, "y": y, "threat_score": goal_proximity * openness})
    candidates.sort(key=lambda c: c["threat_score"], reverse=True)
    return candidates[:k]


def _coverage_counts(defenders: list[tuple[float, float]], carrier_x: float, carrier_y: float, options: list[dict[str, Any]]) -> list[int]:
    """For each ranked option, credit the single closest defender-slot as its coverer.

    overload_score is the resulting per-defender tally: how many of the
    top-ranked threats this defender-slot is the closest one to, among all
    defender-slots visible in the frame. A descriptive count, not a claim
    about who "should" have covered a threat.
    """
    counts = [0 for _ in defenders]
    if not defenders:
        return counts
    for option in options:
        distances = [_distance_point_to_segment(dx, dy, carrier_x, carrier_y, option["x"], option["y"]) for dx, dy in defenders]
        closest_index = min(range(len(distances)), key=lambda i: distances[i])
        counts[closest_index] += 1
    return counts


def _lane_occlusion_features(defender_x: float, defender_y: float, carrier_x: float, carrier_y: float, options: list[dict[str, Any]]) -> dict[str, Any]:
    """Point-to-segment distance from the defender to each carrier->option lane.

    lane_screening_score = 1 / (1 + distance): 1.0 when the defender sits
    exactly on the lane, decaying towards 0 as it moves away.
    screens_top_option is True when the defender is within
    LANE_SCREENING_DISTANCE_THRESHOLD_M of the top-ranked option's lane.
    """
    features: dict[str, Any] = {}
    top_option_1_distance = None
    for i in range(TOP_K_OPTIONS):
        rank = i + 1
        option = options[i] if i < len(options) else None
        if option is None:
            distance = None
            score = None
            target_x = target_y = threat_score = None
        else:
            distance = _distance_point_to_segment(defender_x, defender_y, carrier_x, carrier_y, option["x"], option["y"])
            score = 1.0 / (1.0 + distance)
            target_x, target_y, threat_score = option["x"], option["y"], option["threat_score"]
        if rank == 1:
            top_option_1_distance = distance
        features[f"top_option_{rank}_target_x"] = target_x
        features[f"top_option_{rank}_target_y"] = target_y
        features[f"top_option_{rank}_threat_score"] = threat_score
        features[f"lane_screening_score_option_{rank}"] = score
    features["screens_top_option"] = None if top_option_1_distance is None else bool(top_option_1_distance <= LANE_SCREENING_DISTANCE_THRESHOLD_M)
    return features


def build_passive_defense_rows(events: list[dict[str, Any]], only_with_360: bool = True, verbose: bool = False) -> list[dict[str, Any]]:
    """Return one row per (defending-team player-slot, attacking on-ball event).

    Defender-slots are anonymous freeze-frame opponents of the on-ball
    player's team -- no player identity is attached here.
    """
    if not events:
        return []
    indexed_events = list(enumerate(events))
    indexed_events.sort(key=lambda pair: (int(pair[1].get("match_id") or -1), int(pair[1].get("period") or -1), int(pair[1].get("index") if pair[1].get("index") is not None else pair[0])))
    rows: list[dict[str, Any]] = []
    for (match_id, period, possession), group_iter in groupby(indexed_events, key=lambda pair: (pair[1].get("match_id"), pair[1].get("period"), _possession_group_key(pair[1]))):
        group = list(group_iter)
        if not group:
            continue
        for order_in_possession, (_, row) in enumerate(group):
            event_type = _event_type(row)
            if only_with_360 and not row.get("has_360"):
                continue
            if not _is_on_ball_event(event_type):
                continue
            carrier_loc = _location(row)
            freeze_frame = row.get("freeze_frame")
            if carrier_loc is None or freeze_frame is None:
                continue
            carrier_x, carrier_y = carrier_loc
            on_ball_team = _none_if_missing(row.get("actor_team")) or _none_if_missing(row.get("team"))
            defending_team = _none_if_missing(row.get("defending_team_before_action")) or _none_if_missing(row.get("defending_team"))
            ball_x = _as_float(row.get("ball_x"))
            ball_y = _as_float(row.get("ball_y"))
            attackers, defenders = _freeze_frame_points(freeze_frame, True)
            options = _rank_option_candidates(attackers, defenders)
            coverage_counts = _coverage_counts(defenders, carrier_x, carrier_y, options)
            for defender_slot_index, (defender_x, defender_y) in enumerate(defenders):
                visibility = _visibility_features(row.get("visible_area"), defender_x, defender_y, ball_x, ball_y)
                goal_metrics = _goal_metrics(defender_x, defender_y)
                marking = _marking_features(defender_x, defender_y, attackers)
                zone = _zone_defensive_value(defender_x, defender_y)
                lane_occlusion = _lane_occlusion_features(defender_x, defender_y, carrier_x, carrier_y, options)
                rows.append({
                    "match_id": match_id,
                    "period": period,
                    "possession_sequence_id": possession,
                    "event_id": row.get("id"),
                    "event_order_in_possession": order_in_possession,
                    "defender_slot_index": defender_slot_index,
                    "on_ball_event_type": event_type,
                    "on_ball_team": on_ball_team,
                    "defending_team": defending_team,
                    "phase_label": row.get("phase_label"),
                    "defender_x": defender_x,
                    "defender_y": defender_y,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "carrier_x": carrier_x,
                    "carrier_y": carrier_y,
                    **goal_metrics,
                    **visibility,
                    **marking,
                    **zone,
                    "engagement_distance_to_carrier": _distance(defender_x, defender_y, carrier_x, carrier_y),
                    **lane_occlusion,
                    "overload_score": coverage_counts[defender_slot_index],
                })
    if verbose:
        print(f"[Passive Defense] Built {len(rows)} defender-slot rows")
    return rows
