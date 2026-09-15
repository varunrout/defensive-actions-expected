"""Passive/off-ball defensive feature engineering for DAx.

Builds a 360-only table of anonymous defender-slots observed in the freeze
frame of attacking on-ball events (Pass, Carry, Dribble, Shot). This is a
descriptive, observational signal about defensive positioning -- it makes no
causal claim about what a defender's positioning "caused". Player identity
is deliberately out of scope, permanently -- this project works at the
position/role/archetype level, not the named-player level. Where identity
would otherwise be needed, a per-frame functional role label
(_functional_roles) is used instead, inferred purely from a defender-slot's
position relative to its own team's other visible defenders in that same
frame -- no lineup join, no cross-frame tracking of any kind.
"""
from __future__ import annotations
from itertools import groupby
from math import atan2, hypot
from typing import Any

import pandas as pd

from dax.features.player_defense import (
    _as_float,
    _distance,
    _distance_point_to_segment,
    _exclude_actor_from_frame,
    _freeze_frame_points,
    _goal_metrics,
    _location,
    _none_if_missing,
    _pitch_zone,
    _visibility_features,
)
from dax.targets.short_horizon import add_future_shot_target, add_future_xg_target


ON_BALL_EVENT_TYPES = {"Pass", "Carry", "Dribble", "Shot"}
TOP_K_OPTIONS = 3
OPTION_DENSITY_RADIUS_M = 10.0
LANE_SCREENING_DISTANCE_THRESHOLD_M = 2.0
OPTION_MATCH_DISTANCE_THRESHOLD_M = 5.0
DEEP_RATIO_MAX = 1.0 / 3.0
ADVANCED_RATIO_MIN = 2.0 / 3.0
CENTRAL_RATIO_MAX = 1.0 / 3.0
WIDE_RATIO_MIN = 2.0 / 3.0


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


def _functional_roles(defenders: list[tuple[float, float]]) -> list[str]:
    """Classify each defender-slot's role relative to its OWN team's other
    visible defenders in this same freeze frame -- no player identity, no
    lineup join, no cross-frame tracking of any kind.

    "Deep"/"advanced" and "central"/"wide" only mean anything relative to
    the defenders visible in this frame, not absolute pitch coordinates --
    a deep block and a high line both have a "deepest" member.

    depth_ratio min-max-normalises x across the group (0 = deepest of the
    group, i.e. closest to the defending goal at x=0; 1 = most advanced of
    the group). When every visible defender shares the same x, depth
    carries no signal and depth_ratio is fixed at 0.5 (neither deep nor
    advanced) rather than guessing.

    lateral_ratio is each defender's absolute deviation from the group's
    mean y, divided by the largest such deviation in the group (0 = most
    central, 1 = most wide). When every visible defender shares the same y,
    lateral_ratio is fixed at 0 (nobody is wide relative to a group with no
    width).

    Buckets (first match wins; every combination of depth x laterality is
    covered, so there is no silent default among rows that can be measured):
      - last_line:      deep (depth_ratio <= 1/3) and central (lateral_ratio <= 1/3)
      - wide_cover:      deep (depth_ratio <= 1/3) and wide (lateral_ratio >= 2/3)
      - central_screen:  advanced (depth_ratio >= 2/3) and central (lateral_ratio <= 1/3)
      - advanced_wide:   advanced (depth_ratio >= 2/3) and wide (lateral_ratio >= 2/3)
      - mid_block:       everything else with n >= 2 -- measured, and simply not in the
                          extreme third on one or both axes (ordinary mid-block
                          positioning), not a measurement failure.

    "unclassified" is reserved exclusively for n < 2 (fewer than two visible
    defenders means there is nothing to be "relative to", so every slot in
    that frame is unclassified). unclassified and mid_block mean different
    things -- couldn't-measure vs. measured-and-ordinary -- and must never be
    treated as interchangeable downstream (e.g. neither "missing" imputation
    nor a shot-rate comparison should conflate the two).
    """
    n = len(defenders)
    if n < 2:
        return ["unclassified" for _ in defenders]
    xs = [x for x, _ in defenders]
    ys = [y for _, y in defenders]
    min_x, max_x = min(xs), max(xs)
    mean_y = sum(ys) / n
    deviations = [abs(y - mean_y) for y in ys]
    max_deviation = max(deviations)

    roles = []
    for x, deviation in zip(xs, deviations, strict=False):
        depth_ratio = 0.5 if max_x == min_x else (x - min_x) / (max_x - min_x)
        lateral_ratio = 0.0 if max_deviation == 0 else deviation / max_deviation
        is_deep = depth_ratio <= DEEP_RATIO_MAX
        is_advanced = depth_ratio >= ADVANCED_RATIO_MIN
        is_central = lateral_ratio <= CENTRAL_RATIO_MAX
        is_wide = lateral_ratio >= WIDE_RATIO_MIN
        if is_deep and is_central:
            roles.append("last_line")
        elif is_deep and is_wide:
            roles.append("wide_cover")
        elif is_advanced and is_central:
            roles.append("central_screen")
        elif is_advanced and is_wide:
            roles.append("advanced_wide")
        else:
            roles.append("mid_block")
    return roles


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


def _compute_targets(events: list[dict[str, Any]]) -> dict[int, tuple[int, float]]:
    """Per-event future-shot/future-xG targets, keyed by each event's own
    position in the input list (matching the enumerate() index used to
    build indexed_events below).

    Delegates entirely to the existing add_future_shot_target /
    add_future_xg_target (dax.targets.short_horizon), which already scores
    each event from its own timestamp forward within its own possession --
    this is a per-row computation, not a per-possession broadcast. It must
    run over the FULL event list, not just a has_360 subset a caller may
    have pre-filtered: a shot recorded on a non-360 event still needs to
    count towards a nearby 360 event's 10s window, or the target would be
    understated.
    """
    events_df = pd.DataFrame(events)
    if events_df.empty:
        return {}
    events_df["_orig_idx"] = range(len(events_df))
    if "shot_statsbomb_xg" not in events_df.columns:
        # add_future_xg_target does `df.get(xg_column, 0.0).fillna(...)`, which
        # only works when the column exists (even if all-NaN); a genuinely
        # absent column returns the float default, not a Series.
        events_df["shot_statsbomb_xg"] = 0.0
    possession_column = "possession_sequence_id" if "possession_sequence_id" in events_df.columns else "possession"
    targeted = add_future_shot_target(events_df, possession_column=possession_column)
    targeted = add_future_xg_target(targeted, possession_column=possession_column)
    return {
        int(orig_idx): (int(shot), float(xg))
        for orig_idx, shot, xg in zip(targeted["_orig_idx"], targeted["target_future_shot_10s"], targeted["target_future_xg_10s"], strict=False)
    }


def _next_event_locations(indexed_events: list[tuple[int, dict[str, Any]]]) -> dict[int, tuple[float, float] | None]:
    """The real next event's location for each event, by its own position
    in the input list -- never the possession's eventual outcome. None at
    the end of a match/period, when there is no next event to compare.
    """
    locations: dict[int, tuple[float, float] | None] = {}
    for pos, (orig_idx, row) in enumerate(indexed_events):
        next_location = None
        if pos + 1 < len(indexed_events):
            next_orig_idx, next_row = indexed_events[pos + 1]
            if row.get("match_id") == next_row.get("match_id") and row.get("period") == next_row.get("period"):
                next_location = _location(next_row)
        locations[orig_idx] = next_location
    return locations


def _screened_option_was_avoided(lane_occlusion: dict[str, Any], next_location: tuple[float, float] | None) -> bool | None:
    """Did the real next event in the data land somewhere other than the
    option this defender-slot was screening most tightly?

    "Most tightly screened" = whichever of the top-ranked options has this
    defender's highest lane_screening_score_option_i. A match is decided by
    proximity of the next event's location to that option's snapshotted
    freeze-frame coordinates (within OPTION_MATCH_DISTANCE_THRESHOLD_M) --
    no receiver identity is tracked, so this is a coarse spatial proxy, not
    a confirmed pass-completion match. Descriptive only: this does not
    claim the screening caused the option to be avoided.

    None when there is no next event in the data (end of period/match) or
    this defender had no ranked option to screen at all.
    """
    if next_location is None:
        return None
    best_rank = None
    best_score = None
    for rank in range(1, TOP_K_OPTIONS + 1):
        score = lane_occlusion.get(f"lane_screening_score_option_{rank}")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_rank = rank
    if best_rank is None:
        return None
    target_x = lane_occlusion.get(f"top_option_{best_rank}_target_x")
    target_y = lane_occlusion.get(f"top_option_{best_rank}_target_y")
    if target_x is None or target_y is None:
        return None
    next_x, next_y = next_location
    matched = _distance(next_x, next_y, target_x, target_y) <= OPTION_MATCH_DISTANCE_THRESHOLD_M
    return not matched


def _ball_relative_option_features(lane_occlusion: dict[str, Any], ball_x: float | None, ball_y: float | None) -> dict[str, Any]:
    """Ball-relative (dx/dy/distance/angle) version of each ranked option's
    absolute target coordinates.

    top_option_n_target_x/y are absolute pitch coordinates, which is why they
    correlate strongly with ball_x and with each other (correlation analysis:
    reports/eda/CORRELATION_ANALYSIS.json) -- the same absolute position means
    something different depending on where the ball is. These relative
    features carry the same underlying information without that confound.
    The raw _target_x/_target_y columns are kept alongside these (not
    replaced) so the correlation re-check has both to compare.
    """
    features: dict[str, Any] = {}
    for rank in range(1, TOP_K_OPTIONS + 1):
        target_x = lane_occlusion.get(f"top_option_{rank}_target_x")
        target_y = lane_occlusion.get(f"top_option_{rank}_target_y")
        if ball_x is None or ball_y is None or target_x is None or target_y is None:
            dx = dy = distance_from_ball = angle_from_ball = None
        else:
            dx = target_x - ball_x
            dy = target_y - ball_y
            distance_from_ball = hypot(dx, dy)
            angle_from_ball = atan2(dy, dx)
        features[f"top_option_{rank}_dx"] = dx
        features[f"top_option_{rank}_dy"] = dy
        features[f"top_option_{rank}_distance_from_ball"] = distance_from_ball
        features[f"top_option_{rank}_angle_from_ball"] = angle_from_ball
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
    target_lookup = _compute_targets(events)
    next_location_lookup = _next_event_locations(indexed_events)
    rows: list[dict[str, Any]] = []
    for (match_id, period, possession), group_iter in groupby(indexed_events, key=lambda pair: (pair[1].get("match_id"), pair[1].get("period"), _possession_group_key(pair[1]))):
        group = list(group_iter)
        if not group:
            continue
        for order_in_possession, (orig_idx, row) in enumerate(group):
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
            attackers, defenders = _freeze_frame_points(_exclude_actor_from_frame(freeze_frame), True)
            options = _rank_option_candidates(attackers, defenders)
            coverage_counts = _coverage_counts(defenders, carrier_x, carrier_y, options)
            functional_roles = _functional_roles(defenders)
            target_future_shot_10s, target_future_xg_10s = target_lookup.get(orig_idx, (None, None))
            next_location = next_location_lookup.get(orig_idx)
            for defender_slot_index, (defender_x, defender_y) in enumerate(defenders):
                visibility = _visibility_features(row.get("visible_area"), defender_x, defender_y, ball_x, ball_y)
                goal_metrics = _goal_metrics(defender_x, defender_y)
                marking = _marking_features(defender_x, defender_y, attackers)
                zone = _zone_defensive_value(defender_x, defender_y)
                lane_occlusion = _lane_occlusion_features(defender_x, defender_y, carrier_x, carrier_y, options)
                ball_relative_options = _ball_relative_option_features(lane_occlusion, ball_x, ball_y)
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
                    # carrier_x/carrier_y are intentionally not emitted as
                    # columns: the on-ball event's location IS the ball
                    # location, so they are always identical to ball_x/
                    # ball_y -- ball_x/ball_y is the canonical column.
                    # carrier_x/carrier_y remain as local variables above,
                    # used by _rank_option_candidates/_coverage_counts/
                    # _lane_occlusion_features/engagement_distance_to_carrier.
                    **goal_metrics,
                    **visibility,
                    **marking,
                    **zone,
                    "engagement_distance_to_carrier": _distance(defender_x, defender_y, carrier_x, carrier_y),
                    **lane_occlusion,
                    **ball_relative_options,
                    "overload_score": coverage_counts[defender_slot_index],
                    "defender_functional_role": functional_roles[defender_slot_index],
                    "target_future_shot_10s": target_future_shot_10s,
                    "target_future_xg_10s": target_future_xg_10s,
                    "screened_option_was_avoided": _screened_option_was_avoided(lane_occlusion, next_location),
                })
    if verbose:
        print(f"[Passive Defense] Built {len(rows)} defender-slot rows")
    return rows
