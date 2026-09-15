import math
from pathlib import Path

import pandas as pd
import pytest

from dax.features.player_defense import (
    _freeze_frame_points,
    _support_features,
    _visibility_features,
    build_player_defensive_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL = [0, 0, 120, 0, 120, 80, 0, 80]
SMALL = [40, 20, 80, 20, 80, 60, 40, 60]


def test_goal_side_defenders_use_attacking_goal_direction():
    support = _support_features(
        [
            {"teammate": False, "location": [60, 40]},
            {"teammate": False, "location": [80, 40]},
            {"teammate": False, "location": [100, 40]},
        ],
        x=70,
        y=40,
        ball_x=70,
        actor_is_attacking=True,
        visibility=_visibility_features(FULL, 70, 40, 70, 40),
    )
    assert support["defenders_between_ball_and_attacking_goal"] == 2


def test_freeze_frame_roles_three_state_logic():
    frame = [{"teammate": True, "location": [10, 10]}, {"teammate": False, "location": [20, 20]}]
    assert _freeze_frame_points(frame, True) == ([(10.0, 10.0)], [(20.0, 20.0)])
    assert _freeze_frame_points(frame, False) == ([(20.0, 20.0)], [(10.0, 10.0)])
    assert _freeze_frame_points(frame, None) == ([], [])


def test_unknown_role_context_returns_missing_role_features():
    support = _support_features(
        [{"teammate": True, "location": [10, 10]}, {"teammate": False, "location": [20, 20]}],
        x=15,
        y=15,
        ball_x=15,
        actor_is_attacking=None,
        visibility=_visibility_features(FULL, 15, 15, 15, 15),
    )
    assert support["freeze_frame_roles_known"] is False
    assert support["visible_attacker_count"] is None
    assert support["visible_defender_count"] is None
    assert support["defenders_between_ball_and_attacking_goal"] is None
    assert support["defender_attacker_gap_x"] is None
    assert support["defender_attacker_gap_y"] is None


def test_defender_attacker_gap_is_defender_centroid_minus_attacker_centroid():
    # Attackers at (10, 10) and (30, 10) -> centroid (20, 10).
    # Defenders at (50, 40) and (70, 60) -> centroid (60, 50).
    # gap = defender_centroid - attacker_centroid = (40, 40).
    support = _support_features(
        [
            {"teammate": True, "location": [10, 10]},
            {"teammate": True, "location": [30, 10]},
            {"teammate": False, "location": [50, 40]},
            {"teammate": False, "location": [70, 60]},
        ],
        x=15,
        y=15,
        ball_x=15,
        actor_is_attacking=True,
        visibility=_visibility_features(FULL, 15, 15, 15, 15),
    )
    assert math.isclose(support["attacker_centroid_x"], 20.0)
    assert math.isclose(support["defender_centroid_x"], 60.0)
    assert math.isclose(support["defender_attacker_gap_x"], 40.0)
    assert math.isclose(support["defender_attacker_gap_y"], 40.0)


def test_defender_attacker_gap_is_none_without_defenders_or_attackers():
    support = _support_features(
        [{"teammate": True, "location": [10, 10]}],
        x=15,
        y=15,
        ball_x=15,
        actor_is_attacking=True,
        visibility=_visibility_features(FULL, 15, 15, 15, 15),
    )
    assert support["defender_centroid_x"] is None
    assert support["defender_attacker_gap_x"] is None
    assert support["defender_attacker_gap_y"] is None


def test_local_visibility_middle_large_polygon():
    out = _visibility_features(FULL, 60, 40, 60, 40)
    assert out["action_inside_visible_area"] is True
    assert out["ball_inside_visible_area"] is True
    assert out["local_5m_region_fully_visible"] is True
    assert out["local_10m_region_fully_visible"] is True


def test_local_visibility_close_to_polygon_boundary():
    out = _visibility_features(SMALL, 42, 40, 42, 40)
    assert out["action_inside_visible_area"] is True
    assert out["local_5m_region_fully_visible"] is False
    assert out["local_10m_region_fully_visible"] is False


def test_partially_visible_5m_and_10m_regions():
    partial_5 = _visibility_features([50, 30, 63, 30, 63, 50, 50, 50], 60, 40, 60, 40)
    assert partial_5["local_5m_region_fully_visible"] is False
    assert partial_5["local_10m_region_fully_visible"] is False
    partial_10 = _visibility_features([50, 30, 67, 30, 67, 50, 50, 50], 60, 40, 60, 40)
    assert partial_10["local_5m_region_fully_visible"] is True
    assert partial_10["local_10m_region_fully_visible"] is False


def test_missing_and_malformed_visible_polygon_are_not_visible():
    for area in [None, [1, 2, 3], [0, 0, 0, 0, 0, 0]]:
        out = _visibility_features(area, 60, 40, 60, 40)
        assert out["action_inside_visible_area"] is None
        assert out["ball_inside_visible_area"] is None
        assert out["local_5m_region_fully_visible"] is False
        assert out["local_10m_region_fully_visible"] is False


def test_action_near_pitch_boundary_clips_local_region_to_pitch():
    out = _visibility_features(FULL, 2, 40, 2, 40)
    assert out["local_5m_region_fully_visible"] is True
    assert out["local_10m_region_fully_visible"] is True


def test_ball_and_action_visibility_calculated_independently():
    out = _visibility_features(SMALL, 60, 40, 90, 40)
    assert out["action_inside_visible_area"] is True
    assert out["ball_inside_visible_area"] is False


def _targeted_row(**updates):
    row = {
        "match_id": 1,
        "period": 1,
        "possession": 1,
        "index": 1,
        "minute": 0,
        "second": 1,
        "id": "e1",
        "event_type": "Pressure",
        "player_id": 1,
        "player": "P",
        "team": "B",
        "actor_team": "B",
        "attacking_team_before_action": "A",
        "defending_team_before_action": "B",
        "location": [70, 40],
        "ball_x": 70,
        "ball_y": 40,
        "has_360": True,
        "freeze_frame": [],
        "visible_area": FULL,
        "target_future_shot_10s": 0,
        "target_future_xg_10s": 0.0,
    }
    row.update(updates)
    return row


def test_required_targets_raise_for_missing_key_or_value():
    missing_key = _targeted_row()
    missing_key.pop("target_future_shot_10s")
    with pytest.raises(ValueError, match="target_future_shot_10s"):
        build_player_defensive_actions([missing_key])

    missing_value = _targeted_row(target_future_xg_10s=math.nan)
    with pytest.raises(ValueError, match="target_future_xg_10s"):
        build_player_defensive_actions([missing_value])


def test_targets_can_be_missing_when_not_required():
    row = _targeted_row()
    row.pop("target_future_shot_10s")
    row.pop("target_future_xg_10s")
    out = pd.DataFrame(build_player_defensive_actions([row], require_targets=False))
    assert "target_future_shot_10s" in out.columns
    assert out.loc[0, "target_future_shot_10s"] is None


def test_polygon_points_pair_before_validating_coordinates():
    from dax.features.player_defense import _polygon_points

    assert _polygon_points([None, 0, 10, 0, 10, 10, 0, 10]) == [(10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert _polygon_points([0, None, 10, 0, 10, 10, 0, 10]) == [(10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert _polygon_points([0, 0, 10, 0, 10, 10, 0]) == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    assert _polygon_points(["bad", 0, 10, 0, 10, 10, 0, 10]) == [(10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert _polygon_points([None, 0, 10, None, 0, 10]) == []


def test_malformed_polygon_from_bad_pairs_is_not_visible():
    out = _visibility_features([None, 0, 10, None, 0, 10], 5, 5, 5, 5)
    assert out["visibility_quality_band"] == "missing"
    assert out["local_5m_region_fully_visible"] is False


# --- Fix 1: the acting player must never be their own "nearest defender" ---
#
# StatsBomb 360 freeze frames include the acting player's own position as a
# "teammate": True entry marked "actor": True. Since actor_is_attacking is
# always False for a defensive-action row (the actor IS the defending
# team), that self-entry used to land straight in the "defenders" candidate
# pool, giving nearest_defender_distance (and defender_centroid_x/y,
# defender_spread, visible_defender_count, and everything else built from
# the same pool) a spurious near-zero self-distance in most rows. There is
# no player_id on freeze-frame entries in this codebase's 360 data model
# (defender-slots are inherently anonymous), so the acting-player identity
# check is against the "actor": True marker instead.

def test_nearest_defender_distance_excludes_actors_own_freeze_frame_entry():
    row = _targeted_row(freeze_frame=[
        {"teammate": True, "actor": True, "location": [70, 40]},  # the acting player's own entry
    ])
    out = pd.DataFrame(build_player_defensive_actions([row]))
    assert pd.isna(out.loc[0, "nearest_defender_distance"])
    assert out.loc[0, "visible_defender_count"] == 0


def test_nearest_defender_distance_uses_genuine_defender_not_self():
    row = _targeted_row(freeze_frame=[
        {"teammate": True, "actor": True, "location": [70, 40]},   # self -- must be excluded
        {"teammate": True, "actor": False, "location": [75, 40]},  # genuine teammate/defender, 5m away
        {"teammate": False, "actor": False, "location": [90, 40]},  # attacker, unaffected by the fix
    ])
    out = pd.DataFrame(build_player_defensive_actions([row]))
    assert math.isclose(out.loc[0, "nearest_defender_distance"], 5.0)
    assert out.loc[0, "visible_defender_count"] == 1
    # defender_centroid_x/y and defender_spread share the same pool -- must
    # reflect only the genuine defender, not an average with the actor's
    # own (zero-distance) position.
    assert math.isclose(out.loc[0, "defender_centroid_x"], 75.0)
    assert math.isclose(out.loc[0, "defender_centroid_y"], 40.0)
    assert math.isclose(out.loc[0, "defender_spread"], 0.0)
    assert math.isclose(out.loc[0, "nearest_attacker_distance"], 20.0)


def test_freeze_frame_entries_without_an_actor_marker_are_unaffected():
    # Entries with no "actor" key at all (the norm in older fixtures/tests
    # that predate this fix) must not be excluded -- only an explicit
    # "actor": True marks the acting player's own entry.
    row = _targeted_row(freeze_frame=[
        {"teammate": True, "location": [72, 40]},
    ])
    out = pd.DataFrame(build_player_defensive_actions([row]))
    assert math.isclose(out.loc[0, "nearest_defender_distance"], 2.0)


def test_nearest_defender_distance_distribution_is_sane_on_real_dataset():
    path = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
    if not path.exists():
        pytest.skip("Built player_defensive_actions.parquet is not present in this environment.")
    df = pd.read_parquet(path)
    near_zero_fraction = (df["nearest_defender_distance"] < 0.01).mean()
    # Verified before the fix: 73.6% of rows were < 0.01m (median ~2e-6m) --
    # the actor measuring distance to itself. nearest_attacker_distance,
    # unaffected by the bug, was 20.0% -- used here as a rough sanity
    # ceiling for what "genuinely close" looks like without self-reference.
    assert near_zero_fraction < 0.25


# --- Fix 2: no column should be a silent duplicate of another -------------

def _find_duplicate_column_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    duplicates = []
    columns = list(df.columns)
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            if df[col_a].equals(df[col_b]):
                duplicates.append((col_a, col_b))
    return duplicates


def test_no_duplicate_columns_on_real_player_defensive_actions_dataset():
    path = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
    if not path.exists():
        pytest.skip("Built player_defensive_actions.parquet is not present in this environment.")
    df = pd.read_parquet(path)
    known_exceptions = {
        # Deliberate: kept for the modeling/analysis pipeline (see
        # build_player_dataset.py) even though byte-identical to ball_x/ball_y.
        frozenset({"action_x", "ball_x"}),
        frozenset({"action_y", "ball_y"}),
        # Pre-existing duplicates found by this generic scan but NOT part of
        # the verified fix this test was added for -- flagged, not dropped,
        # since dropping them wasn't requested/verified against consumers.
        # has_360/freeze_frame_roles_known: both happen to be constant-True
        # for this table (build_player_defensive_actions only emits
        # has_360=True rows, and freeze_frame_roles_known is True whenever
        # actor_is_attacking is known) -- a coincidence of current filtering,
        # not a structural guarantee like the pairs above.
        frozenset({"has_360", "freeze_frame_roles_known"}),
        # team/actor_team: event_context.add_event_context sets
        # actor_team = df.get("team", ...), i.e. a literal copy.
        frozenset({"team", "actor_team"}),
        # action_changed_possession/action_ended_possession: player_defense.py
        # falls back to action_changed_possession's value whenever an
        # upstream "action_ended_possession" field is absent, which it
        # currently always is.
        frozenset({"action_changed_possession", "action_ended_possession"}),
    }
    duplicates = _find_duplicate_column_pairs(df)
    unexpected = [pair for pair in duplicates if frozenset(pair) not in known_exceptions]
    assert unexpected == [], f"Unexpected duplicate columns found: {unexpected}"
