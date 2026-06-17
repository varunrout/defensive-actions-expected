import math

import pandas as pd
import pytest

from dax.features.player_defense import (
    _freeze_frame_points,
    _support_features,
    _visibility_features,
    build_player_defensive_actions,
)

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
