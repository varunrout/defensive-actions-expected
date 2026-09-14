import math

import pandas as pd

from dax.features.passive_defense import build_passive_defense_rows

FULL = [0, 0, 120, 0, 120, 80, 0, 80]


def _pass_row(**updates):
    row = {
        "match_id": 1,
        "period": 1,
        "possession": 1,
        "possession_sequence_id": 1,
        "index": 1,
        "minute": 0,
        "second": 1,
        "id": "e1",
        "event_type": "Pass",
        "team": "A",
        "actor_team": "A",
        "defending_team_before_action": "B",
        "location": [60.0, 40.0],
        "ball_x": 60.0,
        "ball_y": 40.0,
        "has_360": True,
        "freeze_frame": [
            {"teammate": True, "location": [65.0, 40.0]},
            {"teammate": False, "location": [55.0, 40.0]},
            {"teammate": False, "location": [50.0, 30.0]},
        ],
        "visible_area": FULL,
        "phase_label": "settled_mid_block_proxy",
    }
    row.update(updates)
    return row


def test_well_formed_event_builds_one_row_per_defender_slot():
    rows = build_passive_defense_rows([_pass_row()])
    assert len(rows) == 2
    assert {r["defender_slot_index"] for r in rows} == {0, 1}
    for row in rows:
        assert row["match_id"] == 1
        assert row["period"] == 1
        assert row["possession_sequence_id"] == 1
        assert row["event_id"] == "e1"
        assert row["on_ball_event_type"] == "Pass"
        assert row["on_ball_team"] == "A"
        assert row["defending_team"] == "B"
        assert row["phase_label"] == "settled_mid_block_proxy"
        assert "distance_to_attacking_goal" in row
        assert "visibility_quality_band" in row


def test_marking_tightness_and_engagement_distance_against_known_fixture():
    rows = build_passive_defense_rows([_pass_row()])
    first = next(r for r in rows if r["defender_slot_index"] == 0)
    second = next(r for r in rows if r["defender_slot_index"] == 1)

    # defender (55, 40) vs nearest attacker (65, 40) -> distance 10
    assert math.isclose(first["marking_tightness"], 10.0)
    assert first["is_goal_side_of_nearest_attacker"] is True  # 55 < 65

    # defender (50, 30) vs nearest attacker (65, 40) -> distance sqrt(15^2+10^2)
    assert math.isclose(second["marking_tightness"], math.sqrt(15**2 + 10**2))
    assert second["is_goal_side_of_nearest_attacker"] is True  # 50 < 65

    # engagement distance to carrier (60, 40)
    assert math.isclose(first["engagement_distance_to_carrier"], 5.0)
    assert math.isclose(second["engagement_distance_to_carrier"], math.sqrt(10**2 + 10**2))


def test_zone_defensive_value_decays_with_distance_to_defending_goal():
    rows = build_passive_defense_rows([_pass_row()])
    first = next(r for r in rows if r["defender_slot_index"] == 0)  # (55, 40): distance to (0,40) = 55
    second = next(r for r in rows if r["defender_slot_index"] == 1)  # (50, 30): further from goal than 55? sqrt(50^2+10^2)
    assert math.isclose(first["zone_defensive_value"], 1.0 / (1.0 + 55.0))
    assert 0.0 < second["zone_defensive_value"] < 1.0
    assert first["defender_zone"] is not None


def test_non_on_ball_event_types_are_skipped():
    row = _pass_row(event_type="Pressure")
    assert build_passive_defense_rows([row]) == []


def test_events_without_360_are_skipped_by_default():
    row = _pass_row(has_360=False)
    assert build_passive_defense_rows([row]) == []


def test_missing_freeze_frame_is_handled_without_crashing():
    row = _pass_row(freeze_frame=None)
    assert build_passive_defense_rows([row]) == []


def test_empty_freeze_frame_produces_no_defender_rows_but_does_not_crash():
    row = _pass_row(freeze_frame=[])
    assert build_passive_defense_rows([row]) == []


def test_missing_location_is_handled_without_crashing():
    row = _pass_row(location=None)
    assert build_passive_defense_rows([row]) == []


def test_no_attackers_in_frame_yields_null_marking_features():
    row = _pass_row(freeze_frame=[{"teammate": False, "location": [55.0, 40.0]}])
    rows = build_passive_defense_rows([row])
    assert len(rows) == 1
    assert rows[0]["marking_tightness"] is None
    assert rows[0]["is_goal_side_of_nearest_attacker"] is None


def test_event_order_in_possession_increments_within_group():
    row1 = _pass_row(id="e1", index=1, second=1)
    row2 = _pass_row(id="e2", index=2, second=3)
    rows = pd.DataFrame(build_passive_defense_rows([row1, row2]))
    orders = rows.groupby("event_id")["event_order_in_possession"].first()
    assert orders["e1"] == 0
    assert orders["e2"] == 1


# --- Phase 3: lane occlusion + option ranking -----------------------------
#
# Fixture geometry (carrier at (60, 40)):
#   A1 = (100, 40) -> distance to attacking goal (120, 40) = 20
#   A2 = (100, 60) -> distance to attacking goal (120, 40) = sqrt(20^2+20^2) ~ 28.284
# Neither candidate has a defender within OPTION_DENSITY_RADIUS_M (10m), so
# openness is 1 for both and ranking is driven purely by goal proximity:
# A1 (closer to goal) must rank above A2.
#
#   D0 = (80, 40) sits exactly on the carrier->A1 lane (distance 0) and is
#        also the closer of the two defenders to the carrier->A2 lane.
#   D1 = (80, 20) sits 20m off the carrier->A1 lane and is the farther
#        defender from the carrier->A2 lane.
# So D0 is the closest covering defender for both threats (overload_score=2)
# and D1 for neither (overload_score=0).

def _lane_fixture_row(**updates):
    row = {
        "match_id": 1,
        "period": 1,
        "possession": 1,
        "possession_sequence_id": 1,
        "index": 1,
        "minute": 0,
        "second": 1,
        "id": "lane1",
        "event_type": "Pass",
        "team": "A",
        "actor_team": "A",
        "defending_team_before_action": "B",
        "location": [60.0, 40.0],
        "ball_x": 60.0,
        "ball_y": 40.0,
        "has_360": True,
        "freeze_frame": [
            {"teammate": True, "location": [100.0, 40.0]},  # A1
            {"teammate": True, "location": [100.0, 60.0]},  # A2
            {"teammate": False, "location": [80.0, 40.0]},  # D0, slot 0
            {"teammate": False, "location": [80.0, 20.0]},  # D1, slot 1
        ],
        "visible_area": FULL,
        "phase_label": "settled_mid_block_proxy",
    }
    row.update(updates)
    return row


def test_option_candidates_are_ranked_by_goal_proximity():
    rows = build_passive_defense_rows([_lane_fixture_row()])
    row = rows[0]
    assert row["top_option_1_target_x"] == 100.0
    assert row["top_option_1_target_y"] == 40.0
    assert row["top_option_2_target_x"] == 100.0
    assert row["top_option_2_target_y"] == 60.0
    assert row["top_option_1_threat_score"] > row["top_option_2_threat_score"]
    expected_score_1 = (1.0 / (1.0 + 20.0)) * 1.0
    assert math.isclose(row["top_option_1_threat_score"], expected_score_1)
    # only two attackers in the frame -> no third option
    assert row["top_option_3_target_x"] is None
    assert row["top_option_3_threat_score"] is None
    assert row["lane_screening_score_option_3"] is None


def test_lane_screening_score_high_on_lane_low_off_lane():
    rows = build_passive_defense_rows([_lane_fixture_row()])
    d0 = next(r for r in rows if r["defender_slot_index"] == 0)
    d1 = next(r for r in rows if r["defender_slot_index"] == 1)

    # D0 sits exactly on the carrier -> option-1 lane.
    assert math.isclose(d0["lane_screening_score_option_1"], 1.0)
    assert d0["screens_top_option"] is True

    # D1 is 20m off the carrier -> option-1 lane.
    assert math.isclose(d1["lane_screening_score_option_1"], 1.0 / (1.0 + 20.0))
    assert d1["screens_top_option"] is False


def test_overload_score_counts_closest_covering_defender_per_threat():
    rows = build_passive_defense_rows([_lane_fixture_row()])
    d0 = next(r for r in rows if r["defender_slot_index"] == 0)
    d1 = next(r for r in rows if r["defender_slot_index"] == 1)

    # D0 is the closer defender to both the option-1 and option-2 lanes.
    assert d0["overload_score"] == 2
    assert d1["overload_score"] == 0


def test_lane_occlusion_features_with_no_attackers_are_null_and_overload_zero():
    row = _lane_fixture_row(freeze_frame=[
        {"teammate": False, "location": [80.0, 40.0]},
    ])
    rows = build_passive_defense_rows([row])
    assert len(rows) == 1
    out = rows[0]
    for rank in (1, 2, 3):
        assert out[f"top_option_{rank}_target_x"] is None
        assert out[f"top_option_{rank}_threat_score"] is None
        assert out[f"lane_screening_score_option_{rank}"] is None
    assert out["screens_top_option"] is None
    assert out["overload_score"] == 0
