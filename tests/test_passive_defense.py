import math
from pathlib import Path

import pandas as pd
import pytest

from dax.features.add_missingness_flags import (
    EXPECTED_FALSE_COUNTS,
    add_passive_defense_flags,
    add_player_defensive_actions_flags,
)
from dax.features.passive_defense import build_passive_defense_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
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


# --- Ball-relative option features -------------------------------------
#
# top_option_n_target_x/y are absolute pitch coordinates, which is why they
# correlated strongly with ball_x/ball_y and with each other (see
# reports/eda/CORRELATION_ANALYSIS.json). The ball-relative dx/dy/distance/
# angle features carry the same information without that confound. Fixture:
# ball at (60, 40); A1=(100, 40) -> dx=40, dy=0; A2=(100, 60) -> dx=40, dy=20.

def test_ball_relative_option_features_match_absolute_targets_minus_ball():
    rows = build_passive_defense_rows([_lane_fixture_row()])
    row = rows[0]

    assert math.isclose(row["top_option_1_dx"], 40.0)
    assert math.isclose(row["top_option_1_dy"], 0.0)
    assert math.isclose(row["top_option_1_distance_from_ball"], 40.0)
    assert math.isclose(row["top_option_1_angle_from_ball"], 0.0, abs_tol=1e-9)

    assert math.isclose(row["top_option_2_dx"], 40.0)
    assert math.isclose(row["top_option_2_dy"], 20.0)
    assert math.isclose(row["top_option_2_distance_from_ball"], math.hypot(40.0, 20.0))
    assert math.isclose(row["top_option_2_angle_from_ball"], math.atan2(20.0, 40.0))

    # only two attackers in the frame -> no third option, ball-relative
    # features for rank 3 stay null just like the absolute target columns.
    assert row["top_option_3_dx"] is None
    assert row["top_option_3_dy"] is None
    assert row["top_option_3_distance_from_ball"] is None
    assert row["top_option_3_angle_from_ball"] is None


def test_ball_relative_option_features_null_when_no_attackers():
    row = _lane_fixture_row(freeze_frame=[
        {"teammate": False, "location": [80.0, 40.0]},
    ])
    rows = build_passive_defense_rows([row])
    out = rows[0]
    for rank in (1, 2, 3):
        assert out[f"top_option_{rank}_dx"] is None
        assert out[f"top_option_{rank}_dy"] is None
        assert out[f"top_option_{rank}_distance_from_ball"] is None
        assert out[f"top_option_{rank}_angle_from_ball"] is None


# --- Functional role labelling ---------------------------------------------
#
# Roles are inferred purely from a defender-slot's depth/width RANK relative
# to its own team's other visible defenders in the same frame -- never
# absolute pitch coordinates. See _functional_roles' docstring for the
# depth_ratio / lateral_ratio formulas and thresholds.

def test_functional_role_back_four_shape():
    # depth (x): two deepest (0, 0), one mid (15), one most advanced (30)
    # lateral (y): two central (45, 35, both near mean), one wide (70)
    row = _pass_row(id="back_four", freeze_frame=[
        {"teammate": False, "location": [0.0, 45.0]},   # deep + central -> last_line
        {"teammate": False, "location": [0.0, 70.0]},   # deep + wide -> wide_cover
        {"teammate": False, "location": [30.0, 35.0]},  # advanced + central -> central_screen
        {"teammate": False, "location": [15.0, 10.0]},  # mid depth, wide -> mid_block (measured, not extreme on depth)
    ])
    rows = build_passive_defense_rows([row])
    roles = {r["defender_slot_index"]: r["defender_functional_role"] for r in rows}
    assert roles[0] == "last_line"
    assert roles[1] == "wide_cover"
    assert roles[2] == "central_screen"
    assert roles[3] == "mid_block"


def test_functional_role_compact_block_is_mid_block_when_depth_is_degenerate():
    # All three defenders share the same x -- no depth signal at all, so
    # nobody can be "deep" or "advanced" relative to the others. n >= 2, so
    # this is a measured-and-ordinary case (mid_block), not "unclassified"
    # -- unclassified is reserved exclusively for n < 2.
    row = _pass_row(id="compact_block", freeze_frame=[
        {"teammate": False, "location": [10.0, 38.0]},
        {"teammate": False, "location": [10.0, 40.0]},
        {"teammate": False, "location": [10.0, 42.0]},
    ])
    rows = build_passive_defense_rows([row])
    roles = {r["defender_slot_index"]: r["defender_functional_role"] for r in rows}
    assert set(roles.values()) == {"mid_block"}


def test_functional_role_stretched_transition_shape():
    row = _pass_row(id="transition", freeze_frame=[
        {"teammate": False, "location": [5.0, 42.0]},   # deep + central -> last_line
        {"teammate": False, "location": [55.0, 5.0]},   # advanced + wide -> advanced_wide
        {"teammate": False, "location": [50.0, 38.0]},  # advanced + central -> central_screen
        {"teammate": False, "location": [8.0, 75.0]},   # deep + wide -> wide_cover
    ])
    rows = build_passive_defense_rows([row])
    roles = {r["defender_slot_index"]: r["defender_functional_role"] for r in rows}
    assert roles[0] == "last_line"
    assert roles[1] == "advanced_wide"
    assert roles[2] == "central_screen"
    assert roles[3] == "wide_cover"


def test_functional_role_advanced_wide_quadrant():
    # The fourth corner: advanced AND wide -- a high, wide-pressing
    # defender (overlapping full-back / winger tracking back high).
    row = _pass_row(id="advanced_wide", freeze_frame=[
        {"teammate": False, "location": [0.0, 40.0]},   # deep anchor, gives the group depth spread
        {"teammate": False, "location": [50.0, 5.0]},    # advanced + wide -> advanced_wide
    ])
    rows = build_passive_defense_rows([row])
    roles = {r["defender_slot_index"]: r["defender_functional_role"] for r in rows}
    assert roles[1] == "advanced_wide"


def test_functional_role_unclassified_with_fewer_than_two_defenders():
    # unclassified is reserved exclusively for n < 2 (nothing to be
    # "relative to") -- never used for a measured-but-ordinary row.
    row = _pass_row(id="lone_defender", freeze_frame=[
        {"teammate": False, "location": [10.0, 40.0]},
    ])
    rows = build_passive_defense_rows([row])
    assert rows[0]["defender_functional_role"] == "unclassified"


# --- Outcome columns: per-row, not per-possession ---------------------------

def test_target_columns_are_computed_per_row_not_broadcast_across_possession():
    # Same possession, same match/period. A shot happens at t=8s. Anchor A
    # (t=0s) has the shot inside its own 10s window; anchor B (t=20s) does
    # not -- the shot is in B's past. A broadcast-across-possession bug
    # would give both the same (1, 0.4) target instead of this asymmetry.
    anchor_a = _pass_row(id="anchor_a", index=1, second=0)
    shot_event = _pass_row(id="shot_event", index=2, second=8)
    shot_event.update({"event_type": "Shot", "has_360": False, "freeze_frame": None, "shot_statsbomb_xg": 0.4})
    anchor_b = _pass_row(id="anchor_b", index=3, second=20)

    rows = build_passive_defense_rows([anchor_a, shot_event, anchor_b])
    row_a = next(r for r in rows if r["event_id"] == "anchor_a")
    row_b = next(r for r in rows if r["event_id"] == "anchor_b")

    assert row_a["target_future_shot_10s"] == 1
    assert math.isclose(row_a["target_future_xg_10s"], 0.4)
    assert row_b["target_future_shot_10s"] == 0
    assert math.isclose(row_b["target_future_xg_10s"], 0.0)


def test_target_columns_use_full_event_stream_even_when_shot_lacks_360():
    # The shot itself is has_360=False and therefore never becomes its own
    # anchor row -- but it must still be visible to anchor_a's horizon scan.
    anchor_a = _pass_row(id="anchor_a", index=1, second=0)
    shot_event = _pass_row(id="shot_event", index=2, second=5)
    shot_event.update({"event_type": "Shot", "has_360": False, "freeze_frame": None, "shot_statsbomb_xg": 0.1})

    rows = build_passive_defense_rows([anchor_a, shot_event])
    assert all(r["event_id"] != "shot_event" for r in rows)
    row_a = next(r for r in rows if r["event_id"] == "anchor_a")
    assert row_a["target_future_shot_10s"] == 1


# --- screened_option_was_avoided --------------------------------------------

def test_screened_option_was_avoided_false_when_next_event_matches_screened_option():
    anchor = _lane_fixture_row(id="anchor", index=1, second=0)
    next_event = {
        "match_id": 1, "period": 1, "index": 2, "minute": 0, "second": 1,
        "id": "next", "event_type": "Pass", "team": "A",
        "location": [99.0, 41.0],  # close to top_option_1's target (100, 40)
        "has_360": False,
    }
    rows = build_passive_defense_rows([anchor, next_event])
    d0 = next(r for r in rows if r["defender_slot_index"] == 0)
    assert d0["screened_option_was_avoided"] is False


def test_screened_option_was_avoided_true_when_next_event_goes_elsewhere():
    anchor = _lane_fixture_row(id="anchor", index=1, second=0)
    next_event = {
        "match_id": 1, "period": 1, "index": 2, "minute": 0, "second": 1,
        "id": "next", "event_type": "Pass", "team": "A",
        "location": [10.0, 10.0],  # far from every ranked option
        "has_360": False,
    }
    rows = build_passive_defense_rows([anchor, next_event])
    d0 = next(r for r in rows if r["defender_slot_index"] == 0)
    assert d0["screened_option_was_avoided"] is True


def test_screened_option_was_avoided_is_none_when_no_next_event():
    anchor = _lane_fixture_row(id="anchor", index=1, second=0)
    rows = build_passive_defense_rows([anchor])
    d0 = next(r for r in rows if r["defender_slot_index"] == 0)
    assert d0["screened_option_was_avoided"] is None


# --- Missingness flags (dax.features.add_missingness_flags) ----------------
#
# Flags are structural (freeze-frame had too few attackers, no next event
# existed, this was a possession's first event), not random missingness.
# Counts against the real built datasets are checked separately, below.

def test_add_passive_defense_flags_matches_notnull_semantics():
    df = pd.DataFrame({
        "top_option_2_threat_score": [0.1, None, 0.2],
        "top_option_3_threat_score": [None, None, 0.05],
        "screened_option_was_avoided": [True, False, None],
    })
    out = add_passive_defense_flags(df, validate=False)
    assert out["has_option_2"].tolist() == [True, False, True]
    assert out["has_option_3"].tolist() == [False, False, True]
    assert out["has_screened_outcome"].tolist() == [True, True, False]
    # existing columns are untouched
    pd.testing.assert_series_equal(out["top_option_2_threat_score"], df["top_option_2_threat_score"])


def test_add_player_defensive_actions_flags_matches_spec_semantics():
    df = pd.DataFrame({
        "visible_attacker_count": [0, 3, None],
        "visible_defender_count": [2, 0, None],
        "phase_label_prev_event": ["settled_mid_block_proxy", None, "box_defence"],
    })
    out = add_player_defensive_actions_flags(df, validate=False)
    assert out["has_visible_attacker"].tolist() == [False, True, False]
    assert out["has_visible_defender"].tolist() == [True, False, False]
    assert out["has_previous_event"].tolist() == [True, False, True]


def test_flag_validation_fails_loudly_on_count_mismatch():
    # A tiny synthetic frame will never match the real dataset's verified
    # False-counts (279 / 2181 / 2581) -- validation must raise, not
    # silently write flags that no longer mean what they're documented to.
    df = pd.DataFrame({
        "top_option_2_threat_score": [0.1, 0.2],
        "top_option_3_threat_score": [0.1, 0.2],
        "screened_option_was_avoided": [True, False],
    })
    with pytest.raises(ValueError, match="has_option_2"):
        add_passive_defense_flags(df, validate=True)


def test_flag_counts_match_verified_values_on_real_datasets():
    passive_path = REPO_ROOT / "data" / "features" / "passive_defense.parquet"
    active_path = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"
    if not passive_path.exists() or not active_path.exists():
        pytest.skip("Built feature parquet files are not present in this environment.")

    passive = add_passive_defense_flags(pd.read_parquet(passive_path))
    assert len(passive) == 1_593_181
    assert int((~passive["has_option_2"]).sum()) == EXPECTED_FALSE_COUNTS["has_option_2"] == 279
    assert int((~passive["has_option_3"]).sum()) == EXPECTED_FALSE_COUNTS["has_option_3"] == 2181
    assert int((~passive["has_screened_outcome"]).sum()) == EXPECTED_FALSE_COUNTS["has_screened_outcome"] == 2581

    active = add_player_defensive_actions_flags(pd.read_parquet(active_path))
    assert len(active) == 56_068
    assert int((~active["has_visible_attacker"]).sum()) == EXPECTED_FALSE_COUNTS["has_visible_attacker"] == 31
    assert int((~active["has_visible_defender"]).sum()) == EXPECTED_FALSE_COUNTS["has_visible_defender"] == 76
    assert int((~active["has_previous_event"]).sum()) == EXPECTED_FALSE_COUNTS["has_previous_event"] == 4534
