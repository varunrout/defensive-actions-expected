import pandas as pd

from dax.features.event_context import add_event_context, validate_event_context
from dax.features.phase_segmentation import label_defensive_phases
from dax.features.player_defense import build_player_defensive_actions
from dax.targets.short_horizon import add_future_shot_target, add_future_xg_target

FULL = [0, 0, 120, 0, 120, 80, 0, 80]
SMALL = [40, 20, 80, 20, 80, 60, 40, 60]


def test_fixture_full_chain_produces_non_degenerate_player_targets():
    events = pd.DataFrame(
        [
            {"match_id": 1, "period": 1, "index": 1, "minute": 0, "second": 0, "possession": 1, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e1", "location": [80, 40], "ball_x": 80, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 2, "minute": 0, "second": 2, "possession": 1, "team": "Team B", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pressure", "id": "pressure", "player_id": 9, "player": "Defender", "position": "Centre Back", "location": [70, 40], "ball_x": 70, "ball_y": 40, "has_360": True, "freeze_frame": [{"teammate": True, "location": [60, 40]}, {"teammate": True, "location": [80, 40]}, {"teammate": True, "location": [100, 40]}], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 3, "minute": 0, "second": 6, "possession": 1, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Shot", "id": "shot", "location": [108, 40], "ball_x": 108, "ball_y": 40, "shot_statsbomb_xg": 0.25, "has_360": True, "freeze_frame": [], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 4, "minute": 0, "second": 20, "possession": 2, "team": "Team B", "possession_team": "Team B", "team_in_possession": "Team B", "home_team": "Team A", "away_team": "Team B", "event_type": "Ball Recovery", "id": "recovery", "player_id": 10, "player": "Recoverer", "position": "Midfield", "location": [62, 42], "ball_x": 62, "ball_y": 42, "has_360": True, "freeze_frame": [], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 5, "minute": 0, "second": 30, "possession": 3, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e5", "location": [50, 40], "ball_x": 50, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 6, "minute": 0, "second": 35, "possession": 4, "team": "Team B", "possession_team": "Team B", "team_in_possession": "Team B", "home_team": "Team A", "away_team": "Team B", "event_type": "Interception", "id": "interception", "player_id": 11, "player": "Interceptor", "position": "Full Back", "location": [58, 38], "ball_x": 58, "ball_y": 38, "has_360": True, "freeze_frame": [], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 7, "minute": 0, "second": 50, "possession": 5, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e7", "location": [55, 40], "ball_x": 55, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": FULL},
            {"match_id": 1, "period": 1, "index": 8, "minute": 0, "second": 52, "possession": 5, "team": None, "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pressure", "id": "unknown", "player_id": 12, "player": "Unknown Context", "position": "Midfield", "location": [42, 40], "ball_x": 42, "ball_y": 40, "has_360": True, "freeze_frame": [{"teammate": True, "location": [43, 40]}, {"teammate": False, "location": [44, 40]}], "visible_area": SMALL},
            {"match_id": 1, "period": 1, "index": 9, "minute": 1, "second": 0, "possession": 5, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Duel", "id": "duel", "player_id": 13, "player": "Duel Player", "position": "Midfield", "location": [60, 44], "ball_x": 60, "ball_y": 44, "has_360": True, "freeze_frame": [{"teammate": True, "location": [59, 45]}], "visible_area": FULL},
        ]
    )
    context = add_event_context(events)
    assert validate_event_context(context) == []

    for event_id in ["recovery", "interception"]:
        row = context.loc[context["id"] == event_id].iloc[0]
        assert row["attacking_team_before_action"] == "Team A"
        assert row["defending_team_before_action"] == "Team B"
        assert bool(row["actor_was_defending"])
        assert bool(row["action_won_possession"])
        assert row["attacking_team_before_action"] != row["defending_team_before_action"]

    phased = pd.DataFrame(label_defensive_phases(context.to_dict("records")))
    targeted = add_future_xg_target(add_future_shot_target(phased))
    rows = build_player_defensive_actions(targeted.to_dict("records"), only_with_360=True, defensive_only=True)
    players = pd.DataFrame(rows)

    assert {"target_future_shot_10s", "target_future_xg_10s"}.issubset(players.columns)
    assert players["target_future_shot_10s"].notna().all()
    assert players["target_future_xg_10s"].notna().all()
    assert players["target_future_shot_10s"].sum() > 0
    assert players["target_future_xg_10s"].sum() > 0

    known = players.dropna(subset=["attacking_team", "defending_team"])
    assert not (known["attacking_team"] == known["defending_team"]).any()
    assert "second_ball" not in players["phase_label"].tolist()

    pressure = players.loc[players["event_id"] == "pressure"].iloc[0]
    assert pressure["defenders_between_ball_and_attacking_goal"] == 2
    assert bool(pressure["local_5m_region_fully_visible"])
    assert bool(pressure["local_10m_region_fully_visible"])

    unknown = players.loc[players["event_id"] == "unknown"].iloc[0]
    assert unknown["freeze_frame_roles_known"] is False or not bool(unknown["freeze_frame_roles_known"])
    assert pd.isna(unknown["visible_attacker_count"])
    assert pd.isna(unknown["visible_defender_count"])
    assert unknown["local_5m_region_fully_visible"] is False or not bool(unknown["local_5m_region_fully_visible"])
