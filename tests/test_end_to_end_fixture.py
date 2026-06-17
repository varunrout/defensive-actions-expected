import pandas as pd

from dax.features.event_context import add_event_context, validate_event_context
from dax.features.phase_segmentation import label_defensive_phases
from dax.features.player_defense import build_player_defensive_actions
from dax.targets.short_horizon import add_future_shot_target, add_future_xg_target


def test_fixture_full_chain_produces_non_degenerate_player_targets():
    events = pd.DataFrame([
        {"match_id": 1, "period": 1, "index": 1, "minute": 0, "second": 0, "possession": 1, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pass", "id": "e1", "location": [80, 40], "ball_x": 80, "ball_y": 40, "has_360": True, "freeze_frame": [], "visible_area": [0, 0, 120, 0, 120, 80, 0, 80]},
        {"match_id": 1, "period": 1, "index": 2, "minute": 0, "second": 2, "possession": 1, "team": "Team B", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Pressure", "id": "e2", "player_id": 9, "player": "Defender", "position": "Centre Back", "location": [90, 42], "ball_x": 90, "ball_y": 42, "has_360": True, "freeze_frame": [{"teammate": True, "location": [88, 41]}, {"teammate": False, "location": [96, 40]}], "visible_area": [0, 0, 120, 0, 120, 80, 0, 80]},
        {"match_id": 1, "period": 1, "index": 3, "minute": 0, "second": 6, "possession": 1, "team": "Team A", "possession_team": "Team A", "team_in_possession": "Team A", "home_team": "Team A", "away_team": "Team B", "event_type": "Shot", "id": "e3", "location": [108, 40], "ball_x": 108, "ball_y": 40, "shot_statsbomb_xg": 0.25, "has_360": True, "freeze_frame": [], "visible_area": [0, 0, 120, 0, 120, 80, 0, 80]},
        {"match_id": 1, "period": 1, "index": 4, "minute": 0, "second": 20, "possession": 2, "team": "Team B", "possession_team": "Team B", "team_in_possession": "Team B", "home_team": "Team A", "away_team": "Team B", "event_type": "Duel", "id": "e4", "player_id": 10, "player": "Midfielder", "position": "Midfield", "location": [60, 44], "ball_x": 60, "ball_y": 44, "has_360": True, "freeze_frame": [{"teammate": True, "location": [59, 45]}], "visible_area": [0, 0, 120, 0, 120, 80, 0, 80]},
    ])
    context = add_event_context(events)
    assert validate_event_context(context) == []
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
