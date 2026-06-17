import pandas as pd

from dax.features.event_context import add_event_context


def _base_events(action_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": 1,
                "period": 1,
                "index": 1,
                "possession": 1,
                "team": "A",
                "possession_team": "A",
                "home_team": "A",
                "away_team": "B",
                "event_type": "Pass",
            },
            {
                "match_id": 1,
                "period": 1,
                "index": 2,
                "possession": 1,
                "team": "B",
                "possession_team": "A",
                "home_team": "A",
                "away_team": "B",
                "event_type": action_type,
            },
        ]
    )


def test_defensive_action_semantics_known_when_context_resolved():
    for action_type in ["Pressure", "Clearance", "Block", "Duel", "50/50", "Foul Committed"]:
        row = add_event_context(_base_events(action_type)).iloc[1]
        assert row["event_semantics_known"] is True or bool(row["event_semantics_known"])
        assert row["attacking_team_before_action"] == "A"
        assert row["defending_team_before_action"] == "B"
        assert bool(row["actor_was_defending"])
        assert bool(row["action_was_under_opponent_possession"])
        assert row["attacking_team_before_action"] != row["defending_team_before_action"]


def test_recovery_and_interception_starting_possession_use_prior_context_and_win_possession():
    for action_type in ["Ball Recovery", "Interception"]:
        events = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "period": 1,
                    "index": 1,
                    "possession": 1,
                    "team": "A",
                    "possession_team": "A",
                    "home_team": "A",
                    "away_team": "B",
                    "event_type": "Pass",
                },
                {
                    "match_id": 1,
                    "period": 1,
                    "index": 2,
                    "possession": 2,
                    "team": "B",
                    "possession_team": "B",
                    "home_team": "A",
                    "away_team": "B",
                    "event_type": action_type,
                },
            ]
        )
        row = add_event_context(events).iloc[1]
        assert row["attacking_team_before_action"] == "A"
        assert row["defending_team_before_action"] == "B"
        assert bool(row["actor_was_defending"])
        assert bool(row["action_won_possession"])
        assert bool(row["action_ended_possession"])
        assert row["attacking_team_before_action"] != row["defending_team_before_action"]
        assert bool(row["event_semantics_known"])


def test_semantics_unknown_when_team_context_unresolved():
    events = pd.DataFrame(
        [
            {
                "match_id": 1,
                "period": 1,
                "index": 1,
                "possession": 1,
                "team": "B",
                "possession_team": "A",
                "event_type": "Pressure",
            }
        ]
    )
    row = add_event_context(events).iloc[0]
    assert not bool(row["event_semantics_known"])
