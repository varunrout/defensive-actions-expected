import pandas as pd

from dax.analysis.notebook_aggregation import (
    PLAYER_NOTEBOOK_AGGREGATION_FEATURES,
    TEAM_NOTEBOOK_AGGREGATION_FEATURES,
)
from dax.features.player_defense import build_player_defensive_actions
from dax.models.baseline_logistic import default_variant_specs, resolve_columns as resolve_logistic_columns
from dax.models.baseline_regression import default_regression_specs, resolve_columns as resolve_regression_columns


def _fixture_player_dataset() -> pd.DataFrame:
    rows = build_player_defensive_actions(
        [
            {
                "match_id": 1,
                "period": 1,
                "possession": 1,
                "index": 2,
                "minute": 0,
                "second": 2,
                "id": "pressure",
                "event_type": "Pressure",
                "player_id": 9,
                "player": "Defender",
                "position": "Centre Back",
                "team": "B",
                "actor_team": "B",
                "attacking_team_before_action": "A",
                "defending_team_before_action": "B",
                "location": [70, 40],
                "ball_x": 70,
                "ball_y": 40,
                "has_360": True,
                "freeze_frame_count": 3,
                "freeze_frame": [
                    {"teammate": True, "location": [60, 40]},
                    {"teammate": True, "location": [80, 40]},
                    {"teammate": True, "location": [100, 40]},
                ],
                "visible_area": [0, 0, 120, 0, 120, 80, 0, 80],
                "phase_label": "high_press",
                "play_pattern": "Regular Play",
                "counterpress": False,
                "action_changed_possession": False,
                "action_ended_possession": False,
                "action_won_possession": False,
                "action_retained_defensive_team_control": True,
                "action_was_under_opponent_possession": True,
                "target_future_shot_10s": 1,
                "target_future_xg_10s": 0.25,
            }
        ]
    )
    return pd.DataFrame(rows)


def test_default_model_specs_exist_in_generated_player_dataset():
    df = _fixture_player_dataset()
    for spec in default_variant_specs():
        missing = [*spec.categorical, *spec.numeric]
        missing = [feature for feature in missing if feature not in df.columns]
        assert missing == []
        resolve_logistic_columns(df, spec)
    for spec in default_regression_specs():
        missing = [*spec.categorical, *spec.numeric]
        missing = [feature for feature in missing if feature not in df.columns]
        assert missing == []
        resolve_regression_columns(df, spec)


def test_default_model_specs_do_not_duplicate_features():
    specs = [*default_variant_specs(), *default_regression_specs()]
    for spec in specs:
        numeric = list(spec.numeric)
        categorical = list(spec.categorical)
        assert numeric == list(dict.fromkeys(numeric)), spec.name
        assert categorical == list(dict.fromkeys(categorical)), spec.name
        assert set(numeric).isdisjoint(categorical), spec.name


def test_notebook_aggregation_features_exist_in_generated_player_dataset():
    df = _fixture_player_dataset()
    referenced_features = {
        source
        for _, source, _ in [
            *TEAM_NOTEBOOK_AGGREGATION_FEATURES,
            *PLAYER_NOTEBOOK_AGGREGATION_FEATURES,
        ]
    }

    missing = sorted(referenced_features - set(df.columns))
    assert missing == []


def test_resolve_columns_fails_for_missing_required_features():
    df = _fixture_player_dataset().drop(columns=["nearest_attacker_distance"])
    spec = default_variant_specs()[2]
    try:
        resolve_logistic_columns(df, spec)
    except ValueError as exc:
        assert "nearest_attacker_distance" in str(exc)
    else:
        raise AssertionError("resolve_columns should fail when required features are missing")
