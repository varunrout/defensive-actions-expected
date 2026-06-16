import pytest
from dax.features.player_defense import _goal_metrics, DEFENSIVE_ACTION_TYPES
from dax.models.attacking_threat import GridThreatModel
from dax.models.baseline_logistic import VariantSpec, default_variant_specs, validate_no_future_features


def test_attacking_goal_geometry():
    assert _goal_metrics(110,40)['distance_to_attacking_goal'] < _goal_metrics(10,40)['distance_to_attacking_goal']
    assert _goal_metrics(10,40)['distance_to_defending_goal'] < _goal_metrics(110,40)['distance_to_defending_goal']
    assert _goal_metrics(100,40)['angle_to_attacking_goal'] != _goal_metrics(100,75)['angle_to_attacking_goal']

def test_shield_excluded():
    assert 'Shield' not in DEFENSIVE_ACTION_TYPES

def test_default_specs_no_future_features():
    for spec in default_variant_specs(): validate_no_future_features(spec)

def test_future_feature_denylist_fails():
    with pytest.raises(ValueError): validate_no_future_features(VariantSpec('bad',[],['possession_duration_total']))

def test_grid_fit_resets_state():
    m=GridThreatModel(smoothing=0)
    m.fit([{'ball_x':10,'ball_y':10,'target_shot_in_10s':1}])
    assert m.predict_point(10,10) == 1
    m.fit([{'ball_x':10,'ball_y':10,'target_shot_in_10s':0}])
    assert m.predict_point(10,10) == 0
