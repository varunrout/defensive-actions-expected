import pandas as pd
from dax.coach_analysis.zones import add_pitch_zones, pitch_zone
from dax.coach_analysis.populations import box_defence_population, apply_visibility_filter
from dax.coach_analysis.sequences import construct_sequences
from dax.coach_analysis.loaders import validate_oof_alignment
from dax.coach_analysis.comparisons import competition_comparison
from dax.coach_analysis.metrics import add_suppression
from dax.coach_analysis.representative_events import select_representative_events


def sample():
    return pd.DataFrame({
        'match_id':[1,1,1,2], 'event_id':[10,11,12,20], 'possession':[5,5,5,6],
        'minute':[1,2,3,4], 'x':[110,85,115,50], 'y':[40,10,20,40],
        'position_group':['Centre Back','Full Back','CB','Midfield'],
        'competition':['World Cup','World Cup','Euros','Euros'],
        'observed_future_shot':[0,1,1,0], 'expected_shot_probability':[.3,.2,.4,.1],
        'observed_future_xg':[0,.05,.2,0], 'expected_future_xg':[.1,.03,.1,.01],
        'visible_teammates':[3,None,2,None]
    })

def test_pitch_zone_classification():
    assert pitch_zone(110,40) == 'penalty box'
    assert add_pitch_zones(sample()).loc[0,'coach_box_lane'] == 'central'

def test_box_defence_population_rules():
    assert len(box_defence_population(sample(), centre_backs_only=True)) == 2

def test_sequence_construction():
    out=construct_sequences(sample())
    assert out.loc[out.event_id.eq(11),'coach_is_repeated_defensive_action'].item() is True

def test_oof_alignment_duplicate_event_handling():
    preds=pd.DataFrame({'match_id':[1,1], 'event_id':[10,10], 'p':[.1,.2]})
    assert validate_oof_alignment(sample(), preds)['duplicate_predictions'] == 1

def test_competition_comparison_denominators():
    out=competition_comparison(sample())
    assert set(out['competition']) == {'World Cup','Euros'}
    assert out['actions'].sum() == 4

def test_suppression_sign_convention():
    out=add_suppression(sample())
    assert out.loc[0,'shot_suppression'] == .3
    assert out.loc[2,'xg_suppression'] < 0

def test_representative_event_selection():
    out=select_representative_events(sample(), n=2)
    assert len(out) == 2
    assert 'reason_selected_for_review' in out.columns

def test_visibility_filtering():
    assert len(apply_visibility_filter(sample(), reliable_only=True)) == 2

def test_empty_population_handling():
    assert box_defence_population(sample().iloc[0:0]).empty
