import pandas as pd
from dax.coach_analysis.zones import add_pitch_zones, pitch_zone
from dax.coach_analysis.populations import box_defence_population, apply_visibility_filter
from dax.coach_analysis.sequences import construct_sequences
from dax.coach_analysis.loaders import (
    select_classification_variant,
    select_regression_variant,
    select_two_part_variant,
    validate_oof_alignment,
)
from dax.coach_analysis.comparisons import competition_comparison
from dax.coach_analysis.metrics import add_suppression, bootstrap_match_level_ci, summary_table
from dax.coach_analysis.representative_events import select_representative_events
from dax.coach_analysis.labels import label_box_defence


def sample():
    return pd.DataFrame({
        'match_id':[1,1,1,2], 'event_id':[10,11,12,20], 'period':[1,1,1,1], 'possession':[5,5,5,6],
        'event_index':[10,20,30,10], 'minute':[1,2,3,4], 'second':[0,5,10,0],
        'action_x':[110,85,115,50], 'action_y':[40,10,20,40],
        'position_group':['Centre Back','Full Back','CB','Midfield'],
        'competition':['World Cup','World Cup','Euros','Euros'], 'team':['A','A','A','B'], 'attacking_team':['X','X','X','Y'],
        'event_type':['Clearance','Pressure','Block','Recovery'], 'action_family':['clearance','pressure','block','recovery'],
        'action_won_possession':[False,False,False,True], 'action_changed_possession':[False,False,False,True],
        'action_retained_defensive_team_control':[True,False,False,True],
        'coach_expected_shot_probability':[.3,.2,.4,.1], 'coach_observed_future_shot':[0,1,1,0],
        'coach_observed_future_xg':[0,.05,.2,0], 'coach_expected_future_xg_r4':[.1,.03,.1,.01],
        'coach_expected_future_xg_two_part':[.12,.02,.09,.02],
        'has_360':[True,False,True,True],
        'local_5m_region_fully_visible':[True,True,False,True],
        'local_10m_region_fully_visible':[True,True,False,True],
        'freeze_frame_roles_known':[True,True,False,True],
        'visibility_quality_band':['high','high','acceptable','low'],
    })

def test_pitch_zone_classification():
    assert pitch_zone(110,40) == 'central box'
    assert add_pitch_zones(sample()).loc[0,'coach_box_lane'] == 'central'


def test_central_box_reachable_and_zone_exclusive():
    zones = [
        pitch_zone(115, 40),
        pitch_zone(115, 20),
        pitch_zone(110, 40),
        pitch_zone(100, 40),
        pitch_zone(82, 10),
    ]
    assert 'central box' in zones
    assert len(zones) == len(set(zones))

def test_box_defence_population_rules():
    assert len(box_defence_population(sample(), centre_backs_only=True)) == 2

def test_sequence_construction():
    events = pd.DataFrame(
        {
            'match_id':[1,1,1,1],
            'period':[1,1,1,1],
            'event_index':[21,22,23,24],
            'team':['X','A','X','A'],
            'type':['Pass','Pressure','Shot','Duel'],
            'possession':[5,5,5,6],
            'minute':[2,2,2,2],
            'second':[6,7,8,9],
        }
    )
    out=construct_sequences(sample().iloc[[1]].copy(), events)
    assert out['coach_next_opposition_event_type'].iloc[0] == 'Pass'
    assert out['coach_second_opposition_event_type'].iloc[0] == 'Shot'

def test_oof_alignment_duplicate_event_handling():
    preds=pd.DataFrame({'match_id':[1,1], 'event_id':[10,10], 'p':[.1,.2]})
    assert validate_oof_alignment(sample(), preds)['duplicate_predictions'] == 1


def test_oof_variant_selection_is_explicit_and_duplicate_variant_rows_fail():
    cls = pd.DataFrame(
        {
            'match_id':[1,1],
            'event_id':[10,10],
            'fold':[0,0],
            'model_variant':['b7_full_with_360','b7_full_with_360'],
            'y_true':[0,0],
            'y_score':[0.4,0.5],
        }
    )
    try:
        select_classification_variant(cls, variant='b7_full_with_360')
        assert False, 'Expected duplicate rows to fail'
    except ValueError as exc:
        assert 'duplicate rows' in str(exc)


def test_regression_and_two_part_variant_selection():
    reg = pd.DataFrame(
        {
            'match_id':[1], 'event_id':[10], 'fold':[0], 'model_variant':['r4_full_with_360'], 'y_true':[0.1], 'y_pred':[0.12]
        }
    )
    two = pd.DataFrame(
        {
            'match_id':[1],
            'event_id':[10],
            'fold':[0],
            'classification_model_variant':['b7_full_with_360'],
            'conditional_model_variant':['conditional_tweedie'],
            'observed_future_shot':[0],
            'observed_future_xg':[0.1],
            'combined_future_xg_prediction':[0.11],
        }
    )
    assert 'coach_expected_future_xg_r4' in select_regression_variant(reg).columns
    assert 'coach_expected_future_xg_two_part' in select_two_part_variant(two).columns

def test_competition_comparison_denominators():
    out=competition_comparison(sample())
    assert set(out['competition']) == {'World Cup','Euros'}
    assert out['actions'].sum() == 4

def test_suppression_sign_convention():
    out=add_suppression(sample())
    assert out.loc[0,'coach_shot_suppression'] == .3
    assert out.loc[2,'coach_xg_suppression_r4'] < 0

def test_representative_event_selection():
    out=select_representative_events(construct_sequences(sample()), n=1)
    assert len(out) >= 1
    assert 'reason_selected_for_review' in out.columns

def test_visibility_filtering():
    filtered = apply_visibility_filter(sample(), reliable_only=True)
    assert len(filtered) == 1
    assert filtered['coach_reliable_visibility'].all()


def test_false_360_not_reliable():
    out = apply_visibility_filter(sample(), reliable_only=False)
    assert out.loc[out['has_360'].eq(False), 'coach_reliable_visibility'].eq(False).all()


def test_tactical_labels_require_observable_evidence():
    out = label_box_defence(construct_sequences(sample()))
    assert 'video_review_required' in set(out['coach_tactical_label'])


def test_match_level_bootstrap_reproducible():
    data = sample()
    data = add_suppression(data)
    a = bootstrap_match_level_ci(data, 'coach_shot_suppression', seed=11)
    b = bootstrap_match_level_ci(data, 'coach_shot_suppression', seed=11)
    assert a == b


def test_minimum_sample_flag_in_summary_table():
    data = add_suppression(sample())
    table = summary_table(data, ['competition'])
    assert table['minimum_sample_flag'].all()

def test_empty_population_handling():
    assert box_defence_population(sample().iloc[0:0]).empty
