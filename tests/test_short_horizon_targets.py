import pandas as pd
from dax.targets.short_horizon import add_future_shot_target, add_future_xg_target


def base(rows):
    for r in rows:
        r.setdefault('match_id',1); r.setdefault('period',1); r.setdefault('possession',1); r.setdefault('attacking_team_before_action','A'); r.setdefault('minute',0); r.setdefault('event_type','Pass')
    return pd.DataFrame(rows)

def val(df,i=0): return int(add_future_shot_target(df).loc[i,'target_future_shot_10s'])

def test_shot_six_seconds_same_possession(): assert val(base([{'index':1,'second':0},{'index':2,'second':6,'event_type':'Shot'}])) == 1
def test_shot_eleven_seconds_returns_zero(): assert val(base([{'index':1,'second':0},{'index':2,'second':11,'event_type':'Shot'}])) == 0
def test_shot_exactly_ten_returns_one(): assert val(base([{'index':1,'second':0},{'index':2,'second':10,'event_type':'Shot'}])) == 1
def test_other_match_zero(): assert val(base([{'index':1,'second':0},{'index':2,'second':6,'event_type':'Shot','match_id':2}])) == 0
def test_other_period_zero(): assert val(base([{'index':1,'second':0},{'index':2,'second':6,'event_type':'Shot','period':2}])) == 0
def test_possession_change_zero(): assert val(base([{'index':1,'second':0},{'index':2,'second':6,'event_type':'Shot','possession':2}])) == 0
def test_lose_regain_new_possession_zero(): assert val(base([{'index':1,'second':0,'possession':1},{'index':2,'second':3,'possession':2,'attacking_team_before_action':'B'},{'index':3,'second':8,'possession':3,'event_type':'Shot'}])) == 0
def test_multiple_shots_same_possession(): assert val(base([{'index':1,'second':0},{'index':2,'second':3,'event_type':'Shot'},{'index':3,'second':5,'event_type':'Shot'}])) == 1
def test_missing_possession_id(): assert val(base([{'index':1,'second':0,'possession':None},{'index':2,'second':3,'event_type':'Shot','possession':None}])) == 0
def test_same_timestamp_different_indices_future_counts(): assert val(base([{'index':1,'second':0},{'index':2,'second':0,'event_type':'Shot'}])) == 1
def test_defensive_action_ends_possession_no_next_inherit(): assert val(base([{'index':1,'second':0,'event_type':'Interception'},{'index':2,'second':2,'possession':2,'event_type':'Shot','attacking_team_before_action':'B'}])) == 0
def test_shot_does_not_count_itself(): assert val(base([{'index':1,'second':0,'event_type':'Shot'}])) == 0

def test_future_xg_sums_multiple_shots():
    df=base([{'index':1,'second':0},{'index':2,'second':3,'event_type':'Shot','shot_statsbomb_xg':0.2},{'index':3,'second':5,'event_type':'Shot','shot_statsbomb_xg':0.1}])
    assert add_future_xg_target(df).loc[0,'target_future_xg_10s'] == 0.3


def test_possession_sequence_id_prevents_backwards_id_leakage():
    df = base([
        {'index': 1, 'second': 0, 'possession': 2, 'attacking_team_before_action': 'A'},
        {'index': 2, 'second': 3, 'possession': 1, 'attacking_team_before_action': 'B'},
        {'index': 3, 'second': 6, 'possession': 2, 'event_type': 'Shot', 'attacking_team_before_action': 'A'},
    ])
    df['possession_sequence_id'] = [1, 2, 3]
    out = add_future_shot_target(df, possession_column='possession_sequence_id')
    assert int(out.loc[0, 'target_future_shot_10s']) == 0

