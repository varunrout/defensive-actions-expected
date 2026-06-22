from __future__ import annotations
import pandas as pd

def construct_sequences(df: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    if df.empty: return df.copy()
    group_cols=group_cols or [c for c in ['match_id','possession','sequence_id'] if c in df.columns]
    order=[c for c in ['match_id','period','minute','second','event_index','index'] if c in df.columns]
    out=df.sort_values(order or list(df.columns[:1])).copy()
    if not group_cols: group_cols=['match_id'] if 'match_id' in out.columns else []
    if group_cols:
        out['coach_action_number_in_sequence']=out.groupby(group_cols).cumcount()+1
        out['coach_is_repeated_defensive_action']=out['coach_action_number_in_sequence']>1
        for col in ['team','player','type','event_type','action_family']:
            if col in out.columns:
                out[f'next_{col}']=out.groupby(group_cols)[col].shift(-1); out[f'second_next_{col}']=out.groupby(group_cols)[col].shift(-2)
    else:
        out['coach_action_number_in_sequence']=range(1,len(out)+1); out['coach_is_repeated_defensive_action']=out['coach_action_number_in_sequence']>1
    return out
