from __future__ import annotations
import pandas as pd

def label_box_defence(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy(); fam=out.get('action_family', pd.Series('', index=out.index)).astype(str).str.lower(); typ=out.get('event_type', out.get('type', pd.Series('', index=out.index))).astype(str).str.lower(); won=out.get('possession_won', out.get('won_possession', pd.Series(False,index=out.index))).fillna(False).astype(bool)
    out['coach_tactical_label']='attack remains alive'
    out.loc[won & fam.str.contains('recovery|interception|tackle'), 'coach_tactical_label']='controlled box defence'
    out.loc[typ.str.contains('clearance') & ~won, 'coach_tactical_label']='emergency survival'
    out.loc[typ.str.contains('block') & ~won, 'coach_tactical_label']='dangerous second-ball defence'
    out.loc[typ.str.contains('pressure') & ~won, 'coach_tactical_label']='high-risk pressure'
    out.loc[typ.str.contains('duel') & ~won, 'coach_tactical_label']='high-risk duel'
    if 'xg_suppression' in out.columns: out.loc[out['xg_suppression']>0.03, 'coach_tactical_label']='successful threat suppression'; out.loc[out['xg_suppression']<-0.03, 'coach_tactical_label']='defensive breakdown'
    return out

def label_rules() -> pd.DataFrame:
    return pd.DataFrame({'label':['controlled box defence','emergency survival','attack remains alive','dangerous second-ball defence','high-risk pressure','high-risk duel','successful threat suppression','defensive breakdown'], 'rule':['possession won by recovery/interception/tackle','clearance without secure possession','default when attack continuation cannot be ruled out','block without secure possession','pressure without possession recovery','duel without possession recovery','observed threat at least 0.03 xG below expected','observed threat at least 0.03 xG above expected']})
