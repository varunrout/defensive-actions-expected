from __future__ import annotations
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

def make_grouped_folds(df:pd.DataFrame,target:str,group_col:str='match_id',n_splits:int=5,seed:int=42)->pd.DataFrame:
    groups=df[group_col]; n_groups=groups.nunique()
    if n_groups<2: raise ValueError('Grouped CV requires at least two groups.')
    k=min(n_splits,int(n_groups))
    y=df[target]
    use_strat=target.endswith('shot_10s') and y.nunique()==2 and int(y.sum())>=k
    splitter=StratifiedGroupKFold(k,shuffle=True,random_state=seed) if use_strat else GroupKFold(k)
    folds=pd.Series(index=df.index,dtype='int64')
    X=df[[group_col]]
    for fold,(_,val) in enumerate(splitter.split(X,y if use_strat else None,groups)):
        folds.iloc[val]=fold
    out=df[[group_col]].copy(); out['fold']=folds.astype(int); out['row_index']=df.index
    return out

def fold_summary(df,target,folds,group_col='match_id'):
    x=df.join(folds['fold']) if 'fold' not in df else df
    return x.groupby('fold').agg(rows=(target,'size'),matches=(group_col,'nunique'),target_mean=(target,'mean'),positive_shots=('target_future_shot_10s','sum'),nonzero_xg=('target_future_xg_10s',lambda s:int((s>0).sum()))).reset_index()
