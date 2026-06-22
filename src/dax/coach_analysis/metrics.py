from __future__ import annotations
import numpy as np
import pandas as pd

def first_existing(df, names):
    return next((n for n in names if n in df.columns), None)
def add_suppression(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    exp_shot=first_existing(out,['coach_expected_shot_b7','expected_shot_probability','pred_shot','shot_probability_oof','y_pred_proba','future_shot_pred'])
    obs_shot=first_existing(out,['coach_observed_shot','target_future_shot_10s','observed_future_shot','future_shot','shot_within_10s','y_true'])
    exp_xg=first_existing(out,['coach_expected_xg_r4','expected_future_xg','pred_future_xg','future_xg_pred','y_pred'])
    obs_xg=first_existing(out,['coach_observed_xg','target_future_xg_10s','observed_future_xg','future_xg','xg_within_10s','y_true_xg'])
    if exp_shot and obs_shot: out['shot_suppression']=out[exp_shot]-out[obs_shot]
    if exp_xg and obs_xg: out['xg_suppression']=out[exp_xg]-out[obs_xg]
    return out

def bootstrap_ci(values, statistic=np.mean, n_boot=500, seed=7):
    arr=np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(arr)==0: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); stats=[statistic(rng.choice(arr, len(arr), True)) for _ in range(n_boot)]
    return (float(statistic(arr)), float(np.quantile(stats,.025)), float(np.quantile(stats,.975)))

def summary_table(df, group_cols):
    if df.empty: return pd.DataFrame(columns=list(group_cols)+['actions','matches','players'])
    obs_shot=first_existing(df,['coach_observed_shot','target_future_shot_10s','observed_future_shot','future_shot','shot_within_10s','y_true'])
    obs_xg=first_existing(df,['coach_observed_xg','target_future_xg_10s','observed_future_xg','future_xg','xg_within_10s','y_true_xg'])
    exp_shot=first_existing(df,['coach_expected_shot_b7','expected_shot_probability','pred_shot','shot_probability_oof','y_pred_proba','future_shot_pred'])
    exp_xg=first_existing(df,['coach_expected_xg_r4','expected_future_xg','pred_future_xg','future_xg_pred','y_pred'])
    poss=first_existing(df,['coach_possession_controlled','action_won_possession','action_changed_possession','possession_won','won_possession','defensive_action_won'])
    agg={}
    for c,n in [(obs_shot,'future_shot_rate'),(obs_xg,'future_xg_per_action'),(exp_shot,'expected_shot_probability'),(exp_xg,'expected_future_xg'),(poss,'possession_win_rate')]:
        if c: agg[n]=(c,'mean')
    base=df.groupby(list(group_cols), dropna=False).agg(actions=(df.columns[0],'size'), matches=('match_id','nunique') if 'match_id' in df.columns else (df.columns[0],'size'), players=('player','nunique') if 'player' in df.columns else (df.columns[0],'size'), **agg).reset_index()
    if {'expected_shot_probability','future_shot_rate'}.issubset(base.columns): base['shot_suppression']=base['expected_shot_probability']-base['future_shot_rate']
    if {'expected_future_xg','future_xg_per_action'}.issubset(base.columns): base['xg_suppression']=base['expected_future_xg']-base['future_xg_per_action']
    return base.sort_values('actions', ascending=False)
