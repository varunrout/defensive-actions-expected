from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def bootstrap_ci_by_match(df, value_col, n=200, seed=42):
    rng=np.random.default_rng(seed); matches=df['match_id'].dropna().unique()
    if len(matches)<2: return (float('nan'),float('nan'),float('nan'))
    vals=[]
    for _ in range(n):
        sample=rng.choice(matches,size=len(matches),replace=True); vals.append(pd.concat([df[df.match_id.eq(m)] for m in sample])[value_col].sum())
    return float(np.mean(vals)), float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5))

def build_player_signals(classification_oof, regression_oof, output, min_actions=30):
    c=pd.read_parquet(classification_oof); r=pd.read_parquet(regression_oof)
    c=c.copy(); r=r.copy(); c['shot_suppression_oof']=c['y_score']-c['y_true']; r['future_xg_suppression_oof']=r['y_pred']-r['y_true']
    rows=[]
    keys=['player_id','player_name','team']
    for key,g in c.groupby(keys,dropna=False):
        rr=r[(r.player_id==key[0])&(r.team==key[2])]
        mean,lo,hi=bootstrap_ci_by_match(g,'shot_suppression_oof')
        rows.append(dict(zip(keys,key),matches=int(g.match_id.nunique()),eligible_actions=int(len(g)),expected_shots=float(g.y_score.sum()),observed_shots=float(g.y_true.sum()),total_shot_suppression=float(g.shot_suppression_oof.sum()),mean_shot_suppression_per_action=float(g.shot_suppression_oof.mean()),expected_future_xg=float(rr.y_pred.sum()) if len(rr) else 0.0,observed_future_xg=float(rr.y_true.sum()) if len(rr) else 0.0,total_future_xg_suppression=float(rr.future_xg_suppression_oof.sum()) if len(rr) else 0.0,mean_future_xg_suppression_per_action=float(rr.future_xg_suppression_oof.mean()) if len(rr) else 0.0,phase_metrics=g.groupby('phase_label').shot_suppression_oof.sum().to_dict(),action_family_metrics=g.groupby('action_family').shot_suppression_oof.sum().to_dict(),position_group_metrics=g.groupby('position_group').shot_suppression_oof.sum().to_dict(),model_variant=str(g.model_variant.iloc[0]),mlflow_run_id=str(g.mlflow_run_id.iloc[0]),shot_suppression_bootstrap_mean=mean,shot_suppression_ci_low=lo,shot_suppression_ci_high=hi,standard_error=float(g.shot_suppression_oof.std(ddof=1)/(len(g)**0.5)) if len(g)>1 else float('nan'),reliability_flag='limited' if len(g)<min_actions or g.match_id.nunique()<3 else 'standard',minimum_sample_flag=bool(len(g)<min_actions)))
    out=pd.DataFrame(rows); Path(output).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(output,index=False); return out
