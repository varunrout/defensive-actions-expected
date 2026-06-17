from __future__ import annotations
import numpy as np
import pandas as pd
FEATURE_GROUP_RULES={"identifiers":["id","match","player","team"],"targets":["target","future"],"spatial":["x","y","distance","zone","lane","height","width"],"possession_semantics":["possession","control","retained"],"phase":["phase"],"360_roles":["attacker","defender","goal_side"],"local_numerical_balance":["balance","within"],"visibility":["visible","visibility","has_360"],"exposure":["exposure","share","rate"],"player_aggregates":["actions","matches"]}
def feature_group(name: str) -> str:
    n=name.lower()
    for g, pats in FEATURE_GROUP_RULES.items():
        if any(p in n for p in pats): return g
    return "action_context"
def numeric_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for c in df.select_dtypes(include="number").columns:
        s=df[c]; q=s.quantile([.01,.05,.25,.5,.75,.95,.99]); iqr=q.loc[.75]-q.loc[.25]; lo=q.loc[.25]-1.5*iqr; hi=q.loc[.75]+1.5*iqr
        rows.append({"feature":c,"group":feature_group(c),"count":int(s.count()),"missing_rate":float(s.isna().mean()),"mean":s.mean(),"std":s.std(),"min":s.min(),"p01":q.loc[.01],"p05":q.loc[.05],"p25":q.loc[.25],"median":q.loc[.5],"p75":q.loc[.75],"p95":q.loc[.95],"p99":q.loc[.99],"max":s.max(),"unique_count":int(s.nunique(dropna=True)),"zero_rate":float((s==0).mean()),"outlier_count":int(((s<lo)|(s>hi)).sum()),"constant_flag":bool(s.nunique(dropna=True)<=1),"near_constant_flag":bool(s.value_counts(normalize=True,dropna=True).head(1).sum()>0.95 if s.count() else False)})
    return pd.DataFrame(rows)
def categorical_diagnostics(df: pd.DataFrame, min_count:int=20) -> pd.DataFrame:
    rows=[]
    for c in df.select_dtypes(exclude="number").columns:
        vc=df[c].value_counts(dropna=False); rows.append({"feature":c,"group":feature_group(c),"unique_values":int(df[c].nunique(dropna=True)),"missing_rate":float(df[c].isna().mean()),"top_value":str(vc.index[0]) if len(vc) else "","top_count":int(vc.iloc[0]) if len(vc) else 0,"rare_category_rate":float(vc[vc<min_count].sum()/len(df)) if len(df) else 0})
    return pd.DataFrame(rows)
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").corr(numeric_only=True)
def target_relationships(df: pd.DataFrame) -> pd.DataFrame:
    nums=[c for c in df.select_dtypes(include="number").columns if c not in ("target_future_shot_10s","target_future_xg_10s")]
    rows=[]
    for c in nums:
        rows.append({"feature":c,"corr_future_shot":df[[c,"target_future_shot_10s"]].corr().iloc[0,1] if "target_future_shot_10s" in df else np.nan,"corr_future_xg":df[[c,"target_future_xg_10s"]].corr().iloc[0,1] if "target_future_xg_10s" in df else np.nan})
    return pd.DataFrame(rows)
def diagnostics_tables(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    return {"numeric_diagnostics":numeric_diagnostics(df),"categorical_diagnostics":categorical_diagnostics(df),"correlations":correlation_matrix(df),"target_relationships":target_relationships(df),"missingness_by_feature_family":numeric_diagnostics(df).groupby("group").agg(features=("feature","count"),mean_missing_rate=("missing_rate","mean")).reset_index()}
