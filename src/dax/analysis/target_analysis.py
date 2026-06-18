from __future__ import annotations
import pandas as pd

def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"rows":len(df),"future_shot_positive":int(df["target_future_shot_10s"].sum()),"future_shot_rate":float(df["target_future_shot_10s"].mean()),"future_xg_mean":float(df["target_future_xg_10s"].mean()),"future_xg_zero_rate":float((df["target_future_xg_10s"]==0).mean())}])

def target_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.groupby(column, dropna=False).agg(rows=(column,"size"),future_shot_rate=("target_future_shot_10s","mean"),future_xg_mean=("target_future_xg_10s","mean")).reset_index().sort_values("rows",ascending=False)
