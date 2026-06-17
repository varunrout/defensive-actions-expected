from __future__ import annotations
import numpy as np
import pandas as pd
from .schemas import coordinate_columns
PITCH_LENGTH=120.0; PITCH_WIDTH=80.0
def add_pitch_zones(df: pd.DataFrame, bins_x:int=6, bins_y:int=4) -> pd.DataFrame:
    x,y=coordinate_columns(df); out=df.copy()
    if not x or not y: out["pitch_zone"]="unknown"; return out
    out["x_bin"]=pd.cut(out[x], np.linspace(0,PITCH_LENGTH,bins_x+1), include_lowest=True, labels=False)
    out["y_bin"]=pd.cut(out[y], np.linspace(0,PITCH_WIDTH,bins_y+1), include_lowest=True, labels=False)
    out["pitch_zone"]=out["x_bin"].astype("Int64").astype(str)+"_"+out["y_bin"].astype("Int64").astype(str)
    return out
def zone_summary(df: pd.DataFrame, bins_x:int=6, bins_y:int=4) -> pd.DataFrame:
    z=add_pitch_zones(df,bins_x,bins_y)
    ag={"rows":("match_id","size"),"future_shot_rate":("target_future_shot_10s","mean"),"future_xg_mean":("target_future_xg_10s","mean")}
    if "possession_won" in z: ag["possession_win_rate"]=("possession_won","mean")
    return z.groupby("pitch_zone",dropna=False).agg(**ag).reset_index()
def player_spatial_profiles(df: pd.DataFrame) -> pd.DataFrame:
    x,y=coordinate_columns(df)
    if not x or not y: return pd.DataFrame()
    return df.groupby(["player_id","player_name","team"],dropna=False).agg(total_actions=("event_id","size"),mean_action_x=(x,"mean"),median_action_x=(x,"median"),mean_action_y=(y,"mean"),median_action_y=(y,"median"),action_width=(y,"std")).reset_index()
