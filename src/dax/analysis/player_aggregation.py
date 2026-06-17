from __future__ import annotations
import pandas as pd
from .spatial_analysis import add_pitch_zones

def build_player_summary(df: pd.DataFrame, min_actions:int=30) -> pd.DataFrame:
    z=add_pitch_zones(df); keys=["player_id","player_name","team"]; g=z.groupby(keys,dropna=False)
    agg={"matches":("match_id","nunique"),"total_actions":("event_id","size"),"future_shot_count":("target_future_shot_10s","sum"),"future_shot_rate":("target_future_shot_10s","mean"),"future_xg_total":("target_future_xg_10s","sum"),"future_xg_mean":("target_future_xg_10s","mean")}
    for c,n in (("has_360","reliable_visibility_actions"),("visible_attacker_count","mean_visible_attacker_count"),("visible_defender_count","mean_visible_defender_count"),("local_numerical_balance","mean_local_numerical_balance"),("nearest_attacker_distance","mean_nearest_attacker_distance"),("nearest_defender_distance","mean_nearest_defender_distance"),("goal_side_defenders","mean_goal_side_defenders"),("possession_won","possession_wins"),("ends_opponent_possession","actions_ending_opponent_possession"),("retained_control","retained_control_actions"),("under_opponent_possession","actions_under_opponent_possession")):
        if c in z: agg[n]=(c,"sum" if n.endswith(("actions","wins","possession")) else "mean")
    out=g.agg(**agg).reset_index(); out["minimum_sample_flag"]=out["total_actions"]<min_actions; out["future_shot_denominator"]=out["total_actions"]; out["future_xg_denominator"]=out["total_actions"]; out["actions_per_match"]=out["total_actions"]/out["matches"].replace(0,pd.NA)
    for count,rate in (("possession_wins","possession_win_rate"),("actions_ending_opponent_possession","actions_ending_opponent_possession_rate"),("retained_control_actions","retained_control_rate"),("actions_under_opponent_possession","under_opponent_possession_rate")):
        if count in out: out[rate]=out[count]/out["total_actions"]
    for col,prefix in (("action_family","action_family"),("phase_label","phase"),("pitch_zone","zone")):
        piv=pd.crosstab([z[k] for k in keys], z[col]).reset_index(); piv.columns=keys+[f"{prefix}_{str(c)}_count" for c in piv.columns[len(keys):]]; out=out.merge(piv,on=keys,how="left")
        for c in [c for c in out.columns if c.startswith(f"{prefix}_") and c.endswith("_count")]: out[c.replace("_count","_share")]=out[c]/out["total_actions"]
    out["role_known_actions"]=out.get("reliable_visibility_actions",0)
    out["numerical_disadvantage_share"]=pd.NA
    if "mean_local_numerical_balance" in out: out["numerical_disadvantage_share"]=(out["mean_local_numerical_balance"]<0).astype(float)
    return out
