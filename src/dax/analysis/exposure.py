from __future__ import annotations
import pandas as pd
def player_exposure(df: pd.DataFrame) -> pd.DataFrame:
    g=df.groupby(["player_id","player_name","team"],dropna=False)
    out=g.agg(matches=("match_id","nunique"),defensive_actions=("event_id","size")).reset_index(); out["actions_per_match"]=out["defensive_actions"]/out["matches"].replace(0,pd.NA)
    for col,name in (("has_360","visible_360_actions"),("high_threat_context","high_threat_actions"),("is_low_block","low_block_actions"),("is_counterpress","counterpress_actions"),("is_transition_defence","transition_defence_actions")):
        if col in df:
            tmp=g[col].sum().reset_index(name=name); out=out.merge(tmp,on=["player_id","player_name","team"],how="left"); out[name.replace("actions","share")]=out[name]/out["defensive_actions"]
    return out
