from __future__ import annotations
import pandas as pd
def phase_tables(df: pd.DataFrame, min_count:int=20) -> dict[str,pd.DataFrame]:
    freq=df["phase_label"].value_counts(dropna=False).rename_axis("phase_label").reset_index(name="rows")
    by_player=df.groupby(["player_id","player_name","phase_label"],dropna=False).size().reset_index(name="actions") if "player_id" in df else pd.DataFrame()
    outcome=df.groupby("phase_label",dropna=False).agg(rows=("phase_label","size"),future_shot_rate=("target_future_shot_10s","mean"),future_xg_mean=("target_future_xg_10s","mean")).reset_index(); outcome["minimum_sample_warning"]=outcome["rows"]<min_count
    return {"phase_frequency":freq,"phase_exposure_by_player":by_player,"phase_outcomes":outcome}
