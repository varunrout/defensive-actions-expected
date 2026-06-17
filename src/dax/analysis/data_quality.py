from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .schemas import validate_processed_events

def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column":df.columns,"missing_count":[int(df[c].isna().sum()) for c in df.columns],"missing_rate":[float(df[c].isna().mean()) for c in df.columns],"dtype":[str(df[c].dtype) for c in df.columns]})

def event_ordering_issues(df: pd.DataFrame) -> pd.DataFrame:
    if not {"match_id","period","index"}.issubset(df.columns): return pd.DataFrame()
    bad=df.sort_values(["match_id","period","index"]).groupby(["match_id","period"])["index"].diff().fillna(1)<0
    return df.loc[bad,["match_id","period","index"]]

def duplicate_identifier_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys=[c for c in ("match_id","event_id") if c in df.columns] or (["id"] if "id" in df.columns else [])
    if not keys: return pd.DataFrame([{"key":"none","duplicate_rows":0}])
    return pd.DataFrame([{"key":"+".join(keys),"duplicate_rows":int(df.duplicated(keys).sum())}])

def processed_event_tables(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    validate_processed_events(df)
    out={"missingness":missingness_summary(df),"duplicates":duplicate_identifier_summary(df)}
    out["overview"]=pd.DataFrame([{"rows":len(df),"matches":df["match_id"].nunique(),"competitions":df["competition"].nunique() if "competition" in df else np.nan,"future_shot_rate":df["target_future_shot_10s"].mean(),"future_xg_mean":df["target_future_xg_10s"].mean(),"future_xg_zero_rate":(df["target_future_xg_10s"]==0).mean()}])
    out["event_counts_by_type"]=df["event_type"].value_counts(dropna=False).rename_axis("event_type").reset_index(name="rows")
    out["rows_per_match"]=df.groupby("match_id").size().reset_index(name="rows")
    out["events_per_possession"]=df.groupby(["match_id","possession"]).size().reset_index(name="events")
    out["phase_distribution"]=df["phase_label"].value_counts(dropna=False).rename_axis("phase_label").reset_index(name="rows")
    out["target_by_phase"]=df.groupby("phase_label",dropna=False).agg(rows=("match_id","size"),future_shot_rate=("target_future_shot_10s","mean"),future_xg_mean=("target_future_xg_10s","mean")).reset_index()
    out["target_by_event_type"]=df.groupby("event_type",dropna=False).agg(rows=("match_id","size"),future_shot_rate=("target_future_shot_10s","mean"),future_xg_mean=("target_future_xg_10s","mean")).reset_index()
    if "has_360" in df: out["coverage_360_by_match"]=df.groupby("match_id").agg(rows=("match_id","size"),has_360_rate=("has_360","mean")).reset_index()
    bad_team=(df["attacking_team_before_action"].isna() | df["defending_team_before_action"].isna() | (df["attacking_team_before_action"]==df["defending_team_before_action"]))
    out["team_context_issues"]=pd.DataFrame([{"invalid_team_context_rows":int(bad_team.sum()),"invalid_team_context_rate":float(bad_team.mean())}])
    return out

def write_tables(tables: dict[str,pd.DataFrame], outdir: str|Path) -> None:
    p=Path(outdir); p.mkdir(parents=True,exist_ok=True)
    for name, table in tables.items(): table.to_csv(p/f"{name}.csv",index=False)
    (p/"metadata.json").write_text(json.dumps({"tables":list(tables)},indent=2),encoding="utf-8")
