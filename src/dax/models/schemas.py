from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import pandas as pd

COMMON_REQUIRED=("match_id","event_id","player_id","team","event_type","action_family","phase_label","action_x","action_y","target_future_shot_10s","target_future_xg_10s")
@dataclass(frozen=True)
class DatasetAudit:
    rows:int; matches:int; shot_positive:int; xg_nonzero:int; shot_rate:float; xg_zero_rate:float; warnings:list[str]

def normalise_model_schema(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    if "player_name" not in out.columns and "player" in out.columns: out["player_name"]=out["player"]
    if "player" not in out.columns and "player_name" in out.columns: out["player"]=out["player_name"]
    if "team" not in out.columns:
        for c in ("actor_team","defending_team","team_name"):
            if c in out.columns: out["team"]=out[c]; break
    if "position_group" not in out.columns: out["position_group"]="unknown"
    return out

def validate_model_dataset(df:pd.DataFrame)->DatasetAudit:
    df=normalise_model_schema(df)
    if df.empty: raise ValueError("Model dataset is empty.")
    missing=[c for c in COMMON_REQUIRED if c not in df.columns]
    if "player_name" not in df.columns and "player" not in df.columns: missing.append("player/player_name")
    if missing: raise ValueError(f"Missing required model fields: {missing}")
    if df["match_id"].isna().any(): raise ValueError("match_id contains missing values.")
    matches=int(df["match_id"].nunique())
    if matches<2: raise ValueError("Model validation requires at least two matches for grouped CV.")
    if df[["target_future_shot_10s","target_future_xg_10s"]].isna().any().any(): raise ValueError("Required targets contain missing values.")
    vals=set(pd.Series(df["target_future_shot_10s"].unique()).dropna().astype(int).tolist())
    if not vals.issubset({0,1}): raise ValueError("target_future_shot_10s must be binary 0/1.")
    if (df["target_future_xg_10s"]<0).any(): raise ValueError("target_future_xg_10s must be non-negative.")
    dup=int(df.duplicated(["event_id"]).sum()) if "event_id" in df else 0
    warnings=[f"event_id is not unique at player-action grain: {dup} duplicates"] if dup else []
    return DatasetAudit(len(df),matches,int(df["target_future_shot_10s"].sum()),int((df["target_future_xg_10s"]>0).sum()),float(df["target_future_shot_10s"].mean()),float((df["target_future_xg_10s"]==0).mean()),warnings)

def dataset_fingerprint(path:str|Path, df:pd.DataFrame)->dict:
    schema=json.dumps([(c,str(t)) for c,t in zip(df.columns,df.dtypes)],sort_keys=True)
    p=Path(path); file_hash=None
    if p.exists() and p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
        file_hash=h.hexdigest()
    return {"path":str(path),"row_count":int(len(df)),"column_count":int(df.shape[1]),"match_count":int(df["match_id"].nunique()) if "match_id" in df else 0,"schema_hash":hashlib.sha256(schema.encode()).hexdigest(),"file_hash":file_hash,"target_summary":{"shot_positive":int(df.get("target_future_shot_10s",pd.Series(dtype=int)).sum()),"xg_nonzero":int((df.get("target_future_xg_10s",pd.Series(dtype=float))>0).sum())}}
