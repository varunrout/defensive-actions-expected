from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import pandas as pd
from .leakage import scan_features
@dataclass(frozen=True)
class FeatureContract:
    task:str; name:str; target:str; model_family:str; feature_scope:str; categorical:list[str]; numeric:list[str]; required:list[str]; optional:list[str]; excluded:list[str]; hyperparameters:dict; requires_360:bool; minimum_usable_rows:int
    @property
    def features(self): return list(dict.fromkeys([*self.categorical,*self.numeric]))
def load_model_config(path:str|Path='configs/models.yaml')->dict:
    return yaml.safe_load(Path(path).read_text())
def _dups(xs): return sorted({x for x in xs if xs.count(x)>1})
def get_contracts(config:dict, task:str)->list[FeatureContract]:
    target=config[task]["target"]; out=[]
    for name,raw in config[task]["variants"].items():
        cats=list(raw.get("categorical_features",[])); nums=list(raw.get("numeric_features",[])); allf=cats+nums
        d=_dups(allf)
        if d: raise ValueError(f"Duplicate features in {name}: {d}")
        scan_features(allf, selected_target=target)
        out.append(FeatureContract(task,name,target,raw["model_family"],raw.get("feature_scope","pre_action_context"),cats,nums,list(raw.get("required_features",[])),list(raw.get("optional_features",[])),list(raw.get("excluded_features",[])),dict(raw.get("hyperparameters",{})),bool(raw.get("requires_360",False)),int(raw.get("minimum_usable_rows",1))))
    return out
def resolve_contract(df:pd.DataFrame, c:FeatureContract)->dict:
    req_missing=[f for f in c.required if f not in df.columns]
    if req_missing: raise ValueError(f"Missing required features for {c.name}: {req_missing}")
    cats=[f for f in c.categorical if f in df.columns]; nums=[f for f in c.numeric if f in df.columns]
    final=list(dict.fromkeys(cats+nums)); scan_features(final, selected_target=c.target)
    miss_opt=[f for f in c.optional if f not in df.columns]
    return {"requested_features":c.features,"available_features":[f for f in c.features if f in df.columns],"missing_required_features":req_missing,"missing_optional_features":miss_opt,"final_features":final,"categorical":cats,"numeric":nums,"rows_retained":int(len(df.dropna(subset=[c.target]))),"rows_excluded":int(df[c.target].isna().sum()),"feature_missingness":{f:float(df[f].isna().mean()) for f in final},"coverage_360":float(df.get("has_360",pd.Series([False]*len(df))).fillna(False).mean())}
