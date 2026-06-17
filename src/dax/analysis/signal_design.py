from __future__ import annotations
import numpy as np
import pandas as pd
def _pct(s): return s.rank(pct=True).fillna(0.5)
def _z(s):
    sd=s.std(); return (s-s.mean())/(sd if sd and not np.isnan(sd) else 1)
def build_descriptive_signals(summary: pd.DataFrame, clusters: pd.DataFrame|None=None, min_actions:int=30) -> pd.DataFrame:
    out=summary[[c for c in ["player_id","player_name","team","total_actions","matches","minimum_sample_flag"] if c in summary]].copy()
    out["activity_index"]=_z(summary.get("actions_per_match", summary["total_actions"]))
    out["possession_win_index"]=_z(summary.get("possession_win_rate", pd.Series(0,index=summary.index)))
    out["threat_suppression_descriptive_index"]=_z(-summary.get("future_shot_rate", pd.Series(0,index=summary.index)))
    phase_cols=[c for c in summary if c.startswith("phase_") and c.endswith("_share")]
    out["phase_versatility_index"]=_z(-(summary[phase_cols].pow(2).sum(axis=1) if phase_cols else pd.Series(0,index=summary.index)))
    out["spatial_aggression_index"]=_z(-summary.get("mean_action_x", summary.get("future_xg_mean", pd.Series(0,index=summary.index))))
    out["transition_defence_exposure_index"]=_z(summary.get("transition_defence_share", pd.Series(0,index=summary.index)))
    out["box_defence_exposure_index"]=_z(summary.get("is_box_defence_share", pd.Series(0,index=summary.index)))
    out["local_numerical_difficulty_index"]=_z(-summary.get("mean_local_numerical_balance", pd.Series(0,index=summary.index)))
    out["visibility_reliability_index"]=_z(summary.get("reliable_visibility_actions", pd.Series(0,index=summary.index))/summary["total_actions"].replace(0,np.nan))
    for c in [c for c in out if c.endswith("_index")]: out[c.replace("_index","_percentile")]=_pct(out[c])
    out["reliable_sample"] = summary["total_actions"]>=min_actions
    out["warnings"] = np.where(out["reliable_sample"], "descriptive provisional signal; not true DAx", "small sample; descriptive provisional signal; not true DAx")
    if clusters is not None and len(clusters): out=out.merge(clusters[[c for c in ["player_id","team","cluster"] if c in clusters]],on=[c for c in ["player_id","team"] if c in out and c in clusters],how="left")
    return out
