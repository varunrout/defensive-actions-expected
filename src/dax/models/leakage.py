from __future__ import annotations
PROHIBITED_EXACT={"event_id","match_id","player_id","player","player_name","team","team_id","fold","target_future_shot_10s","target_future_xg_10s"}
PROHIBITED_PATTERNS=("future","outcome","prediction","pred_","score","suppression","cluster_","observed","dax")
def classify_feature(name:str)->str:
    n=name.lower()
    if n in PROHIBITED_EXACT or n.endswith("_id"): return "identifier" if not n.startswith("target") else "target"
    if n.startswith("target_") or "future" in n: return "target"
    if any(p in n for p in ("outcome","observed","prediction","suppression","dax")): return "post-action"
    if n in {"action_x","action_y","phase_label","event_type","action_family","position_group","play_pattern"}: return "action-time"
    if any(p in n for p in ("visible","attacker","defender","freeze_frame","distance","elapsed","order","zone","phase")): return "pre-action"
    return "uncertain"
def scan_features(features:list[str], *, selected_target:str, allow_uncertain:bool=False)->list[dict]:
    rows=[]; bad=[]
    for f in features:
        cls=classify_feature(f); prohibited=f in PROHIBITED_EXACT or (f.startswith("target_") and f!=selected_target) or cls in {"identifier","target","post-action"} or (cls=="uncertain" and not allow_uncertain)
        rows.append({"feature":f,"classification":cls,"prohibited":prohibited})
        if prohibited: bad.append(f)
    if bad: raise ValueError(f"Prohibited or uncertain leakage-prone features in contract: {bad}")
    return rows
