from __future__ import annotations
import pandas as pd
from .metrics import first_existing, add_suppression
REVIEW_COLUMNS=['match_id','event_id','competition','team','opponent','player','minute','phase','action_type','location','expected_shot_probability','observed_future_shot','expected_future_xg','observed_future_xg','suppression_value','reliability_or_visibility','reason_selected_for_review']
def select_representative_events(df: pd.DataFrame, n=5, reason='representative situation') -> pd.DataFrame:
    if df.empty: return pd.DataFrame(columns=REVIEW_COLUMNS)
    out=add_suppression(df); score=first_existing(out,['xg_suppression','shot_suppression'])
    out['_abs_review_score']=out[score].abs() if score else 0
    out=out.sort_values('_abs_review_score', ascending=False).drop_duplicates([c for c in ['match_id','event_id'] if c in out.columns]).head(n).copy()
    rows=pd.DataFrame()
    for c in REVIEW_COLUMNS:
        if c=='suppression_value': rows[c]=out[score] if score else pd.NA
        elif c=='reliability_or_visibility': rows[c]=out[[x for x in ['reliability_tier','coach_reliable_visibility'] if x in out.columns][0]] if any(x in out.columns for x in ['reliability_tier','coach_reliable_visibility']) else pd.NA
        elif c=='reason_selected_for_review': rows[c]=reason
        elif c=='location': rows[c]=out.apply(lambda r: f"({r.get('x', r.get('location_x', ''))}, {r.get('y', r.get('location_y', ''))})", axis=1)
        else: rows[c]=out[c] if c in out.columns else pd.NA
    return rows
