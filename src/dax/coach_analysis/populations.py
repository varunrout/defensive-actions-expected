from __future__ import annotations
import pandas as pd
from .zones import add_pitch_zones, find_xy_columns

def _contains(s, terms): return s.astype(str).str.lower().str.contains('|'.join(terms), na=False)
def add_visibility_flag(df):
    out=df.copy(); cols=[c for c in out.columns if 'visible' in c.lower() or '360' in c.lower()]
    out['coach_reliable_visibility']=out[cols].notna().any(axis=1) if cols else False
    return out

def box_defence_population(df, centre_backs_only=False):
    out=add_visibility_flag(add_pitch_zones(df))
    x,y=find_xy_columns(out); mask=out['coach_pitch_zone'].isin(['penalty box','central box'])
    if centre_backs_only:
        pos=next((c for c in ['position_group','position','player_position'] if c in out.columns), None)
        if pos: mask &= _contains(out[pos], ['centre','center','cb','central defender'])
    return out[mask].copy()

def wide_defence_population(df):
    out=add_visibility_flag(add_pitch_zones(df)); return out[out['coach_pitch_zone'].isin(['wide final third','box edge']) & out['coach_box_lane'].ne('central')].copy()

def transition_population(df):
    phase=next((c for c in ['phase','phase_label','tactical_phase'] if c in df.columns), None)
    if phase: return df[_contains(df[phase], ['transition','recovery','counter'])].copy()
    return df.iloc[0:0].copy()

def apply_visibility_filter(df, reliable_only=False):
    out=add_visibility_flag(df); return out[out['coach_reliable_visibility']].copy() if reliable_only else out
