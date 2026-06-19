from __future__ import annotations
import numpy as np
import pandas as pd

def find_xy_columns(df: pd.DataFrame) -> tuple[str|None,str|None]:
    candidates=[('x','y'),('location_x','location_y'),('start_x','start_y'),('event_x','event_y')]
    for x,y in candidates:
        if x in df.columns and y in df.columns: return x,y
    return None,None

def normalise_x(x): return np.asarray(x, dtype=float)
def pitch_zone(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y): return 'unknown'
    if x >= 102 and 18 <= y <= 62: return 'penalty box'
    if x >= 102 and 30 <= y <= 50: return 'central box'
    if x >= 90 and 18 <= y <= 62: return 'box edge'
    if x >= 80 and (y < 18 or y > 62): return 'wide final third'
    if x >= 80: return 'final third'
    if x >= 60: return 'middle third'
    return 'defensive half'

def add_pitch_zones(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy(); xcol,ycol=find_xy_columns(out)
    if not xcol: out['coach_pitch_zone']='unknown'; return out
    out['coach_pitch_zone']=[pitch_zone(x,y) for x,y in zip(out[xcol], out[ycol])]
    out['coach_box_lane']=np.select([out[ycol].between(30,50), out[ycol]<30, out[ycol]>50], ['central','left/wide','right/wide'], 'unknown')
    out['coach_box_depth']=np.select([out[xcol]>=114, out[xcol].between(108,114), out[xcol].between(102,108)], ['six-yard/deep','penalty-spot','box-entry'], 'outside')
    return out
