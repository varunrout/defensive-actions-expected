from __future__ import annotations
import numpy as np
import pandas as pd

def find_xy_columns(df: pd.DataFrame) -> tuple[str|None,str|None]:
    candidates=[('action_x','action_y'),('x','y'),('location_x','location_y'),('start_x','start_y'),('event_x','event_y')]
    for x,y in candidates:
        if x in df.columns and y in df.columns: return x,y
    return None,None

def normalise_x(x): return np.asarray(x, dtype=float)
def pitch_zone(x: float, y: float) -> str:
    # Coordinates follow the StatsBomb convention where x increases towards the attacking goal.
    if pd.isna(x) or pd.isna(y):
        return 'unknown'
    x_val = float(x)
    y_val = float(y)
    if x_val >= 114 and 36 <= y_val <= 44:
        return 'six-yard central'
    if x_val >= 114 and 18 <= y_val <= 62:
        return 'six-yard wide'
    if x_val >= 102 and 30 <= y_val <= 50:
        return 'central box'
    if x_val >= 102 and 18 <= y_val < 30:
        return 'left box channel'
    if x_val >= 102 and 50 < y_val <= 62:
        return 'right box channel'
    if 94 <= x_val < 102 and 30 <= y_val <= 50:
        return 'box entry central'
    if 94 <= x_val < 102 and 18 <= y_val <= 62:
        return 'box entry wide'
    if x_val >= 80:
        return 'wide final third'
    return 'outside final third'

def add_pitch_zones(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy(); xcol,ycol=find_xy_columns(out)
    if not xcol: out['coach_pitch_zone']='unknown'; return out
    out['coach_pitch_zone']=[pitch_zone(x,y) for x,y in zip(out[xcol], out[ycol])]
    zone_text = out['coach_pitch_zone'].astype(str)
    out['coach_box_lane']=np.select(
        [out['coach_pitch_zone'].eq('central box'), out['coach_pitch_zone'].eq('left box channel'), out['coach_pitch_zone'].eq('right box channel')],
        ['central', 'left', 'right'],
        'non-box',
    )
    out['coach_box_depth']=np.select(
        [zone_text.str.startswith('six-yard'), zone_text.str.contains('box entry'), zone_text.str.contains('box')],
        ['six-yard', 'entry', 'box'],
        'outside',
    )
    return out
