from __future__ import annotations
import pandas as pd
from .zones import add_pitch_zones

def _contains(s, terms): return s.astype(str).str.lower().str.contains('|'.join(terms), na=False)


def add_visibility_flag(df):
    out = df.copy()
    out['coach_has_360'] = out['has_360'].fillna(False).eq(True) if 'has_360' in out.columns else False
    out['coach_reliable_5m_visibility'] = (
        out['coach_has_360'] & out['local_5m_region_fully_visible'].fillna(False).eq(True)
        if 'local_5m_region_fully_visible' in out.columns
        else False
    )
    out['coach_reliable_10m_visibility'] = (
        out['coach_has_360'] & out['local_10m_region_fully_visible'].fillna(False).eq(True)
        if 'local_10m_region_fully_visible' in out.columns
        else False
    )
    out['coach_freeze_frame_roles_known'] = (
        out['freeze_frame_roles_known'].fillna(False).eq(True) if 'freeze_frame_roles_known' in out.columns else False
    )
    quality = out['visibility_quality_band'].astype(str).str.lower() if 'visibility_quality_band' in out.columns else pd.Series('', index=out.index)
    out['coach_visibility_quality_ok'] = quality.isin({'high', 'acceptable'})
    out['coach_reliable_visibility'] = (
        out['coach_has_360']
        & out['coach_reliable_5m_visibility']
        & out['coach_freeze_frame_roles_known']
        & out['coach_visibility_quality_ok']
    )
    return out

def box_defence_population(df, centre_backs_only=False):
    out=add_visibility_flag(add_pitch_zones(df))
    mask=out['coach_pitch_zone'].isin(['six-yard central','six-yard wide','central box','left box channel','right box channel'])
    if centre_backs_only:
        pos=next((c for c in ['position_group','position','player_position'] if c in out.columns), None)
        if pos: mask &= _contains(out[pos], ['centre','center','cb','central defender'])
    return out[mask].copy()

def wide_defence_population(df):
    out=add_visibility_flag(add_pitch_zones(df)); return out[out['coach_pitch_zone'].isin(['wide final third','left box channel','right box channel','box entry wide'])].copy()

def transition_population(df):
    phase=next((c for c in ['phase','phase_label','tactical_phase'] if c in df.columns), None)
    if phase: return df[_contains(df[phase], ['transition','recovery','counter'])].copy()
    return df.iloc[0:0].copy()

def apply_visibility_filter(df, reliable_only=False):
    out=add_visibility_flag(df); return out[out['coach_reliable_visibility'].eq(True)].copy() if reliable_only else out
