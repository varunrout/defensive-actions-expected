from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_XY_COLUMNS = [('action_x', 'action_y'), ('x', 'y'), ('location_x', 'location_y'), ('start_x', 'start_y'), ('event_x', 'event_y')]


class CoordinateError(ValueError):
    """Raised when action coordinate columns are unavailable."""


def find_xy_columns(df: pd.DataFrame, *, strict: bool = False) -> tuple[str | None, str | None]:
    for x, y in SUPPORTED_XY_COLUMNS:
        if x in df.columns and y in df.columns:
            return x, y
    if strict:
        raise CoordinateError(f"No supported coordinate pair found. Expected one of: {SUPPORTED_XY_COLUMNS}; available columns: {list(df.columns)}")
    return None, None


def defensive_box_zone(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y) or x > 18 or y < 18 or y > 62:
        return 'outside defensive box'
    if x <= 6 and 30 <= y <= 50:
        return 'own six-yard box'
    if x <= 6:
        return 'own deep wide six-yard area'
    if x <= 12 and 30 <= y <= 50:
        return 'own penalty-spot/deep central zone'
    if x <= 18 and 30 <= y <= 50:
        return 'own central box'
    if y < 30:
        return 'own left/wide box'
    return 'own right/wide box'


def attacking_box_zone(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y) or x < 102 or y < 18 or y > 62:
        return 'outside attacking box'
    if x >= 114 and 30 <= y <= 50:
        return 'attacking six-yard box'
    if x >= 114:
        return 'attacking deep wide six-yard area'
    if x >= 108 and 30 <= y <= 50:
        return 'attacking penalty-spot/deep central zone'
    if x >= 102 and 30 <= y <= 50:
        return 'attacking central box'
    if y < 30:
        return 'attacking left/wide box'
    return 'attacking right/wide box'


def pitch_zone(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y):
        return 'unknown'
    defensive = defensive_box_zone(x, y)
    if defensive != 'outside defensive box':
        return defensive
    attacking = attacking_box_zone(x, y)
    if attacking != 'outside attacking box':
        return attacking
    if x >= 80 and (y < 18 or y > 62): return 'wide final third'
    if x >= 80: return 'final third'
    if x >= 60: return 'middle third'
    return 'defensive half'


def add_pitch_zones(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    out = df.copy()
    xcol, ycol = find_xy_columns(out, strict=strict)
    if not xcol or not ycol:
        out['coach_coordinate_x_column'] = pd.NA
        out['coach_coordinate_y_column'] = pd.NA
        out['coach_defensive_box_zone'] = 'unknown'
        out['coach_attacking_box_zone'] = 'unknown'
        out['coach_pitch_zone'] = 'unknown'
        return out
    out['coach_coordinate_x_column'] = xcol
    out['coach_coordinate_y_column'] = ycol
    out['coach_defensive_box_zone'] = [defensive_box_zone(x, y) for x, y in zip(out[xcol], out[ycol], strict=False)]
    out['coach_attacking_box_zone'] = [attacking_box_zone(x, y) for x, y in zip(out[xcol], out[ycol], strict=False)]
    out['coach_pitch_zone'] = [pitch_zone(x, y) for x, y in zip(out[xcol], out[ycol], strict=False)]
    out['coach_box_lane'] = np.select([out[ycol].between(30, 50), out[ycol] < 30, out[ycol] > 50], ['central', 'left/wide', 'right/wide'], 'unknown')
    out['coach_box_depth'] = np.select([out[xcol] <= 6, out[xcol].between(6, 12), out[xcol].between(12, 18)], ['six-yard/deep', 'penalty-spot', 'box-entry'], 'outside')
    return out
