from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .zones import find_xy_columns

def action_pitch_map(df: pd.DataFrame, title: str, path: str | Path | None=None):
    try:
        from mplsoccer import Pitch
        pitch=Pitch(pitch_type='statsbomb'); fig, ax=pitch.draw(figsize=(8,5))
        x,y=find_xy_columns(df)
        if x and not df.empty: pitch.scatter(df[x], df[y], s=18, alpha=.55, ax=ax)
    except Exception:
        fig, ax=plt.subplots(figsize=(8,5)); x,y=find_xy_columns(df)
        if x and not df.empty: ax.scatter(df[x], df[y], s=18, alpha=.55); ax.set_xlim(0,120); ax.set_ylim(0,80)
    ax.set_title(f"{title}\nN={len(df)}")
    if path: fig.savefig(path, bbox_inches='tight', dpi=150)
    return fig, ax
