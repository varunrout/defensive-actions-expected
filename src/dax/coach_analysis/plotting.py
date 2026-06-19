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
        if x and not df.empty:
            pitch.scatter(df[x], df[y], s=18, alpha=.55, ax=ax)
            ax.set_xlim(0, 60)
    except Exception:
        fig, ax=plt.subplots(figsize=(8,5)); x,y=find_xy_columns(df)
        if x and not df.empty: ax.scatter(df[x], df[y], s=18, alpha=.55); ax.set_xlim(0,120); ax.set_ylim(0,80)
    matches = df["match_id"].nunique() if "match_id" in df.columns else 0
    ax.set_title(f"{title}\nActions={len(df)} | Matches={matches} | Population: raw defensive coordinates")
    if path: fig.savefig(path, bbox_inches='tight', dpi=150)
    return fig, ax


def horizontal_metric_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: str | Path | None = None, top_n: int = 15):
    """Create a horizontal bar chart with readable labels and no rotated axes."""
    fig, ax = plt.subplots(figsize=(9, max(3, min(10, 0.4 * min(len(df), top_n) + 1.5))))
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        ax.set_title(f"{title}\nNo data available")
        ax.axis("off")
    else:
        plot_df = df[[label_col, value_col]].dropna().sort_values(value_col).tail(top_n)
        ax.barh(plot_df[label_col].astype(str), plot_df[value_col], color="#4C78A8")
        ax.set_xlabel(value_col.replace("_", " "))
        ax.set_ylabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=0)
        ax.tick_params(axis="y", labelrotation=0)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    return fig, ax
