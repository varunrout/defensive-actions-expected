from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_bar(df: pd.DataFrame, x: str, y: str, out: str|Path, title: str, xlabel: str|None=None, ylabel: str|None=None, dpi:int=150):
    fig, ax=plt.subplots(figsize=(8,4))
    if len(df): ax.bar(df[x].astype(str).head(30), df[y].head(30)); ax.tick_params(axis='x',rotation=45)
    ax.set_title(title); ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y); fig.tight_layout(); Path(out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=dpi); plt.close(fig); return fig

def save_heatmap(matrix: pd.DataFrame, out: str|Path, title: str, dpi:int=150):
    fig, ax=plt.subplots(figsize=(8,6)); im=ax.imshow(matrix.fillna(0).to_numpy(), aspect='auto'); ax.set_title(title); fig.colorbar(im,ax=ax); fig.tight_layout(); Path(out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=dpi); plt.close(fig); return fig

def save_scatter(df: pd.DataFrame, x: str, y: str, out: str|Path, title: str, color: str|None=None, dpi:int=150):
    fig, ax=plt.subplots(figsize=(6,5))
    if len(df): ax.scatter(df[x],df[y],c=df[color] if color and color in df else None)
    ax.set_title(title); ax.set_xlabel(x); ax.set_ylabel(y); fig.tight_layout(); Path(out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=dpi); plt.close(fig); return fig
