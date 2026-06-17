"""Possession-bounded short-horizon observed targets."""
from __future__ import annotations

import pandas as pd


def _time(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("minute", 0), errors="coerce").fillna(0) * 60 + pd.to_numeric(
        df.get("second", 0), errors="coerce"
    ).fillna(0)


def _prepare(events: pd.DataFrame, possession_column: str) -> pd.DataFrame:
    df = events.copy()
    if possession_column not in df.columns:
        if "team_in_possession" in df.columns:
            df[possession_column] = df.groupby(["match_id", "period"], dropna=False)["team_in_possession"].transform(
                lambda s: (s != s.shift()).cumsum()
            )
        else:
            df[possession_column] = 1
    sort_cols = [c for c in ["match_id", "period", possession_column, "index"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["event_time_seconds"] = df.get("event_time_seconds", _time(df))
    return df


def add_future_shot_target(
    events: pd.DataFrame,
    horizon_seconds: float = 10,
    possession_column: str = "possession",
    attacking_team_column: str = "attacking_team_before_action",
) -> pd.DataFrame:
    df = _prepare(events, possession_column)
    out = pd.Series(0, index=df.index, dtype=int)
    etype = df.get("event_type", df.get("type", pd.Series(index=df.index, dtype=object)))
    for _, g in df[df[possession_column].notna()].groupby(["match_id", "period", possession_column], dropna=False):
        times = g["event_time_seconds"].to_numpy()
        idxs = g.index.to_list()
        types = etype.loc[g.index].to_list()
        teams = df.loc[g.index, attacking_team_column].to_list() if attacking_team_column in df else [None] * len(g)
        for pos, idx in enumerate(idxs):
            team = teams[pos]
            for j in range(pos + 1, len(idxs)):
                dt = times[j] - times[pos]
                if dt < 0:
                    continue
                if dt > horizon_seconds:
                    break
                if team is not None and teams[j] is not None and teams[j] != team:
                    continue
                if types[j] == "Shot":
                    out.loc[idx] = 1
                    break
    df["target_future_shot_10s"] = out
    return df


def add_future_xg_target(
    events: pd.DataFrame,
    horizon_seconds: float = 10,
    possession_column: str = "possession",
    attacking_team_column: str = "attacking_team_before_action",
    xg_column: str = "shot_statsbomb_xg",
) -> pd.DataFrame:
    df = add_future_shot_target(events, horizon_seconds, possession_column, attacking_team_column).copy()
    out = pd.Series(0.0, index=df.index, dtype=float)
    etype = df.get("event_type", df.get("type", pd.Series(index=df.index, dtype=object)))
    xg = pd.to_numeric(df.get(xg_column, 0.0), errors="coerce").fillna(0.0)
    for _, g in df[df[possession_column].notna()].groupby(["match_id", "period", possession_column], dropna=False):
        times = g["event_time_seconds"].to_numpy()
        idxs = g.index.to_list()
        types = etype.loc[g.index].to_list()
        teams = df.loc[g.index, attacking_team_column].to_list() if attacking_team_column in df else [None] * len(g)
        for pos, idx in enumerate(idxs):
            total_xg = 0.0
            team = teams[pos]
            for j in range(pos + 1, len(idxs)):
                dt = times[j] - times[pos]
                if dt < 0:
                    continue
                if dt > horizon_seconds:
                    break
                if team is not None and teams[j] is not None and teams[j] != team:
                    continue
                if types[j] == "Shot":
                    total_xg += float(xg.loc[idxs[j]])
            out.loc[idx] = round(total_xg, 10)
    df["target_future_xg_10s"] = out
    return df
