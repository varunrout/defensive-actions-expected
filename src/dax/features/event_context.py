"""Event ordering and team/possession context utilities.

Canonical event order is match_id, period, index. Minute/second are temporal
values only and are not used as primary ordering keys.
"""
from __future__ import annotations

import pandas as pd

ORDER_COLUMNS = ["match_id", "period", "index"]


def event_time_seconds_df(df: pd.DataFrame) -> pd.Series:
    minute = df["minute"] if "minute" in df else pd.Series(0, index=df.index)
    second = df["second"] if "second" in df else pd.Series(0, index=df.index)
    return pd.to_numeric(minute, errors="coerce").fillna(0) * 60 + pd.to_numeric(second, errors="coerce").fillna(0)


def sort_events(events: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ORDER_COLUMNS if c in events.columns]
    return events.sort_values(cols, kind="mergesort").reset_index(drop=True) if cols else events.reset_index(drop=True)


def _opponent_for_row(row: pd.Series, home_team_col: str, away_team_col: str) -> str | None:
    possession_team = row.get("possession_team")
    if home_team_col in row.index and away_team_col in row.index:
        if possession_team == row.get(away_team_col):
            return row.get(home_team_col)
        if possession_team == row.get(home_team_col):
            return row.get(away_team_col)
    return None


def add_event_context(events: pd.DataFrame, home_team_col: str = "home_team", away_team_col: str = "away_team") -> pd.DataFrame:
    df = sort_events(events.copy())
    df["actor_team"] = df.get("team", df.get("team_name"))
    df["possession_team"] = df.get("possession_team", df.get("team_in_possession"))
    if "team_in_possession" not in df.columns:
        df["team_in_possession"] = df["possession_team"]

    group_cols = ["match_id", "period"]
    event_type = df.get("event_type", df.get("type", pd.Series(index=df.index, dtype=object)))
    df["previous_event_team"] = df.groupby(group_cols, dropna=False)["actor_team"].shift(1)
    df["previous_possession_team"] = df.groupby(group_cols, dropna=False)["possession_team"].shift(1)
    df["next_possession_team"] = df.groupby(group_cols, dropna=False)["possession_team"].shift(-1)
    df["possession_id_before_action"] = df.get("possession")
    df["previous_possession_id"] = df.groupby(group_cols, dropna=False)["possession"].shift(1) if "possession" in df else None
    df["possession_id_after_action"] = df.groupby(group_cols, dropna=False)["possession"].shift(-1) if "possession" in df else None
    if "possession" in df:
        previous_possession_id = df.groupby(group_cols, dropna=False)["possession"].shift(1)
        df["event_starts_new_possession"] = previous_possession_id.notna() & (df["possession"] != previous_possession_id)
    else:
        df["event_starts_new_possession"] = False

    df["attacking_team_before_action"] = df["possession_team"]
    df["defending_team_before_action"] = df.apply(lambda r: _opponent_for_row(r, home_team_col, away_team_col), axis=1)

    recovery_mask = event_type.isin(["Ball Recovery", "Interception"])
    use_prior = (
        recovery_mask
        & df["event_starts_new_possession"].fillna(False)
        & df["previous_possession_team"].notna()
        & df["actor_team"].notna()
        & (df["previous_possession_team"] != df["actor_team"])
    )
    df.loc[use_prior, "attacking_team_before_action"] = df.loc[use_prior, "previous_possession_team"]
    df.loc[use_prior, "defending_team_before_action"] = df.loc[use_prior, "actor_team"]

    df["opponent_team"] = df["defending_team_before_action"]
    context_known = df["actor_team"].notna() & df["attacking_team_before_action"].notna() & df["defending_team_before_action"].notna()
    context_consistent = context_known & (df["attacking_team_before_action"] != df["defending_team_before_action"])
    known_event_family = event_type.isin(["Pressure", "Ball Recovery", "Interception", "Clearance", "Block", "Duel", "50/50", "Foul Committed"])
    df["event_semantics_known"] = known_event_family & context_consistent

    df["actor_was_attacking"] = context_consistent & (df["actor_team"] == df["attacking_team_before_action"])
    df["actor_was_defending"] = context_consistent & (df["actor_team"] == df["defending_team_before_action"])
    possession_team_changes_next = df["next_possession_team"].notna() & (df["next_possession_team"] != df["possession_team"])
    df["action_changed_possession"] = possession_team_changes_next | use_prior
    df["action_ended_possession"] = df["action_changed_possession"]
    df["action_won_possession"] = df["action_changed_possession"] & df["actor_was_defending"]
    df["action_retained_defensive_team_control"] = df["actor_was_defending"] & ~df["action_won_possession"]
    df["action_was_under_opponent_possession"] = context_consistent & (df["actor_team"] != df["attacking_team_before_action"])
    df["event_time_seconds"] = event_time_seconds_df(df)
    return df

def validate_event_context(events: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    df = sort_events(events)
    if {"attacking_team_before_action", "defending_team_before_action"}.issubset(df.columns):
        bad = df["attacking_team_before_action"].notna() & (df["attacking_team_before_action"] == df["defending_team_before_action"])
        if bad.any(): issues.append(f"attacking_equals_defending={int(bad.sum())}")
    if {"match_id", "period", "index"}.issubset(df.columns):
        for _, g in df.groupby(["match_id", "period"], dropna=False):
            if not pd.to_numeric(g["index"], errors="coerce").is_monotonic_increasing:
                issues.append("event_order_not_monotonic"); break
    if {"match_id", "period", "possession"}.issubset(df.columns):
        for _, g in df.groupby(["match_id", "period"], dropna=False):
            s = pd.to_numeric(g["possession"], errors="coerce").dropna()
            if not s.is_monotonic_increasing:
                issues.append("possession_moves_backwards"); break
    return issues


def semantic_qa_by_event_type(events: pd.DataFrame) -> pd.DataFrame:
    df = add_event_context(events) if "actor_team" not in events.columns else events.copy()
    typ = df.get("event_type", df.get("type"))
    df = df.assign(_event_type=typ)
    rows=[]
    for name,g in df.groupby("_event_type", dropna=False):
        n=len(g); rows.append({"event_type":name,"events":n,"pct_actor_eq_possession":float((g.actor_team==g.possession_team).mean()) if n else 0,"pct_actor_diff_possession":float((g.actor_team.notna() & g.possession_team.notna() & (g.actor_team!=g.possession_team)).mean()) if n else 0,"pct_changes_possession":float(g.get("action_changed_possession", False).mean()) if n else 0,"pct_missing_team_context":float((g.actor_team.isna() | g.possession_team.isna()).mean()) if n else 0})
    return pd.DataFrame(rows)
