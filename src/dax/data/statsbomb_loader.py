"""
Real StatsBomb data loader using statsbombpy.

This module is the primary data-access layer for the DAx project.
It wraps statsbombpy calls and returns cleaned, typed DataFrames with
consistent column names that feed the rest of the DAx pipeline.

Target tournaments:
  - UEFA Euro 2020       competition_id=55  season_id=43
  - FIFA World Cup 2022  competition_id=43  season_id=106
  - UEFA Euro 2024       competition_id=55  season_id=282

360 data is available for World Cup 2022 and Euro 2024.
Euro 2020 open data does NOT include 360 frames.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsbombpy import sb

# ── Competition registry ──────────────────────────────────────────────────────

TARGET_COMPETITIONS: list[dict[str, Any]] = [
    {"competition_id": 55,  "season_id": 43,  "label": "Euro 2020"},
    {"competition_id": 43,  "season_id": 106, "label": "World Cup 2022"},
    {"competition_id": 55,  "season_id": 282, "label": "Euro 2024"},
]

# 360 freeze-frame data is only available for these season/competition pairs
COMPETITIONS_WITH_360: set[tuple[int, int]] = {
    (43, 106),   # World Cup 2022
    (55, 282),   # Euro 2024
}


# ── Core loaders ─────────────────────────────────────────────────────────────

def load_all_competitions() -> pd.DataFrame:
    """Return the full StatsBomb open-data competition catalogue."""
    return sb.competitions()


def load_target_competitions() -> pd.DataFrame:
    """Return only the three tournaments we care about."""
    all_comps = load_all_competitions()
    keys = {(r["competition_id"], r["season_id"]) for r in TARGET_COMPETITIONS}
    mask = [
        (row.competition_id, row.season_id) in keys
        for row in all_comps.itertuples(index=False)
    ]
    return all_comps[mask].reset_index(drop=True)


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Load all matches for one competition/season."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches["competition_id"] = competition_id
    matches["season_id"] = season_id
    return matches


def load_all_target_matches() -> pd.DataFrame:
    """Load matches for all three target competitions and concatenate."""
    frames: list[pd.DataFrame] = []
    for comp in TARGET_COMPETITIONS:
        matches = load_matches(comp["competition_id"], comp["season_id"])
        matches["competition_label"] = comp["label"]
        frames.append(matches)
    return pd.concat(frames, ignore_index=True)


def load_events(match_id: int) -> pd.DataFrame:
    """
    Load events for a single match using flatten_attrs=True.

    Note: 360 freeze-frame data is loaded separately via load_360_json()
    and joined by event UUID in build_enriched_events().
    """
    events = sb.events(match_id=match_id, flatten_attrs=True)
    events["match_id"] = match_id
    return events


def load_lineups(match_id: int) -> pd.DataFrame:
    """Load lineups for a single match (returns a dict keyed by team name)."""
    raw = sb.lineups(match_id=match_id)
    frames: list[pd.DataFrame] = []
    for team_name, lineup_df in raw.items():
        lineup_df = lineup_df.copy()
        lineup_df["team_name"] = team_name
        lineup_df["match_id"] = match_id
        frames.append(lineup_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_360_json(match_id: int) -> list[dict[str, Any]]:
    """
    Load raw 360 freeze-frame data for one match as a list of dicts.

    Each dict has keys: event_uuid, visible_area, match_id, freeze_frame.

    Uses fmt='json' to bypass the statsbombpy duplicate-index bug in the
    DataFrame formatter for 360 data.

    Returns an empty list if 360 data is unavailable.
    """
    try:
        raw = sb.frames(match_id=match_id, fmt="json")
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def load_360(match_id: int) -> pd.DataFrame:
    """Convenience wrapper: returns 360 data as a DataFrame (may be empty)."""
    raw = load_360_json(match_id)
    if not raw:
        return pd.DataFrame()
    return pd.DataFrame(raw)


# ── Enriched event table ──────────────────────────────────────────────────────

def build_enriched_events(
    match_id: int,
    competition_id: int,
    season_id: int,
    competition_label: str,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """
    Build the core DAx event row for one match.

    Returns a clean DataFrame with:
      match context · ball coordinates · possession context ·
      freeze-frame summary · visibility flag
    """
    has_360 = (competition_id, season_id) in COMPETITIONS_WITH_360
    events = load_events(match_id=match_id)

    if events.empty:
        return pd.DataFrame()

    # ── ball coordinates ──────────────────────────────────────────────────────
    if "location" in events.columns:
        events["ball_x"] = events["location"].apply(
            lambda loc: float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else None
        )
        events["ball_y"] = events["location"].apply(
            lambda loc: float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else None
        )
    else:
        events["ball_x"] = None
        events["ball_y"] = None

    # ── 360 freeze-frame join ─────────────────────────────────────────────────
    if has_360:
        raw_360 = load_360_json(match_id)
        frame_map: dict[str, dict[str, Any]] = {
            f["event_uuid"]: f for f in raw_360 if "event_uuid" in f
        }

        events["freeze_frame"] = events["id"].map(
            lambda eid: frame_map.get(eid, {}).get("freeze_frame", [])
        )
        events["visible_area"] = events["id"].map(
            lambda eid: frame_map.get(eid, {}).get("visible_area")
        )
    else:
        events["freeze_frame"] = [[] for _ in range(len(events))]
        events["visible_area"] = None

    events["freeze_frame_count"] = events["freeze_frame"].apply(
        lambda ff: len(ff) if isinstance(ff, list) else 0
    )
    events["teammate_count"] = events["freeze_frame"].apply(
        lambda ff: sum(1 for p in ff if p.get("teammate") is True)
        if isinstance(ff, list) else 0
    )
    events["opponent_count"] = events["freeze_frame"].apply(
        lambda ff: sum(1 for p in ff if p.get("teammate") is False)
        if isinstance(ff, list) else 0
    )
    events["has_360"] = events["freeze_frame_count"] > 0
    events["visibility_limited"] = events["visible_area"].isna()

    # ── defending team ────────────────────────────────────────────────────────
    def _defending(team_name: Any) -> str | None:
        if team_name == home_team:
            return away_team
        if team_name == away_team:
            return home_team
        return None

    poss_team_col = "possession_team" if "possession_team" in events.columns else None
    if poss_team_col:
        events["team_in_possession"] = events[poss_team_col]
        events["defending_team"] = events[poss_team_col].apply(_defending)
    else:
        events["team_in_possession"] = None
        events["defending_team"] = None

    # ── competition meta ──────────────────────────────────────────────────────
    events["competition_id"] = competition_id
    events["season_id"] = season_id
    events["competition_label"] = competition_label
    events["home_team"] = home_team
    events["away_team"] = away_team

    return events


def build_all_enriched_events(
    max_matches_per_competition: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and enrich all events across all three target competitions.

    Parameters
    ----------
    max_matches_per_competition:
        If provided, limit to first N matches per competition.
        Useful for quick development iterations.
    verbose:
        Print progress.
    """
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(iterable, **kwargs):  # type: ignore[misc]
            return iterable

    all_frames: list[pd.DataFrame] = []

    for comp in TARGET_COMPETITIONS:
        comp_id = comp["competition_id"]
        season_id = comp["season_id"]
        label = comp["label"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading: {label}  (competition={comp_id}, season={season_id})")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matches = load_matches(comp_id, season_id)

        if max_matches_per_competition is not None:
            matches = matches.head(max_matches_per_competition)

        if verbose:
            print(f"  {len(matches)} matches to process")

        for _, match_row in _tqdm(
            matches.iterrows(),
            total=len(matches),
            desc=label,
            disable=not verbose,
        ):
            match_id = int(match_row["match_id"])
            home_team = match_row.get("home_team", "")
            away_team = match_row.get("away_team", "")

            try:
                enriched = build_enriched_events(
                    match_id=match_id,
                    competition_id=comp_id,
                    season_id=season_id,
                    competition_label=label,
                    home_team=home_team,
                    away_team=away_team,
                )
                if not enriched.empty:
                    all_frames.append(enriched)
            except Exception as exc:
                if verbose:
                    print(f"  [WARN] match {match_id} failed: {exc}")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    if verbose:
        print(f"\nDone. Total rows: {len(combined):,}")
    return combined




