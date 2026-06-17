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

PITCH_X_MAX = 120.0
PITCH_Y_MAX = 80.0

# ── Competition registry ──────────────────────────────────────────────────────

TARGET_COMPETITIONS: list[dict[str, Any]] = [
    # {"competition_id": 55,  "season_id": 43,  "label": "Euro 2020"},
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

def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _xy_from_any(location: Any) -> tuple[float | None, float | None]:
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        return _as_float(location[0]), _as_float(location[1])
    if hasattr(location, "tolist"):
        try:
            arr = location.tolist()
            if isinstance(arr, list) and len(arr) >= 2:
                return _as_float(arr[0]), _as_float(arr[1])
        except Exception:
            return None, None
    return None, None


def _normalize_xy(x: float | None, y: float | None, sign: int) -> tuple[float | None, float | None]:
    if x is None or y is None:
        return None, None
    if sign < 0:
        return PITCH_X_MAX - x, PITCH_Y_MAX - y
    return x, y


def _extract_end_x(row: pd.Series) -> float | None:
    for col in [
        "pass_end_location",
        "carry_end_location",
        "dribble_end_location",
        "shot_end_location",
    ]:
        if col not in row.index:
            continue
        x, _ = _xy_from_any(row.get(col))
        if x is not None:
            return x
    return None


def _infer_attack_sign_by_period_team(events: pd.DataFrame, home_team: str, away_team: str) -> dict[tuple[int, str], int]:
    """Infer attacking direction sign by (period, team) using progression deltas.

    sign=+1 means coordinates already left-to-right for that attacking team.
    sign=-1 means coordinates need 180-degree flip to become left-to-right.
    """
    work = events.copy()
    if "period" not in work.columns:
        return {}
    if "possession_team" in work.columns:
        work["_attack_team"] = work["possession_team"]
    else:
        work["_attack_team"] = work.get("team")

    work["_x_raw"] = work["raw_ball_x"]
    work["_end_x"] = work.apply(_extract_end_x, axis=1)
    valid = work[
        work["_attack_team"].notna()
        & work["period"].notna()
        & work["_x_raw"].notna()
        & work["_end_x"].notna()
    ].copy()
    valid["_dx"] = pd.to_numeric(valid["_end_x"], errors="coerce") - pd.to_numeric(valid["_x_raw"], errors="coerce")

    grouped = (
        valid.groupby(["period", "_attack_team"], dropna=False)["_dx"]
        .agg(median_dx="median", abs_median_dx=lambda s: float(s.abs().median()))
        .reset_index()
    )

    signs: dict[tuple[int, str], int] = {}
    confidence: dict[tuple[int, str], float] = {}
    for _, row in grouped.iterrows():
        period = int(row["period"])
        team = str(row["_attack_team"])
        median_dx = float(row["median_dx"])
        signs[(period, team)] = 1 if median_dx >= 0 else -1
        confidence[(period, team)] = float(row["abs_median_dx"])

    periods = sorted({int(p) for p in work["period"].dropna().unique().tolist()})
    for period in periods:
        home_key = (period, home_team)
        away_key = (period, away_team)
        home_sign = signs.get(home_key)
        away_sign = signs.get(away_key)

        if home_sign is not None and away_sign is None:
            signs[away_key] = -home_sign
        elif home_sign is None and away_sign is not None:
            signs[home_key] = -away_sign
        elif home_sign is not None and away_sign is not None and home_sign == away_sign:
            # If inference disagrees with football constraints, flip the weaker side.
            home_conf = confidence.get(home_key, 0.0)
            away_conf = confidence.get(away_key, 0.0)
            if home_conf <= away_conf:
                signs[home_key] = -home_sign
            else:
                signs[away_key] = -away_sign

        # Last-resort fallback if both are missing.
        if home_key not in signs and away_key not in signs:
            fallback_home = 1 if period % 2 == 1 else -1
            signs[home_key] = fallback_home
            signs[away_key] = -fallback_home

    return signs


def _normalize_freeze_frame(frame: Any, sign: int) -> list[dict[str, Any]]:
    if not isinstance(frame, list):
        return []
    out: list[dict[str, Any]] = []
    for player in frame:
        if not isinstance(player, dict):
            continue
        entry = dict(player)
        x, y = _xy_from_any(entry.get("location"))
        x_n, y_n = _normalize_xy(x, y, sign)
        if x_n is not None and y_n is not None:
            entry["location"] = [x_n, y_n]
        out.append(entry)
    return out


def _normalize_visible_area(area: Any, sign: int) -> list[float] | None:
    if not isinstance(area, list) or len(area) < 2:
        return None if area is None else area
    out: list[float] = []
    for i in range(0, len(area), 2):
        if i + 1 >= len(area):
            break
        x = _as_float(area[i])
        y = _as_float(area[i + 1])
        x_n, y_n = _normalize_xy(x, y, sign)
        if x_n is None or y_n is None:
            continue
        out.extend([x_n, y_n])
    return out if out else None

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
        events["raw_ball_x"] = events["location"].apply(
            lambda loc: float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else None
        )
        events["raw_ball_y"] = events["location"].apply(
            lambda loc: float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else None
        )
    else:
        events["raw_ball_x"] = None
        events["raw_ball_y"] = None

    # Infer attacking direction by (period, team) and normalize all coordinates
    direction_map = _infer_attack_sign_by_period_team(events, home_team=home_team, away_team=away_team)

    def _row_sign(row: pd.Series) -> int:
        team_key = row.get("possession_team") if "possession_team" in events.columns else row.get("team")
        period = row.get("period")
        if pd.isna(period) or team_key is None:
            return 1
        return int(direction_map.get((int(period), str(team_key)), 1))

    events["attack_dir_sign"] = events.apply(_row_sign, axis=1)
    events["ball_x"] = events.apply(
        lambda r: _normalize_xy(_as_float(r.get("raw_ball_x")), _as_float(r.get("raw_ball_y")), int(r["attack_dir_sign"]))[0],
        axis=1,
    )
    events["ball_y"] = events.apply(
        lambda r: _normalize_xy(_as_float(r.get("raw_ball_x")), _as_float(r.get("raw_ball_y")), int(r["attack_dir_sign"]))[1],
        axis=1,
    )

    if "location" in events.columns:
        events["location"] = events.apply(
            lambda r: (
                [
                    _normalize_xy(_as_float(r.get("raw_ball_x")), _as_float(r.get("raw_ball_y")), int(r["attack_dir_sign"]))[0],
                    _normalize_xy(_as_float(r.get("raw_ball_x")), _as_float(r.get("raw_ball_y")), int(r["attack_dir_sign"]))[1],
                ]
                if _normalize_xy(_as_float(r.get("raw_ball_x")), _as_float(r.get("raw_ball_y")), int(r["attack_dir_sign"]))[0] is not None
                else r.get("location")
            ),
            axis=1,
        )

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

        events["freeze_frame"] = events.apply(
            lambda r: _normalize_freeze_frame(r.get("freeze_frame"), int(r.get("attack_dir_sign") or 1)),
            axis=1,
        )
        events["visible_area"] = events.apply(
            lambda r: _normalize_visible_area(r.get("visible_area"), int(r.get("attack_dir_sign") or 1)),
            axis=1,
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




