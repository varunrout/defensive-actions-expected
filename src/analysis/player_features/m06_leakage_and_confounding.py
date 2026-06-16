"""Step 06: Leakage and confounding checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AnalysisConfig

SUSPECT_PATTERNS = [
    "target_",
    "_target",
    "future",
    "next_",
    "outcome",
    "in_10s",
]

ALLOWED_FALSE_POSITIVES = {
    "distance_to_left_goal",
    "distance_to_right_goal",
    "nearest_goal_distance",
    "nearest_goal_side",
}


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Flag likely leakage columns and coarse confounding issues."""
    suspect_cols = []
    for c in df.columns:
        c_low = c.lower()
        if c == "target_shot_in_10s" or c in ALLOWED_FALSE_POSITIVES:
            continue
        if any(p in c_low for p in SUSPECT_PATTERNS):
            suspect_cols.append(c)

    # Team-level confounding proxy: variance in team shot rates.
    team_rates = (
        df.groupby("team", dropna=False)["target_shot_in_10s"].agg(size="size", shot_rate="mean").reset_index()
        if "team" in df.columns
        else pd.DataFrame(columns=["team", "size", "shot_rate"])
    )
    team_rates = team_rates[team_rates["size"] >= cfg.min_group_size]
    team_rates.to_csv(cfg.tables_dir / "06_team_rate_confounding.csv", index=False)

    team_rate_std = float(team_rates["shot_rate"].std()) if not team_rates.empty else 0.0

    match_rates = (
        df.groupby("match_id", dropna=False)["target_shot_in_10s"].agg(size="size", shot_rate="mean").reset_index()
    )
    match_rates = match_rates[match_rates["size"] >= cfg.min_group_size]
    match_rates.to_csv(cfg.tables_dir / "06_match_rate_confounding.csv", index=False)
    match_rate_std = float(match_rates["shot_rate"].std()) if not match_rates.empty else 0.0

    leakage_df = pd.DataFrame({"suspect_column": suspect_cols})
    leakage_df.to_csv(cfg.tables_dir / "06_suspect_leakage_columns.csv", index=False)

    return {
        "suspect_leakage_columns": int(len(suspect_cols)),
        "team_rate_std": team_rate_std,
        "match_rate_std": match_rate_std,
        "pass": len(suspect_cols) == 0,
    }



