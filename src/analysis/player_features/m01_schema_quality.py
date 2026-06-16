"""Step 01: Schema and quality checks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AnalysisConfig
from .io import REQUIRED_COLUMNS


KEY_COLUMNS = ["match_id", "event_id", "player_id"]
CORE_MISSINGNESS_COLUMNS = sorted(REQUIRED_COLUMNS | {"phase_label", "action_family", "position_group", "nearest_goal_distance"})


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Run schema/quality checks and save artifacts."""
    missing = (df.isna().mean() * 100.0).rename("missing_pct").reset_index().rename(columns={"index": "feature"})
    missing = missing.sort_values("missing_pct", ascending=False)
    missing.to_csv(cfg.tables_dir / "01_missingness.csv", index=False)

    coord_issues = pd.DataFrame(
        {
            "metric": [
                "action_x_out_of_bounds_pct",
                "action_y_out_of_bounds_pct",
                "target_not_binary_pct",
            ],
            "value": [
                float(((df.get("action_x", pd.Series(dtype=float)) < 0) | (df.get("action_x", pd.Series(dtype=float)) > 120)).mean() * 100.0)
                if "action_x" in df.columns
                else 0.0,
                float(((df.get("action_y", pd.Series(dtype=float)) < 0) | (df.get("action_y", pd.Series(dtype=float)) > 80)).mean() * 100.0)
                if "action_y" in df.columns
                else 0.0,
                float((~df["target_shot_in_10s"].isin([0, 1])).mean() * 100.0),
            ],
        }
    )
    coord_issues.to_csv(cfg.tables_dir / "01_value_range_checks.csv", index=False)

    duplicates_pct = 0.0
    if all(c in df.columns for c in KEY_COLUMNS):
        duplicates_pct = float(df.duplicated(KEY_COLUMNS).mean() * 100.0)

    core_cols = [c for c in CORE_MISSINGNESS_COLUMNS if c in df.columns]
    core_missing_max = float((df[core_cols].isna().mean() * 100.0).max()) if core_cols else float("nan")

    checks = {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "players": int(df["player_id"].nunique()),
        "duplicate_key_pct": duplicates_pct,
        "max_missing_pct": float(missing["missing_pct"].max()),
        "core_max_missing_pct": core_missing_max,
    }
    checks["pass_missingness"] = checks["core_max_missing_pct"] <= cfg.max_missing_pct
    checks["pass_duplicates"] = checks["duplicate_key_pct"] <= cfg.max_duplicate_event_pct
    checks["pass_players"] = checks["players"] >= cfg.min_unique_players
    checks["pass"] = bool(checks["pass_missingness"] and checks["pass_duplicates"] and checks["pass_players"])
    return checks


