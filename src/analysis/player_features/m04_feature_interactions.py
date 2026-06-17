"""Step 04: Interaction analysis for football-context effects."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AnalysisConfig

KEY_NUMERIC = [
    "distance_to_attacking_goal",
    "distance_to_center_line",
    "local_numerical_balance_10m",
    "teammate_opponent_ratio",
    "possession_progress_ratio",
]


def _interaction_table(df: pd.DataFrame, feature: str, context: str, min_size: int) -> pd.DataFrame:
    work = df[[feature, context, "target_future_shot_10s"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=[feature, context])
    if work.empty or work[feature].nunique() < 6:
        return pd.DataFrame()

    work["feature_bin"] = pd.qcut(work[feature], q=4, duplicates="drop")
    out = (
        work.groupby([context, "feature_bin"], observed=True)["target_future_shot_10s"]
        .agg(size="size", shot_rate="mean")
        .reset_index()
    )
    out = out[out["size"] >= min_size]
    out["feature"] = feature
    out["context"] = context
    return out


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Measure interaction variation by phase and role."""
    all_rows = []
    for feature in KEY_NUMERIC:
        if feature not in df.columns:
            continue
        for context in ["phase_label", "position_group"]:
            if context not in df.columns:
                continue
            out = _interaction_table(df, feature, context, cfg.min_group_size)
            if not out.empty:
                all_rows.append(out)

    inter = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    inter.to_csv(cfg.tables_dir / "04_interaction_tables.csv", index=False)

    if inter.empty:
        return {"interaction_rows": 0, "pass": False}

    spreads = (
        inter.groupby(["feature", "context", "feature_bin"], observed=True)["shot_rate"]
        .agg(rate_min="min", rate_max="max")
        .reset_index()
    )
    spreads["spread"] = spreads["rate_max"] - spreads["rate_min"]
    spreads.to_csv(cfg.tables_dir / "04_interaction_spreads.csv", index=False)

    strong_interactions = int((spreads["spread"] >= 0.03).sum())
    return {
        "interaction_rows": int(len(inter)),
        "strong_interaction_cells": strong_interactions,
        "pass": strong_interactions >= 4,
    }

