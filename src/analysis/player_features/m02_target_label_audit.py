"""Step 02: Target and label audit."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AnalysisConfig
from .utils_stats import wilson_interval


def _group_rates(df: pd.DataFrame, col: str, min_size: int) -> pd.DataFrame:
    grouped = (
        df.groupby(col, dropna=False)["target_future_shot_10s"]
        .agg(size="size", shots="sum", shot_rate="mean")
        .reset_index()
    )
    grouped = grouped[grouped["size"] >= min_size].copy()
    grouped[["ci_low", "ci_high"]] = grouped.apply(
        lambda r: pd.Series(wilson_interval(int(r["shots"]), int(r["size"]))), axis=1
    )
    return grouped.sort_values("shot_rate", ascending=False)


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Audit label prevalence and subgroup reliability."""
    base_rate = float(df["target_future_shot_10s"].mean())
    base_ci = wilson_interval(int(df["target_future_shot_10s"].sum()), int(len(df)))

    phase = _group_rates(df, "phase_label", cfg.min_group_size)
    action = _group_rates(df, "action_family", cfg.min_group_size)
    role = _group_rates(df, "position_group", cfg.min_group_size)

    phase.to_csv(cfg.tables_dir / "02_target_by_phase.csv", index=False)
    action.to_csv(cfg.tables_dir / "02_target_by_action_family.csv", index=False)
    role.to_csv(cfg.tables_dir / "02_target_by_position_group.csv", index=False)

    unstable_phase_cells = int((phase["ci_high"] - phase["ci_low"] > 0.08).sum()) if not phase.empty else 0

    return {
        "base_rate": base_rate,
        "base_rate_ci_low": float(base_ci[0]),
        "base_rate_ci_high": float(base_ci[1]),
        "phase_groups": int(len(phase)),
        "action_groups": int(len(action)),
        "role_groups": int(len(role)),
        "unstable_phase_cells": unstable_phase_cells,
        "pass": unstable_phase_cells <= max(2, int(len(phase) * 0.25)) if len(phase) > 0 else False,
    }

