"""Step 05: Stratified consistency checks (phase/role/action)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AnalysisConfig

CHECK_FEATURES = [
    "distance_to_attacking_goal",
    "distance_to_center_line",
    "local_numerical_balance_10m",
    "teammate_opponent_ratio",
]


def _corr_by_group(df: pd.DataFrame, group_col: str, feature: str, min_size: int) -> pd.DataFrame:
    rows = []
    if feature not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    for group, chunk in df.groupby(group_col, dropna=False):
        if len(chunk) < min_size:
            continue
        s = pd.to_numeric(chunk[feature], errors="coerce").dropna()
        if len(s) < min_size:
            continue
        y = chunk.loc[s.index, "target_future_shot_10s"]
        if s.nunique() <= 1 or y.nunique() <= 1:
            continue
        corr = s.corr(y)
        rows.append({"group_col": group_col, "group": str(group), "feature": feature, "n": int(len(s)), "corr": float(corr)})
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Check whether key feature directions are stable within football strata."""
    all_rows = []
    for feature in CHECK_FEATURES:
        for group_col in ["phase_label", "position_group", "action_family"]:
            out = _corr_by_group(df, group_col, feature, cfg.min_group_size)
            if not out.empty:
                all_rows.append(out)

    strat = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    strat.to_csv(cfg.tables_dir / "05_stratified_correlations.csv", index=False)

    if strat.empty:
        return {"rows": 0, "pass": False}

    sign_flip_rows = 0
    for feature, fdf in strat.groupby("feature"):
        has_pos = (fdf["corr"] > 0).any()
        has_neg = (fdf["corr"] < 0).any()
        if has_pos and has_neg:
            sign_flip_rows += 1

    return {
        "rows": int(len(strat)),
        "features_with_sign_flip": int(sign_flip_rows),
        "pass": sign_flip_rows <= 2,
    }


