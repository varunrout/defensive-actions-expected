from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def match_level_bootstrap_ci(
    frame: pd.DataFrame,
    value_col: str,
    match_col: str = "match_id",
    statistic: Callable[[pd.Series], float] | None = None,
    n_boot: int = 1000,
    seed: int = 7,
) -> dict[str, float | int]:
    """Return a match-cluster bootstrap confidence interval for a column mean.

    Matches, not rows, are resampled. This keeps repeated actions from the same
    match together and is the default uncertainty convention for coach reports.
    """
    if statistic is None:
        def statistic(s: pd.Series) -> float:
            return float(pd.to_numeric(s, errors="coerce").mean())
    if frame.empty or value_col not in frame.columns:
        return {"value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "matches": 0, "actions": 0}
    if match_col not in frame.columns:
        values = pd.to_numeric(frame[value_col], errors="coerce").dropna()
        value = float(statistic(values)) if len(values) else np.nan
        return {"value": value, "ci_low": np.nan, "ci_high": np.nan, "matches": 0, "actions": int(len(frame))}
    clean = frame[[match_col, value_col]].dropna(subset=[value_col]).copy()
    if clean.empty:
        return {"value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "matches": 0, "actions": int(len(frame))}
    matches = clean[match_col].drop_duplicates().to_numpy()
    value = float(statistic(clean[value_col]))
    if len(matches) < 2:
        return {"value": value, "ci_low": np.nan, "ci_high": np.nan, "matches": int(len(matches)), "actions": int(len(clean))}
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    grouped = {mid: grp[value_col] for mid, grp in clean.groupby(match_col, sort=False)}
    for _ in range(n_boot):
        sampled = rng.choice(matches, size=len(matches), replace=True)
        vals = pd.concat([grouped[mid] for mid in sampled], ignore_index=True)
        stats.append(float(statistic(vals)))
    return {
        "value": value,
        "ci_low": float(np.nanquantile(stats, 0.025)),
        "ci_high": float(np.nanquantile(stats, 0.975)),
        "matches": int(len(matches)),
        "actions": int(len(clean)),
    }


def add_match_bootstrap_by_group(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    match_col: str = "match_id",
    n_boot: int = 1000,
    seed: int = 7,
) -> pd.DataFrame:
    """Calculate match-level bootstrap intervals for each subgroup."""
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, "value", "ci_low", "ci_high", "matches", "actions"])
    for keys, grp in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        row.update(match_level_bootstrap_ci(grp, value_col, match_col, n_boot=n_boot, seed=seed))
        rows.append(row)
    return pd.DataFrame(rows)
