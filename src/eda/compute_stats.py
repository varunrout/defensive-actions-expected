"""Dataset-agnostic statistics: categorical rates, boolean lift, distributions."""

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

from src.eda.feature_config import (
    CONTINUOUS_BIN_COUNT,
    CONTINUOUS_PERCENTILE_CAP,
    DISCRETE_TOP_N,
    SMALL_N_THRESHOLD,
)


def base_rate(df: pd.DataFrame, target_col: str) -> float:
    return float(df[target_col].mean() * 100)


def categorical_rate_table(df: pd.DataFrame, col: str, target_col: str) -> list[dict]:
    """Shot rate per category, always groupby-mean against the full row population."""
    grouped = df.groupby(df[col].fillna("(missing)"), dropna=False)[target_col].agg(
        n="count", shots="sum", rate=lambda s: s.mean() * 100
    )
    grouped = grouped.sort_values("rate", ascending=False)
    return [
        {"category": str(idx), "n": int(row["n"]), "shots": int(row["shots"]), "rate": float(row["rate"])}
        for idx, row in grouped.iterrows()
    ]


def boolean_lift(df: pd.DataFrame, col: str, target_col: str) -> dict:
    true_mask = df[col] == True  # noqa: E712
    false_mask = df[col] == False  # noqa: E712
    n_true = int(true_mask.sum())
    n_false = int(false_mask.sum())
    rate_true = float(df.loc[true_mask, target_col].mean() * 100) if n_true else float("nan")
    rate_false = float(df.loc[false_mask, target_col].mean() * 100) if n_false else float("nan")
    lift = rate_true - rate_false
    pct_true = float(df[col].mean() * 100)
    return {
        "column": col,
        "rate_true": rate_true,
        "rate_false": rate_false,
        "lift": lift,
        "pct_true": pct_true,
        "n_true": n_true,
        "n_false": n_false,
        "small_n": n_true < SMALL_N_THRESHOLD or n_false < SMALL_N_THRESHOLD,
    }


def continuous_distribution(df: pd.DataFrame, col: str) -> dict:
    x = df[col].dropna().to_numpy(dtype=float)
    p99 = float(np.percentile(x, CONTINUOUS_PERCENTILE_CAP))
    x_min = float(x.min())
    x_max = float(x.max())
    clipped = x[x <= p99]
    if clipped.size == 0 or p99 <= x_min:
        edges = np.linspace(x_min, x_max if x_max > x_min else x_min + 1, CONTINUOUS_BIN_COUNT + 1)
    else:
        edges = np.linspace(x_min, p99, CONTINUOUS_BIN_COUNT + 1)
    counts, _ = np.histogram(clipped, bins=edges)
    n_extreme = int((x > p99).sum())
    return {
        "column": col,
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "skew": float(scipy_skew(x)),
        "p99": p99,
        "max": x_max,
        "min": x_min,
        "n_extreme": n_extreme,
        "n": int(x.size),
        "bin_edges": edges.tolist(),
        "bin_counts": counts.tolist(),
    }


def discrete_distribution(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    counts = s.value_counts().sort_index()
    if len(counts) > DISCRETE_TOP_N:
        head = counts.iloc[: DISCRETE_TOP_N - 1]
        tail_sum = int(counts.iloc[DISCRETE_TOP_N - 1 :].sum())
        bars = [{"value": str(v), "n": int(n), "is_tail": False} for v, n in head.items()]
        bars.append({"value": f"{head.index.max()}+" if len(head) else "20+", "n": tail_sum, "is_tail": True})
    else:
        bars = [{"value": str(v), "n": int(n), "is_tail": False} for v, n in counts.items()]
    return {
        "column": col,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "n": int(s.size),
        "bars": bars,
    }


def check_duplicate_columns(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Verify a known suspected-duplicate column pair; never silently work around it."""
    if col_a not in df.columns or col_b not in df.columns:
        return {"col_a": col_a, "col_b": col_b, "present": False, "is_duplicate": None}
    is_duplicate = bool((df[col_a] == df[col_b]).all())
    return {"col_a": col_a, "col_b": col_b, "present": True, "is_duplicate": is_duplicate}
