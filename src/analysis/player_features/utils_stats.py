"""Statistical helpers for rigorous player-feature analysis."""

from __future__ import annotations

from typing import Callable

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z * z / (4.0 * n * n)))
    return (max(0.0, center - radius), min(1.0, center + radius))


def bootstrap_ci(
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    if values.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        stats.append(stat_fn(sample))
    low = np.percentile(stats, 100.0 * (alpha / 2.0))
    high = np.percentile(stats, 100.0 * (1.0 - alpha / 2.0))
    return (float(low), float(high))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size (non-parametric)."""
    if x.size == 0 or y.size == 0:
        return np.nan
    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    return float((gt - lt) / (x.size * y.size))


def cramers_v(confusion: np.ndarray) -> float:
    """Cramer's V without scipy dependency."""
    if confusion.size == 0:
        return np.nan
    total = confusion.sum()
    if total <= 0:
        return np.nan
    row_sum = confusion.sum(axis=1, keepdims=True)
    col_sum = confusion.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((confusion - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    r, c = confusion.shape
    k = min(r - 1, c - 1)
    if k <= 0:
        return np.nan
    return float(np.sqrt((chi2 / total) / k))


def population_stability_index(base: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two numeric distributions."""
    base = base[np.isfinite(base)]
    current = current[np.isfinite(current)]
    if base.size < 20 or current.size < 20:
        return np.nan

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(base, quantiles))
    if edges.size < 3:
        return np.nan

    base_counts, _ = np.histogram(base, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    base_pct = np.clip(base_counts / max(base_counts.sum(), 1), 1e-6, 1.0)
    curr_pct = np.clip(curr_counts / max(curr_counts.sum(), 1), 1e-6, 1.0)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)

