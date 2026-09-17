"""Mixed-type association methods and tier classification for the
feature-correlation analysis.

A single correlation coefficient doesn't mean the same thing across
continuous, boolean and categorical columns, so each pair is scored with the
type-matched method:

    continuous <-> continuous   Spearman rho   (monotonic, robust to skew)
    boolean    <-> boolean      phi            (Pearson on 0/1)
    boolean    <-> continuous   point-biserial r
    categorical<-> categorical  Cramer's V (bias-corrected); boolean treated
                                 as a 2-level categorical when paired with one
    categorical<-> continuous   correlation ratio eta

Discrete columns are treated as numeric for this purpose and pooled with
continuous columns under the "numeric" type.

V2 (prompt 7/7): categorical methods (Cramer's V, correlation ratio eta) now
have their own REVIEW band (V in [0.3, 0.4), eta in [0.5, 0.6)) below their
COLLAPSE cut, so categorical pairs are no longer tiered straight from
COLLAPSE to DISTINCT with nothing in between -- see generate_review_analysis.py's
Type 5 for how these (plus qualifying COLLAPSE-tier categorical pairs) get
resolved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_PAIR_N = 50

# Candidate-tier thresholds. A pair only reaches DROP if it clears the
# candidate threshold AND the confirm threshold below -- see classify_pair's
# docstring for why: a coefficient like 0.995 or 0.999 clears |r| >= 0.98 but
# is not "genuinely deterministic", so it is downgraded to COLLAPSE with a
# note rather than allowed to fire DROP on a borderline value.
DROP_R = 0.98
DROP_R_CONFIRM = 0.9999
COLLAPSE_R_LOW = 0.90
REVIEW_R_LOW = 0.50

DROP_V = 0.9
DROP_V_CONFIRM = 0.999
COLLAPSE_V_LOW = 0.4
REVIEW_V_LOW = 0.3

DROP_ETA = 0.98
DROP_ETA_CONFIRM = 0.999
COLLAPSE_ETA_LOW = 0.6
REVIEW_ETA_LOW = 0.5


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")


def correlation_ratio(categories, values) -> float:
    cats = pd.Series(categories).astype("category").cat.codes
    df = pd.DataFrame({"c": cats, "v": values}).dropna()
    grand_mean = df["v"].mean()
    ss_between = df.groupby("c")["v"].apply(lambda g: len(g) * (g.mean() - grand_mean) ** 2).sum()
    ss_total = ((df["v"] - grand_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else float("nan")


def classify_pair(method: str, value: float) -> tuple[str, bool]:
    """Return (tier, downgraded_from_drop) for a computed association value.

    `value` is the signed r for spearman/phi/point_biserial, or the unsigned
    magnitude for cramers_v/correlation_ratio.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "distinct", False

    if method in ("spearman", "phi", "point_biserial"):
        a = abs(value)
        if a >= DROP_R:
            if a > DROP_R_CONFIRM:
                return "drop", False
            return "collapse", True
        if a >= COLLAPSE_R_LOW:
            return "collapse", False
        if a >= REVIEW_R_LOW:
            return "review", False
        return "distinct", False

    if method == "cramers_v":
        v = abs(value)
        if v >= DROP_V:
            if v > DROP_V_CONFIRM:
                return "drop", False
            return "collapse", True
        if v >= COLLAPSE_V_LOW:
            return "collapse", False
        if v >= REVIEW_V_LOW:
            return "review", False
        return "distinct", False

    if method == "correlation_ratio":
        e = abs(value)
        if e >= DROP_ETA:
            if e > DROP_ETA_CONFIRM:
                return "drop", False
            return "collapse", True
        if e >= COLLAPSE_ETA_LOW:
            return "collapse", False
        if e >= REVIEW_ETA_LOW:
            return "review", False
        return "distinct", False

    raise ValueError(f"Unknown method: {method}")


def pair_method(type_a: str, type_b: str) -> str:
    types = {type_a, type_b}
    if types == {"numeric"}:
        return "spearman"
    if types == {"boolean"}:
        return "phi"
    if types == {"boolean", "numeric"}:
        return "point_biserial"
    if types == {"categorical"}:
        return "cramers_v"
    if types == {"categorical", "numeric"}:
        return "correlation_ratio"
    if types == {"categorical", "boolean"}:
        # boolean is treated as a 2-level categorical against a categorical partner
        return "cramers_v"
    raise ValueError(f"Unhandled type combination: {type_a}, {type_b}")


def compute_pair(df: pd.DataFrame, col_a: str, type_a: str, col_b: str, type_b: str) -> dict | None:
    """Compute the type-matched association for one column pair.

    Returns None (and the caller should log a skip) when n < MIN_PAIR_N
    after dropping NaNs.
    """
    method = pair_method(type_a, type_b)
    sub = df[[col_a, col_b]].dropna()
    n = len(sub)
    if n < MIN_PAIR_N:
        return None

    a, b = sub[col_a], sub[col_b]

    if method == "spearman":
        value, _ = stats.spearmanr(a.astype(float), b.astype(float))
    elif method == "phi":
        value, _ = stats.pearsonr(a.astype(int), b.astype(int))
    elif method == "point_biserial":
        bool_col, num_col = (col_a, col_b) if type_a == "boolean" else (col_b, col_a)
        value, _ = stats.pointbiserialr(sub[bool_col].astype(int), sub[num_col].astype(float))
    elif method == "cramers_v":
        value = cramers_v(a, b)
    elif method == "correlation_ratio":
        cat_col, num_col = (col_a, col_b) if type_a == "categorical" else (col_b, col_a)
        value = correlation_ratio(sub[cat_col], sub[num_col].astype(float))
    else:
        raise ValueError(method)

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    tier, downgraded = classify_pair(method, value)
    return {
        "feature_a": col_a,
        "feature_b": col_b,
        "type_a": type_a,
        "type_b": type_b,
        "method": method,
        "value": round(float(value), 4),
        "abs_value": round(float(abs(value)), 4),
        "n": int(n),
        "tier": tier,
        "downgraded_from_drop": downgraded,
    }
