"""Step 03: Univariate signal screening."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import CATEGORICAL_CANDIDATES, NUMERIC_CANDIDATES
from .utils_stats import cliffs_delta, cramers_v


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Assess independent feature signal against the target."""
    y = df["target_future_shot_10s"]

    num_rows: list[dict[str, Any]] = []
    for col in NUMERIC_CANDIDATES:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.size < 100 or valid.nunique() < 4:
            continue
        y_valid = y.loc[valid.index]
        corr = np.nan
        if y_valid.nunique() > 1 and valid.nunique() > 1:
            corr = float(valid.corr(y_valid))
        pos = valid[y_valid == 1].to_numpy()
        neg = valid[y_valid == 0].to_numpy()
        delta = cliffs_delta(pos, neg)
        try:
            q = pd.qcut(valid, q=5, duplicates="drop")
            q_rates = pd.DataFrame({"q": q, "y": y.loc[valid.index]}).groupby("q", observed=True)["y"].mean()
            trend = float(q_rates.iloc[-1] - q_rates.iloc[0]) if len(q_rates) >= 2 else np.nan
        except ValueError:
            trend = np.nan
        num_rows.append(
            {
                "feature": col,
                "n_valid": int(valid.size),
                "corr": corr,
                "cliffs_delta": float(delta) if np.isfinite(delta) else np.nan,
                "q5_minus_q1_shot_rate": trend,
            }
        )

    num_df = pd.DataFrame(num_rows).sort_values("corr", key=lambda s: s.abs(), ascending=False)
    num_df.to_csv(cfg.tables_dir / "03_univariate_numeric_signal.csv", index=False)

    cat_rows: list[dict[str, Any]] = []
    for col in CATEGORICAL_CANDIDATES:
        if col not in df.columns:
            continue
        tab = pd.crosstab(df[col].fillna("<NA>"), y)
        if tab.shape[0] < 2:
            continue
        v = cramers_v(tab.to_numpy(dtype=float))
        cat_rows.append({"feature": col, "levels": int(tab.shape[0]), "cramers_v": v})

    cat_df = pd.DataFrame(cat_rows).sort_values("cramers_v", ascending=False)
    cat_df.to_csv(cfg.tables_dir / "03_univariate_categorical_signal.csv", index=False)

    strong_num = int((num_df["corr"].abs() >= 0.03).sum()) if not num_df.empty else 0
    strong_cat = int((cat_df["cramers_v"] >= 0.05).sum()) if not cat_df.empty else 0

    return {
        "numeric_features_tested": int(len(num_df)),
        "categorical_features_tested": int(len(cat_df)),
        "strong_numeric_signal_count": strong_num,
        "strong_categorical_signal_count": strong_cat,
        "pass": strong_num >= 5 and strong_cat >= 2,
    }


