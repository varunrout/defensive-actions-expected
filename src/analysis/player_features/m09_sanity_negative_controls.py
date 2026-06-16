"""Step 09: Sanity checks with negative controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import NUMERIC_CANDIDATES


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Verify signal collapses under permutation and random controls."""
    rng = np.random.default_rng(cfg.random_seed)
    y = df["target_shot_in_10s"]

    feature = next((c for c in NUMERIC_CANDIDATES if c in df.columns), None)
    if feature is None:
        return {"pass": False, "reason": "no numeric feature"}

    s = pd.to_numeric(df[feature], errors="coerce").dropna()
    if len(s) < 200:
        return {"pass": False, "reason": "insufficient rows for sanity test"}

    y_valid = y.loc[s.index]
    if y_valid.nunique() <= 1:
        return {"pass": False, "reason": "constant target in valid sample"}
    observed = abs(float(s.corr(y_valid)))

    perm_corr = []
    for _ in range(200):
        shuffled = pd.Series(rng.permutation(y_valid.to_numpy()), index=y_valid.index)
        perm_corr.append(abs(float(s.corr(shuffled))))

    random_noise = pd.Series(rng.normal(0.0, 1.0, size=len(y_valid)), index=y_valid.index)
    noise_corr = abs(float(random_noise.corr(y_valid)))

    stats_df = pd.DataFrame(
        {
            "observed_abs_corr": [observed],
            "perm_mean_abs_corr": [float(np.mean(perm_corr))],
            "perm_p95_abs_corr": [float(np.percentile(perm_corr, 95))],
            "noise_abs_corr": [noise_corr],
        }
    )
    stats_df.to_csv(cfg.tables_dir / "09_negative_control_summary.csv", index=False)

    pass_check = observed > float(np.percentile(perm_corr, 95)) and observed > noise_corr
    return {
        "feature_tested": feature,
        "observed_abs_corr": observed,
        "perm_p95_abs_corr": float(np.percentile(perm_corr, 95)),
        "noise_abs_corr": noise_corr,
        "pass": bool(pass_check),
    }


