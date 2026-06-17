"""Step 07: Stability and distribution shift checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .io import NUMERIC_CANDIDATES
from .utils_stats import population_stability_index


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Evaluate feature drift across match groups and grouped bootstrap stability."""
    if "match_id" not in df.columns:
        return {"pass": False, "reason": "match_id missing"}

    match_ids = df["match_id"].dropna().drop_duplicates().to_list()
    if len(match_ids) < 4:
        return {"pass": False, "reason": "insufficient match coverage"}

    midpoint = len(match_ids) // 2
    early_ids = set(match_ids[:midpoint])
    late_ids = set(match_ids[midpoint:])

    early = df[df["match_id"].isin(early_ids)]
    late = df[df["match_id"].isin(late_ids)]

    rows = []
    for col in NUMERIC_CANDIDATES:
        if col not in df.columns:
            continue
        base = pd.to_numeric(early[col], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(late[col], errors="coerce").to_numpy(dtype=float)
        psi = population_stability_index(base, current)
        rows.append({"feature": col, "psi_early_vs_late": psi})

    drift = pd.DataFrame(rows).sort_values("psi_early_vs_late", ascending=False)
    drift.to_csv(cfg.tables_dir / "07_feature_drift_psi.csv", index=False)

    # Grouped bootstrap over matches for signal stability.
    rng = np.random.default_rng(cfg.random_seed)
    corr_rows = []
    for col in [c for c in NUMERIC_CANDIDATES if c in df.columns]:
        vals = []
        for _ in range(cfg.bootstrap_iterations):
            sampled_matches = rng.choice(match_ids, size=max(2, int(len(match_ids) * cfg.bootstrap_sample_frac)), replace=True)
            sample = df[df["match_id"].isin(sampled_matches)]
            s = pd.to_numeric(sample[col], errors="coerce").dropna()
            if len(s) < cfg.min_group_size:
                continue
            y = sample.loc[s.index, "target_future_shot_10s"]
            if s.nunique() <= 1 or y.nunique() <= 1:
                continue
            corr = s.corr(y)
            if pd.notna(corr):
                vals.append(float(corr))
        if len(vals) >= 25:
            corr_rows.append(
                {
                    "feature": col,
                    "n_boot": len(vals),
                    "corr_mean": float(np.mean(vals)),
                    "corr_std": float(np.std(vals)),
                    "corr_p10": float(np.percentile(vals, 10)),
                    "corr_p90": float(np.percentile(vals, 90)),
                }
            )
    stability = pd.DataFrame(corr_rows).sort_values("corr_mean", key=lambda s: s.abs(), ascending=False)
    stability.to_csv(cfg.tables_dir / "07_grouped_bootstrap_stability.csv", index=False)

    unstable_drift = int((drift["psi_early_vs_late"] > 0.25).sum()) if not drift.empty else 0
    return {
        "features_tested": int(len(drift)),
        "unstable_drift_features": unstable_drift,
        "stability_features": int(len(stability)),
        "pass": unstable_drift <= 4 and len(stability) >= 5,
    }


