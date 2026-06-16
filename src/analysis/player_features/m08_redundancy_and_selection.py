"""Step 08: Redundancy checks and feature shortlist proposal."""

from __future__ import annotations

from typing import Any

import json

import pandas as pd

from .config import AnalysisConfig
from .io import NUMERIC_CANDIDATES


def run(df: pd.DataFrame, cfg: AnalysisConfig) -> dict[str, Any]:
    """Build core/extended feature sets from missingness and correlation structure."""
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    if not numeric:
        return {"core_feature_count": 0, "extended_feature_count": 0, "pass": False}

    num_df = df[numeric].apply(pd.to_numeric, errors="coerce")
    miss = num_df.isna().mean() * 100.0
    corr = num_df.corr(numeric_only=True)

    corr_pairs = []
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1 :]:
            v = corr.loc[c1, c2]
            if pd.notna(v) and abs(v) >= 0.90:
                corr_pairs.append({"feature_a": c1, "feature_b": c2, "abs_corr": abs(float(v))})
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("abs_corr", ascending=False)
    corr_pairs_df.to_csv(cfg.tables_dir / "08_redundant_feature_pairs.csv", index=False)

    # Keep features with acceptable missingness for core set.
    core = [c for c in numeric if miss[c] <= cfg.max_feature_missing_for_core]
    # Drop one side of very high-correlation pairs from core.
    drop = set()
    for _, row in corr_pairs_df.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        if a in core and b in core:
            # Keep lower missingness feature.
            drop.add(a if miss[a] > miss[b] else b)
    core = [c for c in core if c not in drop]

    extended = [c for c in numeric if miss[c] <= 40.0]

    (cfg.report_dir / "features_core.json").write_text(json.dumps(core, indent=2), encoding="utf-8")
    (cfg.report_dir / "features_extended.json").write_text(json.dumps(extended, indent=2), encoding="utf-8")

    summary = pd.DataFrame(
        {
            "feature": numeric,
            "missing_pct": [float(miss[c]) for c in numeric],
            "in_core": [int(c in core) for c in numeric],
            "in_extended": [int(c in extended) for c in numeric],
        }
    ).sort_values(["in_core", "missing_pct"], ascending=[False, True])
    summary.to_csv(cfg.tables_dir / "08_feature_selection_summary.csv", index=False)

    return {
        "numeric_features_total": int(len(numeric)),
        "core_feature_count": int(len(core)),
        "extended_feature_count": int(len(extended)),
        "high_corr_pairs": int(len(corr_pairs_df)),
        "pass": len(core) >= 8,
    }

