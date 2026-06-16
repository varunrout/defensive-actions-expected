"""Tabular analyses used to justify the player-defense model feature set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import NUMERIC_CANDIDATES


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()),
        "players": int(df["player_id"].nunique()),
        "phases": int(df["phase_label"].nunique(dropna=True)),
        "action_families": int(df["action_family"].nunique(dropna=True)),
        "shot_rate": float(df["target_shot_in_10s"].mean()),
    }


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.isna().mean() * 100).rename("missing_pct").reset_index().rename(columns={"index": "feature"})
    return out.sort_values("missing_pct", ascending=False)


def grouped_rate_table(df: pd.DataFrame, group_col: str, min_size: int = 1) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)["target_shot_in_10s"]
        .agg(size="size", shot_rate="mean")
        .sort_values("shot_rate", ascending=False)
        .reset_index()
    )
    return out[out["size"] >= min_size].reset_index(drop=True)


def phase_position_table(df: pd.DataFrame, min_size: int = 30) -> pd.DataFrame:
    out = (
        df.groupby(["phase_label", "position_group"], dropna=False)["target_shot_in_10s"]
        .agg(size="size", shot_rate="mean")
        .reset_index()
    )
    return out[out["size"] >= min_size].sort_values("shot_rate", ascending=False)


def numeric_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = df["target_shot_in_10s"]

    for col in NUMERIC_CANDIDATES:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.shape[0] < 50 or valid.nunique() <= 1:
            continue

        corr = valid.corr(y.loc[valid.index])
        try:
            q = pd.qcut(valid.rank(method="first"), q=4, duplicates="drop")
            bucket = pd.DataFrame({"bucket": q, "target": y.loc[valid.index]}).groupby("bucket", observed=True)["target"].mean()
            q1 = float(bucket.iloc[0]) if len(bucket) > 0 else np.nan
            q4 = float(bucket.iloc[-1]) if len(bucket) > 0 else np.nan
            lift_q4_vs_q1 = q4 - q1
        except ValueError:
            q1 = np.nan
            q4 = np.nan
            lift_q4_vs_q1 = np.nan

        rows.append(
            {
                "feature": col,
                "n_valid": int(valid.shape[0]),
                "corr_with_target": float(corr) if pd.notna(corr) else np.nan,
                "q1_shot_rate": q1,
                "q4_shot_rate": q4,
                "lift_q4_vs_q1": lift_q4_vs_q1,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("corr_with_target", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def player_reliability_table(df: pd.DataFrame, min_actions: int = 80) -> pd.DataFrame:
    out = (
        df.groupby(["player_id", "player", "position_group"], dropna=False)["target_shot_in_10s"]
        .agg(actions="size", shot_rate="mean")
        .reset_index()
    )
    out = out[out["actions"] >= min_actions].copy()
    out["empirical_bayes_rate"] = (
        (out["shot_rate"] * out["actions"] + df["target_shot_in_10s"].mean() * 200) / (out["actions"] + 200)
    )
    return out.sort_values("actions", ascending=False).reset_index(drop=True)


def bootstrap_signal_stability(df: pd.DataFrame, n_boot: int = 100, sample_frac: float = 0.7) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    available = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    if not available:
        return pd.DataFrame()

    stats: dict[str, list[float]] = {c: [] for c in available}
    for _ in range(n_boot):
        sample = df.sample(frac=sample_frac, replace=True, random_state=int(rng.integers(1, 1_000_000)))
        y = sample["target_shot_in_10s"]
        for col in available:
            s = pd.to_numeric(sample[col], errors="coerce").dropna()
            if s.shape[0] < 30 or s.nunique() <= 1:
                continue
            corr = s.corr(y.loc[s.index])
            if pd.notna(corr):
                stats[col].append(float(corr))

    rows = []
    for col, vals in stats.items():
        if len(vals) < 20:
            continue
        rows.append(
            {
                "feature": col,
                "n_boot": len(vals),
                "corr_mean": float(np.mean(vals)),
                "corr_std": float(np.std(vals)),
                "corr_p10": float(np.percentile(vals, 10)),
                "corr_p90": float(np.percentile(vals, 90)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("corr_mean", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

