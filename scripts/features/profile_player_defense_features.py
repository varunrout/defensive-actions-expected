"""Profile the player defensive dataset to support feature selection.

This script is intentionally lightweight and uses pandas only. It prints:
  - dataset size and action family mix
  - shot rates by feature buckets
  - simple correlations for numeric features

Use it after `scripts/features/build_player_defense_dataset.py`.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FEATURES = REPO_ROOT / "data" / "features"

NUMERIC_FEATURES = [
    "nearest_goal_distance",
    "distance_to_center_line",
    "freeze_support_balance_5m",
    "freeze_support_balance_10m",
    "freeze_support_ratio_5m",
    "freeze_support_ratio_10m",
    "freeze_teammate_nearest_distance",
    "freeze_opponent_nearest_distance",
    "freeze_teammate_mean_distance",
    "freeze_opponent_mean_distance",
    "freeze_teammate_spread",
    "freeze_opponent_spread",
    "teammate_count",
    "opponent_count",
    "possession_progress_ratio",
    "phase_transition_count_so_far",
    "seconds_since_possession_start",
]


def _bucket_summary(df: pd.DataFrame, feature: str, target: str = "target_shot_in_10s") -> None:
    series = pd.to_numeric(df[feature], errors="coerce")
    try:
        buckets = pd.qcut(series.rank(method="first"), q=4, duplicates="drop")
    except ValueError:
        print(f"  {feature}: not enough variation")
        return
    tmp = pd.DataFrame({feature: series, target: df[target], "bucket": buckets})
    stats = tmp.groupby("bucket", observed=True)[target].agg(["size", "mean"])
    print(f"\n{feature} bucket shot rates:")
    print(stats.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile player defensive features.")
    parser.add_argument("--input", type=str, default=str(DATA_FEATURES / "player_defensive_actions.parquet"), help="Input parquet file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_parquet(path)
    print("\n" + "=" * 72)
    print("PLAYER DEFENSIVE FEATURE PROFILE")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Matches: {df['match_id'].nunique():,}")
    print(f"Players: {df['player_id'].nunique():,}")
    print(f"Shot rate: {df['target_shot_in_10s'].mean() * 100:.2f}%")
    print("\nAction families:\n" + df["action_family"].value_counts().to_string())
    print("\nPosition groups:\n" + df["position_group"].value_counts().to_string())

    print("\nTop phase shot rates:")
    phase_stats = df.groupby("phase_label", dropna=False)["target_shot_in_10s"].agg(["size", "mean"]).sort_values("mean", ascending=False)
    print(phase_stats.to_string())

    print("\nTop player-role shot rates (min 100 actions):")
    role_stats = df.groupby("position_group")["target_shot_in_10s"].agg(["size", "mean"])
    role_stats = role_stats[role_stats["size"] >= 100].sort_values("mean", ascending=False)
    print(role_stats.to_string())

    print("\nFeature correlations with target:")
    corr = {}
    for feature in NUMERIC_FEATURES:
        if feature in df.columns:
            s = pd.to_numeric(df[feature], errors="coerce")
            valid = s.dropna()
            if valid.nunique() > 1 and valid.shape[0] > 10:
                corr[feature] = valid.corr(df.loc[valid.index, "target_shot_in_10s"])
    corr_series = pd.Series(corr).sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr_series.to_string())

    for feature in [
        "nearest_goal_distance",
        "freeze_support_balance_5m",
        "freeze_support_ratio_5m",
        "freeze_opponent_nearest_distance",
        "freeze_teammate_spread",
        "opponent_count",
    ]:
        if feature in df.columns:
            _bucket_summary(df, feature)

    print("\nDone.")


if __name__ == "__main__":
    main()



