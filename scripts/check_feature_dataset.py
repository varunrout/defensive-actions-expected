from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/features/player_defensive_actions.parquet")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    if "match_id" in df.columns:
        print(f"Matches: {df['match_id'].nunique():,}")

    if "player_id" in df.columns:
        print(f"Players: {df['player_id'].nunique():,}")

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    if "event_id" in df.columns:
        print("\nDuplicate event IDs:")
        print(df["event_id"].duplicated().sum())

    print("\n" + "=" * 70)
    print("TARGET CHECKS")
    print("=" * 70)

    required_targets = [
        "target_future_shot_10s",
        "target_future_xg_10s",
    ]

    for target in required_targets:
        if target not in df.columns:
            raise ValueError(f"Missing required target: {target}")

    if "target_xt_10s" in df.columns:
        raise ValueError("Deprecated target_xt_10s is still present")

    print("\nFuture shot target:")
    print(df["target_future_shot_10s"].value_counts(dropna=False))
    print(
        "Positive rate:",
        round(df["target_future_shot_10s"].mean(), 6),
    )

    print("\nFuture xG target:")
    print(df["target_future_xg_10s"].describe())
    print(
        "Non-zero rows:",
        int(df["target_future_xg_10s"].gt(0).sum()),
    )
    print(
        "Zero rate:",
        round(df["target_future_xg_10s"].eq(0).mean(), 6),
    )

    assert df["target_future_shot_10s"].notna().all()
    assert df["target_future_xg_10s"].notna().all()
    assert df["target_future_shot_10s"].isin([0, 1]).all()
    assert (df["target_future_xg_10s"] >= 0).all()

    print("\n" + "=" * 70)
    print("TEAM SEMANTICS")
    print("=" * 70)

    if {"attacking_team", "defending_team"}.issubset(df.columns):
        known_teams = (
            df["attacking_team"].notna()
            & df["defending_team"].notna()
        )

        same_team = (
            known_teams
            & (df["attacking_team"] == df["defending_team"])
        )

        print("Rows with known team context:", int(known_teams.sum()))
        print("Rows with identical teams:", int(same_team.sum()))

        assert not same_team.any()

    semantic_columns = [
        "action_changed_possession",
        "action_ended_possession",
        "action_won_possession",
        "action_was_under_opponent_possession",
    ]

    for column in semantic_columns:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION")
    print("=" * 70)

    if "event_type" in df.columns:
        print(df["event_type"].value_counts(dropna=False))

    if "action_family" in df.columns:
        print("\nAction families:")
        print(df["action_family"].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("PHASE DISTRIBUTION")
    print("=" * 70)

    if "phase_label" in df.columns:
        print(df["phase_label"].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("360 AND VISIBILITY COVERAGE")
    print("=" * 70)

    coverage_columns = [
        "has_360",
        "freeze_frame_roles_known",
        "visibility_quality_band",
        "visibility_limited",
        "local_5m_region_fully_visible",
        "local_10m_region_fully_visible",
    ]

    for column in coverage_columns:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].value_counts(dropna=False))

    print("\n" + "=" * 70)
    print("MISSINGNESS")
    print("=" * 70)

    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
    )

    print(missing.head(30).to_string())

    print("\n" + "=" * 70)
    print("NUMERIC FEATURE HEALTH")
    print("=" * 70)

    important_numeric = [
        "distance_to_attacking_goal",
        "angle_to_attacking_goal",
        "visible_attacker_count",
        "visible_defender_count",
        "nearest_attacker_distance",
        "nearest_defender_distance",
        "attackers_within_5m",
        "defenders_within_5m",
        "attackers_within_10m",
        "defenders_within_10m",
        "local_numerical_balance_5m",
        "local_numerical_balance_10m",
        "visible_area_fraction_of_pitch",
    ]

    available_numeric = [
        column
        for column in important_numeric
        if column in df.columns
    ]

    if available_numeric:
        print(df[available_numeric].describe().T.to_string())

    print("\n" + "=" * 70)
    print("AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()