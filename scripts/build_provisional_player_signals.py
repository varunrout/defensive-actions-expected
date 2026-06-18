"""
CLI for building provisional OOF player defensive signals.

Aggregates classification, conditional severity, and combined hurdle predictions
to player-team level with uncertainty estimation via bootstrap confidence intervals.

python scripts/build_provisional_player_signals.py \
  --classification-oof outputs/oof/classification_oof.parquet \
  --two-part-oof outputs/oof/two_part_future_xg_oof.parquet \
  --config configs/models.yaml \
  --output data/features/player_defensive_signals_provisional_oof.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def player_groupby_cols(df: pd.DataFrame) -> list[str]:
    """Return the canonical player aggregation keys present in a frame."""

    columns = ["player_id", "team"]
    if "player_name" in df.columns:
        columns.insert(1, "player_name")
    return columns


def build_player_aggregates(
    two_part_oof: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate two-part OOF predictions to player-team level.
    """
    
    groupby_cols = player_groupby_cols(two_part_oof)
    
    # Core aggregations
    agg_dict = {
        "event_id": "count",  # eligible_actions
        "match_id": "nunique",  # represented_matches
        "oof_shot_probability": "sum",  # total_expected_shots (sum of probabilities)
        "observed_future_shot": "sum",  # total_observed_shots
        "combined_future_xg_prediction": ["sum", "mean"],  # total and mean expected xg
        "observed_future_xg": ["sum", "mean"],  # total and mean observed xg
    }
    
    player_signals = two_part_oof.groupby(groupby_cols, dropna=False).agg(agg_dict).reset_index()
    player_signals.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in player_signals.columns.values]
    
    # Rename for clarity
    player_signals = player_signals.rename(columns={
        "event_id_count": "eligible_actions",
        "match_id_nunique": "represented_matches",
        "oof_shot_probability_sum": "expected_shots",
        "observed_future_shot_sum": "observed_shots",
        "combined_future_xg_prediction_sum": "expected_future_xg",
        "combined_future_xg_prediction_mean": "mean_expected_future_xg",
        "observed_future_xg_sum": "observed_future_xg",
        "observed_future_xg_mean": "mean_observed_future_xg",
    })
    
    # Compute suppression metrics
    player_signals["shot_suppression"] = (
        player_signals["expected_shots"] - player_signals["observed_shots"]
    )
    player_signals["mean_shot_suppression"] = (
        player_signals["shot_suppression"] / player_signals["eligible_actions"]
    )
    
    player_signals["combined_xg_suppression"] = (
        player_signals["expected_future_xg"] - player_signals["observed_future_xg"]
    )
    player_signals["mean_combined_xg_suppression"] = (
        player_signals["combined_xg_suppression"] / player_signals["eligible_actions"]
    )
    
    # Only for rows where a shot occurred: conditional severity suppression
    shot_rows = two_part_oof[two_part_oof["observed_future_shot"] > 0].copy()
    if not shot_rows.empty:
        shot_suppression_agg = shot_rows.groupby(groupby_cols, dropna=False).agg({
            "conditional_xg_prediction": "mean",
            "observed_future_xg": "mean",
        }).reset_index()
        shot_suppression_agg = shot_suppression_agg.rename(columns={
            "conditional_xg_prediction": "conditional_severity_suppression",
            "observed_future_xg": "observed_severity_on_shots",
        })
        shot_suppression_agg["conditional_severity_suppression"] = (
            shot_suppression_agg["conditional_severity_suppression"] - shot_suppression_agg["observed_severity_on_shots"]
        )
        
        player_signals = player_signals.merge(
            shot_suppression_agg[groupby_cols + ["conditional_severity_suppression"]],
            on=groupby_cols,
            how="left"
        )
    else:
        player_signals["conditional_severity_suppression"] = np.nan
    
    # Action family splits
    for action_fam in two_part_oof.get("action_family", pd.Series()).unique():
        if pd.isna(action_fam):
            continue
        subset = two_part_oof[two_part_oof["action_family"] == action_fam]
        agg = (
            subset.groupby(groupby_cols, dropna=False)
            .agg(
                **{
                    f"actions_{action_fam}": ("event_id", "size"),
                    f"xg_{action_fam}": ("combined_future_xg_prediction", "sum"),
                }
            )
            .reset_index()
        )
        player_signals = player_signals.merge(agg, on=groupby_cols, how="left")
    
    # Phase splits
    for phase in two_part_oof.get("phase_label", pd.Series()).unique():
        if pd.isna(phase):
            continue
        subset = two_part_oof[two_part_oof["phase_label"] == phase]
        agg = (
            subset.groupby(groupby_cols, dropna=False)
            .agg(**{f"actions_{phase}": ("event_id", "size")})
            .reset_index()
        )
        player_signals = player_signals.merge(agg, on=groupby_cols, how="left")
    
    # Position group context
    for pos in two_part_oof.get("position_group", pd.Series()).unique():
        if pd.isna(pos):
            continue
        subset = two_part_oof[two_part_oof["position_group"] == pos]
        agg = (
            subset.groupby(groupby_cols, dropna=False)
            .agg(**{f"actions_{pos}": ("event_id", "size")})
            .reset_index()
        )
        player_signals = player_signals.merge(agg, on=groupby_cols, how="left")

    numeric_cols = player_signals.select_dtypes(include=[np.number]).columns
    player_signals.loc[:, numeric_cols] = player_signals.loc[:, numeric_cols].fillna(0)
    return player_signals


def bootstrap_confidence_intervals(data: pd.Series, *, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Returns:
        mean, standard_error, ci_90_lower, ci_90_upper, ci_95_lower, ci_95_upper
    """
    rng = np.random.RandomState(seed)
    
    if len(data) == 0:
        return {
            "bootstrap_mean": np.nan,
            "bootstrap_std": np.nan,
            "ci_90_lower": np.nan,
            "ci_90_upper": np.nan,
            "ci_95_lower": np.nan,
            "ci_95_upper": np.nan,
        }
    
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(sample.mean())
    
    bootstrap_samples = np.array(bootstrap_samples)
    
    return {
        "bootstrap_mean": float(bootstrap_samples.mean()),
        "bootstrap_std": float(bootstrap_samples.std()),
        "ci_90_lower": float(np.percentile(bootstrap_samples, 5)),
        "ci_90_upper": float(np.percentile(bootstrap_samples, 95)),
        "ci_95_lower": float(np.percentile(bootstrap_samples, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_samples, 97.5)),
    }


def add_uncertainty_and_reliability(
    player_signals: pd.DataFrame,
    two_part_oof: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add bootstrap confidence intervals and reliability flags to player signals.
    """
    
    groupby_cols = player_groupby_cols(two_part_oof)
    
    uncertainty_rows = []
    
    for idx, row in player_signals.iterrows():
        player_id = row["player_id"]
        team = row["team"]
        
        # Get all actions for this player-team
        mask = (two_part_oof["player_id"] == player_id) & (two_part_oof["team"] == team)
        player_data = two_part_oof[mask]
        
        if player_data.empty:
            uncertainty_rows.append({
                "player_id": player_id,
                "team": team,
                "action_count": 0,
                "match_count": 0,
                "reliability_tier": "insufficient",
                "minimum_sample_flag": True,
            })
            continue
        
        # Compute CIs on key metrics
        xg_ci = bootstrap_confidence_intervals(player_data["observed_future_xg"])
        shot_ci = bootstrap_confidence_intervals(player_data["observed_future_shot"])
        
        action_count = len(player_data)
        match_count = player_data["match_id"].nunique()
        
        # Reliability tier based on observed distribution
        if action_count < 20 or match_count < 3:
            reliability_tier = "insufficient"
        elif action_count < 50 or match_count < 5:
            reliability_tier = "low"
        elif action_count < 100:
            reliability_tier = "medium"
        else:
            reliability_tier = "high"
        
        uncertainty_rows.append({
            "player_id": player_id,
            "team": team,
            "action_count": int(action_count),
            "match_count": int(match_count),
            "xg_bootstrap_mean": xg_ci["bootstrap_mean"],
            "xg_bootstrap_std": xg_ci["bootstrap_std"],
            "xg_ci_90_lower": xg_ci["ci_90_lower"],
            "xg_ci_90_upper": xg_ci["ci_90_upper"],
            "xg_ci_95_lower": xg_ci["ci_95_lower"],
            "xg_ci_95_upper": xg_ci["ci_95_upper"],
            "shot_bootstrap_mean": shot_ci["bootstrap_mean"],
            "shot_bootstrap_std": shot_ci["bootstrap_std"],
            "reliability_tier": reliability_tier,
            "minimum_sample_flag": action_count < 20,
        })
    
    uncertainty_df = pd.DataFrame(uncertainty_rows)
    return player_signals.merge(uncertainty_df, on=groupby_cols, how="left")


def compute_sensitivity_comparisons(
    two_part_oof: pd.DataFrame,
    regression_oof: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute player signal rank correlations across model variants.
    """
    
    groupby_cols = player_groupby_cols(two_part_oof)
    
    # Aggregate metrics by player
    player_two_part = two_part_oof.groupby(groupby_cols, dropna=False).agg({
        "combined_future_xg_prediction": "sum",
        "observed_future_xg": "sum",
    }).reset_index()
    player_two_part = player_two_part.rename(columns={
        "combined_future_xg_prediction": "expected_xg_two_part",
        "observed_future_xg": "observed_xg",
    })
    player_two_part["suppression_two_part"] = (
        player_two_part["expected_xg_two_part"] - player_two_part["observed_xg"]
    )
    
    result = player_two_part.copy()
    
    # Add one-stage regression comparison if available
    if regression_oof is not None:
        player_regression = regression_oof.groupby(groupby_cols, dropna=False).agg({
            "y_pred": "sum",
            "y_true": "sum",
        }).reset_index()
        player_regression = player_regression.rename(columns={
            "y_pred": "expected_xg_regression",
            "y_true": "observed_xg",
        })
        player_regression["suppression_regression"] = (
            player_regression["expected_xg_regression"] - player_regression["observed_xg"]
        )
        
        result = result.merge(
            player_regression[[*groupby_cols, "expected_xg_regression", "suppression_regression"]],
            on=groupby_cols,
            how="left"
        )
    
    # Compute rank correlations
    if all(col in result.columns for col in ["suppression_two_part", "suppression_regression"]):
        spearman_r, spearman_p = stats.spearmanr(
            result["suppression_two_part"].fillna(0),
            result["suppression_regression"].fillna(0)
        )
        result["spearman_suppression_correlation"] = spearman_r
        result["spearman_suppression_pvalue"] = spearman_p
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Build provisional OOF player defensive signals")
    parser.add_argument("--classification-oof", required=True, help="Path to classification_oof.parquet")
    parser.add_argument("--two-part-oof", required=True, help="Path to two_part_future_xg_oof.parquet")
    parser.add_argument("--regression-oof", help="Path to regression_oof.parquet for sensitivity")
    parser.add_argument("--config", required=True, help="Path to models.yaml")
    parser.add_argument("--output", required=True, help="Output path for player signals parquet")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    two_part_oof = pd.read_parquet(args.two_part_oof)
    regression_oof = pd.read_parquet(args.regression_oof) if args.regression_oof else None
    
    # Build player-level aggregates
    print("Building player aggregates...")
    player_signals = build_player_aggregates(two_part_oof)
    
    # Add uncertainty and reliability
    print("Computing confidence intervals...")
    player_signals = add_uncertainty_and_reliability(player_signals, two_part_oof)
    
    # Add sensitivity comparisons
    print("Computing sensitivity comparisons...")
    sensitivity = compute_sensitivity_comparisons(two_part_oof, regression_oof)
    sensitivity_merge_cols = [column for column in player_groupby_cols(sensitivity) if column in player_signals.columns]
    player_signals = player_signals.merge(
        sensitivity,
        on=sensitivity_merge_cols,
        how="left"
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_signals.to_parquet(output_path, index=False)
    
    print(f"✓ Player signals saved to: {output_path}")
    print(f"  Rows: {len(player_signals)}")
    print(f"  Columns: {len(player_signals.columns)}")


if __name__ == "__main__":
    main()


