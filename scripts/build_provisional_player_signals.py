"""
CLI for building provisional OOF player defensive signals.

The script validates classification OOF alignment, aggregates two-part OOF signals,
computes match-level bootstrap uncertainty, derives data-driven reliability tiers,
and writes player-level sensitivity diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def player_groupby_cols(df: pd.DataFrame) -> list[str]:
    cols = ["player_id", "team"]
    if "player_name" in df.columns:
        cols.insert(1, "player_name")
    return cols


def sanitise_token(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_")


def validate_classification_alignment(two_part_oof: pd.DataFrame, classification_oof: pd.DataFrame) -> tuple[str, int]:
    """Validate that classification probabilities in two-part OOF match source OOF."""

    if "classification_model_variant" not in two_part_oof.columns:
        raise ValueError("Two-part OOF must contain classification_model_variant.")
    variants = sorted(two_part_oof["classification_model_variant"].dropna().unique().tolist())
    if len(variants) != 1:
        raise ValueError(f"Two-part OOF must contain exactly one classification variant. Got: {variants}")
    variant = str(variants[0])

    source = classification_oof.copy()
    if "model_variant" in source.columns:
        source = source.loc[source["model_variant"].eq(variant)].copy()
    if source.empty:
        raise ValueError(f"Classification OOF does not contain variant {variant!r}.")

    merged = two_part_oof[["event_id", "oof_shot_probability"]].merge(
        source[["event_id", "y_score"]],
        on="event_id",
        how="left",
    )
    if merged["y_score"].isna().any():
        missing = int(merged["y_score"].isna().sum())
        raise ValueError(f"Classification OOF is missing {missing} event IDs from two-part OOF.")

    max_diff = float(np.max(np.abs(merged["oof_shot_probability"].to_numpy() - merged["y_score"].to_numpy())))
    if max_diff > 1e-9:
        raise ValueError(f"Classification score mismatch detected (max absolute diff={max_diff:.6g}).")

    return variant, int(len(merged))


def build_player_aggregates(two_part_oof: pd.DataFrame) -> pd.DataFrame:
    group_cols = player_groupby_cols(two_part_oof)

    aggregated = (
        two_part_oof.groupby(group_cols, dropna=False)
        .agg(
            eligible_actions=("event_id", "count"),
            represented_matches=("match_id", "nunique"),
            expected_shots=("oof_shot_probability", "sum"),
            observed_shots=("observed_future_shot", "sum"),
            expected_future_xg=("combined_future_xg_prediction", "sum"),
            mean_expected_future_xg=("combined_future_xg_prediction", "mean"),
            observed_future_xg=("observed_future_xg", "sum"),
            mean_observed_future_xg=("observed_future_xg", "mean"),
        )
        .reset_index()
    )

    aggregated["shot_suppression"] = aggregated["expected_shots"] - aggregated["observed_shots"]
    aggregated["mean_shot_suppression"] = aggregated["shot_suppression"] / aggregated["eligible_actions"].clip(lower=1)

    aggregated["combined_xg_suppression"] = aggregated["expected_future_xg"] - aggregated["observed_future_xg"]
    aggregated["mean_combined_xg_suppression"] = aggregated["combined_xg_suppression"] / aggregated["eligible_actions"].clip(lower=1)

    shot_rows = two_part_oof.loc[two_part_oof["observed_future_shot"].gt(0)].copy()
    if shot_rows.empty:
        aggregated["observed_shot_count"] = 0
        aggregated["conditional_expected_xg_on_shots"] = np.nan
        aggregated["observed_xg_on_shots"] = np.nan
        aggregated["total_conditional_severity_suppression"] = np.nan
        aggregated["mean_conditional_severity_suppression_per_shot"] = np.nan
        aggregated["conditional_severity_reliability_flag"] = False
    else:
        shot_agg = (
            shot_rows.groupby(group_cols, dropna=False)
            .agg(
                observed_shot_count=("event_id", "count"),
                conditional_expected_xg_on_shots=("conditional_xg_prediction", "sum"),
                observed_xg_on_shots=("observed_future_xg", "sum"),
            )
            .reset_index()
        )
        shot_agg["total_conditional_severity_suppression"] = (
            shot_agg["conditional_expected_xg_on_shots"] - shot_agg["observed_xg_on_shots"]
        )
        shot_agg["mean_conditional_severity_suppression_per_shot"] = (
            shot_agg["total_conditional_severity_suppression"] / shot_agg["observed_shot_count"].clip(lower=1)
        )
        shot_agg["conditional_severity_reliability_flag"] = shot_agg["observed_shot_count"].ge(3)

        aggregated = aggregated.merge(shot_agg, on=group_cols, how="left")
        aggregated["observed_shot_count"] = aggregated["observed_shot_count"].fillna(0).astype(int)
        aggregated["conditional_severity_reliability_flag"] = aggregated["conditional_severity_reliability_flag"].fillna(False)

    aggregated["conditional_severity_suppression"] = aggregated["mean_conditional_severity_suppression_per_shot"]

    # Context splits
    for action_fam in sorted(two_part_oof.get("action_family", pd.Series(dtype=object)).dropna().unique().tolist()):
        subset = two_part_oof.loc[two_part_oof["action_family"].eq(action_fam)]
        token = sanitise_token(action_fam)
        split = (
            subset.groupby(group_cols, dropna=False)
            .agg(**{f"actions_{token}": ("event_id", "size"), f"xg_{token}": ("combined_future_xg_prediction", "sum")})
            .reset_index()
        )
        aggregated = aggregated.merge(split, on=group_cols, how="left")

    for phase in sorted(two_part_oof.get("phase_label", pd.Series(dtype=object)).dropna().unique().tolist()):
        subset = two_part_oof.loc[two_part_oof["phase_label"].eq(phase)]
        token = sanitise_token(phase)
        split = subset.groupby(group_cols, dropna=False).agg(**{f"actions_{token}": ("event_id", "size")}).reset_index()
        aggregated = aggregated.merge(split, on=group_cols, how="left")

    for pos in sorted(two_part_oof.get("position_group", pd.Series(dtype=object)).dropna().unique().tolist()):
        subset = two_part_oof.loc[two_part_oof["position_group"].eq(pos)]
        token = sanitise_token(pos)
        split = subset.groupby(group_cols, dropna=False).agg(**{f"actions_{token}": ("event_id", "size")}).reset_index()
        aggregated = aggregated.merge(split, on=group_cols, how="left")

    numeric_cols = aggregated.select_dtypes(include=[np.number]).columns
    # Preserve no-shot conditional severity outputs as missing values.
    non_conditional = [column for column in numeric_cols if column not in {
        "conditional_expected_xg_on_shots",
        "observed_xg_on_shots",
        "total_conditional_severity_suppression",
        "mean_conditional_severity_suppression_per_shot",
    }]
    aggregated.loc[:, non_conditional] = aggregated.loc[:, non_conditional].fillna(0)
    return aggregated


def derive_reliability_thresholds(player_signals: pd.DataFrame) -> dict[str, Any]:
    quantiles = [0.25, 0.5, 0.75, 0.9]
    stats_by_metric: dict[str, dict[str, float]] = {}
    for metric in ["eligible_actions", "represented_matches", "observed_shot_count"]:
        series = pd.to_numeric(player_signals[metric], errors="coerce")
        stats_by_metric[metric] = {
            f"q{int(q * 100)}": float(series.quantile(q)) for q in quantiles
        }

    thresholds = {
        "actions_low": stats_by_metric["eligible_actions"]["q25"],
        "actions_medium": stats_by_metric["eligible_actions"]["q50"],
        "actions_high": stats_by_metric["eligible_actions"]["q75"],
        "matches_low": stats_by_metric["represented_matches"]["q25"],
        "matches_medium": stats_by_metric["represented_matches"]["q50"],
        "matches_high": stats_by_metric["represented_matches"]["q75"],
        "shots_low": stats_by_metric["observed_shot_count"]["q25"],
        "shots_medium": stats_by_metric["observed_shot_count"]["q50"],
    }
    return {"quantiles": stats_by_metric, "thresholds": thresholds}


def assign_reliability_tier(player_signals: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    t = thresholds["thresholds"]
    action = player_signals["eligible_actions"]
    matches = player_signals["represented_matches"]

    tier = pd.Series("insufficient", index=player_signals.index, dtype=object)
    low_mask = action.ge(t["actions_low"]) & matches.ge(t["matches_low"])
    medium_mask = action.ge(t["actions_medium"]) & matches.ge(t["matches_medium"])
    high_mask = action.ge(t["actions_high"]) & matches.ge(t["matches_high"])

    tier.loc[low_mask] = "low"
    tier.loc[medium_mask] = "medium"
    tier.loc[high_mask] = "high"
    return tier


def bootstrap_player_suppression_match_level(
    two_part_oof: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    group_cols = player_groupby_cols(two_part_oof)

    frame = two_part_oof.copy()
    frame["shot_suppression_event"] = frame["oof_shot_probability"] - frame["observed_future_shot"]
    frame["combined_xg_suppression_event"] = frame["combined_future_xg_prediction"] - frame["observed_future_xg"]

    # Per-match aggregates so we resample matches, not events.
    by_match = (
        frame.groupby(group_cols + ["match_id"], dropna=False)
        .agg(
            match_actions=("event_id", "count"),
            total_shot_suppression=("shot_suppression_event", "sum"),
            total_combined_xg_suppression=("combined_xg_suppression_event", "sum"),
        )
        .reset_index()
    )

    shot_rows = frame.loc[frame["observed_future_shot"].gt(0)].copy()
    cond_by_match = (
        shot_rows.groupby(group_cols + ["match_id"], dropna=False)
        .agg(
            shot_rows_count=("event_id", "count"),
            total_conditional_severity_suppression=("conditional_xg_prediction", "sum"),
            observed_xg_on_shots=("observed_future_xg", "sum"),
        )
        .reset_index()
        if not shot_rows.empty
        else pd.DataFrame(columns=group_cols + ["match_id", "shot_rows_count", "total_conditional_severity_suppression", "observed_xg_on_shots"])
    )
    if not cond_by_match.empty:
        cond_by_match["total_conditional_severity_suppression"] = (
            cond_by_match["total_conditional_severity_suppression"] - cond_by_match["observed_xg_on_shots"]
        )

    rng = np.random.RandomState(seed)
    rows: list[dict[str, Any]] = []
    for keys, group in by_match.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {column: value for column, value in zip(group_cols, keys)}
        match_count = int(group["match_id"].nunique())
        if match_count == 0:
            continue

        cond_group = cond_by_match
        for column in group_cols:
            cond_group = cond_group.loc[cond_group[column].eq(key_map[column])]

        stat_samples = {
            "total_shot_suppression": [],
            "mean_shot_suppression_per_action": [],
            "total_combined_xg_suppression": [],
            "mean_combined_xg_suppression_per_action": [],
            "total_conditional_severity_suppression": [],
            "mean_conditional_severity_suppression_per_shot": [],
        }

        group_values = group[["match_actions", "total_shot_suppression", "total_combined_xg_suppression"]].to_numpy(dtype=float)
        cond_values = cond_group[["shot_rows_count", "total_conditional_severity_suppression"]].to_numpy(dtype=float) if not cond_group.empty else np.empty((0, 2), dtype=float)

        for _ in range(n_bootstrap):
            idx = rng.choice(len(group_values), size=len(group_values), replace=True)
            sampled = group_values[idx]
            total_actions = sampled[:, 0].sum()
            total_shot_supp = sampled[:, 1].sum()
            total_combined_supp = sampled[:, 2].sum()

            stat_samples["total_shot_suppression"].append(total_shot_supp)
            stat_samples["mean_shot_suppression_per_action"].append(total_shot_supp / max(total_actions, 1.0))
            stat_samples["total_combined_xg_suppression"].append(total_combined_supp)
            stat_samples["mean_combined_xg_suppression_per_action"].append(total_combined_supp / max(total_actions, 1.0))

            if len(cond_values) > 0:
                cidx = rng.choice(len(cond_values), size=len(cond_values), replace=True)
                sampled_cond = cond_values[cidx]
                total_shot_rows = sampled_cond[:, 0].sum()
                total_cond_supp = sampled_cond[:, 1].sum()
                stat_samples["total_conditional_severity_suppression"].append(total_cond_supp)
                stat_samples["mean_conditional_severity_suppression_per_shot"].append(total_cond_supp / max(total_shot_rows, 1.0))
            else:
                stat_samples["total_conditional_severity_suppression"].append(np.nan)
                stat_samples["mean_conditional_severity_suppression_per_shot"].append(np.nan)

        row = {**key_map, "bootstrap_match_count": match_count}
        for metric, values in stat_samples.items():
            arr = np.asarray(values, dtype=float)
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                row[f"{metric}_bootstrap_mean"] = np.nan
                row[f"{metric}_bootstrap_se"] = np.nan
                row[f"{metric}_ci90_lower"] = np.nan
                row[f"{metric}_ci90_upper"] = np.nan
                row[f"{metric}_ci95_lower"] = np.nan
                row[f"{metric}_ci95_upper"] = np.nan
            else:
                row[f"{metric}_bootstrap_mean"] = float(valid.mean())
                row[f"{metric}_bootstrap_se"] = float(valid.std(ddof=0))
                row[f"{metric}_ci90_lower"] = float(np.percentile(valid, 5))
                row[f"{metric}_ci90_upper"] = float(np.percentile(valid, 95))
                row[f"{metric}_ci95_lower"] = float(np.percentile(valid, 2.5))
                row[f"{metric}_ci95_upper"] = float(np.percentile(valid, 97.5))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_player_sensitivity(player_signals: pd.DataFrame) -> pd.DataFrame:
    """Compute within-output stability flags and ranking diagnostics."""

    if player_signals.empty:
        return pd.DataFrame()

    out = player_signals[[column for column in ["player_id", "team", "combined_xg_suppression", "mean_combined_xg_suppression", "reliability_tier"] if column in player_signals.columns]].copy()
    out["rank_total"] = out["combined_xg_suppression"].rank(ascending=False, method="min")
    out["rank_mean"] = out["mean_combined_xg_suppression"].rank(ascending=False, method="min")
    out["rank_change_total_vs_mean"] = (out["rank_total"] - out["rank_mean"]).abs()
    out["suppression_sign_change_total_vs_mean"] = np.sign(out["combined_xg_suppression"]) != np.sign(out["mean_combined_xg_suppression"])

    spearman = float(out[["combined_xg_suppression", "mean_combined_xg_suppression"]].corr(method="spearman").iloc[0, 1])
    kendall = float(out[["combined_xg_suppression", "mean_combined_xg_suppression"]].corr(method="kendall").iloc[0, 1])
    out["spearman_rank_correlation"] = spearman
    out["kendall_rank_correlation"] = kendall
    out["confidence_interval_overlap"] = np.nan
    out["unstable_player_flag"] = out["rank_change_total_vs_mean"].gt(100) | out["suppression_sign_change_total_vs_mean"]
    return out


def compute_sensitivity_comparisons(two_part_oof: pd.DataFrame, regression_oof: pd.DataFrame | None = None) -> pd.DataFrame:
    """Backward-compatible sensitivity table for older tests/callers."""

    group_cols = player_groupby_cols(two_part_oof)
    result = (
        two_part_oof.groupby(group_cols, dropna=False)
        .agg(
            expected_xg_two_part=("combined_future_xg_prediction", "sum"),
            observed_xg=("observed_future_xg", "sum"),
        )
        .reset_index()
    )
    result["suppression_two_part"] = result["expected_xg_two_part"] - result["observed_xg"]
    if regression_oof is not None:
        reg = (
            regression_oof.groupby(group_cols, dropna=False)
            .agg(
                expected_xg_regression=("y_pred", "sum"),
                observed_xg_regression=("y_true", "sum"),
            )
            .reset_index()
        )
        reg["suppression_regression"] = reg["expected_xg_regression"] - reg["observed_xg_regression"]
        result = result.merge(reg[[*group_cols, "expected_xg_regression", "suppression_regression"]], on=group_cols, how="left")

    if "suppression_regression" in result.columns:
        spearman_r, _ = stats.spearmanr(result["suppression_two_part"].fillna(0), result["suppression_regression"].fillna(0))
        result["spearman_suppression_correlation"] = spearman_r
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provisional OOF player defensive signals")
    parser.add_argument("--classification-oof", required=True, help="Path to classification_oof.parquet")
    parser.add_argument("--two-part-oof", required=True, help="Path to two_part_future_xg_oof.parquet")
    parser.add_argument("--regression-oof", help="Path to regression_oof.parquet for optional context")
    parser.add_argument("--config", required=True, help="Path to models.yaml")
    parser.add_argument("--output", required=True, help="Output path for player signals parquet")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading data...")
    two_part_oof = pd.read_parquet(args.two_part_oof)
    classification_oof = pd.read_parquet(args.classification_oof)

    print("Validating classification alignment...")
    variant, checked_rows = validate_classification_alignment(two_part_oof, classification_oof)
    print(f"  Classification variant: {variant} ({checked_rows} aligned rows)")

    print("Building player aggregates...")
    player_signals = build_player_aggregates(two_part_oof)

    print("Deriving reliability thresholds from data...")
    thresholds = derive_reliability_thresholds(player_signals)
    player_signals["reliability_tier"] = assign_reliability_tier(player_signals, thresholds)
    player_signals["minimum_sample_flag"] = player_signals["reliability_tier"].eq("insufficient")

    print("Computing match-level bootstrap uncertainty...")
    uncertainty = bootstrap_player_suppression_match_level(
        two_part_oof,
        n_bootstrap=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    merge_cols = player_groupby_cols(player_signals)
    player_signals = player_signals.merge(uncertainty, on=merge_cols, how="left")

    print("Computing sensitivity diagnostics...")
    sensitivity = compute_player_sensitivity(player_signals)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_signals.to_parquet(output_path, index=False)

    # Save required reliability thresholds artifact.
    reports_dir = output_path.parents[1] / "models" / "reports" if len(output_path.parents) >= 2 else output_path.parent
    reports_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = reports_dir / "player_signal_reliability_thresholds.json"
    thresholds_payload = {
        "classification_variant": variant,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        **thresholds,
    }
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    sensitivity_path = reports_dir / "player_signal_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)

    print(f"✓ Player signals saved to: {output_path}")
    print(f"  Rows: {len(player_signals)}")
    print(f"  Columns: {len(player_signals.columns)}")
    print(f"  Reliability thresholds: {thresholds_path}")
    print(f"  Sensitivity report: {sensitivity_path}")


if __name__ == "__main__":
    main()


