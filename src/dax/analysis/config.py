"""Configuration loading and validation for pre-modelling analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_ANALYSIS_CONFIG: dict[str, Any] = {
    "minimum_player_actions": 30,
    "minimum_category_sample_size": 20,
    "pitch_grid_dimensions": [6, 4],
    "feature_bins": 10,
    "cluster_count_candidates": [2, 3, 4, 5],
    "random_seed": 42,
    "scaling_method": "standard",
    "missing_value_threshold": 0.5,
    "correlation_threshold": 0.95,
    "output_formats": ["csv", "parquet", "png"],
    "chart_dpi": 150,
    "enable_umap": False,
    "enable_hdbscan": False,
    "minimum_action_threshold_sensitivity": [20, 30, 40, 50],
    "feature_group_sensitivity_sets": {
        "all_configured_groups": ["action_mix", "phase_mix", "spatial_style", "possession_style", "360_context", "difficulty_exposure"],
        "without_360_context": ["action_mix", "phase_mix", "spatial_style", "possession_style", "difficulty_exposure"],
        "without_possession_style": ["action_mix", "phase_mix", "spatial_style", "360_context", "difficulty_exposure"],
        "action_phase_spatial": ["action_mix", "phase_mix", "spatial_style"],
        "spatial_difficulty": ["spatial_style", "difficulty_exposure"],
    },
    "clustering_feature_groups": {
        "action_mix": ["action_family_*_share"],
        "phase_mix": ["phase_*_share"],
        "spatial_style": [
            "mean_action_x",
            "mean_action_y",
            "median_action_x",
            "median_action_y",
            "action_width_std",
            "zone_*_share",
        ],
        "possession_style": [
            "possession_win_rate",
            "opponent_possession_end_rate",
            "retained_control_rate",
            "actions_under_opponent_possession_rate",
        ],
        "360_context": [
            "mean_visible_attacker_count",
            "mean_visible_defender_count",
            "mean_nearest_attacker_distance",
            "mean_nearest_defender_distance",
            "mean_defenders_between_ball_and_attacking_goal",
            "role_known_share",
            "reliable_visibility_share",
        ],
        "difficulty_exposure": [
            "mean_local_numerical_balance_5m",
            "mean_local_numerical_balance_10m",
            "numerical_disadvantage_5m_share",
            "numerical_disadvantage_10m_share",
            "visibility_limited_share",
            "box_defence_share",
        ],
    },
    "primary_clustering_feature_groups": [
        "action_mix",
        "phase_mix",
        "spatial_style",
        "possession_style",
        "360_context",
        "difficulty_exposure",
    ],
}


def load_analysis_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load analysis configuration and validate required keys.

    Parameters
    ----------
    path:
        YAML file path. If missing or ``None``, defaults are used.

    Returns
    -------
    dict
        Validated configuration with defaults filled in.
    """
    config = DEFAULT_ANALYSIS_CONFIG.copy()
    if path and Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        config.update(loaded)
        if "clustering_feature_groups" in loaded:
            merged_groups = DEFAULT_ANALYSIS_CONFIG["clustering_feature_groups"].copy()
            merged_groups.update(loaded["clustering_feature_groups"] or {})
            config["clustering_feature_groups"] = merged_groups
        if "feature_group_sensitivity_sets" in loaded:
            merged_sets = DEFAULT_ANALYSIS_CONFIG["feature_group_sensitivity_sets"].copy()
            merged_sets.update(loaded["feature_group_sensitivity_sets"] or {})
            config["feature_group_sensitivity_sets"] = merged_sets
    validate_analysis_config(config)
    return config


def validate_analysis_config(config: dict[str, Any]) -> None:
    """Raise ``ValueError`` when analysis configuration is invalid."""
    positive_int_keys = ["minimum_player_actions", "minimum_category_sample_size", "feature_bins", "chart_dpi"]
    for key in positive_int_keys:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"analysis config `{key}` must be a positive integer")

    grid = config.get("pitch_grid_dimensions")
    if not isinstance(grid, list) or len(grid) != 2 or any(not isinstance(v, int) or v <= 0 for v in grid):
        raise ValueError("analysis config `pitch_grid_dimensions` must be a two-item positive integer list")

    candidates = config.get("cluster_count_candidates")
    if not isinstance(candidates, list) or not all(isinstance(v, int) and v >= 2 for v in candidates):
        raise ValueError("analysis config `cluster_count_candidates` must be a list of integers >= 2")

    groups = config.get("clustering_feature_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("analysis config `clustering_feature_groups` must be a non-empty mapping")

    primary = config.get("primary_clustering_feature_groups")
    if not isinstance(primary, list) or not primary:
        raise ValueError("analysis config `primary_clustering_feature_groups` must be a non-empty list")
    missing_groups = sorted(set(primary) - set(groups))
    if missing_groups:
        raise ValueError(f"primary clustering groups are not defined: {missing_groups}")

    thresholds = config.get("minimum_action_threshold_sensitivity")
    if not isinstance(thresholds, list) or not all(isinstance(v, int) and v > 0 for v in thresholds):
        raise ValueError("analysis config `minimum_action_threshold_sensitivity` must be a list of positive integers")

    sensitivity_sets = config.get("feature_group_sensitivity_sets")
    if not isinstance(sensitivity_sets, dict):
        raise ValueError("analysis config `feature_group_sensitivity_sets` must be a mapping")
    for name, group_list in sensitivity_sets.items():
        if not isinstance(group_list, list) or sorted(set(group_list) - set(groups)):
            raise ValueError(f"feature-group sensitivity set `{name}` references undefined groups")
