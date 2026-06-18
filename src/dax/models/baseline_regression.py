"""Baseline regression modeling utilities for DAx (observed future-xG target).

This module supports training regression models using observed future xG
as the continuous target instead of the binary target_future_shot_10s.

This enables:
- Dense target signal (all actions get future-xG outcomes, not sparse binary)
- Richer threat assessment (magnitude, not just yes/no)
- Better model convergence (continuous gradients)
- Complementary perspective (spatial threat vs. temporal outcome)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dax.models.evaluation import safe_spearman

# Target configuration
TARGET_COL = "target_future_xg_10s"
GROUP_COL = "match_id"


@dataclass(frozen=True)
class RegressionVariantSpec:
    """Specification for a regression model variant."""

    name: str
    model_type: str  # 'ridge', 'lasso', or 'linear'
    categorical: list[str]
    numeric: list[str]
    alpha: float = 1.0  # Regularization strength for ridge/lasso


def _unique_features(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def _dedupe_spec_features(spec: RegressionVariantSpec) -> RegressionVariantSpec:
    return RegressionVariantSpec(
        name=spec.name,
        model_type=spec.model_type,
        categorical=_unique_features(spec.categorical),
        numeric=_unique_features(spec.numeric),
        alpha=spec.alpha,
    )


def default_regression_specs() -> list[RegressionVariantSpec]:
    """Return future-xG regression baseline variants (V0-V8)."""
    specs = [
        RegressionVariantSpec(
            name="v0_phase_only",
            model_type="ridge",
            categorical=["phase_label"],
            numeric=[],
            alpha=1.0,
        ),
        RegressionVariantSpec(
            name="v1_spatial",
            model_type="ridge",
            categorical=["phase_label", "action_zone", "action_family", "position_group"],
            numeric=[
                "action_x",
                "action_y",
                
                "distance_to_center_line",
            ],
            alpha=1.0,
        ),
        RegressionVariantSpec(
            name="v2_full_baseline",
            model_type="ridge",
            categorical=[
                "phase_label",
                "action_zone",
                "action_family",
                "position_group",
                "event_type",
                "play_pattern",
            ],
            numeric=[
                "action_x",
                "action_y",
                
                "distance_to_center_line",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "attackers_within_10m",
                "nearest_attacker_distance",
                "nearest_defender_distance",
                "attacker_spread",
                "defender_spread",
                "visible_attacker_count",
                "visible_defender_count",
                "attacker_defender_ratio",
                
                "possession_elapsed_seconds",
                
                
                "phase_transitions_observed_so_far",
            ],
            alpha=1.0,
        ),
        RegressionVariantSpec(
            name="v3_context_enhanced",
            model_type="ridge",
            categorical=[
                "phase_label",
                "phase_label_prev_event",
                "action_zone",
                "action_family",
                "position_group",
                "event_type",
                "play_pattern",
                
            ],
            numeric=[
                "period",
                "counterpress",
                "has_360",
                "phase_changed_since_prev_event",
                "event_order_in_possession",
                "action_x",
                "action_y",
                "ball_x",
                "ball_y",
                
                
                
                "distance_to_center_line",
                "is_central_lane",
                "is_wide_lane",
                "is_deep_zone",
                "is_high_zone",
                "freeze_frame_count",
                "visible_attacker_count",
                "visible_defender_count",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "attackers_within_10m",
                "nearest_attacker_distance",
                "nearest_defender_distance",
                "attacker_spread",
                "defender_spread",
                "visible_attacker_count",
                "visible_defender_count",
                "attacker_defender_ratio",
                
                "possession_elapsed_seconds",
                
                
                "phase_transitions_observed_so_far",
            ],
            alpha=0.8,
        ),
        RegressionVariantSpec(
            name="v4_freeze_geometry",
            model_type="ridge",
            categorical=[
                "phase_label",
                "phase_label_prev_event",
                "action_zone",
                "action_family",
                "position_group",
                "position",
                "event_type",
                "play_pattern",
                
            ],
            numeric=[
                "period",
                "counterpress",
                "has_360",
                "phase_changed_since_prev_event",
                "event_order_in_possession",
                "action_x",
                "action_y",
                "ball_x",
                "ball_y",
                
                
                
                "distance_to_center_line",
                "is_central_lane",
                "is_wide_lane",
                "is_deep_zone",
                "is_high_zone",
                "freeze_frame_count",
                "visible_attacker_count",
                "visible_defender_count",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "attackers_within_10m",
                "nearest_attacker_distance",
                "nearest_defender_distance",
                "attacker_centroid_x",
                "attacker_centroid_y",
                "defender_centroid_x",
                "defender_centroid_y",
                "attacker_centroid_x",
                "attacker_centroid_y",
                "defender_centroid_x",
                "defender_centroid_y",
                "attacker_spread",
                "defender_spread",
                "visible_attacker_count",
                "visible_defender_count",
                "attacker_defender_ratio",
                
                "possession_elapsed_seconds",
                
                
                "phase_transitions_observed_so_far",
            ],
            alpha=0.6,
        ),
        RegressionVariantSpec(
            name="v5_interpretable_clustered",
            model_type="linear",
            categorical=[
                "phase_label",
                "phase_label_prev_event",
                "action_zone",
                "action_family",
                "position",
                "event_type",
                
            ],
            numeric=[
                "period",
                "counterpress",
                "has_360",
                "phase_changed_since_prev_event",
                
                
                "distance_to_center_line",
                "is_wide_lane",
                "is_deep_zone",
                "is_high_zone",
                "visible_attacker_count",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "nearest_attacker_distance",
                "defender_centroid_y",
                "attacker_centroid_x",
                "attacker_spread",
                "defender_spread",
                "attacker_defender_ratio",
                
            ],
            alpha=0.0,
        ),
        RegressionVariantSpec(
            name="v6_balanced_clustered",
            model_type="linear",
            categorical=[
                "phase_label",
                "phase_label_prev_event",
                "action_zone",
                "action_family",
                "position_group",
                "position",
                "event_type",
                "play_pattern",
                
            ],
            numeric=[
                "period",
                "counterpress",
                "has_360",
                "phase_changed_since_prev_event",
                
                
                "distance_to_center_line",
                "is_wide_lane",
                "is_deep_zone",
                "is_high_zone",
                "visible_attacker_count",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "nearest_attacker_distance",
                "defender_centroid_y",
                "attacker_centroid_x",
                "attacker_spread",
                "defender_spread",
                "attacker_defender_ratio",
                
                
                "phase_transitions_observed_so_far",
            ],
            alpha=0.0,
        ),
        RegressionVariantSpec(
            name="v7_interpretable_ridge",
            model_type="ridge",
            categorical=[
                "phase_label",
                "phase_label_prev_event",
                "action_zone",
                "action_family",
                "position",
                "event_type",
                
            ],
            numeric=[
                "period",
                "counterpress",
                "has_360",
                "phase_changed_since_prev_event",
                
                
                "distance_to_center_line",
                "is_wide_lane",
                "is_deep_zone",
                "is_high_zone",
                "visible_attacker_count",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "nearest_attacker_distance",
                "defender_centroid_y",
                "attacker_centroid_x",
                "attacker_spread",
                "defender_spread",
                "attacker_defender_ratio",
                
            ],
            alpha=0.6,
        ),
        RegressionVariantSpec(
            name="v8_balanced_ridge",
            model_type="ridge",
            categorical=[
                "phase_label",
                "phase_label_prev_event",
                "action_zone",
                "action_family",
                "position_group",
                "position",
                "event_type",
                "play_pattern",
                
            ],
            numeric=[
                "period",
                "counterpress",
                "has_360",
                "phase_changed_since_prev_event",
                
                
                "distance_to_center_line",
                "is_wide_lane",
                "is_deep_zone",
                "is_high_zone",
                "visible_attacker_count",
                "local_numerical_balance_5m",
                "local_numerical_balance_10m",
                "attackers_within_5m",
                "nearest_attacker_distance",
                "defender_centroid_y",
                "attacker_centroid_x",
                "attacker_spread",
                "defender_spread",
                "attacker_defender_ratio",
                
                
                "phase_transitions_observed_so_far",
            ],
            alpha=0.5,
        ),
    ]
    return [_dedupe_spec_features(spec) for spec in specs]


def resolve_columns(
    df: pd.DataFrame, spec: RegressionVariantSpec, strict: bool = True
) -> RegressionVariantSpec:
    """Validate model feature columns and return the resolved specification."""
    missing_categorical = [c for c in spec.categorical if c not in df.columns]
    missing_numeric = [c for c in spec.numeric if c not in df.columns]
    if strict and (missing_categorical or missing_numeric):
        raise ValueError(
            f"Missing required features for model spec {spec.name}: "
            f"categorical={missing_categorical}, numeric={missing_numeric}"
        )
    categorical = [c for c in spec.categorical if c in df.columns]
    numeric = [c for c in spec.numeric if c in df.columns]
    return RegressionVariantSpec(
        name=spec.name,
        model_type=spec.model_type,
        categorical=categorical,
        numeric=numeric,
        alpha=spec.alpha,
    )


def build_regression_pipeline(spec: RegressionVariantSpec) -> Pipeline:
    """Build preprocessing + regression pipeline for a variant."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, spec.numeric),
            ("cat", categorical_transformer, spec.categorical),
        ],
        remainder="drop",
    )

    # Select regression model based on type
    if spec.model_type == "ridge":
        model = Ridge(alpha=spec.alpha, random_state=42)
    elif spec.model_type == "lasso":
        model = Lasso(alpha=spec.alpha, random_state=42, max_iter=10000)
    else:  # linear
        model = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_xyg_regression(
    df: pd.DataFrame, spec: RegressionVariantSpec
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare features, target and group arrays for regression."""
    needed = [TARGET_COL, GROUP_COL, *spec.categorical, *spec.numeric]
    needed = [c for c in needed if c in df.columns]
    data = df[needed].copy()
    data = data.dropna(subset=[TARGET_COL, GROUP_COL])

    x_cols = [*spec.categorical, *spec.numeric]
    x = data[x_cols].copy()
    y = pd.to_numeric(data[TARGET_COL], errors="coerce").fillna(0).to_numpy()
    groups = data[GROUP_COL].to_numpy()
    return x, y, groups


def grouped_cv_scores_regression(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: Pipeline,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run GroupKFold CV for regression and return fold metrics + OOF predictions."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    oof = np.full(shape=y.shape[0], fill_value=np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups), start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        oof[test_idx] = y_pred

        # Regression metrics
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "spearman": safe_spearman(y_test, y_pred),
                "train_target_mean": float(y_train.mean()),
                "test_target_mean": float(y_test.mean()),
                "test_target_zero_rate": float((y_test == 0).mean()),
            }
        )

    mask = ~np.isnan(oof)
    overall_r2 = float(r2_score(y[mask], oof[mask]))
    overall_rmse = float(np.sqrt(mean_squared_error(y[mask], oof[mask])))
    overall_mae = float(mean_absolute_error(y[mask], oof[mask]))
    overall_spearman = safe_spearman(y[mask], oof[mask])

    return {
        "fold_metrics": fold_rows,
        "oof_predictions": oof,
        "r2": overall_r2,
        "rmse": overall_rmse,
        "mae": overall_mae,
        "spearman": overall_spearman,
        "target_mean": float(y[mask].mean()),
        "target_zero_rate": float((y[mask] == 0).mean()),
    }


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Extract coefficient table from fitted regression pipeline."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    coefs = model.coef_
    out = pd.DataFrame({"feature": feature_names, "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)

