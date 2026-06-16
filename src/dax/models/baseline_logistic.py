"""Baseline logistic modeling utilities for DAx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "target_shot_in_10s"
GROUP_COL = "match_id"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    categorical: list[str]
    numeric: list[str]
    c: float = 1.0


def default_variant_specs() -> list[VariantSpec]:
    """Return default baseline variants V0-V8."""
    return [
        VariantSpec(
            name="v0_phase_only",
            categorical=["phase_label"],
            numeric=[],
        ),
        VariantSpec(
            name="v1_spatial",
            categorical=["phase_label", "action_zone", "action_family", "position_group"],
            numeric=["action_x", "action_y",  "distance_to_center_line"],
        ),
        VariantSpec(
            name="v2_full_baseline",
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
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_support_ratio_10m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_nearest_distance",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_count",
                "opponent_count",
                "teammate_opponent_ratio",
                
                "possession_elapsed_seconds",
                
                
                "phase_transitions_observed_so_far",
            ],
        ),
        VariantSpec(
            name="v3_context_enhanced",
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
                "freeze_teammate_count",
                "freeze_opponent_count",
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_support_ratio_10m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_nearest_distance",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_count",
                "opponent_count",
                "teammate_opponent_ratio",
                
                "possession_elapsed_seconds",
                
                
                "phase_transitions_observed_so_far",
            ],
        ),
        VariantSpec(
            name="v4_freeze_geometry",
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
                "freeze_teammate_count",
                "freeze_opponent_count",
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_support_ratio_10m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_nearest_distance",
                "freeze_teammate_centroid_x",
                "freeze_teammate_centroid_y",
                "freeze_opponent_centroid_x",
                "freeze_opponent_centroid_y",
                "freeze_teammate_centroid_dx",
                "freeze_teammate_centroid_dy",
                "freeze_opponent_centroid_dx",
                "freeze_opponent_centroid_dy",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_count",
                "opponent_count",
                "teammate_opponent_ratio",
                
                "possession_elapsed_seconds",
                
                
                "phase_transitions_observed_so_far",
            ],
        ),
        VariantSpec(
            name="v5_interpretable_clustered",
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
                "freeze_teammate_count",
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_centroid_y",
                "freeze_teammate_centroid_dx",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_opponent_ratio",
                
            ],
        ),
        VariantSpec(
            name="v6_balanced_clustered",
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
                "freeze_teammate_count",
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_centroid_y",
                "freeze_teammate_centroid_dx",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_opponent_ratio",
                
                
                "phase_transitions_observed_so_far",
            ],
        ),
        VariantSpec(
            name="v7_interpretable_ridge",
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
                "freeze_teammate_count",
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_centroid_y",
                "freeze_teammate_centroid_dx",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_opponent_ratio",
                
            ],
            c=0.6,
        ),
        VariantSpec(
            name="v8_balanced_ridge",
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
                "freeze_teammate_count",
                "freeze_support_balance_5m",
                "freeze_support_balance_10m",
                "freeze_support_ratio_5m",
                "freeze_teammate_nearest_distance",
                "freeze_opponent_centroid_y",
                "freeze_teammate_centroid_dx",
                "freeze_teammate_spread",
                "freeze_opponent_spread",
                "teammate_opponent_ratio",
                
                
                "phase_transitions_observed_so_far",
            ],
            c=0.5,
        ),
    ]


def resolve_columns(df: pd.DataFrame, spec: VariantSpec) -> VariantSpec:
    """Keep only columns that exist in the dataframe."""
    categorical = [c for c in spec.categorical if c in df.columns]
    numeric = [c for c in spec.numeric if c in df.columns]
    return VariantSpec(name=spec.name, categorical=categorical, numeric=numeric, c=spec.c)


def build_pipeline(spec: VariantSpec) -> Pipeline:
    """Build preprocessing + logistic pipeline for a variant."""
    validate_no_future_features(spec)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, spec.numeric),
            ("cat", categorical_transformer, spec.categorical),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        solver="liblinear",
        C=spec.c,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def prepare_xyg(df: pd.DataFrame, spec: VariantSpec) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare features, target and group arrays."""
    needed = [TARGET_COL, GROUP_COL, *spec.categorical, *spec.numeric]
    needed = [c for c in needed if c in df.columns]
    data = df[needed].copy()
    data = data.dropna(subset=[TARGET_COL, GROUP_COL])

    x_cols = [*spec.categorical, *spec.numeric]
    x = data[x_cols].copy()
    y = (pd.to_numeric(data[TARGET_COL], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    groups = data[GROUP_COL].to_numpy()
    return x, y, groups


def grouped_cv_scores(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: Pipeline,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run GroupKFold CV and return fold metrics + OOF predictions."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    oof = np.full(shape=y.shape[0], fill_value=np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups), start=1):
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline.fit(x_train, y_train)
        y_score = pipeline.predict_proba(x_test)[:, 1]
        oof[test_idx] = y_score

        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "roc_auc": float(roc_auc_score(y_test, y_score)),
                "avg_precision": float(average_precision_score(y_test, y_score)),
            }
        )

    mask = ~np.isnan(oof)
    overall_auc = float(roc_auc_score(y[mask], oof[mask]))
    overall_ap = float(average_precision_score(y[mask], oof[mask]))

    return {
        "fold_metrics": fold_rows,
        "oof_predictions": oof,
        "roc_auc": overall_auc,
        "avg_precision": overall_ap,
    }


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Extract coefficient table from fitted logistic pipeline."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = pre.get_feature_names_out()
    coefs = model.coef_[0]
    out = pd.DataFrame({"feature": feature_names, "coef": coefs})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


FUTURE_ONLY_FEATURES = {
    "possession_duration_total",
    "possession_event_count_total",
    "possession_progress_ratio",
    "target_xt_10s",
}


def validate_no_future_features(spec: VariantSpec) -> None:
    leaked = FUTURE_ONLY_FEATURES.intersection([*spec.categorical, *spec.numeric])
    if leaked:
        raise ValueError(f"Future-only features in model spec {spec.name}: {sorted(leaked)}")
