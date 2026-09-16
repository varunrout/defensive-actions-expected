"""Canonical DAx model specification accessors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "identifiers": ("match_id", "player_id", "team_id"),
    "targets": ("target_future_shot_10s", "target_future_xg_10s"),
    "spatial_features": ("action_x", "action_y", "distance_to_center_line"),
    "possession_features": ("possession_elapsed_seconds", "event_order_in_possession"),
    "phase_features": ("phase_label", "phase_label_prev_event", "phase_changed_since_prev_event"),
    "360_role_features": ("visible_attacker_count", "visible_defender_count", "attacker_defender_ratio"),
    "visibility_features": ("has_360", "freeze_frame_count"),
}


@dataclass(frozen=True)
class ModelVariantSummary:
    task: Literal["logistic", "regression"]
    name: str
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]


def list_model_variants(task: Literal["logistic", "regression"]) -> list[str]:
    if task == "logistic":
        from dax.models.baseline_logistic import default_variant_specs

        return [spec.name for spec in default_variant_specs()]
    if task == "regression":
        from dax.models.baseline_regression import default_regression_specs

        return [spec.name for spec in default_regression_specs()]
    raise ValueError(f"Unsupported task: {task}")


def canonical_feature_groups() -> dict[str, tuple[str, ...]]:
    return FEATURE_GROUPS.copy()
