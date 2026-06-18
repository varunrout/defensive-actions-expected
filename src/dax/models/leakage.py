"""Prediction-timestamp and leakage checks for model features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

PREDICTION_TIMESTAMP = "defensive_action_timestamp"

IDENTIFIER_FIELDS = {
    "event_id",
    "match_id",
    "player_id",
    "player",
    "player_name",
    "team",
    "team_id",
    "fold",
}
TARGET_FIELDS = {"target_future_shot_10s", "target_future_xg_10s"}
ACTION_TIME_FIELDS = {
    "action_x",
    "action_y",
    "phase_label",
    "event_type",
    "action_family",
    "position_group",
    "play_pattern",
    "action_zone",
    "period",
    "counterpress",
}
PRE_ACTION_PATTERNS = (
    "visible",
    "attacker",
    "defender",
    "freeze_frame",
    "distance",
    "elapsed",
    "order",
    "zone",
    "phase",
    "balance",
    "centroid",
    "spread",
)
POST_ACTION_PATTERNS = (
    "future",
    "outcome",
    "observed",
    "suppression",
    "dax",
    "signal",
    "result",
)
PREDICTION_PATTERNS = ("prediction", "pred_", "y_pred", "y_score", "model_score")
AGGREGATED_OUTCOME_PATTERNS = (
    "player_rate",
    "player_mean",
    "player_xg",
    "player_shot",
    "cluster_outcome",
    "cluster_rate",
)


@dataclass(frozen=True)
class LeakageRecord:
    feature: str
    classification: str
    allowed: bool
    reason: str
    source_timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_feature(name: str) -> str:
    """Classify a feature by its leakage risk."""

    field = name.lower()
    if field in TARGET_FIELDS or field.startswith("target_"):
        return "target"
    if field in IDENTIFIER_FIELDS or field.endswith("_id"):
        return "identifier"
    if any(pattern in field for pattern in PREDICTION_PATTERNS):
        return "post-action"
    if any(pattern in field for pattern in AGGREGATED_OUTCOME_PATTERNS):
        return "post-action"
    if any(pattern in field for pattern in POST_ACTION_PATTERNS):
        return "post-action"
    if field in ACTION_TIME_FIELDS:
        return "action-time"
    if any(pattern in field for pattern in PRE_ACTION_PATTERNS):
        return "pre-action"
    return "uncertain"


def assess_feature(feature: str, *, feature_scope: str, allow_uncertain: bool = False) -> LeakageRecord:
    """Assess one feature for a scope and provide an auditable reason."""

    classification = classify_feature(feature)
    allowed = True
    reason = "available at or before prediction timestamp"
    source_timestamp = "pre_action_or_action_time"

    if classification == "target":
        allowed = False
        reason = "target columns are never allowed as features"
        source_timestamp = "future_outcome"
    elif classification == "identifier":
        allowed = False
        reason = "identifiers are prohibited model features"
        source_timestamp = "identifier"
    elif classification == "post-action" and feature_scope == "pre_action_context":
        allowed = False
        reason = "post-action fields are unavailable for pre-action-context scoring"
        source_timestamp = "post_action_or_future"
    elif classification == "uncertain" and not allow_uncertain:
        allowed = False
        reason = "uncertain timestamp requires explicit approval"
        source_timestamp = "uncertain"
    elif classification == "post-action":
        reason = "allowed only for explicitly diagnostic post-action-observed scope"
        source_timestamp = "post_action_observed"

    return LeakageRecord(feature, classification, allowed, reason, source_timestamp)


def scan_features(
    features: list[str],
    *,
    selected_target: str,
    feature_scope: str = "pre_action_context",
    allow_uncertain: bool = False,
) -> list[dict[str, object]]:
    """Scan features and fail if prohibited fields are present."""

    records = [assess_feature(feature, feature_scope=feature_scope, allow_uncertain=allow_uncertain) for feature in features]
    failures = [record for record in records if not record.allowed]
    selected_target_features = [feature for feature in features if feature.startswith("target_") and feature != selected_target]
    if selected_target_features:
        failures.extend(
            LeakageRecord(
                feature,
                "target",
                False,
                "non-selected target columns are prohibited",
                "future_outcome",
            )
            for feature in selected_target_features
        )
    if failures:
        details = "; ".join(f"{record.feature}: {record.reason}" for record in failures)
        raise ValueError(f"Prohibited leakage-prone features in contract: {details}")
    return [record.to_dict() for record in records]
