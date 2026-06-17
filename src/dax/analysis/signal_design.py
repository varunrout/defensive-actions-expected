"""Transparent provisional descriptive defensive signal components."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalComponentSpec:
    """Metadata for one descriptive signal component."""

    name: str
    source_field: str
    denominator_field: str | None
    direction: int
    minimum_denominator: int
    interpretation: str
    limitation: str


SIGNAL_COMPONENTS: tuple[SignalComponentSpec, ...] = (
    SignalComponentSpec("activity_index", "actions_per_match", "matches", 1, 1, "How often the player records defensive actions per represented match.", "Not minutes-adjusted unless reliable minutes are added."),
    SignalComponentSpec("possession_win_index", "possession_win_rate", "possession_win_rate_denominator", 1, 30, "Share of actions that won possession.", "Descriptive and context-dependent."),
    SignalComponentSpec("threat_suppression_descriptive_index", "future_shot_rate", "future_shot_denominator", -1, 30, "Lower observed future-shot rate after actions.", "Outcome is not model-adjusted and is not causal."),
    SignalComponentSpec("phase_versatility_index", "phase_entropy", "total_actions", 1, 30, "Diversity of rule-based phase exposure.", "Phase labels are tactical proxies, not ground truth."),
    SignalComponentSpec("spatial_aggression_index", "mean_action_x", "total_actions", 1, 30, "Higher average action location toward the attacking goal.", "Depends on normalised coordinate convention."),
    SignalComponentSpec("transition_defence_exposure_index", "phase_transition_defence_share", "total_actions", 1, 30, "Share of actions in transition-defence proxy phases when available.", "Missing if transition-defence phase is not present."),
    SignalComponentSpec("box_defence_exposure_index", "box_defence_share", "box_defence_denominator", 1, 30, "Share of actions inside the canonical defensive penalty box.", "Exposure descriptor only; not a quality measure."),
    SignalComponentSpec("local_numerical_difficulty_index", "mean_local_numerical_balance_10m", "numerical_disadvantage_10m_denominator", -1, 20, "Lower local attacking-minus-defending balance indicates harder visible local context.", "Requires roles-known and locally visible 360 data."),
    SignalComponentSpec("visibility_reliability_index", "reliable_visibility_share", "reliable_visibility_share_denominator", 1, 30, "Share of actions with reliable local 5m and 10m visibility.", "Visibility reliability is coverage, not player quality."),
)


def _standardise(values: pd.Series, direction: int) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    std = valid.std(ddof=0)
    if std == 0 or np.isnan(std):
        z = pd.Series(0.0, index=valid.index)
    else:
        z = (valid - valid.mean()) / std
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[valid.index] = z * direction
    return out


def _phase_entropy(summary: pd.DataFrame) -> pd.Series:
    phase_columns = [column for column in summary.columns if column.startswith("phase_") and column.endswith("_share")]
    if not phase_columns:
        return pd.Series(np.nan, index=summary.index, dtype="float64")
    shares = summary[phase_columns].fillna(0).clip(lower=0)
    entropy = -(shares.where(shares > 0, np.nan) * np.log(shares.where(shares > 0, np.nan))).sum(axis=1)
    max_entropy = np.log(len(phase_columns)) if len(phase_columns) > 1 else 1.0
    return entropy / max_entropy


def _component_source(summary: pd.DataFrame, source_field: str) -> pd.Series:
    if source_field == "phase_entropy":
        return _phase_entropy(summary)
    if source_field in summary.columns:
        return pd.to_numeric(summary[source_field], errors="coerce")
    return pd.Series(np.nan, index=summary.index, dtype="float64")


def build_descriptive_signals(summary: pd.DataFrame, clusters: pd.DataFrame | None = None, min_actions: int = 30) -> pd.DataFrame:
    """Build provisional non-modelled signal components without fabricating missing inputs."""
    identity_columns = [column for column in ["player_id", "player_name", "team", "total_actions", "matches", "minimum_sample_flag"] if column in summary]
    output = summary[identity_columns].copy()
    warning_parts: list[pd.Series] = []

    for spec in SIGNAL_COMPONENTS:
        raw = _component_source(summary, spec.source_field)
        denominator = (
            pd.to_numeric(summary[spec.denominator_field], errors="coerce")
            if spec.denominator_field and spec.denominator_field in summary.columns
            else pd.Series(np.nan, index=summary.index, dtype="float64")
        )
        valid_denominator = denominator >= max(spec.minimum_denominator, min_actions if spec.denominator_field == "total_actions" else spec.minimum_denominator)
        calculable = raw.notna() & valid_denominator
        component = pd.Series(np.nan, index=summary.index, dtype="float64")
        component.loc[calculable] = _standardise(raw.loc[calculable], spec.direction)
        output[spec.name] = component
        output[f"{spec.name}_raw"] = raw
        output[f"{spec.name}_denominator"] = denominator
        output[f"{spec.name}_percentile"] = component.rank(pct=True)
        output[f"{spec.name}_interpretation"] = spec.interpretation
        output[f"{spec.name}_limitation"] = spec.limitation
        warning_parts.append(
            pd.Series(
                np.where(
                    raw.isna(),
                    f"{spec.name}: missing source field {spec.source_field}",
                    np.where(~valid_denominator, f"{spec.name}: denominator below minimum sample rule", ""),
                ),
                index=summary.index,
            )
        )

    output["reliable_sample"] = summary["total_actions"] >= min_actions
    combined_warnings = []
    for idx in summary.index:
        messages = [warnings.loc[idx] for warnings in warning_parts if warnings.loc[idx]]
        messages.append("descriptive provisional signal; not true DAx")
        combined_warnings.append("; ".join(messages))
    output["warnings"] = combined_warnings

    if clusters is not None and not clusters.empty:
        join_keys = [column for column in ["player_id", "team"] if column in output.columns and column in clusters.columns]
        cluster_columns = join_keys + [column for column in ["cluster", "selected_method", "selected_k", "membership_probability", "low_confidence_assignment"] if column in clusters]
        output = output.merge(clusters[cluster_columns], on=join_keys, how="left")
    return output
