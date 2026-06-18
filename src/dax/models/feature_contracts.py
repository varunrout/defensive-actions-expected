"""Feature-contract loading and resolution for modelling variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .leakage import scan_features


@dataclass(frozen=True)
class FeatureContract:
    task: str
    name: str
    target: str
    model_family: str
    feature_scope: str
    categorical: list[str]
    numeric: list[str]
    required: list[str]
    optional: list[str]
    excluded: list[str]
    hyperparameters: dict[str, Any]
    requires_360: bool
    minimum_usable_rows: int
    require_roles_known: bool = False
    require_reliable_5m_visibility: bool = False
    require_reliable_10m_visibility: bool = False

    @property
    def features(self) -> list[str]:
        """Return categorical and numeric features without duplicates."""

        return list(dict.fromkeys([*self.categorical, *self.numeric]))


def load_model_config(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    """Load the model YAML configuration."""

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def duplicate_features(features: list[str]) -> list[str]:
    """Return duplicate feature names in deterministic order."""

    return sorted({feature for feature in features if features.count(feature) > 1})


def contract_from_config(task: str, target: str, name: str, raw: dict[str, Any]) -> FeatureContract:
    """Build and validate a contract from one YAML variant entry."""

    categorical = list(raw.get("categorical_features", []))
    numeric = list(raw.get("numeric_features", []))
    all_features = categorical + numeric
    duplicates = duplicate_features(all_features)
    if duplicates:
        raise ValueError(f"Duplicate features in {name}: {duplicates}")
    feature_scope = raw.get("feature_scope", "pre_action_context")
    scan_features(all_features, selected_target=target, feature_scope=feature_scope)
    return FeatureContract(
        task=task,
        name=name,
        target=target,
        model_family=raw["model_family"],
        feature_scope=feature_scope,
        categorical=categorical,
        numeric=numeric,
        required=list(raw.get("required_features", [])),
        optional=list(raw.get("optional_features", [])),
        excluded=list(raw.get("excluded_features", [])),
        hyperparameters=dict(raw.get("hyperparameters", {})),
        requires_360=bool(raw.get("requires_360", False)),
        minimum_usable_rows=int(raw.get("minimum_usable_rows", 1)),
        require_roles_known=bool(raw.get("require_roles_known", False)),
        require_reliable_5m_visibility=bool(raw.get("require_reliable_5m_visibility", False)),
        require_reliable_10m_visibility=bool(raw.get("require_reliable_10m_visibility", False)),
    )


def get_contracts(config: dict[str, Any], task: str) -> list[FeatureContract]:
    """Return all contracts for a modelling task."""

    target = config[task]["target"]
    return [contract_from_config(task, target, name, raw) for name, raw in config[task]["variants"].items()]


def resolve_contract(df: pd.DataFrame, contract: FeatureContract) -> dict[str, Any]:
    """Resolve a contract against an input dataframe and fail on missing required features."""

    missing_required = [feature for feature in contract.required if feature not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required features for {contract.name}: {missing_required}")

    categorical = [feature for feature in contract.categorical if feature in df.columns]
    numeric = [feature for feature in contract.numeric if feature in df.columns]
    final_features = list(dict.fromkeys([*categorical, *numeric]))
    scan_features(final_features, selected_target=contract.target, feature_scope=contract.feature_scope)
    missing_optional = [feature for feature in contract.optional if feature not in df.columns]
    has_360 = df.get("has_360", pd.Series([False] * len(df), index=df.index)).fillna(False)

    return {
        "requested_features": contract.features,
        "available_features": [feature for feature in contract.features if feature in df.columns],
        "missing_required_features": missing_required,
        "missing_optional_features": missing_optional,
        "final_features": final_features,
        "categorical": categorical,
        "numeric": numeric,
        "rows_retained": int(len(df.dropna(subset=[contract.target]))),
        "rows_excluded": int(df[contract.target].isna().sum()),
        "feature_missingness": {feature: float(df[feature].isna().mean()) for feature in final_features},
        "coverage_360": float(has_360.mean()),
    }
