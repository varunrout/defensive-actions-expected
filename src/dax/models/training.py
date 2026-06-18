"""Training orchestration for leakage-safe defensive-action models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .calibration import calibration_bin_table, fit_calibrated_classifier
from .classification import build_classifier
from .diagnostics import decile_table, save_classification_charts, save_regression_charts, subgroup_metrics
from .evaluation import classification_metrics, regression_metrics
from .feature_contracts import FeatureContract, get_contracts, load_model_config, resolve_contract
from .leakage import scan_features
from .mlflow_tracking import (
    configure_mlflow,
    log_artifact,
    log_artifacts,
    log_json_artifact,
    log_metrics,
    log_params,
    log_sklearn_model,
    sanitise_mlflow_model_name,
    start_parent_run,
    start_variant_run,
)
from .regression import build_regressor
from .schemas import dataset_fingerprint, normalise_model_schema, validate_model_dataset
from .splits import fold_summary, make_grouped_folds

IDENTITY_COLUMNS = ["event_id", "match_id", "player_id", "player_name", "team", "action_family", "phase_label", "position_group"]
CLASSIFICATION_CALIBRATION_METHODS = ["uncalibrated", "platt", "isotonic"]


@dataclass(frozen=True)
class VariantData:
    frame: pd.DataFrame
    rows_excluded: int
    matches_excluded: int
    eligibility: dict[str, Any]


@dataclass(frozen=True)
class VariantResult:
    variant: str
    run_id: str | None
    comparison_row: dict[str, Any]
    oof: pd.DataFrame
    artifacts: list[Path]


def git_sha() -> str | None:
    """Return the current git commit SHA when available."""

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_output_dirs(output_dir: str | Path, task: str) -> dict[str, Path]:
    """Create and return canonical output directories."""

    root = Path(output_dir)
    paths = {
        "root": root,
        "models": root / "models" / task,
        "validation": root / "validation" / task,
        "comparisons": root / "models" / "comparisons",
        "splits": root / "models" / "splits",
        "oof": root / "oof",
        "charts": root / "models" / task / "charts",
        "reports": root / "models" / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def select_variant_rows(df: pd.DataFrame, contract: FeatureContract) -> VariantData:
    """Apply variant-specific eligibility, including strict 360 rules."""

    if not contract.requires_360:
        return VariantData(df.copy(), rows_excluded=0, matches_excluded=0, eligibility={"rule": "all rows", "requires_360": False})

    required_role_features = [
        feature
        for feature in contract.features
        if feature
        in {
            "visible_attacker_count",
            "visible_defender_count",
            "attacker_defender_ratio",
            "nearest_attacker_distance",
            "nearest_defender_distance",
        }
    ]
    if "has_360" not in df.columns:
        raise ValueError(f"Variant {contract.name} requires 360 data but column 'has_360' is missing.")
    mask = df["has_360"].fillna(False).astype(bool)
    rules = ["has_360 is true"]
    if contract.require_roles_known:
        if "freeze_frame_roles_known" not in df.columns:
            raise ValueError(f"Variant {contract.name} requires freeze_frame_roles_known but the column is missing.")
        mask &= df["freeze_frame_roles_known"].fillna(False).astype(bool)
        rules.append("freeze_frame_roles_known is true")
    if contract.require_reliable_5m_visibility:
        if "reliable_5m_visibility" not in df.columns:
            raise ValueError(f"Variant {contract.name} requires reliable_5m_visibility but the column is missing.")
        mask &= df["reliable_5m_visibility"].fillna(False).astype(bool)
        rules.append("reliable_5m_visibility is true")
    if contract.require_reliable_10m_visibility:
        if "reliable_10m_visibility" not in df.columns:
            raise ValueError(f"Variant {contract.name} requires reliable_10m_visibility but the column is missing.")
        mask &= df["reliable_10m_visibility"].fillna(False).astype(bool)
        rules.append("reliable_10m_visibility is true")
    for feature in required_role_features:
        if feature not in df.columns:
            raise ValueError(f"Variant {contract.name} requires 360 role feature {feature!r}.")
        mask &= df[feature].notna()
    if required_role_features:
        rules.append("role-dependent 360 features are non-null: " + ", ".join(required_role_features))
    selected = df.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"Variant {contract.name} has no eligible 360 rows after applying visibility rules.")
    eligibility = {
        "rule": " AND ".join(rules),
        "requires_360": True,
        "required_role_features": required_role_features,
        "rows_retained": int(len(selected)),
        "rows_excluded": int(len(df) - len(selected)),
        "matches_retained": int(selected["match_id"].nunique()),
        "matches_excluded": int(df["match_id"].nunique() - selected["match_id"].nunique()),
        "visibility_quality_distribution": selected.get("visibility_quality_band", pd.Series(["unknown"] * len(selected))).value_counts(dropna=False).to_dict(),
    }
    return VariantData(
        selected,
        rows_excluded=int(len(df) - len(selected)),
        matches_excluded=int(df["match_id"].nunique() - selected["match_id"].nunique()),
        eligibility=eligibility,
    )


def save_fold_metadata(paths: dict[str, Path], contract: FeatureContract, folds: pd.DataFrame, df: pd.DataFrame) -> list[Path]:
    """Save row fold assignments and fold match membership."""

    assignments_path = paths["splits"] / f"{contract.task}_{contract.name}_fold_assignments.parquet"
    membership_path = paths["splits"] / f"{contract.task}_{contract.name}_fold_match_membership.json"
    summary_path = paths["splits"] / f"{contract.task}_{contract.name}_fold_summary.csv"

    folds.to_parquet(assignments_path, index=False)
    membership: dict[str, dict[str, list[str]]] = {}
    for fold in sorted(folds["fold"].unique()):
        validation_matches = sorted(map(str, df.loc[folds["fold"].eq(fold), "match_id"].unique()))
        training_matches = sorted(map(str, df.loc[~folds["fold"].eq(fold), "match_id"].unique()))
        if set(validation_matches).intersection(training_matches):
            raise ValueError(f"Fold {fold} has overlapping training and validation matches.")
        membership[str(int(fold))] = {"training_matches": training_matches, "validation_matches": validation_matches}
    membership_path.write_text(json.dumps(membership, indent=2), encoding="utf-8")
    fold_summary(df, contract.target, folds).to_csv(summary_path, index=False)
    return [assignments_path, membership_path, summary_path]


def validate_fold_support(df: pd.DataFrame, folds: pd.DataFrame, contract: FeatureContract) -> None:
    """Fail clearly when validation folds lack required target support."""

    for fold in sorted(folds["fold"].unique()):
        validation = df.loc[folds["fold"].eq(fold), contract.target]
        if contract.task == "classification" and set(validation.unique()) != {0, 1}:
            raise ValueError(f"Classification validation fold {fold} must contain positive and negative targets.")
        if contract.task == "regression" and int((validation > 0).sum()) == 0:
            raise ValueError(f"Regression validation fold {fold} contains no non-zero future-xG targets.")



def make_valid_variant_folds(
    df: pd.DataFrame,
    contract: FeatureContract,
    *,
    group_col: str,
    requested_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create grouped folds, retrying with fewer folds when support is insufficient."""

    unique_groups = int(df[group_col].nunique())
    max_folds = min(requested_folds, unique_groups)
    attempts: list[dict[str, Any]] = []
    for candidate_folds in range(max_folds, 1, -1):
        try:
            folds = make_grouped_folds(df, contract.target, group_col=group_col, n_splits=candidate_folds, seed=seed)
            validate_fold_support(df, folds, contract)
        except ValueError as exc:
            attempts.append({"fold_count": candidate_folds, "status": "failed", "reason": str(exc)})
            continue
        attempts.append({"fold_count": candidate_folds, "status": "selected", "reason": "target support satisfied"})
        metadata = {
            "requested_fold_count": requested_folds,
            "effective_fold_count": candidate_folds,
            "fallback_sequence": attempts,
            "rule": "retry fewer grouped folds until every validation fold has required target support",
        }
        return folds, metadata
    raise ValueError(f"No valid grouped split exists for {contract.name}: {attempts}")

def make_feature_frame(df: pd.DataFrame, resolved: dict[str, Any]) -> pd.DataFrame:
    """Return the model feature frame, allowing constant models with no columns."""

    features = resolved["final_features"]
    if not features:
        return pd.DataFrame(index=df.index)
    return df.loc[:, features]


def fold_target_counts(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, int]:
    """Return shot and xG support counts for one fold."""

    return {
        "training_positive_shots": int(train["target_future_shot_10s"].sum()),
        "validation_positive_shots": int(validation["target_future_shot_10s"].sum()),
        "training_nonzero_xg": int((train["target_future_xg_10s"] > 0).sum()),
        "validation_nonzero_xg": int((validation["target_future_xg_10s"] > 0).sum()),
    }


def aggregate_fold_metrics(fold_metrics: pd.DataFrame, metric_names: list[str]) -> dict[str, float]:
    """Aggregate fold metrics with mean/std/min/max suffixes."""

    aggregates: dict[str, float] = {}
    for metric in metric_names:
        if metric not in fold_metrics:
            continue
        values = pd.to_numeric(fold_metrics[metric], errors="coerce")
        aggregates[f"fold_{metric}_mean"] = float(values.mean())
        aggregates[f"fold_{metric}_std"] = float(values.std(ddof=0))
        aggregates[f"fold_{metric}_min"] = float(values.min())
        aggregates[f"fold_{metric}_max"] = float(values.max())
    return aggregates


def extract_coefficients(model: Any, contract: FeatureContract, resolved: dict[str, Any]) -> pd.DataFrame:
    """Extract linear coefficients with transformed feature names when available."""

    if not hasattr(model, "named_steps") or "model" not in model.named_steps:
        return pd.DataFrame()
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame()
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
    except Exception:  # noqa: BLE001 - feature names are best-effort artifacts
        names = np.asarray(resolved["final_features"])
    coefficients = np.ravel(estimator.coef_)
    if len(coefficients) != len(names):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "task": contract.task,
            "model_variant": contract.name,
            "feature": names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)


def build_oof_table(df: pd.DataFrame, contract: FeatureContract, folds: pd.DataFrame, predictions: np.ndarray, run_id: str | None) -> pd.DataFrame:
    """Build and validate an OOF table without repeating training match lists."""

    columns = [column for column in IDENTITY_COLUMNS if column in df.columns]
    oof = df.loc[:, columns].copy()
    oof["y_true"] = df[contract.target].to_numpy()
    oof["fold"] = folds["fold"].to_numpy()
    oof["model_variant"] = contract.name
    oof["model_family"] = contract.model_family
    oof["mlflow_run_id"] = run_id
    if contract.task == "classification":
        oof["y_score"] = predictions
    else:
        oof["y_pred"] = predictions
        oof["residual"] = oof["y_pred"] - oof["y_true"]
    if oof.duplicated(["event_id", "model_variant"]).any():
        raise ValueError(f"Duplicate OOF event-variant rows for {contract.name}.")
    prediction_column = "y_score" if contract.task == "classification" else "y_pred"
    if oof[prediction_column].isna().any():
        raise ValueError(f"Missing OOF predictions for {contract.name}.")
    if len(oof) != len(df):
        raise ValueError(f"OOF coverage mismatch for {contract.name}: expected {len(df)}, got {len(oof)}.")
    return oof


def fit_predict_variant(
    df: pd.DataFrame,
    contract: FeatureContract,
    resolved: dict[str, Any],
    folds: pd.DataFrame,
    *,
    seed: int,
    calibration_method: str = "uncalibrated",
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Fit one model per fold and return OOF predictions and fold metrics."""

    predictions = np.full(len(df), np.nan)
    fold_rows: list[dict[str, Any]] = []
    clipping = {"negative_prediction_count": 0, "clipping_count": 0}

    for fold in sorted(folds["fold"].unique()):
        validation_mask = folds["fold"].eq(fold).to_numpy()
        train = df.loc[~validation_mask]
        validation = df.loc[validation_mask]
        x_train = make_feature_frame(train, resolved)
        x_validation = make_feature_frame(validation, resolved)
        y_train = train[contract.target]
        y_validation = validation[contract.target]

        model = build_classifier(contract, resolved) if contract.task == "classification" else build_regressor(contract, resolved)
        fit_start = time.perf_counter()
        if contract.task == "classification":
            model = fit_calibrated_classifier(model, x_train, y_train, method=calibration_method, seed=seed)
        else:
            model.fit(x_train, y_train)
        fit_time = time.perf_counter() - fit_start

        inference_start = time.perf_counter()
        if contract.task == "classification":
            fold_predictions = model.predict_proba(x_validation)[:, 1]
            fold_metric_values = classification_metrics(y_validation, fold_predictions)
        else:
            raw_predictions = np.asarray(model.predict(x_validation), dtype=float)
            clipping["negative_prediction_count"] += int((raw_predictions < 0).sum())
            fold_predictions = np.clip(raw_predictions, 0, None)
            clipping["clipping_count"] += int((raw_predictions != fold_predictions).sum())
            fold_metric_values = regression_metrics(y_validation, fold_predictions)
        inference_time = time.perf_counter() - inference_start
        predictions[validation_mask] = fold_predictions

        row = {
            "fold": int(fold),
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "training_matches": int(train["match_id"].nunique()),
            "validation_matches": int(validation["match_id"].nunique()),
            "target_mean_or_prevalence": float(y_validation.mean()),
            "calibration_method": calibration_method,
            "calibration_training_positives": int(y_train.sum()) if contract.task == "classification" else 0,
            "calibration_training_negatives": int(len(y_train) - y_train.sum()) if contract.task == "classification" else 0,
            "fit_time": fit_time,
            "inference_time": inference_time,
            **fold_target_counts(train, validation),
            **fold_metric_values,
        }
        fold_rows.append(row)

    return predictions, pd.DataFrame(fold_rows), clipping


def train_variant(
    df: pd.DataFrame,
    contract: FeatureContract,
    config: dict[str, Any],
    input_path: str | Path,
    paths: dict[str, Path],
    mlflow: Any | None,
    fingerprint: dict[str, Any],
) -> VariantResult:
    """Train one variant, write artifacts, and log to MLflow."""

    variant_data = select_variant_rows(df, contract)
    variant_df = variant_data.frame.reset_index(drop=True)
    audit = validate_model_dataset(variant_df)
    resolved = resolve_contract(variant_df, contract)
    leakage_audit = scan_features(resolved["final_features"], selected_target=contract.target, feature_scope=contract.feature_scope)

    folds, fold_fallback_metadata = make_valid_variant_folds(
        variant_df,
        contract,
        group_col=config.get("group_column", "match_id"),
        requested_folds=config.get("folds", 5),
        seed=config.get("seed", 42),
    )
    fold_artifacts = save_fold_metadata(paths, contract, folds, variant_df)
    fallback_path = paths["splits"] / f"{contract.task}_{contract.name}_fold_fallback.json"
    fallback_path.write_text(json.dumps(fold_fallback_metadata, indent=2), encoding="utf-8")
    fold_artifacts.append(fallback_path)

    with start_variant_run(mlflow, contract.name) as run:
        run_id = getattr(getattr(run, "info", None), "run_id", None)
        calibration_method = "uncalibrated"
        predictions, fold_metrics, clipping = fit_predict_variant(
            variant_df,
            contract,
            resolved,
            folds,
            seed=config.get("seed", 42),
            calibration_method=calibration_method,
        )
        aggregate = classification_metrics(variant_df[contract.target], predictions) if contract.task == "classification" else regression_metrics(variant_df[contract.target], predictions)
        aggregate.update(aggregate_fold_metrics(fold_metrics, list(aggregate)))
        if contract.task == "regression":
            aggregate.update(
                {
                    "negative_prediction_count_before_clipping": clipping["negative_prediction_count"],
                    "clipping_count": clipping["clipping_count"],
                    "clipping_rate": clipping["clipping_count"] / max(len(variant_df), 1),
                    "future_xg_zero_rate": float((variant_df[contract.target] == 0).mean()),
                }
            )

        x_all = make_feature_frame(variant_df, resolved)
        final_model = build_classifier(contract, resolved) if contract.task == "classification" else build_regressor(contract, resolved)
        final_model.fit(x_all, variant_df[contract.target])
        oof = build_oof_table(variant_df, contract, folds, predictions, run_id)

        model_path = paths["models"] / f"{contract.name}.joblib"
        mlflow_model_name = sanitise_mlflow_model_name(f"{contract.name}/model")
        bundle = {
            "pipeline": final_model,
            "target": contract.target,
            "feature_contract": asdict(contract),
            "final_feature_list": resolved["final_features"],
            "preprocessing_metadata": resolved,
            "training_rows": len(variant_df),
            "training_matches": audit.matches,
            "data_fingerprint": fingerprint,
            "model_version": "0.1.0",
            "git_commit_sha": git_sha(),
            "mlflow_run_id": run_id,
            "mlflow_model_name": mlflow_model_name,
            "mlflow_model_uri": None,
            "mlflow_sklearn_serialization_format": config.get("mlflow", {}).get("sklearn_serialization_format", "cloudpickle"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(bundle, model_path)

        fold_metrics_path = paths["validation"] / f"{contract.name}_fold_metrics.csv"
        fold_metrics.to_csv(fold_metrics_path, index=False)
        oof_path = paths["validation"] / f"{contract.name}_oof_predictions.parquet"
        oof.to_parquet(oof_path, index=False)
        contract_path = paths["models"] / f"{contract.name}_feature_contract.json"
        leakage_path = paths["models"] / f"{contract.name}_leakage_audit.json"
        comparison_row_path = paths["models"] / f"{contract.name}_comparison_row.json"
        schema_path = paths["models"] / f"{contract.name}_input_schema.json"
        config_path = paths["models"] / f"{contract.name}_config_snapshot.json"
        environment_path = paths["models"] / f"{contract.name}_environment.json"

        log_json_artifact(mlflow, {"contract": asdict(contract), "resolved": resolved, "eligibility": variant_data.eligibility}, contract_path)
        log_json_artifact(mlflow, leakage_audit, leakage_path)
        log_json_artifact(mlflow, {"columns": {col: str(dtype) for col, dtype in variant_df.dtypes.items()}}, schema_path)
        log_json_artifact(mlflow, config, config_path)
        log_json_artifact(mlflow, {"git_sha": git_sha(), "timestamp": datetime.now(timezone.utc).isoformat()}, environment_path)

        coefficient_table = extract_coefficients(final_model, contract, resolved)
        coefficient_path = paths["validation"] / f"{contract.name}_coefficients.csv"
        if not coefficient_table.empty:
            coefficient_table.to_csv(coefficient_path, index=False)

        diagnostics_paths: list[Path] = []
        if contract.task == "classification":
            calibration_path = paths["validation"] / f"{contract.name}_calibration_bins.csv"
            decile_path = paths["validation"] / f"{contract.name}_score_deciles.csv"
            calibration_bin_table(oof["y_true"], oof["y_score"]).to_csv(calibration_path, index=False)
            decile_table(oof, "y_score", "y_true").to_csv(decile_path, index=False)
            chart_paths = save_classification_charts(oof, paths["charts"] / contract.name)
            diagnostics_paths.extend([calibration_path, decile_path, *chart_paths])
        else:
            decile_path = paths["validation"] / f"{contract.name}_prediction_deciles.csv"
            subgroup_path = paths["validation"] / f"{contract.name}_subgroup_metrics.csv"
            decile_table(oof, "y_pred", "y_true").to_csv(decile_path, index=False)
            subgroup_metrics(oof, "y_pred", "y_true", ["phase_label", "action_family", "position_group"]).to_csv(subgroup_path, index=False)
            chart_paths = save_regression_charts(oof, paths["charts"] / contract.name)
            diagnostics_paths.extend([decile_path, subgroup_path, *chart_paths])

        comparison_row = {
            "variant": contract.name,
            "model_family": contract.model_family,
            "feature_count": len(resolved["final_features"]),
            "rows": len(variant_df),
            "matches": audit.matches,
            "rows_excluded": variant_data.rows_excluded,
            "matches_excluded": variant_data.matches_excluded,
            "requires_360": contract.requires_360,
            "missing_features": ";".join(resolved["missing_optional_features"]),
            "mlflow_run_id": run_id,
            "recommendation_status": "reference baseline" if "baseline" in contract.model_family or "constant" in contract.model_family else "candidate",
            "calibration_method": calibration_method,
            "effective_fold_count": int(folds["fold"].nunique()),
            **aggregate,
        }
        log_json_artifact(mlflow, comparison_row, comparison_row_path)

        log_params(
            mlflow,
            {
                "target": contract.target,
                "model_family": contract.model_family,
                "model_variant": contract.name,
                "feature_scope": contract.feature_scope,
                "feature_list": resolved["final_features"],
                "categorical_features": resolved["categorical"],
                "numeric_features": resolved["numeric"],
                "fold_count": int(folds["fold"].nunique()),
                "seed": config.get("seed", 42),
                "rows": len(variant_df),
                "matches": audit.matches,
                "target_prevalence": audit.shot_rate,
                "coverage_360": resolved["coverage_360"],
                "hyperparameters": contract.hyperparameters,
                "git_sha": git_sha(),
                "data_fingerprint": fingerprint,
            },
        )
        log_metrics(mlflow, aggregate)
        log_artifacts(mlflow, [model_path, fold_metrics_path, oof_path, *fold_artifacts, *diagnostics_paths], artifact_path=contract.name)
        if not coefficient_table.empty:
            log_artifact(mlflow, coefficient_path, artifact_path=contract.name)
        logged_model = log_sklearn_model(
            mlflow,
            final_model,
            artifact_path=f"{contract.name}/model",
            variant=contract.name,
            serialization_format=config.get("mlflow", {}).get("sklearn_serialization_format", "cloudpickle"),
            trusted_types=config.get("mlflow", {}).get("skops_trusted_types"),
        )
        if logged_model is not None:
            bundle["mlflow_model_name"] = logged_model["name"]
            bundle["mlflow_model_uri"] = logged_model["model_uri"]
            bundle["mlflow_sklearn_serialization_format"] = logged_model["serialization_format"]
            joblib.dump(bundle, model_path)

    return VariantResult(contract.name, run_id, comparison_row, oof, [model_path, fold_metrics_path, oof_path])


def assign_recommendations(comparison: pd.DataFrame, task: str) -> pd.DataFrame:
    """Assign transparent recommendation statuses using multiple metrics."""

    if comparison.empty:
        return comparison
    comparison = comparison.copy()
    comparison["selection_rule"] = (
        "classification: calibration/ECE, log loss, Brier, AP, fold stability, interpretability, complexity, 360 dependence"
        if task == "classification"
        else "regression: MAE, RMSE, bias, non-zero performance, fold stability, interpretability, complexity, 360 dependence"
    )
    non_reference = ~comparison["recommendation_status"].eq("reference baseline")
    if task == "classification" and non_reference.any():
        ordered = comparison.loc[non_reference].sort_values(
            ["expected_calibration_error", "log_loss", "brier_score", "fold_log_loss_std", "feature_count"],
            ascending=[True, True, True, True, True],
        )
    elif task == "regression" and non_reference.any():
        ordered = comparison.loc[non_reference].assign(abs_bias=lambda x: x["prediction_bias"].abs()).sort_values(
            ["mae", "rmse", "abs_bias", "fold_mae_std", "feature_count"],
            ascending=[True, True, True, True, True],
        )
    else:
        ordered = pd.DataFrame()
    if not ordered.empty:
        comparison.loc[ordered.index[0], "recommendation_status"] = "preferred candidate"
    return comparison



def write_population_comparisons(comparison: pd.DataFrame, task: str, comparison_dir: Path) -> None:
    """Write labelled comparison tables for native, common, non-360, and 360 populations."""

    if comparison.empty:
        return
    native = comparison.copy()
    native["comparison_population"] = "native eligible population"
    native.to_csv(comparison_dir / f"{task}_native_population_comparison.csv", index=False)

    completed = comparison[comparison.get("rows", pd.Series(dtype=float)).notna()].copy()
    if not completed.empty:
        common_rows = int(completed["rows"].min())
        common = completed.copy()
        common["comparison_population"] = "common-row diagnostic population"
        common["common_population_rows"] = common_rows
        common["population_warning"] = "metrics are native OOF metrics labelled for row-count comparability; rerun row-restricted scoring for final claims"
        common.to_csv(comparison_dir / f"{task}_common_population_comparison.csv", index=False)

    non_360 = comparison[comparison.get("requires_360", False).eq(False)].copy() if "requires_360" in comparison else pd.DataFrame()
    if not non_360.empty:
        non_360["comparison_population"] = "all-data non-360 variants"
        non_360.to_csv(comparison_dir / f"{task}_all_data_non_360_comparison.csv", index=False)

    only_360 = comparison[comparison.get("requires_360", False).eq(True)].copy() if "requires_360" in comparison else pd.DataFrame()
    if not only_360.empty:
        only_360["comparison_population"] = "360-only eligible variants"
        only_360.to_csv(comparison_dir / f"{task}_360_only_comparison.csv", index=False)

def run_training(
    task: str,
    input_path: str | Path,
    config_path: str | Path = "configs/models.yaml",
    output_dir: str | Path = "outputs",
    mlflow_enabled: bool | None = None,
    tracking_uri: str | None = None,
    n_splits: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Run the modelling workflow for one task."""

    config = load_model_config(config_path)
    if n_splits is not None:
        config["folds"] = n_splits
    df = normalise_model_schema(pd.read_parquet(input_path))
    if max_rows is not None:
        df = df.head(max_rows).copy()
    audit = validate_model_dataset(df)
    fingerprint = dataset_fingerprint(input_path, df)
    paths = ensure_output_dirs(output_dir, task)
    mlflow = configure_mlflow(config, enabled=mlflow_enabled, tracking_uri=tracking_uri)
    experiment = config["mlflow"][f"{task}_experiment"]

    results: list[VariantResult] = []
    skipped_rows: list[dict[str, Any]] = []
    with start_parent_run(mlflow, experiment, f"{config['mlflow'].get('run_name_prefix', 'baseline')}-{task}") as parent_run:
        parent_run_id = getattr(getattr(parent_run, "info", None), "run_id", None)
        log_json_artifact(mlflow, fingerprint, paths["reports"] / f"{task}_dataset_fingerprint.json")
        log_params(mlflow, {"task": task, "rows": audit.rows, "matches": audit.matches, "input_path": str(input_path)})
        for contract in get_contracts(config, task):
            try:
                results.append(train_variant(df, contract, config, input_path, paths, mlflow, fingerprint))
            except ValueError as exc:
                skipped_rows.append({"variant": contract.name, "model_family": contract.model_family, "status": "failed", "reason": str(exc)})

    if results:
        oof_all = pd.concat([result.oof for result in results], ignore_index=True)
        oof_name = "classification_oof.parquet" if task == "classification" else "regression_oof.parquet"
        oof_all.to_parquet(paths["oof"] / oof_name, index=False)
    comparison = pd.DataFrame([result.comparison_row for result in results] + skipped_rows)
    comparison = assign_recommendations(comparison, task)
    comparison_path = paths["comparisons"] / f"{task}_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    write_population_comparisons(comparison, task, paths["comparisons"])
    return {"parent_run_id": parent_run_id, "comparison": str(comparison_path), "comparison_frame": comparison}


def write_legacy_outputs(result: dict[str, Any], task: str, validation_dir: str | None, oof_dir: str | None, output_dir: str | Path) -> None:
    """Write compatibility outputs expected by existing CLI tests."""

    if validation_dir:
        legacy = "baseline" if task == "classification" else "regression"
        directory = Path(validation_dir) / legacy
        directory.mkdir(parents=True, exist_ok=True)
        rows = result["comparison_frame"].to_dict(orient="records") if hasattr(result["comparison_frame"], "to_dict") else []
        variants = {str(row.get("variant")): {**row, "avg_precision": row.get("average_precision", row.get("avg_precision"))} for row in rows}
        summary = {
            "task": task,
            "rows": int(rows[0].get("rows", 0)) if rows else 0,
            "matches": int(rows[0].get("matches", 0)) if rows else 0,
            "target_rate": float(rows[0].get("positive_rate", 0.0)) if rows else 0.0,
            "target_mean": float(rows[0].get("mean_observed", 0.0)) if rows else 0.0,
            "variants": variants,
        }
        metrics_name = "baseline_model_metrics.json" if task == "classification" else "regression_model_metrics.json"
        table_name = "baseline_model_metrics_table.csv" if task == "classification" else "regression_model_metrics_table.csv"
        (directory / metrics_name).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        result["comparison_frame"].to_csv(directory / table_name, index=False)
    if oof_dir:
        source = Path(output_dir) / "oof" / ("classification_oof.parquet" if task == "classification" else "regression_oof.parquet")
        destination = Path(oof_dir) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            destination.write_bytes(source.read_bytes())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-safe DAx models.")
    parser.add_argument("--task", choices=["classification", "regression", "logistic", "all"], required=True)
    parser.add_argument("--input", default="data/features/player_defensive_actions.parquet")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--models-dir")
    parser.add_argument("--validation-dir")
    parser.add_argument("--oof-dir")
    parser.add_argument("--mlflow-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--n-splits", type=int)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = ["classification", "regression"] if args.task == "all" else ["classification" if args.task == "logistic" else args.task]
    for task in tasks:
        result = run_training(
            task,
            args.input,
            args.config,
            args.output_dir,
            (False if args.mlflow_enabled is None and (args.validation_dir or args.oof_dir or args.models_dir) else args.mlflow_enabled),
            args.tracking_uri,
            args.n_splits,
            args.max_rows,
        )
        print({k: v for k, v in result.items() if k != "comparison_frame"})
        write_legacy_outputs(result, task, args.validation_dir, args.oof_dir, args.output_dir)


if __name__ == "__main__":
    main()
