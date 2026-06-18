"""Two-part hurdle-style future-xG modelling: classification + conditional severity."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import safe_spearman
from .feature_contracts import FeatureContract, resolve_contract
from .mlflow_tracking import (
    configure_mlflow,
    log_artifact,
    log_json_artifact,
    log_metrics,
    log_params,
    start_parent_run,
    start_variant_run,
)


@dataclass(frozen=True)
class ConditionalModelSpec:
    """Specification for a conditional severity model variant."""
    name: str
    model_family: str
    hyperparameters: dict[str, Any]


def filter_classification_oof(
    oof: pd.DataFrame,
    classification_variant: str,
) -> pd.DataFrame:
    """Return one classification OOF variant with unique event rows."""

    if "model_variant" not in oof.columns:
        filtered = oof.copy()
        filtered["model_variant"] = classification_variant
    else:
        filtered = oof.loc[oof["model_variant"].eq(classification_variant)].copy()
        if filtered.empty:
            available = sorted(map(str, oof["model_variant"].dropna().unique().tolist()))
            raise ValueError(
                f"Classification OOF does not contain variant {classification_variant!r}. "
                f"Available variants: {available}"
            )

    if filtered.duplicated(["event_id"]).any():
        raise ValueError(
            f"Classification OOF variant {classification_variant!r} contains duplicate event_id rows."
        )
    return filtered.reset_index(drop=True)


def load_classification_oof(oof_path: str | Path) -> pd.DataFrame:
    """Load and validate classification OOF predictions."""
    oof = pd.read_parquet(oof_path)
    required_cols = {"event_id", "y_score", "fold"}
    if not required_cols.issubset(oof.columns):
        raise ValueError(f"Classification OOF must have columns: {required_cols}. Got: {set(oof.columns)}")
    return oof


def build_conditional_regressor(model_family: str, hyperparameters: dict[str, Any], categorical: list[str], numeric: list[str]) -> Pipeline:
    """Build a regressor pipeline for conditional severity modelling."""
    
    prep = ColumnTransformer([
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical),
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ]), numeric)
    ])
    
    if model_family == 'ridge':
        model = Ridge(random_state=42, **hyperparameters)
    elif model_family == 'hist_gradient_boosting_regressor':
        model = HistGradientBoostingRegressor(random_state=42, **hyperparameters)
    elif model_family == 'log_ridge':
        # Ridge on log-transformed target; we'll handle transformation in fit_conditional_fold
        model = Ridge(random_state=42, **hyperparameters)
    else:
        raise ValueError(f"Unknown conditional model family: {model_family}")
    
    return Pipeline([('preprocess', prep), ('model', model)])


def fit_conditional_fold(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame,
    model_family: str,
    hyperparameters: dict[str, Any],
    resolved: dict[str, Any],
    target_col: str = "target_future_xg_10s",
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Fit conditional severity regressor on non-zero training rows.
    
    Returns:
        predictions on validation rows (including zeros where applicable)
        fold metrics dictionary
    """
    
    # Select only non-zero training rows for fitting
    nonzero_mask = (df_train[target_col] > 0).to_numpy()
    df_nonzero = df_train.loc[nonzero_mask].reset_index(drop=True)
    
    if df_nonzero.empty:
        raise ValueError(f"Conditional fold has no non-zero training rows; cannot fit model.")
    
    x_nonzero = df_nonzero[resolved["final_features"]].copy()
    y_nonzero = df_nonzero[target_col].copy()
    
    # Apply log transformation if specified
    if model_family == 'log_ridge':
        y_nonzero_transformed = np.log1p(y_nonzero)
    else:
        y_nonzero_transformed = y_nonzero
    
    # Build and fit model
    regressor = build_conditional_regressor(
        'ridge' if model_family == 'log_ridge' else model_family,
        hyperparameters,
        resolved["categorical"],
        resolved["numeric"]
    )
    
    fit_start = time.perf_counter()
    regressor.fit(x_nonzero, y_nonzero_transformed)
    fit_time = time.perf_counter() - fit_start
    
    # Predict on full validation set (including zeros)
    x_validation = df_validation[resolved["final_features"]].copy()
    inference_start = time.perf_counter()
    raw_preds = np.asarray(regressor.predict(x_validation), dtype=float)
    inference_time = time.perf_counter() - inference_start
    
    # Inverse log transform if necessary
    if model_family == 'log_ridge':
        predictions = np.expm1(raw_preds)
        predictions = np.clip(predictions, 0, None)
    else:
        predictions = np.clip(raw_preds, 0, None)
    
    # Compute metrics on full validation set
    y_validation = df_validation[target_col].to_numpy()
    nonzero_val = y_validation > 0
    
    metrics = {
        "mae": float(mean_absolute_error(y_validation, predictions)),
        "rmse": float(mean_squared_error(y_validation, predictions) ** 0.5),
        "r2": float(r2_score(y_validation, predictions)) if len(y_validation) > 1 else float("nan"),
        "spearman": float(safe_spearman(y_validation, predictions)),
        "mean_prediction": float(predictions.mean()),
        "mean_observed": float(y_validation.mean()),
        "prediction_bias": float((predictions - y_validation).mean()),
        "zero_target_mae": float(mean_absolute_error(y_validation[~nonzero_val], predictions[~nonzero_val])) if (~nonzero_val).any() else float("nan"),
        "nonzero_target_mae": float(mean_absolute_error(y_validation[nonzero_val], predictions[nonzero_val])) if nonzero_val.any() else float("nan"),
        "nonzero_target_rmse": float(mean_squared_error(y_validation[nonzero_val], predictions[nonzero_val]) ** 0.5) if nonzero_val.any() else float("nan"),
        "nonzero_target_spearman": float(safe_spearman(y_validation[nonzero_val], predictions[nonzero_val])),
        "fit_time": fit_time,
        "inference_time": inference_time,
        "training_nonzero_rows": int(len(df_nonzero)),
        "validation_rows": int(len(df_validation)),
        "validation_nonzero_rows": int(nonzero_val.sum()),
        "validation_zero_rows": int((~nonzero_val).sum()),
    }
    
    return predictions, metrics


def train_conditional_variant(
    df: pd.DataFrame,
    classification_oof: pd.DataFrame,
    model_spec: ConditionalModelSpec,
    resolved: dict[str, Any],
    folds: pd.DataFrame,
    *,
    seed: int = 42,
    target_col: str = "target_future_xg_10s",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Train conditional severity model using grouped folds from classification OOF.
    
    Returns:
        OOF predictions array
        DataFrame with fold metrics
    """
    
    predictions = np.full(len(df), np.nan)
    fold_rows: list[dict[str, Any]] = []
    
    # Merge classification OOF to get fold assignments
    df_with_folds = df.copy()
    df_with_folds = df_with_folds.merge(
        classification_oof[["event_id", "fold"]],
        on="event_id",
        how="left"
    )
    if df_with_folds["fold"].isna().any():
        raise ValueError(f"Could not align all rows with classification OOF fold assignments.")
    
    folds = df_with_folds[["fold"]].copy()
    
    for fold in sorted(folds["fold"].unique()):
        validation_mask = folds["fold"].eq(fold).to_numpy()
        train = df.loc[~validation_mask]
        validation = df.loc[validation_mask]
        
        fold_predictions, fold_metrics = fit_conditional_fold(
            train,
            validation,
            model_spec.model_family,
            model_spec.hyperparameters,
            resolved,
            target_col=target_col,
            seed=seed,
        )
        predictions[validation_mask] = fold_predictions
        
        fold_row = {
            "fold": int(fold),
            **fold_metrics
        }
        fold_rows.append(fold_row)
    
    return predictions, pd.DataFrame(fold_rows)


def build_combined_oof(
    df: pd.DataFrame,
    classification_oof: pd.DataFrame,
    conditional_predictions: np.ndarray,
    folds: pd.DataFrame,
    model_spec: ConditionalModelSpec,
    classification_variant: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Build combined OOF table with hurdle predictions."""

    classification_oof = filter_classification_oof(classification_oof, classification_variant)
    
    identity_cols = ["event_id", "match_id", "player_id", "team", "action_family", "phase_label", "position_group"]
    retained_cols = [col for col in identity_cols if col in df.columns]
    
    oof = df.loc[:, retained_cols].copy()
    oof["observed_future_shot"] = df["target_future_shot_10s"].to_numpy()
    oof["observed_future_xg"] = df["target_future_xg_10s"].to_numpy()
    oof["fold"] = folds["fold"].to_numpy()
    
    # Merge classification probability
    oof = oof.merge(
        classification_oof[["event_id", "y_score", "model_variant"]],
        on="event_id",
        how="left"
    )
    if oof["y_score"].isna().any():
        raise ValueError("Could not align all rows with classification OOF probabilities.")
    oof = oof.rename(columns={"y_score": "oof_shot_probability", "model_variant": "classification_model_variant"})
    
    # Add conditional predictions
    oof["conditional_xg_prediction"] = conditional_predictions
    
    # Combine: hurdle prediction
    oof["combined_future_xg_prediction"] = oof["oof_shot_probability"] * oof["conditional_xg_prediction"]
    
    # Add metadata
    oof["conditional_model_variant"] = model_spec.name
    oof["mlflow_run_id"] = run_id
    
    # Validate
    required_cols = {
        "event_id", "observed_future_shot", "observed_future_xg",
        "oof_shot_probability", "conditional_xg_prediction", "combined_future_xg_prediction"
    }
    if not required_cols.issubset(oof.columns):
        raise ValueError(f"Combined OOF missing required columns: {required_cols - set(oof.columns)}")
    
    if oof.duplicated(["event_id", "conditional_model_variant", "classification_model_variant"]).any():
        raise ValueError("Duplicate OOF event-variant-classification combinations.")
    
    if oof[["oof_shot_probability", "conditional_xg_prediction", "combined_future_xg_prediction"]].isna().any().any():
        raise ValueError("Missing OOF predictions in combined table.")
    
    return oof


def compute_hurdle_metrics(oof: pd.DataFrame) -> dict[str, float]:
    """Compute comprehensive metrics for hurdle model evaluation."""
    
    y_true = oof["observed_future_xg"].to_numpy()
    y_combined = oof["combined_future_xg_prediction"].to_numpy()
    y_conditional = oof["conditional_xg_prediction"].to_numpy()
    
    nonzero = y_true > 0
    zero = ~nonzero
    
    metrics = {
        # Overall
        "mae": float(mean_absolute_error(y_true, y_combined)),
        "rmse": float(mean_squared_error(y_true, y_combined) ** 0.5),
        "r2": float(r2_score(y_true, y_combined)) if len(y_true) > 1 else float("nan"),
        "spearman": float(safe_spearman(y_true, y_combined)),
        "mean_prediction": float(y_combined.mean()),
        "mean_observed": float(y_true.mean()),
        "prediction_bias": float((y_combined - y_true).mean()),
        
        # Zero rows
        "zero_target_mae": float(mean_absolute_error(y_true[zero], y_combined[zero])) if zero.any() else float("nan"),
        "false_threat_mass": float(y_combined[zero].sum()),
        "mean_predicted_xg_on_zeros": float(y_combined[zero].mean()) if zero.any() else float("nan"),
        
        # Non-zero rows
        "nonzero_target_mae": float(mean_absolute_error(y_true[nonzero], y_combined[nonzero])) if nonzero.any() else float("nan"),
        "nonzero_target_rmse": float(mean_squared_error(y_true[nonzero], y_combined[nonzero]) ** 0.5) if nonzero.any() else float("nan"),
        "nonzero_target_spearman": float(safe_spearman(y_true[nonzero], y_combined[nonzero])),
        
        # Calibration: E[conditional | nonzero] vs E[y | nonzero]
        "conditional_mean_nonzero": float(y_conditional[nonzero].mean()) if nonzero.any() else float("nan"),
        "observed_mean_nonzero": float(y_true[nonzero].mean()) if nonzero.any() else float("nan"),
        
        # Coverage and support
        "total_rows": int(len(oof)),
        "zero_rows": int(zero.sum()),
        "nonzero_rows": int(nonzero.sum()),
    }
    
    return metrics


def decile_ranking(oof: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction decile ranking and cumulative xG capture."""
    
    df = oof[["combined_future_xg_prediction", "observed_future_xg"]].copy()
    df["decile"] = pd.qcut(df["combined_future_xg_prediction"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    
    result = (
        df.groupby("decile", dropna=False)
        .agg(
            rows=("observed_future_xg", "size"),
            mean_prediction=("combined_future_xg_prediction", "mean"),
            mean_observed=("observed_future_xg", "mean"),
            total_observed=("observed_future_xg", "sum"),
        )
        .reset_index()
    )
    
    # Cumulative capture from highest decile down
    result = result.sort_values("decile", ascending=False).reset_index(drop=True)
    result["cumulative_observed"] = result["total_observed"].cumsum()
    result["cumulative_rows"] = result["rows"].cumsum()
    
    total_xg = result["total_observed"].sum()
    if total_xg > 0:
        result["cumulative_pct_xg"] = 100.0 * result["cumulative_observed"] / total_xg
        result["cumulative_pct_rows"] = 100.0 * result["cumulative_rows"] / len(oof)
    
    return result.sort_values("decile")



