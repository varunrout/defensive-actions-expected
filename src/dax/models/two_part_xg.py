"""Two-part hurdle future-xG modelling utilities."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, Ridge, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import safe_spearman


@dataclass(frozen=True)
class ConditionalModelSpec:
    """Specification for one conditional severity candidate."""

    name: str
    model_family: str
    hyperparameters: dict[str, Any]
    reference_status: str = "candidate"


def load_classification_oof(oof_path: str | Path) -> pd.DataFrame:
    """Load and minimally validate classification OOF predictions."""

    frame = pd.read_parquet(oof_path)
    required = {"event_id", "fold", "y_score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Classification OOF missing required columns: {sorted(required - set(frame.columns))}")
    return frame


def filter_classification_oof(frame: pd.DataFrame, classification_variant: str) -> pd.DataFrame:
    """Filter classification OOF to one variant with unique event IDs."""

    source = frame.copy()
    if "model_variant" in source.columns:
        available = sorted(map(str, source["model_variant"].dropna().unique().tolist()))
        source = source.loc[source["model_variant"].eq(classification_variant)].copy()
        if source.empty:
            raise ValueError(
                f"Classification OOF does not contain variant {classification_variant!r}. "
                f"Available variants: {available}"
            )
    else:
        source["model_variant"] = classification_variant

    if source.duplicated(["event_id"]).any():
        raise ValueError(f"Classification OOF variant {classification_variant!r} contains duplicate event IDs.")
    return source.reset_index(drop=True)


def _build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def build_conditional_regressor(
    model_family: str,
    hyperparameters: dict[str, Any],
    categorical: list[str],
    numeric: list[str],
) -> Pipeline | None:
    """Build conditional severity estimator pipeline; mean baseline returns None."""

    if model_family == "conditional_mean":
        return None

    prep = _build_preprocessor(categorical, numeric)
    if model_family in {"log_ridge", "ridge"}:
        estimator = Ridge(random_state=42, **hyperparameters)
    elif model_family == "gamma":
        estimator = GammaRegressor(**hyperparameters)
    elif model_family == "tweedie":
        estimator = TweedieRegressor(**hyperparameters)
    elif model_family == "hist_gradient_boosting_regressor":
        estimator = HistGradientBoostingRegressor(random_state=42, **hyperparameters)
    else:
        raise ValueError(f"Unknown conditional model family: {model_family}")
    return Pipeline([("preprocess", prep), ("model", estimator)])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def compute_conditional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute conditional metrics on non-zero rows only."""

    if len(y_true) == 0:
        return {
            "conditional_mae": float("nan"),
            "conditional_rmse": float("nan"),
            "conditional_r2": float("nan"),
            "conditional_spearman": float("nan"),
            "conditional_prediction_bias": float("nan"),
            "observed_conditional_mean": float("nan"),
            "predicted_conditional_mean": float("nan"),
            "conditional_sample_count": 0,
        }

    return {
        "conditional_mae": float(mean_absolute_error(y_true, y_pred)),
        "conditional_rmse": _rmse(y_true, y_pred),
        "conditional_r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "conditional_spearman": float(safe_spearman(y_true, y_pred)),
        "conditional_prediction_bias": float((y_pred - y_true).mean()),
        "observed_conditional_mean": float(y_true.mean()),
        "predicted_conditional_mean": float(y_pred.mean()),
        "conditional_sample_count": int(len(y_true)),
    }


def compute_hurdle_metrics(oof: pd.DataFrame, *, prediction_col: str = "combined_future_xg_prediction") -> dict[str, float]:
    """Compute combined hurdle metrics on all rows, with non-zero subgroup reporting."""

    y_true = oof["observed_future_xg"].to_numpy(dtype=float)
    y_pred = oof[prediction_col].to_numpy(dtype=float)
    nonzero = y_true > 0

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "spearman": float(safe_spearman(y_true, y_pred)),
        "prediction_bias": float((y_pred - y_true).mean()),
        "nonzero_mae": float(mean_absolute_error(y_true[nonzero], y_pred[nonzero])) if nonzero.any() else float("nan"),
        "nonzero_rmse": _rmse(y_true[nonzero], y_pred[nonzero]) if nonzero.any() else float("nan"),
        "nonzero_spearman": float(safe_spearman(y_true[nonzero], y_pred[nonzero])) if nonzero.any() else float("nan"),
        "rows": int(len(oof)),
        "matches": int(oof["match_id"].nunique()) if "match_id" in oof.columns else 0,
    }
    zero = ~nonzero
    metrics["zero_target_mae"] = float(mean_absolute_error(y_true[zero], y_pred[zero])) if zero.any() else float("nan")
    metrics["nonzero_target_mae"] = metrics["nonzero_mae"]
    metrics["nonzero_target_rmse"] = metrics["nonzero_rmse"]
    metrics["nonzero_target_spearman"] = metrics["nonzero_spearman"]
    return metrics


def decile_ranking(oof: pd.DataFrame, *, prediction_col: str = "combined_future_xg_prediction") -> pd.DataFrame:
    """Build decile ranking table and cumulative xG capture."""

    frame = oof[[prediction_col, "observed_future_xg"]].copy()
    frame["decile"] = pd.qcut(frame[prediction_col].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    grouped = (
        frame.groupby("decile", dropna=False)
        .agg(
            rows=("observed_future_xg", "size"),
            mean_prediction=(prediction_col, "mean"),
            mean_observed=("observed_future_xg", "mean"),
            total_observed=("observed_future_xg", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("decile", ascending=False).reset_index(drop=True)
    grouped["cumulative_observed"] = grouped["total_observed"].cumsum()
    grouped["cumulative_rows"] = grouped["rows"].cumsum()
    total = grouped["total_observed"].sum()
    grouped["cumulative_pct_xg"] = 0.0 if total <= 0 else 100.0 * grouped["cumulative_observed"] / total
    grouped["cumulative_pct_rows"] = 100.0 * grouped["cumulative_rows"] / max(len(frame), 1)
    return grouped.sort_values("decile").reset_index(drop=True)


def cumulative_capture(deciles: pd.DataFrame, top_pct_rows: float) -> float:
    """Return cumulative observed xG captured by top X percent rows."""

    threshold = top_pct_rows * 100.0
    ordered = deciles.sort_values("decile", ascending=False)
    eligible = ordered.loc[ordered["cumulative_pct_rows"] <= threshold + 1e-9]
    if eligible.empty:
        first = ordered.head(1)
        return float(first["cumulative_pct_xg"].iloc[0]) if not first.empty else float("nan")
    return float(eligible["cumulative_pct_xg"].max())


def fit_conditional_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    spec: ConditionalModelSpec | str,
    hyperparameters: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
    target_col: str = "target_future_xg_10s",
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit conditional model on non-zero train rows and evaluate only on non-zero validation rows."""

    if isinstance(spec, str):
        resolved_dict = resolved if resolved is not None else {}
        spec_obj = ConditionalModelSpec(name=f"conditional_{spec}", model_family=spec, hyperparameters=hyperparameters or {})
    else:
        resolved_dict = hyperparameters if isinstance(hyperparameters, dict) and resolved is None else resolved
        spec_obj = spec
    if resolved_dict is None:
        raise ValueError("Resolved feature metadata is required for fit_conditional_fold.")

    nonzero_train = train.loc[train[target_col].gt(0)].copy()
    if nonzero_train.empty:
        raise ValueError("Conditional fold has no non-zero training rows.")

    x_train = nonzero_train[resolved_dict["final_features"]].copy()
    y_train = nonzero_train[target_col].to_numpy(dtype=float)
    x_validation = validation[resolved_dict["final_features"]].copy()

    fit_start = time.perf_counter()
    model = build_conditional_regressor(spec_obj.model_family, spec_obj.hyperparameters, resolved_dict["categorical"], resolved_dict["numeric"])
    if spec_obj.model_family == "conditional_mean":
        payload: dict[str, Any] = {"mean": float(y_train.mean())}
    elif spec_obj.model_family == "log_ridge":
        payload = {"pipeline": model}
        assert model is not None
        model.fit(x_train, np.log1p(y_train))
    else:
        payload = {"pipeline": model}
        assert model is not None
        model.fit(x_train, y_train)
    fit_time = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    if spec_obj.model_family == "conditional_mean":
        prediction_all = np.full(len(validation), payload["mean"], dtype=float)
    elif spec_obj.model_family == "log_ridge":
        raw = np.asarray(payload["pipeline"].predict(x_validation), dtype=float)
        prediction_all = np.clip(np.expm1(raw), 0, None)
    else:
        raw = np.asarray(payload["pipeline"].predict(x_validation), dtype=float)
        prediction_all = np.clip(raw, 0, None)
    inference_time = time.perf_counter() - inference_start

    nonzero_validation = validation[target_col].to_numpy(dtype=float) > 0
    y_val_conditional = validation.loc[nonzero_validation, target_col].to_numpy(dtype=float)
    y_pred_conditional = prediction_all[nonzero_validation]

    metrics = compute_conditional_metrics(y_val_conditional, y_pred_conditional)
    # Legacy aliases retained for test compatibility.
    metrics["mae"] = metrics["conditional_mae"]
    metrics["rmse"] = metrics["conditional_rmse"]
    metrics["r2"] = metrics["conditional_r2"]
    metrics["spearman"] = metrics["conditional_spearman"]
    metrics["prediction_bias"] = metrics["conditional_prediction_bias"]
    metrics["mean_observed"] = metrics["observed_conditional_mean"]
    metrics["mean_prediction"] = metrics["predicted_conditional_mean"]
    metrics["nonzero_target_mae"] = metrics["conditional_mae"]
    metrics["nonzero_target_rmse"] = metrics["conditional_rmse"]
    metrics["nonzero_target_spearman"] = metrics["conditional_spearman"]
    metrics.update(
        {
            "fit_time": float(fit_time),
            "inference_time": float(inference_time),
            "validation_rows": int(len(validation)),
            "validation_nonzero_rows": int(nonzero_validation.sum()),
            "training_nonzero_rows": int(len(nonzero_train)),
        }
    )
    return prediction_all, metrics


def train_conditional_variant(
    df: pd.DataFrame,
    folds: pd.Series,
    spec: ConditionalModelSpec,
    resolved: dict[str, Any],
    *,
    target_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run fold training for one conditional variant."""

    prediction = np.full(len(df), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(pd.Series(folds).dropna().unique().tolist()):
        val_mask = folds.eq(fold).to_numpy()
        train = df.loc[~val_mask]
        validation = df.loc[val_mask]
        fold_prediction, metrics = fit_conditional_fold(train, validation, spec, resolved=resolved, target_col=target_col)
        prediction[val_mask] = fold_prediction
        fold_rows.append({"fold": int(fold), **metrics})

    if np.isnan(prediction).any():
        raise ValueError(f"Conditional OOF has missing predictions for {spec.name}.")
    return prediction, pd.DataFrame(fold_rows)


def fit_final_conditional_model(
    df: pd.DataFrame,
    spec: ConditionalModelSpec,
    resolved: dict[str, Any],
    *,
    target_col: str,
) -> tuple[dict[str, Any], int, int]:
    """Fit final conditional model on all non-zero rows and return serializable payload."""

    nonzero = df.loc[df[target_col].gt(0)].copy()
    if nonzero.empty:
        raise ValueError("Final conditional model cannot fit: no non-zero rows.")

    x = nonzero[resolved["final_features"]].copy()
    y = nonzero[target_col].to_numpy(dtype=float)
    payload: dict[str, Any]
    if spec.model_family == "conditional_mean":
        payload = {"model_family": spec.model_family, "mean": float(y.mean())}
    elif spec.model_family == "log_ridge":
        pipeline = build_conditional_regressor(spec.model_family, spec.hyperparameters, resolved["categorical"], resolved["numeric"])
        assert pipeline is not None
        pipeline.fit(x, np.log1p(y))
        payload = {"model_family": spec.model_family, "pipeline": pipeline}
    else:
        pipeline = build_conditional_regressor(spec.model_family, spec.hyperparameters, resolved["categorical"], resolved["numeric"])
        assert pipeline is not None
        pipeline.fit(x, y)
        payload = {"model_family": spec.model_family, "pipeline": pipeline}
    return payload, int(len(nonzero)), int(nonzero["match_id"].nunique())


def predict_conditional(payload: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Predict conditional severity using a saved payload."""

    family = payload.get("model_family")
    if family == "conditional_mean":
        return np.full(len(features), float(payload["mean"]), dtype=float)
    if family == "log_ridge":
        raw = np.asarray(payload["pipeline"].predict(features), dtype=float)
        return np.clip(np.expm1(raw), 0, None)
    raw = np.asarray(payload["pipeline"].predict(features), dtype=float)
    return np.clip(raw, 0, None)


def build_combined_oof(
    df: pd.DataFrame,
    classification_oof: pd.DataFrame,
    conditional_prediction: np.ndarray,
    folds: pd.Series | pd.DataFrame,
    model_spec: ConditionalModelSpec | None = None,
    classification_variant: str | None = None,
    run_id: str | None = None,
    *,
    conditional_variant: str | None = None,
) -> pd.DataFrame:
    """Build combined hurdle OOF for one classification+conditional pair."""

    if isinstance(folds, pd.DataFrame):
        folds_series = folds["fold"]
    else:
        folds_series = folds
    cond_variant = conditional_variant or (model_spec.name if model_spec is not None else "conditional_model")
    class_variant = classification_variant or str(classification_oof.get("model_variant", pd.Series(["classification_model"])).iloc[0])
    if "model_variant" in classification_oof.columns:
        classification_oof = filter_classification_oof(classification_oof, class_variant)

    class_frame = classification_oof[["event_id", "y_score"]].copy()
    keep = [
        "event_id",
        "match_id",
        "player_id",
        "player_name",
        "team",
        "action_family",
        "phase_label",
        "position_group",
        "reliable_5m_visibility",
        "has_360",
        "target_future_shot_10s",
        "target_future_xg_10s",
    ]
    merged = df[[column for column in keep if column in df.columns]].copy()
    merged["fold"] = np.asarray(folds_series)
    merged = merged.merge(class_frame, on="event_id", how="left")
    if merged["y_score"].isna().any():
        raise ValueError("Could not align all rows with classification OOF scores.")

    merged = merged.rename(
        columns={
            "target_future_shot_10s": "observed_future_shot",
            "target_future_xg_10s": "observed_future_xg",
            "y_score": "oof_shot_probability",
        }
    )
    merged["conditional_xg_prediction"] = np.asarray(conditional_prediction, dtype=float)
    merged["combined_future_xg_prediction"] = merged["oof_shot_probability"] * merged["conditional_xg_prediction"]
    merged["classification_model_variant"] = class_variant
    merged["conditional_model_variant"] = cond_variant
    merged["mlflow_run_id"] = run_id

    if merged["combined_future_xg_prediction"].isna().any():
        raise ValueError("Combined OOF has missing predictions.")
    if merged.duplicated(["event_id", "classification_model_variant", "conditional_model_variant"]).any():
        raise ValueError("Combined OOF has duplicate event/variant rows.")
    return merged


def fold_metric_summary(oof: pd.DataFrame) -> dict[str, float]:
    """Compute fold mean/std for MAE on combined prediction."""

    rows: list[float] = []
    for fold in sorted(oof["fold"].dropna().unique().tolist()):
        subset = oof.loc[oof["fold"].eq(fold)]
        rows.append(float(mean_absolute_error(subset["observed_future_xg"], subset["combined_future_xg_prediction"])))
    if not rows:
        return {"fold_mean": float("nan"), "fold_std": float("nan")}
    arr = np.asarray(rows, dtype=float)
    return {"fold_mean": float(arr.mean()), "fold_std": float(arr.std(ddof=0))}


def safe_numeric(value: Any) -> float:
    """Convert a value to finite float where possible."""

    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return val if math.isfinite(val) else float("nan")



