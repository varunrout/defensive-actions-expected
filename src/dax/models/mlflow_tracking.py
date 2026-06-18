"""MLflow integration helpers for the modelling workflow.

All direct MLflow calls live here so training code can be exercised with MLflow
explicitly disabled and so missing/unreachable tracking backends fail with a
clear, actionable error.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import inspect
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

INSTALL_COMMAND = 'python -m pip install -e ".[dev,visualization,models]"'


class MLflowConfigurationError(RuntimeError):
    """Raised when MLflow is requested but cannot be configured."""


@dataclass
class DisabledRun(AbstractContextManager):
    """Context-manager compatible run object used when tracking is disabled."""

    run_id: str | None = None

    @property
    def info(self) -> Any:
        return type("Info", (), {"run_id": self.run_id})()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


def import_mlflow() -> Any:
    """Import MLflow or raise a clear installation error."""

    try:
        import mlflow  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatch tests
        raise MLflowConfigurationError(
            "MLflow tracking is enabled but the 'mlflow' package is unavailable. "
            f"Install the modelling dependencies with: {INSTALL_COMMAND}"
        ) from exc
    return mlflow


def resolve_tracking_uri(config: dict[str, Any], tracking_uri: str | None = None) -> str | None:
    """Return the effective tracking URI without silently changing destinations."""

    return tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or config.get("tracking_uri")


def configure_mlflow(
    config: dict[str, Any] | None,
    *,
    enabled: bool | None = None,
    tracking_uri: str | None = None,
) -> Any | None:
    """Configure MLflow and return the module, or ``None`` when disabled.

    When enabled and an explicit remote/file URI is supplied, this function keeps
    that exact destination. It does not fall back to a different tracking store.
    """

    mlflow_config = (config or {}).get("mlflow", config or {})
    effective_enabled = bool(mlflow_config.get("enabled", True) if enabled is None else enabled)
    if not effective_enabled:
        return None

    mlflow = import_mlflow()
    uri = resolve_tracking_uri(mlflow_config, tracking_uri)
    if uri:
        mlflow.set_tracking_uri(uri)

    # Force a lightweight connectivity check for explicitly remote stores so
    # unreachable URIs fail before training starts. Local file stores are created
    # lazily by MLflow and should not require a server.
    if uri and uri.startswith(("http://", "https://")):
        try:
            mlflow.tracking.MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
        except Exception as exc:  # noqa: BLE001 - preserve MLflow-specific error context
            raise MLflowConfigurationError(f"Unable to reach configured MLflow tracking URI {uri!r}.") from exc
    return mlflow


def start_parent_run(mlflow: Any | None, experiment_name: str, run_name: str) -> AbstractContextManager:
    """Start a parent run, or a disabled run when MLflow is off."""

    if mlflow is None:
        return DisabledRun()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def start_variant_run(mlflow: Any | None, run_name: str) -> AbstractContextManager:
    """Start a nested variant run."""

    if mlflow is None:
        return DisabledRun()
    return mlflow.start_run(run_name=run_name, nested=True)


def log_params(mlflow: Any | None, params: dict[str, Any]) -> None:
    """Log parameters, serialising complex values safely."""

    if mlflow is None:
        return
    for key, value in params.items():
        serialised = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else value
        mlflow.log_param(key, str(serialised)[:1000])


def log_metrics(mlflow: Any | None, metrics: dict[str, Any], *, prefix: str = "") -> list[str]:
    """Log finite numeric metrics and mark undefined non-finite metrics.

    Local metric tables are built before this helper is called, so NaN and
    infinite analytical results remain preserved there. MLflow's SQLite metric
    table cannot safely replay duplicate non-finite metric rows during model
    logging, so only finite values are sent to ``mlflow.log_metric``.
    """

    skipped: list[str] = []
    if mlflow is None:
        return skipped
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}"
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(metric_value):
            skipped.append(metric_name)
            status_name = f"metric_status_{metric_name}"[:250]
            mlflow.log_param(status_name, "undefined_non_finite")
            if hasattr(mlflow, "set_tag"):
                mlflow.set_tag(status_name, "undefined_non_finite")
            continue
        mlflow.log_metric(metric_name, metric_value)
    return skipped


def write_json_artifact(obj: Any, path: str | Path) -> Path:
    """Write a JSON artifact and return its path."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return artifact_path


def log_artifact(mlflow: Any | None, path: str | Path, artifact_path: str | None = None) -> None:
    """Log a file artifact when MLflow is enabled."""

    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifacts(mlflow: Any | None, paths: Iterable[str | Path], artifact_path: str | None = None) -> None:
    """Log many file artifacts when present."""

    for path in paths:
        if Path(path).exists():
            log_artifact(mlflow, path, artifact_path=artifact_path)


def log_json_artifact(mlflow: Any | None, obj: Any, path: str | Path, artifact_path: str | None = None) -> Path:
    """Write and optionally log a JSON artifact."""

    json_path = write_json_artifact(obj, path)
    log_artifact(mlflow, json_path, artifact_path=artifact_path)
    return json_path


INVALID_MODEL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def sanitise_mlflow_model_name(value: str | None) -> str:
    """Return a MLflow logged-model compatible name.

    Names are restricted to predictable alphanumeric, underscore, and hyphen
    characters. Empty/all-invalid values receive a stable fallback name.
    """

    raw = (value or "").strip()
    safe = INVALID_MODEL_NAME_CHARS.sub("_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "model"



def resolve_sklearn_serialization_format(
    mlflow: Any,
    requested: str = "cloudpickle",
    *,
    trusted_types: list[str] | None = None,
) -> Any:
    """Resolve an explicit MLflow sklearn serialization format.

    CloudPickle is the repository default because the modelling package includes
    trusted project-owned custom estimators. Skops requires an explicit trusted
    types policy so the caller does not accidentally rely on a permissive load
    path.
    """

    normalised = (requested or "cloudpickle").lower()
    if normalised == "cloudpickle":
        return getattr(mlflow.sklearn, "SERIALIZATION_FORMAT_CLOUDPICKLE", "cloudpickle")
    if normalised == "pickle":
        return getattr(mlflow.sklearn, "SERIALIZATION_FORMAT_PICKLE", "pickle")
    if normalised == "skops":
        if not trusted_types:
            raise MLflowConfigurationError("skops serialization requires explicit trusted-types configuration.")
        return getattr(mlflow.sklearn, "SERIALIZATION_FORMAT_SKOPS", "skops")
    raise MLflowConfigurationError(f"Unsupported sklearn serialization format: {requested!r}.")

def log_sklearn_model(
    mlflow: Any | None,
    model: Any,
    artifact_path: str,
    *,
    variant: str | None = None,
    serialization_format: str = "cloudpickle",
    trusted_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """Log a scikit-learn compatible model and return MLflow model metadata.

    Current MLflow versions prefer ``name=`` and ``sk_model=``. Older versions
    used ``artifact_path=``. Both paths receive the same sanitised value.
    """

    if mlflow is None:
        return None
    safe_model_name = sanitise_mlflow_model_name(artifact_path)
    resolved_serialization_format = resolve_sklearn_serialization_format(
        mlflow,
        serialization_format,
        trusted_types=trusted_types,
    )
    try:
        parameters = inspect.signature(mlflow.sklearn.log_model).parameters
        kwargs: dict[str, Any] = {
            "sk_model": model,
            "serialization_format": resolved_serialization_format,
        }
        if "name" in parameters:
            kwargs["name"] = safe_model_name
        elif "artifact_path" in parameters:
            kwargs["artifact_path"] = safe_model_name
        else:
            raise MLflowConfigurationError("mlflow.sklearn.log_model supports neither 'name' nor 'artifact_path'.")

        replay_disable_options = {
            "log_model_metrics": False,
            "log_metrics": False,
            "associate_logged_model_metrics": False,
        }
        for option, disabled_value in replay_disable_options.items():
            if option in parameters:
                kwargs[option] = disabled_value
        model_info = mlflow.sklearn.log_model(**kwargs)
    except Exception as exc:  # noqa: BLE001 - model logging should not hide training results
        variant_text = f" for variant {variant!r}" if variant else ""
        raise MLflowConfigurationError(f"Failed to log sklearn model{variant_text} as {safe_model_name!r}.") from exc
    return {"name": safe_model_name, "model_info": model_info, "model_uri": getattr(model_info, "model_uri", None), "serialization_format": serialization_format}
