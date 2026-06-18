"""MLflow integration helpers for the modelling workflow.

All direct MLflow calls live here so training code can be exercised with MLflow
explicitly disabled and so missing/unreachable tracking backends fail with a
clear, actionable error.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
import os
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


def log_metrics(mlflow: Any | None, metrics: dict[str, Any], *, prefix: str = "") -> None:
    """Log numeric metrics."""

    if mlflow is None:
        return
    for key, value in metrics.items():
        try:
            mlflow.log_metric(f"{prefix}{key}", float(value))
        except (TypeError, ValueError):
            continue


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


def log_sklearn_model(mlflow: Any | None, model: Any, artifact_path: str) -> None:
    """Log a scikit-learn compatible model when supported."""

    if mlflow is None:
        return
    try:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001 - model logging should not hide training results
        raise MLflowConfigurationError(f"Failed to log sklearn model to MLflow artifact path {artifact_path!r}.") from exc
