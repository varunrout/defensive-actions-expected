from __future__ import annotations

import sys
import types

import pytest

from dax.models.mlflow_tracking import (
    INSTALL_COMMAND,
    MLflowConfigurationError,
    configure_mlflow,
    log_sklearn_model,
    sanitise_mlflow_model_name,
    start_parent_run,
    start_variant_run,
)


class FakeRun:
    def __init__(self, run_id):
        self.info = types.SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMlflow(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.tracking_uri = None
        self.experiments = []
        self.runs = []
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.sklearn = types.SimpleNamespace(
            SERIALIZATION_FORMAT_CLOUDPICKLE="cloudpickle",
            SERIALIZATION_FORMAT_PICKLE="pickle",
            SERIALIZATION_FORMAT_SKOPS="skops",
            log_model=lambda sk_model=None, artifact_path=None, name=None, serialization_format=None: self.artifacts.append(("model", name or artifact_path, serialization_format)) or types.SimpleNamespace(model_uri=f"models:/{name or artifact_path}"),
        )
        self.tracking = types.SimpleNamespace(MlflowClient=lambda tracking_uri=None: types.SimpleNamespace(search_experiments=lambda max_results=1: []))

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiments.append(name)

    def start_run(self, run_name=None, nested=False):
        run = FakeRun(f"run-{len(self.runs)+1}")
        self.runs.append({"run_name": run_name, "nested": nested, "run_id": run.info.run_id})
        return run

    def log_param(self, key, value):
        self.params.append((key, value))

    def log_metric(self, key, value):
        self.metrics.append((key, value))

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


def test_mlflow_disabled_returns_none():
    assert configure_mlflow({"mlflow": {"enabled": True}}, enabled=False) is None


def test_mlflow_enabled_but_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(MLflowConfigurationError, match="Install the modelling dependencies"):
        configure_mlflow({"mlflow": {"enabled": True}}, enabled=True)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_mlflow_enabled_installed_and_run_nesting(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    mlflow = configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "sqlite:////tmp/mlflow-test.db"}}, enabled=True)
    assert mlflow is fake
    assert fake.tracking_uri == "sqlite:////tmp/mlflow-test.db"
    with start_parent_run(mlflow, "classification-exp", "parent") as parent:
        with start_variant_run(mlflow, "variant") as child:
            assert parent.info.run_id == "run-1"
            assert child.info.run_id == "run-2"
    assert fake.experiments == ["classification-exp"]
    assert fake.runs[0]["nested"] is False
    assert fake.runs[1]["nested"] is True
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_configuration_tracking_uri_used_without_env_or_explicit(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "sqlite:////tmp/config-only.db"}}, enabled=True)
    assert fake.tracking_uri == "sqlite:////tmp/config-only.db"
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_no_tracking_uri_is_set_when_none_supplied(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": None}}, enabled=True)
    assert fake.tracking_uri is None
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_environment_and_explicit_tracking_uri_overrides(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/env-mlflow.db")
    configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "sqlite:////tmp/config-mlflow.db"}}, enabled=True)
    assert fake.tracking_uri == "sqlite:////tmp/env-mlflow.db"
    configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "sqlite:////tmp/config-mlflow.db"}}, enabled=True, tracking_uri="sqlite:////tmp/explicit-mlflow.db")
    assert fake.tracking_uri == "sqlite:////tmp/explicit-mlflow.db"
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_unreachable_remote_uri_fails(monkeypatch):
    fake = FakeMlflow()
    fake.tracking = types.SimpleNamespace(
        MlflowClient=lambda tracking_uri=None: types.SimpleNamespace(search_experiments=lambda max_results=1: (_ for _ in ()).throw(RuntimeError("down")))
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with pytest.raises(MLflowConfigurationError, match="Unable to reach"):
        configure_mlflow({"mlflow": {"enabled": True}}, enabled=True, tracking_uri="http://127.0.0.1:1")
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_sanitise_mlflow_model_name():
    cases = {
        "b0_constant/model": "b0_constant_model",
        "name:with:colon": "name_with_colon",
        "name.with.periods": "name_with_periods",
        "name%with%percent": "name_with_percent",
        "name\"with'quotes": "name_with_quotes",
        r"name\with\backslash": "name_with_backslash",
        "name with spaces": "name_with_spaces",
        "": "model",
        "////": "model",
        "--a///b:::c..d%%e--": "--a_b_c_d_e--",
    }
    for raw, expected in cases.items():
        assert sanitise_mlflow_model_name(raw) == expected


def test_log_sklearn_model_uses_safe_model_name(monkeypatch):
    fake = FakeMlflow()
    result = log_sklearn_model(fake, object(), "b0_constant/model", variant="b0_constant")
    assert result["name"] == "b0_constant_model"
    assert result["model_uri"] == "models:/b0_constant_model"
    assert ("model", "b0_constant_model", "cloudpickle") in fake.artifacts


def test_log_sklearn_model_uses_pickle_when_requested():
    fake = FakeMlflow()
    result = log_sklearn_model(fake, object(), "r0_constant/model", serialization_format="pickle")
    assert result["serialization_format"] == "pickle"
    assert ("model", "r0_constant_model", "pickle") in fake.artifacts


def test_skops_requires_trusted_types():
    fake = FakeMlflow()
    with pytest.raises(MLflowConfigurationError, match="trusted-types"):
        log_sklearn_model(fake, object(), "b0_constant/model", serialization_format="skops")


def test_log_metrics_skips_non_finite_and_records_status():
    from dax.models.mlflow_tracking import log_metrics

    fake = FakeMlflow()
    fake.tags = []
    fake.set_tag = lambda key, value: fake.tags.append((key, value))

    metrics = {"finite": 1.25, "nan_metric": float("nan"), "pos_inf": float("inf"), "neg_inf": float("-inf"), "text": "not numeric"}
    skipped = log_metrics(fake, metrics)

    assert fake.metrics == [("finite", 1.25)]
    assert skipped == ["nan_metric", "pos_inf", "neg_inf"]
    assert ("metric_status_nan_metric", "undefined_non_finite") in fake.params
    assert ("metric_status_pos_inf", "undefined_non_finite") in fake.params
    assert ("metric_status_neg_inf", "undefined_non_finite") in fake.params
    assert ("metric_status_nan_metric", "undefined_non_finite") in fake.tags
    assert metrics["nan_metric"] != metrics["nan_metric"]


def test_log_metrics_prefixes_status_names():
    from dax.models.mlflow_tracking import log_metrics

    fake = FakeMlflow()
    skipped = log_metrics(fake, {"spearman": float("nan")}, prefix="fold_")

    assert skipped == ["fold_spearman"]
    assert fake.metrics == []
    assert ("metric_status_fold_spearman", "undefined_non_finite") in fake.params


def test_log_sklearn_model_disables_metric_replay_when_signature_supports_it():
    fake = FakeMlflow()
    calls = []

    def log_model(sk_model=None, name=None, serialization_format=None, log_model_metrics=True):
        calls.append({"name": name, "log_model_metrics": log_model_metrics, "serialization_format": serialization_format})
        return types.SimpleNamespace(model_uri=f"models:/{name}")

    fake.sklearn.log_model = log_model
    result = log_sklearn_model(fake, object(), "r0_constant/model")

    assert result["model_uri"] == "models:/r0_constant_model"
    assert calls == [{"name": "r0_constant_model", "log_model_metrics": False, "serialization_format": "cloudpickle"}]


def test_log_sklearn_model_does_not_pass_unsupported_metric_replay_option():
    fake = FakeMlflow()
    calls = []

    def log_model(sk_model=None, name=None, serialization_format=None):
        calls.append({"name": name, "serialization_format": serialization_format})
        return types.SimpleNamespace(model_uri=f"models:/{name}")

    fake.sklearn.log_model = log_model
    result = log_sklearn_model(fake, object(), "r0_constant/model")

    assert result["model_uri"] == "models:/r0_constant_model"
    assert calls == [{"name": "r0_constant_model", "serialization_format": "cloudpickle"}]
