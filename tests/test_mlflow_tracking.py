from __future__ import annotations

import sys
import types

import pytest

from dax.models.mlflow_tracking import INSTALL_COMMAND, MLflowConfigurationError, configure_mlflow, start_parent_run, start_variant_run


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
        self.sklearn = types.SimpleNamespace(log_model=lambda model, artifact_path: self.artifacts.append(("model", artifact_path)))
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
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    mlflow = configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "file:///tmp/mlruns-test"}}, enabled=True)
    assert mlflow is fake
    assert fake.tracking_uri == "file:///tmp/mlruns-test"
    with start_parent_run(mlflow, "classification-exp", "parent") as parent:
        with start_variant_run(mlflow, "variant") as child:
            assert parent.info.run_id == "run-1"
            assert child.info.run_id == "run-2"
    assert fake.experiments == ["classification-exp"]
    assert fake.runs[0]["nested"] is False
    assert fake.runs[1]["nested"] is True
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)


def test_environment_and_explicit_tracking_uri_overrides(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///env-store")
    configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "file:///config-store"}}, enabled=True)
    assert fake.tracking_uri == "file:///env-store"
    configure_mlflow({"mlflow": {"enabled": True, "tracking_uri": "file:///config-store"}}, enabled=True, tracking_uri="file:///explicit-store")
    assert fake.tracking_uri == "file:///explicit-store"
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
