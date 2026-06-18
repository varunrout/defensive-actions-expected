from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dax.models.calibration import fit_calibrated_classifier
from dax.models.feature_contracts import get_contracts, load_model_config
from dax.models.training import run_training, select_variant_rows


def production_fixture() -> pd.DataFrame:
    rows = []
    phases = ["press", "block", "transition", "set_piece"]
    families = ["pressure", "duel", "interception", "recovery"]
    positions = ["DF", "MF", "FW", "GK"]
    event_types = ["Pressure", "Duel", "Interception", "Ball Recovery"]
    for match in range(1, 9):
        for i in range(6):
            has_360 = match % 2 == 0
            shot = int(i % 3 == 0)
            xg = 0.08 + 0.01 * i if i % 3 == 0 else 0.0
            roles_known = has_360 and not (match == 2 and i == 5)
            rows.append(
                {
                    "match_id": match,
                    "event_id": f"m{match}_e{i}",
                    "player_id": 100 + (i % 5),
                    "player_name": f"Player {i % 5}",
                    "team": f"Team {match % 3}",
                    "event_type": event_types[i % len(event_types)],
                    "action_family": families[i % len(families)],
                    "phase_label": phases[i % len(phases)],
                    "action_x": float(10 + i * 8 + match),
                    "action_y": float(20 + i * 4),
                    "position_group": positions[i % len(positions)],
                    "action_zone": "central" if i % 2 else "wide",
                    "distance_to_center_line": float(abs(40 - (20 + i * 4))),
                    "play_pattern": "Regular Play" if match % 2 else "Counter",
                    "possession_elapsed_seconds": float(i * 3),
                    "event_order_in_possession": i + 1,
                    "has_360": has_360,
                    "freeze_frame_roles_known": roles_known,
                    "reliable_5m_visibility": has_360 and i != 4,
                    "reliable_10m_visibility": has_360 and i != 4,
                    "visibility_quality_band": "high" if roles_known else "unknown",
                    "visible_attacker_count": float(3 + i) if roles_known else np.nan,
                    "visible_defender_count": float(4 + i) if roles_known else np.nan,
                    "attacker_defender_ratio": float((3 + i) / (4 + i)) if roles_known else np.nan,
                    "nearest_attacker_distance": float(2 + i) if roles_known else np.nan,
                    "nearest_defender_distance": float(1 + i) if roles_known else np.nan,
                    "target_future_shot_10s": shot,
                    "target_future_xg_10s": xg,
                }
            )
    return pd.DataFrame(rows)


def test_360_eligibility_uses_canonical_visibility_fields():
    cfg = load_model_config("configs/models.yaml")
    contract = next(c for c in get_contracts(cfg, "classification") if c.name == "b5_360_geometry")
    selected = select_variant_rows(production_fixture(), contract)
    assert selected.eligibility["requires_360"] is True
    assert "freeze_frame_roles_known is true" in selected.eligibility["rule"]
    assert selected.eligibility["visibility_quality_distribution"]["high"] > 0
    assert selected.rows_excluded > 0


def test_fixture_workflow_all_families_and_population_tables(tmp_path: Path):
    data_path = tmp_path / "features.parquet"
    production_fixture().to_parquet(data_path, index=False)
    class_result = run_training("classification", data_path, output_dir=tmp_path / "out", mlflow_enabled=False, n_splits=3)
    reg_result = run_training("regression", data_path, output_dir=tmp_path / "out", mlflow_enabled=False, n_splits=3)
    class_comparison = class_result["comparison_frame"]
    reg_comparison = reg_result["comparison_frame"]
    assert "hist_gradient_boosting_classifier" in set(class_comparison["model_family"])
    assert "hist_gradient_boosting_regressor" in set(reg_comparison["model_family"])
    assert (tmp_path / "out/models/comparisons/classification_native_population_comparison.csv").exists()
    assert (tmp_path / "out/models/comparisons/classification_360_only_comparison.csv").exists()
    assert (tmp_path / "out/models/comparisons/regression_all_data_non_360_comparison.csv").exists()
    assert pd.read_parquet(tmp_path / "out/oof/classification_oof.parquet").duplicated(["event_id", "model_variant"]).sum() == 0
    assert (tmp_path / "out/validation/classification/b1_phase_only_fold_metrics.csv").exists()


class RecordingEstimator:
    def __init__(self):
        self.fit_indices_ = None

    def fit(self, x, y):
        self.fit_indices_ = list(x.index)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        return np.c_[np.full(len(x), 0.6), np.full(len(x), 0.4)]


def test_uncalibrated_calibration_uses_outer_training_only():
    x_train = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
    y_train = pd.Series([0, 1, 0, 1], index=x_train.index)
    model = fit_calibrated_classifier(RecordingEstimator(), x_train, y_train, method="uncalibrated")
    assert model.fit_indices_ == [10, 11, 12, 13]
    assert 99 not in model.fit_indices_


def test_isotonic_calibration_downgrades_by_failing_when_support_is_insufficient():
    x_train = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
    y_train = pd.Series([0, 0, 0, 1], index=x_train.index)
    with pytest.raises(ValueError, match="Insufficient training support"):
        fit_calibrated_classifier(RecordingEstimator(), x_train, y_train, method="isotonic")


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="MLflow is not installed in this environment")
def test_enabled_mlflow_sqlite_tracking_integration(tmp_path: Path):
    data_path = tmp_path / "features.parquet"
    production_fixture().to_parquet(data_path, index=False)
    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    result = run_training("classification", data_path, output_dir=tmp_path / "out", mlflow_enabled=True, tracking_uri=tracking_uri, n_splits=2)
    assert db_path.exists()
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("dax-shot-classification")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert any(run.info.run_id == result["parent_run_id"] for run in runs)
    nested = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == result["parent_run_id"]]
    assert nested
    oof = pd.read_parquet(tmp_path / "out/oof/classification_oof.parquet")
    assert oof["mlflow_run_id"].notna().all()
    bundle = next((tmp_path / "out/models/classification").glob("*.joblib"))
    assert bundle.exists()
    assert any(run.data.metrics for run in nested)
    assert any(run.data.params for run in nested)
    if hasattr(client, "search_logged_models"):
        logged_models = client.search_logged_models(experiment_ids=[experiment.experiment_id])
        assert any("b0_constant_model" in getattr(model, "name", "") for model in logged_models)

class FakeCalibratedCV:
    calls = []

    def __init__(self, estimator, method, cv):
        self.estimator = estimator
        self.method = method
        self.cv = cv

    def fit(self, x, y):
        FakeCalibratedCV.calls.append({"method": self.method, "indices": list(x.index), "positives": int(y.sum()), "rows": len(y)})
        self.estimator.fit(x, y)
        return self

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)


def test_platt_and_isotonic_calibration_receive_only_outer_training_rows(monkeypatch):
    import dax.models.calibration as calibration_module

    monkeypatch.setattr(calibration_module, "CalibratedClassifierCV", FakeCalibratedCV)
    x_train = pd.DataFrame({"x": range(12)}, index=list(range(100, 112)))
    y_train = pd.Series([0, 1] * 6, index=x_train.index)
    outer_validation_indices = {200, 201, 202}
    FakeCalibratedCV.calls.clear()
    fit_calibrated_classifier(RecordingEstimator(), x_train, y_train, method="platt")
    fit_calibrated_classifier(RecordingEstimator(), x_train, y_train, method="isotonic")
    assert [call["method"] for call in FakeCalibratedCV.calls] == ["sigmoid", "isotonic"]
    for call in FakeCalibratedCV.calls:
        assert set(call["indices"]) == set(x_train.index)
        assert not outer_validation_indices.intersection(call["indices"])
        assert call["positives"] == 6
