from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_two_part_xg.py"
SPEC = importlib.util.spec_from_file_location("train_two_part_xg_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_aligned_classification_oof_rebuilds_stale_variant_artifact(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "event_id": ["event-1", "event-2"],
            "target_future_xg_10s": [0.0, 0.2],
        }
    )
    output_dir = tmp_path / "outputs"
    validation_dir = output_dir / "validation" / "classification"
    validation_dir.mkdir(parents=True)
    requested_variant = "b6_full_without_360"
    variant_path = validation_dir / f"{requested_variant}_oof_predictions.parquet"
    pd.DataFrame(
        {
            "event_id": ["fixture-1", "fixture-2"],
            "fold": [0, 1],
            "y_score": [0.1, 0.2],
            "model_variant": [requested_variant, requested_variant],
        }
    ).to_parquet(variant_path, index=False)

    calls: list[tuple[object, ...]] = []

    def fake_run_training(task, input_path, config_path, output_dir_arg, mlflow_enabled=None, **kwargs):
        calls.append((task, input_path, config_path, output_dir_arg, mlflow_enabled, kwargs))
        pd.DataFrame(
            {
                "event_id": ["event-1", "event-2"],
                "fold": [0, 1],
                "y_score": [0.3, 0.7],
                "model_variant": [requested_variant, requested_variant],
            }
        ).to_parquet(variant_path, index=False)
        return {"comparison": str(output_dir_arg)}

    monkeypatch.setattr(MODULE, "run_training", fake_run_training)

    aligned_df, aligned_oof, selected_path = MODULE.load_aligned_classification_oof(
        df,
        tmp_path / "input.parquet",
        output_dir / "oof" / "classification_oof.parquet",
        tmp_path / "models.yaml",
        output_dir,
        requested_variant,
        mlflow_enabled=False,
    )

    assert len(calls) == 1
    assert calls[0][0] == "classification"
    assert selected_path == variant_path
    assert aligned_df["event_id"].tolist() == ["event-1", "event-2"]
    assert aligned_oof["event_id"].tolist() == ["event-1", "event-2"]
    assert aligned_oof["y_score"].tolist() == [0.3, 0.7]

