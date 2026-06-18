"""Validation helpers for modelling outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def assert_oof_predictions(oof: pd.DataFrame, score_col: str, fold_membership: dict[str, dict[str, list[str]]] | None = None) -> bool:
    """Validate row-level OOF coverage and optional fold isolation metadata."""

    keys = ["event_id", "model_variant"] if "event_id" in oof.columns else ["match_id", "player_id", "model_variant"]
    if oof.duplicated(keys).any():
        raise ValueError("Duplicate OOF event-variant predictions detected.")
    if oof[score_col].isna().any():
        raise ValueError("Missing OOF predictions detected.")
    if fold_membership is not None:
        for fold, metadata in fold_membership.items():
            training = set(map(str, metadata.get("training_matches", [])))
            validation = set(map(str, metadata.get("validation_matches", [])))
            if training.intersection(validation):
                raise ValueError(f"Fold {fold} has a match in both training and validation membership.")
    return True


def validate_outputs(task: str, output_dir: str | Path = "outputs") -> dict[str, int]:
    """Validate canonical OOF output for one task."""

    path = Path(output_dir) / "oof" / ("classification_oof.parquet" if task == "classification" else "regression_oof.parquet")
    if not path.exists():
        raise FileNotFoundError(path)
    oof = pd.read_parquet(path)
    assert_oof_predictions(oof, "y_score" if task == "classification" else "y_pred")
    return {"rows": len(oof), "variants": int(oof["model_variant"].nunique())}
