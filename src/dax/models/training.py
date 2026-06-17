from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dax.models.baseline_logistic import (
    GROUP_COL as LOGISTIC_GROUP_COL,
    TARGET_COL as LOGISTIC_TARGET_COL,
    build_pipeline,
    coefficient_table as logistic_coefficient_table,
    default_variant_specs,
    grouped_cv_scores,
    prepare_xyg,
    resolve_columns as resolve_logistic_columns,
)
from dax.models.baseline_regression import (
    GROUP_COL as REGRESSION_GROUP_COL,
    TARGET_COL as REGRESSION_TARGET_COL,
    build_regression_pipeline,
    coefficient_table as regression_coefficient_table,
    default_regression_specs,
    grouped_cv_scores_regression,
    prepare_xyg_regression,
    resolve_columns as resolve_regression_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supported DAx baseline models.")
    parser.add_argument("--task", choices=("all", "logistic", "regression"), default="all")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input player dataset parquet.")
    parser.add_argument("--models-dir", type=str, default=str(REPO_ROOT / "outputs" / "models"))
    parser.add_argument("--validation-dir", type=str, default=str(REPO_ROOT / "outputs" / "validation"))
    parser.add_argument("--oof-dir", type=str, default=str(REPO_ROOT / "outputs" / "oof"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned training task without running it.")
    return parser.parse_args(argv)


def _effective_splits(groups: pd.Series, requested: int) -> int:
    unique_groups = int(groups.nunique())
    if unique_groups < 2:
        raise ValueError("Training requires at least two unique match_id groups.")
    effective = min(requested, unique_groups)
    if effective != requested:
        print(f"[WARN] Requested n_splits={requested}, using {effective} because only {unique_groups} groups are available.")
    return effective


def train_logistic_models(
    input_path: str | Path,
    *,
    models_dir: str | Path,
    validation_dir: str | Path,
    oof_dir: str | Path,
    n_splits: int = 5,
    max_rows: int | None = None,
) -> Path:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_file}")

    models_path = Path(models_dir)
    validation_path = Path(validation_dir)
    oof_path = Path(oof_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    oof_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_file)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if LOGISTIC_TARGET_COL not in df.columns or LOGISTIC_GROUP_COL not in df.columns:
        raise ValueError(f"Dataset must contain {LOGISTIC_TARGET_COL!r} and {LOGISTIC_GROUP_COL!r}")

    effective_splits = _effective_splits(df[LOGISTIC_GROUP_COL], n_splits)
    print("=" * 72)
    print("TRAIN BASELINE LOGISTIC MODELS")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Matches: {df[LOGISTIC_GROUP_COL].nunique():,}")
    print(f"Target rate: {df[LOGISTIC_TARGET_COL].mean() * 100:.3f}%")

    summary: dict[str, object] = {
        "input_path": str(input_file),
        "rows": int(len(df)),
        "matches": int(df[LOGISTIC_GROUP_COL].nunique()),
        "target_rate": float(df[LOGISTIC_TARGET_COL].mean()),
        "variants": {},
    }
    all_oof_rows: list[pd.DataFrame] = []

    for spec in default_variant_specs():
        resolved = resolve_logistic_columns(df, spec)
        if not resolved.categorical and not resolved.numeric:
            print(f"[SKIP] {resolved.name}: no usable columns found")
            continue

        x, y, groups = prepare_xyg(df, resolved)
        pipeline = build_pipeline(resolved)
        cv = grouped_cv_scores(x=x, y=y, groups=groups, pipeline=pipeline, n_splits=effective_splits)
        pipeline.fit(x, y)

        coefficients = logistic_coefficient_table(pipeline)
        coefficients_path = validation_path / f"baseline_{resolved.name}_coefficients.csv"
        coefficients.to_csv(coefficients_path, index=False)

        model_path = models_path / f"logistic_{resolved.name}.joblib"
        joblib.dump(
            {
                "variant": resolved.name,
                "categorical": resolved.categorical,
                "numeric": resolved.numeric,
                "pipeline": pipeline,
                "target": LOGISTIC_TARGET_COL,
                "group": LOGISTIC_GROUP_COL,
            },
            model_path,
        )

        fold_df = pd.DataFrame(cv["fold_metrics"])
        fold_path = validation_path / f"baseline_{resolved.name}_fold_metrics.csv"
        fold_df.to_csv(fold_path, index=False)

        oof = pd.DataFrame(
            {
                "variant": resolved.name,
                "y_true": y.astype(int),
                "y_score": np.asarray(cv["oof_predictions"], dtype=float),
                "match_id": groups,
            }
        )
        all_oof_rows.append(oof)

        summary["variants"][resolved.name] = {
            "categorical": resolved.categorical,
            "numeric": resolved.numeric,
            "roc_auc": float(cv["roc_auc"]),
            "avg_precision": float(cv["avg_precision"]),
            "model_path": str(model_path),
            "coefficients_path": str(coefficients_path),
            "fold_metrics_path": str(fold_path),
        }
        print(
            f"[DONE] {resolved.name}: AUC={cv['roc_auc']:.4f}, AP={cv['avg_precision']:.4f}, "
            f"features={len(resolved.categorical) + len(resolved.numeric)}"
        )

    if all_oof_rows:
        all_oof = pd.concat(all_oof_rows, ignore_index=True)
        logistic_oof = oof_path / "baseline_oof_predictions.parquet"
        all_oof.to_parquet(logistic_oof, index=False)
        summary["oof_path"] = str(logistic_oof)

    summary_path = validation_path / "baseline_model_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"Saved summary: {summary_path}")
    print("Done.")
    return summary_path


def train_regression_models(
    input_path: str | Path,
    *,
    models_dir: str | Path,
    validation_dir: str | Path,
    oof_dir: str | Path,
    n_splits: int = 5,
    max_rows: int | None = None,
) -> Path:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_file}")

    models_path = Path(models_dir)
    validation_path = Path(validation_dir)
    oof_path = Path(oof_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    oof_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_file)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if REGRESSION_TARGET_COL not in df.columns or REGRESSION_GROUP_COL not in df.columns:
        raise ValueError(f"Dataset must contain {REGRESSION_TARGET_COL!r} and {REGRESSION_GROUP_COL!r}")

    effective_splits = _effective_splits(df[REGRESSION_GROUP_COL], n_splits)
    print("=" * 72)
    print("TRAIN BASELINE REGRESSION MODELS (FUTURE-XG TARGET)")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Matches: {df[REGRESSION_GROUP_COL].nunique():,}")
    print(f"Target mean: {df[REGRESSION_TARGET_COL].mean():.4f}")

    summary: dict[str, object] = {
        "input_path": str(input_file),
        "rows": int(len(df)),
        "matches": int(df[REGRESSION_GROUP_COL].nunique()),
        "target": REGRESSION_TARGET_COL,
        "target_mean": float(df[REGRESSION_TARGET_COL].mean()),
        "target_std": float(df[REGRESSION_TARGET_COL].std()),
        "variants": {},
    }
    all_oof_rows: list[pd.DataFrame] = []

    for spec in default_regression_specs():
        resolved = resolve_regression_columns(df, spec)
        if not resolved.categorical and not resolved.numeric:
            print(f"[SKIP] {resolved.name}: no usable columns found")
            continue

        x, y, groups = prepare_xyg_regression(df, resolved)
        pipeline = build_regression_pipeline(resolved)
        cv = grouped_cv_scores_regression(x=x, y=y, groups=groups, pipeline=pipeline, n_splits=effective_splits)
        pipeline.fit(x, y)

        coefficients = regression_coefficient_table(pipeline)
        coefficients_path = validation_path / f"regression_{resolved.name}_coefficients.csv"
        coefficients.to_csv(coefficients_path, index=False)

        model_path = models_path / f"regression_{resolved.name}.joblib"
        joblib.dump(
            {
                "variant": resolved.name,
                "model_type": resolved.model_type,
                "categorical": resolved.categorical,
                "numeric": resolved.numeric,
                "pipeline": pipeline,
                "target": REGRESSION_TARGET_COL,
                "group": REGRESSION_GROUP_COL,
            },
            model_path,
        )

        fold_df = pd.DataFrame(cv["fold_metrics"])
        fold_path = validation_path / f"regression_{resolved.name}_fold_metrics.csv"
        fold_df.to_csv(fold_path, index=False)

        oof = pd.DataFrame(
            {
                "variant": resolved.name,
                "y_true": y.astype(float),
                "y_pred": np.asarray(cv["oof_predictions"], dtype=float),
                "match_id": groups,
            }
        )
        all_oof_rows.append(oof)

        summary["variants"][resolved.name] = {
            "model_type": resolved.model_type,
            "categorical": resolved.categorical,
            "numeric": resolved.numeric,
            "r2": float(cv["r2"]),
            "rmse": float(cv["rmse"]),
            "mae": float(cv["mae"]),
            "spearman": float(cv["spearman"]),
            "model_path": str(model_path),
            "coefficients_path": str(coefficients_path),
            "fold_metrics_path": str(fold_path),
        }
        print(
            f"[DONE] {resolved.name}: R²={cv['r2']:.4f}, RMSE={cv['rmse']:.4f}, "
            f"MAE={cv['mae']:.4f}, Spearman={cv['spearman']:.4f}, "
            f"features={len(resolved.categorical) + len(resolved.numeric)}"
        )

    if all_oof_rows:
        all_oof = pd.concat(all_oof_rows, ignore_index=True)
        regression_oof = oof_path / "regression_oof_predictions.parquet"
        all_oof.to_parquet(regression_oof, index=False)
        summary["oof_path"] = str(regression_oof)

    summary_path = validation_path / "regression_model_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"Saved summary: {summary_path}")
    print("Done.")
    return summary_path


def _task_paths(root: str | Path, leaf: str) -> Path:
    return Path(root) / leaf


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"[dry-run] train task={args.task} input={args.input}")
        return 0

    if args.task in {"all", "logistic"}:
        train_logistic_models(
            args.input,
            models_dir=_task_paths(args.models_dir, "baseline"),
            validation_dir=_task_paths(args.validation_dir, "baseline"),
            oof_dir=_task_paths(args.oof_dir, "baseline"),
            n_splits=args.n_splits,
            max_rows=args.max_rows,
        )
    if args.task in {"all", "regression"}:
        train_regression_models(
            args.input,
            models_dir=_task_paths(args.models_dir, "regression"),
            validation_dir=_task_paths(args.validation_dir, "regression"),
            oof_dir=_task_paths(args.oof_dir, "regression"),
            n_splits=args.n_splits,
            max_rows=args.max_rows,
        )
    return 0
