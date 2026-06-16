"""Train DAx baseline regression variants for xT-based target.

Usage:
  python scripts/models/train_baseline_regression.py
  python scripts/models/train_baseline_regression.py --models-dir outputs/models/regression --max-rows 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dax.models.baseline_regression import (  # noqa: E402
    TARGET_COL,
    GROUP_COL,
    build_regression_pipeline,
    coefficient_table,
    default_regression_specs,
    grouped_cv_scores_regression,
    prepare_xyg_regression,
    resolve_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline regression models for DAx (xT-based target)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"),
        help="Input parquet file.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "regression"),
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "regression"),
        help="Directory to save validation artifacts.",
    )
    parser.add_argument(
        "--oof-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "oof" / "regression"),
        help="Directory to save out-of-fold prediction artifacts.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional quick-run row cap"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    models_dir = Path(args.models_dir)
    validation_dir = Path(args.validation_dir)
    oof_dir = Path(args.oof_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    df = pd.read_parquet(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    if TARGET_COL not in df.columns or GROUP_COL not in df.columns:
        raise ValueError(f"Dataset must contain {TARGET_COL!r} and {GROUP_COL!r}")

    print("=" * 72)
    print("TRAIN BASELINE REGRESSION MODELS (xT-BASED TARGET)")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Matches: {df[GROUP_COL].nunique():,}")
    print(f"Target (xT) stats:")
    print(f"  Mean: {df[TARGET_COL].mean():.4f}")
    print(f"  Median: {df[TARGET_COL].median():.4f}")
    print(f"  Std: {df[TARGET_COL].std():.4f}")
    print(f"  Range: [{df[TARGET_COL].min():.4f}, {df[TARGET_COL].max():.4f}]")
    print(f"  Missing: {df[TARGET_COL].isna().sum():,}")

    summary: dict[str, object] = {
        "input_path": str(input_path),
        "rows": int(len(df)),
        "matches": int(df[GROUP_COL].nunique()),
        "target": TARGET_COL,
        "target_mean": float(df[TARGET_COL].mean()),
        "target_std": float(df[TARGET_COL].std()),
        "variants": {},
    }

    all_oof_rows: list[pd.DataFrame] = []

    for spec in default_regression_specs():
        resolved = resolve_columns(df, spec)
        if not resolved.categorical and not resolved.numeric:
            print(f"[SKIP] {resolved.name}: no usable columns found")
            continue

        x, y, groups = prepare_xyg_regression(df, resolved)
        pipe = build_regression_pipeline(resolved)

        cv = grouped_cv_scores_regression(
            x=x, y=y, groups=groups, pipeline=pipe, n_splits=args.n_splits
        )
        pipe.fit(x, y)

        coef_df = coefficient_table(pipe)
        coef_path = validation_dir / f"regression_{resolved.name}_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)

        model_path = models_dir / f"regression_{resolved.name}.joblib"
        payload = {
            "variant": resolved.name,
            "model_type": resolved.model_type,
            "categorical": resolved.categorical,
            "numeric": resolved.numeric,
            "pipeline": pipe,
            "target": TARGET_COL,
            "group": GROUP_COL,
        }
        joblib.dump(payload, model_path)

        fold_df = pd.DataFrame(cv["fold_metrics"])
        fold_path = validation_dir / f"regression_{resolved.name}_fold_metrics.csv"
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
            "mape": float(cv["mape"]),
            "spearman": float(cv["spearman"]),
            "model_path": str(model_path),
            "coefficients_path": str(coef_path),
            "fold_metrics_path": str(fold_path),
        }

        print(
            f"[DONE] {resolved.name}: R²={cv['r2']:.4f}, RMSE={cv['rmse']:.4f}, "
            f"MAE={cv['mae']:.4f}, Spearman={cv['spearman']:.4f}, "
            f"features={len(resolved.categorical) + len(resolved.numeric)}"
        )

    if all_oof_rows:
        oof_df = pd.concat(all_oof_rows, ignore_index=True)
        oof_path = oof_dir / "regression_oof_predictions.parquet"
        oof_df.to_parquet(oof_path, index=False)
        summary["oof_path"] = str(oof_path)

    summary_path = validation_dir / "regression_model_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print(f"Saved summary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()

