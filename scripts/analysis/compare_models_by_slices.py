"""Compare trained classification/regression variants overall and by contextual slices.

Outputs:
- overall leaderboard for logistic and regression families
- slice-level metrics per variant
- best variant per slice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dax.models.baseline_logistic import TARGET_COL as LOG_TARGET_COL  # noqa: E402
from dax.models.baseline_regression import TARGET_COL as REG_TARGET_COL  # noqa: E402


DEFAULT_SLICES = [
    "phase_label",
    "action_zone",
    "position_group",
    "play_pattern",
    "action_family",
    "counterpress",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained models by slices")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"),
        help="Input dataset path.",
    )
    parser.add_argument(
        "--baseline-models-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "baseline"),
        help="Directory with logistic_*.joblib files.",
    )
    parser.add_argument(
        "--regression-models-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "regression"),
        help="Directory with regression_*.joblib files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "comparison" / "slices"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--slice-cols",
        type=str,
        nargs="*",
        default=DEFAULT_SLICES,
        help="Slice columns to evaluate.",
    )
    parser.add_argument(
        "--min-slice-size",
        type=int,
        default=200,
        help="Minimum rows required for a slice segment.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick runs.",
    )
    return parser.parse_args()


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    corr = spearmanr(y_true, y_pred).correlation
    return float(corr) if corr is not None and not np.isnan(corr) else np.nan


def _variant_from_filename(name: str, prefix: str) -> str:
    stem = Path(name).stem
    return stem[len(prefix) + 1 :]


def _predict_logistic(df: pd.DataFrame, model_payload: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = [c for c in model_payload["categorical"] + model_payload["numeric"] if c in df.columns]
    x = df[cols].copy()
    y = (pd.to_numeric(df[LOG_TARGET_COL], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    mask = ~x.isna().all(axis=1)
    x_valid = x.loc[mask]
    y_valid = y[mask.to_numpy()]
    y_score = model_payload["pipeline"].predict_proba(x_valid)[:, 1]
    return df.loc[mask].copy(), y_valid, np.asarray(y_score, dtype=float)


def _predict_regression(df: pd.DataFrame, model_payload: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = [c for c in model_payload["categorical"] + model_payload["numeric"] if c in df.columns]
    x = df[cols].copy()
    y = pd.to_numeric(df[REG_TARGET_COL], errors="coerce").fillna(0).to_numpy()
    mask = ~x.isna().all(axis=1)
    x_valid = x.loc[mask]
    y_valid = y[mask.to_numpy()]
    y_pred = model_payload["pipeline"].predict(x_valid)
    return df.loc[mask].copy(), y_valid, np.asarray(y_pred, dtype=float)


def _overall_metrics_logistic(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "avg_precision": _safe_avg_precision(y_true, y_score),
    }


def _overall_metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def _slice_rows_logistic(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    variant: str,
    family: str,
    slice_cols: list[str],
    min_slice_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in slice_cols:
        if col not in frame.columns:
            continue
        for value, idx in frame.groupby(col).groups.items():
            seg_idx = np.asarray(list(idx), dtype=int)
            seg_size = len(seg_idx)
            if seg_size < min_slice_size:
                continue
            yt = y_true[frame.index.get_indexer(seg_idx)]
            ys = y_score[frame.index.get_indexer(seg_idx)]
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "slice_col": col,
                    "slice_value": str(value),
                    "n": seg_size,
                    "roc_auc": _safe_roc_auc(yt, ys),
                    "avg_precision": _safe_avg_precision(yt, ys),
                }
            )
    return rows


def _slice_rows_regression(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variant: str,
    family: str,
    slice_cols: list[str],
    min_slice_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in slice_cols:
        if col not in frame.columns:
            continue
        for value, idx in frame.groupby(col).groups.items():
            seg_idx = np.asarray(list(idx), dtype=int)
            seg_size = len(seg_idx)
            if seg_size < min_slice_size:
                continue
            yt = y_true[frame.index.get_indexer(seg_idx)]
            yp = y_pred[frame.index.get_indexer(seg_idx)]
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "slice_col": col,
                    "slice_value": str(value),
                    "n": seg_size,
                    "r2": float(r2_score(yt, yp)),
                    "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                    "mae": float(mean_absolute_error(yt, yp)),
                    "spearman": _safe_spearman(yt, yp),
                }
            )
    return rows


def _best_by_slice(slice_df: pd.DataFrame, family: str) -> pd.DataFrame:
    if family == "logistic":
        metric = "roc_auc"
        work = slice_df.dropna(subset=[metric]).copy()
        if work.empty:
            return work
        best_idx = work.groupby(["slice_col", "slice_value"])[metric].idxmax()
        return work.loc[best_idx].sort_values(["slice_col", metric], ascending=[True, False]).reset_index(drop=True)

    metric = "r2"
    work = slice_df.dropna(subset=[metric]).copy()
    if work.empty:
        return work
    best_idx = work.groupby(["slice_col", "slice_value"])[metric].idxmax()
    return work.loc[best_idx].sort_values(["slice_col", metric], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    baseline_dir = Path(args.baseline_models_dir)
    regression_dir = Path(args.regression_models_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    df = pd.read_parquet(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    # --- Logistic family ---
    logistic_files = sorted(baseline_dir.glob("logistic_*.joblib"))
    logistic_overall: list[dict[str, Any]] = []
    logistic_slice_rows: list[dict[str, Any]] = []

    for path in logistic_files:
        variant = _variant_from_filename(path.name, "logistic")
        payload = joblib.load(path)
        frame, y_true, y_score = _predict_logistic(df, payload)
        metrics = _overall_metrics_logistic(y_true, y_score)
        logistic_overall.append(
            {
                "family": "logistic",
                "variant": variant,
                "n": int(len(y_true)),
                **metrics,
            }
        )
        logistic_slice_rows.extend(
            _slice_rows_logistic(
                frame=frame,
                y_true=y_true,
                y_score=y_score,
                variant=variant,
                family="logistic",
                slice_cols=args.slice_cols,
                min_slice_size=args.min_slice_size,
            )
        )

    logistic_overall_df = pd.DataFrame(logistic_overall).sort_values("roc_auc", ascending=False)
    logistic_slices_df = pd.DataFrame(logistic_slice_rows)
    logistic_best_slice_df = _best_by_slice(logistic_slices_df, family="logistic")

    # --- Regression family ---
    regression_files = sorted(regression_dir.glob("regression_*.joblib"))
    regression_overall: list[dict[str, Any]] = []
    regression_slice_rows: list[dict[str, Any]] = []

    for path in regression_files:
        variant = _variant_from_filename(path.name, "regression")
        payload = joblib.load(path)
        frame, y_true, y_pred = _predict_regression(df, payload)
        metrics = _overall_metrics_regression(y_true, y_pred)
        regression_overall.append(
            {
                "family": "regression",
                "variant": variant,
                "n": int(len(y_true)),
                **metrics,
            }
        )
        regression_slice_rows.extend(
            _slice_rows_regression(
                frame=frame,
                y_true=y_true,
                y_pred=y_pred,
                variant=variant,
                family="regression",
                slice_cols=args.slice_cols,
                min_slice_size=args.min_slice_size,
            )
        )

    regression_overall_df = pd.DataFrame(regression_overall).sort_values("r2", ascending=False)
    regression_slices_df = pd.DataFrame(regression_slice_rows)
    regression_best_slice_df = _best_by_slice(regression_slices_df, family="regression")

    # Save outputs
    logistic_overall_path = out_dir / "leaderboard_logistic_overall.csv"
    regression_overall_path = out_dir / "leaderboard_regression_overall.csv"
    logistic_slices_path = out_dir / "slice_metrics_logistic.csv"
    regression_slices_path = out_dir / "slice_metrics_regression.csv"
    logistic_best_path = out_dir / "best_variant_by_slice_logistic.csv"
    regression_best_path = out_dir / "best_variant_by_slice_regression.csv"

    logistic_overall_df.to_csv(logistic_overall_path, index=False)
    regression_overall_df.to_csv(regression_overall_path, index=False)
    logistic_slices_df.to_csv(logistic_slices_path, index=False)
    regression_slices_df.to_csv(regression_slices_path, index=False)
    logistic_best_slice_df.to_csv(logistic_best_path, index=False)
    regression_best_slice_df.to_csv(regression_best_path, index=False)

    summary = {
        "rows": int(len(df)),
        "slice_cols": args.slice_cols,
        "min_slice_size": int(args.min_slice_size),
        "top_logistic_variant": logistic_overall_df.iloc[0]["variant"] if not logistic_overall_df.empty else "",
        "top_regression_variant": regression_overall_df.iloc[0]["variant"] if not regression_overall_df.empty else "",
        "outputs": {
            "leaderboard_logistic_overall": str(logistic_overall_path),
            "leaderboard_regression_overall": str(regression_overall_path),
            "slice_metrics_logistic": str(logistic_slices_path),
            "slice_metrics_regression": str(regression_slices_path),
            "best_variant_by_slice_logistic": str(logistic_best_path),
            "best_variant_by_slice_regression": str(regression_best_path),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("MODEL COMPARISON BY SLICES")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    if not logistic_overall_df.empty:
        top = logistic_overall_df.iloc[0]
        print(f"Top logistic: {top['variant']} (AUC={top['roc_auc']:.4f}, AP={top['avg_precision']:.4f})")
    if not regression_overall_df.empty:
        top = regression_overall_df.iloc[0]
        print(f"Top regression: {top['variant']} (R2={top['r2']:.4f}, Spearman={top['spearman']:.4f})")
    print(f"Saved: {logistic_overall_path}")
    print(f"Saved: {regression_overall_path}")
    print(f"Saved: {logistic_slices_path}")
    print(f"Saved: {regression_slices_path}")
    print(f"Saved: {logistic_best_path}")
    print(f"Saved: {regression_best_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()


