"""
CLI for training and evaluating two-part hurdle future-xG models.

This workflow evaluates both b7_full_with_360 and b6_full_without_360 with
identical conditional candidates and identical fold assignments on shared rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from dax.models.feature_contracts import get_contracts, resolve_contract
from dax.models.mlflow_tracking import (
    configure_mlflow,
    log_artifact,
    log_json_artifact,
    log_metrics,
    log_params,
    log_sklearn_model,
    start_parent_run,
    start_variant_run,
)
from dax.models.schemas import dataset_fingerprint
from dax.models.training import git_sha, run_training
from dax.models.two_part_xg import (
    ConditionalModelSpec,
    build_combined_oof,
    compute_conditional_metrics,
    compute_hurdle_metrics,
    cumulative_capture,
    decile_ranking,
    filter_classification_oof,
    fit_final_conditional_model,
    fold_metric_summary,
    load_classification_oof,
    predict_conditional,
    safe_numeric,
    train_conditional_variant,
)

matplotlib.use("Agg", force=True)

DEFAULT_CLASSIFICATION_VARIANTS = ["b7_full_with_360", "b6_full_without_360"]


def _compute_match_level_suppression_ci(
    oof: pd.DataFrame,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute per-player bootstrap CI for combined_xg_suppression.

    Returns a DataFrame with player_id, team, ci95_lower, ci95_upper.
    """
    group_cols = [c for c in ["player_id", "team"] if c in oof.columns]
    if not group_cols:
        return pd.DataFrame(columns=["ci95_lower", "ci95_upper"])
    oof = oof.copy()
    oof["supp_event"] = oof["combined_future_xg_prediction"] - oof["observed_future_xg"]
    by_match = (
        oof.groupby(group_cols + ["match_id"], dropna=False)
        .agg(total_supp=("supp_event", "sum"))
        .reset_index()
    )
    rng = np.random.RandomState(seed)
    rows: list[dict] = []
    for keys, group in by_match.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        vals = group["total_supp"].to_numpy(dtype=float)
        n = len(vals)
        samples = np.array([vals[rng.choice(n, size=n, replace=True)].sum() for _ in range(n_bootstrap)])
        rows.append({
            **key_map,
            "ci95_lower": float(np.percentile(samples, 2.5)),
            "ci95_upper": float(np.percentile(samples, 97.5)),
        })
    return pd.DataFrame(rows)
DEFAULT_REGRESSION_CONTRACT = "r4_full_with_360"
BENCHMARK_VARIANTS = ["r0_constant", "r4_full_with_360", "r6_nonlinear_candidate"]


def build_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "oof": root / "oof",
        "reports": root / "models" / "reports",
        "bundles": root / "models" / "two_part_xg",
        "charts": root / "models" / "two_part_xg" / "charts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def conditional_specs() -> list[ConditionalModelSpec]:
    return [
        ConditionalModelSpec("conditional_mean_baseline", "conditional_mean", {}, reference_status="reference baseline"),
        ConditionalModelSpec("conditional_log_ridge", "log_ridge", {"alpha": 1.0}),
        ConditionalModelSpec("conditional_gamma", "gamma", {"alpha": 0.0, "max_iter": 500}),
        ConditionalModelSpec("conditional_tweedie", "tweedie", {"power": 1.5, "alpha": 0.0, "max_iter": 500}),
        ConditionalModelSpec(
            "conditional_hgb",
            "hist_gradient_boosting_regressor",
            {"max_iter": 150, "learning_rate": 0.05, "max_depth": None},
        ),
    ]


def candidate_classification_oof_paths(oof_path: str | Path, output_dir: str | Path, classification_variant: str) -> list[Path]:
    primary = Path(oof_path)
    output_root = Path(output_dir)
    candidates = [
        primary,
        output_root / "validation" / "classification" / f"{classification_variant}_oof_predictions.parquet",
    ]
    if len(primary.parents) >= 2:
        candidates.append(primary.parents[1] / "validation" / "classification" / f"{classification_variant}_oof_predictions.parquet")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_classification_variant(
    df: pd.DataFrame,
    oof_path: str | Path,
    output_dir: str | Path,
    classification_variant: str,
) -> tuple[pd.DataFrame, Path, list[str]]:
    input_events = set(df["event_id"].unique())
    best: tuple[int, int, pd.DataFrame, Path] | None = None
    checked: list[str] = []

    for candidate in candidate_classification_oof_paths(oof_path, output_dir, classification_variant):
        if not candidate.exists():
            checked.append(f"{candidate} [missing]")
            continue
        try:
            frame = filter_classification_oof(load_classification_oof(candidate), classification_variant)
        except Exception as exc:  # noqa: BLE001
            checked.append(f"{candidate} [invalid: {exc}]")
            continue

        event_ids = set(frame["event_id"].unique())
        overlap = len(input_events & event_ids)
        extraneous = len(event_ids - input_events)
        checked.append(f"{candidate} [rows={len(frame)}, overlap={overlap}, extraneous={extraneous}]")
        score = (overlap, -extraneous)
        if best is None or score > (best[0], best[1]):
            best = (overlap, -extraneous, frame, candidate)

    if best is None or best[0] == 0:
        raise ValueError(
            f"Could not find aligned classification OOF for {classification_variant!r}.\n"
            f"Checked:\n  - "
            + "\n  - ".join(checked)
        )

    selected = best[2]
    if selected["event_id"].duplicated().any():
        raise ValueError(f"Classification variant {classification_variant!r} has duplicate event IDs.")
    overlap_df = df.loc[df["event_id"].isin(selected["event_id"])].copy().reset_index(drop=True)
    selected = selected.set_index("event_id").loc[overlap_df["event_id"]].reset_index()
    return selected, best[3], checked


def load_aligned_classification_oof(
    df: pd.DataFrame,
    input_path: str | Path,
    oof_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    classification_variant: str,
    *,
    mlflow_enabled: bool,
    allow_rebuild: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Backward-compatible loader for one classification variant alignment."""

    try:
        selected, selected_path, _ = resolve_classification_variant(df, oof_path, output_dir, classification_variant)
    except ValueError:
        if not allow_rebuild:
            raise
        run_training("classification", input_path, config_path, output_dir, mlflow_enabled=mlflow_enabled)
        selected, selected_path, _ = resolve_classification_variant(df, oof_path, output_dir, classification_variant)

    aligned_df = df.loc[df["event_id"].isin(selected["event_id"])].copy().reset_index(drop=True)
    aligned_oof = selected.set_index("event_id").loc[aligned_df["event_id"]].reset_index()
    return aligned_df, aligned_oof, selected_path


def load_regression_variant_oof(regression_oof_path: str | Path, output_dir: str | Path, variant: str) -> pd.DataFrame:
    candidates = [
        Path(regression_oof_path),
        Path(output_dir) / "validation" / "regression" / f"{variant}_oof_predictions.parquet",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        frame = pd.read_parquet(candidate)
        if "model_variant" in frame.columns:
            frame = frame.loc[frame["model_variant"].eq(variant)].copy()
        if not frame.empty and {"event_id", "y_true", "y_pred"}.issubset(frame.columns):
            return frame.reset_index(drop=True)
    raise ValueError(f"Could not load regression OOF for benchmark {variant!r}. Tried: {candidates}")


def make_common_row_metrics(oof: pd.DataFrame) -> dict[str, float]:
    hurdle = compute_hurdle_metrics(oof)
    deciles = decile_ranking(oof)
    fold_stats = fold_metric_summary(oof)
    return {
        **hurdle,
        "top_10_pct_xg_capture": cumulative_capture(deciles, 0.10),
        "top_20_pct_xg_capture": cumulative_capture(deciles, 0.20),
        "top_30_pct_xg_capture": cumulative_capture(deciles, 0.30),
        **fold_stats,
    }


def selection_with_reasons(comparison: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    frame = comparison.copy()
    frame["status"] = "candidate"
    frame["selection_reason"] = "meets execution requirements"

    frame.loc[frame["conditional_model"].eq("conditional_mean_baseline"), "status"] = "reference baseline"
    frame.loc[frame["conditional_model"].eq("conditional_mean_baseline"), "selection_reason"] = "constant conditional reference"

    lower_better = ["mae", "rmse", "nonzero_mae", "nonzero_rmse", "fold_std", "prediction_bias_abs"]
    higher_better = ["r2", "spearman", "nonzero_spearman", "top_10_pct_xg_capture", "top_20_pct_xg_capture", "top_30_pct_xg_capture"]

    frame["prediction_bias_abs"] = frame["prediction_bias"].abs()
    for column in lower_better:
        frame[f"rank_{column}"] = frame[column].rank(method="average", ascending=True)
    for column in higher_better:
        frame[f"rank_{column}"] = frame[column].rank(method="average", ascending=False)

    rank_cols = [column for column in frame.columns if column.startswith("rank_")]
    frame["selection_score"] = frame[rank_cols].mean(axis=1)

    for classification_variant in sorted(frame["classification_variant"].unique().tolist()):
        subset_idx = frame.index[frame["classification_variant"].eq(classification_variant) & frame["status"].eq("candidate")]
        if len(subset_idx) == 0:
            continue
        ordered = frame.loc[subset_idx].sort_values("selection_score")
        preferred_idx = ordered.index[0]

        r4_rows = (
            benchmark.loc[
                benchmark["classification_variant"].eq(classification_variant)
                & benchmark["conditional_model"].eq(frame.loc[preferred_idx, "conditional_model"])
                & benchmark["benchmark_variant"].eq("r4_full_with_360")
            ]
            if not benchmark.empty and "classification_variant" in benchmark.columns
            else pd.DataFrame()
        )
        if not r4_rows.empty:
            r4_row = r4_rows.iloc[0]
            better_or_equal = bool(
                safe_numeric(r4_row.get("candidate_mae")) <= safe_numeric(r4_row.get("benchmark_mae"))
                and safe_numeric(r4_row.get("candidate_rmse")) <= safe_numeric(r4_row.get("benchmark_rmse"))
                and safe_numeric(r4_row.get("candidate_nonzero_mae")) <= safe_numeric(r4_row.get("benchmark_nonzero_mae"))
            )
            if not better_or_equal:
                frame.loc[preferred_idx, "status"] = "insufficient evidence"
                frame.loc[preferred_idx, "selection_reason"] = "weaker than r4 on shared rows for core error metrics"
            else:
                frame.loc[preferred_idx, "status"] = "preferred candidate"
                frame.loc[preferred_idx, "selection_reason"] = "best multi-metric score and not weaker than r4"
        else:
            frame.loc[preferred_idx, "status"] = "insufficient evidence"
            frame.loc[preferred_idx, "selection_reason"] = "missing r4 shared-row benchmark"

        backup_candidates = [idx for idx in ordered.index.tolist()[1:] if frame.loc[idx, "status"] == "candidate"]
        if backup_candidates:
            frame.loc[backup_candidates[0], "status"] = "backup candidate"
            frame.loc[backup_candidates[0], "selection_reason"] = "second-best multi-metric score"

        rejected = [idx for idx in subset_idx if frame.loc[idx, "status"] == "candidate"]
        frame.loc[rejected, "status"] = "rejected"
        frame.loc[rejected, "selection_reason"] = "lower multi-metric score than preferred/backup"

    return frame


def save_diagnostics(oof: pd.DataFrame, fold_metrics: pd.DataFrame, chart_dir: Path, comparison_rows: pd.DataFrame) -> list[Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    shot_rows = oof.loc[oof["observed_future_xg"].gt(0)].copy()
    if not shot_rows.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(shot_rows["conditional_xg_prediction"], shot_rows["observed_future_xg"], alpha=0.4)
        ax.set_xlabel("Predicted conditional xG")
        ax.set_ylabel("Observed xG on shot rows")
        ax.set_title("Conditional predicted vs observed (shot rows)")
        path = chart_dir / "conditional_predicted_vs_observed.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        residual = shot_rows["conditional_xg_prediction"] - shot_rows["observed_future_xg"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(residual, bins=30)
        ax.set_title("Conditional residual distribution (shot rows)")
        path = chart_dir / "conditional_residual_distribution.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(shot_rows["conditional_xg_prediction"], residual, alpha=0.4)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Conditional residual vs prediction")
        path = chart_dir / "conditional_residual_vs_prediction.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        cond_deciles = shot_rows[["conditional_xg_prediction", "observed_future_xg"]].copy()
        cond_deciles["decile"] = pd.qcut(cond_deciles["conditional_xg_prediction"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
        cond_table = cond_deciles.groupby("decile", dropna=False)["observed_future_xg"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cond_table["decile"], cond_table["observed_future_xg"], marker="o")
        ax.set_title("Conditional prediction deciles (shot rows)")
        path = chart_dir / "conditional_prediction_deciles.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Combined charts
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(oof["combined_future_xg_prediction"], oof["observed_future_xg"], alpha=0.4)
    ax.set_title("Combined predicted vs observed")
    path = chart_dir / "combined_predicted_vs_observed.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    residual_all = oof["combined_future_xg_prediction"] - oof["observed_future_xg"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual_all, bins=30)
    ax.set_title("Combined residual distribution")
    path = chart_dir / "combined_residual_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(oof["combined_future_xg_prediction"], residual_all, alpha=0.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Combined residual vs prediction")
    path = chart_dir / "combined_residual_vs_prediction.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    deciles = decile_ranking(oof)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(deciles["decile"], deciles["mean_observed"])
    ax.set_title("Observed xG by prediction decile")
    path = chart_dir / "combined_observed_xg_by_decile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    ordered = deciles.sort_values("decile", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ordered["cumulative_pct_rows"], ordered["cumulative_pct_xg"], marker="o")
    ax.set_xlabel("Top rows (%)")
    ax.set_ylabel("Cumulative observed xG capture (%)")
    ax.set_title("Cumulative xG capture curve")
    path = chart_dir / "combined_cumulative_xg_capture.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    zero_rows = oof.loc[oof["observed_future_xg"].eq(0), "combined_future_xg_prediction"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(zero_rows, bins=30)
    ax.set_title("Zero-row prediction distribution")
    path = chart_dir / "combined_zero_row_prediction_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    nonzero_rows = oof.loc[oof["observed_future_xg"].gt(0), "combined_future_xg_prediction"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nonzero_rows, bins=30)
    ax.set_title("Non-zero-row prediction distribution")
    path = chart_dir / "combined_nonzero_row_prediction_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    if not fold_metrics.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fold_metrics["fold"], fold_metrics["conditional_mae"], marker="o")
        ax.set_title("Conditional fold metric chart (MAE)")
        path = chart_dir / "fold_metric_chart.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not comparison_rows.empty:
        plot_rows = comparison_rows.loc[comparison_rows["benchmark_variant"].isin(["r4_full_with_360", "r6_nonlinear_candidate"])].copy()
        if not plot_rows.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(plot_rows["benchmark_variant"], plot_rows["candidate_mae"] - plot_rows["benchmark_mae"])
            ax.axhline(0, color="black", linewidth=1)
            ax.set_ylabel("MAE delta (candidate - benchmark)")
            ax.set_title("Comparison with r4 and r6")
            path = chart_dir / "comparison_with_r4_r6.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-part hurdle future-xG models")
    parser.add_argument("--input", required=True)
    parser.add_argument("--classification-oof", required=True)
    parser.add_argument("--regression-oof", default="outputs/oof/regression_oof.parquet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classification-variants", nargs="+", default=DEFAULT_CLASSIFICATION_VARIANTS)
    parser.add_argument("--rebuild-classification-oof", action="store_true")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.input)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dirs = build_output_dirs(args.output_dir)
    fingerprint = dataset_fingerprint(args.input, df)

    mlflow = None
    if not args.disable_mlflow:
        mlflow = configure_mlflow(config.get("mlflow", {}))

    classification_variants = list(dict.fromkeys(args.classification_variants))
    missing_variants = [variant for variant in DEFAULT_CLASSIFICATION_VARIANTS if variant not in classification_variants]
    if missing_variants:
        print(f"Warning: expected variants missing from run request: {missing_variants}")

    classification_frames: dict[str, pd.DataFrame] = {}
    classification_sources: dict[str, Path] = {}
    checked_messages: dict[str, list[str]] = {}

    rebuilt = False
    for variant in classification_variants:
        try:
            frame, source, checked = resolve_classification_variant(df, args.classification_oof, args.output_dir, variant)
        except ValueError as exc:
            if not args.rebuild_classification_oof or rebuilt:
                rebuild_cmd = (
                    f"python scripts/train_models.py --task classification --input {args.input} "
                    f"--config {args.config} --output-dir {args.output_dir}"
                )
                raise ValueError(
                    f"{exc}\n\nClassification OOF artifacts are stale or missing. "
                    f"Run:\n{rebuild_cmd}\n"
                    "or rerun this command with --rebuild-classification-oof."
                ) from exc
            print("Rebuilding classification OOF artifacts because --rebuild-classification-oof was set...")
            run_training("classification", args.input, args.config, args.output_dir, mlflow_enabled=not args.disable_mlflow)
            rebuilt = True
            frame, source, checked = resolve_classification_variant(df, args.classification_oof, args.output_dir, variant)
        classification_frames[variant] = frame
        classification_sources[variant] = source
        checked_messages[variant] = checked

    fold_anchor = "b6_full_without_360" if "b6_full_without_360" in classification_frames else classification_variants[0]
    fold_map = classification_frames[fold_anchor].set_index("event_id")["fold"]

    regression_contracts = {contract.name: contract for contract in get_contracts(config, "regression")}
    if DEFAULT_REGRESSION_CONTRACT not in regression_contracts:
        raise ValueError(f"Regression contract {DEFAULT_REGRESSION_CONTRACT!r} missing from config.")

    all_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    variant_outputs: dict[tuple[str, str], pd.DataFrame] = {}

    run_name = f"two-part-future-xg-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with start_parent_run(mlflow, "dax-two-part-future-xg", run_name):
        for classification_variant in classification_variants:
            class_oof = classification_frames[classification_variant]
            aligned_df = df.loc[df["event_id"].isin(class_oof["event_id"])].copy().reset_index(drop=True)
            class_oof = class_oof.set_index("event_id").loc[aligned_df["event_id"]].reset_index()

            aligned_df["shared_fold"] = aligned_df["event_id"].map(fold_map)
            if aligned_df["shared_fold"].isna().any():
                raise ValueError(f"Could not assign shared folds for {classification_variant}.")
            folds = aligned_df["shared_fold"].astype(int)

            resolved = resolve_contract(aligned_df, regression_contracts[DEFAULT_REGRESSION_CONTRACT])
            for spec in conditional_specs():
                run_label = f"{classification_variant}__{spec.name}"
                with start_variant_run(mlflow, run_label) as variant_run:
                    run_id = getattr(getattr(variant_run, "info", None), "run_id", None)
                    try:
                        conditional_prediction, fold_metrics = train_conditional_variant(
                            aligned_df,
                            folds,
                            spec,
                            resolved,
                            target_col="target_future_xg_10s",
                        )
                    except Exception as exc:  # noqa: BLE001
                        status = "skipped" if spec.model_family in {"gamma", "tweedie"} else "failed"
                        reason = str(exc)
                        row = {
                            "classification_variant": classification_variant,
                            "conditional_model": spec.name,
                            "status": status,
                            "selection_reason": reason,
                        }
                        all_rows.append(row)
                        log_params(mlflow, {"classification_variant": classification_variant, "conditional_variant": spec.name, "status": status})
                        log_json_artifact(mlflow, {"error": reason}, dirs["reports"] / f"{run_label}_error.json")
                        continue

                    combined_oof = build_combined_oof(
                        aligned_df,
                        class_oof,
                        conditional_prediction,
                        folds,
                        conditional_variant=spec.name,
                        classification_variant=classification_variant,
                        run_id=run_id,
                    )
                    variant_outputs[(classification_variant, spec.name)] = combined_oof

                    conditional_eval = compute_conditional_metrics(
                        combined_oof.loc[combined_oof["observed_future_xg"].gt(0), "observed_future_xg"].to_numpy(dtype=float),
                        combined_oof.loc[combined_oof["observed_future_xg"].gt(0), "conditional_xg_prediction"].to_numpy(dtype=float),
                    )
                    hurdle_eval = make_common_row_metrics(combined_oof)
                    deciles = decile_ranking(combined_oof)

                    fold_stats = fold_metric_summary(combined_oof)
                    row = {
                        "classification_variant": classification_variant,
                        "conditional_model": spec.name,
                        "model_family": spec.model_family,
                        "status": "candidate",
                        "rows": int(len(combined_oof)),
                        "matches": int(combined_oof["match_id"].nunique()),
                        "mae": hurdle_eval["mae"],
                        "rmse": hurdle_eval["rmse"],
                        "r2": hurdle_eval["r2"],
                        "spearman": hurdle_eval["spearman"],
                        "nonzero_mae": hurdle_eval["nonzero_mae"],
                        "nonzero_rmse": hurdle_eval["nonzero_rmse"],
                        "nonzero_spearman": hurdle_eval["nonzero_spearman"],
                        "prediction_bias": hurdle_eval["prediction_bias"],
                        "top_decile_xg_capture": hurdle_eval["top_10_pct_xg_capture"],
                        "top_10_pct_xg_capture": hurdle_eval["top_10_pct_xg_capture"],
                        "top_20_pct_xg_capture": hurdle_eval["top_20_pct_xg_capture"],
                        "top_30_pct_xg_capture": hurdle_eval["top_30_pct_xg_capture"],
                        "fold_mean": fold_stats["fold_mean"],
                        "fold_std": fold_stats["fold_std"],
                        **conditional_eval,
                    }
                    all_rows.append(row)

                    run_root = dirs["reports"]
                    oof_path = run_root / f"{run_label}_combined_oof.parquet"
                    fold_path = run_root / f"{run_label}_conditional_fold_metrics.csv"
                    cond_path = run_root / f"{run_label}_conditional_metrics.json"
                    hurdle_path = run_root / f"{run_label}_hurdle_metrics.json"
                    decile_path = run_root / f"{run_label}_combined_deciles.csv"
                    combined_oof.to_parquet(oof_path, index=False)
                    fold_metrics.to_csv(fold_path, index=False)
                    cond_path.write_text(json.dumps(conditional_eval, indent=2, default=str), encoding="utf-8")
                    hurdle_path.write_text(json.dumps(hurdle_eval, indent=2, default=str), encoding="utf-8")
                    deciles.to_csv(decile_path, index=False)

                    final_payload, training_nonzero_rows, training_match_count = fit_final_conditional_model(
                        aligned_df,
                        spec,
                        resolved,
                        target_col="target_future_xg_10s",
                    )

                    sample_features = aligned_df[resolved["final_features"]].head(5).copy()
                    _ = predict_conditional(final_payload, sample_features)

                    bundle_path = dirs["bundles"] / f"{run_label}.joblib"
                    bundle = {
                        "conditional_model": final_payload,
                        "feature_contract": regression_contracts[DEFAULT_REGRESSION_CONTRACT].__dict__,
                        "target_definition": "target_future_xg_10s conditioned on target_future_xg_10s > 0",
                        "model_family": spec.model_family,
                        "hyperparameters": spec.hyperparameters,
                        "dataset_fingerprint": fingerprint,
                        "git_sha": git_sha(),
                        "classification_variant": classification_variant,
                        "mlflow_run_id": run_id,
                        "mlflow_model_uri": None,
                        "training_nonzero_row_count": training_nonzero_rows,
                        "training_match_count": training_match_count,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    joblib.dump(bundle, bundle_path)

                    loaded_bundle = joblib.load(bundle_path)
                    _ = predict_conditional(loaded_bundle["conditional_model"], sample_features)

                    # Common-row benchmark rows for this candidate
                    for benchmark_variant in BENCHMARK_VARIANTS:
                        benchmark_oof = load_regression_variant_oof(args.regression_oof, args.output_dir, benchmark_variant)
                        shared_ids = sorted(set(combined_oof["event_id"]).intersection(set(benchmark_oof["event_id"])))
                        if not shared_ids:
                            benchmark_rows.append(
                                {
                                    "classification_variant": classification_variant,
                                    "conditional_model": spec.name,
                                    "benchmark_variant": benchmark_variant,
                                    "rows": 0,
                                    "selection_reason": "no shared rows",
                                }
                            )
                            continue

                        candidate_shared = combined_oof.set_index("event_id").loc[shared_ids].reset_index()
                        benchmark_shared = benchmark_oof.set_index("event_id").loc[shared_ids].reset_index()
                        benchmark_frame = candidate_shared[["event_id", "match_id", "fold", "observed_future_xg"]].copy()
                        benchmark_frame["combined_future_xg_prediction"] = benchmark_shared["y_pred"].to_numpy()
                        candidate_metrics = make_common_row_metrics(candidate_shared)
                        benchmark_metrics = make_common_row_metrics(benchmark_frame)

                        benchmark_rows.append(
                            {
                                "classification_variant": classification_variant,
                                "conditional_model": spec.name,
                                "benchmark_variant": benchmark_variant,
                                "rows": int(len(shared_ids)),
                                "matches": int(candidate_shared["match_id"].nunique()),
                                "candidate_mae": candidate_metrics["mae"],
                                "benchmark_mae": benchmark_metrics["mae"],
                                "candidate_rmse": candidate_metrics["rmse"],
                                "benchmark_rmse": benchmark_metrics["rmse"],
                                "candidate_r2": candidate_metrics["r2"],
                                "benchmark_r2": benchmark_metrics["r2"],
                                "candidate_spearman": candidate_metrics["spearman"],
                                "benchmark_spearman": benchmark_metrics["spearman"],
                                "candidate_nonzero_mae": candidate_metrics["nonzero_mae"],
                                "benchmark_nonzero_mae": benchmark_metrics["nonzero_mae"],
                                "candidate_nonzero_rmse": candidate_metrics["nonzero_rmse"],
                                "benchmark_nonzero_rmse": benchmark_metrics["nonzero_rmse"],
                                "candidate_nonzero_spearman": candidate_metrics["nonzero_spearman"],
                                "benchmark_nonzero_spearman": benchmark_metrics["nonzero_spearman"],
                                "candidate_top_10_pct_xg_capture": candidate_metrics["top_10_pct_xg_capture"],
                                "benchmark_top_10_pct_xg_capture": benchmark_metrics["top_10_pct_xg_capture"],
                            }
                        )

                    benchmark_df_partial = pd.DataFrame(benchmark_rows)
                    chart_paths = save_diagnostics(
                        combined_oof,
                        fold_metrics,
                        dirs["charts"] / run_label,
                        benchmark_df_partial.loc[
                            benchmark_df_partial["classification_variant"].eq(classification_variant)
                            & benchmark_df_partial["conditional_model"].eq(spec.name)
                        ].copy(),
                    )

                    log_params(
                        mlflow,
                        {
                            "classification_variant": classification_variant,
                            "conditional_variant": spec.name,
                            "conditional_family": spec.model_family,
                            "conditional_hyperparameters": spec.hyperparameters,
                            "feature_contract": regression_contracts[DEFAULT_REGRESSION_CONTRACT].__dict__,
                            "cloudpickle_serialization": True,
                        },
                    )
                    log_metrics(mlflow, conditional_eval, prefix="conditional_")
                    log_metrics(mlflow, hurdle_eval, prefix="hurdle_")
                    for artifact in [oof_path, fold_path, cond_path, hurdle_path, decile_path, bundle_path, *chart_paths]:
                        log_artifact(mlflow, artifact, artifact_path=run_label)

                    if spec.model_family != "conditional_mean":
                        mlflow_model = log_sklearn_model(
                            mlflow,
                            final_payload["pipeline"],
                            artifact_path=f"{run_label}/model",
                            variant=run_label,
                            serialization_format="cloudpickle",
                        )
                        if mlflow_model is not None:
                            # Update bundle with MLflow URI, resave, and re-log the
                            # final bundle so the artifact and local file are identical.
                            bundle["mlflow_model_uri"] = mlflow_model.get("model_uri")
                            joblib.dump(bundle, bundle_path)
                            log_artifact(mlflow, bundle_path, artifact_path=run_label)

    comparison_df = pd.DataFrame(all_rows)
    benchmark_df = pd.DataFrame(benchmark_rows)

    requested_columns = [
        "classification_variant",
        "conditional_model",
        "rows",
        "matches",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "nonzero_mae",
        "nonzero_rmse",
        "nonzero_spearman",
        "prediction_bias",
        "top_decile_xg_capture",
        "fold_mean",
        "fold_std",
    ]

    common_rows_records: list[dict[str, Any]] = []
    if {"b7_full_with_360", "b6_full_without_360"}.issubset(set(classification_variants)):
        for spec_name in sorted(comparison_df["conditional_model"].dropna().unique().tolist()):
            a = variant_outputs.get(("b7_full_with_360", spec_name))
            b = variant_outputs.get(("b6_full_without_360", spec_name))
            if a is None or b is None:
                continue
            shared = sorted(set(a["event_id"]).intersection(set(b["event_id"])))
            if not shared:
                continue
            for class_variant, source in [("b7_full_with_360", a), ("b6_full_without_360", b)]:
                subset = source.set_index("event_id").loc[shared].reset_index()
                metrics = make_common_row_metrics(subset)
                common_rows_records.append(
                    {
                        "classification_variant": class_variant,
                        "conditional_model": spec_name,
                        "rows": int(len(subset)),
                        "matches": int(subset["match_id"].nunique()),
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "spearman": metrics["spearman"],
                        "nonzero_mae": metrics["nonzero_mae"],
                        "nonzero_rmse": metrics["nonzero_rmse"],
                        "nonzero_spearman": metrics["nonzero_spearman"],
                        "prediction_bias": metrics["prediction_bias"],
                        "top_decile_xg_capture": metrics["top_10_pct_xg_capture"],
                        "fold_mean": metrics["fold_mean"],
                        "fold_std": metrics["fold_std"],
                    }
                )

    classification_variant_comparison = pd.DataFrame(common_rows_records)
    comparison_path = dirs["reports"] / "two_part_classification_variant_comparison.csv"
    if classification_variant_comparison.empty:
        comparison_df.loc[:, requested_columns].to_csv(comparison_path, index=False)
    else:
        classification_variant_comparison.loc[:, requested_columns].to_csv(comparison_path, index=False)

    benchmark_path = dirs["reports"] / "two_part_vs_one_stage_common_rows.csv"
    benchmark_df.to_csv(benchmark_path, index=False)

    selection_df = selection_with_reasons(comparison_df, benchmark_df)
    selection_path = dirs["reports"] / "two_part_model_selection.csv"
    selection_df.to_csv(selection_path, index=False)

    # Save preferred OOF per classification variant
    preferred_rows = selection_df.loc[selection_df["status"].eq("preferred candidate")].copy()
    for _, row in preferred_rows.iterrows():
        key = (str(row["classification_variant"]), str(row["conditional_model"]))
        oof = variant_outputs.get(key)
        if oof is None:
            continue
        oof.to_parquet(dirs["oof"] / f"two_part_future_xg_oof_{row['classification_variant']}.parquet", index=False)
    if not preferred_rows.empty:
        first = preferred_rows.iloc[0]
        key = (str(first["classification_variant"]), str(first["conditional_model"]))
        if key in variant_outputs:
            variant_outputs[key].to_parquet(dirs["oof"] / "two_part_future_xg_oof.parquet", index=False)

    # If no candidate beat r4: save best exploratory OOF and label it clearly.
    exploratory_oof_path: Path | None = None
    exploratory_label: str | None = None
    if preferred_rows.empty and variant_outputs:
        exploratory_candidates = selection_df.loc[
            selection_df["status"].isin(["insufficient evidence", "backup candidate"])
        ].copy()
        if exploratory_candidates.empty:
            exploratory_candidates = selection_df.copy()
        if not exploratory_candidates.empty:
            # Use selection_score to pick the best exploratory candidate; lower is better.
            if "selection_score" in exploratory_candidates.columns:
                exploratory_candidates = exploratory_candidates.sort_values("selection_score")
            best_exploratory = exploratory_candidates.iloc[0]
            key = (str(best_exploratory["classification_variant"]), str(best_exploratory["conditional_model"]))
            if key in variant_outputs:
                exploratory_oof = variant_outputs[key]
                exploratory_oof_path = dirs["oof"] / "two_part_future_xg_oof_exploratory.parquet"
                exploratory_oof.to_parquet(exploratory_oof_path, index=False)
                exploratory_label = f"{key[0]}/{key[1]}"
                print(
                    f"\nNOTE: No hurdle candidate beat r4 on shared-row metrics. "
                    f"Saved exploratory OOF for '{exploratory_label}' to: {exploratory_oof_path}"
                )

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    sensitivity_rows: list[dict[str, Any]] = []

    def _player_suppression(source_oof: pd.DataFrame, pred_col: str, obs_col: str) -> pd.DataFrame:
        grp = source_oof.groupby(["player_id", "team"], dropna=False)
        agg = grp[[pred_col, obs_col]].sum().reset_index()
        agg["suppression"] = agg[pred_col] - agg[obs_col]
        return agg

    def _pair_rows(
        name_a: str,
        sup_a: pd.DataFrame,
        name_b: str,
        sup_b: pd.DataFrame,
        ci_a: pd.DataFrame | None = None,
        ci_b: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        merged = sup_a.merge(sup_b, on=["player_id", "team"], suffixes=("_a", "_b"), how="inner")
        if merged.empty:
            return []
        merged["rank_a"] = merged["suppression_a"].rank(ascending=False, method="min")
        merged["rank_b"] = merged["suppression_b"].rank(ascending=False, method="min")
        merged["rank_change"] = (merged["rank_a"] - merged["rank_b"]).abs()
        merged["sign_change"] = np.sign(merged["suppression_a"]) != np.sign(merged["suppression_b"])
        merged["unstable"] = merged["rank_change"].gt(100) | merged["sign_change"]
        valid_a = merged["suppression_a"].dropna()
        valid_b = merged["suppression_b"].dropna()
        if len(valid_a) >= 2 and len(valid_b) >= 2:
            spearman_val = float(merged[["suppression_a", "suppression_b"]].corr(method="spearman").iloc[0, 1])
            kendall_val = float(merged[["suppression_a", "suppression_b"]].corr(method="kendall").iloc[0, 1])
        else:
            spearman_val = float("nan")
            kendall_val = float("nan")
        # CI overlap between model A and model B per player.
        ci_merged: pd.DataFrame | None = None
        if ci_a is not None and ci_b is not None and not ci_a.empty and not ci_b.empty:
            ci_merged = ci_a.merge(ci_b, on=["player_id", "team"], suffixes=("_a", "_b"), how="inner")
        out = []
        for _, r in merged.iterrows():
            ci_overlap = float("nan")
            if ci_merged is not None:
                ci_row = ci_merged.loc[
                    ci_merged["player_id"].eq(r["player_id"]) & ci_merged["team"].eq(r["team"])
                ]
                if not ci_row.empty:
                    lo_a = float(ci_row.iloc[0]["ci95_lower_a"])
                    hi_a = float(ci_row.iloc[0]["ci95_upper_a"])
                    lo_b = float(ci_row.iloc[0]["ci95_lower_b"])
                    hi_b = float(ci_row.iloc[0]["ci95_upper_b"])
                    ci_overlap = bool(lo_a <= hi_b and lo_b <= hi_a)
            out.append({
                "comparison_pair": f"{name_a}_vs_{name_b}",
                "player_id": r["player_id"],
                "team": r["team"],
                f"suppression_{name_a}": r["suppression_a"],
                f"suppression_{name_b}": r["suppression_b"],
                f"rank_{name_a}": r["rank_a"],
                f"rank_{name_b}": r["rank_b"],
                "rank_change": r["rank_change"],
                "suppression_sign_change": bool(r["sign_change"]),
                "spearman_rank_correlation": spearman_val,
                "kendall_rank_correlation": kendall_val,
                "confidence_interval_overlap": ci_overlap,
                "unstable_player_flag": bool(r["unstable"]),
            })
        return out

    has_b7 = "b7_full_with_360" in classification_variants
    has_b6 = "b6_full_without_360" in classification_variants

    # Attempt to load r4 for comparisons.
    r4_player_oof: pd.DataFrame | None = None
    try:
        r4_raw = load_regression_variant_oof(args.regression_oof, args.output_dir, "r4_full_with_360")
        if {"player_id", "team", "y_pred", "y_true"}.issubset(r4_raw.columns):
            r4_player_oof = r4_raw
    except Exception:  # noqa: BLE001
        pass

    def _get_preferred_oof(class_variant: str) -> pd.DataFrame | None:
        preferred_row = preferred_rows.loc[preferred_rows["classification_variant"].eq(class_variant)]
        if preferred_row.empty:
            # Fall back to best insufficient-evidence candidate.
            ie_rows = selection_df.loc[
                selection_df["classification_variant"].eq(class_variant)
                & selection_df["status"].isin(["insufficient evidence", "backup candidate"])
            ]
            if ie_rows.empty:
                return None
            if "selection_score" in ie_rows.columns:
                ie_rows = ie_rows.sort_values("selection_score")
            preferred_row = ie_rows.head(1)
        cond_model = str(preferred_row.iloc[0]["conditional_model"])
        return variant_outputs.get((class_variant, cond_model))

    b7_oof = _get_preferred_oof("b7_full_with_360") if has_b7 else None
    b6_oof = _get_preferred_oof("b6_full_without_360") if has_b6 else None

    if b7_oof is not None and b6_oof is not None:
        sup_b7 = _player_suppression(b7_oof, "combined_future_xg_prediction", "observed_future_xg")
        sup_b6 = _player_suppression(b6_oof, "combined_future_xg_prediction", "observed_future_xg")
        ci_b7 = _compute_match_level_suppression_ci(b7_oof)
        ci_b6 = _compute_match_level_suppression_ci(b6_oof)
        sensitivity_rows.extend(_pair_rows("b7", sup_b7, "b6", sup_b6, ci_b7, ci_b6))

        if r4_player_oof is not None:
            sup_r4 = _player_suppression(r4_player_oof, "y_pred", "y_true")
            ci_r4 = _compute_match_level_suppression_ci(
                r4_player_oof.rename(columns={"y_pred": "combined_future_xg_prediction", "y_true": "observed_future_xg"})
            )
            sensitivity_rows.extend(_pair_rows("b7", sup_b7, "r4", sup_r4, ci_b7, ci_r4))
            sensitivity_rows.extend(_pair_rows("b6", sup_b6, "r4", sup_r4, ci_b6, ci_r4))

    # all eligible vs reliable_5m_visibility rows for the combined OOF.
    for class_variant, src_oof in [(v, _get_preferred_oof(v)) for v in classification_variants]:
        if src_oof is None:
            continue
        if "reliable_5m_visibility" not in src_oof.columns:
            continue
        reliable = src_oof.loc[src_oof["reliable_5m_visibility"].eq(True)].copy()
        if reliable.empty or len(reliable) == len(src_oof):
            continue
        sup_all = _player_suppression(src_oof, "combined_future_xg_prediction", "observed_future_xg")
        sup_rel = _player_suppression(reliable, "combined_future_xg_prediction", "observed_future_xg")
        ci_all = _compute_match_level_suppression_ci(src_oof)
        ci_rel = _compute_match_level_suppression_ci(reliable)
        variant_tag = "b7" if class_variant == "b7_full_with_360" else "b6"
        sensitivity_rows.extend(
            _pair_rows(f"{variant_tag}_all", sup_all, f"{variant_tag}_reliable", sup_rel, ci_all, ci_rel)
        )

    sensitivity_path = dirs["reports"] / "player_signal_sensitivity.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)

    # ── Expanded evaluation report ───────────────────────────────────────────
    report_path = dirs["reports"] / "two_part_xg_evaluation.md"

    def _fmt(val: Any, decimals: int = 4) -> str:
        try:
            f = float(val)
            import math
            return f"{f:.{decimals}f}" if math.isfinite(f) else str(val)
        except (TypeError, ValueError):
            return str(val)

    has_preferred = not preferred_rows.empty
    any_beats_r4 = (
        has_preferred and
        not benchmark_df.empty and
        benchmark_df.loc[
            benchmark_df["benchmark_variant"].eq("r4_full_with_360") &
            benchmark_df["rows"].gt(0)
        ].apply(
            lambda r: safe_numeric(r.get("candidate_mae", np.nan)) <= safe_numeric(r.get("benchmark_mae", np.nan)),
            axis=1,
        ).any()
    )

    lines = [
        "# Two-Part xG Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Dataset and OOF Alignment",
        "",
    ]
    for variant in classification_variants:
        source = classification_sources.get(variant)
        checked = checked_messages.get(variant, [])
        lines.append(f"- `{variant}` source: `{source}`")
        lines.append(f"- `{variant}` checked paths: {len(checked)}")

    lines += ["", "## 2. Best Candidates", ""]
    for cv in classification_variants:
        pref = preferred_rows.loc[preferred_rows["classification_variant"].eq(cv)]
        if not pref.empty:
            r = pref.iloc[0]
            lines.append(f"- **{cv}** preferred: `{r['conditional_model']}` "
                         f"(MAE={_fmt(r.get('mae'))}, RMSE={_fmt(r.get('rmse'))}, "
                         f"Spearman={_fmt(r.get('spearman'))}, "
                         f"fold_std={_fmt(r.get('fold_std'))})")
        else:
            ie = selection_df.loc[
                selection_df["classification_variant"].eq(cv) &
                selection_df["status"].eq("insufficient evidence")
            ]
            if not ie.empty:
                r = ie.iloc[0]
                lines.append(f"- **{cv}** best insufficient-evidence: `{r['conditional_model']}` "
                             f"(MAE={_fmt(r.get('mae'))}, status=insufficient evidence)")
            else:
                lines.append(f"- **{cv}**: no preferred or insufficient-evidence candidate found")

    lines += ["", "## 3. r4 Benchmark Comparison (Exact Common Rows)", ""]
    r4_bench = benchmark_df.loc[benchmark_df["benchmark_variant"].eq("r4_full_with_360") & benchmark_df["rows"].gt(0)]
    if r4_bench.empty:
        lines.append("- No shared rows found with r4 benchmark.")
    else:
        for _, r in r4_bench.iterrows():
            delta_mae = safe_numeric(r.get("candidate_mae")) - safe_numeric(r.get("benchmark_mae"))
            delta_rmse = safe_numeric(r.get("candidate_rmse")) - safe_numeric(r.get("benchmark_rmse"))
            lines.append(
                f"- `{r['classification_variant']}` / `{r['conditional_model']}` "
                f"vs r4 ({int(r['rows'])} shared rows): "
                f"ΔMAE={_fmt(delta_mae)} ({_fmt(r.get('candidate_mae'))} vs {_fmt(r.get('benchmark_mae'))}), "
                f"ΔRMSE={_fmt(delta_rmse)} ({_fmt(r.get('candidate_rmse'))} vs {_fmt(r.get('benchmark_rmse'))}), "
                f"candidate_nonzero_MAE={_fmt(r.get('candidate_nonzero_mae'))}"
            )

    lines += ["", "## 4. Conditional Severity Metrics", ""]
    cond_cols = ["conditional_mae", "conditional_rmse", "conditional_r2", "conditional_spearman",
                 "conditional_sample_count", "observed_conditional_mean", "predicted_conditional_mean"]
    for cv in classification_variants:
        cv_rows = selection_df.loc[
            selection_df["classification_variant"].eq(cv) &
            selection_df["status"].isin(["preferred candidate", "insufficient evidence"])
        ]
        if cv_rows.empty:
            continue
        r = cv_rows.iloc[0]
        parts = [f"n_shots={_fmt(r.get('conditional_sample_count'), 0)}"]
        for col in cond_cols:
            if col in r.index:
                parts.append(f"{col.replace('conditional_', '')}={_fmt(r.get(col))}")
        lines.append(f"- **{cv}** / `{r['conditional_model']}`: {', '.join(parts)}")

    lines += ["", "## 5. Hurdle Metrics (All Rows)", ""]
    for cv in classification_variants:
        cv_rows = selection_df.loc[
            selection_df["classification_variant"].eq(cv) &
            selection_df["status"].isin(["preferred candidate", "insufficient evidence"])
        ]
        if cv_rows.empty:
            continue
        r = cv_rows.iloc[0]
        lines.append(
            f"- **{cv}** / `{r['conditional_model']}`: "
            f"MAE={_fmt(r.get('mae'))}, RMSE={_fmt(r.get('rmse'))}, R²={_fmt(r.get('r2'))}, "
            f"Spearman={_fmt(r.get('spearman'))}, nonzero_MAE={_fmt(r.get('nonzero_mae'))}, "
            f"nonzero_Spearman={_fmt(r.get('nonzero_spearman'))}, "
            f"top10%_xG_capture={_fmt(r.get('top_decile_xg_capture'))}"
        )

    lines += ["", "## 6. Fold Stability", ""]
    for cv in classification_variants:
        cv_rows = selection_df.loc[
            selection_df["classification_variant"].eq(cv) &
            selection_df["status"].isin(["preferred candidate", "insufficient evidence"])
        ]
        if cv_rows.empty:
            continue
        r = cv_rows.iloc[0]
        lines.append(
            f"- **{cv}** / `{r['conditional_model']}`: "
            f"fold_mean_MAE={_fmt(r.get('fold_mean'))}, fold_std_MAE={_fmt(r.get('fold_std'))}"
        )

    lines += ["", "## 7. xG Capture", ""]
    for cv in classification_variants:
        cv_rows = selection_df.loc[
            selection_df["classification_variant"].eq(cv) &
            selection_df["status"].isin(["preferred candidate", "insufficient evidence"])
        ]
        if cv_rows.empty:
            continue
        r = cv_rows.iloc[0]
        lines.append(
            f"- **{cv}**: top10%={_fmt(r.get('top_10_pct_xg_capture'))}, "
            f"top20%={_fmt(r.get('top_20_pct_xg_capture'))}, "
            f"top30%={_fmt(r.get('top_30_pct_xg_capture'))}"
            if "top_20_pct_xg_capture" in r.index
            else f"- **{cv}**: top10%={_fmt(r.get('top_decile_xg_capture'))}"
        )

    lines += ["", "## 8. Sensitivity Results", ""]
    if sensitivity_rows:
        sensitivity_df_report = pd.DataFrame(sensitivity_rows)
        for pair in sorted(sensitivity_df_report["comparison_pair"].unique().tolist()):
            pair_data = sensitivity_df_report.loc[sensitivity_df_report["comparison_pair"].eq(pair)]
            spearman_val = pair_data["spearman_rank_correlation"].iloc[0] if not pair_data.empty else float("nan")
            kendall_val = pair_data["kendall_rank_correlation"].iloc[0] if not pair_data.empty else float("nan")
            sign_changes = int(pair_data["suppression_sign_change"].sum()) if "suppression_sign_change" in pair_data.columns else 0
            unstable = int(pair_data["unstable_player_flag"].sum()) if "unstable_player_flag" in pair_data.columns else 0
            ci_overlap_computed = not pair_data["confidence_interval_overlap"].isna().all() if "confidence_interval_overlap" in pair_data.columns else False
            lines.append(
                f"- **{pair}** ({len(pair_data)} players): "
                f"Spearman={_fmt(spearman_val)}, Kendall={_fmt(kendall_val)}, "
                f"sign_changes={sign_changes}, unstable={unstable}, "
                f"CI_overlap_computed={'yes' if ci_overlap_computed else 'no'}"
            )
    else:
        lines.append("- Sensitivity not computed (preferred candidates missing).")

    lines += ["", "## 9. Reliability Thresholds", ""]
    lines.append("- Derived from player-signal CLI (`build_provisional_player_signals.py`).")
    lines.append("- Thresholds: `player_signal_reliability_thresholds.json`.")
    lines.append("- conditional_severity_reliability_flag uses the data-derived q50 of observed_shot_count.")

    lines += ["", "## 10. Model Selection Summary", ""]
    lines.append(f"- Hurdle model improves on r4: **{'YES' if any_beats_r4 else 'NO'}**")
    if not has_preferred:
        lines.append("- **Player signals are EXPLORATORY ONLY**: no candidate beat r4 on all core error metrics.")
        if exploratory_label:
            lines.append(f"  Exploratory OOF saved: `{exploratory_oof_path}`  (candidate: `{exploratory_label}`)")
    lines.append("")
    lines.append("All candidate statuses:")
    status_counts = selection_df["status"].value_counts().to_dict() if not selection_df.empty else {}
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")

    lines += ["", "## 11. Merge Recommendation", ""]
    if any_beats_r4:
        lines.append(
            "- At least one hurdle candidate **beats r4** on shared-row metrics. "
            "Review the preferred candidate selection and conditional diagnostics before promoting."
        )
    else:
        lines.append(
            "- **No hurdle candidate beat r4** on shared-row core error metrics (MAE, RMSE, nonzero_MAE). "
            "Do NOT promote hurdle model as default; treat all outputs as exploratory. "
            "Consider richer feature contracts or extended training data before re-evaluation."
        )

    lines += ["", "## 12. Artifacts", ""]
    lines.append(f"- Classification-variant comparison: `{comparison_path}`")
    lines.append(f"- Common-row benchmark comparison: `{benchmark_path}`")
    lines.append(f"- Selection table: `{selection_path}`")
    lines.append(f"- Sensitivity report: `{sensitivity_path}`")
    lines.append(f"- Diagnostics charts: `{dirs['charts']}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"- Classification-variant comparison: {comparison_path}")
    print(f"- Common-row benchmark comparison: {benchmark_path}")
    print(f"- Selection table: {selection_path}")
    print(f"- Evaluation report: {report_path}")


if __name__ == "__main__":
    main()



