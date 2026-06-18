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

        r4_rows = benchmark.loc[
            benchmark["classification_variant"].eq(classification_variant)
            & benchmark["conditional_model"].eq(frame.loc[preferred_idx, "conditional_model"])
            & benchmark["benchmark_variant"].eq("r4_full_with_360")
        ]
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
                            bundle["mlflow_model_uri"] = mlflow_model.get("model_uri")
                            joblib.dump(bundle, bundle_path)

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

    # Sensitivity output for player-level suppression stability
    sensitivity_rows: list[dict[str, Any]] = []
    if {"b7_full_with_360", "b6_full_without_360"}.issubset(set(classification_variants)):
        def _preferred_variant(class_variant: str) -> tuple[str, pd.DataFrame] | None:
            row = preferred_rows.loc[preferred_rows["classification_variant"].eq(class_variant)]
            if row.empty:
                return None
            conditional_model = str(row.iloc[0]["conditional_model"])
            oof = variant_outputs.get((class_variant, conditional_model))
            return (conditional_model, oof) if oof is not None else None

        preferred_b7 = _preferred_variant("b7_full_with_360")
        preferred_b6 = _preferred_variant("b6_full_without_360")
        if preferred_b7 is not None and preferred_b6 is not None:
            r4 = load_regression_variant_oof(args.regression_oof, args.output_dir, "r4_full_with_360")
            b7_player = preferred_b7[1].groupby(["player_id", "team"], dropna=False)[["combined_future_xg_prediction", "observed_future_xg"]].sum().reset_index()
            b6_player = preferred_b6[1].groupby(["player_id", "team"], dropna=False)[["combined_future_xg_prediction", "observed_future_xg"]].sum().reset_index()
            r4_player = r4.groupby(["player_id", "team"], dropna=False)[["y_pred", "y_true"]].sum().reset_index()
            merged = b7_player.merge(b6_player, on=["player_id", "team"], suffixes=("_b7", "_b6"), how="inner")
            merged = merged.merge(r4_player, on=["player_id", "team"], how="left")
            merged["supp_b7"] = merged["combined_future_xg_prediction_b7"] - merged["observed_future_xg_b7"]
            merged["supp_b6"] = merged["combined_future_xg_prediction_b6"] - merged["observed_future_xg_b6"]
            merged["supp_r4"] = merged["y_pred"] - merged["y_true"]
            merged["rank_b7"] = merged["supp_b7"].rank(ascending=False, method="min")
            merged["rank_b6"] = merged["supp_b6"].rank(ascending=False, method="min")
            merged["rank_change_b7_vs_b6"] = (merged["rank_b7"] - merged["rank_b6"]).abs()
            merged["suppression_sign_change_b7_vs_b6"] = np.sign(merged["supp_b7"]) != np.sign(merged["supp_b6"])
            merged["unstable_player_flag"] = merged["rank_change_b7_vs_b6"].gt(100) | merged["suppression_sign_change_b7_vs_b6"]
            global_spearman = merged[["supp_b7", "supp_b6"]].corr(method="spearman").iloc[0, 1]
            global_kendall = merged[["supp_b7", "supp_b6"]].corr(method="kendall").iloc[0, 1]
            for _, row in merged.iterrows():
                sensitivity_rows.append(
                    {
                        "player_id": row["player_id"],
                        "team": row["team"],
                        "suppression_b7": row["supp_b7"],
                        "suppression_b6": row["supp_b6"],
                        "suppression_r4": row["supp_r4"],
                        "spearman_rank_correlation_b7_vs_b6": global_spearman,
                        "kendall_rank_correlation_b7_vs_b6": global_kendall,
                        "rank_change_b7_vs_b6": row["rank_change_b7_vs_b6"],
                        "suppression_sign_change_b7_vs_b6": bool(row["suppression_sign_change_b7_vs_b6"]),
                        "confidence_interval_overlap": np.nan,
                        "unstable_player_flag": bool(row["unstable_player_flag"]),
                    }
                )

    sensitivity_path = dirs["reports"] / "player_signal_sensitivity.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)

    # Final markdown report
    report_path = dirs["reports"] / "two_part_xg_evaluation.md"
    lines = [
        "# Two-Part xG Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset and OOF alignment",
        "",
    ]
    for variant in classification_variants:
        source = classification_sources.get(variant)
        checked = checked_messages.get(variant, [])
        lines.append(f"- `{variant}` source: `{source}`")
        lines.append(f"- `{variant}` checks: {len(checked)} paths")
    lines += [
        "",
        "## Classification variant comparison (exact common rows)",
        "",
        f"- Output: `{comparison_path}`",
        "",
        "## Conditional model results",
        "",
        f"- Output: `{selection_path}`",
        "",
        "## Hurdle model results",
        "",
        f"- Output: `{benchmark_path}`",
        "",
        "## Fold stability",
        "",
        "- Reported via `fold_mean` and `fold_std` in comparison tables.",
        "",
        "## Decile ranking",
        "",
        "- Stored per variant as `*_combined_deciles.csv` under reports.",
        "",
        "## Diagnostics",
        "",
        f"- Charts saved under `{dirs['charts']}`",
        "",
        "## Model selection",
        "",
        "- Statuses: preferred candidate, backup candidate, reference baseline, insufficient evidence, rejected.",
        "",
        "## Player-signal reliability",
        "",
        "- Generated by the provisional player-signal CLI.",
        "",
        "## Sensitivity findings",
        "",
        f"- Output: `{sensitivity_path}`",
        "",
        "## Limitations",
        "",
        "- Reliability CI overlap is not computed in this CLI (left as NaN in sensitivity output).",
        "- Interpretation should rely on common-row benchmark comparisons against r4/r6.",
        "",
        "## Merge recommendation",
        "",
        "- Review `two_part_vs_one_stage_common_rows.csv` and do not promote hurdle models unless multi-metric evidence beats one-stage references.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"- Classification-variant comparison: {comparison_path}")
    print(f"- Common-row benchmark comparison: {benchmark_path}")
    print(f"- Selection table: {selection_path}")
    print(f"- Evaluation report: {report_path}")


if __name__ == "__main__":
    main()



