"""
CLI for training two-part hurdle future-xG models.

python scripts/train_two_part_xg.py \
  --input data/features/player_defensive_actions.parquet \
  --classification-oof outputs/oof/classification_oof.parquet \
  --config configs/models.yaml \
  --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from dax.models.feature_contracts import get_contracts, resolve_contract
from dax.models.mlflow_tracking import configure_mlflow, start_parent_run
from dax.models.training import run_training
from dax.models.two_part_xg import (
    ConditionalModelSpec,
    build_combined_oof,
    compute_hurdle_metrics,
    decile_ranking,
    filter_classification_oof,
    load_classification_oof,
    train_conditional_variant,
)


DEFAULT_CLASSIFICATION_VARIANT = "b7_full_with_360"
DEFAULT_REGRESSION_CONTRACT = "r4_full_with_360"


def build_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create canonical output directory structure."""
    root = Path(output_dir)
    dirs = {
        "oof": root / "oof",
        "models": root / "models" / "two_part_xg",
        "models_charts": root / "models" / "two_part_xg" / "charts",
        "models_comparisons": root / "models" / "comparisons",
        "validation": root / "models" / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def candidate_classification_oof_paths(
    oof_path: str | Path,
    output_dir: str | Path,
    classification_variant: str,
) -> list[Path]:
    """Return candidate OOF paths, preferring the caller path then the per-variant artifact."""

    primary = Path(oof_path)
    output_root = Path(output_dir)
    candidates = [primary]
    fallbacks = [
        output_root / "validation" / "classification" / f"{classification_variant}_oof_predictions.parquet",
    ]
    if len(primary.parents) >= 2:
        fallbacks.append(primary.parents[1] / "validation" / "classification" / f"{classification_variant}_oof_predictions.parquet")
    for candidate in fallbacks:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def ensure_feature_alignment(
    df: pd.DataFrame,
    classification_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align input rows to a single-variant classification OOF population."""
    if "event_id" not in df.columns:
        raise ValueError("Input data must have 'event_id' column.")
    if "event_id" not in classification_oof.columns:
        raise ValueError("Classification OOF must have 'event_id' column.")

    if df["event_id"].duplicated().any():
        raise ValueError("Input data contains duplicate event_id rows; two-part training requires unique events.")

    input_events = set(df["event_id"].unique())
    oof_events = set(classification_oof["event_id"].unique())

    missing_in_input = oof_events - input_events
    if missing_in_input:
        raise ValueError(
            f"Classification OOF contains {len(missing_in_input)} events not present in the input data. "
            "This usually means the supplied OOF file was built from a different dataset or a fixture."
        )

    if not oof_events:
        raise ValueError("Classification OOF has no event rows after variant filtering.")

    aligned_df = df.loc[df["event_id"].isin(oof_events)].copy().reset_index(drop=True)
    if aligned_df.empty:
        raise ValueError("Classification OOF has zero overlapping events with the input dataset.")

    aligned_classification_oof = (
        classification_oof.set_index("event_id")
        .loc[aligned_df["event_id"]]
        .reset_index()
    )
    return aligned_df, aligned_classification_oof


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
    """Load the best matching classification OOF source for the requested variant."""

    input_events = set(df["event_id"].unique())
    best_candidate: tuple[int, int, pd.DataFrame, Path] | None = None
    attempted: list[str] = []

    for candidate_path in candidate_classification_oof_paths(oof_path, output_dir, classification_variant):
        if not candidate_path.exists():
            attempted.append(f"{candidate_path} [missing]")
            continue
        try:
            candidate_oof = filter_classification_oof(load_classification_oof(candidate_path), classification_variant)
        except Exception as exc:  # noqa: BLE001 - preserve per-candidate diagnostics
            attempted.append(f"{candidate_path} [invalid: {exc}]")
            continue

        oof_events = set(candidate_oof["event_id"].unique())
        overlap_count = len(input_events & oof_events)
        extraneous_count = len(oof_events - input_events)
        attempted.append(
            f"{candidate_path} [rows={len(candidate_oof)}, overlap={overlap_count}, extraneous={extraneous_count}]"
        )
        score = (overlap_count, -extraneous_count)
        if best_candidate is None or score > (best_candidate[0], best_candidate[1]):
            best_candidate = (overlap_count, -extraneous_count, candidate_oof, candidate_path)

    if best_candidate is None or best_candidate[0] == 0:
        if allow_rebuild:
            print(
                f"No usable classification OOF found for {classification_variant}; "
                "rebuilding classification artifacts from the current dataset..."
            )
            run_training(
                "classification",
                input_path,
                config_path,
                output_dir,
                mlflow_enabled=mlflow_enabled,
            )
            return load_aligned_classification_oof(
                df,
                input_path,
                oof_path,
                config_path,
                output_dir,
                classification_variant,
                mlflow_enabled=mlflow_enabled,
                allow_rebuild=False,
            )
        attempted_text = "\n  - ".join(["", *attempted]) if attempted else "\n  - no candidate files were checked"
        raise ValueError(
            f"Could not find a classification OOF for variant {classification_variant!r} that overlaps the input dataset."
            f"\nChecked:{attempted_text}"
        )

    selected_oof = best_candidate[2]
    selected_path = best_candidate[3]
    try:
        aligned_df, aligned_oof = ensure_feature_alignment(df, selected_oof)
    except ValueError:
        if not allow_rebuild:
            raise
        print(
            f"Classification OOF candidate {selected_path} is incompatible with the current dataset; "
            "rebuilding classification artifacts..."
        )
        run_training(
            "classification",
            input_path,
            config_path,
            output_dir,
            mlflow_enabled=mlflow_enabled,
        )
        return load_aligned_classification_oof(
            df,
            input_path,
            oof_path,
            config_path,
            output_dir,
            classification_variant,
            mlflow_enabled=mlflow_enabled,
            allow_rebuild=False,
        )
    return aligned_df, aligned_oof, selected_path


def train_conditional_variants(
    df: pd.DataFrame,
    classification_oof: pd.DataFrame,
    specs: list[ConditionalModelSpec],
    config: dict[str, Any],
    mlflow: Any | None,
    dirs: dict[str, Path],
    classification_variant: str,
) -> list[dict[str, Any]]:
    """Train all conditional model variants and return results."""
    
    results = []
    target_col = "target_future_xg_10s"
    
    regression_contracts = {contract.name: contract for contract in get_contracts(config, "regression")}
    if DEFAULT_REGRESSION_CONTRACT not in regression_contracts:
        raise ValueError(f"Regression contract {DEFAULT_REGRESSION_CONTRACT!r} was not found in the model config.")
    regression_contract = regression_contracts[DEFAULT_REGRESSION_CONTRACT]
    resolved = resolve_contract(df, regression_contract)

    if "fold" not in classification_oof.columns:
        raise ValueError("Classification OOF must have 'fold' column.")
    folds = classification_oof[["fold"]].copy()

    run_name = f"two-part-future-xg-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with start_parent_run(mlflow, "dax-two-part-future-xg", run_name) as parent_run:
        parent_run_id = getattr(getattr(parent_run, "info", None), "run_id", None)

        for spec in specs:
            print(f"Training conditional variant: {spec.name}")

            try:
                predictions, fold_metrics = train_conditional_variant(
                    df,
                    classification_oof,
                    spec,
                    resolved,
                    folds,
                    seed=config.get("seed", 42),
                    target_col=target_col,
                )
                
                # Build combined OOF
                combined_oof = build_combined_oof(
                    df,
                    classification_oof,
                    predictions,
                    folds,
                    spec,
                    classification_variant,
                    run_id=parent_run_id,
                )

                # Compute hurdle metrics
                hurdle_metrics = compute_hurdle_metrics(combined_oof)

                # Compute deciles
                deciles = decile_ranking(combined_oof)
                deciles.insert(0, "conditional_variant", spec.name)

                result = {
                    "variant": spec.name,
                    "model_family": spec.model_family,
                    "parent_run_id": parent_run_id,
                    "combined_oof": combined_oof,
                    "fold_metrics": fold_metrics,
                    "hurdle_metrics": hurdle_metrics,
                    "deciles": deciles,
                    "status": "success",
                }

                # Save intermediate outputs
                oof_path = dirs["validation"] / f"two_part_conditional_{spec.name}_oof.parquet"
                combined_oof.to_parquet(oof_path, index=False)

                fold_metrics_path = dirs["validation"] / f"two_part_conditional_{spec.name}_fold_metrics.csv"
                fold_metrics.to_csv(fold_metrics_path, index=False)

                metrics_path = dirs["validation"] / f"two_part_conditional_{spec.name}_metrics.json"
                metrics_path.write_text(json.dumps(hurdle_metrics, indent=2, default=str), encoding="utf-8")

                deciles_path = dirs["validation"] / f"two_part_conditional_{spec.name}_deciles.csv"
                deciles.to_csv(deciles_path, index=False)

                result["oof_path"] = oof_path
                result["fold_metrics_path"] = fold_metrics_path
                result["metrics_path"] = metrics_path
                result["deciles_path"] = deciles_path

                results.append(result)
                print(f"  ✓ {spec.name}: MAE={hurdle_metrics['mae']:.4f}, RMSE={hurdle_metrics['rmse']:.4f}")

            except Exception as e:
                results.append({
                    "variant": spec.name,
                    "status": "failed",
                    "error": str(e),
                })
                print(f"  ✗ {spec.name}: {str(e)}")

    return results


def select_best_variants(results: list[dict[str, Any]]) -> dict[str, str]:
    """Select preferred and backup conditional models and best hurdle model."""
    
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        raise ValueError("No successful conditional model variants.")
    
    # Sort by MAE (lower is better)
    sorted_by_mae = sorted(successful, key=lambda r: r["hurdle_metrics"]["mae"])
    
    selection = {
        "preferred_conditional": sorted_by_mae[0]["variant"],
        "preferred_conditional_mae": sorted_by_mae[0]["hurdle_metrics"]["mae"],
        "backup_conditional": sorted_by_mae[1]["variant"] if len(sorted_by_mae) > 1 else sorted_by_mae[0]["variant"],
        "backup_conditional_mae": sorted_by_mae[1]["hurdle_metrics"]["mae"] if len(sorted_by_mae) > 1 else sorted_by_mae[0]["hurdle_metrics"]["mae"],
    }
    
    return selection


def main():
    parser = argparse.ArgumentParser(description="Train two-part hurdle future-xG models")
    parser.add_argument("--input", required=True, help="Path to player_defensive_actions.parquet")
    parser.add_argument("--classification-oof", required=True, help="Path to classification_oof.parquet")
    parser.add_argument("--config", required=True, help="Path to models.yaml")
    parser.add_argument("--output-dir", required=True, help="Output directory root")
    parser.add_argument(
        "--classification-variant",
        default=DEFAULT_CLASSIFICATION_VARIANT,
        help=f"Classification model variant to pair with the conditional model (default: {DEFAULT_CLASSIFICATION_VARIANT})",
    )
    parser.add_argument("--disable-mlflow", action="store_true", help="Disable MLflow tracking")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.input)
    aligned_df, classification_oof, classification_oof_path = load_aligned_classification_oof(
        df,
        args.input,
        args.classification_oof,
        args.config,
        args.output_dir,
        args.classification_variant,
        mlflow_enabled=not args.disable_mlflow,
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if classification_oof_path != Path(args.classification_oof):
        print(f"Using fallback classification OOF: {classification_oof_path}")
    print(
        f"Using classification variant: {args.classification_variant} "
        f"({len(aligned_df):,} aligned rows from {len(df):,} input rows)"
    )

    # Setup MLflow
    mlflow = None
    if not args.disable_mlflow:
        mlflow = configure_mlflow(config.get("mlflow", {}))

    # Output directories
    dirs = build_output_dirs(args.output_dir)

    # Define conditional model variants to test
    conditional_specs = [
        ConditionalModelSpec(
            name="conditional_ridge_baseline",
            model_family="ridge",
            hyperparameters={"alpha": 1.0}
        ),
        ConditionalModelSpec(
            name="conditional_log_ridge",
            model_family="log_ridge",
            hyperparameters={"alpha": 1.0}
        ),
        ConditionalModelSpec(
            name="conditional_hgb",
            model_family="hist_gradient_boosting_regressor",
            hyperparameters={"max_iter": 100, "learning_rate": 0.05}
        ),
    ]

    # Train variants
    print("Training conditional variants...")
    results = train_conditional_variants(
        aligned_df,
        classification_oof,
        conditional_specs,
        config,
        mlflow,
        dirs,
        args.classification_variant,
    )

    # Select best models
    print("Selecting best models...")
    selection = select_best_variants(results)

    successful = [result for result in results if result["status"] == "success"]
    comparison_rows = [
        {
            "variant": result["variant"],
            "model_family": result.get("model_family"),
            "status": result["status"],
            "mae": result.get("hurdle_metrics", {}).get("mae", np.nan),
            "rmse": result.get("hurdle_metrics", {}).get("rmse", np.nan),
            "r2": result.get("hurdle_metrics", {}).get("r2", np.nan),
            "spearman": result.get("hurdle_metrics", {}).get("spearman", np.nan),
        }
        for result in results
    ]
    comparison_path = dirs["validation"] / "two_part_conditional_comparison.csv"
    pd.DataFrame(comparison_rows).sort_values(["status", "mae"], na_position="last").to_csv(comparison_path, index=False)

    preferred_result = next(result for result in successful if result["variant"] == selection["preferred_conditional"])
    preferred_oof_path = dirs["oof"] / "two_part_future_xg_oof.parquet"
    preferred_result["combined_oof"].to_parquet(preferred_oof_path, index=False)

    # Save selection summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classification_variant": args.classification_variant,
        "classification_oof_source": str(classification_oof_path),
        "aligned_rows": int(len(aligned_df)),
        "input_rows": int(len(df)),
        "selection": selection,
        "variant_results": [
            {
                "variant": r["variant"],
                "status": r["status"],
                "mae": r.get("hurdle_metrics", {}).get("mae", np.nan),
                "rmse": r.get("hurdle_metrics", {}).get("rmse", np.nan),
                "r2": r.get("hurdle_metrics", {}).get("r2", np.nan),
            }
            for r in results
        ]
    }

    summary_path = dirs["validation"] / "two_part_model_selection.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n✓ Training complete:")
    print(f"  Preferred conditional model: {selection['preferred_conditional']} (MAE={selection['preferred_conditional_mae']:.4f})")
    print(f"  Preferred OOF: {preferred_oof_path}")
    print(f"  Comparison table: {comparison_path}")
    print(f"  Results saved to: {summary_path}")


if __name__ == "__main__":
    main()



