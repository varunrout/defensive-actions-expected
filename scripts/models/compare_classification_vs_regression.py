"""Compare binary classification vs xT regression approaches.

This script loads both trained models and compares their predictions
on the same test data to evaluate which approach better captures
defensive threat dynamics.

Usage:
  python scripts/models/compare_classification_vs_regression.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import warnings
warnings.filterwarnings("ignore")

# Add src to path for imports
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_models(
    logistic_dir: Path, regression_dir: Path, variant: str = "v2_full_baseline"
) -> tuple[dict, dict]:
    """Load both logistic and regression models for the same variant."""
    logistic_path = logistic_dir / f"logistic_{variant}.joblib"
    regression_path = regression_dir / f"regression_{variant}.joblib"

    if not logistic_path.exists():
        raise FileNotFoundError(f"Missing logistic model: {logistic_path}")
    if not regression_path.exists():
        raise FileNotFoundError(f"Missing regression model: {regression_path}")

    logistic_model = joblib.load(logistic_path)
    regression_model = joblib.load(regression_path)

    return logistic_model, regression_model


def compare_predictions(
    df: pd.DataFrame,
    logistic_model: dict,
    regression_model: dict,
) -> dict[str, object]:
    """Compare predictions from both models on held-out data."""
    print("\nPreparing test data...")

    # Prepare features for logistic model
    categorical_cols = logistic_model["categorical"]
    numeric_cols = logistic_model["numeric"]
    feature_cols = [c for c in categorical_cols + numeric_cols if c in df.columns]
    X_test = df[feature_cols].copy()

    # Remove rows with missing targets or groups
    test_data = X_test.dropna(subset=feature_cols)
    if test_data.empty:
        raise ValueError("No valid test data after removing missing values")

    # Get predictions
    print(f"Test set size: {len(test_data):,} actions")

    logistic_pipe = logistic_model["pipeline"]
    regression_pipe = regression_model["pipeline"]

    # Logistic predictions (probabilities)
    p_shot = logistic_pipe.predict_proba(test_data)[:, 1]

    # Regression predictions (xT scores)
    threat_xt = regression_pipe.predict(test_data)

    # Also get binary predictions from regression via median split
    xt_median = threat_xt.median() if isinstance(threat_xt, pd.Series) else np.median(threat_xt)
    threat_binary = (threat_xt >= xt_median).astype(int)

    results = {
        "p_shot": p_shot,
        "threat_xt": threat_xt,
        "threat_binary": threat_binary,
        "xt_median": float(xt_median),
    }

    return results


def compute_correlations(results: dict[str, object]) -> dict[str, float]:
    """Compute correlations between prediction approaches."""
    p_shot = results["p_shot"]
    threat_xt = results["threat_xt"]
    threat_binary = results["threat_binary"]

    # Pearson correlation
    corr_pearson = np.corrcoef(p_shot, threat_xt)[0, 1]

    # Spearman correlation
    corr_spearman = spearmanr(p_shot, threat_xt).correlation

    # Agreement on binary splits (both high or both low)
    agreement = (p_shot >= 0.5) == threat_binary
    agreement_rate = agreement.mean()

    # Confusion matrix style
    tn = ((p_shot < 0.5) & (threat_binary == 0)).sum()
    tp = ((p_shot >= 0.5) & (threat_binary == 1)).sum()
    fn = ((p_shot < 0.5) & (threat_binary == 1)).sum()
    fp = ((p_shot >= 0.5) & (threat_binary == 0)).sum()

    return {
        "pearson_correlation": float(corr_pearson),
        "spearman_correlation": float(corr_spearman),
        "binary_agreement": float(agreement_rate),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare binary classification vs xT regression models"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"),
        help="Input parquet file for test data.",
    )
    parser.add_argument(
        "--logistic-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "baseline"),
        help="Directory containing trained logistic models.",
    )
    parser.add_argument(
        "--regression-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "regression"),
        help="Directory containing trained regression models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "comparison"),
        help="Directory to save comparison results.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="v2_full_baseline",
        help="Model variant to compare (v0, v1, v2, etc.).",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Optional row cap for quick testing"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    logistic_dir = Path(args.logistic_dir)
    regression_dir = Path(args.regression_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("COMPARE BINARY CLASSIFICATION vs xT REGRESSION")
    print("=" * 72)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    # Load data
    print(f"\nLoading test data from: {input_path}")
    df = pd.read_parquet(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    print(f"Test rows: {len(df):,}")

    # Load models
    print(f"\nLoading models (variant: {args.variant})...")
    try:
        logistic_model, regression_model = load_models(
            logistic_dir, regression_dir, args.variant
        )
        print("✓ Both models loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error loading models: {e}")
        print(f"  Make sure to train both models first:")
        print(f"    python scripts/models/train_baseline_logistic.py")
        print(f"    python scripts/models/train_baseline_regression.py")
        return

    # Compare predictions
    print(f"\nGenerating predictions...")
    try:
        results = compare_predictions(df, logistic_model, regression_model)
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return

    # Compute correlations
    print(f"\nComputing correlations...")
    correlations = compute_correlations(results)

    # Print results
    print("\n" + "=" * 72)
    print("COMPARISON RESULTS")
    print("=" * 72)
    print(f"\nCorrelation between P(shot) and E[xT]:")
    print(f"  Pearson:  {correlations['pearson_correlation']:.4f}")
    print(f"  Spearman: {correlations['spearman_correlation']:.4f}")

    print(f"\nBinary agreement (both high or both low):")
    print(f"  Agreement rate: {correlations['binary_agreement']:.1%}")
    print(f"  True Negatives:  {correlations['true_negatives']:,}")
    print(f"  True Positives:  {correlations['true_positives']:,}")
    print(f"  False Negatives: {correlations['false_negatives']:,}")
    print(f"  False Positives: {correlations['false_positives']:,}")

    # Load model metrics
    print(f"\n" + "=" * 72)
    print("MODEL METRICS COMPARISON")
    print("=" * 72)

    logistic_metrics_path = (
        Path(args.logistic_dir).parent.parent
        / "validation"
        / "baseline"
        / "baseline_model_metrics.json"
    )
    regression_metrics_path = (
        Path(args.regression_dir).parent.parent
        / "validation"
        / "regression"
        / "regression_model_metrics.json"
    )

    summary = {"comparison": correlations}

    if logistic_metrics_path.exists():
        with open(logistic_metrics_path) as f:
            logistic_metrics = json.load(f)
        variant_metrics = logistic_metrics["variants"].get(args.variant, {})
        print(f"\nLogistic ({args.variant}):")
        print(f"  ROC-AUC: {variant_metrics.get('roc_auc', 'N/A')}")
        print(f"  Avg Precision: {variant_metrics.get('avg_precision', 'N/A')}")
        summary["logistic_metrics"] = variant_metrics
    else:
        print(f"\n⚠ Logistic metrics not found: {logistic_metrics_path}")

    if regression_metrics_path.exists():
        with open(regression_metrics_path) as f:
            regression_metrics = json.load(f)
        variant_metrics = regression_metrics["variants"].get(args.variant, {})
        print(f"\nRegression ({args.variant}):")
        print(f"  R²: {variant_metrics.get('r2', 'N/A'):.4f}")
        print(f"  RMSE: {variant_metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAE: {variant_metrics.get('mae', 'N/A'):.4f}")
        print(f"  Spearman: {variant_metrics.get('spearman', 'N/A'):.4f}")
        summary["regression_metrics"] = variant_metrics
    else:
        print(f"\n⚠ Regression metrics not found: {regression_metrics_path}")

    # Save comparison report
    report_path = output_dir / f"comparison_{args.variant}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 72)
    print(f"Saved comparison report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

