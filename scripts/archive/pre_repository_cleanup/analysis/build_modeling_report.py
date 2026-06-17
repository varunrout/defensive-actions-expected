from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portfolio-level comparison report across all trained models."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"),
        help="Input parquet used for scoring cross-task alignment.",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=str(
            REPO_ROOT / "outputs" / "validation" / "baseline" / "baseline_model_metrics.json"
        ),
        help="Path to logistic metrics JSON.",
    )
    parser.add_argument(
        "--regression-metrics",
        type=str,
        default=str(
            REPO_ROOT / "outputs" / "validation" / "regression" / "regression_model_metrics.json"
        ),
        help="Path to regression metrics JSON.",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "baseline"),
        help="Directory with saved logistic models.",
    )
    parser.add_argument(
        "--regression-model-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "models" / "regression"),
        help="Directory with saved regression models.",
    )
    parser.add_argument(
        "--slice-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "comparison" / "slices_latest"),
        help="Directory with saved slice comparison CSVs.",
    )
    parser.add_argument(
        "--feature-stability-logistic",
        type=str,
        default=str(
            REPO_ROOT
            / "outputs"
            / "validation"
            / "analysis"
            / "feature_stability"
            / "logistic_v4_freeze_geometry"
        ),
        help="Directory with logistic feature stability outputs.",
    )
    parser.add_argument(
        "--feature-stability-regression",
        type=str,
        default=str(
            REPO_ROOT
            / "outputs"
            / "validation"
            / "analysis"
            / "feature_stability"
            / "regression_v3_context_enhanced"
        ),
        help="Directory with regression feature stability outputs.",
    )
    parser.add_argument(
        "--feature-selection-logistic",
        type=str,
        default=str(
            REPO_ROOT
            / "outputs"
            / "validation"
            / "analysis"
            / "feature_selection"
            / "logistic_v4_freeze_geometry"
        ),
        help="Directory with logistic clustering/PCA feature-selection outputs.",
    )
    parser.add_argument(
        "--feature-selection-regression",
        type=str,
        default=str(
            REPO_ROOT
            / "outputs"
            / "validation"
            / "analysis"
            / "feature_selection"
            / "regression_v4_freeze_geometry"
        ),
        help="Directory with regression clustering/PCA feature-selection outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "comparison" / "model_portfolio"),
        help="Directory where portfolio-level CSV/JSON artifacts will be saved.",
    )
    parser.add_argument(
        "--doc-path",
        type=str,
        default=str(REPO_ROOT / "docs" / "analysis" / "MODEL_PORTFOLIO_REPORT.md"),
        help="Markdown document to generate.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def variant_order(variant: str) -> int:
    try:
        return int(variant.split("_", 1)[0].replace("v", ""))
    except Exception:
        return 999


def variant_family(variant: str) -> str:
    if variant.startswith("v0"):
        return "phase-only"
    if variant.startswith("v1"):
        return "spatial"
    if variant.startswith("v2"):
        return "full baseline"
    if variant.startswith("v3"):
        return "context enhanced"
    if variant.startswith("v4"):
        return "freeze geometry"
    if variant.startswith("v5"):
        return "clustered interpretable"
    if variant.startswith("v6"):
        return "clustered balanced"
    if variant.startswith("v7"):
        return "clustered interpretable ridge"
    if variant.startswith("v8"):
        return "clustered balanced ridge"
    return "other"


def build_logistic_df(metrics: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, payload in metrics["variants"].items():
        rows.append(
            {
                "family": "logistic",
                "variant": variant,
                "variant_order": variant_order(variant),
                "stage": variant_family(variant),
                "model_type": "logistic_l2",
                "categorical_features": len(payload["categorical"]),
                "numeric_features": len(payload["numeric"]),
                "feature_count": len(payload["categorical"]) + len(payload["numeric"]),
                "roc_auc": payload["roc_auc"],
                "avg_precision": payload["avg_precision"],
                "model_path": rel(payload["model_path"]),
                "coefficients_path": rel(payload["coefficients_path"]),
                "fold_metrics_path": rel(payload["fold_metrics_path"]),
            }
        )
    return pd.DataFrame(rows).sort_values("variant_order").reset_index(drop=True)


def build_regression_df(metrics: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, payload in metrics["variants"].items():
        rows.append(
            {
                "family": "regression",
                "variant": variant,
                "variant_order": variant_order(variant),
                "stage": variant_family(variant),
                "model_type": payload["model_type"],
                "categorical_features": len(payload["categorical"]),
                "numeric_features": len(payload["numeric"]),
                "feature_count": len(payload["categorical"]) + len(payload["numeric"]),
                "r2": payload["r2"],
                "rmse": payload["rmse"],
                "mae": payload["mae"],
                "spearman": payload["spearman"],
                "model_path": rel(payload["model_path"]),
                "coefficients_path": rel(payload["coefficients_path"]),
                "fold_metrics_path": rel(payload["fold_metrics_path"]),
            }
        )
    return pd.DataFrame(rows).sort_values("variant_order").reset_index(drop=True)


def fold_summary(df: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        fold_df = pd.read_csv(REPO_ROOT / row["fold_metrics_path"])
        record = {
            "family": row["family"],
            "variant": row["variant"],
            "feature_count": row["feature_count"],
        }
        for metric in metric_cols:
            if metric in fold_df.columns:
                record[f"{metric}_mean"] = fold_df[metric].mean()
                record[f"{metric}_std"] = fold_df[metric].std(ddof=1)
                record[f"{metric}_min"] = fold_df[metric].min()
                record[f"{metric}_max"] = fold_df[metric].max()
        rows.append(record)
    return pd.DataFrame(rows)


def compute_cross_task_alignment(
    data_path: Path,
    logistic_df: pd.DataFrame,
    baseline_model_dir: Path,
    regression_model_dir: Path,
) -> pd.DataFrame:
    data = pd.read_parquet(data_path)
    rows: list[dict] = []

    for variant in logistic_df["variant"].tolist():
        logistic_path = baseline_model_dir / f"logistic_{variant}.joblib"
        regression_path = regression_model_dir / f"regression_{variant}.joblib"
        if not logistic_path.exists() or not regression_path.exists():
            continue

        logistic_model = joblib.load(logistic_path)
        regression_model = joblib.load(regression_path)
        feature_cols = list(
            dict.fromkeys(
                logistic_model["categorical"]
                + logistic_model["numeric"]
                + regression_model["categorical"]
                + regression_model["numeric"]
            )
        )
        feature_cols = [col for col in feature_cols if col in data.columns]
        scored = data.dropna(subset=feature_cols)
        if scored.empty:
            continue

        x = scored[feature_cols].copy()
        p_shot = logistic_model["pipeline"].predict_proba(x)[:, 1]
        threat_xt = regression_model["pipeline"].predict(x)
        xt_median = float(np.median(threat_xt))
        threat_binary = (threat_xt >= xt_median).astype(int)
        agreement = ((p_shot >= 0.5) == threat_binary).mean()

        rows.append(
            {
                "variant": variant,
                "n_scored": len(scored),
                "pearson_correlation": float(np.corrcoef(p_shot, threat_xt)[0, 1]),
                "spearman_correlation": float(spearmanr(p_shot, threat_xt).correlation),
                "binary_agreement": float(agreement),
                "xt_median": xt_median,
            }
        )

    return pd.DataFrame(rows).sort_values("variant", key=lambda s: s.map(variant_order)).reset_index(drop=True)


def format_value(value: object, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None, decimals: int = 4) -> str:
    rename = rename or {}
    headers = [rename.get(col, col) for col in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df[columns].iterrows():
        values = [format_value(row[col], decimals=decimals) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize_slice_wins(slice_path: Path, family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not slice_path.exists():
        empty = pd.DataFrame(columns=["variant", "slice_wins"])
        return empty, empty
    df = pd.read_csv(slice_path)
    wins = (
        df.groupby("variant").size().reset_index(name="slice_wins").sort_values("slice_wins", ascending=False)
    )
    by_dim = (
        df.groupby(["slice_col", "variant"]).size().reset_index(name="wins").sort_values(["slice_col", "wins"], ascending=[True, False])
    )
    by_dim["family"] = family
    return wins, by_dim


def top_clusters(cluster_path: Path, max_clusters: int = 6) -> pd.DataFrame:
    if not cluster_path.exists():
        return pd.DataFrame(columns=["cluster_id", "features", "cluster_size", "max_abs_corr_within_cluster"])
    clusters = pd.read_csv(cluster_path)
    summary = (
        clusters.groupby("cluster_id")
        .agg(
            features=("feature", lambda s: ", ".join(s.tolist())),
            cluster_size=("cluster_size", "max"),
            max_abs_corr_within_cluster=("max_abs_corr_within_cluster", "max"),
        )
        .reset_index()
        .sort_values(["cluster_size", "max_abs_corr_within_cluster"], ascending=[False, False])
        .head(max_clusters)
    )
    return summary


def selected_feature_count(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if "selected" in df.columns:
        return int(df["selected"].sum())
    return len(df)


def build_markdown(
    logistic_metrics: dict,
    regression_metrics: dict,
    logistic_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    logistic_direct_df: pd.DataFrame,
    regression_direct_df: pd.DataFrame,
    cross_task_df: pd.DataFrame,
    logistic_fold_df: pd.DataFrame,
    regression_fold_df: pd.DataFrame,
    logistic_slice_wins: pd.DataFrame,
    regression_slice_wins: pd.DataFrame,
    logistic_slice_by_dim: pd.DataFrame,
    regression_slice_by_dim: pd.DataFrame,
    logistic_stability_summary: dict,
    regression_stability_summary: dict,
    logistic_clusters: pd.DataFrame,
    regression_clusters: pd.DataFrame,
    logistic_fs_manifest: dict,
    regression_fs_manifest: dict,
    artifacts: dict[str, str],
) -> str:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    best_logistic = logistic_df.sort_values(["roc_auc", "avg_precision"], ascending=False).iloc[0]
    best_regression = regression_df.sort_values(["r2", "spearman"], ascending=False).iloc[0]
    best_logistic_direct = logistic_direct_df.sort_values(["roc_auc", "avg_precision"], ascending=False).iloc[0]
    best_regression_direct = regression_direct_df.sort_values(["r2", "spearman"], ascending=False).iloc[0]
    top_logistic_fold = logistic_fold_df.sort_values(
        ["roc_auc_mean", "avg_precision_mean"], ascending=False
    ).iloc[0]
    top_regression_fold = regression_fold_df.sort_values(["r2_mean", "rmse_mean"], ascending=[False, True]).iloc[0]
    top_cross_task = (
        cross_task_df.sort_values("spearman_correlation", ascending=False).iloc[0]
        if not cross_task_df.empty
        else None
    )

    ridge_delta_logistic = {
        "v5_vs_v7_auc": float(
            logistic_df.loc[logistic_df["variant"] == "v7_interpretable_ridge", "roc_auc"].iloc[0]
            - logistic_df.loc[logistic_df["variant"] == "v5_interpretable_clustered", "roc_auc"].iloc[0]
        ),
        "v6_vs_v8_auc": float(
            logistic_df.loc[logistic_df["variant"] == "v8_balanced_ridge", "roc_auc"].iloc[0]
            - logistic_df.loc[logistic_df["variant"] == "v6_balanced_clustered", "roc_auc"].iloc[0]
        ),
    }
    ridge_delta_regression = {
        "v5_vs_v7_r2": float(
            regression_df.loc[regression_df["variant"] == "v7_interpretable_ridge", "r2"].iloc[0]
            - regression_df.loc[regression_df["variant"] == "v5_interpretable_clustered", "r2"].iloc[0]
        ),
        "v6_vs_v8_r2": float(
            regression_df.loc[regression_df["variant"] == "v8_balanced_ridge", "r2"].iloc[0]
            - regression_df.loc[regression_df["variant"] == "v6_balanced_clustered", "r2"].iloc[0]
        ),
    }

    logistic_slice_leader = logistic_slice_wins.iloc[0] if not logistic_slice_wins.empty else None
    regression_slice_leader = regression_slice_wins.iloc[0] if not regression_slice_wins.empty else None

    logistic_top_dims = (
        logistic_slice_by_dim.groupby("slice_col").first().reset_index()[["slice_col", "variant", "wins"]]
        if not logistic_slice_by_dim.empty
        else pd.DataFrame(columns=["slice_col", "variant", "wins"])
    )
    regression_top_dims = (
        regression_slice_by_dim.groupby("slice_col").first().reset_index()[["slice_col", "variant", "wins"]]
        if not regression_slice_by_dim.empty
        else pd.DataFrame(columns=["slice_col", "variant", "wins"])
    )

    sections: list[str] = []
    sections.append("# Model Portfolio Report")
    sections.append("")
    sections.append(f"Generated: {created_at}")
    sections.append("")
    sections.append(
        "> Methodology note: this report summarizes the metric artifacts supplied to the script. It must not be read as current corrected-model evidence until the full pipeline has been rerun and the models have been retrained."
    )
    sections.append("")
    sections.append("## Executive Summary")
    sections.append("")
    sections.append(
        f"- **Dataset / validation context:** {logistic_metrics['rows']:,} defensive actions across {logistic_metrics['matches']} matches; grouped CV by `match_id`; logistic target base rate {logistic_metrics['target_rate']:.2%}; regression target mean {regression_metrics['target_mean']:.4f}."
    )
    sections.append(
        f"- **Top OOF classification model in supplied metrics:** `{best_logistic['variant']}` with ROC-AUC {best_logistic['roc_auc']:.4f} and AP {best_logistic['avg_precision']:.4f}."
    )
    sections.append(
        f"- **Top OOF regression model in supplied metrics:** `{best_regression['variant']}` with R2 {best_regression['r2']:.4f}, RMSE {best_regression['rmse']:.4f}, MAE {best_regression['mae']:.4f}, and Spearman {best_regression['spearman']:.4f}."
    )
    sections.append(
        f"- **Top direct-score / slice-analysis model in supplied metrics:** classification `{best_logistic_direct['variant']}` and regression `{best_regression_direct['variant']}`. These are useful for relative segment behavior but are more optimistic than OOF CV because they score saved models on the full dataset."
    )
    sections.append(
        "- **Feature-selection context:** compact and interpretable variants should be compared from regenerated metrics before any current recommendation is made."
    )
    sections.append(
        f"- **Regularization takeaway:** on filtered feature sets, Ridge had almost no portfolio-level lift (`v5`→`v7` logistic ΔAUC {ridge_delta_logistic['v5_vs_v7_auc']:+.5f}; `v6`→`v8` logistic ΔAUC {ridge_delta_logistic['v6_vs_v8_auc']:+.5f}; `v5`→`v7` regression ΔR2 {ridge_delta_regression['v5_vs_v7_r2']:+.5f}; `v6`→`v8` regression ΔR2 {ridge_delta_regression['v6_vs_v8_r2']:+.5f})."
    )

    sections.append("")
    sections.append("## 1. Validation Protocol")
    sections.append("")
    sections.append("- **Classification family:** logistic regression with L2 penalty predicting `target_future_shot_10s`.")
    sections.append("- **Regression family:** linear or Ridge regression predicting `target_future_xg_10s`.")
    sections.append("- **Split strategy:** `GroupKFold` by `match_id` to prevent leakage across events from the same match.")
    sections.append("- **Primary metrics:** ROC-AUC / Average Precision for classification; R2 / RMSE / MAE / Spearman for regression.")
    sections.append("- **Secondary analyses:** slice leaderboards, cross-task alignment, correlation clustering, tactical-vs-proxy ablations, and feature-selection manifests for clustered variants.")

    sections.append("")
    sections.append("## 2. OOF Leaderboard — Classification")
    sections.append("")
    logistic_table = logistic_df.copy()
    logistic_table["delta_auc_vs_best"] = logistic_table["roc_auc"] - best_logistic["roc_auc"]
    logistic_table["delta_ap_vs_best"] = logistic_table["avg_precision"] - best_logistic["avg_precision"]
    sections.append(
        markdown_table(
            logistic_table,
            ["variant", "stage", "feature_count", "roc_auc", "avg_precision", "delta_auc_vs_best", "delta_ap_vs_best"],
            rename={
                "variant": "Variant",
                "stage": "Feature set",
                "feature_count": "Features",
                "roc_auc": "ROC-AUC",
                "avg_precision": "AP",
                "delta_auc_vs_best": "ΔAUC vs best",
                "delta_ap_vs_best": "ΔAP vs best",
            },
        )
    )
    sections.append("")
    sections.append(
        f"**Readout:** In the supplied OOF classification metrics, `{best_logistic['variant']}` ranks first by ROC-AUC and average precision. Treat this as artifact-specific until corrected models are retrained."
    )

    sections.append("")
    sections.append("## 3. OOF Leaderboard — Regression")
    sections.append("")
    regression_table = regression_df.copy()
    regression_table["delta_r2_vs_best"] = regression_table["r2"] - best_regression["r2"]
    regression_table["delta_spearman_vs_best"] = regression_table["spearman"] - best_regression["spearman"]
    sections.append(
        markdown_table(
            regression_table,
            ["variant", "stage", "model_type", "feature_count", "r2", "rmse", "mae", "spearman", "delta_r2_vs_best"],
            rename={
                "variant": "Variant",
                "stage": "Feature set",
                "model_type": "Model",
                "feature_count": "Features",
                "r2": "R2",
                "rmse": "RMSE",
                "mae": "MAE",
                "spearman": "Spearman",
                "delta_r2_vs_best": "ΔR2 vs best",
            },
        )
    )
    sections.append("")
    sections.append(
        f"**Readout:** In the supplied OOF regression metrics, `{best_regression['variant']}` ranks first by R2 and Spearman. Treat this as artifact-specific until corrected models are retrained."
    )

    sections.append("")
    sections.append("## 4. Direct-Score Leaderboard and Slice Caveat")
    sections.append("")
    sections.append(
        "The slice-analysis pipeline scores saved models on the full dataset, so the absolute values below are more optimistic than OOF CV. Use them to compare relative behavior across segments rather than to select the final model alone."
    )
    sections.append("")
    sections.append("### Classification direct-score leaderboard")
    sections.append("")
    sections.append(
        markdown_table(
            logistic_direct_df,
            ["variant", "roc_auc", "avg_precision"],
            rename={"variant": "Variant", "roc_auc": "ROC-AUC", "avg_precision": "AP"},
        )
    )
    sections.append("")
    sections.append("### Regression direct-score leaderboard")
    sections.append("")
    sections.append(
        markdown_table(
            regression_direct_df,
            ["variant", "r2", "rmse", "mae", "spearman"],
            rename={"variant": "Variant", "r2": "R2", "rmse": "RMSE", "mae": "MAE", "spearman": "Spearman"},
        )
    )
    sections.append("")
    sections.append(
        f"**Readout:** In the supplied direct-score files, `{best_logistic_direct['variant']}` leads classification and `{best_regression_direct['variant']}` leads regression. Direct scores are segment diagnostics, not deployment evidence."
    )

    sections.append("")
    sections.append("## 5. Fold Stability")
    sections.append("")
    logistic_fold_view = logistic_fold_df[logistic_fold_df["variant"].isin(["v2_full_baseline", "v3_context_enhanced", "v4_freeze_geometry", "v6_balanced_clustered", "v8_balanced_ridge"])].copy()
    regression_fold_view = regression_fold_df[regression_fold_df["variant"].isin(["v2_full_baseline", "v3_context_enhanced", "v4_freeze_geometry", "v6_balanced_clustered", "v8_balanced_ridge"])].copy()
    sections.append("### Classification fold spread")
    sections.append("")
    sections.append(
        markdown_table(
            logistic_fold_view,
            ["variant", "roc_auc_mean", "roc_auc_std", "avg_precision_mean", "avg_precision_std"],
            rename={
                "variant": "Variant",
                "roc_auc_mean": "Mean AUC",
                "roc_auc_std": "AUC SD",
                "avg_precision_mean": "Mean AP",
                "avg_precision_std": "AP SD",
            },
        )
    )
    sections.append("")
    sections.append("### Regression fold spread")
    sections.append("")
    sections.append(
        markdown_table(
            regression_fold_view,
            ["variant", "r2_mean", "r2_std", "rmse_mean", "rmse_std"],
            rename={
                "variant": "Variant",
                "r2_mean": "Mean R2",
                "r2_std": "R2 SD",
                "rmse_mean": "Mean RMSE",
                "rmse_std": "RMSE SD",
            },
        )
    )
    sections.append("")
    sections.append(
        f"**Readout:** The supplied fold summaries rank `{top_logistic_fold['variant']}` first for classification mean AUC/AP and `{top_regression_fold['variant']}` first for regression mean R2/RMSE."
    )

    sections.append("")
    sections.append("## 6. Cross-Task Alignment (P(shot) vs E[future xG))")
    sections.append("")
    sections.append(
        markdown_table(
            cross_task_df,
            ["variant", "pearson_correlation", "spearman_correlation", "binary_agreement", "n_scored"],
            rename={
                "variant": "Variant",
                "pearson_correlation": "Pearson",
                "spearman_correlation": "Spearman",
                "binary_agreement": "Agreement",
                "n_scored": "Rows scored",
            },
        )
    )
    sections.append("")
    if top_cross_task is not None:
        sections.append(
            f"**Readout:** `{top_cross_task['variant']}` has the highest supplied cross-task Spearman correlation ({top_cross_task['spearman_correlation']:.4f}). Use this as an alignment diagnostic, not a ranking of model quality."
        )

    sections.append("")
    sections.append("## 7. Slice Analysis")
    sections.append("")
    if logistic_slice_leader is not None and regression_slice_leader is not None:
        sections.append(
            f"- **Slice-win leader (classification):** `{logistic_slice_leader['variant']}` with {int(logistic_slice_leader['slice_wins'])} best-segment wins."
        )
        sections.append(
            f"- **Slice-win leader (regression):** `{regression_slice_leader['variant']}` with {int(regression_slice_leader['slice_wins'])} best-segment wins."
        )
    sections.append("")
    sections.append("### Slice-win counts")
    sections.append("")
    sections.append(
        markdown_table(
            logistic_slice_wins,
            ["variant", "slice_wins"],
            rename={"variant": "Classification variant", "slice_wins": "Wins"},
            decimals=0,
        )
    )
    sections.append("")
    sections.append(
        markdown_table(
            regression_slice_wins,
            ["variant", "slice_wins"],
            rename={"variant": "Regression variant", "slice_wins": "Wins"},
            decimals=0,
        )
    )
    sections.append("")
    sections.append("### Dominant winner by slice dimension")
    sections.append("")
    sections.append(
        markdown_table(
            logistic_top_dims,
            ["slice_col", "variant", "wins"],
            rename={"slice_col": "Dimension", "variant": "Classification winner", "wins": "Wins"},
            decimals=0,
        )
    )
    sections.append("")
    sections.append(
        markdown_table(
            regression_top_dims,
            ["slice_col", "variant", "wins"],
            rename={"slice_col": "Dimension", "variant": "Regression winner", "wins": "Wins"},
            decimals=0,
        )
    )
    sections.append("")
    sections.append(
        "**Readout:** Slice-win tables show which variants lead each supplied segment file. Because slice scores are not OOF, they should be used as qualitative diagnostics."
    )

    sections.append("")
    sections.append("## 8. Correlation Clusters and Coefficient Stability")
    sections.append("")
    sections.append("### Logistic stability study (`v4_freeze_geometry`) — top correlation clusters")
    sections.append("")
    sections.append(
        markdown_table(
            logistic_clusters,
            ["cluster_id", "cluster_size", "max_abs_corr_within_cluster", "features"],
            rename={
                "cluster_id": "Cluster",
                "cluster_size": "Size",
                "max_abs_corr_within_cluster": "Max abs corr",
                "features": "Features",
            },
        )
    )
    sections.append("")
    sections.append("### Regression stability study (`v3_context_enhanced`) — top correlation clusters")
    sections.append("")
    sections.append(
        markdown_table(
            regression_clusters,
            ["cluster_id", "cluster_size", "max_abs_corr_within_cluster", "features"],
            rename={
                "cluster_id": "Cluster",
                "cluster_size": "Size",
                "max_abs_corr_within_cluster": "Max abs corr",
                "features": "Features",
            },
        )
    )
    sections.append("")
    logistic_ablation = pd.read_csv(Path(logistic_stability_summary["outputs"]["tactical_vs_proxy_ablation"]))
    regression_ablation = pd.read_csv(Path(regression_stability_summary["outputs"]["tactical_vs_proxy_ablation"]))
    logistic_tactical_drop = logistic_ablation.query("scenario == 'tactical_only'")["delta_vs_full_primary"].iloc[0]
    logistic_proxy_drop = logistic_ablation.query("scenario == 'proxy_only'")["delta_vs_full_primary"].iloc[0]
    regression_tactical_drop = regression_ablation.query("scenario == 'tactical_only'")["delta_vs_full_primary"].iloc[0]
    regression_proxy_drop = regression_ablation.query("scenario == 'proxy_only'")["delta_vs_full_primary"].iloc[0]
    sections.append(
        f"- Logistic tactical-only ablation drops AUC by {logistic_tactical_drop:.4f}; proxy-only drops AUC by {logistic_proxy_drop:.4f}."
    )
    sections.append(
        f"- Regression tactical-only ablation drops R2 by {regression_tactical_drop:.4f}; proxy-only drops R2 by {regression_proxy_drop:.4f}."
    )
    sections.append(
        "- Interpretation: possession-time and progression proxies matter, but they are not the core source of signal. They can inflate coefficients if left unconstrained, especially in linear models with many correlated inputs."
    )

    sections.append("")
    sections.append("## 9. Feature-Selection Outcomes (Clustering / PCA)")
    sections.append("")
    sections.append(
        f"- Logistic `v5` manifest keeps **{logistic_fs_manifest['v5']['total_features']}** features; `v6` keeps **{logistic_fs_manifest['v6']['total_features']}**."
    )
    sections.append(
        f"- Regression `v5` manifest keeps **{regression_fs_manifest['v5']['total_features']}** features; `v6` keeps **{regression_fs_manifest['v6']['total_features']}**."
    )
    sections.append(
        f"- Stability-selected logistic interpretable set marks **{selected_feature_count(Path(logistic_stability_summary['outputs']['interpretable_feature_set_recommended']))}** raw features as keepers."
    )
    sections.append(
        f"- Stability-selected regression interpretable set marks **{selected_feature_count(Path(regression_stability_summary['outputs']['interpretable_feature_set_recommended']))}** raw features as keepers."
    )
    sections.append("")
    sections.append("**Readout:**")
    sections.append("")
    sections.append("1. Correlation and feature-selection artifacts identify candidate redundancy bundles for reviewer inspection.")
    sections.append("2. Compact and interpretable variants should be re-evaluated after regenerated corrected metrics are available.")
    sections.append("3. Ridge deltas above are computed from the supplied metrics and should not be presented as current findings until retraining is complete.")

    sections.append("")
    sections.append("## 10. Variant Feature Set Notes")
    sections.append("")
    sections.append("- **`v0_phase_only`**: phase taxonomy only.")
    sections.append("- **`v1_spatial`**: phase, action category, role group, and basic action geometry.")
    sections.append("- **`v2_full_baseline`**: adds generated support, visibility, local balance, and possession-context features.")
    sections.append("- **`v3_context_enhanced`**: adds previous-phase, event-order, ball-location, and phase-transition context.")
    sections.append("- **`v4_freeze_geometry`**: adds player position and generated freeze-frame centroid geometry.")
    sections.append("- **`v5_interpretable_clustered`**: compact feature-governed linear set.")
    sections.append("- **`v6_balanced_clustered`**: compact feature-governed set with additional categorical context.")
    sections.append("- **`v7_interpretable_ridge`**: Ridge version of `v5`.")
    sections.append("- **`v8_balanced_ridge`**: Ridge version of `v6`.")

    sections.append("")
    sections.append("## 11. Metric-Derived Leaders")
    sections.append("")
    sections.append("### Supplied artifact leaders")
    sections.append("")
    sections.append(f"- **OOF classification leader:** `{best_logistic['variant']}`.")
    sections.append(f"- **OOF regression leader:** `{best_regression['variant']}`.")
    sections.append(f"- **Direct-score classification leader:** `{best_logistic_direct['variant']}`.")
    sections.append(f"- **Direct-score regression leader:** `{best_regression_direct['variant']}`.")
    sections.append("")
    sections.append("### Regeneration framing")
    sections.append("")
    sections.append("1. Regenerate the player defensive-action dataset with the corrected targets and canonical feature schema.")
    sections.append("2. Retrain all logistic and regression variants from the regenerated dataset.")
    sections.append("3. Rebuild this report from the regenerated metrics before making current model-selection claims.")

    sections.append("")
    sections.append("## 12. Caveats")
    sections.append("")
    sections.append("- Slice leaderboards are not OOF; they score saved models on the same population used for fit aggregation, so they are best used qualitatively.")
    sections.append("- Regression MAPE is not decision-useful here because the target distribution includes near-zero values; prioritize R2, RMSE, MAE, and Spearman instead.")
    sections.append("- Coefficients remain associative, not causal. Strong coefficients on possession-lifecycle variables can reflect timing/progression context rather than coachable tactical mechanisms.")
    sections.append("- The feature-stability studies were run on `logistic_v4_freeze_geometry` and `regression_v3_context_enhanced`; that is appropriate because they are the main candidates, but it is not a guarantee that every smaller variant behaves identically.")

    sections.append("")
    sections.append("## 13. Artifact Index")
    sections.append("")
    sections.append("### Generated by `build_modeling_report.py`")
    sections.append("")
    for name, path in artifacts.items():
        sections.append(f"- `{path}` — {name.replace('_', ' ')}")
    sections.append("")
    sections.append("### Upstream validation artifacts referenced")
    sections.append("")
    sections.append(f"- `{rel(logistic_metrics['oof_path'])}`")
    sections.append(f"- `{rel(regression_metrics['oof_path'])}`")
    sections.append(f"- `{rel(logistic_stability_summary['outputs']['tactical_vs_proxy_ablation'])}`")
    sections.append(f"- `{rel(regression_stability_summary['outputs']['tactical_vs_proxy_ablation'])}`")
    sections.append(f"- `{rel(logistic_fs_manifest['artifacts']['corr_heatmap'])}`")
    sections.append(f"- `{rel(regression_fs_manifest['artifacts']['corr_heatmap'])}`")
    sections.append(f"- `{artifacts['slice_summary_json']}`")

    return "\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()

    data_path = Path(args.input)
    baseline_metrics_path = Path(args.baseline_metrics)
    regression_metrics_path = Path(args.regression_metrics)
    baseline_model_dir = Path(args.baseline_model_dir)
    regression_model_dir = Path(args.regression_model_dir)
    slice_dir = Path(args.slice_dir)
    feature_stability_logistic_dir = Path(args.feature_stability_logistic)
    feature_stability_regression_dir = Path(args.feature_stability_regression)
    feature_selection_logistic_dir = Path(args.feature_selection_logistic)
    feature_selection_regression_dir = Path(args.feature_selection_regression)
    output_dir = Path(args.output_dir)
    doc_path = Path(args.doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    logistic_metrics = load_json(baseline_metrics_path)
    regression_metrics = load_json(regression_metrics_path)
    logistic_df = build_logistic_df(logistic_metrics)
    regression_df = build_regression_df(regression_metrics)

    logistic_fold_df = fold_summary(logistic_df, ["roc_auc", "avg_precision"])
    regression_fold_df = fold_summary(regression_df, ["r2", "rmse", "mae", "mape"])

    logistic_direct_df = pd.read_csv(slice_dir / "leaderboard_logistic_overall.csv")
    regression_direct_df = pd.read_csv(slice_dir / "leaderboard_regression_overall.csv")
    cross_task_df = compute_cross_task_alignment(
        data_path=data_path,
        logistic_df=logistic_df,
        baseline_model_dir=baseline_model_dir,
        regression_model_dir=regression_model_dir,
    )

    logistic_slice_wins, logistic_slice_by_dim = summarize_slice_wins(
        slice_dir / "best_variant_by_slice_logistic.csv", "logistic"
    )
    regression_slice_wins, regression_slice_by_dim = summarize_slice_wins(
        slice_dir / "best_variant_by_slice_regression.csv", "regression"
    )

    logistic_stability_summary = load_json(feature_stability_logistic_dir / "summary.json")
    regression_stability_summary = load_json(feature_stability_regression_dir / "summary.json")
    logistic_clusters = top_clusters(feature_stability_logistic_dir / "correlation_clusters.csv")
    regression_clusters = top_clusters(feature_stability_regression_dir / "correlation_clusters.csv")
    logistic_fs_manifest = load_json(feature_selection_logistic_dir / "feature_selection_manifest.json")
    regression_fs_manifest = load_json(feature_selection_regression_dir / "feature_selection_manifest.json")

    combined_leaderboard_path = output_dir / "model_leaderboard_all.csv"
    cross_task_path = output_dir / "cross_task_alignment.csv"
    fold_summary_path = output_dir / "fold_stability_summary.csv"
    slice_wins_path = output_dir / "slice_win_counts.csv"
    summary_json_path = output_dir / "model_portfolio_summary.json"

    combined = pd.concat([logistic_df, regression_df], ignore_index=True, sort=False)
    combined.to_csv(combined_leaderboard_path, index=False)
    cross_task_df.to_csv(cross_task_path, index=False)
    pd.concat([logistic_fold_df, regression_fold_df], ignore_index=True, sort=False).to_csv(
        fold_summary_path, index=False
    )
    pd.concat(
        [
            logistic_slice_wins.assign(family="logistic"),
            regression_slice_wins.assign(family="regression"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(slice_wins_path, index=False)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_rows": logistic_metrics["rows"],
        "matches": logistic_metrics["matches"],
        "top_oof_logistic_variant": logistic_df.sort_values(["roc_auc", "avg_precision"], ascending=False).iloc[0]["variant"],
        "top_oof_regression_variant": regression_df.sort_values(["r2", "spearman"], ascending=False).iloc[0]["variant"],
        "top_direct_logistic_variant": logistic_direct_df.sort_values(["roc_auc", "avg_precision"], ascending=False).iloc[0]["variant"],
        "top_direct_regression_variant": regression_direct_df.sort_values(["r2", "spearman"], ascending=False).iloc[0]["variant"],
        "artifact_paths": {
            "model_leaderboard_all_csv": rel(combined_leaderboard_path),
            "cross_task_alignment_csv": rel(cross_task_path),
            "fold_stability_summary_csv": rel(fold_summary_path),
            "slice_win_counts_csv": rel(slice_wins_path),
            "report_markdown": rel(doc_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    artifacts = {
        "model_leaderboard_all_csv": rel(combined_leaderboard_path),
        "cross_task_alignment_csv": rel(cross_task_path),
        "fold_stability_summary_csv": rel(fold_summary_path),
        "slice_win_counts_csv": rel(slice_wins_path),
        "model_portfolio_summary_json": rel(summary_json_path),
        "slice_summary_json": rel(slice_dir / "summary.json"),
    }
    markdown = build_markdown(
        logistic_metrics=logistic_metrics,
        regression_metrics=regression_metrics,
        logistic_df=logistic_df,
        regression_df=regression_df,
        logistic_direct_df=logistic_direct_df,
        regression_direct_df=regression_direct_df,
        cross_task_df=cross_task_df,
        logistic_fold_df=logistic_fold_df,
        regression_fold_df=regression_fold_df,
        logistic_slice_wins=logistic_slice_wins,
        regression_slice_wins=regression_slice_wins,
        logistic_slice_by_dim=logistic_slice_by_dim,
        regression_slice_by_dim=regression_slice_by_dim,
        logistic_stability_summary=logistic_stability_summary,
        regression_stability_summary=regression_stability_summary,
        logistic_clusters=logistic_clusters,
        regression_clusters=regression_clusters,
        logistic_fs_manifest=logistic_fs_manifest,
        regression_fs_manifest=regression_fs_manifest,
        artifacts=artifacts,
    )
    doc_path.write_text(markdown, encoding="utf-8")

    print("Saved portfolio leaderboard:", combined_leaderboard_path)
    print("Saved cross-task alignment:", cross_task_path)
    print("Saved fold stability summary:", fold_summary_path)
    print("Saved slice win counts:", slice_wins_path)
    print("Saved summary JSON:", summary_json_path)
    print("Saved report:", doc_path)


if __name__ == "__main__":
    main()


