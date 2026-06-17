"""Generate a feature stability report for DAx baseline variants.

Outputs:
1) correlation clusters (numeric features)
2) fold-by-fold coefficient sign stability
3) tactical vs proxy ablation table
4) recommended interpretable feature set CSV
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dax.models.baseline_logistic import (  # noqa: E402
    GROUP_COL as LOG_GROUP_COL,
    TARGET_COL as LOG_TARGET_COL,
    VariantSpec,
    build_pipeline,
    default_variant_specs,
    grouped_cv_scores,
    prepare_xyg,
    resolve_columns,
)
from dax.models.baseline_regression import (  # noqa: E402
    GROUP_COL as REG_GROUP_COL,
    TARGET_COL as REG_TARGET_COL,
    RegressionVariantSpec,
    build_regression_pipeline,
    default_regression_specs,
    grouped_cv_scores_regression,
    prepare_xyg_regression,
    resolve_columns as resolve_reg_columns,
)

# Explicit proxy candidates we usually do not want to over-weight tactically.
PROXY_COLUMNS = {
    "possession_event_count_total",
    "possession_duration_total",
    "seconds_since_possession_start",
    "event_order_in_possession",
    "phase_transition_count_so_far",
    "play_pattern",
    "position_group",
}


@dataclass
class ReportConfig:
    task: str
    variant: str
    n_splits: int
    corr_threshold: float
    min_sign_consistency: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature stability diagnostics for baseline DAx models")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "features" / "player_defensive_actions.parquet"),
        help="Input parquet dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["logistic", "regression"],
        default="logistic",
        help="Model family used for coefficient diagnostics.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="v4_freeze_geometry",
        help="Variant name from the selected model family.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="GroupKFold split count.")
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.8,
        help="Absolute Pearson threshold for correlation clustering.",
    )
    parser.add_argument(
        "--min-sign-consistency",
        type=float,
        default=0.8,
        help="Minimum sign consistency to include a feature in the recommended set.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for fast iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "analysis" / "feature_stability"),
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def _get_spec(task: str, variant: str, df: pd.DataFrame) -> tuple[Any, str, str]:
    if task == "logistic":
        specs = {s.name: s for s in default_variant_specs()}
        if variant not in specs:
            raise ValueError(f"Unknown logistic variant: {variant}")
        spec = resolve_columns(df, specs[variant])
        return spec, LOG_TARGET_COL, LOG_GROUP_COL

    specs = {s.name: s for s in default_regression_specs()}
    if variant not in specs:
        raise ValueError(f"Unknown regression variant: {variant}")
    spec = resolve_reg_columns(df, specs[variant])
    return spec, REG_TARGET_COL, REG_GROUP_COL


def _numeric_correlation_clusters(df: pd.DataFrame, numeric_cols: list[str], threshold: float) -> pd.DataFrame:
    usable = [c for c in numeric_cols if c in df.columns]
    if len(usable) < 2:
        return pd.DataFrame(columns=["cluster_id", "feature", "cluster_size", "max_abs_corr_within_cluster"])

    num_df = df[usable].apply(pd.to_numeric, errors="coerce")
    medians = num_df.median(numeric_only=True)
    num_df = num_df.fillna(medians)
    corr = num_df.corr().abs().fillna(0.0)

    visited: set[str] = set()
    clusters: list[list[str]] = []

    for col in corr.columns:
        if col in visited:
            continue
        stack = [col]
        comp: set[str] = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            neighbors = corr.index[corr.loc[node] >= threshold].tolist()
            for n in neighbors:
                if n not in comp:
                    stack.append(n)
        visited.update(comp)
        if len(comp) > 1:
            clusters.append(sorted(comp))

    rows: list[dict[str, Any]] = []
    for idx, cluster in enumerate(clusters, start=1):
        sub = corr.loc[cluster, cluster]
        sub_values = sub.to_numpy(copy=True)
        np.fill_diagonal(sub_values, np.nan)
        max_corr = float(np.nanmax(sub_values)) if sub_values.size else np.nan
        for feature in cluster:
            rows.append(
                {
                    "cluster_id": idx,
                    "feature": feature,
                    "cluster_size": len(cluster),
                    "max_abs_corr_within_cluster": max_corr,
                }
            )
    return pd.DataFrame(rows).sort_values(["cluster_id", "feature"]).reset_index(drop=True)


def _prepare_xyg_for_task(df: pd.DataFrame, spec: Any, task: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if task == "logistic":
        return prepare_xyg(df, spec)
    return prepare_xyg_regression(df, spec)


def _build_pipeline_for_task(spec: Any, task: str):
    if task == "logistic":
        return build_pipeline(spec)
    return build_regression_pipeline(spec)


def _cv_metrics_for_task(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline,
    task: str,
    n_splits: int,
) -> dict[str, float]:
    if task == "logistic":
        cv = grouped_cv_scores(x=x, y=y, groups=groups, pipeline=pipeline, n_splits=n_splits)
        return {
            "primary_metric": float(cv["roc_auc"]),
            "secondary_metric": float(cv["avg_precision"]),
            "primary_metric_name": "roc_auc",
            "secondary_metric_name": "avg_precision",
        }

    cv = grouped_cv_scores_regression(x=x, y=y, groups=groups, pipeline=pipeline, n_splits=n_splits)
    return {
        "primary_metric": float(cv["r2"]),
        "secondary_metric": float(cv["spearman"]),
        "primary_metric_name": "r2",
        "secondary_metric_name": "spearman",
    }


def _fold_coefficients(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, spec: Any, task: str, n_splits: int) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=n_splits)
    rows: list[pd.DataFrame] = []

    for fold, (train_idx, _) in enumerate(gkf.split(x, y, groups), start=1):
        pipe = _build_pipeline_for_task(spec, task)
        pipe.fit(x.iloc[train_idx], y[train_idx])
        feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
        model = pipe.named_steps["model"]
        coef = model.coef_[0] if task == "logistic" else model.coef_

        fold_df = pd.DataFrame({"feature": feature_names, "coef": np.asarray(coef, dtype=float)})
        fold_df["fold"] = fold
        rows.append(fold_df)

    return pd.concat(rows, ignore_index=True)


def _sign_stability(fold_coefs: pd.DataFrame, n_splits: int, zero_eps: float = 1e-8) -> pd.DataFrame:
    work = fold_coefs.copy()
    work["sign"] = 0
    work.loc[work["coef"] > zero_eps, "sign"] = 1
    work.loc[work["coef"] < -zero_eps, "sign"] = -1

    out_rows: list[dict[str, Any]] = []
    for feature, part in work.groupby("feature"):
        pos = int((part["sign"] == 1).sum())
        neg = int((part["sign"] == -1).sum())
        zero = int((part["sign"] == 0).sum())
        nz = pos + neg
        consistency = float(max(pos, neg) / nz) if nz > 0 else np.nan

        out_rows.append(
            {
                "feature": feature,
                "mean_coef": float(part["coef"].mean()),
                "std_coef": float(part["coef"].std(ddof=0)),
                "abs_mean_coef": float(part["coef"].abs().mean()),
                "pos_folds": pos,
                "neg_folds": neg,
                "zero_folds": zero,
                "nonzero_folds": nz,
                "sign_consistency": consistency,
                "fold_coverage": float(len(part) / n_splits),
            }
        )

    return pd.DataFrame(out_rows).sort_values(["sign_consistency", "abs_mean_coef"], ascending=[False, False]).reset_index(drop=True)


def _is_proxy_raw_feature(raw_feature: str) -> bool:
    return raw_feature in PROXY_COLUMNS


def _raw_feature_from_encoded(feature: str, numeric_cols: list[str], categorical_cols: list[str]) -> str:
    """Map transformed feature names back to raw feature names.

    For one-hot columns, prefer longest matching categorical prefix to handle
    names like `position_group_center_back` correctly.
    """
    if feature.startswith("num__"):
        return feature[5:]
    if feature.startswith("cat__"):
        rest = feature[5:]
        for cat_col in sorted(categorical_cols, key=len, reverse=True):
            prefix = f"{cat_col}_"
            if rest == cat_col or rest.startswith(prefix):
                return cat_col
        return rest
    return feature


def _tactical_proxy_ablation(df: pd.DataFrame, spec: Any, task: str, n_splits: int) -> pd.DataFrame:
    all_cat = [c for c in spec.categorical if c in df.columns]
    all_num = [n for n in spec.numeric if n in df.columns]

    proxy_cat = [c for c in all_cat if _is_proxy_raw_feature(c)]
    proxy_num = [n for n in all_num if _is_proxy_raw_feature(n)]

    tactical_cat = [c for c in all_cat if c not in proxy_cat]
    tactical_num = [n for n in all_num if n not in proxy_num]

    scenarios: list[tuple[str, list[str], list[str]]] = [
        ("full", all_cat, all_num),
        ("tactical_only", tactical_cat, tactical_num),
        ("proxy_only", proxy_cat, proxy_num),
    ]

    rows: list[dict[str, Any]] = []
    for label, cat_cols, num_cols in scenarios:
        if not cat_cols and not num_cols:
            rows.append(
                {
                    "scenario": label,
                    "categorical_features": 0,
                    "numeric_features": 0,
                    "total_features": 0,
                    "primary_metric": np.nan,
                    "secondary_metric": np.nan,
                    "primary_metric_name": "",
                    "secondary_metric_name": "",
                }
            )
            continue

        if task == "logistic":
            current_spec = VariantSpec(name=f"{spec.name}_{label}", categorical=cat_cols, numeric=num_cols)
        else:
            current_spec = RegressionVariantSpec(
                name=f"{spec.name}_{label}",
                model_type=spec.model_type,
                categorical=cat_cols,
                numeric=num_cols,
                alpha=spec.alpha,
            )

        x, y, groups = _prepare_xyg_for_task(df, current_spec, task)
        pipe = _build_pipeline_for_task(current_spec, task)
        metrics = _cv_metrics_for_task(x, y, groups, pipe, task, n_splits)

        rows.append(
            {
                "scenario": label,
                "categorical_features": len(cat_cols),
                "numeric_features": len(num_cols),
                "total_features": len(cat_cols) + len(num_cols),
                **metrics,
            }
        )

    out = pd.DataFrame(rows)
    if "full" in set(out["scenario"]):
        base = out.loc[out["scenario"] == "full", "primary_metric"].iloc[0]
        out["delta_vs_full_primary"] = out["primary_metric"] - base
    return out


def _recommended_feature_set(
    spec: Any,
    sign_df: pd.DataFrame,
    corr_clusters: pd.DataFrame,
    ablation_df: pd.DataFrame,
    min_sign_consistency: float,
    n_splits: int,
) -> pd.DataFrame:
    sign_work = sign_df.copy()
    sign_work["raw_feature"] = sign_work["feature"].map(
        lambda f: _raw_feature_from_encoded(
            feature=f,
            numeric_cols=spec.numeric,
            categorical_cols=spec.categorical,
        )
    )

    # Aggregate sign stability at raw-feature level.
    raw_stability = (
        sign_work.groupby("raw_feature", as_index=False)
        .agg(
            stability_score=("sign_consistency", "mean"),
            abs_mean_coef=("abs_mean_coef", "sum"),
            encoded_terms=("feature", "count"),
            nonzero_folds_max=("nonzero_folds", "max"),
        )
        .sort_values(["stability_score", "abs_mean_coef"], ascending=[False, False])
        .reset_index(drop=True)
    )

    cluster_map: dict[str, int] = {}
    if not corr_clusters.empty:
        for _, row in corr_clusters.iterrows():
            cluster_map[str(row["feature"])] = int(row["cluster_id"])

    # Keep at most one feature per high-correlation cluster.
    selected_cluster_ids: set[int] = set()
    rows: list[dict[str, Any]] = []

    full_primary = ablation_df.loc[ablation_df["scenario"] == "full", "primary_metric"].iloc[0]
    tactical_primary = ablation_df.loc[ablation_df["scenario"] == "tactical_only", "primary_metric"].iloc[0]
    allow_proxy = bool((full_primary - tactical_primary) > 0.01) if not np.isnan(tactical_primary) else True

    min_nonzero = max(1, int(np.ceil(0.6 * n_splits)))

    all_features = [*spec.numeric, *spec.categorical]
    for raw_feature in all_features:
        row = raw_stability[raw_stability["raw_feature"] == raw_feature]
        stability_score = float(row["stability_score"].iloc[0]) if not row.empty else np.nan
        abs_mean_coef = float(row["abs_mean_coef"].iloc[0]) if not row.empty else 0.0
        nonzero_max = int(row["nonzero_folds_max"].iloc[0]) if not row.empty else 0
        encoded_terms = int(row["encoded_terms"].iloc[0]) if not row.empty else 0

        cluster_id = cluster_map.get(raw_feature)
        is_proxy = _is_proxy_raw_feature(raw_feature)

        selected = True
        reason = "stable tactical signal"

        if np.isnan(stability_score) or stability_score < min_sign_consistency:
            selected = False
            reason = "low sign stability"
        elif nonzero_max < min_nonzero:
            selected = False
            reason = "weak coefficient presence across folds"
        elif is_proxy and not allow_proxy:
            selected = False
            reason = "proxy feature removable with small metric loss"
        elif cluster_id is not None:
            if cluster_id in selected_cluster_ids:
                selected = False
                reason = "redundant high-correlation cluster"
            else:
                selected_cluster_ids.add(cluster_id)
                reason = "cluster representative"

        rows.append(
            {
                "feature": raw_feature,
                "feature_type": "numeric" if raw_feature in spec.numeric else "categorical",
                "is_proxy": is_proxy,
                "cluster_id": cluster_id,
                "stability_score": stability_score,
                "nonzero_folds_max": nonzero_max,
                "encoded_terms": encoded_terms,
                "abs_mean_coef_sum": abs_mean_coef,
                "selected": selected,
                "selection_reason": reason,
            }
        )

    return pd.DataFrame(rows).sort_values(["selected", "stability_score", "abs_mean_coef_sum"], ascending=[False, False, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = ReportConfig(
        task=args.task,
        variant=args.variant,
        n_splits=args.n_splits,
        corr_threshold=args.corr_threshold,
        min_sign_consistency=args.min_sign_consistency,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) / f"{args.task}_{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input dataset: {input_path}")

    df = pd.read_parquet(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    spec, target_col, group_col = _get_spec(args.task, args.variant, df)
    if target_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Dataset must include {target_col!r} and {group_col!r}")

    x, y, groups = _prepare_xyg_for_task(df, spec, args.task)

    corr_clusters = _numeric_correlation_clusters(df=x, numeric_cols=spec.numeric, threshold=args.corr_threshold)
    corr_path = output_dir / "correlation_clusters.csv"
    corr_clusters.to_csv(corr_path, index=False)

    fold_coefs = _fold_coefficients(x=x, y=y, groups=groups, spec=spec, task=args.task, n_splits=args.n_splits)
    fold_coef_path = output_dir / "fold_coefficients.csv"
    fold_coefs.to_csv(fold_coef_path, index=False)

    sign_stability = _sign_stability(fold_coefs=fold_coefs, n_splits=args.n_splits)
    sign_path = output_dir / "coefficient_sign_stability.csv"
    sign_stability.to_csv(sign_path, index=False)

    ablation = _tactical_proxy_ablation(df=df, spec=spec, task=args.task, n_splits=args.n_splits)
    ablation_path = output_dir / "tactical_vs_proxy_ablation.csv"
    ablation.to_csv(ablation_path, index=False)

    recommended = _recommended_feature_set(
        spec=spec,
        sign_df=sign_stability,
        corr_clusters=corr_clusters,
        ablation_df=ablation,
        min_sign_consistency=args.min_sign_consistency,
        n_splits=args.n_splits,
    )
    recommended_path = output_dir / "interpretable_feature_set_recommended.csv"
    recommended.to_csv(recommended_path, index=False)

    summary = {
        "config": asdict(config),
        "input_path": str(input_path),
        "rows": int(len(df)),
        "variant_features": {
            "categorical": spec.categorical,
            "numeric": spec.numeric,
            "total": len(spec.categorical) + len(spec.numeric),
        },
        "outputs": {
            "correlation_clusters": str(corr_path),
            "fold_coefficients": str(fold_coef_path),
            "coefficient_sign_stability": str(sign_path),
            "tactical_vs_proxy_ablation": str(ablation_path),
            "interpretable_feature_set_recommended": str(recommended_path),
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("FEATURE STABILITY REPORT")
    print("=" * 72)
    print(f"Task: {args.task}")
    print(f"Variant: {args.variant}")
    print(f"Rows: {len(df):,}")
    print(f"Saved: {corr_path}")
    print(f"Saved: {sign_path}")
    print(f"Saved: {ablation_path}")
    print(f"Saved: {recommended_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

