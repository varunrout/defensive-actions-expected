"""Feature selection analysis using correlation clustering and PCA.

This script helps decide what to keep/remove for new variants by producing:
- correlation clusters over numeric features
- PCA diagnostics (explained variance + top loadings)
- feature selection tables for v5 (interpretable) and v6 (balanced)

It writes a JSON manifest that can be used to define new model variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dax.models.baseline_logistic import (  # noqa: E402
    GROUP_COL as LOG_GROUP_COL,
    TARGET_COL as LOG_TARGET_COL,
    default_variant_specs,
    prepare_xyg,
    resolve_columns,
)
from dax.models.baseline_regression import (  # noqa: E402
    GROUP_COL as REG_GROUP_COL,
    TARGET_COL as REG_TARGET_COL,
    default_regression_specs,
    prepare_xyg_regression,
    resolve_columns as resolve_reg_columns,
)

PROXY_FEATURES = {
    "event_order_in_possession",
    "seconds_since_possession_start",
    "possession_duration_total",
    "possession_event_count_total",
    "phase_transition_count_so_far",
    "play_pattern",
    "position_group",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature selection via clustering + PCA")
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
        help="Task family to analyze.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="v4_freeze_geometry",
        help="Base variant to analyze and prune into v5/v6.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.8,
        help="Absolute correlation threshold for cluster edges.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for faster iteration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "validation" / "analysis" / "feature_selection"),
        help="Output directory root.",
    )
    return parser.parse_args()


def _resolve_task_spec(df: pd.DataFrame, task: str, variant: str) -> tuple[Any, str, str]:
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


def _prepare_task_xyg(df: pd.DataFrame, task: str, spec: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if task == "logistic":
        return prepare_xyg(df, spec)
    return prepare_xyg_regression(df, spec)


def _target_corr_numeric(x: pd.DataFrame, y: np.ndarray, numeric_cols: list[str]) -> dict[str, float]:
    corr_map: dict[str, float] = {}
    y_series = pd.Series(y)

    for col in numeric_cols:
        if col not in x.columns:
            corr_map[col] = 0.0
            continue
        s = pd.to_numeric(x[col], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            corr_map[col] = 0.0
            continue
        c = float(s.corr(y_series)) if s.notna().sum() > 3 else 0.0
        if np.isnan(c):
            c = 0.0
        corr_map[col] = abs(c)
    return corr_map


def _build_clusters(corr_abs: pd.DataFrame, threshold: float) -> list[list[str]]:
    visited: set[str] = set()
    clusters: list[list[str]] = []

    for col in corr_abs.columns:
        if col in visited:
            continue
        stack = [col]
        comp: set[str] = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            neighbors = corr_abs.index[corr_abs.loc[node] >= threshold].tolist()
            for nxt in neighbors:
                if nxt not in comp:
                    stack.append(nxt)
        visited.update(comp)
        if len(comp) > 1:
            clusters.append(sorted(comp))
    return clusters


def _plot_corr_heatmap(corr_abs: pd.DataFrame, order: list[str], out_path: Path) -> None:
    if not order:
        return
    matrix = corr_abs.loc[order, order].to_numpy(copy=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Absolute Correlation Heatmap (Cluster-Ordered)")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|corr|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def _plot_pca(numeric_df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    if numeric_df.shape[1] < 2:
        return paths

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(numeric_df)

    pca_full = PCA()
    pca_full.fit(x_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)

    fig_var, ax_var = plt.subplots(figsize=(8, 5))
    ax_var.plot(range(1, len(cum) + 1), cum, marker="o", linewidth=1.5)
    ax_var.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax_var.axhline(0.9, color="gray", linestyle=":", linewidth=1)
    ax_var.set_xlabel("Number of components")
    ax_var.set_ylabel("Cumulative explained variance")
    ax_var.set_title("PCA Cumulative Explained Variance")
    ax_var.set_ylim(0, 1.02)
    fig_var.tight_layout()
    var_path = out_dir / "pca_explained_variance.png"
    fig_var.savefig(var_path, dpi=150)
    paths["explained_variance"] = str(var_path)

    pca_2 = PCA(n_components=min(2, numeric_df.shape[1]))
    pca_2.fit(x_scaled)
    loadings = pd.DataFrame(
        pca_2.components_.T,
        index=numeric_df.columns,
        columns=[f"PC{i + 1}" for i in range(pca_2.n_components_)],
    )

    fig_load, axes = plt.subplots(1, pca_2.n_components_, figsize=(12, 5))
    if pca_2.n_components_ == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        pc_name = f"PC{idx + 1}"
        top = loadings[pc_name].abs().sort_values(ascending=False).head(12).index.tolist()
        view = loadings.loc[top, pc_name].sort_values(ascending=True)
        ax.barh(view.index, view.values, color="steelblue")
        ax.set_title(f"Top loadings: {pc_name}")
        ax.set_xlabel("Loading")

    fig_load.tight_layout()
    load_path = out_dir / "pca_top_loadings.png"
    fig_load.savefig(load_path, dpi=150)
    paths["top_loadings"] = str(load_path)
    return paths


def _select_features(
    spec: Any,
    corr_abs: pd.DataFrame,
    clusters: list[list[str]],
    target_corr_abs: dict[str, float],
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    cluster_id_map: dict[str, int] = {}
    rep_v5: set[str] = set()
    rep_v6: set[str] = set()

    for i, cluster in enumerate(clusters, start=1):
        for f in cluster:
            cluster_id_map[f] = i

        ordered = sorted(cluster, key=lambda c: target_corr_abs.get(c, 0.0), reverse=True)
        non_proxy = [c for c in ordered if c not in PROXY_FEATURES]

        if non_proxy:
            rep_v5.add(non_proxy[0])
        rep_v6.add(ordered[0])

    rows: list[dict[str, Any]] = []

    v5_num: list[str] = []
    v6_num: list[str] = []

    for n in spec.numeric:
        cluster_id = cluster_id_map.get(n)
        is_proxy = n in PROXY_FEATURES

        if cluster_id is None:
            keep_v5 = not is_proxy
            keep_v6 = True
            reason_v5 = "drop proxy" if is_proxy else "non-cluster numeric"
            reason_v6 = "non-cluster numeric"
        else:
            keep_v5 = n in rep_v5
            keep_v6 = n in rep_v6
            reason_v5 = "cluster representative" if keep_v5 else "redundant in cluster"
            if is_proxy and not keep_v5:
                reason_v5 = "proxy + redundant in cluster"
            reason_v6 = "cluster representative" if keep_v6 else "redundant in cluster"

        if keep_v5:
            v5_num.append(n)
        if keep_v6:
            v6_num.append(n)

        rows.append(
            {
                "feature": n,
                "feature_type": "numeric",
                "is_proxy": is_proxy,
                "cluster_id": cluster_id,
                "target_corr_abs": float(target_corr_abs.get(n, 0.0)),
                "keep_v5": keep_v5,
                "reason_v5": reason_v5,
                "keep_v6": keep_v6,
                "reason_v6": reason_v6,
            }
        )

    v5_cat = [c for c in spec.categorical if c not in PROXY_FEATURES]
    v6_cat = list(spec.categorical)

    for c in spec.categorical:
        is_proxy = c in PROXY_FEATURES
        keep_v5 = c in v5_cat
        keep_v6 = c in v6_cat
        rows.append(
            {
                "feature": c,
                "feature_type": "categorical",
                "is_proxy": is_proxy,
                "cluster_id": np.nan,
                "target_corr_abs": np.nan,
                "keep_v5": keep_v5,
                "reason_v5": "drop proxy categorical" if not keep_v5 else "kept categorical",
                "keep_v6": keep_v6,
                "reason_v6": "kept categorical",
            }
        )

    decisions = pd.DataFrame(rows)
    return decisions, v5_cat, v5_num, v6_cat, v6_num


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing dataset: {input_path}")

    df = pd.read_parquet(input_path)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    spec, target_col, group_col = _resolve_task_spec(df=df, task=args.task, variant=args.variant)
    if target_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Dataset must contain {target_col!r} and {group_col!r}")

    x, y, _ = _prepare_task_xyg(df=df, task=args.task, spec=spec)

    out_dir = Path(args.output_dir) / f"{args.task}_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = [c for c in spec.numeric if c in x.columns]
    numeric_df = x[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    variable_numeric_cols = [c for c in numeric_cols if numeric_df[c].nunique(dropna=False) > 1]
    variable_numeric_df = numeric_df[variable_numeric_cols]

    if len(variable_numeric_cols) >= 2:
        corr_abs = variable_numeric_df.corr().abs().fillna(0.0)
    else:
        corr_abs = pd.DataFrame(index=variable_numeric_cols, columns=variable_numeric_cols, data=1.0)

    clusters = (
        _build_clusters(corr_abs=corr_abs, threshold=args.corr_threshold)
        if len(variable_numeric_cols) >= 2
        else []
    )
    target_corr_abs = _target_corr_numeric(x=x, y=y, numeric_cols=numeric_cols)

    cluster_rows: list[dict[str, Any]] = []
    for i, cluster in enumerate(clusters, start=1):
        ordered = sorted(cluster, key=lambda c: target_corr_abs.get(c, 0.0), reverse=True)
        for rank, feature in enumerate(ordered, start=1):
            cluster_rows.append(
                {
                    "cluster_id": i,
                    "cluster_size": len(cluster),
                    "feature": feature,
                    "target_corr_abs": float(target_corr_abs.get(feature, 0.0)),
                    "rank_by_target_corr": rank,
                    "is_proxy": feature in PROXY_FEATURES,
                }
            )

    cluster_df = pd.DataFrame(cluster_rows)
    cluster_csv = out_dir / "correlation_clusters.csv"
    cluster_df.to_csv(cluster_csv, index=False)

    order: list[str] = []
    for cluster in clusters:
        order.extend(cluster)
    order.extend([c for c in variable_numeric_cols if c not in set(order)])

    corr_heatmap = out_dir / "correlation_heatmap_clustered.png"
    if len(variable_numeric_cols) >= 2:
        _plot_corr_heatmap(corr_abs=corr_abs, order=order, out_path=corr_heatmap)

    pca_paths = _plot_pca(numeric_df=variable_numeric_df, out_dir=out_dir)

    decisions, v5_cat, v5_num, v6_cat, v6_num = _select_features(
        spec=spec,
        corr_abs=corr_abs,
        clusters=clusters,
        target_corr_abs=target_corr_abs,
    )

    decisions_csv = out_dir / "feature_selection_decisions.csv"
    decisions.sort_values(["feature_type", "feature"]).to_csv(decisions_csv, index=False)

    v5_df = pd.DataFrame(
        [{"feature": f, "feature_type": "categorical"} for f in v5_cat]
        + [{"feature": f, "feature_type": "numeric"} for f in v5_num]
    )
    v6_df = pd.DataFrame(
        [{"feature": f, "feature_type": "categorical"} for f in v6_cat]
        + [{"feature": f, "feature_type": "numeric"} for f in v6_num]
    )

    v5_csv = out_dir / "selected_features_v5.csv"
    v6_csv = out_dir / "selected_features_v6.csv"
    v5_df.to_csv(v5_csv, index=False)
    v6_df.to_csv(v6_csv, index=False)

    manifest = {
        "task": args.task,
        "base_variant": args.variant,
        "rows": int(len(df)),
        "corr_threshold": float(args.corr_threshold),
        "v5": {
            "name": "v5_interpretable_clustered",
            "categorical": v5_cat,
            "numeric": v5_num,
            "total_features": len(v5_cat) + len(v5_num),
        },
        "v6": {
            "name": "v6_balanced_clustered",
            "categorical": v6_cat,
            "numeric": v6_num,
            "total_features": len(v6_cat) + len(v6_num),
        },
        "artifacts": {
            "clusters_csv": str(cluster_csv),
            "decisions_csv": str(decisions_csv),
            "selected_v5_csv": str(v5_csv),
            "selected_v6_csv": str(v6_csv),
            "corr_heatmap": str(corr_heatmap) if corr_heatmap.exists() else "",
            **pca_paths,
        },
    }

    manifest_path = out_dir / "feature_selection_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("FEATURE SELECTION VIA CLUSTERING + PCA")
    print("=" * 72)
    print(f"Task: {args.task}")
    print(f"Base variant: {args.variant}")
    print(f"Rows analyzed: {len(df):,}")
    print(f"V5 features: {len(v5_cat) + len(v5_num)}")
    print(f"V6 features: {len(v6_cat) + len(v6_num)}")
    print(f"Saved: {cluster_csv}")
    print(f"Saved: {decisions_csv}")
    print(f"Saved: {v5_csv}")
    print(f"Saved: {v6_csv}")
    if corr_heatmap.exists():
        print(f"Saved: {corr_heatmap}")
    if "explained_variance" in pca_paths:
        print(f"Saved: {pca_paths['explained_variance']}")
    if "top_loadings" in pca_paths:
        print(f"Saved: {pca_paths['top_loadings']}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()



