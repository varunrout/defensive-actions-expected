from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_OOF_PATH = PROJECT_ROOT / "outputs" / "oof" / "baseline" / "baseline_oof_predictions.parquet"
REGRESSION_OOF_PATH = PROJECT_ROOT / "outputs" / "oof" / "regression" / "regression_oof_predictions.parquet"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "features" / "player_defensive_actions.parquet"

OUT_DIR = PROJECT_ROOT / "outputs" / "validation" / "comparison" / "defensibility"
DOC_PATH = PROJECT_ROOT / "docs" / "analysis" / "DEFENSIBILITY_ADDENDUM.md"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        frac = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(frac - conf)
    return float(ece)


def bootstrap_metric_by_match(
    df: pd.DataFrame,
    score_col: str,
    metric_name: str,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    match_ids = df["match_id"].unique()
    index_by_match = {mid: idx.to_numpy() for mid, idx in df.groupby("match_id").groups.items()}

    values: list[float] = []
    for _ in range(n_boot):
        sampled_matches = rng.choice(match_ids, size=len(match_ids), replace=True)
        sampled_idx = np.concatenate([index_by_match[mid] for mid in sampled_matches])
        sample = df.loc[sampled_idx]
        y_true = sample["y_true"].to_numpy()
        y_score = sample[score_col].to_numpy()

        if metric_name == "auc":
            if len(np.unique(y_true)) < 2:
                continue
            val = roc_auc_score(y_true, y_score)
        elif metric_name == "ap":
            val = average_precision_score(y_true, y_score)
        elif metric_name == "brier":
            val = brier_score_loss(y_true, y_score)
        elif metric_name == "r2":
            val = r2_score(y_true, y_score)
        elif metric_name == "rmse":
            val = rmse(y_true, y_score)
        elif metric_name == "mae":
            val = mean_absolute_error(y_true, y_score)
        elif metric_name == "spearman":
            corr = spearmanr(y_true, y_score).correlation
            val = float(corr if corr is not None else np.nan)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        if np.isfinite(val):
            values.append(float(val))

    return np.array(values, dtype=float)


def summarize_bootstrap(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "std": float(np.std(values)),
        "n_boot_effective": int(values.size),
    }


def classification_ci_tables(baseline_oof: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    winner_rows: list[dict] = []

    variants = sorted(baseline_oof["variant"].unique().tolist())

    # Store bootstrap samples so we can compute winner frequency robustly.
    auc_samples: dict[str, np.ndarray] = {}

    for variant in variants:
        dfv = baseline_oof[baseline_oof["variant"] == variant]
        auc_vals = bootstrap_metric_by_match(dfv, "y_score", "auc", n_boot, rng)
        ap_vals = bootstrap_metric_by_match(dfv, "y_score", "ap", n_boot, rng)
        brier_vals = bootstrap_metric_by_match(dfv, "y_score", "brier", n_boot, rng)

        auc_samples[variant] = auc_vals

        row = {
            "variant": variant,
            "auc": roc_auc_score(dfv["y_true"], dfv["y_score"]),
            "ap": average_precision_score(dfv["y_true"], dfv["y_score"]),
            "brier": brier_score_loss(dfv["y_true"], dfv["y_score"]),
        }
        row.update({f"auc_{k}": v for k, v in summarize_bootstrap(auc_vals).items()})
        row.update({f"ap_{k}": v for k, v in summarize_bootstrap(ap_vals).items()})
        row.update({f"brier_{k}": v for k, v in summarize_bootstrap(brier_vals).items()})
        rows.append(row)

    # Winner frequency by bootstrap index (truncate to common effective bootstrap length)
    min_len = min(v.size for v in auc_samples.values())
    if min_len > 0:
        winners = []
        variants_arr = np.array(variants)
        auc_matrix = np.vstack([auc_samples[v][:min_len] for v in variants])
        best_idx = np.argmax(auc_matrix, axis=0)
        for idx in best_idx:
            winners.append(variants_arr[idx])
        winner_series = pd.Series(winners).value_counts(normalize=True).sort_values(ascending=False)
        winner_rows = [{"variant": k, "winner_frequency_auc": float(v)} for k, v in winner_series.items()]

    return pd.DataFrame(rows).sort_values("auc", ascending=False), pd.DataFrame(winner_rows)


def regression_ci_tables(reg_oof: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    winner_rows: list[dict] = []

    variants = sorted(reg_oof["variant"].unique().tolist())
    r2_samples: dict[str, np.ndarray] = {}

    for variant in variants:
        dfv = reg_oof[reg_oof["variant"] == variant]
        r2_vals = bootstrap_metric_by_match(dfv, "y_pred", "r2", n_boot, rng)
        rmse_vals = bootstrap_metric_by_match(dfv, "y_pred", "rmse", n_boot, rng)
        mae_vals = bootstrap_metric_by_match(dfv, "y_pred", "mae", n_boot, rng)
        sp_vals = bootstrap_metric_by_match(dfv, "y_pred", "spearman", n_boot, rng)

        r2_samples[variant] = r2_vals

        row = {
            "variant": variant,
            "r2": r2_score(dfv["y_true"], dfv["y_pred"]),
            "rmse": rmse(dfv["y_true"].to_numpy(), dfv["y_pred"].to_numpy()),
            "mae": mean_absolute_error(dfv["y_true"], dfv["y_pred"]),
            "spearman": float(spearmanr(dfv["y_true"], dfv["y_pred"]).correlation),
        }
        row.update({f"r2_{k}": v for k, v in summarize_bootstrap(r2_vals).items()})
        row.update({f"rmse_{k}": v for k, v in summarize_bootstrap(rmse_vals).items()})
        row.update({f"mae_{k}": v for k, v in summarize_bootstrap(mae_vals).items()})
        row.update({f"spearman_{k}": v for k, v in summarize_bootstrap(sp_vals).items()})
        rows.append(row)

    min_len = min(v.size for v in r2_samples.values())
    if min_len > 0:
        winners = []
        variants_arr = np.array(variants)
        r2_matrix = np.vstack([r2_samples[v][:min_len] for v in variants])
        best_idx = np.argmax(r2_matrix, axis=0)
        for idx in best_idx:
            winners.append(variants_arr[idx])
        winner_series = pd.Series(winners).value_counts(normalize=True).sort_values(ascending=False)
        winner_rows = [{"variant": k, "winner_frequency_r2": float(v)} for k, v in winner_series.items()]

    return pd.DataFrame(rows).sort_values("r2", ascending=False), pd.DataFrame(winner_rows)


def plot_classification_calibration(baseline_oof: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    top_variants = ["v3_context_enhanced", "v4_freeze_geometry", "v6_balanced_clustered", "v5_interpretable_clustered"]
    plt.figure(figsize=(8, 6))
    rows: list[dict] = []

    for variant in top_variants:
        dfv = baseline_oof[baseline_oof["variant"] == variant]
        y_true = dfv["y_true"].to_numpy()
        y_score = dfv["y_score"].to_numpy()

        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=variant)

        rows.append(
            {
                "variant": variant,
                "ece": ece_score(y_true, y_score, n_bins=10),
                "brier": brier_score_loss(y_true, y_score),
                "avg_pred": float(np.mean(y_score)),
                "base_rate": float(np.mean(y_true)),
            }
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Calibration Reliability Curves (OOF)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    return pd.DataFrame(rows).sort_values("ece")


def plot_regression_decile_bias(reg_oof: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    top_variants = ["v3_context_enhanced", "v4_freeze_geometry", "v6_balanced_clustered", "v5_interpretable_clustered"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    rows: list[dict] = []

    for ax, variant in zip(axes.flatten(), top_variants):
        dfv = reg_oof[reg_oof["variant"] == variant].copy()
        dfv["pred_bin"] = pd.qcut(dfv["y_pred"], q=10, labels=False, duplicates="drop")
        grp = dfv.groupby("pred_bin", as_index=False).agg(
            pred_mean=("y_pred", "mean"),
            obs_mean=("y_true", "mean"),
            n=("y_true", "size"),
        )
        grp["bias"] = grp["obs_mean"] - grp["pred_mean"]
        ax.plot(grp["pred_mean"], grp["obs_mean"], marker="o")
        minv = float(min(grp["pred_mean"].min(), grp["obs_mean"].min()))
        maxv = float(max(grp["pred_mean"].max(), grp["obs_mean"].max()))
        ax.plot([minv, maxv], [minv, maxv], "k--", alpha=0.5)
        ax.set_title(variant)
        ax.set_xlabel("Predicted mean by decile")
        ax.set_ylabel("Observed mean by decile")

        rows.append(
            {
                "variant": variant,
                "mean_abs_decile_bias": float(np.mean(np.abs(grp["bias"]))),
                "max_abs_decile_bias": float(np.max(np.abs(grp["bias"]))),
            }
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return pd.DataFrame(rows).sort_values("mean_abs_decile_bias")


def slice_gap_table(
    baseline_oof: pd.DataFrame,
    reg_oof: pd.DataFrame,
    feat: pd.DataFrame,
) -> pd.DataFrame:
    # Align using dataset row order per variant: OOF tables are stacked by variant and keep original row sequence.
    # We only compare v3 vs v4 where slice-specialist behavior matters most.
    feat_small = feat[["phase_label", "action_zone", "position_group"]].reset_index(drop=True)

    def add_slices(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for variant in ["v3_context_enhanced", "v4_freeze_geometry"]:
            d = df[df["variant"] == variant].copy().reset_index(drop=True)
            merged = pd.concat([d, feat_small], axis=1)
            merged["variant"] = variant
            parts.append(merged)
        return pd.concat(parts, ignore_index=True)

    cls = add_slices(baseline_oof)
    reg = add_slices(reg_oof)

    rows: list[dict] = []
    for dim in ["phase_label", "action_zone", "position_group"]:
        for slice_val, g in cls.groupby(dim):
            g3 = g[g["variant"] == "v3_context_enhanced"]
            g4 = g[g["variant"] == "v4_freeze_geometry"]
            if len(g3) < 200 or len(g4) < 200 or g3["y_true"].nunique() < 2 or g4["y_true"].nunique() < 2:
                continue
            auc3 = roc_auc_score(g3["y_true"], g3["y_score"])
            auc4 = roc_auc_score(g4["y_true"], g4["y_score"])
            rows.append(
                {
                    "task": "classification",
                    "dimension": dim,
                    "slice": str(slice_val),
                    "v3_metric": auc3,
                    "v4_metric": auc4,
                    "delta_v4_minus_v3": auc4 - auc3,
                    "n_rows": int(len(g3)),
                }
            )

        for slice_val, g in reg.groupby(dim):
            g3 = g[g["variant"] == "v3_context_enhanced"]
            g4 = g[g["variant"] == "v4_freeze_geometry"]
            if len(g3) < 200 or len(g4) < 200:
                continue
            r23 = r2_score(g3["y_true"], g3["y_pred"])
            r24 = r2_score(g4["y_true"], g4["y_pred"])
            rows.append(
                {
                    "task": "regression",
                    "dimension": dim,
                    "slice": str(slice_val),
                    "v3_metric": r23,
                    "v4_metric": r24,
                    "delta_v4_minus_v3": r24 - r23,
                    "n_rows": int(len(g3)),
                }
            )

    out = pd.DataFrame(rows)
    return out.sort_values(["task", "delta_v4_minus_v3"], ascending=[True, False])


def render_markdown(
    cls_ci: pd.DataFrame,
    cls_winners: pd.DataFrame,
    reg_ci: pd.DataFrame,
    reg_winners: pd.DataFrame,
    calib: pd.DataFrame,
    decile_bias: pd.DataFrame,
    slice_gap: pd.DataFrame,
    n_boot: int,
) -> str:
    top_cls = cls_ci.iloc[0]
    top_reg = reg_ci.iloc[0]

    slice_cls = slice_gap[slice_gap["task"] == "classification"].head(10)
    slice_reg = slice_gap[slice_gap["task"] == "regression"].head(10)

    lines: list[str] = []

    def df_to_markdown(df: pd.DataFrame) -> str:
        cols = df.columns.tolist()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [header, sep]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)
    lines.append("# Defensibility Addendum")
    lines.append("")
    lines.append("This addendum strengthens the evidence package with uncertainty quantification, calibration diagnostics, and slice-level gap stress tests.")
    lines.append("")
    lines.append("## What was added")
    lines.append("")
    lines.append(f"- Match-bootstrap confidence intervals for each variant (`n_boot={n_boot}`).")
    lines.append("- Winner-frequency analysis (how often each model is best under resampling).")
    lines.append("- Classification calibration checks (ECE + reliability curves).")
    lines.append("- Regression decile-bias checks (observed vs predicted by risk bands).")
    lines.append("- Slice-level `v4 - v3` gap table across `phase_label`, `action_zone`, and `position_group`.")
    lines.append("")
    lines.append("## Key outcomes")
    lines.append("")
    lines.append(
        f"- Classification top mean AUC remains `{top_cls['variant']}`: {top_cls['auc_mean']:.4f} "
        f"(95% CI {top_cls['auc_ci_low']:.4f}, {top_cls['auc_ci_high']:.4f})."
    )
    lines.append(
        f"- Regression top mean R2 remains `{top_reg['variant']}`: {top_reg['r2_mean']:.4f} "
        f"(95% CI {top_reg['r2_ci_low']:.4f}, {top_reg['r2_ci_high']:.4f})."
    )
    if not cls_winners.empty:
        cw = cls_winners.iloc[0]
        lines.append(f"- Classification ranking robustness: `{cw['variant']}` wins {cw['winner_frequency_auc']:.1%} of bootstrap runs.")
    if not reg_winners.empty:
        rw = reg_winners.iloc[0]
        lines.append(f"- Regression ranking robustness: `{rw['variant']}` wins {rw['winner_frequency_r2']:.1%} of bootstrap runs.")
    best_cal = calib.iloc[0]
    lines.append(
        f"- Best calibration among tested classifiers: `{best_cal['variant']}` (ECE={best_cal['ece']:.4f}, "
        f"Brier={best_cal['brier']:.4f})."
    )
    best_bias = decile_bias.iloc[0]
    lines.append(
        f"- Most stable regression decile bias: `{best_bias['variant']}` "
        f"(mean abs decile bias={best_bias['mean_abs_decile_bias']:.4f})."
    )
    lines.append("")

    lines.append("## Tables")
    lines.append("")
    lines.append("### Classification confidence intervals")
    lines.append("")
    lines.append(df_to_markdown(cls_ci[["variant", "auc", "auc_ci_low", "auc_ci_high", "ap", "ap_ci_low", "ap_ci_high", "brier", "brier_ci_low", "brier_ci_high"]]))
    lines.append("")
    lines.append("### Regression confidence intervals")
    lines.append("")
    lines.append(df_to_markdown(reg_ci[["variant", "r2", "r2_ci_low", "r2_ci_high", "rmse", "rmse_ci_low", "rmse_ci_high", "mae", "mae_ci_low", "mae_ci_high"]]))
    lines.append("")
    if not cls_winners.empty:
        lines.append("### Classification winner frequency")
        lines.append("")
        lines.append(df_to_markdown(cls_winners))
        lines.append("")
    if not reg_winners.empty:
        lines.append("### Regression winner frequency")
        lines.append("")
        lines.append(df_to_markdown(reg_winners))
        lines.append("")

    lines.append("### Largest positive `v4 - v3` slice gaps (classification AUC)")
    lines.append("")
    if not slice_cls.empty:
        lines.append(df_to_markdown(slice_cls[["dimension", "slice", "v3_metric", "v4_metric", "delta_v4_minus_v3", "n_rows"]]))
    else:
        lines.append("No eligible slices met minimum row thresholds.")
    lines.append("")

    lines.append("### Largest positive `v4 - v3` slice gaps (regression R2)")
    lines.append("")
    if not slice_reg.empty:
        lines.append(df_to_markdown(slice_reg[["dimension", "slice", "v3_metric", "v4_metric", "delta_v4_minus_v3", "n_rows"]]))
    else:
        lines.append("No eligible slices met minimum row thresholds.")
    lines.append("")

    lines.append("## New charts")
    lines.append("")
    lines.append("- `outputs/validation/comparison/defensibility/calibration_reliability_curves.png`")
    lines.append("- `outputs/validation/comparison/defensibility/regression_decile_bias.png`")
    lines.append("")

    lines.append("## Defensibility checklist (critical gaps)")
    lines.append("")
    lines.append("- Added now: uncertainty CIs, rank robustness, calibration, decile bias, focused slice stress tests.")
    lines.append("- Still recommended next: external temporal holdout by competition season, decision-curve utility analysis, and model monitoring thresholds for deployment drift.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build defensibility addendum with extra charts and uncertainty diagnostics.")
    parser.add_argument("--n-boot", type=int, default=250, help="Bootstrap iterations by match.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_oof = pd.read_parquet(BASELINE_OOF_PATH)
    reg_oof = pd.read_parquet(REGRESSION_OOF_PATH)
    feat = pd.read_parquet(FEATURE_DATA_PATH)

    cls_ci, cls_winners = classification_ci_tables(baseline_oof, n_boot=args.n_boot, seed=args.seed)
    reg_ci, reg_winners = regression_ci_tables(reg_oof, n_boot=args.n_boot, seed=args.seed + 1)

    calib = plot_classification_calibration(baseline_oof, OUT_DIR / "calibration_reliability_curves.png")
    decile_bias = plot_regression_decile_bias(reg_oof, OUT_DIR / "regression_decile_bias.png")
    slice_gap = slice_gap_table(baseline_oof, reg_oof, feat)

    cls_ci.to_csv(OUT_DIR / "classification_metric_ci.csv", index=False)
    cls_winners.to_csv(OUT_DIR / "classification_winner_frequency.csv", index=False)
    reg_ci.to_csv(OUT_DIR / "regression_metric_ci.csv", index=False)
    reg_winners.to_csv(OUT_DIR / "regression_winner_frequency.csv", index=False)
    calib.to_csv(OUT_DIR / "classification_calibration_summary.csv", index=False)
    decile_bias.to_csv(OUT_DIR / "regression_decile_bias_summary.csv", index=False)
    slice_gap.to_csv(OUT_DIR / "slice_gap_v4_vs_v3.csv", index=False)

    summary = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "best_classification_auc": cls_ci.iloc[0]["variant"],
        "best_regression_r2": reg_ci.iloc[0]["variant"],
        "best_calibration_ece": calib.iloc[0]["variant"],
        "lowest_regression_decile_bias": decile_bias.iloc[0]["variant"],
    }
    (OUT_DIR / "defensibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = render_markdown(
        cls_ci=cls_ci,
        cls_winners=cls_winners,
        reg_ci=reg_ci,
        reg_winners=reg_winners,
        calib=calib,
        decile_bias=decile_bias,
        slice_gap=slice_gap,
        n_boot=args.n_boot,
    )
    DOC_PATH.write_text(markdown, encoding="utf-8")

    print("[DONE] Wrote defensibility artifacts to:", OUT_DIR)
    print("[DONE] Wrote addendum report:", DOC_PATH)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()



