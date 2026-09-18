"""Post-lock validation pass on the passive-binary baselines (p0-p3).

Prompt 49 built p0-p3 on the real 1,593,181-row passive_defense dataset and
produced CV metrics plus a single held-out-test readout for all 4 variants.
p1_unweighted is the current pick (best CV and held-out PR-AUC among
p1/p2/p3, same class-weighting-destroys-calibration mechanism as the active
leg, checked directly against this leg's own numbers). This script closes 4
remaining validation gaps on that pick, mirroring
scripts/models/validate_active_binary_baselines.py but adapted for two real
structural differences this leg has and the active leg doesn't:

  1. Paired significance test: is p1's CV PR-AUC edge over p2/p3 real or
     fold noise? (paired t-test + Wilcoxon on per-fold average precision,
     same as the active leg's item 1.)
  2. Tournament-stratified check: does p1 hold up across the tournaments
     actually present in this leg's matches, not just in aggregate?
  3. Cluster-aware bootstrap on the held-out test set. This REPLACES the
     active leg's player-disjoint recheck -- passive_defense.parquet has no
     player_id column (Phase 4 of the passive build dropped frame-to-frame
     player tracking as unreliable), so there is nothing to group a
     disjoint split on. Instead, this quantifies the row-correlation
     caveat Prompt 49's own report flagged and explicitly deferred:
     every metric reported so far (CV and held-out) treats each row as an
     independent Bernoulli trial, but rows sharing the same event_id (the
     on-ball-attacking-event identifier -- verified against
     src/eda/feature_config.py's PASSIVE row_description and the real
     column list on the locked parquet, not assumed) describe the same
     attacking context through different defender-slots and are not
     independent. A naive row-level bootstrap and a cluster bootstrap
     (resampling whole event_id groups) are run side by side on p1's and
     p2's held-out test scores; if the cluster-aware CI is meaningfully
     wider, that's the caveat made concrete -- the naive significance test
     in item 1 is then likely overconfident, quantified by the actual
     width ratio, not just asserted.
  4. Error analysis: where does p1 miss on the held-out test set, broken
     down by phase_label and by defender_functional_role -- the two slice
     dimensions this leg actually has (there is no player identity to
     slice by, unlike the active leg's phase_label/position breakdown).
  5. Chart-integrity pass: the 16 baseline PNGs exist, are non-empty, and
     are pixel-consistent per chart type across variants.

Everything here reuses DesignMatrixBuilder, fit_predict_fold, load_data and
the feature/variant constants from train_passive_binary_baseline.py
directly, and loads the same frozen canonical split via dax.models.splits
-- no split is recomputed, no variant hyperparameter changes, and the
existing comparison/held-out-readout CSVs from prompt 49 are not touched.

Usage:
    python scripts/models/validate_passive_binary_baselines.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/models/ -> repo root
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_passive_binary_baseline as tb  # noqa: E402  (needs sys.path set first)
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "models" / "validation"
CHARTS_DIR = REPO_ROOT / "outputs" / "models" / "classification" / "charts"
MATCHES_GLOB = str(REPO_ROOT / "data" / "raw" / "matches" / "*.json")

# The on-ball-attacking-event identifier this leg's row unit is keyed to
# (feature_config.py PASSIVE["row_description"]: "one row per visible
# defender-slot per on-ball attacking event"). Verified directly against the
# real column list on data/features/passive_defense.parquet -- it is
# "event_id", not "on_ball_event_id" (no such column exists on this parquet).
EVENT_GROUP_COL = "event_id"

VARIANTS_FOR_SIGNIFICANCE = ["p2_weighted", "p3_weighted_interactions"]
ERROR_ANALYSIS_GROUP_COLS = ["phase_label", "defender_functional_role"]

CHART_VARIANTS = ["p0_dummy", "p1_unweighted", "p2_weighted", "p3_weighted_interactions"]
CHART_NAMES = ["calibration_curve.png", "precision_recall_curve.png", "roc_curve.png", "prediction_distribution.png"]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def build_competition_map() -> dict[int, str]:
    """match_id -> "{competition_name} {season}" from data/raw/matches/*.json.

    Same map the active leg's Prompt 38 built -- same matches, same source
    files, nothing leg-specific to rebuild.
    """
    comp: dict[int, str] = {}
    for path in glob.glob(MATCHES_GLOB):
        matches = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in matches:
            comp[m["match_id"]] = f"{m['competition_name']} {m['season']}"
    return comp


def item1_significance_test(df_all: pd.DataFrame) -> dict:
    """Paired fold-level PR-AUC comparison: p1 vs p2, p1 vs p3.

    Refits all three variants on each of the 5 canonical CV folds (same
    folds/assignment as the Prompt 49 baseline run) and runs a paired
    t-test + Wilcoxon signed-rank test on the per-fold PR-AUC differences.
    This test assumes row-level independence within each fold's PR-AUC
    estimate -- item 3 (the cluster-aware bootstrap) revisits that
    assumption directly on the held-out test set.
    """
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)

    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    variants = ["p1_unweighted", *VARIANTS_FOR_SIGNIFICANCE]
    per_fold_ap: dict[str, list[float]] = {v: [] for v in variants}
    for fold_id in sorted(df["fold"].unique()):
        train_mask = (df["fold"] != fold_id).to_numpy()
        test_mask_f = (df["fold"] == fold_id).to_numpy()
        train_df, test_df = df.loc[train_mask], df.loc[test_mask_f]
        y_train, y_test = y_all[train_mask], y_all[test_mask_f]
        for variant in variants:
            score, _, _ = tb.fit_predict_fold(train_df, test_df, y_train, y_test, variant)
            m = classification_metrics(y_test, score)
            per_fold_ap[variant].append(m["average_precision"])

    result: dict = {
        "assumes_row_level_independence_within_fold": True,
        "note": "see item 3 (cluster_bootstrap_p1_vs_p2.json) for how much this independence assumption matters",
        "per_fold_average_precision": per_fold_ap,
    }
    a = np.array(per_fold_ap["p1_unweighted"])
    for other in VARIANTS_FOR_SIGNIFICANCE:
        b = np.array(per_fold_ap[other])
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        if len(set(diff.round(12))) > 1:
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float("nan"), float("nan")
        result[f"p1_vs_{other}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        }
    return result


def _fit_once_and_score_test(df_all: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit `variant` once on all train+val rows, score the held-out test
    set once. Shared by items 2, 3 and 4 so all three read from the exact
    same single fit/score pass per variant."""
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)

    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test = test_df[tb.TARGET_COL].to_numpy()
    score_test, _, _ = tb.fit_predict_fold(trainval_df, test_df, y_trainval, y_test, variant)
    return test_df, y_test, score_test


def item2_tournament_stratified(df_all: pd.DataFrame, comp_map: dict[int, str], test_df: pd.DataFrame, score_test: np.ndarray) -> dict:
    """p1, fit once on all train+val rows (same fit as the Prompt 49 held-out
    readout), scored on the held-out test set, broken down by tournament."""
    trainval_df = df_all.loc[~canonical_test_mask(df_all, group_col=tb.GROUP_COL)].reset_index(drop=True)

    test_df = test_df.copy()
    test_df["competition"] = test_df[tb.GROUP_COL].map(comp_map)
    test_df["_score"] = score_test

    unmapped = test_df["competition"].isna().sum()

    out: dict = {}
    for comp, sub in test_df.groupby("competition", dropna=False):
        y = sub[tb.TARGET_COL].to_numpy()
        s = sub["_score"].to_numpy()
        matches = sub[tb.GROUP_COL].nunique()
        entry: dict = {"rows": int(len(sub)), "matches": int(matches)}
        if matches < 2 or y.sum() < 2:
            entry["note"] = "too few matches/positives for a stable readout"
        else:
            entry.update(classification_metrics(y, s))
        out[str(comp)] = entry

    trainval_df = trainval_df.copy()
    trainval_df["competition"] = trainval_df[tb.GROUP_COL].map(comp_map)
    train_composition = {str(k): int(v) for k, v in trainval_df.groupby("competition")[tb.GROUP_COL].nunique().items()}
    test_composition = {str(k): int(v) for k, v in test_df.groupby("competition")[tb.GROUP_COL].nunique().items()}

    return {
        "held_out_test_by_tournament": out,
        "train_val_match_counts_by_tournament": train_composition,
        "test_match_counts_by_tournament": test_composition,
        "held_out_rows_with_unmapped_tournament": int(unmapped),
    }


def _bootstrap_ci(values: np.ndarray) -> dict:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "width": float(hi - lo)}


def item3_cluster_bootstrap(y_test: np.ndarray, score_p1: np.ndarray, score_p2: np.ndarray, event_ids: np.ndarray) -> dict:
    """Naive row-level bootstrap vs cluster bootstrap (grouped by
    EVENT_GROUP_COL) on the held-out test set, for p1's PR-AUC and for
    (p1 - p2)'s PR-AUC difference. Quantifies whether the row-correlation
    caveat Prompt 49 flagged actually widens the uncertainty on p1's
    apparent advantage, and by how much."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_test)

    # -- Naive row-level bootstrap: resample individual rows. --
    naive_p1, naive_diff = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b = y_test[idx]
        if y_b.sum() == 0:
            continue
        ap_p1 = average_precision_score(y_b, score_p1[idx])
        ap_p2 = average_precision_score(y_b, score_p2[idx])
        naive_p1.append(ap_p1)
        naive_diff.append(ap_p1 - ap_p2)

    # -- Cluster bootstrap: resample whole event_id groups. Every row
    # belonging to a sampled event goes into that bootstrap sample together
    # (with repetition if the same event is drawn more than once), which is
    # what actually preserves the within-event row correlation under
    # resampling instead of averaging it away. --
    event_to_rows: dict = {}
    for i, eid in enumerate(event_ids):
        event_to_rows.setdefault(eid, []).append(i)
    unique_events = np.array(list(event_to_rows.keys()))
    n_events = len(unique_events)
    event_row_lists = [np.array(event_to_rows[e]) for e in unique_events]

    cluster_p1, cluster_diff = [], []
    for _ in range(N_BOOTSTRAP):
        sampled_event_idx = rng.integers(0, n_events, size=n_events)
        idx = np.concatenate([event_row_lists[j] for j in sampled_event_idx])
        y_b = y_test[idx]
        if y_b.sum() == 0:
            continue
        ap_p1 = average_precision_score(y_b, score_p1[idx])
        ap_p2 = average_precision_score(y_b, score_p2[idx])
        cluster_p1.append(ap_p1)
        cluster_diff.append(ap_p1 - ap_p2)

    naive_p1_ci = _bootstrap_ci(np.array(naive_p1))
    naive_diff_ci = _bootstrap_ci(np.array(naive_diff))
    cluster_p1_ci = _bootstrap_ci(np.array(cluster_p1))
    cluster_diff_ci = _bootstrap_ci(np.array(cluster_diff))

    rows_per_event = np.array([len(v) for v in event_to_rows.values()])

    return {
        "n_bootstrap_iterations": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "held_out_test_rows": int(n),
        "held_out_test_n_events": int(n_events),
        "avg_rows_per_event": float(rows_per_event.mean()),
        "median_rows_per_event": float(np.median(rows_per_event)),
        "p1_average_precision": {
            "naive_row_level": naive_p1_ci,
            "cluster_by_event": cluster_p1_ci,
            "cluster_to_naive_width_ratio": float(cluster_p1_ci["width"] / naive_p1_ci["width"]),
        },
        "p1_minus_p2_average_precision": {
            "naive_row_level": naive_diff_ci,
            "cluster_by_event": cluster_diff_ci,
            "cluster_to_naive_width_ratio": float(cluster_diff_ci["width"] / naive_diff_ci["width"]),
            "naive_ci_excludes_zero": bool(naive_diff_ci["ci_lo"] > 0 or naive_diff_ci["ci_hi"] < 0),
            "cluster_ci_excludes_zero": bool(cluster_diff_ci["ci_lo"] > 0 or cluster_diff_ci["ci_hi"] < 0),
        },
    }


def item4_error_analysis(test_df: pd.DataFrame, y_test: np.ndarray, score_test: np.ndarray) -> dict:
    """Where does p1 (fit on train+val, scored on held-out test, same fit as
    item 2/3) miss, by phase_label and by defender_functional_role -- the
    two slice dimensions this leg actually has (no player identity)."""
    test_df = test_df.copy()
    test_df["_score"] = score_test
    test_df["_y"] = y_test

    out: dict = {}
    for group_col in ERROR_ANALYSIS_GROUP_COLS:
        rows = []
        for val, sub in test_df.groupby(group_col, dropna=False):
            y = sub["_y"].to_numpy()
            s = sub["_score"].to_numpy()
            row: dict = {
                group_col: str(val),
                "rows": int(len(sub)),
                "positive_rate": float(y.mean()),
                "mean_predicted": float(s.mean()),
                "calibration_gap": float(s.mean() - y.mean()),
            }
            if y.sum() >= 2 and len(np.unique(y)) > 1:
                m = classification_metrics(y, s)
                row["average_precision"] = m["average_precision"]
                row["roc_auc"] = m["roc_auc"]
                row["log_loss"] = m["log_loss"]
            else:
                row["skipped_reason"] = "fewer than 2 positives or single-class slice"
            rows.append(row)
        out[group_col] = sorted(rows, key=lambda r: -r["rows"])

    out["independence_caveat"] = (
        "Rows in different phase_label/defender_functional_role groups can still share the same "
        f"{EVENT_GROUP_COL} (a defending-team slot in one group sees teammates classified into other "
        "groups at the same on-ball event) -- these per-group counts and metrics are not fully "
        "independent of each other either, the same row-correlation property documented for the "
        "dataset as a whole (see item 3's cluster bootstrap)."
    )
    return out


def item5_chart_integrity() -> dict:
    """Automated integrity pass on the 16 baseline PNGs: existence,
    non-empty, and pixel-dimension consistency per chart type across the
    4 variants. Not a visual audit."""
    issues: list[str] = []
    sizes_by_chart: dict[str, list[tuple[int, int]]] = {name: [] for name in CHART_NAMES}
    files_checked = 0

    for variant in CHART_VARIANTS:
        for chart_name in CHART_NAMES:
            path = CHARTS_DIR / variant / chart_name
            if not path.exists():
                issues.append(f"missing: {variant}/{chart_name}")
                continue
            size_bytes = path.stat().st_size
            if size_bytes < 512:
                issues.append(f"near-zero file size ({size_bytes}B): {variant}/{chart_name}")
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    dims = img.size
            except Exception as exc:  # noqa: BLE001 -- report, don't crash the validation run
                issues.append(f"unreadable image: {variant}/{chart_name} ({exc})")
                continue
            files_checked += 1
            sizes_by_chart[chart_name].append(dims)

    consistent_by_chart = {}
    for chart_name, sizes in sizes_by_chart.items():
        unique_sizes = sorted(set(sizes))
        consistent_by_chart[chart_name] = {
            "n_files": len(sizes),
            "unique_dimensions": [list(s) for s in unique_sizes],
            "consistent": len(unique_sizes) <= 1,
        }
        if len(unique_sizes) > 1:
            issues.append(f"inconsistent dimensions for {chart_name}: {unique_sizes}")

    return {
        "expected_files": len(CHART_VARIANTS) * len(CHART_NAMES),
        "files_checked_ok": files_checked,
        "consistent_by_chart_type": consistent_by_chart,
        "issues": issues,
        "all_ok": files_checked == len(CHART_VARIANTS) * len(CHART_NAMES) and not issues,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = tb.load_data()
    load_canonical_split()
    comp_map = build_competition_map()

    assert EVENT_GROUP_COL in df_all.columns, f"{EVENT_GROUP_COL} not found on the locked passive_defense parquet"

    print("[1/5] paired significance test (p1 vs p2, p1 vs p3)...")
    sig = item1_significance_test(df_all)
    (OUT_DIR / "significance_p1_vs_p2_p3.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[fit] p1_unweighted and p2_weighted, once each on train+val, scored on held-out test...")
    test_df, y_test, score_p1 = _fit_once_and_score_test(df_all, "p1_unweighted")
    _, _, score_p2 = _fit_once_and_score_test(df_all, "p2_weighted")

    print("[2/5] tournament-stratified held-out test check...")
    tourn = item2_tournament_stratified(df_all, comp_map, test_df, score_p1)
    (OUT_DIR / "tournament_stratified_p1.json").write_text(json.dumps(tourn, indent=2, default=str), encoding="utf-8")

    print(f"[3/5] cluster-aware bootstrap (naive row-level vs cluster-by-{EVENT_GROUP_COL})...")
    boot = item3_cluster_bootstrap(y_test, score_p1, score_p2, test_df[EVENT_GROUP_COL].to_numpy())
    (OUT_DIR / "cluster_bootstrap_p1_vs_p2.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    print("[4/5] error analysis by phase_label / defender_functional_role...")
    errs = item4_error_analysis(test_df, y_test, score_p1)
    (OUT_DIR / "error_analysis_p1.json").write_text(json.dumps(errs, indent=2, default=str), encoding="utf-8")

    print("[5/5] chart integrity pass...")
    charts = item5_chart_integrity()
    print("  all_ok:", charts["all_ok"], "issues:", charts["issues"] or "none")

    print("\n=== SIGNIFICANCE (p1 vs p2/p3, paired over 5 canonical CV folds) ===")
    for other in VARIANTS_FOR_SIGNIFICANCE:
        r = sig[f"p1_vs_{other}"]
        print(f"p1_vs_{other}: mean_diff={r['mean_diff']:+.4f} std_diff={r['std_diff']:.4f} "
              f"paired_t_p={r['paired_t_pvalue']:.4f} wilcoxon_p={r['wilcoxon_pvalue']}")

    print("\n=== TOURNAMENT (held-out test, p1) ===")
    for comp, m in tourn["held_out_test_by_tournament"].items():
        print(comp, m)

    print("\n=== CLUSTER BOOTSTRAP (p1 AP, p1-p2 AP diff; naive row-level vs cluster-by-event) ===")
    print("p1 AP naive:", boot["p1_average_precision"]["naive_row_level"])
    print("p1 AP cluster:", boot["p1_average_precision"]["cluster_by_event"])
    print("p1-p2 diff naive:", boot["p1_minus_p2_average_precision"]["naive_row_level"])
    print("p1-p2 diff cluster:", boot["p1_minus_p2_average_precision"]["cluster_by_event"])
    print("width ratio (diff, cluster/naive):", boot["p1_minus_p2_average_precision"]["cluster_to_naive_width_ratio"])

    print("\n=== ERROR ANALYSIS: phase_label (held-out test) ===")
    for row in errs["phase_label"]:
        print(row)
    print("\n=== ERROR ANALYSIS: defender_functional_role (held-out test) ===")
    for row in errs["defender_functional_role"]:
        print(row)

    print("\nDone. Outputs written under", OUT_DIR)


if __name__ == "__main__":
    main()
