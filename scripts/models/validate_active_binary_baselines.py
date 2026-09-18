"""Post-lock validation pass on the active-binary baselines (v0-v3).

Prompt 36 built v0-v3 on the real dataset; prompt 37 added v1's held-out
test readout. v1_unweighted is the current pick (best CV and held-out
PR-AUC among v1/v2/v3, no overfitting signal). This script closes 4
remaining validation gaps on that pick before it is treated as settled:

  1. Paired significance test: is v1's CV PR-AUC edge over v2/v3 real or
     fold noise? (paired t-test + Wilcoxon on per-fold average precision)
  2. Tournament-stratified check: does v1 hold up across FIFA World Cup
     2022 / UEFA Euro 2020 / UEFA Euro 2024, not just in aggregate?
  3. Player-disjoint train/test run: an actual GroupKFold(player_id) fit
     and score, not just the fold-balance diagnostic from prompt 31's
     PLAYER_LEVEL_VALIDITY_CHECK.json.
  4. Error analysis: where does v1 miss on the held-out test set, broken
     down by phase_label and (see deviation note below) position?
  5. Chart-integrity pass: the 16 baseline PNGs exist, are non-empty, and
     are pixel-consistent per chart type across variants.

Deviation from the original brief: `defender_functional_role` does not
exist on data/features/player_defensive_actions.parquet (the ACTIVE
dataset these baselines are trained on) -- it is defined in
src/dax/features/passive_defense.py and only appears in the PASSIVE leg's
parquet (data/features/passive_defense.parquet), per
src/eda/feature_config.py's PASSIVE["categorical"]. Item 4 therefore uses
`position` instead: it is the closest per-row, per-player categorical
already in ACTIVE["categorical"] (the acting player's on-pitch position),
and is the actual defender-identity dimension available on this leg.

Everything here reuses DesignMatrixBuilder, fit_predict_fold, load_data
and the feature/variant constants from train_active_binary_baseline.py
directly, and loads the same frozen canonical split via dax.models.splits
-- no split is recomputed, no variant hyperparameter changes, and the
existing comparison/held-out-readout CSVs from prompts 36/37 are not
touched.

Usage:
    python scripts/models/validate_active_binary_baselines.py
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
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/<subfolder>/ -> repo root (prompt 48 move: was parents[1] at scripts/ root)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))  # prompt 48: sibling ladder scripts now live in scripts/models/, not scripts/
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402  (needs sys.path set first)
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

OUT_DIR = REPO_ROOT / "outputs" / "models" / "validation"
CHARTS_DIR = REPO_ROOT / "outputs" / "models" / "classification" / "charts"
MATCHES_GLOB = str(REPO_ROOT / "data" / "raw" / "matches" / "*.json")

VARIANTS_FOR_SIGNIFICANCE = ["v2_weighted", "v3_weighted_interactions"]
ERROR_ANALYSIS_GROUP_COLS = ["phase_label", "position"]

CHART_VARIANTS = ["v0_dummy", "v1_unweighted", "v2_weighted", "v3_weighted_interactions"]
CHART_NAMES = ["calibration_curve.png", "precision_recall_curve.png", "roc_curve.png", "prediction_distribution.png"]


def build_competition_map() -> dict[int, str]:
    """match_id -> "{competition_name} {season}" from data/raw/matches/*.json."""
    comp: dict[int, str] = {}
    for path in glob.glob(MATCHES_GLOB):
        matches = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in matches:
            comp[m["match_id"]] = f"{m['competition_name']} {m['season']}"
    return comp


def item1_significance_test(df_all: pd.DataFrame) -> dict:
    """Paired fold-level PR-AUC comparison: v1 vs v2, v1 vs v3.

    Refits all three variants on each of the 5 canonical CV folds (same
    folds/assignment as the original baseline run) and runs a paired
    t-test + Wilcoxon signed-rank test on the per-fold PR-AUC differences.
    """
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)

    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    df = trainval_df.loc[folds["row_index"]].copy()
    df["fold"] = folds["fold"].to_numpy()
    y_all = df[tb.TARGET_COL].to_numpy()

    variants = ["v1_unweighted", *VARIANTS_FOR_SIGNIFICANCE]
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

    result: dict = {"per_fold_average_precision": per_fold_ap}
    a = np.array(per_fold_ap["v1_unweighted"])
    for other in VARIANTS_FOR_SIGNIFICANCE:
        b = np.array(per_fold_ap[other])
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        if len(set(diff.round(12))) > 1:
            w_stat, w_p = stats.wilcoxon(a, b)
        else:
            w_stat, w_p = float("nan"), float("nan")
        result[f"v1_vs_{other}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(p_val),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "n_folds": int(len(a)),
        }
    return result


def item2_tournament_stratified(df_all: pd.DataFrame, comp_map: dict[int, str]) -> dict:
    """v1, fit once on all train+val rows (same fit as the prompt-37 held-out
    readout), scored on the held-out test set, broken down by tournament."""
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)

    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test_all = test_df[tb.TARGET_COL].to_numpy()
    score_test, _, _ = tb.fit_predict_fold(trainval_df, test_df, y_trainval, y_test_all, "v1_unweighted")

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


def item3_player_disjoint(df_all: pd.DataFrame) -> dict:
    """Actual player-disjoint train/test run: GroupKFold on player_id (not
    match_id), 5 folds, v1 spec, on train+val rows only -- the frozen
    held-out test set (23 matches) is excluded entirely and never touched."""
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)

    if "player_id" not in trainval_df.columns:
        return {"error": "player_id column not found in the dataset"}

    groups = trainval_df["player_id"].astype(str)
    y = trainval_df[tb.TARGET_COL].to_numpy()
    n_players = int(groups.nunique())
    gkf = GroupKFold(n_splits=5)

    fold_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(trainval_df, y, groups)):
        train_df, test_df = trainval_df.iloc[train_idx], trainval_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        overlap = set(train_df["player_id"]) & set(test_df["player_id"])
        assert len(overlap) == 0, f"GroupKFold leaked {len(overlap)} players into fold {fold_id}"

        score, _, _ = tb.fit_predict_fold(train_df, test_df, y_train, y_test, "v1_unweighted")
        m = classification_metrics(y_test, score)
        m["fold"] = fold_id
        m["n_train"] = int(len(train_df))
        m["n_test"] = int(len(test_df))
        m["n_test_players"] = int(test_df["player_id"].nunique())
        m["player_overlap_train_test"] = int(len(overlap))
        fold_rows.append(m)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in fold_df.columns if c not in {"fold", "n_train", "n_test", "n_test_players", "player_overlap_train_test"}]
    return {
        "n_players_total": n_players,
        "fold_metrics": fold_df.to_dict(orient="records"),
        "mean": fold_df[metric_cols].mean().to_dict(),
        "std": fold_df[metric_cols].std(ddof=0).to_dict(),
        "max_player_overlap_any_fold": int(fold_df["player_overlap_train_test"].max()),
        "player_disjoint_confirmed_every_fold": bool((fold_df["player_overlap_train_test"] == 0).all()),
    }


def item4_error_analysis(df_all: pd.DataFrame) -> dict:
    """Where does v1 (fit on train+val, scored on held-out test, same fit as
    item 2) miss, by phase_label and by position (see module docstring for
    why position replaces the requested defender_functional_role)."""
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)

    y_trainval = trainval_df[tb.TARGET_COL].to_numpy()
    y_test_all = test_df[tb.TARGET_COL].to_numpy()
    score_test, _, _ = tb.fit_predict_fold(trainval_df, test_df, y_trainval, y_test_all, "v1_unweighted")

    test_df = test_df.copy()
    test_df["_score"] = score_test
    test_df["_y"] = test_df[tb.TARGET_COL]

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

    print("[1/5] paired significance test (v1 vs v2, v1 vs v3)...")
    sig = item1_significance_test(df_all)
    (OUT_DIR / "significance_v1_vs_v2_v3.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")

    print("[2/5] tournament-stratified held-out test check...")
    tourn = item2_tournament_stratified(df_all, comp_map)
    (OUT_DIR / "tournament_stratified_v1.json").write_text(json.dumps(tourn, indent=2, default=str), encoding="utf-8")

    print("[3/5] player-disjoint train/test run...")
    disjoint = item3_player_disjoint(df_all)
    (OUT_DIR / "player_disjoint_v1.json").write_text(json.dumps(disjoint, indent=2, default=str), encoding="utf-8")

    print("[4/5] error analysis by phase_label / position...")
    errs = item4_error_analysis(df_all)
    (OUT_DIR / "error_analysis_v1.json").write_text(json.dumps(errs, indent=2, default=str), encoding="utf-8")

    print("[5/5] chart integrity pass...")
    charts = item5_chart_integrity()
    print("  all_ok:", charts["all_ok"], "issues:", charts["issues"] or "none")

    print("\n=== SIGNIFICANCE (v1 vs v2/v3, paired over 5 canonical CV folds) ===")
    for other in VARIANTS_FOR_SIGNIFICANCE:
        r = sig[f"v1_vs_{other}"]
        print(f"v1_vs_{other}: mean_diff={r['mean_diff']:+.4f} std_diff={r['std_diff']:.4f} "
              f"paired_t_p={r['paired_t_pvalue']:.4f} wilcoxon_p={r['wilcoxon_pvalue']}")

    print("\n=== TOURNAMENT (held-out test, v1) ===")
    for comp, m in tourn["held_out_test_by_tournament"].items():
        print(comp, m)

    print("\n=== PLAYER-DISJOINT (v1, 5-fold GroupKFold on player_id) ===")
    print("n_players_total:", disjoint.get("n_players_total"))
    print("mean:", disjoint.get("mean"))
    print("player_disjoint_confirmed_every_fold:", disjoint.get("player_disjoint_confirmed_every_fold"))

    print("\n=== ERROR ANALYSIS: phase_label (held-out test) ===")
    for row in errs["phase_label"]:
        print(row)
    print("\n=== ERROR ANALYSIS: position (held-out test; defender_functional_role not in ACTIVE dataset, see docstring) ===")
    for row in errs["position"]:
        print(row)

    print("\nDone. Outputs written under", OUT_DIR)


if __name__ == "__main__":
    main()
