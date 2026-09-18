"""Full promotion validation for v1e_gradient_boosting (prompt 46).

Rung 4's own ladder gate (prompt 45 -- paired significance, held-out-test
confirmation, player-disjoint recheck) answered "is this rung's change
real?". This script answers the bigger question, same depth of scrutiny
v1c went through in prompt 43 before it replaced v1_unweighted as the
standing reference model: should v1e_gradient_boosting now replace
v1c_systematic_interactions?

Three checks, all on train+val 5-fold canonical CV out-of-fold (OOF)
predictions -- the same population prompt 43 used for v1c's promotion
audit, and the held-out test set is not read again for any purpose here:

  1. Tournament-stratified PR-AUC (WC2022 vs Euro2024), v1e vs v1c vs v1d.
  2. Error-slice breakdown by phase_label and position, v1e vs v1c
     (and v1, for the original weak-spot lineage), with particular
     attention to the Goalkeeper/Right Attacking Midfield calibration
     caveat flagged when v1c was promoted.
  3. Behavioral sanity check: marginal mean-predicted-probability shape
     across 10 quantile bins for v1e's top 8 gain-importance features,
     cross-checked against EDA-documented shapes (especially the 4
     features already confirmed U-shaped by rung 1's quadratic terms and
     v1c's surviving squared L1 terms).

Reuse, not re-run: v1's and v1c's CV-OOF breakdowns are loaded directly
from prompt 43's tournament_stratified_v1c.json / error_analysis_v1c.json
rather than recomputed. v1d has no saved per-row OOF (only aggregate
metrics were persisted in prompt 44), so its OOF is regenerated here at
its already-locked hyperparameters (n_estimators=600, max_depth=None,
min_samples_leaf=5 -- no re-tuning) purely to build the tournament
comparison prompt 46 asks for; v1e's OOF is regenerated the same way, at
its already-locked hyperparameters from prompt 45
(learning_rate=0.01, num_leaves=63, min_child_samples=30). Neither
"recompute" changes any existing artifact or output file -- this mirrors
exactly what prompt 43 did to get v1's own OOF numbers.

Usage:
    python scripts/validate_v1e_promotion.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
import train_v1c_systematic_interactions as v1c_mod  # noqa: E402
import train_v1d_random_forest as v1d_mod  # noqa: E402
import train_v1e_gradient_boosting as v1e_mod  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_grouped_folds, canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
MATCHES_GLOB = str(REPO_ROOT / "data" / "raw" / "matches" / "*.json")

V1E_PARAMS = {"learning_rate": 0.01, "num_leaves": 63, "min_child_samples": 30}
V1D_PARAMS = {"n_estimators": 600, "max_depth": None, "min_samples_leaf": 5}
V1C_C = 0.1

TOP8_GAIN_FEATURES = [
    "match_time_seconds", "distance_to_attacking_box", "attacking_goal_centrality",
    "nearest_attacker_distance", "defender_spread", "attacker_spread",
    "possession_elapsed_seconds", "attacker_defender_ratio",
]
U_SHAPED_DOCUMENTED = {
    "defender_spread", "distance_to_attacking_box", "visible_defender_count",
    "defenders_between_ball_and_attacking_goal",
}


def build_competition_map() -> dict[int, str]:
    comp: dict[int, str] = {}
    for path in glob.glob(MATCHES_GLOB):
        matches = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in matches:
            comp[m["match_id"]] = f"{m['competition_name']} {m['season']}"
    return comp


def ordered_trainval_df(trainval_df: pd.DataFrame) -> pd.DataFrame:
    """Same row ordering canonical_grouped_folds imposes -- what every
    run_cv/run_cv_* function in this project already uses internally."""
    folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
    return trainval_df.loc[folds["row_index"]]


def tournament_breakdown(df_ordered: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray, comp_map: dict) -> dict:
    work = df_ordered.copy()
    work["_y"] = y_true
    work["_score"] = scores
    work["competition"] = work[tb.GROUP_COL].map(comp_map)
    out: dict = {}
    for comp, sub in work.groupby("competition", dropna=False):
        y = sub["_y"].to_numpy()
        s = sub["_score"].to_numpy()
        matches = sub[tb.GROUP_COL].nunique()
        entry: dict = {"rows": int(len(sub)), "matches": int(matches)}
        if matches < 2 or y.sum() < 2:
            entry["note"] = "too few matches/positives for a stable readout"
        else:
            entry.update(classification_metrics(y, s))
        out[str(comp)] = entry
    return out


def error_slice_breakdown(df_ordered: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray) -> dict:
    work = df_ordered.copy()
    work["_y"] = y_true
    work["_score"] = scores
    out: dict = {}
    for group_col in ["phase_label", "position"]:
        rows = []
        for val, sub in work.groupby(group_col, dropna=False):
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


def behavioral_sanity(df_ordered: pd.DataFrame, scores: np.ndarray, features: list[str]) -> dict:
    out: dict = {}
    for feat in features:
        values = df_ordered[feat].to_numpy(dtype=float)
        finite = np.isfinite(values)
        try:
            bins = pd.qcut(values[finite], q=10, duplicates="drop")
        except ValueError:
            bins = pd.cut(values[finite], bins=10)
        bin_df = pd.DataFrame({"bin": bins, "score": scores[finite], "value": values[finite]})
        grouped = bin_df.groupby("bin", observed=True)
        table = []
        for interval, sub in grouped:
            table.append({
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "n_rows": int(len(sub)),
                "mean_feature_value": float(sub["value"].mean()),
                "mean_predicted_probability": float(sub["score"].mean()),
            })
        table.sort(key=lambda r: r["bin_left"])
        out[feat] = {
            "n_bins": len(table),
            "n_rows_with_finite_value": int(finite.sum()),
            "n_rows_missing_or_nonfinite": int((~finite).sum()),
            "bins": table,
            "documented_u_shaped": feat in U_SHAPED_DOCUMENTED,
        }
    return out


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = tb.load_data()
    load_canonical_split()
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    print(f"[split] train+val rows={len(trainval_df)} matches={trainval_df[tb.GROUP_COL].nunique()} "
          "(held-out test set NOT touched in this script)")

    comp_map = build_competition_map()
    trainval_comp = trainval_df[tb.GROUP_COL].map(comp_map)
    tournaments_present = sorted(trainval_comp.dropna().unique().tolist())
    print(f"[tournaments] present in train+val: {tournaments_present}")
    assert set(tournaments_present) == {"FIFA World Cup 2022", "UEFA Euro 2024"}, (
        f"Expected exactly WC2022 + Euro2024, got {tournaments_present} -- re-checked, not assumed."
    )

    df_ordered = ordered_trainval_df(trainval_df)

    # -- Reuse v1's and v1c's existing CV-OOF breakdowns from prompt 43 -- no recompute. --
    prior_tournament = json.loads((VALIDATION_DIR / "tournament_stratified_v1c.json").read_text(encoding="utf-8"))
    prior_error = json.loads((VALIDATION_DIR / "error_analysis_v1c.json").read_text(encoding="utf-8"))
    v1_tourn = prior_tournament["v1_unweighted_by_tournament"]
    v1c_tourn = prior_tournament["v1c_systematic_interactions_by_tournament"]
    v1_err = prior_error["v1_unweighted"]
    v1c_err = prior_error["v1c_systematic_interactions"]

    # -- Regenerate v1d's and v1e's CV-OOF at their already-locked hyperparameters
    #    (no re-tuning) -- neither has saved per-row OOF from prompts 44/45. --
    print("[cv] v1d_random_forest OOF (train+val, 5 canonical folds, locked params)...")
    v1d_cv = v1d_mod.run_cv_rf(trainval_df, V1D_PARAMS)
    print("[cv] v1e_gradient_boosting OOF (train+val, 5 canonical folds, locked params)...")
    v1e_cv = v1e_mod.run_cv_gbm(trainval_df, V1E_PARAMS)

    y_v1d = v1d_cv["y_all"]
    y_v1e = v1e_cv["y_all"]
    assert np.array_equal(y_v1d, y_v1e), "v1d and v1e CV row ordering diverged -- folds must match exactly"
    assert len(y_v1e) == len(df_ordered), "OOF length mismatch vs reconstructed df_ordered"

    print("[tournament] breakdown for v1d and v1e (v1/v1c reused from prompt 43)...")
    v1d_tourn = tournament_breakdown(df_ordered, y_v1d, v1d_cv["oof_scores"], comp_map)
    v1e_tourn = tournament_breakdown(df_ordered, y_v1e, v1e_cv["oof_scores"], comp_map)

    train_composition = {str(k): int(v) for k, v in trainval_df.assign(competition=trainval_comp)
                          .groupby("competition")[tb.GROUP_COL].nunique().items()}

    tournament_output = {
        "population": "train+val, 5-fold canonical CV out-of-fold predictions (NOT held-out test; "
                       "same population and method as tournament_stratified_v1c.json)",
        "v1e_params": V1E_PARAMS,
        "v1d_params_for_reference": V1D_PARAMS,
        "train_val_match_counts_by_tournament": train_composition,
        "v1_unweighted_by_tournament": v1_tourn,
        "v1c_systematic_interactions_by_tournament": v1c_tourn,
        "v1d_random_forest_by_tournament": v1d_tourn,
        "v1e_gradient_boosting_by_tournament": v1e_tourn,
    }
    (VALIDATION_DIR / "tournament_stratified_v1e.json").write_text(
        json.dumps(tournament_output, indent=2, default=str), encoding="utf-8"
    )

    print("[error-slice] phase_label / position breakdown for v1e (v1/v1c reused from prompt 43)...")
    v1e_err = error_slice_breakdown(df_ordered, y_v1e, v1e_cv["oof_scores"])
    error_output = {
        "population": "train+val, 5-fold canonical CV out-of-fold predictions (same population/method as error_analysis_v1c.json)",
        "v1e_params": V1E_PARAMS,
        "v1_unweighted": v1_err,
        "v1c_systematic_interactions": v1c_err,
        "v1e_gradient_boosting": v1e_err,
    }
    (VALIDATION_DIR / "error_analysis_v1e.json").write_text(
        json.dumps(error_output, indent=2, default=str), encoding="utf-8"
    )

    print("[behavioral sanity] marginal mean-predicted-probability shape, top 8 gain features...")
    sanity = behavioral_sanity(df_ordered, v1e_cv["oof_scores"], TOP8_GAIN_FEATURES)
    behavioral_output = {
        "population": "train+val, 5-fold canonical CV out-of-fold predictions",
        "v1e_params": V1E_PARAMS,
        "note": "Marginal view only -- one feature binned at a time, nothing else held fixed. "
                "Not a controlled/partial-dependence view.",
        "features": sanity,
    }
    (VALIDATION_DIR / "behavioral_sanity_v1e.json").write_text(
        json.dumps(behavioral_output, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== TOURNAMENT (train+val CV OOF) ===")
    for comp in v1_tourn:
        print(f"{comp}: v1={v1_tourn[comp].get('average_precision'):.4f} "
              f"v1c={v1c_tourn[comp].get('average_precision'):.4f} "
              f"v1d={v1d_tourn[comp].get('average_precision'):.4f} "
              f"v1e={v1e_tourn[comp].get('average_precision'):.4f}")

    print("\n=== ERROR SLICES: phase_label (v1c vs v1e, train+val CV OOF) ===")
    v1c_by_key = {r["phase_label"]: r for r in v1c_err["phase_label"]}
    v1e_by_key = {r["phase_label"]: r for r in v1e_err["phase_label"]}
    for key in v1c_by_key:
        a, b = v1c_by_key[key], v1e_by_key.get(key, {})
        print(f"{key}: rows={a['rows']} v1c AP={a.get('average_precision')} v1e AP={b.get('average_precision')} "
              f"| v1c cal_gap={a['calibration_gap']:+.4f} v1e cal_gap={b.get('calibration_gap', float('nan')):+.4f}")

    print("\n=== ERROR SLICES: Goalkeeper / Right Attacking Midfield caveat check (position) ===")
    v1c_pos = {r["position"]: r for r in v1c_err["position"]}
    v1e_pos = {r["position"]: r for r in v1e_err["position"]}
    for key in ["Goalkeeper", "Right Attacking Midfield"]:
        a, b = v1c_pos.get(key, {}), v1e_pos.get(key, {})
        print(f"{key}: v1c cal_gap={a.get('calibration_gap')} v1e cal_gap={b.get('calibration_gap')}")

    print("\n=== BEHAVIORAL SANITY: shape summary (first/last bin mean predicted prob) ===")
    for feat, info in sanity.items():
        bins = info["bins"]
        first, last = bins[0], bins[-1]
        mid = bins[len(bins) // 2]
        print(f"{feat} (documented U-shaped={info['documented_u_shaped']}): "
              f"first_bin={first['mean_predicted_probability']:.4f} "
              f"mid_bin={mid['mean_predicted_probability']:.4f} "
              f"last_bin={last['mean_predicted_probability']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
