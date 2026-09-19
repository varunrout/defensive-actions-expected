"""Full promotion validation (prompt 52): does p1e_gradient_boosting_calibrated
replace p1_unweighted as the passive-binary leg's standing reference model?

Mirrors the active leg's Prompts 43 (tournament + error-slice + sanity-check
promotion framework) and 46 (raw-vs-calibrated decision logic, and the trap of
an aggregate calibration number hiding a slice-level problem), adapted for
this leg's real differences: no player-disjoint recheck (no player_id -- the
cluster-by-event_id bootstrap from Prompt 50 is used everywhere the active
leg used one), and error slices are phase_label / defender_functional_role
(the passive leg's real per-row categorical dimensions, confirmed against
src/eda/feature_config.py's PASSIVE["categorical"], not player-identity
slices this leg does not have).

This script does NOT retrain p1, p1c, or p1e_gradient_boosting_calibrated.
Every score used here comes from either:
  (a) an already-fitted, already-saved joblib model, scored via a freshly
      *fit* (not trained-from-scratch-and-tuned) preprocessing builder --
      OneHotEncoder/StandardScaler/PolynomialFeatures fits are cheap
      deterministic preprocessing, not model training, and reproduce
      byte-identical inputs to what the saved model was originally trained
      on; or
  (b) an existing validation JSON from Prompt 49/50/51, read and reused
      rather than recomputed (p1's tournament/error-slice numbers).

One deliberate exception, flagged explicitly where it happens (item 4): true
CV-OOF predictions for p1e_gradient_boosting_calibrated were never persisted
as data in Prompt 51 (only turned into chart PNGs), and regenerating them
would require refitting the calibrated GBM across 5 folds -- exactly the
retraining this prompt's constraints prohibit. In-sample train+val
predictions from the already-fitted FINAL model are used instead for that
one marginal-shape check, explicitly labelled as in-sample, not OOF.

Usage:
    python scripts/models/validate_p1e_calibrated_promotion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_p1c_systematic_interactions as p1c_mod  # noqa: E402
import train_p1d_random_forest as p1d_mod  # noqa: E402
import train_passive_binary_baseline as tb  # noqa: E402
import validate_passive_binary_baselines as vpb  # noqa: E402
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
MODELS_DIR = tb.MODELS_DIR

P1_JOBLIB = MODELS_DIR / "p1_unweighted.joblib"
P1C_JOBLIB = MODELS_DIR / "p1c_systematic_interactions.joblib"
P1E_CAL_JOBLIB = MODELS_DIR / "p1e_gradient_boosting_calibrated.joblib"

# Confirmed against src/eda/feature_config.py PASSIVE["categorical"] -- the
# passive leg's real per-row categorical slice dimensions (no player
# identity/position field exists on this leg).
ERROR_ANALYSIS_GROUP_COLS = ["phase_label", "defender_functional_role"]

# Top 6 features by GBM gain importance, read from p1e_gradient_boosting.json
# (Prompt 51's own already-computed ranking, not recomputed here).
BEHAVIORAL_SANITY_FEATURES = [
    "defender_x", "ball_x", "top_option_3_threat_score",
    "phase_label", "ball_y", "top_option_1_threat_score",
]
N_BINS = 10


def score_with_saved_model(trainval_df: pd.DataFrame, test_df: pd.DataFrame, builder, joblib_path: Path) -> np.ndarray:
    """Fit a fresh preprocessing builder (cheap, deterministic; NOT model
    training) on train+val, transform the held-out test set, and score it
    with an already-fitted saved model -- reproduces that model's original
    held-out readout without refitting the model itself."""
    builder.fit(trainval_df)
    x_test, _ = builder.transform(test_df)
    model = joblib.load(joblib_path)
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    df_all = tb.load_data()
    load_canonical_split()
    comp_map = vpb.build_competition_map()

    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    test_df = df_all.loc[test_mask].reset_index(drop=True)
    y_test = test_df[tb.TARGET_COL].to_numpy()
    print(f"[split] train+val rows={len(trainval_df)}; held-out test rows={len(test_df)}")

    # ---- Item 1: tournament-stratified check ----------------------------
    print("[1/5] tournament-stratified check (p1c, p1e_gradient_boosting_calibrated)...")
    p1c_tourn_path = VALIDATION_DIR / "tournament_stratified_p1c.json"
    score_p1c = score_with_saved_model(trainval_df, test_df, p1c_mod.InteractionDesignMatrixBuilder(), P1C_JOBLIB)
    score_p1e_cal = score_with_saved_model(trainval_df, test_df, p1d_mod.RFDesignMatrixBuilder(), P1E_CAL_JOBLIB)

    tourn_p1c = vpb.item2_tournament_stratified(df_all, comp_map, test_df, score_p1c)
    p1c_tourn_path.write_text(json.dumps(tourn_p1c, indent=2, default=str), encoding="utf-8")

    tourn_p1e_cal = vpb.item2_tournament_stratified(df_all, comp_map, test_df, score_p1e_cal)
    p1_tourn = json.loads((VALIDATION_DIR / "tournament_stratified_p1.json").read_text(encoding="utf-8"))
    tourn_output = {
        "p1_unweighted": p1_tourn["held_out_test_by_tournament"],
        "p1c_systematic_interactions": tourn_p1c["held_out_test_by_tournament"],
        "p1e_gradient_boosting_calibrated": tourn_p1e_cal["held_out_test_by_tournament"],
        "train_val_match_counts_by_tournament": tourn_p1e_cal["train_val_match_counts_by_tournament"],
        "test_match_counts_by_tournament": tourn_p1e_cal["test_match_counts_by_tournament"],
        "note": "Only FIFA World Cup 2022 and UEFA Euro 2024 are represented in this leg's matches "
                "(UEFA Euro 2020 contributes zero rows, confirmed in Prompt 50 -- reused, not re-derived).",
    }
    (VALIDATION_DIR / "tournament_stratified_p1e_calibrated.json").write_text(json.dumps(tourn_output, indent=2), encoding="utf-8")

    # ---- Item 2: error-slice analysis ------------------------------------
    print("[2/5] error-slice analysis by phase_label / defender_functional_role...")
    errs_p1c = vpb.item4_error_analysis(test_df, y_test, score_p1c)
    (VALIDATION_DIR / "error_analysis_p1c.json").write_text(json.dumps(errs_p1c, indent=2, default=str), encoding="utf-8")

    errs_p1e_cal = vpb.item4_error_analysis(test_df, y_test, score_p1e_cal)
    (VALIDATION_DIR / "error_analysis_p1e_calibrated.json").write_text(json.dumps(errs_p1e_cal, indent=2, default=str), encoding="utf-8")

    errs_p1 = json.loads((VALIDATION_DIR / "error_analysis_p1.json").read_text(encoding="utf-8"))

    # -- 3-way comparison table per slice: does p1e_calibrated improve, tie, or regress vs p1 and p1c? --
    def slice_lookup(errs: dict, group_col: str) -> dict[str, dict]:
        return {row[group_col]: row for row in errs[group_col]}

    comparison_by_group: dict[str, list[dict]] = {}
    for group_col in ERROR_ANALYSIS_GROUP_COLS:
        p1_by_key = slice_lookup(errs_p1, group_col)
        p1c_by_key = slice_lookup(errs_p1c, group_col)
        p1e_by_key = slice_lookup(errs_p1e_cal, group_col)
        rows = []
        for key, p1e_row in p1e_by_key.items():
            p1_row = p1_by_key.get(key, {})
            p1c_row = p1c_by_key.get(key, {})
            entry = {
                group_col: key,
                "rows": p1e_row["rows"],
                "p1_average_precision": p1_row.get("average_precision"),
                "p1c_average_precision": p1c_row.get("average_precision"),
                "p1e_calibrated_average_precision": p1e_row.get("average_precision"),
                "p1_calibration_gap": p1_row.get("calibration_gap"),
                "p1c_calibration_gap": p1c_row.get("calibration_gap"),
                "p1e_calibrated_calibration_gap": p1e_row.get("calibration_gap"),
            }
            if entry["p1e_calibrated_average_precision"] is not None and entry["p1c_average_precision"] is not None:
                entry["p1e_vs_p1c_ap_delta"] = entry["p1e_calibrated_average_precision"] - entry["p1c_average_precision"]
            if entry["p1e_calibrated_average_precision"] is not None and entry["p1_average_precision"] is not None:
                entry["p1e_vs_p1_ap_delta"] = entry["p1e_calibrated_average_precision"] - entry["p1_average_precision"]
            if entry["p1e_calibrated_calibration_gap"] is not None and entry["p1c_calibration_gap"] is not None:
                entry["p1e_vs_p1c_calibration_gap_delta"] = abs(entry["p1e_calibrated_calibration_gap"]) - abs(entry["p1c_calibration_gap"])
            rows.append(entry)
        comparison_by_group[group_col] = sorted(rows, key=lambda r: -r["rows"])

    # ---- Item 3: slice-level calibration check (folded into the same output) ----
    print("[3/5] slice-level calibration comparison (the Prompt 46 trap check)...")
    slice_calibration_output = {
        "note": (
            "Aggregate ECE alone (Prompt 51: p1e_calibrated 0.0044 vs p1c 0.0043 held-out, near-tied) "
            "is NOT used to settle this -- per-slice calibration_gap (mean predicted - positive rate) "
            "is compared directly below, the same check that caught the active leg's raw v1e looking "
            "well-calibrated in aggregate while being 6-19x worse than v1c at the phase_label slice level."
        ),
        "by_group": comparison_by_group,
    }

    # ---- Item 4: behavioral sanity check ---------------------------------
    print("[4/5] behavioral sanity check on the 6 dominant features (in-sample train+val)...")
    print("  [note] true CV-OOF predictions for p1e_gradient_boosting_calibrated were not persisted "
          "as data in Prompt 51 (only chart PNGs) -- regenerating them would require refitting the "
          "calibrated GBM across 5 folds, which this prompt's constraints prohibit. Using in-sample "
          "train+val predictions from the already-fitted FINAL model instead (explicitly in-sample, "
          "not OOF).")
    rf_builder = p1d_mod.RFDesignMatrixBuilder().fit(trainval_df)
    x_trainval, _ = rf_builder.transform(trainval_df)
    p1e_cal_model = joblib.load(P1E_CAL_JOBLIB)
    score_trainval_insample = p1e_cal_model.predict_proba(x_trainval)[:, 1]

    behavioral_sanity: dict = {
        "in_sample_not_oof": True,
        "reason": "See module docstring / [4/5] log line -- true CV-OOF predictions were never persisted for this variant.",
        "n_rows": len(trainval_df),
        "features": {},
    }
    for feat in BEHAVIORAL_SANITY_FEATURES:
        col = trainval_df[feat]
        if pd.api.types.is_numeric_dtype(col):
            bins = pd.qcut(col, q=N_BINS, duplicates="drop")
            table = pd.DataFrame({"bin": bins, "score": score_trainval_insample, "y": trainval_df[tb.TARGET_COL]})
            grouped = table.groupby("bin", observed=True).agg(
                n=("score", "size"), mean_predicted=("score", "mean"), positive_rate=("y", "mean")
            ).reset_index()
            grouped["bin"] = grouped["bin"].astype(str)
            behavioral_sanity["features"][feat] = {"binning": "qcut_10", "table": grouped.to_dict(orient="records")}
        else:
            table = pd.DataFrame({"cat": col, "score": score_trainval_insample, "y": trainval_df[tb.TARGET_COL]})
            grouped = table.groupby("cat", observed=True).agg(
                n=("score", "size"), mean_predicted=("score", "mean"), positive_rate=("y", "mean")
            ).reset_index().sort_values("cat")
            behavioral_sanity["features"][feat] = {"binning": "categorical", "table": grouped.to_dict(orient="records")}

    (VALIDATION_DIR / "behavioral_sanity_p1e_calibrated.json").write_text(json.dumps(behavioral_sanity, indent=2, default=str), encoding="utf-8")

    # ---- Item 5: cluster bootstrap vs p1c --------------------------------
    print("[5/5] cluster-by-event_id bootstrap, p1e_gradient_boosting_calibrated vs p1c (held-out test)...")
    boot = vpb.item3_cluster_bootstrap(
        y_test, score_p1e_cal, score_p1c, test_df[vpb.EVENT_GROUP_COL].to_numpy(),
        label_a="p1e_gradient_boosting_calibrated", label_b="p1c_systematic_interactions",
    )
    (VALIDATION_DIR / "cluster_bootstrap_p1e_calibrated_vs_p1c.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")

    (VALIDATION_DIR / "slice_level_calibration_p1e_calibrated_vs_p1c_vs_p1.json").write_text(
        json.dumps(slice_calibration_output, indent=2, default=str), encoding="utf-8"
    )

    # ---- Console summary --------------------------------------------------
    print("\n=== TOURNAMENT (held-out test) ===")
    for comp in tourn_output["p1_unweighted"]:
        print(comp, "p1:", tourn_output["p1_unweighted"][comp].get("average_precision"),
              "p1c:", tourn_output["p1c_systematic_interactions"].get(comp, {}).get("average_precision"),
              "p1e_cal:", tourn_output["p1e_gradient_boosting_calibrated"].get(comp, {}).get("average_precision"))

    print("\n=== ERROR-SLICE + CALIBRATION COMPARISON ===")
    for group_col, rows in comparison_by_group.items():
        print(f"--- {group_col} ---")
        for r in rows:
            print(f"  {r[group_col]}: n={r['rows']} p1_AP={r['p1_average_precision']} "
                  f"p1c_AP={r['p1c_average_precision']} p1e_cal_AP={r['p1e_calibrated_average_precision']} "
                  f"| calib_gap p1={r['p1_calibration_gap']} p1c={r['p1c_calibration_gap']} "
                  f"p1e_cal={r['p1e_calibrated_calibration_gap']}")

    print("\n=== CLUSTER BOOTSTRAP (p1e_calibrated vs p1c) ===")
    diff_key = "p1e_gradient_boosting_calibrated_minus_p1c_systematic_interactions_average_precision"
    print("diff naive:", boot[diff_key]["naive_row_level"])
    print("diff cluster:", boot[diff_key]["cluster_by_event"])
    print("width ratio:", boot[diff_key]["cluster_to_naive_width_ratio"])
    print("naive CI excludes zero:", boot[diff_key]["naive_ci_excludes_zero"])
    print("cluster CI excludes zero:", boot[diff_key]["cluster_ci_excludes_zero"])

    print("\nDone. Outputs written under", VALIDATION_DIR)


if __name__ == "__main__":
    main()
