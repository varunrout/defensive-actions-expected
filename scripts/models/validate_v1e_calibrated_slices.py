"""Follow-up to validate_v1e_promotion.py: does v1e_gradient_boosting_calibrated
fix the slice-specific calibration problem the raw variant showed?

validate_v1e_promotion.py found that raw v1e_gradient_boosting's held-out
aggregate ECE (0.0068) looked better than v1c's (0.0103), but on CV-OOF
error slices the raw variant's calibration gap is 6-19x larger than v1c's
on 6 of 7 phase_label slices -- the aggregate number did not generalize.
This script checks whether the isotonic-calibrated variant (locked at
n_estimators=382, prompt 45's already-chosen recipe, no re-tuning) fixes
that, using the exact same tournament/error-slice methods and CV-OOF
population as validate_v1e_promotion.py, so results merge directly.

Usage:
    python scripts/validate_v1e_calibrated_slices.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/<subfolder>/ -> repo root (prompt 48 move: was parents[1] at scripts/ root)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))  # prompt 48: sibling ladder scripts now live in scripts/models/, not scripts/
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as tb  # noqa: E402
import train_v1e_gradient_boosting as v1e_mod  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402
from validate_v1e_promotion import (  # noqa: E402
    V1E_PARAMS, VALIDATION_DIR, build_competition_map, error_slice_breakdown,
    ordered_trainval_df, tournament_breakdown,
)

FIXED_N_ESTIMATORS = 382
CALIB_METHOD = "isotonic"


def main() -> None:
    df_all = tb.load_data()
    load_canonical_split()
    test_mask = canonical_test_mask(df_all, group_col=tb.GROUP_COL)
    trainval_df = df_all.loc[~test_mask].reset_index(drop=True)
    df_ordered = ordered_trainval_df(trainval_df)
    comp_map = build_competition_map()

    print(f"[cv] v1e_gradient_boosting_calibrated OOF ({CALIB_METHOD}, n_estimators={FIXED_N_ESTIMATORS} fixed)...")
    calib_cv = v1e_mod.run_cv_gbm_calibrated(trainval_df, V1E_PARAMS, FIXED_N_ESTIMATORS, CALIB_METHOD)
    y = calib_cv["y_all"]
    scores = calib_cv["oof_scores"]

    tourn = tournament_breakdown(df_ordered, y, scores, comp_map)
    err = error_slice_breakdown(df_ordered, y, scores)

    # -- Merge into the existing tournament/error JSONs from validate_v1e_promotion.py. --
    tourn_path = VALIDATION_DIR / "tournament_stratified_v1e.json"
    tourn_data = json.loads(tourn_path.read_text(encoding="utf-8"))
    tourn_data["v1e_gradient_boosting_calibrated_by_tournament"] = tourn
    tourn_data["v1e_calibrated_method"] = CALIB_METHOD
    tourn_data["v1e_calibrated_n_estimators_fixed"] = FIXED_N_ESTIMATORS
    tourn_path.write_text(json.dumps(tourn_data, indent=2, default=str), encoding="utf-8")

    err_path = VALIDATION_DIR / "error_analysis_v1e.json"
    err_data = json.loads(err_path.read_text(encoding="utf-8"))
    err_data["v1e_gradient_boosting_calibrated"] = err
    err_data["v1e_calibrated_method"] = CALIB_METHOD
    err_data["v1e_calibrated_n_estimators_fixed"] = FIXED_N_ESTIMATORS
    err_path.write_text(json.dumps(err_data, indent=2, default=str), encoding="utf-8")

    print("\n=== TOURNAMENT: v1e_calibrated ===")
    for comp, m in tourn.items():
        print(comp, m.get("average_precision"), m.get("expected_calibration_error"))

    print("\n=== ERROR SLICES: phase_label calibration gap, v1c vs v1e_raw vs v1e_calibrated ===")
    v1c_err = json.loads((VALIDATION_DIR / "error_analysis_v1c.json").read_text(encoding="utf-8"))["v1c_systematic_interactions"]
    v1c_by_key = {r["phase_label"]: r for r in v1c_err["phase_label"]}
    v1e_raw_by_key = {r["phase_label"]: r for r in err_data["v1e_gradient_boosting"]["phase_label"]}
    v1e_cal_by_key = {r["phase_label"]: r for r in err["phase_label"]}
    for key in v1c_by_key:
        c, raw, cal = v1c_by_key[key], v1e_raw_by_key.get(key, {}), v1e_cal_by_key.get(key, {})
        print(f"{key}: v1c={c['calibration_gap']:+.4f} v1e_raw={raw.get('calibration_gap', float('nan')):+.4f} "
              f"v1e_calibrated={cal.get('calibration_gap', float('nan')):+.4f}")

    print("\n=== Goalkeeper / Right Attacking Midfield (position) ===")
    v1c_pos = {r["position"]: r for r in v1c_err["position"]}
    v1e_raw_pos = {r["position"]: r for r in err_data["v1e_gradient_boosting"]["position"]}
    v1e_cal_pos = {r["position"]: r for r in err["position"]}
    for key in ["Goalkeeper", "Right Attacking Midfield"]:
        print(key, "v1c=", v1c_pos.get(key, {}).get("calibration_gap"),
              "v1e_raw=", v1e_raw_pos.get(key, {}).get("calibration_gap"),
              "v1e_calibrated=", v1e_cal_pos.get(key, {}).get("calibration_gap"))

    print("\nDone.")


if __name__ == "__main__":
    main()
