"""Dashboard data export: curated matches re-scored across all 6 reference models (prompt 83).

Produces dashboard_data/match_explorer/{match_id}.json -- one file per curated
match -- for the separate portfolio-website (Next.js) repo to import. Data only:
no frontend code lives in this repo.

Re-scoring. As of prompt 83, every prediction came from an already-promoted
.joblib artifact, loaded and scored ONLY (never .fit). As of prompt 86 (see
below), 2 of the 6 legs still work exactly that way; the other 4 are refit at
their already-locked hyperparameters (no new tuning) because no post-fix
artifact was ever persisted to disk:

  Active-binary       v1e_gradient_boosting_calibrated  saved joblib, predict only (acb.score_classifier)
  Active-continuous   c1d_random_forest x v1e P(shot)   Prompt 86: c1d refit at locked params (c1d_mod.fit_predict_c1d)
  Active-xT           x1c_random_forest (two-stage)     Prompt 86: refit at locked params (x1c_mod.fit_predict)
  Passive-binary      p1e_gradient_boosting_calibrated  Prompt 86: refit (calibrated) at locked params (p1e_mod.fit_predict_gbm_calibrated)
  Passive-continuous  d1_lognormal_glm x p1e P(shot)    saved joblib, predict only, unchanged since prompt 83
  Passive-xT          y1c_random_forest (two-stage)     Prompt 86: refit at locked params (y1c_mod.fit_predict)

Design-matrix builders (imputation medians, one-hot category sets) are fitted
on the canonical train+val rows exactly as in those scripts, which is why every
leg is scored on its FULL held-out test set first: before anything is written,
each leg's recorded held-out headline metric is reproduced from these scores and
compared to the on-disk readout. Only then are the curated matches (all
three are canonical held-out TEST matches, so every prediction is out-of-sample)
sliced out.

--- Prompt 86 update (post coordinate-frame fix) ---

Prompt 85 fixed a coordinate-frame bug in the loader and rebuilt the locked
feature parquets in place; see reports/modeling/COORDINATE_FRAME_FIX_AND_REPIPELINE.md.
Prompt 85's own Phase 4/5 validation scripts (scripts/analysis/phase4_held_out_refit.py,
scripts/analysis/phase5_xt_promotion_audit.py) established the corrected headline
numbers, but refit every model fresh, in memory, at each leg's already-locked
hyperparameters -- they never saved new .joblib artifacts to the standing
outputs/models/{classification,regression}/ paths (those paths still hold the
PRE-FIX-trained weights). For the 2 legs whose promotion was RECONFIRMED unchanged
(v1e_gradient_boosting_calibrated, d1_lognormal_glm), this script keeps loading
those pre-fix standing artifacts and scoring them on the corrected features, as
before -- no retraining for these two, per Phase 4's own "don't touch" verdict.
For the 4 legs whose promotion was REOPENED/refreshed on corrected data
(p1e_gradient_boosting_calibrated, c1d_random_forest, x1c_random_forest,
y1c_random_forest -- same rung/architecture/hyperparameters in every case, only
the numbers moved), this script now refits each at its already-locked
hyperparameters on the corrected train+val split, mirroring Phase 4/5's own
already-validated methodology exactly (no new tuning) -- this is the only way to
produce dashboard predictions consistent with the corrected headline numbers now
published in the ladder docs, since no persisted post-fix artifact exists to load
instead. The hard bit-exact `assert_close` reproduction gate from prompt 83 is
relaxed to a logged comparison against Phase 4/5's own corrected numbers
(outputs/models/validation/phase4_held_out_refit.json,
outputs/models/validation/phase5_xt_promotion_audit.json) rather than the
pre-fix CSV/JSON readouts, since those pre-fix numbers are no longer the right
target to reproduce.

Display coordinates. A read-only check in this prompt found that the locked
feature parquets' coordinates are NOT in one consistent frame: raw StatsBomb
event/360 locations are always in the acting team's own frame, but
statsbomb_loader._infer_attack_sign_by_period_team flips one possession team
per period, so roughly half the rows end up in the opposite frame from the other
half (and the rule-based phase_label, which reads ball_x, inherits the
inconsistency). The locked files are NOT modified and the models are scored on
the features exactly as they were trained. For DISPLAY only, this script derives
a consistent frame from the raw StatsBomb location and flags each row whose
model-input frame disagrees with the phase labeller's assumed (attacking-team)
frame. See dashboard_data/README.md, "Known data-quality issue".

Usage:
    .venv/Scripts/python.exe scripts/dashboard/export_match_explorer.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_active_binary_baseline as ab  # noqa: E402
import train_active_continuous_baseline as acb  # noqa: E402
import train_active_xt_baseline as xb  # noqa: E402
import train_c1d_random_forest as c1d_mod  # noqa: E402
import train_p1e_gradient_boosting as p1e_mod  # noqa: E402
import train_passive_binary_baseline as pb  # noqa: E402
import train_passive_continuous_baseline as pcb  # noqa: E402
import train_passive_xt_baseline as yb  # noqa: E402
import train_x1c_random_forest as x1c_mod  # noqa: E402
import train_y1c_random_forest as y1c_mod  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402
from dax.models.two_part_xg import compute_hurdle_metrics  # noqa: E402

OUT_DIR = REPO_ROOT / "dashboard_data" / "match_explorer"
VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
COMPARISONS_DIR = REPO_ROOT / "outputs" / "models" / "comparisons"
RAW_DIR = REPO_ROOT / "data" / "raw"

# Prompt 86: already-locked hyperparameters for the 4 reopened legs, refit here
# on corrected features (Phase 4/5's own already-validated configuration, no
# new tuning) -- exact values from outputs/models/validation/phase4_held_out_refit.json
# and phase5_xt_promotion_audit.json / each leg's own *.json artifact metadata.
C1D_PARAMS = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10}
P1E_PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 500}
P1E_N_ESTIMATORS_FIXED = 116
P1E_CALIBRATION_METHOD = "isotonic"
X1C_CLF_PARAMS = {"max_depth": None, "min_samples_leaf": 20}
X1C_REG_PARAMS = {"max_depth": None, "min_samples_leaf": 20}
Y1C_CLF_PARAMS = {"max_depth": None, "min_samples_leaf": 150}
Y1C_REG_PARAMS = {"max_depth": None, "min_samples_leaf": 500}

# Curated in prompt 83 on frame-corrected defensive-style contrast only (no model
# output or target value was a selection criterion) -- rationale in
# dashboard_data/README.md section 2.
CURATED_MATCHES = {
    3938643: "High press vs deep block (Euro 2024): France defend high, Poland sit deep.",
    3857294: "High line vs deep block (WC 2022): Netherlands defend high, Qatar sit deep.",
    3857298: "Open, transition-heavy game (WC 2022): highest counterpress rate of the 23 held-out matches.",
}

SIG = 4  # significant figures kept for model outputs in the JSON (display precision)


def rnd(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return 0.0
    return float(f"{value:.{SIG}g}")


def rnd_xy(x, y) -> dict:
    return {"x": round(float(x), 2), "y": round(float(y), 2)}


def assert_close(label: str, got: float, expected: float, tol: float = 1e-9) -> None:
    ok = abs(got - expected) <= tol * max(1.0, abs(expected))
    print(f"  [verify] {label}: reproduced={got:.10f} recorded={expected:.10f} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise AssertionError(f"{label} did not reproduce ({got} vs {expected}) -- not the same scoring path, stopping.")


def compare_logged(label: str, got: float, reference: float, tol: float = 0.02) -> None:
    """Prompt 86: logged (non-fatal) comparison, used for the 4 legs refit on
    corrected data. The pre-fix hard `assert_close` reproduction gate no longer
    applies -- there is no persisted post-fix artifact to bit-exactly reproduce,
    only Phase 4/5's own in-memory refit numbers (JSON), computed by a fresh fit
    with its own random draws inside sklearn/LightGBM's early stopping / internal
    CV, so small non-determinism is expected here even at identical hyperparameters."""
    diff = abs(got - reference)
    ok = diff <= tol * max(1.0, abs(reference))
    print(f"  [compare] {label}: this refit={got:.6f} Phase-4/5 recorded={reference:.6f} "
          f"diff={diff:.6f} -> {'within tolerance' if ok else 'OUTSIDE TOLERANCE, check'}")


def load_match_meta() -> dict[int, dict]:
    meta: dict[int, dict] = {}
    for path in glob.glob(str(RAW_DIR / "matches" / "*.json")):
        for m in json.loads(Path(path).read_text(encoding="utf-8")):
            meta[int(m["match_id"])] = m
    return meta


def load_raw_events(match_id: int) -> dict[str, dict]:
    events = json.loads((RAW_DIR / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    return {e["id"]: e for e in events}


def split(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = canonical_test_mask(df, group_col=group_col)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Active legs
# ---------------------------------------------------------------------------

def score_active() -> pd.DataFrame:
    print("\n[active] loading player_defensive_actions.parquet (read-only)...")
    df_all = ab.load_data()
    trainval_full, test_full = split(df_all, ab.GROUP_COL)

    print("[active-binary] v1e_gradient_boosting_calibrated -- pre-fix standing artifact, NOT retrained "
          "(Phase 4 verdict: reconfirmed/closed, do not touch), scored on corrected features")
    p_shot = acb.score_classifier(trainval_full, test_full)
    y = test_full[ab.TARGET_COL].to_numpy()
    rec = pd.read_csv(COMPARISONS_DIR / "active_binary_baseline_held_out_test_readout.csv")
    rec = rec[rec["variant"] == "v1e_gradient_boosting_calibrated"].iloc[0]
    # Prompt 86: this is the pre-fix artifact scored on corrected test-row features --
    # a small drift from the pre-fix CSV record is expected (36.6% of v1e's own gain
    # importance sits on features whose values changed) even though the leg's own
    # fold-level significance test found this drift indistinguishable from noise
    # (Phase 4: fold-AP shift +0.0008, p=0.929). Logged, not a hard gate.
    compare_logged("v1e held-out PR-AUC", average_precision_score(y, p_shot), rec["average_precision"])
    compare_logged("v1e held-out ROC-AUC", roc_auc_score(y, p_shot), rec["roc_auc"])

    print("[active-continuous] c1d_random_forest -- Prompt 86: refit at locked hyperparameters "
          "on corrected features (Phase 4 verdict: reopened, promotion stands, numbers updated)")
    pos_all = df_all[df_all[ab.TARGET_COL] == 1].reset_index(drop=True)
    pos_trainval = pos_all.loc[~canonical_test_mask(pos_all, group_col=ab.GROUP_COL)].reset_index(drop=True)
    # Fit once on positive train+val rows (as c1d always was), predict on the FULL
    # held-out test set (matches validate_c1d_promotion.score_c1d_predict_only's own
    # target_df=test_full convention -- the hurdle multiplies by p_shot for every row).
    c1d_log_pred, c1d_sigma2, _, _, _ = c1d_mod.fit_predict_c1d(pos_trainval, test_full, C1D_PARAMS)
    e_xg = acb.backtransform(c1d_log_pred, c1d_sigma2, "lognormal_corrected")
    hurdle = p_shot * e_xg
    metrics = compute_hurdle_metrics(pd.DataFrame({
        "observed_future_xg": test_full[acb.TARGET_XG_COL].to_numpy(),
        "combined_future_xg_prediction": hurdle,
        "match_id": test_full[ab.GROUP_COL].to_numpy(),
    }))
    phase4_c1d = json.loads((VALIDATION_DIR / "phase4_held_out_refit.json").read_text(encoding="utf-8"))["c1d_random_forest"]
    print(f"  [info] c1d hurdle RMSE={metrics['rmse']:.6f} R2={metrics['r2']:.6f} (hurdle pipeline itself was not "
          f"separately re-scored by Phase 4/5 -- only c1d's own regression-only common_log_rmse/corrected_r2 were, "
          f"a different metric on a different scale, not directly comparable to this hurdle RMSE/R2; "
          f"Phase-4 corrected regression-only numbers for reference: common_log_rmse={phase4_c1d['common_log_rmse']:.4f}, "
          f"corrected_r2={phase4_c1d['corrected_r2']:.4f})")

    out = test_full[["match_id", "event_id"]].copy()
    out["v1e_p_shot"] = p_shot
    out["c1d_e_xg_given_shot"] = e_xg
    out["c1d_hurdle_expected_xg"] = hurdle

    print("[active-xT] x1c_random_forest -- Prompt 86: refit at locked hyperparameters on corrected "
          "features AND corrected target_xt_delta_v2 (Phase 5 verdict: promoted, same rung, numbers updated)")
    dfx = xb.load_data()
    trainval_x, test_x = split(dfx, xb.GROUP_COL)
    combined, extra_x = x1c_mod.fit_predict(trainval_x, test_x, X1C_CLF_PARAMS, X1C_REG_PARAMS)
    p_nonzero = extra_x["clf"].predict_proba(extra_x["clf_builder"].transform(test_x)[0])[:, 1]
    e_delta = extra_x["reg"].predict(extra_x["reg_builder"].transform(test_x)[0])
    m = xb.signed_regression_metrics(test_x[xb.TARGET_COL].to_numpy(), combined)
    phase5_x = json.loads((VALIDATION_DIR / "phase5_xt_promotion_audit.json").read_text(encoding="utf-8"))["active_xt"]["held_out_test"]["x1c_random_forest"]
    compare_logged("x1c pipeline RMSE", m["rmse"], phase5_x["rmse"])
    compare_logged("x1c pipeline R2", m["r2"], phase5_x["r2"])

    xt = test_x[["event_id", xb.TARGET_COL]].copy()
    xt["x1c_p_nonzero"] = p_nonzero
    xt["x1c_e_delta_given_nonzero"] = e_delta
    xt["x1c_expected_delta"] = combined
    out = out.merge(xt, on="event_id", how="left", validate="one_to_one")
    feature_cols = ["event_id", "period", "player", "team", "attacking_team", "defending_team", "event_type",
                    "phase_label", "action_x", "action_y", "counterpress", "action_won_possession",
                    ab.TARGET_COL, acb.TARGET_XG_COL, "has_360"]
    out = out.merge(test_full[feature_cols], on="event_id", how="left", validate="one_to_one")
    return out


# ---------------------------------------------------------------------------
# Passive legs
# ---------------------------------------------------------------------------

PASSIVE_KEY = ["event_id", "defender_slot_index"]


def score_passive() -> pd.DataFrame:
    print("\n[passive] loading passive_defense.parquet (read-only)...")
    df_all = pb.load_data()
    assert not df_all.duplicated(PASSIVE_KEY).any(), "passive rows must be unique on (event_id, defender_slot_index)"
    trainval_full, test_full = split(df_all, pb.GROUP_COL)

    print("[passive-binary] p1e_gradient_boosting_calibrated -- Prompt 86: refit (calibrated) at locked "
          "hyperparameters on corrected features (Phase 4 verdict: reopened, promotion stands, numbers updated)")
    y_trainval = trainval_full[pb.TARGET_COL].to_numpy()
    y_test = test_full[pb.TARGET_COL].to_numpy()
    p_shot, _, _ = p1e_mod.fit_predict_gbm_calibrated(
        trainval_full, test_full, y_trainval, y_test, P1E_PARAMS, P1E_N_ESTIMATORS_FIXED, P1E_CALIBRATION_METHOD,
    )
    y = y_test
    phase4_p1e = json.loads((VALIDATION_DIR / "phase4_held_out_refit.json").read_text(encoding="utf-8"))["p1e_gradient_boosting"]
    compare_logged("p1e held-out PR-AUC", average_precision_score(y, p_shot), phase4_p1e["average_precision"])
    compare_logged("p1e held-out ROC-AUC", roc_auc_score(y, p_shot), phase4_p1e["roc_auc"])

    print("[passive-continuous] d1_lognormal_glm -- pre-fix standing artifact, NOT retrained "
          "(Phase 4 verdict: reconfirmed/closed, do not touch), scored on corrected features")
    pos_all = df_all[df_all[pb.TARGET_COL] == 1].reset_index(drop=True)
    pos_trainval = pos_all.loc[~canonical_test_mask(pos_all, group_col=pb.GROUP_COL)].reset_index(drop=True)
    d1_model = joblib.load(pcb.REG_DIR / "d1_lognormal_glm.joblib")
    d1_builder = pb.DesignMatrixBuilder(add_interactions=False, add_quadratic=False).fit(pos_trainval)
    x_pos_trainval, _ = d1_builder.transform(pos_trainval)
    assert x_pos_trainval.shape[1] == d1_model.n_features_in_, (
        f"d1 builder width {x_pos_trainval.shape[1]} != saved model's {d1_model.n_features_in_} -- "
        "design matrix drifted from what the artifact was fitted on, stopping."
    )
    resid = np.log(pos_trainval[pcb.TARGET_XG_COL].to_numpy()) - d1_model.predict(x_pos_trainval)
    d1_sigma2 = float(np.var(resid, ddof=1))
    e_xg = pcb.backtransform(d1_model.predict(d1_builder.transform(test_full)[0]), d1_sigma2, "lognormal_corrected")
    hurdle = p_shot * e_xg
    metrics = compute_hurdle_metrics(pd.DataFrame({
        "observed_future_xg": test_full[pcb.TARGET_XG_COL].to_numpy(),
        "combined_future_xg_prediction": hurdle,
        "match_id": test_full[pb.GROUP_COL].to_numpy(),
    }))
    rec_d1 = json.loads((VALIDATION_DIR / "hurdle_pipeline_readout_passive.json").read_text(encoding="utf-8"))["hurdle_pipeline_metrics"]
    # Prompt 86: p_shot now comes from a refit p1e (corrected features), so the hurdle
    # product (p1e x d1) will not bit-exactly reproduce the pre-fix hurdle readout even
    # though d1 itself is untouched -- logged, not a hard gate, same reasoning as v1e above.
    compare_logged("d1 hurdle RMSE", metrics["rmse"], rec_d1["rmse"])
    compare_logged("d1 hurdle R2", metrics["r2"], rec_d1["r2"])

    keep = ["match_id", "event_id", "defender_slot_index", "period", "on_ball_event_type", "on_ball_team",
            "defending_team", "phase_label", "defender_x", "defender_y", "ball_x", "ball_y",
            "defender_functional_role", "visibility_limited", pb.TARGET_COL, pcb.TARGET_XG_COL]
    out = test_full[keep].copy()
    out["p1e_p_shot"] = p_shot
    out["d1_e_xg_given_shot"] = e_xg
    out["d1_hurdle_expected_xg"] = hurdle
    del df_all, trainval_full, test_full, pos_all, pos_trainval

    print("[passive-xT] y1c_random_forest -- Prompt 86: refit at locked hyperparameters on corrected "
          "features AND corrected target_xt_delta_passive (Phase 5 verdict: promoted, same rung, numbers updated)")
    dfy = yb.load_data()
    trainval_y, test_y = split(dfy, yb.GROUP_COL)
    del dfy
    combined, extra_y = y1c_mod.fit_predict(trainval_y, test_y, Y1C_CLF_PARAMS, Y1C_REG_PARAMS)
    p_nonzero = extra_y["clf"].predict_proba(extra_y["clf_builder"].transform(test_y)[0])[:, 1]
    e_delta = extra_y["reg"].predict(extra_y["reg_builder"].transform(test_y)[0])
    m = yb.signed_regression_metrics(test_y[yb.TARGET_COL].to_numpy(), combined)
    phase5_y = json.loads((VALIDATION_DIR / "phase5_xt_promotion_audit.json").read_text(encoding="utf-8"))["passive_xt"]["held_out_test"]["y1c_random_forest"]
    compare_logged("y1c pipeline RMSE", m["rmse"], phase5_y["rmse"])
    compare_logged("y1c pipeline R2", m["r2"], phase5_y["r2"])

    yt = test_y[PASSIVE_KEY + [yb.TARGET_COL]].copy()
    yt["y1c_p_nonzero"] = p_nonzero
    yt["y1c_e_delta_given_nonzero"] = e_delta
    yt["y1c_expected_delta"] = combined
    return out.merge(yt, on=PASSIVE_KEY, how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# Frame handling (display only -- model inputs are never altered)
# ---------------------------------------------------------------------------

def frame_state(stored_x, stored_y, raw_loc) -> str:
    """'raw' if the parquet coordinate equals the raw StatsBomb location (acting
    team's own frame), 'flipped' if it equals the 180-degree rotation, else 'unknown'."""
    if not isinstance(raw_loc, list) or len(raw_loc) < 2:
        return "unknown"
    rx, ry = float(raw_loc[0]), float(raw_loc[1])
    if abs(stored_x - rx) < 1e-6 and abs(stored_y - ry) < 1e-6:
        return "raw"
    if abs(stored_x - (120.0 - rx)) < 1e-6 and abs(stored_y - (80.0 - ry)) < 1e-6:
        return "flipped"
    return "unknown"


def other(team: str, home: str, away: str) -> str:
    return away if team == home else home


def to_frame(x: float, y: float, from_team: str, to_team: str) -> tuple[float, float]:
    return (x, y) if from_team == to_team else (120.0 - x, 80.0 - y)


# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------

def build_match(match_id: int, meta: dict, active: pd.DataFrame, passive: pd.DataFrame) -> dict:
    raw = load_raw_events(match_id)
    home, away = meta["home_team"], meta["away_team"]
    comp = "WC2022" if meta["competition_name"] == "FIFA World Cup" else "Euro2024"
    events: list[dict] = []
    qa = {"active_rows": 0, "active_rows_feature_frame_inconsistent": 0, "active_rows_without_active_xt": 0,
          "passive_events": 0, "passive_defender_rows": 0, "passive_events_feature_frame_inconsistent": 0,
          "passive_defender_rows_without_passive_xt": 0, "rows_with_unknown_frame": 0}

    for r in active.itertuples(index=False):
        e = raw[r.event_id]
        actor = e["team"]
        state = frame_state(r.action_x, r.action_y, e.get("location"))
        if state == "unknown":
            qa["rows_with_unknown_frame"] += 1
        stored_frame_team = actor if state == "raw" else other(actor, home, away)
        consistent = None if state == "unknown" else stored_frame_team == r.attacking_team
        own_x, own_y = e["location"][0], e["location"][1]
        mx, my = to_frame(own_x, own_y, actor, home)
        preds = {
            "active_binary": {"probability": rnd(r.v1e_p_shot)},
            "active_continuous": {"expected_value": rnd(r.c1d_hurdle_expected_xg),
                                  "expected_xg_given_shot": rnd(r.c1d_e_xg_given_shot)},
        }
        observed = {"shot_within_10s": int(getattr(r, ab.TARGET_COL)),
                    "xg_within_10s": rnd(getattr(r, acb.TARGET_XG_COL))}
        if pd.notna(r.x1c_expected_delta):
            preds["active_xt"] = {"expected_delta": rnd(r.x1c_expected_delta), "p_nonzero": rnd(r.x1c_p_nonzero),
                                  "expected_delta_given_nonzero": rnd(r.x1c_e_delta_given_nonzero)}
            observed["xt_delta"] = rnd(getattr(r, xb.TARGET_COL))
        else:
            qa["active_rows_without_active_xt"] += 1
        qa["active_rows"] += 1
        qa["active_rows_feature_frame_inconsistent"] += int(consistent is False)
        events.append({
            "event_id": r.event_id,
            "timestamp": e["timestamp"], "minute": int(e["minute"]), "second": int(e["second"]),
            "period": int(r.period),
            "team": actor, "player": r.player,
            "phase": "active",
            "location": rnd_xy(own_x, own_y),
            "location_match_frame": rnd_xy(mx, my),
            "on_ball_event_type": r.event_type,
            "counterpress": bool(r.counterpress) if pd.notna(r.counterpress) else None,
            "won_possession": bool(r.action_won_possession) if pd.notna(r.action_won_possession) else None,
            "phase_label": r.phase_label,
            "phase_label_frame_consistent": consistent,
            "predictions": preds,
            "observed": observed,
        })

    for event_id, grp in passive.groupby("event_id", sort=False):
        e = raw[event_id]
        first = grp.iloc[0]
        actor = e["team"]
        defending = first["defending_team"]
        state = frame_state(first["ball_x"], first["ball_y"], e.get("location"))
        if state == "unknown":
            qa["rows_with_unknown_frame"] += 1
            continue  # cannot place this event's defenders in a known frame -- dropped from display, counted
        stored_frame_team = actor if state == "raw" else other(actor, home, away)
        consistent = stored_frame_team == first["on_ball_team"]
        bx, by = to_frame(first["ball_x"], first["ball_y"], stored_frame_team, defending)
        bmx, bmy = to_frame(first["ball_x"], first["ball_y"], stored_frame_team, home)
        defenders = []
        for d in grp.sort_values("defender_slot_index").itertuples(index=False):
            dx, dy = to_frame(d.defender_x, d.defender_y, stored_frame_team, defending)
            dmx, dmy = to_frame(d.defender_x, d.defender_y, stored_frame_team, home)
            preds = {
                "passive_binary": {"probability": rnd(d.p1e_p_shot)},
                "passive_continuous": {"expected_value": rnd(d.d1_hurdle_expected_xg),
                                       "expected_xg_given_shot": rnd(d.d1_e_xg_given_shot)},
            }
            if pd.notna(d.y1c_expected_delta):
                preds["passive_xt"] = {"expected_delta": rnd(d.y1c_expected_delta), "p_nonzero": rnd(d.y1c_p_nonzero),
                                       "expected_delta_given_nonzero": rnd(d.y1c_e_delta_given_nonzero)}
            else:
                qa["passive_defender_rows_without_passive_xt"] += 1
            defenders.append({
                "defender_slot_index": int(d.defender_slot_index),
                "location": rnd_xy(dx, dy),
                "location_match_frame": rnd_xy(dmx, dmy),
                "functional_role": d.defender_functional_role,
                "predictions": preds,
            })
        observed = {"shot_within_10s": int(first[pb.TARGET_COL]), "xg_within_10s": rnd(first[pcb.TARGET_XG_COL])}
        if pd.notna(first[yb.TARGET_COL]):
            observed["xt_delta"] = rnd(first[yb.TARGET_COL])
        qa["passive_events"] += 1
        qa["passive_defender_rows"] += len(defenders)
        qa["passive_events_feature_frame_inconsistent"] += int(not consistent)
        events.append({
            "event_id": event_id,
            "timestamp": e["timestamp"], "minute": int(e["minute"]), "second": int(e["second"]),
            "period": int(first["period"]),
            "team": defending, "player": None,
            "phase": "passive",
            "location": rnd_xy(bx, by),
            "location_match_frame": rnd_xy(bmx, bmy),
            "on_ball_event_type": first["on_ball_event_type"],
            "on_ball_team": first["on_ball_team"],
            "visibility_limited": bool(first["visibility_limited"]) if pd.notna(first["visibility_limited"]) else None,
            "phase_label": first["phase_label"],
            "phase_label_frame_consistent": consistent,
            "defenders": defenders,
            "observed": observed,
        })

    events.sort(key=lambda ev: (ev["period"], ev["timestamp"], 0 if ev["phase"] == "passive" else 1))
    return {
        "schema_version": 1,
        "match_id": str(match_id),
        "competition": comp,
        "competition_stage": meta.get("competition_stage"),
        "teams": {"home": home, "away": away},
        "score": {"home": int(meta["home_score"]), "away": int(meta["away_score"])},
        "date": meta["match_date"],
        "canonical_split": "test",
        "why_selected": CURATED_MATCHES[match_id],
        "reference_models": {
            "active_binary": "v1e_gradient_boosting_calibrated",
            "active_continuous": "c1d_random_forest x v1e_gradient_boosting_calibrated P(shot) (hurdle)",
            "active_xt": "x1c_random_forest (two-stage)",
            "passive_binary": "p1e_gradient_boosting_calibrated",
            "passive_continuous": "d1_lognormal_glm x p1e_gradient_boosting_calibrated P(shot) (hurdle)",
            "passive_xt": "y1c_random_forest (two-stage)",
        },
        "coordinate_frames": {
            "location": "StatsBomb 120x80 pitch, in the acting/defending team's OWN frame: own goal at x=0, "
                        "opponent goal at x=120 (so larger x = defending further up the pitch).",
            "location_match_frame": f"Same pitch, one fixed frame for the whole match: {home} (home) attack toward "
                                    f"x=120, {away} toward x=0, both halves (StatsBomb does not record ends switching).",
        },
        "data_quality": qa,
        "events": events,
    }


def main() -> None:
    load_canonical_split()
    assignment = json.loads((REPO_ROOT / "outputs" / "models" / "splits" / "match_assignment.json").read_text(encoding="utf-8"))
    for mid in CURATED_MATCHES:
        assert assignment[str(mid)] == "test", f"{mid} is not a canonical held-out test match"
    meta = load_match_meta()

    active = score_active()
    passive = score_passive()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[write] match explorer files...")
    for mid in CURATED_MATCHES:
        payload = build_match(mid, meta[mid], active[active["match_id"] == mid], passive[passive["match_id"] == mid])
        path = OUT_DIR / f"{mid}.json"
        path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        print(f"  {path.relative_to(REPO_ROOT)}: {path.stat().st_size / 1e6:.2f} MB, qa={payload['data_quality']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
