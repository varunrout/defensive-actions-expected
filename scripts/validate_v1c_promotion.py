"""Full promotion validation for v1c_systematic_interactions (prompt 43).

Rung 2's own ladder gate (prompt 41 -- paired significance, held-out-test
confirmation, player-disjoint recheck) answers "is this rung's isolated
change real?". This script answers the bigger question: should
v1c_systematic_interactions replace v1_unweighted as the standing
baseline everyone downstream builds on? That needs the same depth of
scrutiny v1 itself went through in prompt 38, not just the 3-gate ladder
check.

Two checks, mirroring prompt 38 items (b) and (d) for v1:
  1. Tournament-stratified PR-AUC (WC2022 vs Euro2024).
  2. Error-slice breakdown by phase_label and position, directly
     comparable against v1's own numbers.

IMPORTANT population note -- a deliberate deviation, documented rather
than silently resolved: prompt 38's actual tournament_stratified_v1.json
and error_analysis_v1.json were built by fitting v1 once on all
train+val rows and scoring the FROZEN HELD-OUT TEST SET (verified: both
files' row counts sum to 10,660 / 23 matches, i.e. the test set, not
train+val's 45,408 / 92). This prompt's brief describes that precedent
as running on "train+val CV predictions" and simultaneously forbids
scoring v1c's held-out test set a second time (it was already read once,
labelled FINAL_HELD_OUT_TEST_NOT_FOR_SELECTION, in prompt 41). Those two
statements are in tension for v1c specifically. Per the explicit,
binding constraint ("do not score the held-out test set again for any
new purpose"), this script uses genuine 5-fold cross-validated
out-of-fold (OOF) predictions on train+val for BOTH v1 and v1c -- not
v1's original held-out-test-based numbers, and not an in-sample scoring
of the single all-train+val fit. This keeps the comparison population
identical and fair for both models, at the cost of not being a byte-for
-byte replay of prompt 38's own (held-out-test-based) v1 numbers, which
remain available unchanged in tournament_stratified_v1.json /
error_analysis_v1.json for reference.

Neither v1's nor v1c's specification/hyperparameters are changed here --
v1's OOF predictions reuse train_active_binary_baseline.run_cv exactly
as already defined; v1c's OOF predictions reuse
train_v1c_systematic_interactions.run_cv_v1c at the already-locked C=0.1
(no re-tuning). No existing output file for v0/v1/v2/v3/v1b_quadratic or
v1c is modified; this script only writes two new validation JSONs.

Usage:
    python scripts/validate_v1c_promotion.py
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
from dax.models.evaluation import classification_metrics  # noqa: E402
from dax.models.splits import canonical_test_mask, load_canonical_split  # noqa: E402

VALIDATION_DIR = REPO_ROOT / "outputs" / "models" / "validation"
CHOSEN_C = 0.1  # locked in prompt 41; not re-tuned here
MATCHES_GLOB = str(REPO_ROOT / "data" / "raw" / "matches" / "*.json")


def build_competition_map() -> dict[int, str]:
    comp: dict[int, str] = {}
    for path in glob.glob(MATCHES_GLOB):
        matches = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in matches:
            comp[m["match_id"]] = f"{m['competition_name']} {m['season']}"
    return comp


def tournament_breakdown(df: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray, comp_map: dict) -> dict:
    work = df.copy()
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


def error_slice_breakdown(df: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray) -> dict:
    work = df.copy()
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
        f"Expected exactly WC2022 + Euro2024, got {tournaments_present} -- do not assume, this must be re-checked."
    )

    # -- v1's train+val CV OOF predictions (spec unchanged; recomputed only
    #    because per-row OOF scores were never persisted to disk). --
    print("[cv] v1_unweighted OOF (train+val, 5 canonical folds)...")
    v1_cv = tb.run_cv(trainval_df, "v1_unweighted")

    # -- v1c's train+val CV OOF predictions at the already-locked C=0.1
    #    (no re-tuning -- reuses run_cv_v1c from prompt 41's script as-is). --
    print(f"[cv] v1c_systematic_interactions OOF (train+val, 5 canonical folds, C={CHOSEN_C})...")
    v1c_cv = v1c_mod.run_cv_v1c(trainval_df, CHOSEN_C)

    # Both run_cv/run_cv_v1c apply canonical_grouped_folds identically (same
    # function, same seed), so both "df"s are row-for-row the same ordering.
    df_ordered = v1c_cv.get("df")
    if df_ordered is None:
        # train_active_binary_baseline.run_cv returns "df"; run_cv_v1c does not
        # -- reconstruct the identical ordering it used internally (same
        # canonical_grouped_folds call, same .loc[] reorder, no reset_index,
        # matching both run_cv's and run_cv_v1c's own internal construction).
        from dax.models.splits import canonical_grouped_folds
        folds = canonical_grouped_folds(trainval_df, group_col=tb.GROUP_COL)
        df_ordered = trainval_df.loc[folds["row_index"]]

    y_v1 = v1_cv["y_all"]
    y_v1c = v1c_cv["y_all"]
    assert np.array_equal(y_v1, y_v1c), "v1 and v1c CV row ordering diverged -- folds must match exactly"

    print("[tournament] breakdown for v1 and v1c on train+val CV OOF...")
    tourn_v1 = tournament_breakdown(df_ordered, y_v1, v1_cv["oof_scores"], comp_map)
    tourn_v1c = tournament_breakdown(df_ordered, y_v1c, v1c_cv["oof_scores"], comp_map)

    train_composition = {str(k): int(v) for k, v in trainval_df.assign(competition=trainval_comp)
                          .groupby("competition")[tb.GROUP_COL].nunique().items()}

    tournament_output = {
        "population": "train+val, 5-fold canonical CV out-of-fold predictions (NOT held-out test; "
                       "see module docstring for why this differs from tournament_stratified_v1.json's "
                       "held-out-test population)",
        "chosen_C_v1c": CHOSEN_C,
        "train_val_match_counts_by_tournament": train_composition,
        "v1_unweighted_by_tournament": tourn_v1,
        "v1c_systematic_interactions_by_tournament": tourn_v1c,
        "reference_v1_held_out_test_numbers_from_prompt_38": {
            "FIFA World Cup 2022": {"average_precision": 0.35962030466893663, "roc_auc": 0.8333764933046118},
            "UEFA Euro 2024": {"average_precision": 0.39130913717769134, "roc_auc": 0.7971029378117802},
            "note": "for context only -- different population (held-out test) than the CV-OOF numbers above",
        },
    }
    (VALIDATION_DIR / "tournament_stratified_v1c.json").write_text(
        json.dumps(tournament_output, indent=2, default=str), encoding="utf-8"
    )

    print("[error-slice] phase_label / position breakdown for v1 and v1c on train+val CV OOF...")
    errs_v1 = error_slice_breakdown(df_ordered, y_v1, v1_cv["oof_scores"])
    errs_v1c = error_slice_breakdown(df_ordered, y_v1c, v1c_cv["oof_scores"])
    error_output = {
        "population": "train+val, 5-fold canonical CV out-of-fold predictions (NOT held-out test; "
                       "see module docstring for why this differs from error_analysis_v1.json's "
                       "held-out-test population -- same slice keys, so still directly diffable in shape)",
        "chosen_C_v1c": CHOSEN_C,
        "v1_unweighted": errs_v1,
        "v1c_systematic_interactions": errs_v1c,
    }
    (VALIDATION_DIR / "error_analysis_v1c.json").write_text(
        json.dumps(error_output, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== TOURNAMENT (train+val CV OOF) ===")
    for comp in tourn_v1:
        v1m = tourn_v1[comp]
        v1cm = tourn_v1c.get(comp, {})
        print(f"{comp}: v1 PR-AUC={v1m.get('average_precision'):.4f} "
              f"v1c PR-AUC={v1cm.get('average_precision', float('nan')):.4f}")

    print("\n=== ERROR SLICES: phase_label (v1 vs v1c, train+val CV OOF) ===")
    v1_by_key = {r["phase_label"]: r for r in errs_v1["phase_label"]}
    v1c_by_key = {r["phase_label"]: r for r in errs_v1c["phase_label"]}
    for key in v1_by_key:
        a, b = v1_by_key[key], v1c_by_key.get(key, {})
        print(f"{key}: rows={a['rows']} v1 AP={a.get('average_precision')} v1c AP={b.get('average_precision')} "
              f"| v1 cal_gap={a['calibration_gap']:+.4f} v1c cal_gap={b.get('calibration_gap', float('nan')):+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
