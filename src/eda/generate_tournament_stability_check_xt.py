"""CLI entrypoint: tournament stability check against `target_xt_delta`,
ACTIVE features only -- the xT sibling of
generate_tournament_stability_check_xg.py (the confirmed generator of the
xg portal's TOURNAMENT_STABILITY_CHECK.json/.html).

Carried over UNCHANGED (all target-independent or scale-free):
  - the tournament split itself (WC2022 vs Euro2024, via
    src.eda.tournament_mapping.load_match_tournament_map)
  - the INVESTIGATE feature list, filtered to its 6 ACTIVE entries -- the
    4 passive entries are out of scope for this prompt. The list is
    imported, not re-derived from this portal's own flagged-inconsistent
    features, exactly as the xg version does: the point is to mirror the
    same population-mixing hypothesis test on the same features, so the
    comparison across targets stays meaningful.
  - bin edges computed once on the pooled data and REUSED for both
    tournaments (so a shape difference is about the target, not about
    different cut points)
  - classify_shape()'s heuristic
  - nearest_defender_distance's exclusion (separate known self-reference bug)

ADAPTED: flat_margin, per Adaptation 1 in src/eda/xt_common.py.

Usage:
    python -m src.eda.generate_tournament_stability_check_xt
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_common as xc
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, N_QUANTILE_BINS, classify_shape
from src.eda.generate_numerical_xt_target_analysis import _bin_table
from src.eda.generate_tournament_stability_check import INVESTIGATE
from src.eda.tournament_mapping import load_match_tournament_map

OUTPUT_PATH = xc.OUT_DIR / "TOURNAMENT_STABILITY_CHECK.json"
TARGET = xc.TARGET

ACTIVE_INVESTIGATE = [f for f, ds in INVESTIGATE.items() if ds == "active"]


def _load_train_test_finding(feature: str) -> dict:
    """Restate this portal's own atlas finding for context -- read directly
    from the JSON just written by generate_numerical_xt_target_analysis,
    never re-typed by hand. Same approach as the xg version."""
    atlas_path = xc.OUT_DIR / "active_numerical_target_atlas.json"
    data = json.loads(atlas_path.read_text(encoding="utf-8"))
    entry = next((f for f in data["features"] if f["feature"] == feature), None)
    if entry is None:
        return {"in_atlas": False,
                "note": f"{feature} is not in the reconstructed numerical pool this portal's atlas covers."}
    cc = entry["consistency_check"]
    return {
        "in_atlas": True,
        "overall_shape": entry["shape"],
        "spearman_rho": entry["spearman_rho"],
        "train_val_shape": cc.get("train_val_shape"),
        "test_shape": cc.get("test_shape"),
        "train_test_consistent": cc.get("consistent"),
    }


def investigate_feature(feature: str, df: pd.DataFrame, tournament_map: dict, flat_margin: float) -> dict:
    tt = _load_train_test_finding(feature)
    tournament = df["match_id"].map(tournament_map)
    n_unmapped = int(tournament.isna().sum())

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD
    _, edges = _bin_table(df, feature, is_discrete_cardinality, None)

    per_tournament = {}
    for label in ("WC2022", "Euro2024"):
        sub = df[tournament == label]
        bins, _ = _bin_table(sub, feature, is_discrete_cardinality, edges)
        values = [b["mean_xt"] for b in bins]
        per_tournament[label] = {
            "n_rows": int(len(sub)),
            "n_matches": int(sub["match_id"].nunique()),
            "shape": classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin) if values else "insufficient data",
            "mean_xt_range": round(max(values) - min(values), 8) if values else None,
            "bin_mean_min": round(min(values), 8) if values else None,
            "bin_mean_max": round(max(values), 8) if values else None,
            "curve_crosses_zero": bool(values and min(values) < 0 < max(values)),
            "bins": bins,
        }

    agree = per_tournament["WC2022"]["shape"] == per_tournament["Euro2024"]["shape"]
    if agree:
        verdict = "arbitrary train/test noise"
        reason = (
            f"Same shape classification ({per_tournament['WC2022']['shape']}) in both WC2022 and Euro2024 -- a "
            "genuine tournament-level population difference would be expected to show up here, and it doesn't."
        )
    else:
        verdict = "genuine tournament-level difference"
        reason = (
            f"Shape classification differs between tournaments (WC2022: {per_tournament['WC2022']['shape']}, "
            f"Euro2024: {per_tournament['Euro2024']['shape']}) -- consistent with a real difference between the "
            "two populations (rules, pitch dimensions, squad quality), not just arbitrary train/test noise."
        )

    # Adaptation 3: does the direction of the relationship itself differ
    # between tournaments? This question has no xg counterpart (xg's binned
    # means are non-negative by construction, so a sign flip is impossible).
    sign_flip = (
        per_tournament["WC2022"]["curve_crosses_zero"] != per_tournament["Euro2024"]["curve_crosses_zero"]
    )

    return {
        "feature": feature,
        "dataset": "active",
        "train_test_finding": tt,
        "n_unmapped_rows": n_unmapped,
        "per_tournament": per_tournament,
        "verdict": verdict,
        "verdict_reason": reason,
        "zero_crossing_differs_between_tournaments": sign_flip,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tournament_map = load_match_tournament_map()

    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    flat_margin = scale["flat_margin"]

    results = [investigate_feature(f, df, tournament_map, flat_margin) for f in ACTIVE_INVESTIGATE]

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": TARGET,
        "target_scale": scale,
        "tournaments": ["WC2022", "Euro2024"],
        "n_quantile_bins": N_QUANTILE_BINS,
        "flat_margin": flat_margin,
        "step_0_scope_note": (
            "The xg portal's TOURNAMENT_STABILITY_CHECK covers all 10 features in "
            "generate_tournament_stability_check.INVESTIGATE (6 active + 4 passive). This prompt is scoped to "
            "the active leg, so the same imported list is filtered to its 6 ACTIVE entries and the 4 passive "
            "ones are omitted -- a scope reduction, not a methodology change. The list is imported rather than "
            "re-derived from this portal's own flagged-inconsistent features, exactly as the xg version does, "
            "so the same features are compared across all three targets."
        ),
        "excluded_from_investigation": {
            "nearest_defender_distance": "already-known self-reference bug (73.6% of rows ~0m) -- a separate "
                                         "issue, not investigated here. Carried over unchanged.",
        },
        "methodology": (
            "xT-delta counterpart to the xg and binary TOURNAMENT_STABILITY_CHECKs -- same tournament split, "
            "same imported feature list (active half), same bin edges computed once on the pooled data and "
            "reused for both tournaments, same classify_shape() heuristic. The ONLY change is flat_margin, "
            "which is translated from the xg suite's 0.5x-mean-xg convention via xg's own standard-deviation "
            "fraction, because a multiple of this target's near-zero NEGATIVE mean would be meaningless. One "
            "field has no xg counterpart: zero_crossing_differs_between_tournaments, which asks whether the "
            "binned curve changes DIRECTION between tournaments -- impossible to ask of a non-negative target."
        ),
        "prompt_64_cross_references": {"distribution_shape": xc.PROMPT_64_SHAPE_FINDING},
        "features": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    for r in results:
        print(f"{r['feature']}: {r['verdict']}")
        print(f"  WC2022 shape={r['per_tournament']['WC2022']['shape']}  "
              f"Euro2024 shape={r['per_tournament']['Euro2024']['shape']}  "
              f"zero-crossing differs={r['zero_crossing_differs_between_tournaments']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
