"""CLI entrypoint: prompt 24 Part B -- for every feature prompt 21 flagged
as train/test-inconsistent (except nearest_defender_distance, a separate
known self-reference bug), split by TOURNAMENT (WC2022 vs Euro2024) instead
of by the train/test split, and recompute the same binned shot-rate curve.

This checks a different hypothesis than the train/test check did: not
"does this generalise across an arbitrary match split" but "is this
actually two different populations mixed together" (rule differences,
pitch dimensions, squad quality between a World Cup and a Euros). Reuses
the exact same binning (same edges as the unconditional pass, from
active/passive_numerical_target_atlas.json) and classify_shape() as
prompt 21/22, just swapping the split variable.

Verdict per feature: if the shape classification matches between
tournaments, the earlier train/test instability looks like arbitrary
noise (a real population-mixing artefact would show up here too, and
doesn't). If it differs, that's consistent with a genuine tournament-level
difference, not just noise.

Usage:
    python -m src.eda.generate_tournament_stability_check
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import (
    DISCRETE_CARDINALITY_THRESHOLD,
    N_QUANTILE_BINS,
    TARGET,
    _bin_table,
    classify_shape,
)
from src.eda.tournament_mapping import load_match_tournament_map

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "TOURNAMENT_STABILITY_CHECK.json"

# feature -> dataset key. nearest_defender_distance deliberately excluded
# (already-known self-reference bug, a separate issue per prompt 24).
INVESTIGATE = {
    "defenders_within_10m": "active",
    "defenders_within_5m": "active",
    "events_elapsed_in_possession": "active",
    "local_numerical_balance_5m": "active",
    "match_time_seconds": "active",
    "attackers_within_5m": "active",
    "top_option_1_distance_from_ball": "passive",
    "top_option_3_dx": "passive",
    "top_option_3_dy": "passive",
    "angle_to_attacking_goal": "passive",
}


def _load_train_test_finding(feature: str, dataset_key: str) -> dict:
    """Restate prompt 21's already-known train/test finding for context --
    read directly from the existing atlas JSON, not re-typed by hand."""
    atlas_path = REPO_ROOT / "reports" / "eda" / f"{dataset_key}_numerical_target_atlas.json"
    data = json.loads(atlas_path.read_text(encoding="utf-8"))
    entry = next(f for f in data["features"] if f["feature"] == feature)
    cc = entry["consistency_check"]
    return {
        "overall_shape": entry["shape"],
        "spearman_rho": entry["spearman_rho"],
        "train_val_shape": cc.get("train_val_shape"),
        "test_shape": cc.get("test_shape"),
        "train_test_consistent": cc.get("consistent"),
    }


def investigate_feature(feature: str, dataset_key: str, df: pd.DataFrame, tournament_map: dict[int, str]) -> dict:
    train_test_finding = _load_train_test_finding(feature, dataset_key)

    match_ids = df["match_id"]
    tournament = match_ids.map(tournament_map)
    n_unmapped = int(tournament.isna().sum())

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    # Same edges as the unconditional pass: computed once on the full
    # population, then applied to each tournament subset -- apples-to-apples,
    # the same discipline the train/test consistency check already used.
    _, edges = _bin_table(df, feature, is_discrete_cardinality, None)

    per_tournament = {}
    for label in ("WC2022", "Euro2024"):
        sub_df = df[tournament == label]
        bins, _ = _bin_table(sub_df, feature, is_discrete_cardinality, edges)
        rates = [b["shot_rate_pct"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], rates) if rates else "insufficient data"
        per_tournament[label] = {
            "n_rows": int(len(sub_df)),
            "n_matches": int(sub_df["match_id"].nunique()),
            "shape": shape,
            "rate_range_pp": round(max(rates) - min(rates), 3) if rates else None,
            "bins": bins,
        }

    tournaments_agree = per_tournament["WC2022"]["shape"] == per_tournament["Euro2024"]["shape"]
    if tournaments_agree:
        verdict = "arbitrary train/test noise"
        verdict_reason = (
            f"Same shape classification ({per_tournament['WC2022']['shape']}) in both WC2022 and Euro2024 -- a "
            "genuine tournament-level population difference would be expected to show up here, and it doesn't. "
            "The train/test instability prompt 21 found looks like noise from an arbitrary match split, not a "
            "real population-mixing artefact."
        )
    else:
        verdict = "genuine tournament-level difference"
        verdict_reason = (
            f"Shape classification differs between tournaments (WC2022: {per_tournament['WC2022']['shape']}, "
            f"Euro2024: {per_tournament['Euro2024']['shape']}) -- consistent with a real difference between the "
            "two populations (rules, pitch dimensions, squad quality), not just arbitrary train/test noise."
        )

    return {
        "feature": feature,
        "dataset": dataset_key,
        "train_test_finding": train_test_finding,
        "n_unmapped_rows": n_unmapped,
        "per_tournament": per_tournament,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def main() -> None:
    tournament_map = load_match_tournament_map()

    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}

    results = []
    for feature, dataset_key in INVESTIGATE.items():
        results.append(investigate_feature(feature, dataset_key, dfs[dataset_key], tournament_map))

    output = {
        "target": TARGET,
        "tournaments": ["WC2022", "Euro2024"],
        "n_quantile_bins": N_QUANTILE_BINS,
        "excluded_from_investigation": {
            "nearest_defender_distance": "already-known self-reference bug (73.6% of rows ~0m) -- a separate issue, not investigated here per prompt 24's explicit instruction.",
        },
        "methodology": (
            "For each feature, bin edges are computed once on the full population (same edges the train/test "
            "consistency check used), then applied to each tournament subset -- apples-to-apples. Shape "
            "classification uses the same classify_shape() heuristic as prompts 21/22. Verdict: matching shapes "
            "across tournaments -> the earlier train/test instability looks like arbitrary noise; differing "
            "shapes -> consistent with a genuine tournament-level difference."
        ),
        "features": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    for r in results:
        print(f"{r['feature']} ({r['dataset']}): {r['verdict']}")
        print(f"  WC2022 shape={r['per_tournament']['WC2022']['shape']}  Euro2024 shape={r['per_tournament']['Euro2024']['shape']}")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
