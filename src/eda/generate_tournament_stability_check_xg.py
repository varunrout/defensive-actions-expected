"""CLI entrypoint: the continuous-target (xG) counterpart to
generate_tournament_stability_check.py -- same tournament-split method
(WC2022 vs Euro2024, same bin edges as the unconditional pass, same
classify_shape() heuristic), against mean target_future_xg_10s instead of
shot-rate percentage.

Same 10 features investigated as the binary-target version (mirrors the
population-mixing hypothesis test, not re-derived from the xG atlas's own
flagged-inconsistent list) -- nearest_defender_distance excluded for the
same reason (a separate, already-known self-reference bug).

flat_margin for classify_shape() is RELATIVE to each dataset's own overall
mean xG (same convention as generate_numerical_xg_target_analysis.py), not
a fixed percentage-point margin -- xG has mean ~0.008, nothing like a
0-100% rate.

Usage:
    python -m src.eda.generate_tournament_stability_check_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, N_QUANTILE_BINS, classify_shape
from src.eda.generate_numerical_xg_target_analysis import FLAT_MARGIN_RATIO, TARGET, _bin_table
from src.eda.generate_tournament_stability_check import INVESTIGATE
from src.eda.tournament_mapping import load_match_tournament_map

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "TOURNAMENT_STABILITY_CHECK.json"


def _load_train_test_finding(feature: str, dataset_key: str) -> dict:
    """Restate the xG atlas's already-known train/test finding for context --
    read directly from the existing atlas JSON, not re-typed by hand."""
    atlas_path = REPO_ROOT / "reports" / "eda_xg" / f"{dataset_key}_numerical_target_atlas.json"
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


def investigate_feature(feature: str, dataset_key: str, df: pd.DataFrame, tournament_map: dict[int, str], flat_margin: float) -> dict:
    train_test_finding = _load_train_test_finding(feature, dataset_key)

    match_ids = df["match_id"]
    tournament = match_ids.map(tournament_map)
    n_unmapped = int(tournament.isna().sum())

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    _, edges = _bin_table(df, feature, is_discrete_cardinality, None)

    per_tournament = {}
    for label in ("WC2022", "Euro2024"):
        sub_df = df[tournament == label]
        bins, _ = _bin_table(sub_df, feature, is_discrete_cardinality, edges)
        values = [b["mean_xg"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin) if values else "insufficient data"
        per_tournament[label] = {
            "n_rows": int(len(sub_df)),
            "n_matches": int(sub_df["match_id"].nunique()),
            "shape": shape,
            "mean_xg_range": round(max(values) - min(values), 6) if values else None,
            "bins": bins,
        }

    tournaments_agree = per_tournament["WC2022"]["shape"] == per_tournament["Euro2024"]["shape"]
    if tournaments_agree:
        verdict = "arbitrary train/test noise"
        verdict_reason = (
            f"Same shape classification ({per_tournament['WC2022']['shape']}) in both WC2022 and Euro2024 -- a "
            "genuine tournament-level population difference would be expected to show up here, and it doesn't."
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
    flat_margins = {ds: FLAT_MARGIN_RATIO * float(df[TARGET].mean()) for ds, df in dfs.items()}

    results = []
    for feature, dataset_key in INVESTIGATE.items():
        results.append(investigate_feature(feature, dataset_key, dfs[dataset_key], tournament_map, flat_margins[dataset_key]))

    output = {
        "target": TARGET,
        "tournaments": ["WC2022", "Euro2024"],
        "n_quantile_bins": N_QUANTILE_BINS,
        "flat_margin_ratio": FLAT_MARGIN_RATIO,
        "flat_margins_by_dataset": {k: round(v, 6) for k, v in flat_margins.items()},
        "excluded_from_investigation": {
            "nearest_defender_distance": "already-known self-reference bug (73.6% of rows ~0m) -- a separate issue, not investigated here.",
        },
        "methodology": (
            "Continuous-target (xG) counterpart to TOURNAMENT_STABILITY_CHECK.json (binary target) -- same 10 "
            "features, same tournament split, same bin edges reused across tournaments, same classify_shape() "
            "heuristic, but flat_margin is relative to each dataset's own overall mean xG rather than a fixed "
            "percentage-point margin."
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
