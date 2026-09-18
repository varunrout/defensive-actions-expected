"""CLI entrypoint: prompt 28 -- SLICE_STRATIFICATION_V2 (binary target).
Two slicer dimensions prompts 24/25 never tried: `defender_archetype_name`
(the derived clustering label from prompts 18/19 -- a real column on
passive_defense.parquet, not a locked feature) and the locked boolean flags
on both datasets (the cheapest possible category, 2 values, never used as a
slicer before this).

New file, NOT an extension of SLICE_STRATIFICATION.json (V1) -- that file's
118 cells (prompts 24/25) are locked and untouched. Same method, different
slicer set: V1 reused unchanged for the archetype-vs-boolean question, this
file is where the archetype and boolean slicers live.

Method notes:
  - defender_archetype_name: reuses generate_slice_stratification._slice_one
    unchanged, on a pre-filtered df (null archetype rows dropped first --
    same treatment 'unclassified' gets elsewhere; those rows are the same
    ones behind defender_functional_role's 'unclassified', n=1026).
  - Boolean slicers: a 2-category slicer has no meaningfully separate
    "overall" to diverge from, so `diverges_from_overall` is REDEFINED here
    to mean "does the True category's shape differ from the False
    category's shape" -- both categories share the same value (they either
    diverge from each other or they don't). This is a real method
    difference from V1's category-vs-pooled-overall comparison, stated here
    rather than left implicit. `driven_by_single_category` doesn't apply to
    a 2-category slicer and is always null.

Grid: active 12 features x 9 boolean slicers = 108 cells. Passive 13
features x (1 archetype slicer + 6 boolean slicers) = 91 cells. Total 199,
all new (V1's slicer set and V2's don't overlap).

Usage:
    python -m src.eda.generate_slice_stratification_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, TARGET, _bin_table, classify_shape
from src.eda.generate_slice_stratification import SMALL_N_THRESHOLD, _slice_one

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "shot_target" / "SLICE_STRATIFICATION_V2.json"

ACTIVE_FEATURES = [
    "defender_spread", "attacker_spread", "attacking_goal_centrality", "attacker_defender_ratio",
    "defenders_within_10m", "visible_attacker_count", "visible_defender_count", "defenders_within_5m",
    "angle_to_attacking_goal", "possession_elapsed_seconds", "attackers_within_10m",
    "defenders_between_ball_and_attacking_goal",
]
PASSIVE_FEATURES = [
    "marking_tightness", "lane_screening_score_option_1", "overload_score", "defender_x",
    "top_option_1_threat_score", "top_option_2_threat_score", "top_option_3_threat_score",
    "lane_screening_score_option_2", "lane_screening_score_option_3", "attacking_goal_centrality",
    "top_option_2_distance_from_ball", "top_option_3_distance_from_ball", "engagement_distance_to_carrier",
]

ACTIVE_BOOLEAN_SLICERS = [
    "is_in_defending_box", "is_in_attacking_box", "counterpress", "phase_changed_since_prev_event",
    "action_was_under_opponent_possession", "action_retained_defensive_team_control",
    "has_previous_event", "has_visible_attacker", "has_visible_defender",
]
PASSIVE_BOOLEAN_SLICERS = ["is_in_defending_box", "is_in_attacking_box", "is_wide_lane", "has_option_2", "has_option_3", "is_goal_side_of_nearest_attacker"]

ARCHETYPE_SLICER = "defender_archetype_name"


def _slice_boolean(feature: str, slicer: str, dataset_key: str, df: pd.DataFrame) -> dict:
    """A 2-category (True/False) version of _slice_one -- see module
    docstring for why diverges_from_overall is redefined here."""
    # A few boolean columns carry nulls (e.g. is_goal_side_of_nearest_attacker,
    # stored as object with True/False/None) -- those rows are excluded from
    # this slicer's categories, same null treatment as everywhere else.
    df_valid = df.loc[df[slicer].notna()]
    bool_col = df_valid[slicer].astype(bool)

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    overall_bins, edges = _bin_table(df, feature, is_discrete_cardinality, None)
    overall_rates = [b["shot_rate_pct"] for b in overall_bins]
    overall_shape = classify_shape([b["bin"] for b in overall_bins], overall_rates) if overall_rates else "insufficient data"

    categories: dict[str, dict] = {}
    shapes: dict[bool, str] = {}
    for flag_value in (True, False):
        cat_df = df_valid.loc[bool_col == flag_value]
        n_rows = len(cat_df)
        bins, _ = _bin_table(cat_df, feature, is_discrete_cardinality, edges)
        rates = [b["shot_rate_pct"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], rates) if rates else "insufficient data"
        shapes[flag_value] = shape
        categories[str(flag_value)] = {
            "n_rows": n_rows,
            "shape": shape,
            "rate_range_pp": round(max(rates) - min(rates), 3) if rates else None,
            "small_n": n_rows < SMALL_N_THRESHOLD,
            "structural_caution": None,
            "bins": bins,
        }

    diverges = shapes[True] != shapes[False]
    for cat in categories.values():
        cat["diverges_from_overall"] = diverges

    n_diverging = 2 if diverges else 0

    return {
        "feature": feature,
        "slicer": slicer,
        "slicer_type": "boolean",
        "dataset": dataset_key,
        "overall_shape": overall_shape,
        "overall_rate_range_pp": round(max(overall_rates) - min(overall_rates), 3) if overall_rates else None,
        "n_categories": 2,
        "categories": categories,
        "n_categories_diverging": n_diverging,
        "driven_by_single_category": None,
    }


def main() -> None:
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    passive_df_archetype = passive_df[passive_df[ARCHETYPE_SLICER].notna()]

    results: dict[str, dict] = {"active": {}, "passive": {}}
    n_cells = 0

    for feature in ACTIVE_FEATURES:
        results["active"][feature] = {}
        for slicer in ACTIVE_BOOLEAN_SLICERS:
            cell = _slice_boolean(feature, slicer, "active", active_df)
            results["active"][feature][slicer] = cell
            n_cells += 1
            print(f"[active] {feature} x {slicer}: overall={cell['overall_shape']}, diverges(True vs False)={cell['n_categories_diverging'] > 0}")

    for feature in PASSIVE_FEATURES:
        results["passive"][feature] = {}
        archetype_cell = _slice_one(feature, ARCHETYPE_SLICER, "passive", passive_df_archetype)
        archetype_cell["slicer_type"] = "categorical"
        results["passive"][feature][ARCHETYPE_SLICER] = archetype_cell
        n_cells += 1
        print(f"[passive] {feature} x {ARCHETYPE_SLICER}: overall={archetype_cell['overall_shape']}, "
              f"{archetype_cell['n_categories_diverging']}/{archetype_cell['n_categories']} categories diverge")

        for slicer in PASSIVE_BOOLEAN_SLICERS:
            cell = _slice_boolean(feature, slicer, "passive", passive_df)
            results["passive"][feature][slicer] = cell
            n_cells += 1
            print(f"[passive] {feature} x {slicer}: overall={cell['overall_shape']}, diverges(True vs False)={cell['n_categories_diverging'] > 0}")

    assert n_cells == 199, f"Expected exactly 199 cells, computed {n_cells}."

    # --- Cross-dataset summary (V2-only, kept separate from V1's) ---
    by_feature = {}
    genuine_divergences = []
    for dataset, features in results.items():
        for feature, slicers in features.items():
            diverging_slicers, stable_slicers = [], []
            for slicer, cell in slicers.items():
                has_div = False
                for cat_label, cat in cell["categories"].items():
                    is_genuine = cat["diverges_from_overall"] and not cat["small_n"] and not cat.get("structural_caution")
                    if is_genuine:
                        has_div = True
                        genuine_divergences.append({
                            "dataset": dataset, "feature": feature, "slicer": slicer, "slicer_type": cell["slicer_type"],
                            "category": cat_label, "category_shape": cat["shape"], "overall_shape": cell["overall_shape"],
                        })
                (diverging_slicers if has_div else stable_slicers).append(slicer)
            by_feature[f"{dataset}/{feature}"] = {
                "dataset": dataset, "feature": feature,
                "n_slicers_tested": len(slicers),
                "diverging_slicers": sorted(diverging_slicers),
                "stable_slicers": sorted(stable_slicers),
                "needs_interaction_term_signal": bool(diverging_slicers),
            }
    genuine_divergences.sort(key=lambda r: (r["dataset"], r["feature"], r["slicer"], r["category"]))

    archetype_divergences = [d for d in genuine_divergences if d["slicer"] == ARCHETYPE_SLICER]
    boolean_divergences = [d for d in genuine_divergences if d["slicer"] != ARCHETYPE_SLICER]
    n_archetype_cells = sum(1 for f in results["passive"].values() for s in f if s == ARCHETYPE_SLICER)
    n_boolean_cells = 199 - n_archetype_cells
    closing_note = (
        f"defender_archetype_name alone produced {len(archetype_divergences)} genuine divergence(s) across "
        f"{n_archetype_cells} cells ({len(archetype_divergences) / n_archetype_cells:.2f}/cell). The 15 boolean "
        f"slicers combined (9 active + 6 passive) produced {len(boolean_divergences)} genuine divergence(s) across "
        f"{n_boolean_cells} cells ({len(boolean_divergences) / n_boolean_cells:.2f}/cell). "
        + (
            "The archetype slicer is at least as informative per cell as the combined boolean slicers -- it does not "
            "need several separate boolean flags to match its divergence rate."
            if (len(archetype_divergences) / n_archetype_cells) >= (len(boolean_divergences) / n_boolean_cells) else
            "The combined boolean slicers still produce more divergence per cell than the archetype slicer alone -- "
            "archetype does not substitute for testing the individual boolean flags."
        )
    )

    output = {
        "target": TARGET,
        "small_n_threshold": SMALL_N_THRESHOLD,
        "v1_cross_reference": (
            "See reports/analysis/shot_target/SLICE_STRATIFICATION.json (V1, prompts 24/25) for the original 118-cell grid over "
            "locked categorical slicers (phase_label, position/position_group, event_type, play_pattern, "
            "phase_label_prev_event, period, defender_functional_role, on_ball_event_type). V1 is untouched by "
            "this file -- V2 tests a disjoint slicer set (defender_archetype_name + every locked boolean flag) "
            "against the same feature pools, not a superset or a replacement."
        ),
        "methodology": (
            "Same method as V1 (_bin_table/classify_shape, same edges reused across categories, small_n threshold "
            f"{SMALL_N_THRESHOLD}) with one real difference: a boolean slicer only has 2 categories (True/False), "
            "so there's no meaningfully separate 'overall' to diverge from. For boolean-slicer cells, "
            "diverges_from_overall is REDEFINED to mean 'does the True category's shape differ from the False "
            "category's shape' -- both categories carry the same value since a 2-way split either diverges or it "
            "doesn't, symmetrically. driven_by_single_category doesn't apply to a 2-category slicer and is always "
            "null for boolean cells. defender_archetype_name (passive only, categorical, 8 non-null values) reuses "
            "V1's exact category logic unchanged, computed on rows with a non-null archetype (the same ~1026 rows "
            "behind defender_functional_role's 'unclassified' are excluded here too)."
        ),
        "cross_dataset_summary": {
            "by_feature": by_feature,
            "n_features_with_at_least_one_diverging_slicer": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal"]),
            "n_features_fully_stable": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal"]),
            "genuine_divergences": genuine_divergences,
            "n_genuine_divergences": len(genuine_divergences),
            "closing_note_archetype_vs_boolean": closing_note,
        },
        "results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n{n_cells} cells computed. Genuine divergences: {len(genuine_divergences)}")
    print(closing_note)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
