"""CLI entrypoint: prompt 25 -- full feature x slicer cross-product slice
analysis. Extends reports/analysis/shot_target/SLICE_STRATIFICATION.json (prompt 24 Part C's
7 ad-hoc features, each sliced 2 ways = 14 leaf cells) with the full,
explicit cross-product of locked numerical features against locked
categorical slicers, per dataset. Same method as prompt 24 (_slice_one,
reused unchanged) -- broader coverage only, no new method.

Correction from the prompt's stated premise, verified against the live
feature_config.py before running (same discipline the prompt itself asks
for on slicer names): the prompt calls `distance_to_defending_goal` "the
locked sibling of dropped zone_defensive_value". That's not what
feature_config.py says. zone_defensive_value's own drop reason (in
REDUNDANCY_DROPPED_PASSIVE) does name distance_to_defending_goal as the
column it duplicates -- but distance_to_defending_goal is ALSO in
REDUNDANCY_DROPPED_PASSIVE, under its own entry: "Cluster 4
(defender-position, drop-to-one): see distance_to_attacking_box", which
resolves to `defender_x` (kept: "the most primitive measurement... upstream
of every derived distance/zone column here"). PASSIVE["continuous"] confirms
this: distance_to_defending_goal is absent, defender_x is present. The
actually-locked, model-facing sibling is `defender_x`, not
distance_to_defending_goal -- used here instead, so Pool A's 2 new cells
don't repeat the exact mistake this prompt exists to fix.

Slicer names were checked against feature_config.py's live locked
categorical lists (ACTIVE/PASSIVE["categorical"]) before running -- all
names in the prompt matched exactly, no corrections needed there.

Usage:
    python -m src.eda.generate_slice_stratification_expand
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_slice_stratification import OUTPUT_PATH, SMALL_N_THRESHOLD, _slice_one

REPO_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_SLICERS_ALL = ["phase_label", "position", "event_type", "play_pattern", "phase_label_prev_event", "period"]
ACTIVE_SLICERS_REDUCED = ["event_type", "play_pattern", "phase_label_prev_event", "period"]
PASSIVE_SLICERS_ALL = ["phase_label", "defender_functional_role", "on_ball_event_type", "period"]

# Pool A (passive, already-validated features) -- distance_to_defending_goal
# corrected to defender_x, see module docstring. marking_tightness,
# lane_screening_score_option_1, overload_score already fully covered from
# prompt 24 (confirmed on read-back below) -- no new cells for them.
POOL_A_NEW = [
    ("passive", "defender_x", ["defender_functional_role", "phase_label"]),
]

# Pool B active -- 3 features already have position_group/phase_label from
# prompt 24 (position_group isn't itself one of the 6 locked slicers below,
# but it's the same population cut as `position` in spirit -- per the
# prompt's explicit instruction, treated as already covering `position` and
# `phase_label` for these 3, so only the remaining 4 slicers are new here).
POOL_B_ACTIVE_REDUCED_FEATURES = ["defender_spread", "attacking_goal_centrality", "attacker_defender_ratio"]
POOL_B_ACTIVE_FULL_FEATURES = [
    "attacker_spread", "defenders_within_10m", "visible_attacker_count", "visible_defender_count",
    "defenders_within_5m", "angle_to_attacking_goal", "possession_elapsed_seconds",
    "attackers_within_10m", "defenders_between_ball_and_attacking_goal",
]

POOL_B_PASSIVE_FEATURES = [
    "top_option_1_threat_score", "top_option_2_threat_score", "top_option_3_threat_score",
    "lane_screening_score_option_2", "lane_screening_score_option_3", "attacking_goal_centrality",
    "top_option_2_distance_from_ball", "top_option_3_distance_from_ball", "engagement_distance_to_carrier",
]


def build_plan() -> list[tuple[str, str, str]]:
    """Flat list of (dataset, feature, slicer) new cells -- 104 total."""
    plan: list[tuple[str, str, str]] = []
    for dataset, feature, slicers in POOL_A_NEW:
        for slicer in slicers:
            plan.append((dataset, feature, slicer))
    for feature in POOL_B_ACTIVE_REDUCED_FEATURES:
        for slicer in ACTIVE_SLICERS_REDUCED:
            plan.append(("active", feature, slicer))
    for feature in POOL_B_ACTIVE_FULL_FEATURES:
        for slicer in ACTIVE_SLICERS_ALL:
            plan.append(("active", feature, slicer))
    for feature in POOL_B_PASSIVE_FEATURES:
        for slicer in PASSIVE_SLICERS_ALL:
            plan.append(("passive", feature, slicer))
    return plan


def build_summary(results: dict) -> dict:
    by_feature = {}
    genuine_divergences = []
    for dataset, features in results.items():
        for feature, slicers in features.items():
            diverging_slicers = []
            stable_slicers = []
            for slicer, cell in slicers.items():
                cell_has_divergence = False
                for cat_label, cat in cell["categories"].items():
                    if cat["diverges_from_overall"] and not cat["small_n"] and not cat.get("structural_caution"):
                        cell_has_divergence = True
                        genuine_divergences.append({
                            "dataset": dataset,
                            "feature": feature,
                            "slicer": slicer,
                            "category": cat_label,
                            "category_shape": cat["shape"],
                            "overall_shape": cell["overall_shape"],
                        })
                (diverging_slicers if cell_has_divergence else stable_slicers).append(slicer)

            by_feature[f"{dataset}/{feature}"] = {
                "dataset": dataset,
                "feature": feature,
                "n_slicers_tested": len(slicers),
                "diverging_slicers": sorted(diverging_slicers),
                "stable_slicers": sorted(stable_slicers),
                "needs_interaction_term_signal": bool(diverging_slicers),
            }

    genuine_divergences.sort(key=lambda r: (r["dataset"], r["feature"], r["slicer"], r["category"]))

    return {
        "by_feature": by_feature,
        "n_features_with_at_least_one_diverging_slicer": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal"]),
        "n_features_fully_stable": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal"]),
        "genuine_divergences": genuine_divergences,
        "n_genuine_divergences": len(genuine_divergences),
    }


def main() -> None:
    existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    # Prompt 24's "7" was 7 FEATURES (loosely called "7 feature x slicer
    # combinations"), each sliced 2 ways -- 14 individual (feature, slicer)
    # leaf cells, confirmed by reading the file back before this assertion.
    n_existing_cells = sum(len(slicers) for feats in existing["results"].values() for slicers in feats.values())
    assert n_existing_cells == 14, (
        f"Expected exactly 14 pre-existing (feature, slicer) cells (prompt 24's 7 features x 2 slicers each) "
        f"before extending, found {n_existing_cells} -- aborting rather than risk duplicating or clobbering entries."
    )

    plan = build_plan()
    assert len(plan) == 104, f"Expected exactly 104 new (feature, slicer) cells, built {len(plan)} -- aborting."

    for dataset, feature, slicer in plan:
        existing_slicers = existing["results"].get(dataset, {}).get(feature, {})
        if slicer in existing_slicers:
            raise AssertionError(f"Cell ({dataset}, {feature}, {slicer}) already exists in prompt 24's output -- aborting rather than duplicate.")

    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}

    n_added = 0
    for dataset, feature, slicer in plan:
        cell = _slice_one(feature, slicer, dataset, dfs[dataset])
        existing["results"].setdefault(dataset, {}).setdefault(feature, {})[slicer] = cell
        n_added += 1
        flag = f" -- DRIVEN BY {cell['driven_by_single_category']}" if cell["driven_by_single_category"] else ""
        print(f"{dataset}/{feature} x {slicer}: overall={cell['overall_shape']}, "
              f"{cell['n_categories_diverging']}/{cell['n_categories']} categories diverge{flag}")

    assert n_added == 104, f"Expected to add exactly 104 cells, added {n_added}."

    existing["small_n_threshold"] = SMALL_N_THRESHOLD  # unchanged, restated for clarity after extension
    existing["prompt_25_extension_note"] = (
        "104 new (feature, slicer) cells added by prompt 25 -- the full explicit cross-product of locked "
        "numerical features against locked categorical slicers, replacing prompt 24 Part C's ad-hoc selection "
        "(7 features x 2 slicers = 14 cells) with full coverage. Those 14 prompt-24 cells above this note are "
        "byte-for-byte unchanged. "
        "Correction: the prompt's Pool A used `distance_to_defending_goal` as 'the locked sibling of dropped "
        "zone_defensive_value' -- feature_config.py shows distance_to_defending_goal is ALSO dropped (Cluster 4, "
        "defender-position, drop-to-one), with `defender_x` as the actually-kept, model-facing sibling. `defender_x` "
        "is used for Pool A's 2 new cells instead. Slicer names were confirmed against feature_config.py's live "
        "locked categorical lists -- all matched exactly, no other corrections needed. `position` (active) and "
        "`phase_label_prev_event` (active) introduced 2 new structural_caution categories not present in prompt "
        "24: 'Goalkeeper' (near-zero shot exposure, not a peer outfield position) and 'nan' under "
        "phase_label_prev_event (first event of a possession has no previous phase to report, not a small sample "
        "of a real one)."
    )
    existing["cross_dataset_summary"] = build_summary(existing["results"])

    OUTPUT_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    summary = existing["cross_dataset_summary"]
    print(f"\n{n_added} new cells added ({sum(len(s) for f in existing['results'].values() for s in f.values())} total).")
    print(f"Features with >=1 diverging slicer: {summary['n_features_with_at_least_one_diverging_slicer']}")
    print(f"Features fully stable across all tested slicers: {summary['n_features_fully_stable']}")
    print(f"Genuine divergences (flagged, filtered of small_n/structural_caution): {summary['n_genuine_divergences']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
