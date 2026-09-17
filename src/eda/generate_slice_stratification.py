"""CLI entrypoint: prompt 24 Part C -- categorical stratification. For a set
of features with an already-established pattern (from prompts 21/22), slice
by a categorical column and check whether the shape classification holds
inside every category or is driven by one slice of the data.

Reuses the exact same binning (_bin_table, same edges as the unconditional
pass) and classify_shape() as prompts 21/22/24-B -- this is the same method,
just conditioned on a category instead of a train/test split or a
tournament.

Feature x slicer combinations (7 features, each sliced 2 ways = 14 cells):
  passive x defender_functional_role, passive x phase_label:
    zone_defensive_value, marking_tightness, lane_screening_score_option_1,
    overload_score (control case -- a "clean" monotonic finding, checked
    the same way as the messy ones)
  active x position_group, active x phase_label:
    defender_spread, attacking_goal_centrality, attacker_defender_ratio

Flags, not buried in a table:
  - "unclassified" (passive defender_functional_role) is a structural
    category (n<2 visible defenders), not a peer behavioural role -- always
    flagged, regardless of row count.
  - any category below SMALL_N_THRESHOLD rows -- flagged, not silently
    included as if it were as trustworthy as the large categories.
  - any category whose shape classification differs from the overall
    pattern -- flagged as a divergence.
  - if the overall pattern is reproduced by only one category (of the
    non-small-n ones), flagged as "driven by a single category."

Usage:
    python -m src.eda.generate_slice_stratification
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, TARGET, _bin_table, classify_shape

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "SLICE_STRATIFICATION.json"

SMALL_N_THRESHOLD = 1000
STRUCTURAL_CAUTION_CATEGORIES = {
    ("passive", "defender_functional_role", "unclassified"): (
        "Structural category (n<2 visible defenders -- nothing to be 'relative to'), not a peer behavioural "
        "role like the other 4. Same caution already applied in the archetype clustering work -- never present "
        "this as a fifth role."
    ),
    # Added for prompt 25's full cross-product (position and
    # phase_label_prev_event weren't slicers used in prompt 24).
    ("active", "position", "Goalkeeper"): (
        "Structurally different, not a peer position: goalkeepers almost never face the ball 1v1 the way outfield "
        "defenders do, and the position carries near-zero shot exposure in this dataset's defensive-action frame -- "
        "not a smaller version of the same role."
    ),
    ("active", "phase_label_prev_event", "nan"): (
        "Not a phase label -- this is the first event of the possession, so there IS no previous event to have a "
        "phase. A structurally different case (no previous state), not a small sample of a real phase."
    ),
}

FEATURE_SLICER_PLAN: list[tuple[str, str, list[str]]] = [
    ("passive", "zone_defensive_value", ["defender_functional_role", "phase_label"]),
    ("passive", "marking_tightness", ["defender_functional_role", "phase_label"]),
    ("passive", "lane_screening_score_option_1", ["defender_functional_role", "phase_label"]),
    ("passive", "overload_score", ["defender_functional_role", "phase_label"]),
    ("active", "defender_spread", ["position_group", "phase_label"]),
    ("active", "attacking_goal_centrality", ["position_group", "phase_label"]),
    ("active", "attacker_defender_ratio", ["position_group", "phase_label"]),
]


def _slice_one(feature: str, slicer: str, dataset_key: str, df: pd.DataFrame) -> dict:
    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    # Same edges for the overall pattern and every category slice --
    # apples-to-apples, same discipline as the train/test and tournament checks.
    overall_bins, edges = _bin_table(df, feature, is_discrete_cardinality, None)
    overall_rates = [b["shot_rate_pct"] for b in overall_bins]
    overall_shape = classify_shape([b["bin"] for b in overall_bins], overall_rates) if overall_rates else "insufficient data"

    categories: dict[str, dict] = {}
    matching_categories: list[str] = []
    non_small_n_categories: list[str] = []

    for cat_value, cat_df in df.groupby(slicer, dropna=False, observed=True):
        cat_label = str(cat_value)
        n_rows = len(cat_df)
        bins, _ = _bin_table(cat_df, feature, is_discrete_cardinality, edges)
        rates = [b["shot_rate_pct"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], rates) if rates else "insufficient data"
        rate_range = round(max(rates) - min(rates), 3) if rates else None

        is_small_n = n_rows < SMALL_N_THRESHOLD
        structural_caution = STRUCTURAL_CAUTION_CATEGORIES.get((dataset_key, slicer, cat_label))
        diverges = shape != overall_shape

        categories[cat_label] = {
            "n_rows": n_rows,
            "shape": shape,
            "rate_range_pp": rate_range,
            "diverges_from_overall": diverges,
            "small_n": is_small_n,
            "structural_caution": structural_caution,
            "bins": bins,
        }

        if not is_small_n and structural_caution is None:
            non_small_n_categories.append(cat_label)
            if not diverges:
                matching_categories.append(cat_label)

    driven_by_single_category = (
        len(matching_categories) == 1 and len(non_small_n_categories) > 1
    )

    return {
        "feature": feature,
        "slicer": slicer,
        "dataset": dataset_key,
        "overall_shape": overall_shape,
        "overall_rate_range_pp": round(max(overall_rates) - min(overall_rates), 3) if overall_rates else None,
        "n_categories": len(categories),
        "categories": categories,
        "n_categories_diverging": sum(1 for c in categories.values() if c["diverges_from_overall"] and not c["small_n"] and not c["structural_caution"]),
        "driven_by_single_category": (
            matching_categories[0] if driven_by_single_category else None
        ),
    }


def main() -> None:
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}

    results: dict[str, dict] = {"active": {}, "passive": {}}
    for dataset_key, feature, slicers in FEATURE_SLICER_PLAN:
        results[dataset_key][feature] = {}
        for slicer in slicers:
            cell = _slice_one(feature, slicer, dataset_key, dfs[dataset_key])
            results[dataset_key][feature][slicer] = cell
            flag = f" -- DRIVEN BY {cell['driven_by_single_category']}" if cell["driven_by_single_category"] else ""
            print(f"{dataset_key}/{feature} x {slicer}: overall={cell['overall_shape']}, "
                  f"{cell['n_categories_diverging']}/{cell['n_categories']} categories diverge{flag}")

    output = {
        "target": TARGET,
        "small_n_threshold": SMALL_N_THRESHOLD,
        "methodology": (
            "Same binning (_bin_table, same edges as the unconditional pass) and classify_shape() as prompts "
            "21/22 -- conditioned on a category instead of a train/test split. A category is flagged small_n if "
            f"under {SMALL_N_THRESHOLD} rows; 'unclassified' (passive defender_functional_role) is always flagged "
            "structurally regardless of row count, since it's a different kind of category (n<2 visible "
            "defenders), not a peer behavioural role."
        ),
        "feature_slicer_plan": [
            {"dataset": ds, "feature": f, "slicers": sl} for ds, f, sl in FEATURE_SLICER_PLAN
        ],
        "results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
