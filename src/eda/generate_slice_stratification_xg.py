"""CLI entrypoint: the continuous-target (xG) counterpart to
generate_slice_stratification.py -- same 7 features, same 2 slicers each
(14 cells), same category-divergence / small-n / driven-by-one-category
logic, against mean target_future_xg_10s instead of shot-rate percentage.

flat_margin for classify_shape() is relative to each dataset's own overall
mean xG (same convention as generate_numerical_xg_target_analysis.py).

Usage:
    python -m src.eda.generate_slice_stratification_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, classify_shape
from src.eda.generate_numerical_xg_target_analysis import FLAT_MARGIN_RATIO, TARGET, _bin_table
from src.eda.generate_slice_stratification import FEATURE_SLICER_PLAN, SMALL_N_THRESHOLD, STRUCTURAL_CAUTION_CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "SLICE_STRATIFICATION.json"


def _slice_one(feature: str, slicer: str, dataset_key: str, df: pd.DataFrame, flat_margin: float) -> dict:
    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    overall_bins, edges = _bin_table(df, feature, is_discrete_cardinality, None)
    overall_values = [b["mean_xg"] for b in overall_bins]
    overall_shape = classify_shape([b["bin"] for b in overall_bins], overall_values, flat_margin=flat_margin) if overall_values else "insufficient data"

    categories: dict[str, dict] = {}
    matching_categories: list[str] = []
    non_small_n_categories: list[str] = []

    for cat_value, cat_df in df.groupby(slicer, dropna=False, observed=True):
        cat_label = str(cat_value)
        n_rows = len(cat_df)
        bins, _ = _bin_table(cat_df, feature, is_discrete_cardinality, edges)
        values = [b["mean_xg"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin) if values else "insufficient data"
        value_range = round(max(values) - min(values), 6) if values else None

        is_small_n = n_rows < SMALL_N_THRESHOLD
        structural_caution = STRUCTURAL_CAUTION_CATEGORIES.get((dataset_key, slicer, cat_label))
        diverges = shape != overall_shape

        categories[cat_label] = {
            "n_rows": n_rows,
            "shape": shape,
            "mean_xg_range": value_range,
            "diverges_from_overall": diverges,
            "small_n": is_small_n,
            "structural_caution": structural_caution,
            "bins": bins,
        }

        if not is_small_n and structural_caution is None:
            non_small_n_categories.append(cat_label)
            if not diverges:
                matching_categories.append(cat_label)

    driven_by_single_category = len(matching_categories) == 1 and len(non_small_n_categories) > 1

    return {
        "feature": feature,
        "slicer": slicer,
        "dataset": dataset_key,
        "overall_shape": overall_shape,
        "overall_mean_xg_range": round(max(overall_values) - min(overall_values), 6) if overall_values else None,
        "n_categories": len(categories),
        "categories": categories,
        "n_categories_diverging": sum(1 for c in categories.values() if c["diverges_from_overall"] and not c["small_n"] and not c["structural_caution"]),
        "driven_by_single_category": matching_categories[0] if driven_by_single_category else None,
    }


def main() -> None:
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}
    flat_margins = {ds: FLAT_MARGIN_RATIO * float(df[TARGET].mean()) for ds, df in dfs.items()}

    results: dict[str, dict] = {"active": {}, "passive": {}}
    for dataset_key, feature, slicers in FEATURE_SLICER_PLAN:
        results[dataset_key][feature] = {}
        for slicer in slicers:
            cell = _slice_one(feature, slicer, dataset_key, dfs[dataset_key], flat_margins[dataset_key])
            results[dataset_key][feature][slicer] = cell
            flag = f" -- DRIVEN BY {cell['driven_by_single_category']}" if cell["driven_by_single_category"] else ""
            print(f"{dataset_key}/{feature} x {slicer}: overall={cell['overall_shape']}, "
                  f"{cell['n_categories_diverging']}/{cell['n_categories']} categories diverge{flag}")

    output = {
        "target": TARGET,
        "small_n_threshold": SMALL_N_THRESHOLD,
        "flat_margin_ratio": FLAT_MARGIN_RATIO,
        "flat_margins_by_dataset": {k: round(v, 6) for k, v in flat_margins.items()},
        "methodology": (
            "Continuous-target (xG) counterpart to SLICE_STRATIFICATION.json (binary target) -- same 7 features, "
            "same 2 slicers each, same category-divergence / small-n / driven-by-one-category logic, but "
            "flat_margin is relative to each dataset's own overall mean xG rather than a fixed percentage-point "
            "margin. 'unclassified' (passive defender_functional_role) is always flagged structurally regardless "
            "of row count, same as the binary-target version."
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
