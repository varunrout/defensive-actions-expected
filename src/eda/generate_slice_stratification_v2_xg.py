"""CLI entrypoint: prompt 28 -- SLICE_STRATIFICATION_V2 (xG / continuous
target). Same 199-cell grid as generate_slice_stratification_v2.py
(defender_archetype_name + every locked boolean flag as slicers), computed
against target_future_xg_10s instead, WITH the shot-conditional fields
built in from the start (n_given_shot, bins_given_shot, shape_given_shot,
diverges_from_overall_given_shot, occurrence_vs_quality) -- V1 did this in
two passes (prompt 25 then prompt 26's retrofit); V2 does it in one run
since the method is already established.

Same boolean-slicer scoping note as the binary V2 file: with only 2
categories, diverges_from_overall (and its _given_shot counterpart) means
"does True's shape differ from False's shape", not "vs a pooled overall".

Usage:
    python -m src.eda.generate_slice_stratification_v2_xg
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, classify_shape
from src.eda.generate_numerical_xg_target_analysis import FLAT_MARGIN_RATIO, SHOT_COL, TARGET, _bin_table
from src.eda.generate_slice_stratification_v2 import (
    ACTIVE_BOOLEAN_SLICERS,
    ACTIVE_FEATURES,
    ARCHETYPE_SLICER,
    PASSIVE_BOOLEAN_SLICERS,
    PASSIVE_FEATURES,
)
from src.eda.generate_slice_stratification_xg import SMALL_N_THRESHOLD, _slice_one

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "analysis" / "xg_target" / "SLICE_STRATIFICATION_V2.json"


def _given_shot_fields(cat_df: pd.DataFrame, feature: str, is_discrete_cardinality: bool, edges, flat_margin_given_shot: float, ref_shape_given_shot: str, already_small_n: bool, small_n_threshold: int) -> dict:
    shot_df = cat_df.loc[cat_df[SHOT_COL] == 1]
    n_given_shot = len(shot_df)
    bins_given_shot, _ = _bin_table(shot_df, feature, is_discrete_cardinality, edges)
    values = [b["mean_xg"] for b in bins_given_shot]
    shape_given_shot = classify_shape([b["bin"] for b in bins_given_shot], values, flat_margin=flat_margin_given_shot) if values else "insufficient data"
    diverges_given_shot = shape_given_shot != ref_shape_given_shot

    too_small = already_small_n or n_given_shot < small_n_threshold or shape_given_shot == "insufficient data"
    if too_small:
        occurrence_vs_quality = "inconclusive"
    elif shape_given_shot == "flat":
        occurrence_vs_quality = "occurrence-only"
    else:
        occurrence_vs_quality = "occurrence+quality"

    return {
        "n_given_shot": n_given_shot,
        "bins_given_shot": bins_given_shot,
        "shape_given_shot": shape_given_shot,
        "diverges_from_overall_given_shot": diverges_given_shot,
        "occurrence_vs_quality": occurrence_vs_quality,
    }


def _slice_boolean_xg(feature: str, slicer: str, dataset_key: str, df: pd.DataFrame, flat_margin: float, flat_margin_given_shot: float, small_n_threshold: int) -> dict:
    df_valid = df.loc[df[slicer].notna()]
    bool_col = df_valid[slicer].astype(bool)

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    overall_bins, edges = _bin_table(df, feature, is_discrete_cardinality, None)
    overall_values = [b["mean_xg"] for b in overall_bins]
    overall_shape = classify_shape([b["bin"] for b in overall_bins], overall_values, flat_margin=flat_margin) if overall_values else "insufficient data"

    shot_df_all = df.loc[df[SHOT_COL] == 1]
    overall_bins_given_shot, _ = _bin_table(shot_df_all, feature, is_discrete_cardinality, edges)
    overall_values_given_shot = [b["mean_xg"] for b in overall_bins_given_shot]
    overall_shape_given_shot = (
        classify_shape([b["bin"] for b in overall_bins_given_shot], overall_values_given_shot, flat_margin=flat_margin_given_shot)
        if overall_values_given_shot else "insufficient data"
    )

    categories: dict[str, dict] = {}
    shapes: dict[bool, str] = {}
    shapes_given_shot: dict[bool, str] = {}
    small_n_flags: dict[bool, bool] = {}

    for flag_value in (True, False):
        cat_df = df_valid.loc[bool_col == flag_value]
        n_rows = len(cat_df)
        bins, _ = _bin_table(cat_df, feature, is_discrete_cardinality, edges)
        values = [b["mean_xg"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin) if values else "insufficient data"
        shapes[flag_value] = shape
        is_small_n = n_rows < small_n_threshold
        small_n_flags[flag_value] = is_small_n

        given_shot_fields = _given_shot_fields(cat_df, feature, is_discrete_cardinality, edges, flat_margin_given_shot, overall_shape_given_shot, is_small_n, small_n_threshold)
        shapes_given_shot[flag_value] = given_shot_fields["shape_given_shot"]

        categories[str(flag_value)] = {
            "n_rows": n_rows,
            "shape": shape,
            "mean_xg_range": round(max(values) - min(values), 6) if values else None,
            "small_n": is_small_n,
            "structural_caution": None,
            "bins": bins,
            **given_shot_fields,
        }

    diverges = shapes[True] != shapes[False]
    diverges_given_shot = shapes_given_shot[True] != shapes_given_shot[False]
    for cat in categories.values():
        cat["diverges_from_overall"] = diverges
        cat["diverges_from_overall_given_shot"] = diverges_given_shot

    return {
        "feature": feature,
        "slicer": slicer,
        "slicer_type": "boolean",
        "dataset": dataset_key,
        "overall_shape": overall_shape,
        "overall_mean_xg_range": round(max(overall_values) - min(overall_values), 6) if overall_values else None,
        "overall_shape_given_shot": overall_shape_given_shot,
        "n_categories": 2,
        "categories": categories,
        "n_categories_diverging": 2 if diverges else 0,
        "driven_by_single_category": None,
    }


def _slice_archetype_xg(feature: str, df: pd.DataFrame, flat_margin: float, flat_margin_given_shot: float, small_n_threshold: int) -> dict:
    """Same as V1's _slice_one (xg), then retrofit given-shot fields onto
    every category in one pass -- V1 needed a separate retrofit script
    (prompt 26) because it ran after V1 already existed; V2 has no such
    history to preserve, so this just does both in the same call."""
    cell = _slice_one(feature, ARCHETYPE_SLICER, "passive", df, flat_margin)
    cell["slicer_type"] = "categorical"

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD
    _, edges = _bin_table(df, feature, is_discrete_cardinality, None)

    shot_df_all = df.loc[df[SHOT_COL] == 1]
    overall_bins_given_shot, _ = _bin_table(shot_df_all, feature, is_discrete_cardinality, edges)
    overall_values_given_shot = [b["mean_xg"] for b in overall_bins_given_shot]
    overall_shape_given_shot = (
        classify_shape([b["bin"] for b in overall_bins_given_shot], overall_values_given_shot, flat_margin=flat_margin_given_shot)
        if overall_values_given_shot else "insufficient data"
    )
    cell["overall_shape_given_shot"] = overall_shape_given_shot

    for cat_label, cat in cell["categories"].items():
        cat_df = df.loc[df[ARCHETYPE_SLICER].astype(str) == cat_label]
        given_shot_fields = _given_shot_fields(cat_df, feature, is_discrete_cardinality, edges, flat_margin_given_shot, overall_shape_given_shot, cat["small_n"], small_n_threshold)
        cat.update(given_shot_fields)

    return cell


def main() -> None:
    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    passive_df_archetype = passive_df[passive_df[ARCHETYPE_SLICER].notna()]

    flat_margins = {
        "active": FLAT_MARGIN_RATIO * float(active_df[TARGET].mean()),
        "passive": FLAT_MARGIN_RATIO * float(passive_df[TARGET].mean()),
    }
    flat_margins_given_shot = {
        "active": FLAT_MARGIN_RATIO * float(active_df.loc[active_df[SHOT_COL] == 1, TARGET].mean()),
        "passive": FLAT_MARGIN_RATIO * float(passive_df.loc[passive_df[SHOT_COL] == 1, TARGET].mean()),
    }
    small_n_threshold = SMALL_N_THRESHOLD

    results: dict[str, dict] = {"active": {}, "passive": {}}
    n_cells = 0

    for feature in ACTIVE_FEATURES:
        results["active"][feature] = {}
        for slicer in ACTIVE_BOOLEAN_SLICERS:
            cell = _slice_boolean_xg(feature, slicer, "active", active_df, flat_margins["active"], flat_margins_given_shot["active"], small_n_threshold)
            results["active"][feature][slicer] = cell
            n_cells += 1
            print(f"[active] {feature} x {slicer}: overall={cell['overall_shape']}, given-shot overall={cell['overall_shape_given_shot']}, diverges={cell['n_categories_diverging'] > 0}")

    for feature in PASSIVE_FEATURES:
        results["passive"][feature] = {}
        archetype_cell = _slice_archetype_xg(feature, passive_df_archetype, flat_margins["passive"], flat_margins_given_shot["passive"], small_n_threshold)
        results["passive"][feature][ARCHETYPE_SLICER] = archetype_cell
        n_cells += 1
        print(f"[passive] {feature} x {ARCHETYPE_SLICER}: overall={archetype_cell['overall_shape']}, "
              f"{archetype_cell['n_categories_diverging']}/{archetype_cell['n_categories']} categories diverge")

        for slicer in PASSIVE_BOOLEAN_SLICERS:
            cell = _slice_boolean_xg(feature, slicer, "passive", passive_df, flat_margins["passive"], flat_margins_given_shot["passive"], small_n_threshold)
            results["passive"][feature][slicer] = cell
            n_cells += 1
            print(f"[passive] {feature} x {slicer}: overall={cell['overall_shape']}, given-shot overall={cell['overall_shape_given_shot']}, diverges={cell['n_categories_diverging'] > 0}")

    assert n_cells == 199, f"Expected exactly 199 cells, computed {n_cells}."

    by_feature = {}
    genuine_divergences = []
    genuine_divergences_given_shot = []
    for dataset, features in results.items():
        for feature, slicers in features.items():
            diverging, stable = [], []
            diverging_gs, stable_gs = [], []
            for slicer, cell in slicers.items():
                has_div = False
                has_div_gs = False
                for cat_label, cat in cell["categories"].items():
                    is_genuine = cat["diverges_from_overall"] and not cat["small_n"] and not cat.get("structural_caution")
                    is_genuine_gs = cat["diverges_from_overall_given_shot"] and cat["occurrence_vs_quality"] != "inconclusive" and not cat.get("structural_caution")
                    if is_genuine:
                        has_div = True
                        genuine_divergences.append({"dataset": dataset, "feature": feature, "slicer": slicer, "slicer_type": cell["slicer_type"], "category": cat_label, "category_shape": cat["shape"], "overall_shape": cell["overall_shape"]})
                    if is_genuine_gs:
                        has_div_gs = True
                        genuine_divergences_given_shot.append({"dataset": dataset, "feature": feature, "slicer": slicer, "slicer_type": cell["slicer_type"], "category": cat_label, "category_shape_given_shot": cat["shape_given_shot"], "overall_shape_given_shot": cell["overall_shape_given_shot"]})
                (diverging if has_div else stable).append(slicer)
                (diverging_gs if has_div_gs else stable_gs).append(slicer)
            by_feature[f"{dataset}/{feature}"] = {
                "dataset": dataset, "feature": feature,
                "n_slicers_tested": len(slicers),
                "diverging_slicers": sorted(diverging),
                "stable_slicers": sorted(stable),
                "needs_interaction_term_signal": bool(diverging),
                "diverging_slicers_given_shot": sorted(diverging_gs),
                "stable_slicers_given_shot": sorted(stable_gs),
                "needs_interaction_term_signal_given_shot": bool(diverging_gs),
            }
    genuine_divergences.sort(key=lambda r: (r["dataset"], r["feature"], r["slicer"], r["category"]))
    genuine_divergences_given_shot.sort(key=lambda r: (r["dataset"], r["feature"], r["slicer"], r["category"]))

    archetype_div = [d for d in genuine_divergences if d["slicer"] == ARCHETYPE_SLICER]
    boolean_div = [d for d in genuine_divergences if d["slicer"] != ARCHETYPE_SLICER]
    n_archetype_cells = sum(1 for f in results["passive"].values() for s in f if s == ARCHETYPE_SLICER)
    n_boolean_cells = 199 - n_archetype_cells
    closing_note = (
        f"Unconditionally, defender_archetype_name produced {len(archetype_div)} genuine divergence(s) across "
        f"{n_archetype_cells} cells ({len(archetype_div) / n_archetype_cells:.2f}/cell) vs the 15 boolean slicers "
        f"combined at {len(boolean_div)} across {n_boolean_cells} cells ({len(boolean_div) / n_boolean_cells:.2f}/cell). "
        + (
            "The archetype slicer matches or exceeds the combined boolean slicers' per-cell divergence rate for xG too."
            if (len(archetype_div) / n_archetype_cells) >= (len(boolean_div) / n_boolean_cells) else
            "The combined boolean slicers still out-produce the archetype slicer per cell for xG, same conclusion as binary."
        )
    )

    output = {
        "target": TARGET,
        "small_n_threshold": small_n_threshold,
        "flat_margin_ratio": FLAT_MARGIN_RATIO,
        "flat_margins_by_dataset": {k: round(v, 6) for k, v in flat_margins.items()},
        "flat_margins_given_shot_by_dataset": {k: round(v, 6) for k, v in flat_margins_given_shot.items()},
        "v1_cross_reference": (
            "See reports/analysis/xg_target/SLICE_STRATIFICATION.json (V1, prompts 25/26) for the original 118-cell grid over "
            "locked categorical slicers. V1 is untouched by this file -- V2 tests a disjoint slicer set "
            "(defender_archetype_name + every locked boolean flag)."
        ),
        "methodology": (
            "Continuous-target (xG) counterpart to SLICE_STRATIFICATION_V2.json (binary), same slicer grid. Same "
            "boolean-slicer scoping difference as the binary V2 file: diverges_from_overall (and "
            "diverges_from_overall_given_shot) for a boolean slicer means True's shape differs from False's shape, "
            "not vs a pooled overall. Shot-conditional fields (n_given_shot, bins_given_shot, shape_given_shot, "
            "diverges_from_overall_given_shot, occurrence_vs_quality) are computed in this same run, not a later "
            "retrofit -- flat_margin for the given-shot classification is scaled to each dataset's own "
            "shot-conditional mean xG (flat_margins_given_shot_by_dataset), same convention as V1's retrofit."
        ),
        "cross_dataset_summary": {
            "by_feature": by_feature,
            "n_features_with_at_least_one_diverging_slicer": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal"]),
            "n_features_fully_stable": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal"]),
            "genuine_divergences": genuine_divergences,
            "n_genuine_divergences": len(genuine_divergences),
            "n_features_with_at_least_one_diverging_slicer_given_shot": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal_given_shot"]),
            "n_features_fully_stable_given_shot": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal_given_shot"]),
            "genuine_divergences_given_shot": genuine_divergences_given_shot,
            "n_genuine_divergences_given_shot": len(genuine_divergences_given_shot),
            "closing_note_archetype_vs_boolean": closing_note,
        },
        "results": results,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n{n_cells} cells computed. Unconditional genuine divergences: {len(genuine_divergences)}, given-shot: {len(genuine_divergences_given_shot)}")
    print(closing_note)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
