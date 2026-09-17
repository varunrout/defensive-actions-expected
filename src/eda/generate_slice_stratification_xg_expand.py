"""CLI entrypoint: prompt 26 -- xG counterpart to prompt 25's full feature x
slicer cross-product, plus the shot-conditional retrofit prompt 23 already
applied to the numerical-target atlases and confound tests but never to
slice stratification.

Part A: the exact same 104-cell (feature, slicer) grid as prompt 25
(generate_slice_stratification_expand.build_plan(), unchanged), computed
against target_future_xg_10s / reports/eda_xg/ instead of the binary target.

Same correction as prompt 25 (both prompts reuse the same premise, verified
once against the live feature_config.py in that script's docstring):
Pool A's `distance_to_defending_goal` is NOT the locked sibling of
zone_defensive_value -- it's ALSO dropped (Cluster 4, defender-position),
with `defender_x` as the actually-kept column. Reusing prompt 25's *actual*
implemented plan (defender_x) rather than its literal restated text keeps
this prompt's stated goal -- "direct comparability between the binary and
xg results" -- true; using a different substitute feature here would break
exactly that comparability.

Part B: every category in every one of the resulting 118 cells (14
pre-existing + 104 new) gets 5 new fields, computed on the
target_future_shot_10s == 1 subset with the SAME bin edges as the
unconditional curve:
  - n_given_shot
  - bins_given_shot (mean_xg_given_shot per bin)
  - shape_given_shot (classify_shape, flat_margin scaled to the
    shot-conditional mean xG, not the unconditional one -- the conditional
    scale is roughly 10x higher)
  - diverges_from_overall_given_shot (vs the cell's own conditional overall
    shape, not the unconditional one)
  - occurrence_vs_quality: "inconclusive" if the category was already
    small_n unconditionally, or n_given_shot < small_n_threshold; otherwise
    "occurrence-only" if shape_given_shot == "flat", else "occurrence+quality"

Usage:
    python -m src.eda.generate_slice_stratification_xg_expand
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eda.feature_config import ACTIVE, PASSIVE
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, classify_shape
from src.eda.generate_numerical_xg_target_analysis import FLAT_MARGIN_RATIO, SHOT_COL, TARGET, _bin_table
from src.eda.generate_slice_stratification_expand import build_plan
from src.eda.generate_slice_stratification_xg import _slice_one

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "reports" / "eda_xg" / "SLICE_STRATIFICATION.json"


def _given_shot_fields_for_category(
    cat_df: pd.DataFrame,
    feature: str,
    is_discrete_cardinality: bool,
    edges,
    flat_margin_given_shot: float,
    cell_overall_shape_given_shot: str,
    small_n_threshold: int,
    already_small_n_unconditional: bool,
) -> dict:
    shot_df = cat_df.loc[cat_df[SHOT_COL] == 1]
    n_given_shot = len(shot_df)

    bins_given_shot, _ = _bin_table(shot_df, feature, is_discrete_cardinality, edges)
    values = [b["mean_xg"] for b in bins_given_shot]
    shape_given_shot = (
        classify_shape([b["bin"] for b in bins_given_shot], values, flat_margin=flat_margin_given_shot)
        if values else "insufficient data"
    )
    diverges_given_shot = shape_given_shot != cell_overall_shape_given_shot

    too_small = already_small_n_unconditional or n_given_shot < small_n_threshold or shape_given_shot == "insufficient data"
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


def retrofit_cell(cell: dict, df: pd.DataFrame, flat_margin_given_shot: float, small_n_threshold: int) -> None:
    """Mutates `cell` in place, adding shot-conditional fields to every
    category -- does not touch any existing (unconditional) field."""
    feature = cell["feature"]
    slicer = cell["slicer"]

    n_unique = df[feature].dropna().nunique()
    is_discrete_cardinality = n_unique <= DISCRETE_CARDINALITY_THRESHOLD
    _, edges = _bin_table(df, feature, is_discrete_cardinality, None)

    # Cell-level conditional overall shape -- the reference every category's
    # conditional shape is compared against, same role overall_shape plays
    # for the unconditional side.
    shot_df_all = df.loc[df[SHOT_COL] == 1]
    overall_bins_given_shot, _ = _bin_table(shot_df_all, feature, is_discrete_cardinality, edges)
    overall_values_given_shot = [b["mean_xg"] for b in overall_bins_given_shot]
    overall_shape_given_shot = (
        classify_shape([b["bin"] for b in overall_bins_given_shot], overall_values_given_shot, flat_margin=flat_margin_given_shot)
        if overall_values_given_shot else "insufficient data"
    )
    cell["overall_shape_given_shot"] = overall_shape_given_shot

    for cat_label, cat in cell["categories"].items():
        # dropna(False) groupby in _slice_one stringifies NaN to "nan" --
        # match that here so the category subset lines up with the one
        # _slice_one already computed the unconditional fields from.
        if cat_label == "nan" and df[slicer].isna().any():
            cat_df = df[df[slicer].isna()]
        else:
            cat_df = df[df[slicer].astype(str) == cat_label]

        fields = _given_shot_fields_for_category(
            cat_df, feature, is_discrete_cardinality, edges,
            flat_margin_given_shot, overall_shape_given_shot, small_n_threshold,
            already_small_n_unconditional=cat["small_n"],
        )
        cat.update(fields)


def main() -> None:
    existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    n_existing_cells = sum(len(slicers) for feats in existing["results"].values() for slicers in feats.values())
    assert n_existing_cells == 14, (
        f"Expected exactly 14 pre-existing (feature, slicer) cells before extending, found {n_existing_cells} -- "
        "aborting rather than risk duplicating or clobbering entries."
    )

    plan = build_plan()  # same 104-cell plan as prompt 25 (defender_x, not distance_to_defending_goal -- see docstring)
    assert len(plan) == 104, f"Expected exactly 104 new (feature, slicer) cells, built {len(plan)} -- aborting."
    for dataset, feature, slicer in plan:
        if slicer in existing["results"].get(dataset, {}).get(feature, {}):
            raise AssertionError(f"Cell ({dataset}, {feature}, {slicer}) already exists -- aborting rather than duplicate.")

    active_df = pd.read_parquet(REPO_ROOT / ACTIVE["parquet_path"])
    passive_df = pd.read_parquet(REPO_ROOT / PASSIVE["parquet_path"])
    dfs = {"active": active_df, "passive": passive_df}
    flat_margins = {ds: FLAT_MARGIN_RATIO * float(df[TARGET].mean()) for ds, df in dfs.items()}
    flat_margins_given_shot = {ds: FLAT_MARGIN_RATIO * float(df.loc[df[SHOT_COL] == 1, TARGET].mean()) for ds, df in dfs.items()}
    small_n_threshold = existing["small_n_threshold"]

    # --- Part A: 104 new cells, same grid as prompt 25 ---
    n_added = 0
    for dataset, feature, slicer in plan:
        cell = _slice_one(feature, slicer, dataset, dfs[dataset], flat_margins[dataset])
        existing["results"].setdefault(dataset, {}).setdefault(feature, {})[slicer] = cell
        n_added += 1
        flag = f" -- DRIVEN BY {cell['driven_by_single_category']}" if cell["driven_by_single_category"] else ""
        print(f"[A] {dataset}/{feature} x {slicer}: overall={cell['overall_shape']}, "
              f"{cell['n_categories_diverging']}/{cell['n_categories']} categories diverge{flag}")
    assert n_added == 104, f"Expected to add exactly 104 cells in Part A, added {n_added}."

    # --- Part B: shot-conditional retrofit on all 118 cells (14 old + 104 new) ---
    n_retrofitted = 0
    for dataset, features in existing["results"].items():
        for feature, slicers in features.items():
            for slicer, cell in slicers.items():
                retrofit_cell(cell, dfs[dataset], flat_margins_given_shot[dataset], small_n_threshold)
                n_retrofitted += 1
    total_cells = n_existing_cells + n_added
    assert n_retrofitted == total_cells, f"Expected to retrofit all {total_cells} cells, retrofitted {n_retrofitted}."

    # --- Cross-dataset, cross-conditioning summary ---
    by_feature = {}
    genuine_divergences = []
    genuine_divergences_given_shot = []
    conditioning_agreement = []
    for dataset, features in existing["results"].items():
        for feature, slicers in features.items():
            diverging_slicers, stable_slicers = [], []
            diverging_slicers_given_shot, stable_slicers_given_shot = [], []
            for slicer, cell in slicers.items():
                has_div = False
                has_div_given_shot = False
                for cat_label, cat in cell["categories"].items():
                    is_genuine = cat["diverges_from_overall"] and not cat["small_n"] and not cat.get("structural_caution")
                    is_genuine_given_shot = (
                        cat["diverges_from_overall_given_shot"] and cat["occurrence_vs_quality"] != "inconclusive"
                        and not cat.get("structural_caution")
                    )
                    if is_genuine:
                        has_div = True
                        genuine_divergences.append({"dataset": dataset, "feature": feature, "slicer": slicer, "category": cat_label, "category_shape": cat["shape"], "overall_shape": cell["overall_shape"]})
                    if is_genuine_given_shot:
                        has_div_given_shot = True
                        genuine_divergences_given_shot.append({"dataset": dataset, "feature": feature, "slicer": slicer, "category": cat_label, "category_shape_given_shot": cat["shape_given_shot"], "overall_shape_given_shot": cell["overall_shape_given_shot"]})

                    if not cat.get("structural_caution") and cat["occurrence_vs_quality"] != "inconclusive":
                        agrees = is_genuine == is_genuine_given_shot
                        conditioning_agreement.append({
                            "dataset": dataset, "feature": feature, "slicer": slicer, "category": cat_label,
                            "diverges_unconditional": is_genuine, "diverges_given_shot": is_genuine_given_shot,
                            "agrees": agrees,
                            "note": "same conclusion conditional and unconditional" if agrees else (
                                "stable unconditionally but occurrence-only-driven divergence appears once conditioned on a shot"
                                if is_genuine_given_shot and not is_genuine else
                                "diverges unconditionally but that divergence is occurrence-only -- flat once conditioned on a shot"
                            ),
                        })

                (diverging_slicers if has_div else stable_slicers).append(slicer)
                (diverging_slicers_given_shot if has_div_given_shot else stable_slicers_given_shot).append(slicer)

            by_feature[f"{dataset}/{feature}"] = {
                "dataset": dataset, "feature": feature,
                "n_slicers_tested": len(slicers),
                "diverging_slicers": sorted(diverging_slicers),
                "stable_slicers": sorted(stable_slicers),
                "needs_interaction_term_signal": bool(diverging_slicers),
                "diverging_slicers_given_shot": sorted(diverging_slicers_given_shot),
                "stable_slicers_given_shot": sorted(stable_slicers_given_shot),
                "needs_interaction_term_signal_given_shot": bool(diverging_slicers_given_shot),
            }

    genuine_divergences.sort(key=lambda r: (r["dataset"], r["feature"], r["slicer"], r["category"]))
    genuine_divergences_given_shot.sort(key=lambda r: (r["dataset"], r["feature"], r["slicer"], r["category"]))
    n_disagreements = sum(1 for r in conditioning_agreement if not r["agrees"])

    existing["small_n_threshold"] = small_n_threshold
    existing["flat_margins_given_shot_by_dataset"] = {k: round(v, 6) for k, v in flat_margins_given_shot.items()}
    existing["prompt_26_extension_note"] = (
        "104 new (feature, slicer) cells added by prompt 26 Part A -- same grid as prompt 25 (binary target), "
        "computed against target_future_xg_10s instead, for direct comparability. Correction (same one prompt 25 "
        "made): Pool A uses `defender_x`, not `distance_to_defending_goal` -- the latter is also dropped in "
        "feature_config.py (Cluster 4, defender-position), defender_x is the actually-kept sibling. Part B "
        "retrofitted shot-conditional fields (n_given_shot, bins_given_shot, shape_given_shot, "
        "diverges_from_overall_given_shot, occurrence_vs_quality) onto every category of all 118 cells (14 "
        "pre-existing + 104 new) -- xg is structurally zero exactly where no shot occurs (STRUCTURAL_ZERO_CHECK.json), "
        "so the unconditional mean_xg per category mostly re-derives the occurrence pattern in different units; "
        "the shot-conditional curve is where genuinely new chance-quality information shows up, same principle "
        "prompt 23 established for the numerical-target atlases and confound tests."
    )
    existing["cross_dataset_summary"] = {
        "by_feature": by_feature,
        "n_features_with_at_least_one_diverging_slicer": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal"]),
        "n_features_fully_stable": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal"]),
        "genuine_divergences": genuine_divergences,
        "n_genuine_divergences": len(genuine_divergences),
        "n_features_with_at_least_one_diverging_slicer_given_shot": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal_given_shot"]),
        "n_features_fully_stable_given_shot": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal_given_shot"]),
        "genuine_divergences_given_shot": genuine_divergences_given_shot,
        "n_genuine_divergences_given_shot": len(genuine_divergences_given_shot),
        "conditioning_agreement": conditioning_agreement,
        "n_conditioning_disagreements": n_disagreements,
    }

    OUTPUT_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    summary = existing["cross_dataset_summary"]
    print(f"\n{n_added} new cells added, {n_retrofitted} cells retrofitted with shot-conditional fields ({total_cells} total).")
    print(f"Unconditional genuine divergences: {summary['n_genuine_divergences']}")
    print(f"Given-shot genuine divergences: {summary['n_genuine_divergences_given_shot']}")
    print(f"Categories where conditioning changes the divergence conclusion: {summary['n_conditioning_disagreements']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
