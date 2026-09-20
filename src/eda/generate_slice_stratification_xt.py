"""CLI entrypoint: SLICE_STRATIFICATION V1 (xT delta), ACTIVE features only
-- the xT sibling of generate_slice_stratification_xg.py +
generate_slice_stratification_xg_expand.py (the confirmed two-script
generator of the xg portal's SLICE_STRATIFICATION.json).

Those two run in sequence in the xg portal for historical reasons (the
14-cell V1 grid came from prompt 24, the 104-cell expansion from prompt 25,
and the shot-conditional fields were retrofitted onto both by prompt 26).
This portal has no such history to preserve, so the same cells are built in
ONE pass, with the conditional fields computed inline -- exactly the
simplification generate_slice_stratification_v2_xg.py's own docstring makes
for V2 ("V1 did this in two passes; V2 does it in one run since the method
is already established").

Carried over UNCHANGED: FEATURE_SLICER_PLAN's active entries,
build_plan()'s active cells, SMALL_N_THRESHOLD = 1000,
STRUCTURAL_CAUTION_CATEGORIES, classify_shape(), and the
divergence / small-n / driven-by-one-category logic.

ADAPTED: flat_margin (Adaptation 1), and the conditional panel becomes
"given a non-zero delta" (Adaptation 2), which also renames
`occurrence_vs_quality` -- that name encodes the xg-specific
occurrence/quality distinction and would be actively misleading here. The
xT question is instead "is this category's pattern carried by WHICH rows
moved at all, or by HOW FAR they moved", so the field is named
`zero_mass_vs_magnitude` with values `zero-mass-driven` /
`magnitude-driven` / `inconclusive`.

Usage:
    python -m src.eda.generate_slice_stratification_xt
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_common as xc
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, classify_shape
from src.eda.generate_numerical_xt_target_analysis import _bin_table
from src.eda.generate_slice_stratification import (
    FEATURE_SLICER_PLAN,
    SMALL_N_THRESHOLD,
    STRUCTURAL_CAUTION_CATEGORIES,
)
from src.eda.generate_slice_stratification_expand import build_plan

OUTPUT_PATH = xc.OUT_DIR / "SLICE_STRATIFICATION.json"
TARGET = xc.TARGET


def _conditional_fields(cat_df: pd.DataFrame, feature: str, is_disc: bool, edges,
                        flat_margin_nz: float, ref_shape_nz: str, already_small_n: bool) -> dict:
    nz = xc.nonzero_subset(cat_df)
    n_nz = len(nz)
    bins, _ = _bin_table(nz, feature, is_disc, edges)
    values = [b["mean_xt"] for b in bins]
    shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin_nz) if values else "insufficient data"

    too_small = already_small_n or n_nz < SMALL_N_THRESHOLD or shape == "insufficient data"
    if too_small:
        verdict = "inconclusive"
    elif shape == "flat":
        verdict = "zero-mass-driven"
    else:
        verdict = "magnitude-driven"

    return {
        "n_given_nonzero_delta": n_nz,
        "bins_given_nonzero_delta": bins,
        "shape_given_nonzero_delta": shape,
        "diverges_from_overall_given_nonzero_delta": shape != ref_shape_nz,
        "zero_mass_vs_magnitude": verdict,
    }


def slice_one_xt(feature: str, slicer: str, df: pd.DataFrame, flat_margin: float, flat_margin_nz: float) -> dict:
    n_unique = df[feature].dropna().nunique()
    is_disc = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    overall_bins, edges = _bin_table(df, feature, is_disc, None)
    overall_values = [b["mean_xt"] for b in overall_bins]
    overall_shape = classify_shape([b["bin"] for b in overall_bins], overall_values, flat_margin=flat_margin) if overall_values else "insufficient data"

    nz_all = xc.nonzero_subset(df)
    ob_nz, _ = _bin_table(nz_all, feature, is_disc, edges)
    ov_nz = [b["mean_xt"] for b in ob_nz]
    overall_shape_nz = classify_shape([b["bin"] for b in ob_nz], ov_nz, flat_margin=flat_margin_nz) if ov_nz else "insufficient data"

    categories: dict[str, dict] = {}
    matching, non_small = [], []
    for cat_value, cat_df in df.groupby(slicer, dropna=False, observed=True):
        label = str(cat_value)
        n_rows = len(cat_df)
        bins, _ = _bin_table(cat_df, feature, is_disc, edges)
        values = [b["mean_xt"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin) if values else "insufficient data"

        is_small_n = n_rows < SMALL_N_THRESHOLD
        caution = STRUCTURAL_CAUTION_CATEGORIES.get(("active", slicer, label))
        diverges = shape != overall_shape

        categories[label] = {
            "n_rows": n_rows,
            "shape": shape,
            "mean_xt_range": round(max(values) - min(values), 8) if values else None,
            "bin_mean_min": round(min(values), 8) if values else None,
            "bin_mean_max": round(max(values), 8) if values else None,
            "curve_crosses_zero": bool(values and min(values) < 0 < max(values)),
            "diverges_from_overall": diverges,
            "small_n": is_small_n,
            "structural_caution": caution,
            "bins": bins,
            **_conditional_fields(cat_df, feature, is_disc, edges, flat_margin_nz, overall_shape_nz, is_small_n),
        }

        if not is_small_n and caution is None:
            non_small.append(label)
            if not diverges:
                matching.append(label)

    driven = len(matching) == 1 and len(non_small) > 1

    return {
        "feature": feature,
        "slicer": slicer,
        "dataset": "active",
        "overall_shape": overall_shape,
        "overall_shape_given_nonzero_delta": overall_shape_nz,
        "overall_mean_xt_range": round(max(overall_values) - min(overall_values), 8) if overall_values else None,
        "n_categories": len(categories),
        "categories": categories,
        "n_categories_diverging": sum(
            1 for c in categories.values()
            if c["diverges_from_overall"] and not c["small_n"] and not c["structural_caution"]
        ),
        "driven_by_single_category": matching[0] if driven else None,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    fm, fm_nz = scale["flat_margin"], scale["flat_margin_nonzero"]

    # Cells: the active entries of prompt 24's FEATURE_SLICER_PLAN, plus the
    # active entries of prompt 25's build_plan() -- both imported, not
    # re-derived. Passive cells from both are dropped (out of scope).
    cells: list[tuple[str, str]] = []
    for ds, feature, slicers in FEATURE_SLICER_PLAN:
        if ds != "active":
            continue
        cells.extend((feature, s) for s in slicers)
    n_from_plan = len(cells)
    for ds, feature, slicer in build_plan():
        if ds != "active":
            continue
        if (feature, slicer) not in cells:
            cells.append((feature, slicer))
    n_from_expand = len(cells) - n_from_plan

    results: dict[str, dict] = {}
    for feature, slicer in cells:
        cell = slice_one_xt(feature, slicer, df, fm, fm_nz)
        results.setdefault(feature, {})[slicer] = cell
        flag = f" -- DRIVEN BY {cell['driven_by_single_category']}" if cell["driven_by_single_category"] else ""
        print(f"active/{feature} x {slicer}: overall={cell['overall_shape']}, "
              f"{cell['n_categories_diverging']}/{cell['n_categories']} diverge{flag}")

    # --- Cross-cell summary, same structure as the xg expand script ---
    by_feature, genuine, genuine_nz, agreement = {}, [], [], []
    for feature, slicers in results.items():
        div, stable, div_nz, stable_nz = [], [], [], []
        for slicer, cell in slicers.items():
            has_div = has_div_nz = False
            for label, cat in cell["categories"].items():
                is_g = cat["diverges_from_overall"] and not cat["small_n"] and not cat["structural_caution"]
                is_g_nz = (cat["diverges_from_overall_given_nonzero_delta"]
                           and cat["zero_mass_vs_magnitude"] != "inconclusive"
                           and not cat["structural_caution"])
                if is_g:
                    has_div = True
                    genuine.append({"feature": feature, "slicer": slicer, "category": label,
                                    "category_shape": cat["shape"], "overall_shape": cell["overall_shape"]})
                if is_g_nz:
                    has_div_nz = True
                    genuine_nz.append({"feature": feature, "slicer": slicer, "category": label,
                                       "category_shape_given_nonzero_delta": cat["shape_given_nonzero_delta"],
                                       "overall_shape_given_nonzero_delta": cell["overall_shape_given_nonzero_delta"]})
                if not cat["structural_caution"] and cat["zero_mass_vs_magnitude"] != "inconclusive":
                    agrees = is_g == is_g_nz
                    agreement.append({
                        "feature": feature, "slicer": slicer, "category": label,
                        "diverges_unconditional": is_g, "diverges_given_nonzero_delta": is_g_nz, "agrees": agrees,
                        "note": "same conclusion either way" if agrees else (
                            "stable unconditionally, but a real magnitude-level divergence appears once the zero mass is removed"
                            if is_g_nz and not is_g else
                            "diverges unconditionally, but that divergence is carried by the zero mass -- flat once it is removed"
                        ),
                    })
            (div if has_div else stable).append(slicer)
            (div_nz if has_div_nz else stable_nz).append(slicer)

        by_feature[f"active/{feature}"] = {
            "dataset": "active", "feature": feature, "n_slicers_tested": len(slicers),
            "diverging_slicers": sorted(div), "stable_slicers": sorted(stable),
            "needs_interaction_term_signal": bool(div),
            "diverging_slicers_given_nonzero_delta": sorted(div_nz),
            "stable_slicers_given_nonzero_delta": sorted(stable_nz),
            "needs_interaction_term_signal_given_nonzero_delta": bool(div_nz),
        }

    genuine.sort(key=lambda r: (r["feature"], r["slicer"], r["category"]))
    genuine_nz.sort(key=lambda r: (r["feature"], r["slicer"], r["category"]))

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": TARGET,
        "target_scale": scale,
        "small_n_threshold": SMALL_N_THRESHOLD,
        "flat_margin": fm,
        "flat_margin_given_nonzero_delta": fm_nz,
        "n_cells": len(cells),
        "cell_provenance": (
            f"{n_from_plan} cell(s) from generate_slice_stratification.FEATURE_SLICER_PLAN's ACTIVE entries "
            f"(prompt 24), plus {n_from_expand} from generate_slice_stratification_expand.build_plan()'s ACTIVE "
            "entries (prompt 25). Both lists are imported, not re-derived. Passive cells from both are omitted "
            "-- the passive leg is out of scope for this target, a scope reduction rather than a method change. "
            "The xg portal builds the same cells across two scripts plus a third retrofit pass for historical "
            "reasons; this portal has no such history, so they are built in one pass with the conditional "
            "fields computed inline."
        ),
        "methodology": (
            "xT-delta counterpart to SLICE_STRATIFICATION.json -- same cells, same SMALL_N_THRESHOLD=1000, same "
            "STRUCTURAL_CAUTION_CATEGORIES, same classify_shape(), same divergence / small-n / "
            "driven-by-one-category logic, all imported unchanged. flat_margin is translated per Adaptation 1. "
            "The conditional panel is 'given a non-zero delta' rather than 'given a shot' (Adaptation 2), and "
            "the xg version's `occurrence_vs_quality` field is RENAMED to `zero_mass_vs_magnitude` with values "
            "zero-mass-driven / magnitude-driven / inconclusive -- 'occurrence vs quality' encodes an "
            "xg-specific distinction that does not exist for this target and reusing the name would have been "
            "actively misleading. Each category also records whether its binned curve crosses zero, which the "
            "xg version cannot ask (its binned means are non-negative by construction)."
        ),
        "prompt_64_cross_references": {"distribution_shape": xc.PROMPT_64_SHAPE_FINDING},
        "cross_dataset_summary": {
            "by_feature": by_feature,
            "n_features_with_at_least_one_diverging_slicer": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal"]),
            "n_features_fully_stable": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal"]),
            "genuine_divergences": genuine,
            "n_genuine_divergences": len(genuine),
            "genuine_divergences_given_nonzero_delta": genuine_nz,
            "n_genuine_divergences_given_nonzero_delta": len(genuine_nz),
            "conditioning_agreement": agreement,
            "n_conditioning_disagreements": sum(1 for r in agreement if not r["agrees"]),
        },
        "results": {"active": results},
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    s = output["cross_dataset_summary"]
    print(f"\n{len(cells)} cells. Unconditional genuine divergences: {s['n_genuine_divergences']}, "
          f"given non-zero delta: {s['n_genuine_divergences_given_nonzero_delta']}, "
          f"conditioning flips: {s['n_conditioning_disagreements']}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
