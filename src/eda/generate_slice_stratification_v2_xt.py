"""CLI entrypoint: SLICE_STRATIFICATION V2 (xT delta), ACTIVE only -- the
xT sibling of generate_slice_stratification_v2_xg.py (the confirmed
generator of the xg portal's SLICE_STRATIFICATION_V2.json).

V2's slicer set is disjoint from V1's: V1 slices by locked CATEGORICAL
columns, V2 by `defender_archetype_name` (passive-only) plus every locked
BOOLEAN flag. `defender_archetype_name` does not exist in the active
parquet, so the active half of V2 is the boolean-slicer half only --
ACTIVE_FEATURES x ACTIVE_BOOLEAN_SLICERS, both imported unchanged from
generate_slice_stratification_v2. The archetype-vs-boolean closing
comparison the xg V2 makes is therefore NOT reproducible here and is
reported as such rather than replaced with a different comparison.

Same boolean-slicer scoping note as both the binary and xg V2 files: with
only 2 categories, `diverges_from_overall` means "does True's shape differ
from False's shape", not "vs a pooled overall".

ADAPTED: flat_margin (Adaptation 1); the conditional panel and its verdict
field (Adaptation 2/3), named `zero_mass_vs_magnitude` for the same reason
as V1.

Usage:
    python -m src.eda.generate_slice_stratification_v2_xt
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_common as xc
from src.eda.generate_numerical_target_analysis import DISCRETE_CARDINALITY_THRESHOLD, classify_shape
from src.eda.generate_numerical_xt_target_analysis import _bin_table
from src.eda.generate_slice_stratification import SMALL_N_THRESHOLD
from src.eda.generate_slice_stratification_v2 import ACTIVE_BOOLEAN_SLICERS, ACTIVE_FEATURES, ARCHETYPE_SLICER
from src.eda.generate_slice_stratification_xt import _conditional_fields

OUTPUT_PATH = xc.OUT_DIR / "SLICE_STRATIFICATION_V2.json"
TARGET = xc.TARGET


def slice_boolean_xt(feature: str, slicer: str, df: pd.DataFrame, flat_margin: float, flat_margin_nz: float) -> dict:
    df_valid = df.loc[df[slicer].notna()]
    bool_col = df_valid[slicer].astype(bool)

    n_unique = df[feature].dropna().nunique()
    is_disc = n_unique <= DISCRETE_CARDINALITY_THRESHOLD

    overall_bins, edges = _bin_table(df, feature, is_disc, None)
    ov = [b["mean_xt"] for b in overall_bins]
    overall_shape = classify_shape([b["bin"] for b in overall_bins], ov, flat_margin=flat_margin) if ov else "insufficient data"

    nz_all = xc.nonzero_subset(df)
    ob_nz, _ = _bin_table(nz_all, feature, is_disc, edges)
    ov_nz = [b["mean_xt"] for b in ob_nz]
    overall_shape_nz = classify_shape([b["bin"] for b in ob_nz], ov_nz, flat_margin=flat_margin_nz) if ov_nz else "insufficient data"

    categories, shapes, shapes_nz = {}, {}, {}
    for flag in (True, False):
        cat_df = df_valid.loc[bool_col == flag]
        n_rows = len(cat_df)
        bins, _ = _bin_table(cat_df, feature, is_disc, edges)
        values = [b["mean_xt"] for b in bins]
        shape = classify_shape([b["bin"] for b in bins], values, flat_margin=flat_margin) if values else "insufficient data"
        shapes[flag] = shape
        is_small_n = n_rows < SMALL_N_THRESHOLD

        cond = _conditional_fields(cat_df, feature, is_disc, edges, flat_margin_nz, overall_shape_nz, is_small_n)
        shapes_nz[flag] = cond["shape_given_nonzero_delta"]

        categories[str(flag)] = {
            "n_rows": n_rows,
            "shape": shape,
            "mean_xt_range": round(max(values) - min(values), 8) if values else None,
            "bin_mean_min": round(min(values), 8) if values else None,
            "bin_mean_max": round(max(values), 8) if values else None,
            "curve_crosses_zero": bool(values and min(values) < 0 < max(values)),
            "small_n": is_small_n,
            "structural_caution": None,
            "bins": bins,
            **cond,
        }

    diverges = shapes[True] != shapes[False]
    diverges_nz = shapes_nz[True] != shapes_nz[False]
    for cat in categories.values():
        cat["diverges_from_overall"] = diverges
        cat["diverges_from_overall_given_nonzero_delta"] = diverges_nz

    return {
        "feature": feature,
        "slicer": slicer,
        "slicer_type": "boolean",
        "dataset": "active",
        "overall_shape": overall_shape,
        "overall_shape_given_nonzero_delta": overall_shape_nz,
        "overall_mean_xt_range": round(max(ov) - min(ov), 8) if ov else None,
        "n_categories": 2,
        "categories": categories,
        "n_categories_diverging": 2 if diverges else 0,
        "driven_by_single_category": None,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    fm, fm_nz = scale["flat_margin"], scale["flat_margin_nonzero"]

    results: dict[str, dict] = {}
    n_cells = 0
    for feature in ACTIVE_FEATURES:
        results[feature] = {}
        for slicer in ACTIVE_BOOLEAN_SLICERS:
            cell = slice_boolean_xt(feature, slicer, df, fm, fm_nz)
            results[feature][slicer] = cell
            n_cells += 1
            print(f"[active] {feature} x {slicer}: overall={cell['overall_shape']}, "
                  f"non-zero overall={cell['overall_shape_given_nonzero_delta']}, "
                  f"diverges={cell['n_categories_diverging'] > 0}")

    expected = len(ACTIVE_FEATURES) * len(ACTIVE_BOOLEAN_SLICERS)
    assert n_cells == expected, f"Expected {expected} cells, computed {n_cells}."

    by_feature, genuine, genuine_nz = {}, [], []
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
                    genuine.append({"feature": feature, "slicer": slicer, "slicer_type": "boolean",
                                    "category": label, "category_shape": cat["shape"],
                                    "overall_shape": cell["overall_shape"]})
                if is_g_nz:
                    has_div_nz = True
                    genuine_nz.append({"feature": feature, "slicer": slicer, "slicer_type": "boolean",
                                       "category": label,
                                       "category_shape_given_nonzero_delta": cat["shape_given_nonzero_delta"],
                                       "overall_shape_given_nonzero_delta": cell["overall_shape_given_nonzero_delta"]})
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
        "n_cells": n_cells,
        "archetype_slicer_not_applicable": (
            f"The xg and binary V2 files also slice by `{ARCHETYPE_SLICER}`, and close with a comparison of the "
            "archetype slicer's per-cell divergence rate against the boolean slicers' combined rate. "
            f"`{ARCHETYPE_SLICER}` is a PASSIVE column -- it does not exist in the active parquet -- so the "
            "active half of V2 is the boolean-slicer half only, and that closing comparison is NOT reproducible "
            "here. Reported as not applicable rather than replaced with a different comparison that would look "
            "like the same conclusion."
        ),
        "v1_cross_reference": (
            "See reports/analysis/xt_target/SLICE_STRATIFICATION.json (V1) for the locked-categorical-slicer "
            "grid. V1 is untouched by this file -- V2 tests a disjoint slicer set (every locked boolean flag)."
        ),
        "boolean_slicer_scoping_note": (
            "Same scoping difference the binary and xg V2 files state: with only 2 categories, "
            "diverges_from_overall (and its non-zero-delta counterpart) means True's shape differs from False's "
            "shape, not 'vs a pooled overall'."
        ),
        "methodology": (
            "xT-delta counterpart to SLICE_STRATIFICATION_V2.json, active half. ACTIVE_FEATURES and "
            "ACTIVE_BOOLEAN_SLICERS are imported unchanged, as are SMALL_N_THRESHOLD and classify_shape(). "
            "flat_margin is translated per Adaptation 1; the conditional panel is 'given a non-zero delta' "
            "(Adaptation 2) and its verdict field is named zero_mass_vs_magnitude rather than reusing the "
            "xg-specific occurrence_vs_quality name."
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
            "n_features_with_at_least_one_diverging_slicer_given_nonzero_delta": sum(1 for v in by_feature.values() if v["needs_interaction_term_signal_given_nonzero_delta"]),
            "n_features_fully_stable_given_nonzero_delta": sum(1 for v in by_feature.values() if not v["needs_interaction_term_signal_given_nonzero_delta"]),
            "closing_note_archetype_vs_boolean": "NOT APPLICABLE -- see archetype_slicer_not_applicable.",
        },
        "results": {"active": results},
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    s = output["cross_dataset_summary"]
    print(f"\n{n_cells} cells. Unconditional genuine divergences: {s['n_genuine_divergences']}, "
          f"given non-zero delta: {s['n_genuine_divergences_given_nonzero_delta']}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
