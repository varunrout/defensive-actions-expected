"""CLI entrypoint: SLICE_STRATIFICATION V1 + V2 against
`target_xt_delta_passive` -- PASSIVE features only. Computes both JSONs and
renders both HTML reports in one pass.

STEP-0 DISPOSITION: genuine passive build, for BOTH versions.
`reports/analysis/xg_target/SLICE_STRATIFICATION.json` and
`..._V2.json` each cover BOTH legs inside one file -- confirmed by reading
them: both carry `results.active` AND `results.passive`, and a
`flat_margins_by_dataset` block with an entry per leg. This portal's
existing files of those names are Prompt 67's ACTIVE-ONLY output, because
Prompt 65 scoped the suite to one leg and filtered the imported plans down
to their active entries. Merging a passive section into those files would
mean editing active-leg report content, which this pass does not do, so the
passive halves are separate `PASSIVE_*` files.

**V2 is where the two legs genuinely differ in what they can report, and it
differs in the passive leg's FAVOUR.** The active V2 records its closing
archetype-vs-boolean comparison as NOT APPLICABLE, because
`defender_archetype_name` is a passive-only column. Here it IS applicable,
so the comparison the active half could not make is made -- and it is
labelled as this leg's own content rather than presented as something the
portal has always had.

Carried over UNCHANGED, all imported: `FEATURE_SLICER_PLAN`'s passive
entries, `build_plan()`'s passive entries, `PASSIVE_FEATURES`,
`PASSIVE_BOOLEAN_SLICERS`, `ARCHETYPE_SLICER`, `SMALL_N_THRESHOLD = 1000`
(a ROW COUNT, so scale-free and untouchable by this leg's tails),
`STRUCTURAL_CAUTION_CATEGORIES`'s passive entry, `classify_shape()`, and
the divergence / small-n / driven-by-one-category logic.

ADAPTED: `flat_margin` and `flat_margin_given_nonzero_delta`, recomputed
from THIS leg's own standard deviations (Adaptation 1); the conditional
panel is "given a non-zero delta" (Adaptation 2); the xg version's
`occurrence_vs_quality` field stays renamed to `zero_mass_vs_magnitude`
(Prompt 65's rename, carried over because the reasoning is the same).

Usage:
    python -m src.eda.generate_slice_stratification_xt_passive
"""

from __future__ import annotations

import json

from src.eda import xt_passive_common as pc  # re-points xt_common; MUST precede the generator imports
from src.eda import xt_common as xc
from src.eda import generate_slice_stratification_xt as s1
from src.eda import generate_slice_stratification_v2_xt as s2
from src.eda import generate_slice_stratification_report_xt as sr1
from src.eda.generate_slice_stratification import (
    FEATURE_SLICER_PLAN,
    SMALL_N_THRESHOLD,
    STRUCTURAL_CAUTION_CATEGORIES,
)
from src.eda.generate_slice_stratification_expand import build_plan
from src.eda.generate_slice_stratification_v2 import (
    ARCHETYPE_SLICER,
    PASSIVE_BOOLEAN_SLICERS,
    PASSIVE_FEATURES,
)

V1_JSON = xc.OUT_DIR / "PASSIVE_SLICE_STRATIFICATION.json"
V1_HTML = xc.OUT_DIR / "PASSIVE_SLICE_STRATIFICATION.html"
V2_JSON = xc.OUT_DIR / "PASSIVE_SLICE_STRATIFICATION_V2.json"
V2_HTML = xc.OUT_DIR / "PASSIVE_SLICE_STRATIFICATION_V2.html"
TARGET = xc.TARGET

# `slice_one_xt` looks the caution table up with a hardcoded "active" dataset
# key -- the one active-specific line in an otherwise dataset-agnostic
# function. Rebound here by declared substitution so the PASSIVE caution
# entries are the ones that apply, rather than duplicating the function.
# The live table is asserted to still carry the entry this depends on.
_PASSIVE_CAUTIONS = {
    ("active", slicer, label): reason
    for (ds, slicer, label), reason in STRUCTURAL_CAUTION_CATEGORIES.items()
    if ds == "passive"
}
assert ("active", "defender_functional_role", "unclassified") in _PASSIVE_CAUTIONS, (
    "generate_slice_stratification.STRUCTURAL_CAUTION_CATEGORIES no longer carries the passive "
    "('defender_functional_role', 'unclassified') caution. The passive slice reports depend on it being "
    "applied, so it must not be silently lost."
)
s1.STRUCTURAL_CAUTION_CATEGORIES = _PASSIVE_CAUTIONS


def _retarget(cell: dict) -> dict:
    """`slice_one_xt` / `slice_boolean_xt` stamp `"dataset": "active"`.
    Corrected here rather than by editing those functions."""
    cell["dataset"] = "passive"
    return cell


def _rollup(results: dict, *, with_agreement: bool, slicer_type_key: bool) -> dict:
    """The cross-cell summary, structurally identical to the one both active
    slice scripts build -- reproduced here rather than imported because both
    build it inline inside their own `main()`."""
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
                extra = {"slicer_type": cell.get("slicer_type", "categorical")} if slicer_type_key else {}
                if is_g:
                    has_div = True
                    genuine.append({"feature": feature, "slicer": slicer, **extra, "category": label,
                                    "category_shape": cat["shape"], "overall_shape": cell["overall_shape"]})
                if is_g_nz:
                    has_div_nz = True
                    genuine_nz.append({
                        "feature": feature, "slicer": slicer, **extra, "category": label,
                        "category_shape_given_nonzero_delta": cat["shape_given_nonzero_delta"],
                        "overall_shape_given_nonzero_delta": cell["overall_shape_given_nonzero_delta"]})
                if with_agreement and not cat["structural_caution"] and cat["zero_mass_vs_magnitude"] != "inconclusive":
                    agrees = is_g == is_g_nz
                    agreement.append({
                        "feature": feature, "slicer": slicer, "category": label,
                        "diverges_unconditional": is_g, "diverges_given_nonzero_delta": is_g_nz,
                        "agrees": agrees,
                        "note": "same conclusion either way" if agrees else (
                            "stable unconditionally, but a real magnitude-level divergence appears once the "
                            "zero mass is removed" if is_g_nz and not is_g else
                            "diverges unconditionally, but that divergence is carried by the zero mass -- flat "
                            "once it is removed"
                        ),
                    })
            (div if has_div else stable).append(slicer)
            (div_nz if has_div_nz else stable_nz).append(slicer)

        by_feature[f"passive/{feature}"] = {
            "dataset": "passive", "feature": feature, "n_slicers_tested": len(slicers),
            "diverging_slicers": sorted(div), "stable_slicers": sorted(stable),
            "needs_interaction_term_signal": bool(div),
            "diverging_slicers_given_nonzero_delta": sorted(div_nz),
            "stable_slicers_given_nonzero_delta": sorted(stable_nz),
            "needs_interaction_term_signal_given_nonzero_delta": bool(div_nz),
        }

    genuine.sort(key=lambda r: (r["feature"], r["slicer"], r["category"]))
    genuine_nz.sort(key=lambda r: (r["feature"], r["slicer"], r["category"]))

    summary = {
        "by_feature": by_feature,
        "n_features_with_at_least_one_diverging_slicer": sum(
            1 for v in by_feature.values() if v["needs_interaction_term_signal"]),
        "n_features_fully_stable": sum(
            1 for v in by_feature.values() if not v["needs_interaction_term_signal"]),
        "genuine_divergences": genuine,
        "n_genuine_divergences": len(genuine),
        "genuine_divergences_given_nonzero_delta": genuine_nz,
        "n_genuine_divergences_given_nonzero_delta": len(genuine_nz),
        "n_features_with_at_least_one_diverging_slicer_given_nonzero_delta": sum(
            1 for v in by_feature.values() if v["needs_interaction_term_signal_given_nonzero_delta"]),
        "n_features_fully_stable_given_nonzero_delta": sum(
            1 for v in by_feature.values() if not v["needs_interaction_term_signal_given_nonzero_delta"]),
    }
    if with_agreement:
        summary["conditioning_agreement"] = agreement
        summary["n_conditioning_disagreements"] = sum(1 for r in agreement if not r["agrees"])
    return summary


def _render(data: dict, version: str, out_path) -> None:
    """The shared renderer reads `data["results"]["active"]`. A shallow copy
    with the key remapped is passed to it; the SAVED JSON keeps the honest
    `results.passive` key. Declared here rather than storing passive cells
    under an "active" key on disk, which would be a lie in the artefact."""
    view = {**data, "results": {"active": data["results"]["passive"]}}
    html = sr1.build_report(view, version=version)
    subs = [
        (f"SLICE STRATIFICATION {version} -- XT DELTA &amp;middot; ACTIVE-BINARY LEG ONLY",
         f"SLICE STRATIFICATION {version} -- XT DELTA &amp;middot; PASSIVE LEG"),
    ]
    if version == "V1":
        subs += [
            ("Categorical Stratification of Established Patterns (xT delta)",
             "Categorical Stratification of Established Patterns (xT delta, passive)"),
            ("Every ACTIVE locked numerical feature crossed with every locked categorical slicer, against mean "
             "xT delta. Each category carries an unconditional curve and one restricted to rows with a non-zero "
             "delta. Active-binary leg only.",
             "Every PASSIVE locked numerical feature crossed with every locked categorical slicer, against mean "
             "xT delta. Each category carries an unconditional curve and one restricted to rows with a non-zero "
             "delta. One row per visible defender-slot per on-ball event; the target is one value per event."),
        ]
    else:
        subs += [
            ("Boolean-Flag Slicers (xT delta)", "Archetype + Boolean-Flag Slicers (xT delta, passive)"),
            ("V2's disjoint slicer set, active half: every ACTIVE locked numerical feature crossed with every "
             "locked boolean flag. The archetype slicer the passive half uses does not exist in the active "
             "parquet, so neither does its closing comparison. Active-binary leg only.",
             "V2's disjoint slicer set, passive half: every PASSIVE locked numerical feature crossed with the "
             "defender_archetype_name slicer AND every locked boolean flag. The archetype slicer exists only "
             "on this leg, so the closing archetype-vs-boolean comparison the active half had to report as not "
             "applicable is made here."),
            ("Not applicable here, said so rather than substituted.",
             "Applicable here, unlike the active half."),
        ]
    for needle, replacement in subs:
        assert needle in html, (
            f"Expected {needle!r} in generate_slice_stratification_report_xt's {version} output so it could be "
            "relabelled for the passive leg. It is not there -- that module must have changed."
        )
        html = html.replace(needle, replacement)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:,.0f} KB)")


def _common(scale: dict, n_rows: int, n_events: int) -> dict:
    return {
        "dataset": "passive",
        "leg": "passive (this portal also covers the active-binary leg)",
        "target": TARGET,
        "target_scale": scale,
        "n_rows": n_rows,
        "n_unique_events": n_events,
        "small_n_threshold": SMALL_N_THRESHOLD,
        "flat_margin": scale["flat_margin"],
        "flat_margin_given_nonzero_delta": scale["flat_margin_nonzero"],
        "threshold_recomputation": {
            "std_derived_flat_margin": scale["flat_margin"],
            "std_derived_flat_margin_given_nonzero_delta": scale["flat_margin_nonzero"],
            "mad_derived_robust_flat_margin": scale["robust_scale"]["robust_flat_margin"],
            "active_v2_std_derived_flat_margin": 0.004929,
            "pct_rows_inside_std_derived_margin": scale["robust_scale"]["pct_rows_inside_flat_margin"],
            "note": (
                "flat_margin is RECOMPUTED from this leg's own standard deviation, not transferred from the "
                "active leg's 0.004929. That is the single threshold in this report that is not scale-free, "
                "and it matters in a specific direction: every verdict here is a SHAPE comparison, so a margin "
                "that is too wide makes categories look 'flat' and agree with each other, understating "
                "divergence. Prompt 67 found the std-derived margin already spanning 53.4% of rows on the "
                f"active leg; on this leg it spans "
                f"{scale['robust_scale']['pct_rows_inside_flat_margin']:.1f}%, so the effect is stronger here. "
                "The literal translated margin is KEPT as the headline so the two halves of this portal stay "
                "comparable, and the MAD-derived robust alternative is published beside it. SMALL_N_THRESHOLD "
                "(1000) is a ROW COUNT and is carried over completely unchanged -- re-checked, not assumed: a "
                "count cannot be distorted by tail weight."
            ),
        },
        "row_grain_note": pc.PROMPT_68_SHARED_TARGET_FINDING,
        "prompt_68_cross_references": {
            "distribution_shape": pc.PROMPT_68_SHAPE_FINDING,
            "shared_target_row_grain": pc.PROMPT_68_SHARED_TARGET_FINDING,
        },
    }


def run_v1(df, scale, n_rows, n_events) -> None:
    fm, fm_nz = scale["flat_margin"], scale["flat_margin_nonzero"]

    cells: list[tuple[str, str]] = []
    for ds, feature, slicers in FEATURE_SLICER_PLAN:
        if ds != "passive":
            continue
        cells.extend((feature, s) for s in slicers)
    n_from_plan = len(cells)
    for ds, feature, slicer in build_plan():
        if ds != "passive":
            continue
        if (feature, slicer) not in cells:
            cells.append((feature, slicer))
    n_from_expand = len(cells) - n_from_plan

    results: dict[str, dict] = {}
    for feature, slicer in cells:
        cell = _retarget(s1.slice_one_xt(feature, slicer, df, fm, fm_nz))
        results.setdefault(feature, {})[slicer] = cell
        flag = f" -- DRIVEN BY {cell['driven_by_single_category']}" if cell["driven_by_single_category"] else ""
        print(f"passive/{feature} x {slicer}: overall={cell['overall_shape']}, "
              f"{cell['n_categories_diverging']}/{cell['n_categories']} diverge{flag}")

    output = {
        **_common(scale, n_rows, n_events),
        "n_cells": len(cells),
        "cell_provenance": (
            f"{n_from_plan} cell(s) from generate_slice_stratification.FEATURE_SLICER_PLAN's PASSIVE entries "
            f"(prompt 24), plus {n_from_expand} from generate_slice_stratification_expand.build_plan()'s "
            "PASSIVE entries (prompt 25). Both lists are imported, not re-derived, and the active cells from "
            "both are omitted -- they are covered by this portal's own active half, not missing. The xg portal "
            "builds the same cells across two scripts plus a third retrofit pass for historical reasons; this "
            "pass has no such history, so they are built in one go with the conditional fields inline, exactly "
            "as Prompt 65 did for the active leg."
        ),
        "methodology": (
            "xT-delta counterpart to SLICE_STRATIFICATION.json's PASSIVE half -- same cells, same "
            "SMALL_N_THRESHOLD=1000, same STRUCTURAL_CAUTION_CATEGORIES (its passive entry, "
            "defender_functional_role='unclassified', applies here and is the reason that category is excluded "
            "from every divergence count below), same classify_shape(), same divergence / small-n / "
            "driven-by-one-category logic, all imported unchanged. flat_margin is translated per Adaptation 1 "
            "from THIS leg's own std. The conditional panel is 'given a non-zero delta' rather than 'given a "
            "shot' (Adaptation 2), and the xg version's `occurrence_vs_quality` field stays RENAMED to "
            "`zero_mass_vs_magnitude` -- Prompt 65's rename, carried over because the reasoning is unchanged: "
            "'occurrence vs quality' encodes an xg-specific distinction this target does not have. Each "
            "category also records whether its binned curve crosses zero, which the xg version cannot ask."
        ),
        "cross_dataset_summary": _rollup(results, with_agreement=True, slicer_type_key=False),
        "results": {"passive": results},
    }
    V1_JSON.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    s = output["cross_dataset_summary"]
    print(f"\nV1: {len(cells)} cells. Unconditional genuine divergences: {s['n_genuine_divergences']}, "
          f"given non-zero delta: {s['n_genuine_divergences_given_nonzero_delta']}, "
          f"conditioning flips: {s['n_conditioning_disagreements']}")
    _render(output, "V1", V1_HTML)


def run_v2(df, scale, n_rows, n_events) -> None:
    fm, fm_nz = scale["flat_margin"], scale["flat_margin_nonzero"]
    df_arch = df[df[ARCHETYPE_SLICER].notna()]

    results: dict[str, dict] = {}
    n_cells = 0
    for feature in PASSIVE_FEATURES:
        results[feature] = {}
        arch_cell = _retarget(s1.slice_one_xt(feature, ARCHETYPE_SLICER, df_arch, fm, fm_nz))
        arch_cell["slicer_type"] = "categorical (archetype)"
        results[feature][ARCHETYPE_SLICER] = arch_cell
        n_cells += 1
        print(f"[passive] {feature} x {ARCHETYPE_SLICER}: overall={arch_cell['overall_shape']}, "
              f"{arch_cell['n_categories_diverging']}/{arch_cell['n_categories']} diverge")
        for slicer in PASSIVE_BOOLEAN_SLICERS:
            cell = _retarget(s2.slice_boolean_xt(feature, slicer, df, fm, fm_nz))
            results[feature][slicer] = cell
            n_cells += 1
            print(f"[passive] {feature} x {slicer}: overall={cell['overall_shape']}, "
                  f"diverges={cell['n_categories_diverging'] > 0}")

    expected = len(PASSIVE_FEATURES) * (1 + len(PASSIVE_BOOLEAN_SLICERS))
    assert n_cells == expected, f"Expected {expected} cells, computed {n_cells}."

    summary = _rollup(results, with_agreement=False, slicer_type_key=True)

    # The closing comparison the ACTIVE half had to report as not applicable.
    arch_div = [d for d in summary["genuine_divergences"] if d["slicer"] == ARCHETYPE_SLICER]
    bool_div = [d for d in summary["genuine_divergences"] if d["slicer"] != ARCHETYPE_SLICER]
    n_arch_cells = len(PASSIVE_FEATURES)
    n_bool_cells = n_cells - n_arch_cells
    summary["closing_note_archetype_vs_boolean"] = (
        f"APPLICABLE ON THIS LEG, unlike the active half. Unconditionally, {ARCHETYPE_SLICER} produced "
        f"{len(arch_div)} genuine divergence(s) across {n_arch_cells} cells "
        f"({len(arch_div) / n_arch_cells:.2f}/cell) versus the {len(PASSIVE_BOOLEAN_SLICERS)} boolean slicers' "
        f"{len(bool_div)} across {n_bool_cells} cells ({len(bool_div) / n_bool_cells:.2f}/cell). This is the "
        "comparison the ACTIVE half of this portal reports as NOT APPLICABLE, because "
        f"{ARCHETYPE_SLICER} is a passive-only column that does not exist in the active parquet. It is made "
        "here as this leg's own content, not presented as something the portal previously had. Read it as a "
        "descriptive rate, not a significance test: the archetype slicer has 8 non-null categories against "
        "each boolean slicer's 2, and more categories mechanically offer more chances to diverge, so a higher "
        "per-cell rate is expected before any tactical interpretation is reached for."
    )

    output = {
        **_common(scale, n_rows, n_events),
        "n_cells": n_cells,
        "archetype_slicer_not_applicable": (
            f"APPLICABLE HERE -- this key keeps the active half's name so the two files stay diffable, and its "
            f"VALUE says the opposite. `{ARCHETYPE_SLICER}` is a PASSIVE column with 8 non-null values, so both "
            "the archetype slicer and its closing archetype-vs-boolean comparison run on this leg. The active "
            "half of this portal reports both as not applicable, correctly, because the column does not exist "
            "in the active parquet. See `cross_dataset_summary.closing_note_archetype_vs_boolean`."
        ),
        "v1_cross_reference": (
            "See reports/analysis/xt_target/PASSIVE_SLICE_STRATIFICATION.json (V1) for the locked-categorical-"
            "slicer grid. V1 is untouched by this file -- V2 tests a disjoint slicer set (the archetype slicer "
            "plus every locked boolean flag)."
        ),
        "boolean_slicer_scoping_note": (
            "Same scoping difference the binary, xg and active xT V2 files state: for the BOOLEAN cells, with "
            "only 2 categories, diverges_from_overall (and its non-zero-delta counterpart) means True's shape "
            f"differs from False's shape, not 'vs a pooled overall'. The {ARCHETYPE_SLICER} cells do compare "
            "each category against a pooled overall, so the two slicer kinds' divergence flags are not "
            "measuring quite the same thing -- which is a second reason the closing comparison above is a "
            "descriptive rate rather than a test."
        ),
        "methodology": (
            "xT-delta counterpart to SLICE_STRATIFICATION_V2.json's PASSIVE half. PASSIVE_FEATURES, "
            "PASSIVE_BOOLEAN_SLICERS and ARCHETYPE_SLICER are imported unchanged, as are SMALL_N_THRESHOLD and "
            "classify_shape(). The archetype cells reuse generate_slice_stratification_xt.slice_one_xt (the "
            "categorical-slicer function) on the archetype-non-null subset, exactly as the binary and xg V2 "
            "files do; the boolean cells reuse generate_slice_stratification_v2_xt.slice_boolean_xt. "
            "flat_margin is translated per Adaptation 1 from THIS leg's own std; the conditional panel is "
            "'given a non-zero delta' (Adaptation 2) and its verdict field is named zero_mass_vs_magnitude "
            "rather than reusing the xg-specific occurrence_vs_quality name."
        ),
        "cross_dataset_summary": summary,
        "results": {"passive": results},
    }
    V2_JSON.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nV2: {n_cells} cells. Unconditional genuine divergences: {summary['n_genuine_divergences']}, "
          f"given non-zero delta: {summary['n_genuine_divergences_given_nonzero_delta']}")
    print(f"  archetype {len(arch_div)}/{n_arch_cells} cells vs boolean {len(bool_div)}/{n_bool_cells} cells")
    _render(output, "V2", V2_HTML)


def main() -> None:
    V1_JSON.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    scale = xc.target_scale(df)
    n_rows, n_events = len(df), int(df["event_id"].nunique())
    run_v1(df, scale, n_rows, n_events)
    run_v2(df, scale, n_rows, n_events)


if __name__ == "__main__":
    main()
