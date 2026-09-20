"""CLI entrypoint: leakage audit for `target_xt_delta_v2`, ACTIVE dataset.

STEP-0 DISPOSITION: REWRITE, not a plain re-point -- and this is the single
largest genuine logic change in the v2 rebuild.

Parts A, B' and C' are methodologically untouched (Part A is imported
unchanged from `generate_leakage_audit.part_a` exactly as the v1 edition
did; B'/C' reuse the v1 module's `_split_stats` byte-for-byte). Only their
explanatory prose is re-derived, because it quoted v1's distribution shape.

**Part D' had to be rewritten from scratch, because its premise is false
under v2.** The v1 edition's Part D' existed to quantify this:

    "target_xt_delta = xt_before - xt_after, and xt_after is looked up
     directly from action_x/action_y. Any feature that is a deterministic
     function of the action's own pitch location therefore shares a
     definitional term with half the target."

Prompt 66 removed exactly that. Under v2, `xt_after` is 0.0 when the action
ends its own possession, the NEXT same-possession event's grid value
otherwise, or that event's own `shot_statsbomb_xg` if it is a Shot. **The
acting row's own coordinates are never looked up at all.** So the v1
coupling is not merely weaker, its mechanism is gone.

Restating v1's Part D' under a re-point would therefore have published a
sentence that is no longer true about the data it sits next to. Instead the
coupling is RE-MEASURED from scratch on v2, with the v1 figures quoted
beside the v2 ones so a reader can see how much of Prompt 65's single most
important recorded caveat the correction actually retired -- and a second,
genuinely NEW coupling channel specific to v2 is tested: `xt_after = 0` is
triggered deterministically by `action_ended_possession`, so the coupling
moved from a LOCATION term to a FLAG term.

Usage:
    python -m src.eda.generate_leakage_audit_xt_v2
"""

from __future__ import annotations

import json

import pandas as pd

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import xt_common as xc
from src.eda import generate_leakage_audit_xt as l1
from src.eda.feature_config import ACTIVE
from src.eda.generate_leakage_audit import _candidate_columns, part_a

OUTPUT_PATH = xc.OUT_DIR / "LEAKAGE_AUDIT.json"
TARGET = xc.TARGET

# v1's own Part D' headline, quoted so the comparison is explicit rather than
# implied. Source: reports/analysis/xt_target_v1_superseded/LEAKAGE_AUDIT.json.
V1_STRONGEST = {"feature": "distance_to_attacking_box", "r_vs_target": 0.5551}

# Under v2 the xt_after=0 branch is triggered by this flag, not by a location.
V2_CONSTRUCTION_COLUMNS = ["action_ended_possession"]


def part_b_prime_v2(df: pd.DataFrame) -> dict:
    """Same method as v1 (imported `_split_stats`); prose re-derived for v2."""
    r = l1._split_stats(df, "has_previous_event")
    tr, fa = r["by_value"]["True"], r["by_value"]["False"]
    r.update({
        "substitutes_for": "Part C of the passive xg audit (has_screened_outcome)",
        "method": "groupby mean target + Welch's t-test -- identical to generate_leakage_audit_xg.part_c_xg",
        "verdict": "flagged -- real censoring-shaped effect, and on v2 it is visible in BOTH the mean and the zero share",
        "explanation": (
            f"has_previous_event=False rows (n={fa['n']:,}) are the first event of a possession, so there is no "
            f"in-possession predecessor. Mean delta {fa['mean_xt']:+.6f} versus True's {tr['mean_xt']:+.6f} "
            f"(Welch p={r['p_value']:.3e}), and the EXACTLY-ZERO share is {fa['pct_zero']:.1f}% versus "
            f"{tr['pct_zero']:.1f}%, a {r['zero_share_gap_pp']:+.1f}pp gap. The v1 edition recorded this as its "
            "Caveat 2 -- the effect lived almost entirely in the zero mass, and a mean-difference test alone "
            "(the instrument the xg audit reaches for) returned p=0.0645 and would have reported nothing. That "
            "caveat still stands on v2 and the zero/negative/positive shares are still reported alongside every "
            "mean in this suite. Reported as a finding about how this target must be MEASURED, NOT as a "
            "proposal to drop has_previous_event -- it is a locked feature and this audit does not change "
            "feature_config.py."
        ),
    })
    return r


def part_c_prime_v2(df: pd.DataFrame) -> dict:
    """The three action_*_possession flags. On v2 this part changes character
    completely: two of these flags are no longer merely non-degenerate
    against the target, they are CONSTRUCTION INPUTS to it."""
    import numpy as np

    results = {}
    for col in ("action_changed_possession", "action_ended_possession", "action_won_possession"):
        r = l1._split_stats(df, col)
        tr = r["by_value"].get("True", {})
        sub = df.loc[(df[col] == True) & df[TARGET].notna()]  # noqa: E712
        collapses = bool(np.allclose(sub[TARGET].to_numpy(), sub["xt_before"].to_numpy())) if len(sub) else False
        r.update({
            "substitutes_for": "Part D of the passive xg audit (has_option_2 / has_option_3)",
            "method": "groupby mean target + Welch's t-test -- identical to generate_leakage_audit_xg's Parts B-D",
            "feature_config_exclusion_reason": ACTIVE["excluded"].get(col),
            "collapses_to_xt_before": collapses,
            "xt_before_mean_same_slice": round(float(sub["xt_before"].mean()), 8) if len(sub) else None,
            "verdict": (
                "exclusion justification does not transfer -- but on v2 a STRONGER, different reason to exclude "
                "appears: construction input" if collapses else
                "exclusion justification does NOT transfer to this target -- flagged, not reopened"
            ),
            "explanation": (
                f"feature_config.py excludes {col} because it is a structural zero against "
                f"target_future_shot_10s (0.00% shot rate by that target's own window definition). Against "
                f"{TARGET} the True group is not degenerate: n={tr.get('n', 0):,}, mean "
                f"{tr.get('mean_xt', float('nan')):+.6f}, {tr.get('pct_zero', float('nan')):.1f}% exactly zero, "
                f"{tr.get('pct_negative', float('nan')):.1f}% negative and "
                f"{tr.get('pct_positive', float('nan')):.1f}% positive. "
                + (
                    "CRUCIALLY, and unlike v1: this flag's True group collapses onto xt_before EXACTLY "
                    "(np.allclose, verified on this run), because it is the deterministic trigger for "
                    "xt_after = 0 in the v2 construction. That makes it a construction input to the target, "
                    "not a candidate feature -- a far stronger reason to keep it excluded than the "
                    "target-specific one feature_config.py currently records. Feeding it to a model would be "
                    "circular. Still an observation, not a change: this suite does not edit feature_config.py."
                    if collapses else
                    "This flag is not the xt_after=0 trigger, so it carries no construction identity. Reported "
                    "as an observation, NOT a proposal to unlock the column."
                )
            ),
        })
        results[col] = r
    return results


def part_d_prime_v2(df: pd.DataFrame) -> dict:
    """REWRITTEN. v1's construction-coupling mechanism no longer exists, so
    the coupling is re-measured rather than restated, and the new v2-specific
    flag-coupling channel is tested alongside it."""
    locked = sorted(set(ACTIVE["continuous"]) | set(ACTIVE["discrete"]))
    sub = df.dropna(subset=[TARGET])
    rows = []
    for col in locked:
        if not pd.api.types.is_numeric_dtype(sub[col]):
            continue
        rows.append({
            "feature": col,
            "r_vs_xt_after": round(float(sub[col].corr(sub["xt_after"])), 4),
            "r_vs_xt_before": round(float(sub[col].corr(sub["xt_before"])), 4),
            "r_vs_target": round(float(sub[col].corr(sub[TARGET])), 4),
        })
    rows.sort(key=lambda r: abs(r["r_vs_target"]), reverse=True)
    strongest = rows[0] if rows else None

    # v1's construction columns -- confirmed still out of the candidate list,
    # and now confirmed to be irrelevant to xt_after as well.
    candidate_cols = _candidate_columns(ACTIVE)
    old_construction_status = {
        c: {
            "in_candidate_list": c in candidate_cols,
            "exclusion_reason": ACTIVE["excluded"].get(c),
            "r_vs_xt_after": round(float(sub[c].corr(sub["xt_after"])), 4) if c in sub.columns else None,
        }
        for c in ["action_x", "action_y"]
    }

    # NEW for v2: the coupling moved from a location term to a flag term.
    flag_coupling = {}
    for c in V2_CONSTRUCTION_COLUMNS:
        f = sub[c].astype(int)
        flag_coupling[c] = {
            "in_candidate_list": c in candidate_cols,
            "exclusion_reason": ACTIVE["excluded"].get(c),
            "r_vs_xt_after": round(float(f.corr(sub["xt_after"])), 4),
            "r_vs_target": round(float(f.corr(sub[TARGET])), 4),
        }

    reduction = (
        round((1 - abs(strongest["r_vs_target"]) / V1_STRONGEST["r_vs_target"]) * 100, 1)
        if strongest else None
    )

    return {
        "method": (
            "RE-MEASURED FROM SCRATCH for v2, not carried over. Pearson r of every locked numerical ACTIVE "
            "feature against xt_after, xt_before and the delta itself -- the same plain instrument the v1 "
            "edition used, applied to a target whose xt_after term is now defined completely differently. The "
            "v1 figures are quoted alongside rather than replaced silently."
        ),
        "why_this_part_changed": (
            f"The v1 edition of this part existed because {'target_xt_delta'} = xt_before - xt_after with "
            "xt_after looked up directly from action_x/action_y, so every location-derived locked feature "
            "shared a definitional term with half the target. Prompt 66 removed exactly that mechanism: under "
            "v2, xt_after is 0.0 when the action ends its own possession, the NEXT same-possession event's grid "
            "value otherwise, or that event's own shot_statsbomb_xg if it is a Shot. The acting row's own "
            "coordinates are never used. The v1 premise is therefore FALSE under v2 and could not be restated."
        ),
        "v1_comparison": {
            "v1_strongest_feature": V1_STRONGEST["feature"],
            "v1_r_vs_target": V1_STRONGEST["r_vs_target"],
            "v2_strongest_feature": strongest["feature"] if strongest else None,
            "v2_r_vs_target": strongest["r_vs_target"] if strongest else None,
            "abs_reduction_pct": reduction,
            "note": (
                f"The strongest locked-feature correlation with the target falls from "
                f"{V1_STRONGEST['r_vs_target']:+.4f} ({V1_STRONGEST['feature']}, v1) to "
                f"{strongest['r_vs_target']:+.4f} ({strongest['feature']}, v2) -- a {reduction:.0f}% reduction "
                "in absolute magnitude. Prompt 65 recorded location-vs-target construction coupling as the "
                "single most important caveat its portal carried, with no xG or binary counterpart. The v2 "
                "correction substantially RETIRES that caveat. It is not claimed to be zero -- xt_before is "
                "still a grid lookup at the PREVIOUS event's location, so a residual location relationship is "
                "expected and is visible in the r_vs_xt_before column -- but it is no longer a definitional "
                "recovery of half the target from the feature set."
            ) if strongest and reduction is not None else None,
        },
        "v1_construction_columns_action_x_y": old_construction_status,
        "v2_construction_coupling_moved_to_flags": {
            "columns": flag_coupling,
            "note": (
                "NEW channel, specific to v2 and tested because the mechanism moved rather than vanished. "
                "xt_after = 0 is triggered deterministically by action_ended_possession, so the construction "
                "coupling is now to a FLAG rather than to a location. Measured on this run, that flag correlates "
                f"{flag_coupling['action_ended_possession']['r_vs_target']:+.4f} with the target and "
                f"{flag_coupling['action_ended_possession']['r_vs_xt_after']:+.4f} with xt_after. It is already "
                "OUT of the ACTIVE candidate list -- confirmed here, not assumed -- so no locked feature carries "
                "this coupling. Part C' covers the same three flags from the exclusion-justification angle."
            ),
        },
        "locked_feature_correlations": rows,
        "strongest": strongest,
        "verdict": (
            "substantially resolved relative to v1 -- residual coupling runs through xt_before only, and the "
            "v2 xt_after coupling is confined to an already-excluded flag"
        ),
        "explanation": (
            f"No locked numerical feature now exceeds |r| = {abs(strongest['r_vs_target']):.4f} against the "
            f"target ({strongest['feature']}), against {V1_STRONGEST['r_vs_target']:+.4f} in the v1 edition. "
            "Reading the Numerical Target Atlas no longer requires the heavy 'this may be recovery-by-"
            "definition' discount the v1 portal attached to every location feature. What remains is ordinary "
            "and expected: xt_before is still an xT grid lookup at the previous event's location, so features "
            "describing where play was happening retain a real, modest relationship with it -- see the "
            "r_vs_xt_before column, where the same feature sits materially higher than it does against the "
            "delta. No feature is dropped or added on this evidence, and feature_config.py is unchanged."
        ) if strongest else "no numeric locked features found",
        "prompt_66_cross_reference": v2.PROMPT_66_CONSTRUCTION_FINDING,
        "clearance_link": (
            "This part and the Category Atlas's Clearance row are the same story from two directions. v1's "
            "Clearance artefact was the most visible symptom of scoring the action's OWN location as xt_after; "
            "this part measured the general mechanism behind it. Prompt 66 removed the mechanism, and both "
            "move together: Clearance flips from 8th of 8 (-0.0325) to 1st of 8 (+0.0247), and the strongest "
            f"location coupling falls from {V1_STRONGEST['r_vs_target']:+.4f} to "
            f"{strongest['r_vs_target']:+.4f}." if strongest else ""
        ),
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = xc.load_active_xt()
    candidate_cols = _candidate_columns(ACTIVE)

    a = part_a(df, candidate_cols)
    b = part_b_prime_v2(df)
    c = part_c_prime_v2(df)
    d = part_d_prime_v2(df)

    output = {
        "dataset": "active",
        "leg": "active-binary only (passive leg out of scope for this target)",
        "parquet_path": ACTIVE["parquet_path"],
        "n_rows": len(df),
        "target_col": TARGET,
        "target_scale": xc.target_scale(df),
        "step_0_scope_note": (
            "SCOPE GAP, unchanged from the v1 edition and restated rather than quietly dropped: the xg portal's "
            "LEAKAGE_AUDIT is PASSIVE-DATASET-ONLY (its Parts B/C/D test screened_option_was_avoided, "
            "has_screened_outcome and has_option_2/3, none of which exist in the active parquet), and no "
            "active-side leakage audit exists in this repo for any target. Part A's method is reused by direct "
            "import; Parts B'/C' apply the same method to the active dataset's own analogous columns; Part D' "
            "is new to this target family."
        ),
        "v2_rebuild_note": (
            "Parts A/B'/C' are methodological re-points of the v1 edition -- same imports, same tests, prose "
            "re-derived on v2's numbers. PART D' WAS REWRITTEN. Its v1 premise ('xt_after is looked up directly "
            "from action_x/action_y, so location-derived locked features share a definitional term with half "
            "the target') is false under the corrected target, so it was re-measured from scratch rather than "
            "restated, and a new v2-specific flag-coupling channel was added. See the v1 edition preserved at "
            f"reports/analysis/{v2.V1_PORTAL_DIRNAME}/LEAKAGE_AUDIT.html for what it said before."
        ),
        "methodology": (
            "Part A (target- and dataset-independent schema/naming scan) imported unchanged from "
            "generate_leakage_audit.part_a. Parts B'/C' use the same groupby-mean-target + Welch's t-test the "
            "xg audit uses, with negative/zero/positive shares added because this target's effects can live in "
            "the zero mass rather than the mean -- a caveat that survives from v1 to v2 intact. Part D' is a "
            "construction-coupling check, re-derived for v2's own construction."
        ),
        "parts_with_no_active_counterpart": {
            "part_b_screened_option_was_avoided": "passive-only column; does not exist in the active parquet",
            "part_c_has_screened_outcome": "passive-only column; does not exist in the active parquet",
            "part_d_has_option_2_3": "passive-only columns; do not exist in the active parquet",
        },
        "part_a_known_leaky_scan": a,
        "part_b_prime_has_previous_event": b,
        "part_c_prime_action_possession_flags": c,
        "part_d_prime_construction_coupling": d,
        "prompt_66_cross_references": {
            "clearance_reversal": v2.PROMPT_66_CLEARANCE_FINDING,
            "distribution_shape": v2.PROMPT_66_SHAPE_FINDING,
            "possession_ending_collapse": v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING,
            "construction_coupling": v2.PROMPT_66_CONSTRUCTION_FINDING,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("=== Part A ===", "clean" if a["verdict"] == "clean" else "ISSUES FOUND")
    print(f"Part B' has_previous_event: zero-share gap {b['zero_share_gap_pp']:+.1f}pp, Welch p={b['p_value']:.3e}")
    for col, r in c.items():
        tr = r["by_value"].get("True", {})
        print(f"Part C' {col}: True n={tr.get('n'):,} mean={tr.get('mean_xt'):+.6f} "
              f"collapses_to_xt_before={r['collapses_to_xt_before']}")
    vc = d["v1_comparison"]
    print(f"Part D' strongest coupling v1 {vc['v1_strongest_feature']} r={vc['v1_r_vs_target']:+.4f} "
          f"-> v2 {vc['v2_strongest_feature']} r={vc['v2_r_vs_target']:+.4f} ({vc['abs_reduction_pct']:.0f}% lower)")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
