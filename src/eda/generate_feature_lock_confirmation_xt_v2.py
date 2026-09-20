"""CLI entrypoint: post-lock confirmation + pattern findings for
`target_xt_delta_v2`, ACTIVE-BINARY LEG ONLY.

STEP-0 DISPOSITION: REWRITE (prose only -- every computation is re-pointed).

`generate_feature_lock_confirmation_xt.py` aggregates this portal's own
JSONs, so re-pointing alone already produces correct NUMBERS for v2. Its
structural logic is target-agnostic and is reused here by direct import:
the correlation_diff is still read from the binary confirmation and
labelled identical-by-construction, the counts still come from
`feature_config.ACTIVE`, and every roll-up still reads the JSON this
portal just wrote.

Four of its narrative strings, however, state v1 findings as fact:

  - `category_atlas.finding` asserts "Prompt 64's Clearance/action_x
    artefact resurfaces exactly where it was predicted to" and quotes
    -0.0315. Under v2 the artefact reverses; Clearance ranks 1st of 8.
  - `part_d_prime_construction_coupling.implication_for_the_lock` and the
    document `verdict` both name construction coupling "the single most
    important caveat this portal carries". The v2 Leakage Audit measures
    that coupling as 74% weaker and substantially retired.
  - `part_c_prime...implication_for_the_lock` describes the possession
    flags as carrying "real row-level variance" without noting that two of
    them are now construction inputs to the target.

Those four are replaced here. Everything else is the v1 module's own output.

Usage:
    python -m src.eda.generate_feature_lock_confirmation_xt_v2
"""

from __future__ import annotations

import json

from src.eda import xt_v2_common as v2  # re-points xt_common; MUST precede the generator import
from src.eda import xt_common as xc
from src.eda import generate_feature_lock_confirmation_xt as f1
from src.eda.feature_config import ACTIVE

OUT_DIR = xc.OUT_DIR
OUTPUT_PATH = OUT_DIR / "FEATURE_LOCK_CONFIRMATION_XT.json"


def build_leakage_confirmation_v2() -> dict:
    lc = f1.build_leakage_confirmation()
    la = json.loads((OUT_DIR / "LEAKAGE_AUDIT.json").read_text(encoding="utf-8"))
    d = la["part_d_prime_construction_coupling"]
    c = la["part_c_prime_action_possession_flags"]
    vc = d["v1_comparison"]

    lc["source"] = "reports/analysis/xt_target/LEAKAGE_AUDIT.json (v2)"

    lc["part_c_prime_action_possession_flags"]["implication_for_the_lock"] = (
        "These three columns remain EXCLUDED in feature_config.py and this document does not change that. What "
        "changed from v1 is the REASON the exclusion is right. v1 recorded that their stated justification "
        "('structural zero, 0.00% shot rate by target-window definition') is specific to target_future_shot_10s "
        "and does not hold against the xT target, where the same rows carry real row-level variance -- which "
        "left an open question about whether they should be unlocked. Under v2 that question effectively "
        "closes in the opposite direction: action_ended_possession is the deterministic TRIGGER for "
        "xt_after = 0, so its True group's delta equals xt_before exactly (np.allclose, verified in the Leakage "
        "Audit Part C'). It is a construction input to the target, not a candidate feature, and modelling on it "
        "would be circular. The lock is unchanged; the justification for it is now stronger, not weaker."
    )

    lc["part_d_prime_construction_coupling"] = {
        "strongest": d["strongest"],
        "verdict": d["verdict"],
        "v1_comparison": vc,
        "implication_for_the_lock": (
            "No feature is dropped on this evidence, and none is proposed for dropping -- the locked set stays "
            "as-is, exactly as under v1. What changed is the weight of the caveat attached to reading the "
            "Numerical Target Atlas. v1 recorded construction coupling as the single most important caveat its "
            "portal carried, with no xG or binary counterpart: because xt_after was looked up from "
            "action_x/action_y, location-derived locked features shared a definitional term with half the "
            f"target, and the strongest sat at r={vc['v1_r_vs_target']:+.4f}. Prompt 66 removed that mechanism "
            f"entirely, and the strongest locked-feature correlation falls to r={vc['v2_r_vs_target']:+.4f} "
            f"({vc['abs_reduction_pct']:.0f}% lower). The caveat is substantially RETIRED rather than carried "
            "forward. A residual location relationship remains through xt_before, which is unchanged from v1 "
            "and is still a grid lookup at the previous event's location -- that is ordinary and expected, not "
            "recovery-by-definition of half the target."
        ),
    }

    lc["verdict"] = (
        "The locked ACTIVE feature set is internally unchanged for xT-delta work -- correlation/redundancy is "
        "target-agnostic, so nothing about the feature list moves. The two caveats v1 recorded now stand very "
        "differently: (1) construction coupling between location-derived features and xt_after is SUBSTANTIALLY "
        f"RETIRED by the v2 correction ({vc['v1_r_vs_target']:+.4f} -> {vc['v2_r_vs_target']:+.4f} on the "
        "strongest feature), while (2) the fact that this target's effects can live in its exactly-zero mass "
        "SURVIVES intact and, with v2's zero share rising from 11.3% to 20.2%, matters more rather than less."
    )
    return lc


def build_pattern_findings_v2() -> dict:
    pf = f1.build_pattern_findings()
    ca = pf["category_atlas"]
    et_rank = ca["clearance_rank"]

    ca["finding"] = (
        f"Prompt 64's Clearance/action_x artefact does NOT resurface -- it reverses, exactly as Prompt 66 "
        f"reported. Clearance ranks {et_rank} at {ca['clearance_mean_xt']:+.6f} mean delta over "
        f"n={ca['clearance_n']:,} rows ({ca['clearance_pct_negative']:.1f}% negative). The v1 edition of this "
        f"confirmation recorded the same category at 8 of 8 and -0.032482, and flagged it as a measurement "
        f"artefact of the xt_after definition. Under the corrected xt_after the sign flips and the rank "
        f"inverts. Confirmed independently on this run, not quoted."
    )
    ca["v1_comparison"] = {
        "v1_clearance_rank": "8 of 8 event_type categories",
        "v1_clearance_mean_xt": -0.032482,
        "v2_clearance_rank": et_rank,
        "v2_clearance_mean_xt": ca["clearance_mean_xt"],
    }
    ca["prompt_66_cross_reference"] = v2.PROMPT_66_CLEARANCE_FINDING

    pf["flag_ledger"]["possession_flags_construction_note"] = v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING
    return pf


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary_lock = json.loads(f1.BINARY_LOCK_PATH.read_text(encoding="utf-8"))
    active_count = sum(len(ACTIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))
    lc = build_leakage_confirmation_v2()
    vc = lc["part_d_prime_construction_coupling"]["v1_comparison"]

    output = {
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": xc.TARGET,
        "supersedes": (
            f"the v1 edition preserved at reports/analysis/{v2.V1_PORTAL_DIRNAME}/"
            "FEATURE_LOCK_CONFIRMATION_XT.html"
        ),
        "confirmed_counts": {
            "active": active_count,
            "active_expected": f1.EXPECTED_ACTIVE_COUNT,
            "active_matches_expected": active_count == f1.EXPECTED_ACTIVE_COUNT,
        },
        "changes_since_last_full_run": binary_lock["changes_since_last_full_run"],
        "correlation_diff": {"active": binary_lock["correlation_diff"]["active"]},
        "correlation_diff_note": (
            "IDENTICAL to reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json's active correlation_diff, "
            "by construction, not independently re-derived -- correlation clustering (feature vs feature) never "
            "references a target column, so re-running it against target_xt_delta_v2 would produce the exact "
            "same pairs and tiers. This is also precisely why the target correction does not touch the seven "
            "target-independent reports: nothing about xt_after's definition can move a feature-vs-feature "
            "statistic. Same discipline generate_feature_lock_confirmation_xg.py applies."
        ),
        "diff_is_empty": binary_lock["correlation_diff"]["active"]["n_added"] == 0
                         and binary_lock["correlation_diff"]["active"]["n_removed"] == 0
                         and binary_lock["correlation_diff"]["active"]["n_tier_changed"] == 0,
        "any_newly_risky_pairs_found": bool(binary_lock["correlation_diff"]["active"]["newly_risky_pairs"]),
        "leakage_confirmation_xt": lc,
        "pattern_analysis_findings": build_pattern_findings_v2(),
        "prompt_66_cross_references": {
            "clearance_reversal": v2.PROMPT_66_CLEARANCE_FINDING,
            "distribution_shape": v2.PROMPT_66_SHAPE_FINDING,
            "possession_ending_collapse": v2.PROMPT_66_POSSESSION_COLLAPSE_FINDING,
            "construction_coupling": v2.PROMPT_66_CONSTRUCTION_FINDING,
            "correlation_with_old_target": v2.PROMPT_66_CORRELATION_FINDING,
        },
        "verdict": (
            f"The locked ACTIVE feature set ({active_count} features) is confirmed internally consistent for "
            "xT-delta v2 work -- same count as the binary, xG and v1 xT confirmations, since feature_config.py "
            "is not target-specific and correlation/redundancy is target-agnostic. No feature is added, dropped "
            "or re-tiered by this document, and none was by its v1 edition either. What the target correction "
            "changes is the caveat set, not the lock: construction coupling, which v1 called the single most "
            f"important caveat it carried, is substantially retired ({vc['v1_r_vs_target']:+.4f} -> "
            f"{vc['v2_r_vs_target']:+.4f} on the strongest locked feature, {vc['abs_reduction_pct']:.0f}% "
            "lower), while the zero-mass measurement caveat survives intact and applies to a larger share of "
            "rows than before (20.2% exactly zero, up from 11.3%). One threshold does NOT transfer cleanly and "
            "is reported rather than forced: the std-derived flat_margin now spans the majority of the "
            "distribution -- see `target_scale.threshold_transfer_warning` in this portal's other JSONs."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"active: {active_count} (expected {f1.EXPECTED_ACTIVE_COUNT}) "
          f"{'OK' if active_count == f1.EXPECTED_ACTIVE_COUNT else 'MISMATCH'}")
    print(f"correlation diff empty: {output['diff_is_empty']} (reused from the binary confirmation)")
    print(f"construction coupling: {vc['v1_r_vs_target']:+.4f} (v1) -> {vc['v2_r_vs_target']:+.4f} (v2)")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
