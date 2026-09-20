"""CLI entrypoint: post-lock confirmation + pattern findings for the
xT-delta target, ACTIVE-BINARY LEG ONLY -- the xT sibling of
generate_feature_lock_confirmation_xg.py (which builds
FEATURE_LOCK_CONFIRMATION_XG.json) plus the pattern-findings section
generate_feature_lock_pattern_findings.py appends to it. Confirmed by
reading all three: the xg portal's FEATURE_LOCK_CONFIRMATION_XG.html is a
single document carrying BOTH the lock confirmation and the pattern
findings, so both are produced here in one file too.

Structural point carried over exactly from the xg version, not re-derived:
the correlation/redundancy machinery is TARGET-AGNOSTIC (feature vs
feature, never feature vs target). Re-running it against a different target
would produce byte-identical pairs and tiers. So `correlation_diff` is READ
DIRECTLY from reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json
and labelled as identical by construction, rather than silently recomputed
-- the same discipline generate_feature_lock_confirmation_xg.py applies.

What IS target-specific is the leakage side and the pattern findings, both
rebuilt here from this portal's own JSONs.

Usage:
    python -m src.eda.generate_feature_lock_confirmation_xt
"""

from __future__ import annotations

import json

from src.eda import xt_common as xc
from src.eda.feature_config import ACTIVE

BINARY_LOCK_PATH = xc.REPO_ROOT / "reports" / "analysis" / "shot_target" / "FEATURE_LOCK_CONFIRMATION.json"
OUT_DIR = xc.OUT_DIR
OUTPUT_PATH = OUT_DIR / "FEATURE_LOCK_CONFIRMATION_XT.json"

EXPECTED_ACTIVE_COUNT = 34


def _load(name: str) -> dict:
    return json.loads((OUT_DIR / name).read_text(encoding="utf-8"))


def build_leakage_confirmation() -> dict:
    la = _load("LEAKAGE_AUDIT.json")
    b = la["part_b_prime_has_previous_event"]
    c = la["part_c_prime_action_possession_flags"]
    d = la["part_d_prime_construction_coupling"]

    return {
        "source": "reports/analysis/xt_target/LEAKAGE_AUDIT.json",
        "scope_difference_from_xg": (
            "The xg portal's lock confirmation folds in that portal's LEAKAGE_AUDIT Parts C and D, which are "
            "PASSIVE-only checks (has_screened_outcome, has_option_2/3). Neither column exists in the active "
            "parquet, so neither can be confirmed here. This portal's leakage audit substitutes active-side "
            "analogues and adds one new check; those are what is folded in below. Stated plainly rather than "
            "presented as the same confirmation."
        ),
        "part_a_verdict": la["part_a_known_leaky_scan"]["verdict"],
        "part_b_prime_has_previous_event": {
            "zero_share_gap_pp": b["zero_share_gap_pp"],
            "p_value_on_the_mean": b["p_value"],
            "verdict": b["verdict"],
            "implication_for_the_lock": (
                "has_previous_event stays a locked candidate feature. The finding is about how this target must "
                "be MEASURED (its effects can sit in the zero mass rather than the mean), not about whether the "
                "feature belongs in the list. A mean-difference test alone -- the instrument the xg audit uses "
                f"-- returns p={b['p_value']:.3g} here and would have reported nothing, while the exactly-zero "
                f"share differs by {b['zero_share_gap_pp']:+.1f}pp."
            ),
        },
        "part_c_prime_action_possession_flags": {
            "columns": list(c.keys()),
            "verdicts": {k: v["verdict"] for k, v in c.items()},
            "implication_for_the_lock": (
                "These three columns remain EXCLUDED in feature_config.py and this document does not change "
                "that. What it records is that their stated exclusion reason -- 'structural zero (0.00% shot "
                "rate by target-window definition)' -- is specific to target_future_shot_10s and does not hold "
                "against target_xt_delta, where the same rows carry real row-level variance. Any future "
                "decision to model this target seriously would need to revisit that exclusion on its own "
                "merits; it is not revisited here."
            ),
        },
        "part_d_prime_construction_coupling": {
            "strongest": d["strongest"],
            "verdict": d["verdict"],
            "implication_for_the_lock": (
                "No feature is dropped on this evidence, and none is proposed for dropping. The locked set "
                "stays as-is. The finding is interpretive: because xt_after is looked up from action_x/action_y, "
                "location-derived locked features share a definitional term with half the target, so a strong "
                "correlation in the Numerical Target Atlas is partly recovery-by-definition rather than "
                "discovered defensive signal. This has no counterpart in the xg or binary confirmations, and is "
                "the single most important caveat this portal carries."
            ),
        },
        "verdict": (
            "The locked ACTIVE feature set is internally unchanged for xT-delta work -- correlation/redundancy "
            "is target-agnostic, so nothing about the feature list moves. Two target-specific caveats are "
            "recorded rather than resolved: (1) construction coupling between location-derived features and "
            "xt_after, and (2) the fact that this target's effects can live in its zero mass, which a "
            "mean-difference test misses."
        ),
    }


def build_pattern_findings() -> dict:
    atlas = _load("active_numerical_target_atlas.json")
    cat = _load("active_category_atlas.json")
    flag = _load("active_flag_ledger.json")
    confound = _load("CONFOUND_ANALYSIS.json")
    tourn = _load("TOURNAMENT_STABILITY_CHECK.json")
    v1 = _load("SLICE_STRATIFICATION.json")
    v2 = _load("SLICE_STRATIFICATION_V2.json")
    inter = _load("FEATURE_INTERACTION_ANALYSIS.json")
    review = _load("REVIEW_ANALYSIS.json")
    dist = _load("active_distribution_atlas.json")

    et = cat["columns"]["event_type"]["categories"]
    clearance = next(r for r in et if r["category"] == xc.CLEARANCE_EVENT_TYPE)
    clearance_rank = [r["category"] for r in et].index(xc.CLEARANCE_EVENT_TYPE) + 1

    flag_ranked = sorted(
        ({"column": k, **v["lift_stats"]} for k, v in flag["columns"].items()),
        key=lambda e: abs(e["lift"]), reverse=True,
    )

    return {
        "source_note": (
            "Every number in this section is read from this portal's own JSON outputs, not re-typed by hand and "
            "not carried over from the xg or binary portals. Where a finding has an xg or binary counterpart, "
            "the comparison is made in the individual report, not restated here."
        ),
        "target_shape": dist["target_distribution"],
        "numerical_vs_target_rankings": {
            "active_top": [
                {"feature": f["feature"], "spearman_rho": f["spearman_rho"], "shape": f["shape"],
                 "binned_curve_crosses_zero": f.get("binned_curve_crosses_zero")}
                for f in atlas["features"][:5]
            ],
            "n_curves_crossing_zero": atlas["n_features_whose_binned_curve_crosses_zero"],
            "n_features": atlas["n_features_analyzed"],
            "n_train_test_inconsistent": atlas["n_features_flagged_inconsistent"],
            "note": (
                "Top 5 of the reconstructed ACTIVE pool by |Spearman rho|, read from the atlas's own sort order. "
                "Ranking is by ABSOLUTE rho in both directions -- on this target a strong negative rho is as "
                "much a finding as a strong positive one."
            ),
        },
        "category_atlas": {
            "clearance_rank": f"{clearance_rank} of {len(et)} event_type categories",
            "clearance_mean_xt": clearance["mean_xt"],
            "clearance_n": clearance["n"],
            "clearance_pct_negative": clearance["pct_negative"],
            "finding": (
                f"Prompt 64's Clearance/action_x artefact resurfaces exactly where it was predicted to: "
                f"Clearance ranks {clearance_rank} of {len(et)} event_type categories at "
                f"{clearance['mean_xt']:+.6f} mean delta over n={clearance['n']:,} rows "
                f"({clearance['pct_negative']:.1f}% negative). Prompt 64 measured -0.0315 inside the "
                "action_ended_possession slice; this is the same artefact on the full active dataset."
            ),
        },
        "flag_ledger": {
            "top_by_abs_lift": [
                {"column": e["column"], "lift": e["lift"], "mean_xt_true": e["mean_xt_true"],
                 "mean_xt_false": e["mean_xt_false"], "n_true": e["n_true"]}
                for e in flag_ranked[:5]
            ],
            "possession_flags_no_longer_degenerate": {
                k: {"mean_xt_true": v["mean_xt_true"], "n_true": v["n_true"],
                    "pct_zero_true": v["true_direction"]["pct_zero"], "lift": v["lift"]}
                for k, v in flag["off_pool_possession_flags"]["lifts"].items()
            },
            "possession_flags_off_pool_note": flag["off_pool_possession_flags"]["note"],
        },
        "confound_tests": {
            "n_tests": len(confound["tests"]),
            "tests": [{"name": t["name"], "title": t["title"], "verdict": t["verdict"]["verdict"],
                       "verdict_given_nonzero_delta": t["given_nonzero_delta"]["verdict"]["verdict"],
                       "is_new_test": bool(t.get("is_new_test"))}
                      for t in confound["tests"]],
            "note": confound["step_0_scope_note"],
        },
        "tournament_stability": {
            "n_features": len(tourn["features"]),
            "features": [{"feature": f["feature"], "verdict": f["verdict"]} for f in tourn["features"]],
            "n_genuine": sum(1 for f in tourn["features"] if f["verdict"] == "genuine tournament-level difference"),
        },
        "slice_stratification": {
            "v1": {"n_cells": v1["n_cells"],
                   "n_genuine_divergences": v1["cross_dataset_summary"]["n_genuine_divergences"],
                   "n_genuine_divergences_given_nonzero_delta": v1["cross_dataset_summary"]["n_genuine_divergences_given_nonzero_delta"],
                   "n_conditioning_disagreements": v1["cross_dataset_summary"]["n_conditioning_disagreements"]},
            "v2": {"n_cells": v2["n_cells"],
                   "n_genuine_divergences": v2["cross_dataset_summary"]["n_genuine_divergences"],
                   "n_genuine_divergences_given_nonzero_delta": v2["cross_dataset_summary"]["n_genuine_divergences_given_nonzero_delta"],
                   "archetype_comparison": v2["cross_dataset_summary"]["closing_note_archetype_vs_boolean"]},
        },
        "numeric_interaction": {
            "n_pairs": inter["n_pairs_total"],
            "n_agree": inter["n_pairs_where_unconditional_and_conditional_agree"],
            "n_sign_flipping": inter["n_pairs_with_sign_flipping_stratum_deltas"],
            "pairs": [{"feature_a": p["feature_a"], "feature_b": p["feature_b"],
                       "classification": p["classification"],
                       "within_stratum_deltas_change_sign": p["within_stratum_deltas_change_sign"]}
                      for p in inter["summary_table"]],
        },
        "review_tier": {
            "n_review_pairs": review["datasets"]["active"]["n_review_pairs"],
            "n_needs_human_call": len(review["datasets"]["active"]["needs_human_call"]),
            "n_type2_opposite_sign": review["datasets"]["active"]["n_type2_opposite_sign_pairs"],
        },
        "cross_referenced_target_independent_sections": {
            "correlation_atlas_v1_v2_v3": (
                "Not rebuilt. Correlation clustering is feature-vs-feature and never references a target -- "
                "re-running it against target_xt_delta would produce byte-identical pairs and tiers. Linked from "
                "this portal's index to reports/analysis/xg_target/ as-is."
            ),
            "vif": "Not rebuilt -- multicollinearity is a property of the feature matrix alone. Linked as-is.",
            "slicer_redundancy": "Not rebuilt -- slicer-vs-slicer redundancy, no target term. Linked as-is.",
            "player_level_validity": "Not rebuilt -- row concentration and player overlap, no target term. Linked as-is.",
            "player_grouped_split_check": "Not rebuilt -- an identity-leakage stress test on the split itself. Linked as-is.",
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    binary_lock = json.loads(BINARY_LOCK_PATH.read_text(encoding="utf-8"))

    active_count = sum(len(ACTIVE[k]) for k in ("categorical", "boolean", "continuous", "discrete"))

    output = {
        "leg": "active-binary only (passive leg out of scope for this target)",
        "target": xc.TARGET,
        "confirmed_counts": {
            "active": active_count,
            "active_expected": EXPECTED_ACTIVE_COUNT,
            "active_matches_expected": active_count == EXPECTED_ACTIVE_COUNT,
        },
        "changes_since_last_full_run": binary_lock["changes_since_last_full_run"],
        "correlation_diff": {"active": binary_lock["correlation_diff"]["active"]},
        "correlation_diff_note": (
            "IDENTICAL to reports/analysis/shot_target/FEATURE_LOCK_CONFIRMATION.json's active correlation_diff, "
            "by construction, not independently re-derived -- correlation clustering (feature vs feature) never "
            "references a target column, so re-running it against target_xt_delta would produce the exact same "
            "pairs and tiers. Reused directly rather than silently recomputed, so this file's numbers are "
            "traceable to that source. Same discipline generate_feature_lock_confirmation_xg.py applies."
        ),
        "diff_is_empty": binary_lock["correlation_diff"]["active"]["n_added"] == 0
                         and binary_lock["correlation_diff"]["active"]["n_removed"] == 0
                         and binary_lock["correlation_diff"]["active"]["n_tier_changed"] == 0,
        "any_newly_risky_pairs_found": bool(binary_lock["correlation_diff"]["active"]["newly_risky_pairs"]),
        "leakage_confirmation_xt": build_leakage_confirmation(),
        "pattern_analysis_findings": build_pattern_findings(),
        "prompt_64_cross_references": {
            "clearance_action_x_artefact": xc.PROMPT_64_CLEARANCE_FINDING,
            "distribution_shape": xc.PROMPT_64_SHAPE_FINDING,
        },
        "verdict": (
            f"The locked ACTIVE feature set ({active_count} features) is confirmed internally consistent for "
            "xT-delta work -- same count as the binary and xG confirmations, since feature_config.py is not "
            "target-specific and correlation/redundancy is target-agnostic. No feature is added, dropped or "
            "re-tiered by this document. Two target-specific caveats are RECORDED, not resolved: construction "
            "coupling between location-derived locked features and the target's own xt_after term (the single "
            "most important caveat this portal carries, with no xG or binary counterpart), and the fact that "
            "this target's effects can sit in its exactly-zero mass where a mean-difference test will miss them."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"active: {active_count} (expected {EXPECTED_ACTIVE_COUNT}) "
          f"{'OK' if active_count == EXPECTED_ACTIVE_COUNT else 'MISMATCH'}")
    print(f"correlation diff empty: {output['diff_is_empty']} (reused from the binary confirmation)")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
