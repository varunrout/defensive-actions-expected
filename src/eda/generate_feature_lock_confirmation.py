"""CLI entrypoint: post-lock correlation confirmation pass (prompt 14).

Not another COLLAPSE/REVIEW redundancy round -- that work is closed. This
re-runs the correlation machinery fresh against the final locked candidate
lists (after the functional-role bucket rename, the VIF drop, and the
leakage drop) and diffs the result pair-for-pair against the last confirmed
run, so any tier crossing is named explicitly rather than silently absorbed
into an unchanged-looking summary.

Usage:
    python -m src.eda.generate_feature_lock_confirmation --before-correlation <path> --before-review <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eda.feature_config import ACTIVE, PASSIVE

REPO_ROOT = Path(__file__).resolve().parents[2]
CORRELATION_PATH = REPO_ROOT / "reports" / "eda" / "CORRELATION_ANALYSIS.json"
REVIEW_PATH = REPO_ROOT / "reports" / "eda" / "REVIEW_ANALYSIS.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "eda" / "FEATURE_LOCK_CONFIRMATION.json"

EXPECTED_ACTIVE_COUNT = 34
EXPECTED_PASSIVE_COUNT = 38

CHANGES_SINCE_LAST_FULL_RUN = [
    {
        "change": "defender_functional_role bucket fix",
        "detail": "4 categories -> 5: added advanced_wide (the previously-missing deep x wide quadrant), renamed the residual n>=2 fallback from unclassified to mid_block. unclassified now reserved exclusively for n<2.",
        "feature_count_impact": "none (category-count change on an existing categorical column, not a column add/drop)",
    },
    {
        "change": "V2 methodology-gap fixes",
        "detail": "Added a categorical REVIEW band to correlation.py's tiering and Type 5/Type 6 review resolution logic. V1 pipeline (correlation.py, generate_review_analysis.py) reverted to its pre-V2 state after a scoping correction; V2 lives in separate _v2-suffixed modules.",
        "feature_count_impact": "none (V1 candidate lists untouched; V2 is a parallel analysis track)",
    },
    {
        "change": "VIF drop",
        "detail": "local_numerical_balance_5m/10m dropped from the active candidate list -- near-exact linear combination of attackers_within_Nm - defenders_within_Nm (already kept), invisible to pairwise correlation, caught by VIF (condition number ~1.36e15 pre-drop, unbounded VIF for 6 columns).",
        "feature_count_impact": "active: 36 -> 34",
    },
    {
        "change": "Leakage drop",
        "detail": "has_screened_outcome dropped from the passive candidate list -- censoring-mechanism proxy for target_future_shot_10s's own 10s window being truncated at end-of-period/match (chi2=136.5, p=1.54e-31), not defensive signal.",
        "feature_count_impact": "passive: 39 -> 38",
    },
    {
        "change": "Football sanity check",
        "detail": "Validation-only pass (recomputation checks against stored values). No feature-list or pipeline changes.",
        "feature_count_impact": "none",
    },
    {
        "change": "Passive archetype clustering",
        "detail": "New analysis module (src/dax/analysis/passive_archetypes.py) clustering within each defender_functional_role bucket. Does not touch feature_config.py's candidate lists.",
        "feature_count_impact": "none",
    },
    {
        "change": "Canonical match-grouped split",
        "detail": "Frozen TEST/CV match assignment for model training. Does not touch feature_config.py's candidate lists.",
        "feature_count_impact": "none",
    },
]


def _index_pairs(correlation_data: dict) -> dict[str, dict[frozenset, dict]]:
    """dataset -> {frozenset({feature_a, feature_b}): pair_record}, DROP/COLLAPSE/REVIEW
    tiers only -- these are exhaustive lists. distinct_sample is deliberately excluded:
    it's a top-10-by-magnitude sample, not the full DISTINCT population, so a pair
    entering/leaving that sample is sampling noise, not a real tier change, and diffing
    it would produce spurious added/removed entries unrelated to the actual feature set."""
    out: dict[str, dict[frozenset, dict]] = {}
    for ds_key, ds in correlation_data["datasets"].items():
        pairs: dict[frozenset, dict] = {}
        for tier in ("drop", "collapse", "review"):
            for r in ds["pairs"].get(tier, []):
                key = frozenset({r["feature_a"], r["feature_b"]})
                pairs[key] = {**r, "tier": tier}
        out[ds_key] = pairs
    return out


def diff_correlation(before: dict, after: dict) -> dict:
    before_idx = _index_pairs(before)
    after_idx = _index_pairs(after)

    result = {}
    for ds_key in ("active", "passive"):
        b = before_idx.get(ds_key, {})
        a = after_idx.get(ds_key, {})

        b_keys = set(b.keys())
        a_keys = set(a.keys())

        added = sorted(
            ({"feature_a": a[k]["feature_a"], "feature_b": a[k]["feature_b"], "tier": a[k]["tier"], "value": a[k]["value"]} for k in (a_keys - b_keys)),
            key=lambda r: (r["feature_a"], r["feature_b"]),
        )
        removed = sorted(
            ({"feature_a": b[k]["feature_a"], "feature_b": b[k]["feature_b"], "tier": b[k]["tier"], "value": b[k]["value"]} for k in (b_keys - a_keys)),
            key=lambda r: (r["feature_a"], r["feature_b"]),
        )

        tier_changed = []
        for k in (a_keys & b_keys):
            if a[k]["tier"] != b[k]["tier"]:
                tier_changed.append({
                    "feature_a": a[k]["feature_a"],
                    "feature_b": a[k]["feature_b"],
                    "before_tier": b[k]["tier"],
                    "after_tier": a[k]["tier"],
                    "before_value": b[k]["value"],
                    "after_value": a[k]["value"],
                })
        tier_changed.sort(key=lambda r: (r["feature_a"], r["feature_b"]))

        # Only DROP/COLLAPSE crossings matter for Part B's trigger condition.
        newly_risky = [
            c for c in tier_changed
            if c["after_tier"] in ("drop", "collapse") and c["before_tier"] not in ("drop", "collapse")
        ]

        result[ds_key] = {
            "n_pairs_before": len(b),
            "n_pairs_after": len(a),
            "n_added": len(added),
            "n_removed": len(removed),
            "n_tier_changed": len(tier_changed),
            "added_pairs": added,
            "removed_pairs": removed,
            "tier_changed_pairs": tier_changed,
            "newly_risky_pairs": newly_risky,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-correlation", required=True, help="Path to the pre-rerun CORRELATION_ANALYSIS.json snapshot")
    parser.add_argument("--before-review", required=True, help="Path to the pre-rerun REVIEW_ANALYSIS.json snapshot (currently informational only, not diffed pair-for-pair)")
    args = parser.parse_args()

    active_count = len(ACTIVE["categorical"]) + len(ACTIVE["boolean"]) + len(ACTIVE["continuous"]) + len(ACTIVE["discrete"])
    passive_count = len(PASSIVE["categorical"]) + len(PASSIVE["boolean"]) + len(PASSIVE["continuous"]) + len(PASSIVE["discrete"])

    before_correlation = json.loads(Path(args.before_correlation).read_text(encoding="utf-8"))
    after_correlation = json.loads(CORRELATION_PATH.read_text(encoding="utf-8"))

    diff = diff_correlation(before_correlation, after_correlation)
    any_newly_risky = any(diff[ds]["newly_risky_pairs"] for ds in diff)
    diff_is_empty = all(
        diff[ds]["n_added"] == 0 and diff[ds]["n_removed"] == 0 and diff[ds]["n_tier_changed"] == 0
        for ds in diff
    )

    output = {
        "confirmed_counts": {
            "active": active_count,
            "active_expected": EXPECTED_ACTIVE_COUNT,
            "active_matches_expected": active_count == EXPECTED_ACTIVE_COUNT,
            "passive": passive_count,
            "passive_expected": EXPECTED_PASSIVE_COUNT,
            "passive_matches_expected": passive_count == EXPECTED_PASSIVE_COUNT,
        },
        "changes_since_last_full_run": CHANGES_SINCE_LAST_FULL_RUN,
        "correlation_diff": diff,
        "diff_is_empty": diff_is_empty,
        "any_newly_risky_pairs_found": any_newly_risky,
        "part_b_triggered": any_newly_risky,
        "verdict": (
            "No new DROP/COLLAPSE crossings found -- the locked feature set is internally consistent after all "
            "landed changes." if not any_newly_risky else
            "NEW DROP/COLLAPSE crossing(s) found -- see newly_risky_pairs. These require resolution under the "
            "existing V1/V2 Type 1-6 framework before this report can call the feature set locked."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"active: {active_count} (expected {EXPECTED_ACTIVE_COUNT}) {'OK' if active_count == EXPECTED_ACTIVE_COUNT else 'MISMATCH'}")
    print(f"passive: {passive_count} (expected {EXPECTED_PASSIVE_COUNT}) {'OK' if passive_count == EXPECTED_PASSIVE_COUNT else 'MISMATCH'}")
    print()
    for ds_key, d in diff.items():
        print(f"=== {ds_key} ===")
        print(f"  pairs before={d['n_pairs_before']} after={d['n_pairs_after']} added={d['n_added']} removed={d['n_removed']} tier_changed={d['n_tier_changed']}")
        for p in d["tier_changed_pairs"]:
            print(f"    TIER CHANGE: {p['feature_a']} <-> {p['feature_b']}: {p['before_tier']} ({p['before_value']}) -> {p['after_tier']} ({p['after_value']})")
        for p in d["added_pairs"]:
            print(f"    ADDED: {p['feature_a']} <-> {p['feature_b']} [{p['tier']}] {p['value']}")
        for p in d["removed_pairs"]:
            print(f"    REMOVED: {p['feature_a']} <-> {p['feature_b']} [{p['tier']}] {p['value']}")

    print(f"\nverdict: {output['verdict']}")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
