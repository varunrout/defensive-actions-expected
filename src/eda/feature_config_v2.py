"""V2 (prompt 7/7) additions to feature_config.py -- kept in a separate
module so the V1 pipeline (correlation.py, generate_review_analysis.py,
generate_correlation_reports.py, generate_reports.py) is untouched by the V2
review-resolution logic. feature_config.py's ACTIVE/PASSIVE/DATASETS remain
the single source of truth for the candidate feature lists; this module only
adds the extra lookup tables the V2 scripts need.
"""

from __future__ import annotations

# Explicit continuous<->continuous "engineered family" groups for Type 6
# review resolution -- deliberately hardcoded, not inferred by
# string-matching, so a pair only auto-resolves when it's a genuine
# same-measurement-at-a-different-radius/rank/option relationship, not just
# a shared substring. Each group's `always_needs_human_call=True` marks a
# family gated on a not-yet-built baseline model (Part B of the structural
# redesign, PASSIVE_COLLAPSE_OPTION_RANKS) -- pairs inside it stay
# needs_human_call regardless of |r|, because the whole point of that
# deferral is that correlation isn't what decides it.
FAMILY_GROUPS_ACTIVE = [
    {"columns": frozenset({"attackers_within_5m", "attackers_within_10m"}), "reason_tag": "within-radius counts, different radius"},
    {"columns": frozenset({"defenders_within_5m", "defenders_within_10m"}), "reason_tag": "within-radius counts, different radius"},
    {"columns": frozenset({"local_numerical_balance_5m", "local_numerical_balance_10m"}), "reason_tag": "numerical-balance, different radius"},
    {"columns": frozenset({"nearest_attacker_distance", "nearest_defender_distance"}), "reason_tag": "nearest-distance, different side"},
    {"columns": frozenset({"attacker_spread", "defender_spread"}), "reason_tag": "spread, different side"},
]

FAMILY_GROUPS_PASSIVE = [
    {"columns": frozenset({"lane_screening_score_option_1", "lane_screening_score_option_2", "lane_screening_score_option_3"}), "reason_tag": "lane-screening score, different ranked option"},
    {
        "columns": frozenset({"top_option_1_threat_score", "top_option_2_threat_score", "top_option_3_threat_score"}),
        "reason_tag": "ranked-option threat score, different rank",
        "always_needs_human_call": True,
        "always_reason": (
            "Cluster 5 (prompt 5/5 & 6/6): deliberately deferred to baseline-model feature-importance evidence "
            "(PASSIVE_COLLAPSE_OPTION_RANKS, prompt 4/4 Part B), not resolved by correlation alone -- stays "
            "needs_human_call regardless of |r|."
        ),
    },
    {"columns": frozenset({"top_option_1_dx", "top_option_2_dx", "top_option_3_dx"}), "reason_tag": "ball-relative dx, different ranked option"},
    {"columns": frozenset({"top_option_1_dy", "top_option_2_dy", "top_option_3_dy"}), "reason_tag": "ball-relative dy, different ranked option"},
    {"columns": frozenset({"top_option_1_distance_from_ball", "top_option_2_distance_from_ball", "top_option_3_distance_from_ball"}), "reason_tag": "ball-relative distance, different ranked option"},
    {"columns": frozenset({"top_option_1_angle_from_ball", "top_option_2_angle_from_ball", "top_option_3_angle_from_ball"}), "reason_tag": "ball-relative angle, different ranked option"},
]

FAMILY_GROUPS = {"active": FAMILY_GROUPS_ACTIVE, "passive": FAMILY_GROUPS_PASSIVE}

# Part B (prompt 7/7): verdicts resolved by checking real data, not guessed.
# Both marginally exceed Type 5's 0.85 relabeling threshold on a strict
# reading but were confirmed NOT to be relabeling/duplication once actually
# checked -- documented here so they aren't re-litigated by a future
# threshold-only pass.
PART_B_RESOLVED_PAIRS = {
    frozenset({"period", "match_time_seconds"}): (
        "keep_both",
        "Checked directly: match_time_seconds is a cumulative match clock (second half continues from ~2700s, "
        "doesn't reset to 0), but 9,652 rows (17% of the active dataset) sit in a genuine overlap window "
        "(2,702s-3,537s) where both period=1 (first-half stoppage running long) and period=2 (early second half) "
        "occur. period disambiguates information match_time_seconds cannot recover alone for that 17% -- not a "
        "structural duplicate, despite eta=0.8658 exceeding the 0.85 relabeling threshold.",
    ),
    frozenset({"phase_label", "distance_to_attacking_box"}): (
        "keep_both",
        "eta=0.8531, marginally over the Type-5 relabeling threshold on a strict reading, but qualitatively "
        "identical to the rest of the phase_label family that resolved as keep-both (tactical label vs raw "
        "geometry) -- treated as part of that same verdict, not a separate human call.",
    ),
}
