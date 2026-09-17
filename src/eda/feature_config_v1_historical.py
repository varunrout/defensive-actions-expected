"""Reconstructed V1 candidate feature lists -- what CORRELATION_ANALYSIS.json
actually looked like at stage 01, before ANY redundancy-driven drop had
landed (51 active / 44 passive). This is the historical baseline the
eda_pipeline.html audit trail describes; it does not exist as a live JSON
snapshot anywhere in the repo -- CORRELATION_ANALYSIS.json itself was
regenerated at stage 14 against the FINAL locked lists, overwriting whatever
it originally held.

Built as: feature_config.py's live ACTIVE/PASSIVE lists, minus the columns
engineered *during* the reduction (defender_attacker_gap_x/y, and the
passive ball-relative top_option_n_dx/dy/distance_from_ball/angle_from_ball,
none of which existed at stage 01), plus every column
feature_config.REDUNDANCY_DROPPED_ACTIVE/PASSIVE later dropped, plus
passive's has_screened_outcome (dropped at stage 10 for leakage, so it was
still a candidate at stage 01). Column existence in the current parquet
files is asserted at import time, not assumed.

Usage: imported by generate_correlation_analysis_v1_historical.py, never by
the live pipeline scripts (generate_correlation_analysis.py etc. keep
reading feature_config.py's current lists, unchanged).
"""

from __future__ import annotations

from src.eda.feature_config import ACTIVE, PASSIVE, TARGET_COL

# Columns engineered *by* the reduction itself -- didn't exist as candidates
# at stage 01, so they're excluded from the historical reconstruction even
# though they're in feature_config.py's current lists.
_ACTIVE_ENGINEERED_DURING_REDUCTION = {"defender_attacker_gap_x", "defender_attacker_gap_y"}
_PASSIVE_ENGINEERED_DURING_REDUCTION = {
    f"top_option_{n}_{suffix}" for n in (1, 2, 3) for suffix in ("dx", "dy", "distance_from_ball", "angle_from_ball")
}

ACTIVE_V1_HISTORICAL = {
    **ACTIVE,
    "label": "Active Defensive Actions (V1 historical reconstruction)",
    "categorical": [c for c in ACTIVE["categorical"]] + ["position_group", "action_family", "action_zone"],
    "boolean": [c for c in ACTIVE["boolean"]] + ["is_central_lane", "is_wide_lane", "is_high_zone", "is_deep_zone"],
    "continuous": (
        [c for c in ACTIVE["continuous"] if c not in _ACTIVE_ENGINEERED_DURING_REDUCTION]
        + [
            "distance_to_center_line", "distance_to_attacking_goal", "distance_to_defending_goal",
            "distance_to_defending_box", "attacker_centroid_x", "attacker_centroid_y",
            "defender_centroid_x", "defender_centroid_y",
        ]
    ),
    "discrete": (
        [c for c in ACTIVE["discrete"]]
        + [
            "events_elapsed_in_possession", "phase_transitions_observed_so_far",
            "local_numerical_balance_5m", "local_numerical_balance_10m",
        ]
    ),
}

PASSIVE_V1_HISTORICAL = {
    **PASSIVE,
    "label": "Passive / Off-Ball Defensive Positioning (V1 historical reconstruction)",
    "categorical": [c for c in PASSIVE["categorical"]] + ["defender_zone"],
    "boolean": (
        [c for c in PASSIVE["boolean"]]
        + ["is_central_lane", "is_high_zone", "is_deep_zone", "screens_top_option", "has_screened_outcome"]
    ),
    "continuous": (
        [c for c in PASSIVE["continuous"] if c not in _PASSIVE_ENGINEERED_DURING_REDUCTION]
        + [
            "zone_defensive_value", "distance_to_center_line", "distance_to_attacking_box",
            "distance_to_defending_box", "distance_to_attacking_goal", "distance_to_defending_goal",
            "top_option_1_target_x", "top_option_1_target_y",
            "top_option_2_target_x", "top_option_2_target_y",
            "top_option_3_target_x", "top_option_3_target_y",
        ]
    ),
    "discrete": [c for c in PASSIVE["discrete"]],
}

# has_screened_outcome was in EXCLUDED_COLUMNS_PASSIVE (a leakage exclusion),
# not REDUNDANCY_DROPPED_PASSIVE, so it must not stay in `excluded` for this
# historical view -- at stage 01 it was a live candidate, not excluded yet.
PASSIVE_V1_HISTORICAL["excluded"] = {
    k: v for k, v in PASSIVE["excluded"].items() if k != "has_screened_outcome"
}

EXPECTED_ACTIVE_COUNT = 51
EXPECTED_PASSIVE_COUNT = 44


def _count(cfg: dict) -> int:
    return len(cfg["categorical"]) + len(cfg["boolean"]) + len(cfg["continuous"]) + len(cfg["discrete"])


_active_n, _passive_n = _count(ACTIVE_V1_HISTORICAL), _count(PASSIVE_V1_HISTORICAL)
if _active_n != EXPECTED_ACTIVE_COUNT or _passive_n != EXPECTED_PASSIVE_COUNT:
    raise AssertionError(
        f"V1 historical reconstruction doesn't match the documented stage-01 baseline: "
        f"active={_active_n} (expected {EXPECTED_ACTIVE_COUNT}), passive={_passive_n} (expected {EXPECTED_PASSIVE_COUNT})"
    )

DATASETS_V1_HISTORICAL = {"active": ACTIVE_V1_HISTORICAL, "passive": PASSIVE_V1_HISTORICAL}
