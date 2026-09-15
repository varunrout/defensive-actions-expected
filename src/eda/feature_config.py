"""Hardcoded feature classification for the EDA report generator.

Column types are declared explicitly per dataset rather than re-derived from
dtypes, because dtype alone can't distinguish e.g. a boolean stored as int64
from a genuine discrete count, or a categorical stored as int64 (period).
"""

from __future__ import annotations

import pandas as pd

TARGET_COL = "target_future_shot_10s"
SECONDARY_TARGET_COL = "target_future_xg_10s"

SMALL_N_THRESHOLD = 500
DISCRETE_TOP_N = 20
CONTINUOUS_BIN_COUNT = 24
CONTINUOUS_PERCENTILE_CAP = 99

# Columns excluded from every report, with the reason surfaced in the footer.
EXCLUDED_COLUMNS_COMMON = {
    "visibility_quality_band": "camera/360-tracking-coverage metadata, not a football signal (confound: crowded/dangerous zones are both harder to fully capture on camera and where shots cluster)",
    "ball_inside_visible_area": "camera/360-tracking-coverage metadata, not a football signal",
    "action_inside_visible_area": "camera/360-tracking-coverage metadata, not a football signal",
    "local_5m_region_fully_visible": "camera/360-tracking-coverage metadata, not a football signal",
    "local_10m_region_fully_visible": "camera/360-tracking-coverage metadata, not a football signal",
    "visibility_limited": "camera/360-tracking-coverage metadata, not a football signal",
    "visible_area_polygon_area": "camera/360-tracking-coverage metadata, not a football signal",
    "visible_area_fraction_of_pitch": "camera/360-tracking-coverage metadata, not a football signal",
    "freeze_frame_count": "data-collection artifact, not a defensive feature",
    "carrier_x": "verified 100%-duplicate of ball_x (already dropped from the passive parquet)",
    "carrier_y": "verified 100%-duplicate of ball_y (already dropped from the passive parquet)",
}

EXCLUDED_COLUMNS_ACTIVE = {
    **EXCLUDED_COLUMNS_COMMON,
    "action_x": "verified 100%-duplicate of ball_x as of this pass -- known unresolved data issue",
    "action_y": "verified 100%-duplicate of ball_y as of this pass -- known unresolved data issue",
    "action_changed_possession": "structural zero (0.00% shot rate by target-window definition), not a genuine finding",
    "action_ended_possession": "structural zero (0.00% shot rate by target-window definition), not a genuine finding",
    "action_won_possession": "structural zero (0.00% shot rate by target-window definition), not a genuine finding",
}

EXCLUDED_COLUMNS_PASSIVE = {
    **EXCLUDED_COLUMNS_COMMON,
    # Prompt 10 (leakage audit) -- this is a LEAKAGE exclusion, not a
    # correlation-tier verdict (hence here, not REDUNDANCY_DROPPED_PASSIVE).
    # has_screened_outcome is False only when screened_option_was_avoided
    # could not be computed (end of period/match, no next event in the data
    # to compare against) -- it is a proxy for whether target_future_shot_10s's
    # own 10-second measurement window was truncated, not a defensive signal.
    # chi2 test of independence vs target_future_shot_10s: chi2=136.5,
    # p=1.54e-31, dof=1 (reports/eda/LEAKAGE_AUDIT.json). has_screened_outcome=False
    # (2,581 rows) shows a 0.50% shot rate vs 5.97% for True -- a ~11.9x gap driven
    # by target-window truncation (a false negative -- no time left for a shot --
    # is far more likely than a genuine successful defensive outcome when the
    # match/period ends early), not by anything defensively meaningful. A model
    # fed this feature would learn "predict near-zero shot probability whenever
    # the target's own measurement window was cut short" -- circular with the
    # target's construction, not causal, not useful, and it would look like a
    # strong feature in offline evaluation while encoding nothing about defending.
    "has_screened_outcome": (
        "Leakage exclusion (not a correlation-tier verdict): censoring-mechanism proxy for target_future_shot_10s's "
        "own 10s window being truncated at end-of-period/match, not defensive signal -- chi2=136.5, p=1.54e-31 vs "
        "target_future_shot_10s (0.50% shot rate when False [n=2,581] vs 5.97% when True); see reports/eda/LEAKAGE_AUDIT.json"
    ),
}

# Redundancy-driven removals from reports/eda/CORRELATION_ANALYSIS.json (DROP
# tier) and reports/eda/REVIEW_ANALYSIS.json (Type 1/2 drop verdicts) --
# regenerate both before trusting these against a changed feature build. Each
# entry names which surviving column/pair drove the removal, for traceability.
#
# NOTE on passive `is_wide_lane`: the Cowork-pass prompt this pipeline
# replicates listed `is_wide_lane` as a passive-side drop too, but the actual
# computed correlations in this run do not support it -- its correlation
# with attacking_goal_centrality/distance_to_center_line tops out at 0.65
# (below the 0.75 Type-1 structural threshold) and with is_central_lane at
# -0.44 (below the 0.5 Type-2 threshold). Per the instruction to confirm
# against the actual JSON output rather than trust a possibly-stale list,
# `is_wide_lane` is KEPT on the passive side. If a future correlation re-run
# shows it crossing a threshold, drop it then, from that evidence.
REDUNDANCY_DROPPED_ACTIVE = {
    "is_central_lane": "Type 1 REVIEW resolution: point-biserial r=0.854 vs attacking_goal_centrality (kept)",
    "is_wide_lane": "Type 1 REVIEW resolution: point-biserial r=-0.785 vs attacking_goal_centrality (kept)",
    "is_high_zone": "Type 2 REVIEW resolution: near-total subset of is_in_attacking_box (kept, higher lift +11.99pp vs +7.22pp)",
    "is_deep_zone": "Type 2 REVIEW resolution: near-total subset of is_in_defending_box (kept, higher lift +12.04pp vs +7.22pp)",
    "position_group": "CORRELATION_ANALYSIS DROP tier: Cramer's V=1.0 vs position (kept, more granular)",
    "action_family": "CORRELATION_ANALYSIS DROP tier: Cramer's V=1.0 vs event_type (kept, more granular)",
    "distance_to_center_line": "CORRELATION_ANALYSIS DROP tier: Spearman r=-1.0 vs attacking_goal_centrality (kept)",
    # COLLAPSE-tier cluster resolutions (prompt 5/5) -- see REVIEW_METHODOLOGY.html
    # and reports/eda/CORRELATION_ANALYSIS.json for the r/eta values.
    "distance_to_attacking_goal": "Cluster 1 (goal-proximity, drop-to-one): mutually r/eta >= 0.94 with distance_to_attacking_box, distance_to_defending_goal/box, action_zone -- kept distance_to_attacking_box (is_in_attacking_box lift +11.99pp is the most predictive single cut)",
    "distance_to_defending_goal": "Cluster 1 (goal-proximity, drop-to-one): see distance_to_attacking_goal",
    "distance_to_defending_box": "Cluster 1 (goal-proximity, drop-to-one): see distance_to_attacking_goal",
    "action_zone": "Cluster 1 (goal-proximity, drop-to-one): see distance_to_attacking_goal",
    "attacker_centroid_x": "Cluster 2 (attacker/defender centroid, MERGE not drop): r=0.984 with defender_centroid_x -- replaced by defender_attacker_gap_x/y (defensive compactness relative to attacking shape), not simply dropped, since the two centroids describe different teams' shapes",
    "attacker_centroid_y": "Cluster 2 (attacker/defender centroid, MERGE not drop): see attacker_centroid_x",
    "defender_centroid_x": "Cluster 2 (attacker/defender centroid, MERGE not drop): see attacker_centroid_x",
    "defender_centroid_y": "Cluster 2 (attacker/defender centroid, MERGE not drop): see attacker_centroid_x",
    "events_elapsed_in_possession": "Cluster 3 (possession clock, drop-to-one): mutually r=0.90-0.95 with possession_elapsed_seconds/phase_transitions_observed_so_far -- kept possession_elapsed_seconds (most granular, continuous)",
    "phase_transitions_observed_so_far": "Cluster 3 (possession clock, drop-to-one): see events_elapsed_in_possession",
    # Prompt 9/9, VIF (multicollinearity) analysis -- joint multicollinearity
    # invisible to pairwise correlation, caught by VIF, not a COLLAPSE/REVIEW
    # tier verdict. local_numerical_balance_5m is a near-exact linear
    # combination of two columns already kept (attackers_within_5m -
    # defenders_within_5m, matching on effectively all rows in this build);
    # local_numerical_balance_10m likewise for the 10m pair. Pairwise
    # correlation never surfaced this (max pairwise r among the six affected
    # columns was well under 0.90) because the redundancy is a joint linear
    # dependency across three columns at once. VIF for these six columns was
    # effectively infinite (correlation-matrix condition number ~1e14-1e15,
    # SVD failed to converge) before the drop -- see reports/eda/VIF_ANALYSIS.json.
    # Dropping local_numerical_balance_5m/10m loses no information: both are
    # exactly recoverable from attackers_within_Nm - defenders_within_Nm,
    # which stay in the candidate list.
    "local_numerical_balance_5m": "VIF analysis: near-exact linear combination of attackers_within_5m - defenders_within_5m (both kept); joint multicollinearity invisible to pairwise correlation, caught by VIF, not a COLLAPSE/REVIEW tier verdict -- see reports/eda/VIF_ANALYSIS.json",
    "local_numerical_balance_10m": "VIF analysis: near-exact linear combination of attackers_within_10m - defenders_within_10m (both kept); joint multicollinearity invisible to pairwise correlation, caught by VIF, not a COLLAPSE/REVIEW tier verdict -- see reports/eda/VIF_ANALYSIS.json",
}

REDUNDANCY_DROPPED_PASSIVE = {
    "is_central_lane": "Type 1 REVIEW resolution: point-biserial r=0.849 vs attacking_goal_centrality (kept)",
    "is_high_zone": "Type 2 REVIEW resolution: near-total subset of is_in_attacking_box (kept, higher lift +11.85pp vs +6.41pp)",
    "is_deep_zone": "Type 2 REVIEW resolution: near-total subset of is_in_defending_box (kept, higher lift +12.31pp vs +7.02pp)",
    "screens_top_option": "Type 1 REVIEW resolution: point-biserial r=0.832 vs lane_screening_score_option_1 (kept)",
    "zone_defensive_value": "CORRELATION_ANALYSIS DROP tier: Spearman r=-1.0 vs distance_to_defending_goal (kept)",
    "distance_to_center_line": "CORRELATION_ANALYSIS DROP tier: Spearman r=-1.0 vs attacking_goal_centrality (kept)",
    # Cluster 4 (prompt 5/5): defender-position, drop-to-one.
    "distance_to_attacking_box": "Cluster 4 (defender-position, drop-to-one): mutually r/eta >= 0.92 with defender_x and the rest of this cluster -- kept defender_x (the most primitive measurement; it's upstream of every derived distance/zone column here, unlike the active side where the raw coordinate isn't itself a kept feature)",
    "distance_to_defending_box": "Cluster 4 (defender-position, drop-to-one): see distance_to_attacking_box",
    "distance_to_attacking_goal": "Cluster 4 (defender-position, drop-to-one): see distance_to_attacking_box",
    "distance_to_defending_goal": "Cluster 4 (defender-position, drop-to-one): see distance_to_attacking_box",
    "defender_zone": "Cluster 4 (defender-position, drop-to-one): see distance_to_attacking_box",
    # Prompt 6/6: transition period over for Part A (prompt 4/4). The raw
    # absolute target_x/y columns were deliberately kept alongside the
    # ball-relative replacements until those were confirmed working (they
    # were -- see reports/eda/CORRELATION_ANALYSIS.json). Coordinate-frame
    # cleanup, not a redundancy judgment call: no information is lost, since
    # dx/dy/distance_from_ball/angle_from_ball are strictly more useful for
    # the same signal (they generalise across the pitch; raw coordinates
    # don't). The underlying target_x/y computation stays in
    # passive_defense.py (dx = target_x - ball_x needs it as an intermediate
    # value) -- only removed from the candidate feature list here.
    "top_option_1_target_x": "Raw absolute coordinate, superseded by ball-relative top_option_1_dx/dy/distance_from_ball/angle_from_ball (prompt 4/4 Part A); correlated 0.68-0.88 with ball_x/defender_x/threat_score as a coordinate-frame artefact, not genuine redundancy -- see prompt 6/6",
    "top_option_1_target_y": "Raw absolute coordinate, superseded by ball-relative top_option_1_dx/dy/distance_from_ball/angle_from_ball (prompt 4/4 Part A); correlated 0.68-0.88 with ball_x/defender_x/threat_score as a coordinate-frame artefact, not genuine redundancy -- see prompt 6/6",
    "top_option_2_target_x": "Raw absolute coordinate, superseded by ball-relative top_option_2_dx/dy/distance_from_ball/angle_from_ball (prompt 4/4 Part A); correlated 0.68-0.88 with ball_x/defender_x/threat_score as a coordinate-frame artefact, not genuine redundancy -- see prompt 6/6",
    "top_option_2_target_y": "Raw absolute coordinate, superseded by ball-relative top_option_2_dx/dy/distance_from_ball/angle_from_ball (prompt 4/4 Part A); correlated 0.68-0.88 with ball_x/defender_x/threat_score as a coordinate-frame artefact, not genuine redundancy -- see prompt 6/6",
    "top_option_3_target_x": "Raw absolute coordinate, superseded by ball-relative top_option_3_dx/dy/distance_from_ball/angle_from_ball (prompt 4/4 Part A); correlated 0.68-0.88 with ball_x/defender_x/threat_score as a coordinate-frame artefact, not genuine redundancy -- see prompt 6/6",
    "top_option_3_target_y": "Raw absolute coordinate, superseded by ball-relative top_option_3_dx/dy/distance_from_ball/angle_from_ball (prompt 4/4 Part A); correlated 0.68-0.88 with ball_x/defender_x/threat_score as a coordinate-frame artefact, not genuine redundancy -- see prompt 6/6",
}

# Part B (deferred): whether to collapse top_option_2/3_* into aggregate
# "backup option" features (max threat score, spread vs rank 1). Defaults to
# False -- raw ranked columns stay in the feature set until a baseline
# model's feature importance for top_option_2_*/top_option_3_* says whether
# they're pulling their own weight. See add_option_rank_aggregates below,
# which is written but NOT wired into any default pipeline.
PASSIVE_COLLAPSE_OPTION_RANKS = False


def add_option_rank_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse top_option_2/3_* into aggregate "backup option" features.

    Scaffolded per the Part B structural-redesign deferral in
    reports/eda/REVIEW_METHODOLOGY.html -- NOT called by any default pipeline.
    Only flip PASSIVE_COLLAPSE_OPTION_RANKS to True, and start calling this,
    once a baseline model's feature importance for top_option_2_*/
    top_option_3_* shows they aren't pulling their own weight individually.
    """
    out = df.copy()
    threat_cols = [f"top_option_{n}_threat_score" for n in (2, 3)]
    out["backup_max_threat_score"] = out[threat_cols].max(axis=1)
    out["threat_score_spread"] = out["top_option_1_threat_score"] - out[threat_cols].max(axis=1)
    screen_cols = [f"lane_screening_score_option_{n}" for n in (2, 3)]
    out["backup_max_lane_screening_score"] = out[screen_cols].max(axis=1)
    return out


ACTIVE = {
    "key": "active",
    "label": "Active Defensive Actions",
    "parquet_path": "data/features/player_defensive_actions.parquet",
    "target_col": TARGET_COL,
    "row_description": "one row per actual defensive action",
    "categorical": [
        "phase_label", "position", "event_type",
        "play_pattern", "phase_label_prev_event", "period",
    ],
    "boolean": [
        "is_in_defending_box", "is_in_attacking_box", "counterpress", "phase_changed_since_prev_event",
        "action_was_under_opponent_possession", "action_retained_defensive_team_control",
        "has_previous_event", "has_visible_attacker", "has_visible_defender",
    ],
    "continuous": [
        "match_time_seconds", "possession_elapsed_seconds", "angle_to_attacking_goal",
        "distance_to_attacking_box", "attacking_goal_centrality",
        "nearest_attacker_distance", "nearest_defender_distance",
        "defender_attacker_gap_x", "defender_attacker_gap_y",
        "attacker_spread", "defender_spread", "attacker_defender_ratio",
    ],
    "discrete": [
        "visible_attacker_count", "visible_defender_count", "attackers_within_5m",
        "defenders_within_5m", "attackers_within_10m", "defenders_within_10m",
        "defenders_between_ball_and_attacking_goal",
    ],
    "excluded": {**EXCLUDED_COLUMNS_ACTIVE, **REDUNDANCY_DROPPED_ACTIVE},
    "duplicate_check": {
        "pair": ("action_x", "ball_x"),
        "pair_y": ("action_y", "ball_y"),
    },
}

PASSIVE = {
    "key": "passive",
    "label": "Passive / Off-Ball Defensive Positioning",
    "parquet_path": "data/features/passive_defense.parquet",
    "target_col": TARGET_COL,
    "row_description": "one row per visible defender-slot per on-ball attacking event",
    "categorical": [
        "on_ball_event_type", "phase_label", "defender_functional_role", "period",
    ],
    "boolean": [
        "is_in_defending_box", "is_in_attacking_box",
        "is_wide_lane", "has_option_2", "has_option_3",
        "is_goal_side_of_nearest_attacker",
    ],
    "continuous": [
        "defender_x", "defender_y", "ball_x", "ball_y",
        "angle_to_attacking_goal", "attacking_goal_centrality",
        "marking_tightness", "engagement_distance_to_carrier",
        # Cluster 5 (prompt 5/5) update, prompt 6/6: this cluster was
        # originally two pairs -- top_option_1_target_x <-> threat_score
        # (r=0.908) and top_option_2_threat_score <-> top_option_3_threat_score
        # (r=0.902) -- deliberately left unresolved because target_x/y were
        # about to be superseded by the ball-relative transform (prompt 4/4
        # Part A). Now that target_x/y are dropped from this list (prompt
        # 6/6, coordinate-frame cleanup, see REDUNDANCY_DROPPED_PASSIVE),
        # the target_x<->threat_score half of the cluster is resolved BY that
        # drop, not by a redundancy judgment call -- it no longer exists as
        # a pair once target_x is gone. Cluster 5 now reduces to just
        # top_option_2_threat_score <-> top_option_3_threat_score (r=0.902),
        # which is STILL unresolved and still gated behind the same Part B
        # baseline-model evidence (PASSIVE_COLLAPSE_OPTION_RANKS) as before --
        # don't let this comment's narrowed scope be mistaken for that pair
        # being resolved too.
        "top_option_1_threat_score",
        "lane_screening_score_option_1", "top_option_2_threat_score", "lane_screening_score_option_2",
        "top_option_3_threat_score",
        "lane_screening_score_option_3",
        "top_option_1_dx", "top_option_1_dy", "top_option_1_distance_from_ball", "top_option_1_angle_from_ball",
        "top_option_2_dx", "top_option_2_dy", "top_option_2_distance_from_ball", "top_option_2_angle_from_ball",
        "top_option_3_dx", "top_option_3_dy", "top_option_3_distance_from_ball", "top_option_3_angle_from_ball",
    ],
    "discrete": ["overload_score", "defender_slot_index"],
    "excluded": {**EXCLUDED_COLUMNS_PASSIVE, **REDUNDANCY_DROPPED_PASSIVE},
    "duplicate_check": None,
}

DATASETS = {"active": ACTIVE, "passive": PASSIVE}
