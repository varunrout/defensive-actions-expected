"""Coordinate-frame bug: per-reference-model importance-weighted exposure
(Prompt 84, Task 3). Read-only -- uses each of the 6 already-promoted
reference models' own already-computed importance/coefficient artifact.
No retraining.

Task 2's frame-dependent feature classification (hardcoded below, one line
of reasoning per feature) is applied to each model's own full importance
list, aggregated back to the original locked feature (from
src/eda/feature_config.py's ACTIVE/PASSIVE lists) via longest-categorical-
prefix matching -- the same aggregation method this project's own promotion
audits (validate_x1c_promotion.py, validate_y1c_promotion.py) already use.

Classification principle (applied consistently, not per-feature guesswork):
  - A feature that stores or is derived from a RAW coordinate value, or that
    references a FIXED external pitch point/boundary (attacking goal at
    (120,40), box edge at x=102/18), or that is a SIGNED direction-dependent
    quantity (a signed gap/dx/dy, a goal-side inequality, an absolute
    bearing angle) is FRAME-DEPENDENT: its value is wrong for whichever rows
    are frame-inconsistent.
  - A feature that is a purely RELATIVE, isometry-invariant quantity between
    simultaneously-flipped points (a Euclidean distance or magnitude, a
    count of players within a radius, a spread/std-dev of positions, a
    count-ratio) is FRAME-INDEPENDENT even under the bug, because the same
    180-degree point-reflection is applied to every spatial input in that
    row at once (ball_x/ball_y AND the freeze-frame player positions), and
    a point-reflection through the pitch centre is a distance- and
    ratio-preserving isometry. Two features are also invariant for a
    non-obvious reason and are called out explicitly:
      - `is_wide_lane` (y <= 12 or y >= 68): symmetric about y=40, so the
        rule evaluates identically whether y or 80-y is supplied.
      - `top_option_*_distance_from_ball` (hypot(dx, dy)): dx and dy both
        flip sign together under the bug, and hypot(-dx, -dy) == hypot(dx, dy).
  - A non-spatial feature (categorical event/role/period label not derived
    from a coordinate threshold, a StatsBomb-provided qualifier, a pure
    count, a temporal quantity) is FRAME-INDEPENDENT trivially.

Usage:
    .venv/Scripts/python.exe scripts/analysis/coordinate_frame_model_exposure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
OUT_PATH = REPO_ROOT / "outputs" / "models" / "validation" / "coordinate_frame_model_exposure.json"

from eda.feature_config import ACTIVE, PASSIVE  # noqa: E402

ACTIVE_CATEGORICAL = list(ACTIVE["categorical"])
PASSIVE_CATEGORICAL = list(PASSIVE["categorical"])

# --- Task 2 classification: ACTIVE's 34 locked features -------------------
ACTIVE_DEP = {
    "phase_label": "rule-based label assumes possession team attacks toward x=120 (the bug's own motivating example)",
    "phase_label_prev_event": "same rule-based labeller, previous event",
    "is_in_defending_box": "fixed zone boundary near x=0/18",
    "is_in_attacking_box": "fixed zone boundary near x=102/120",
    "phase_changed_since_prev_event": "derived from phase_label, which is frame-dependent",
    "angle_to_attacking_goal": "angle to the FIXED point (120,40)",
    "distance_to_attacking_box": "distance to a fixed box boundary near x=102",
    "attacking_goal_centrality": "based on |y-40| assuming goal is at fixed (120,40)",
    "defender_attacker_gap_x": "signed x-difference between centroids -- flips sign under the bug",
    "defender_attacker_gap_y": "signed y-difference between centroids -- flips sign under the bug",
    "defenders_between_ball_and_attacking_goal": "counts defenders with px >= ball_x -- direction-dependent",
}
ACTIVE_INV = {
    "position": "StatsBomb player role label, not coordinate-derived",
    "event_type": "categorical event-type label, not coordinate-derived",
    "play_pattern": "categorical play-pattern label, not coordinate-derived",
    "period": "non-spatial",
    "counterpress": "StatsBomb-provided event qualifier, not coordinate-derived",
    "action_was_under_opponent_possession": "possession-identity flag, not coordinate-derived",
    "action_retained_defensive_team_control": "possession-identity flag, not coordinate-derived",
    "has_previous_event": "existence flag, not coordinate-derived",
    "has_visible_attacker": "existence flag, not coordinate-derived",
    "has_visible_defender": "existence flag, not coordinate-derived",
    "match_time_seconds": "temporal",
    "possession_elapsed_seconds": "temporal",
    "nearest_attacker_distance": "Euclidean distance -- isometry-invariant",
    "nearest_defender_distance": "Euclidean distance -- isometry-invariant",
    "attacker_spread": "dispersion among simultaneously-flipped points -- isometry-invariant",
    "defender_spread": "dispersion among simultaneously-flipped points -- isometry-invariant",
    "attacker_defender_ratio": "ratio of player counts, not coordinate-derived",
    "visible_attacker_count": "pure count",
    "visible_defender_count": "pure count",
    "attackers_within_5m": "count within radius -- distance-based, isometry-invariant",
    "defenders_within_5m": "count within radius -- distance-based, isometry-invariant",
    "attackers_within_10m": "count within radius -- distance-based, isometry-invariant",
    "defenders_within_10m": "count within radius -- distance-based, isometry-invariant",
}
assert set(ACTIVE_DEP) | set(ACTIVE_INV) == set(ACTIVE["categorical"]) | set(ACTIVE["boolean"]) | set(ACTIVE["continuous"]) | set(ACTIVE["discrete"]), \
    "ACTIVE classification must cover every locked feature exactly once"

# --- Task 2 classification: PASSIVE's 38 locked features -------------------
PASSIVE_DEP = {
    "phase_label": "rule-based label, same frame-dependence as active leg",
    "defender_functional_role": "relative x-ranking among defenders (e.g. 'last_line' = deepest) assumes a fixed attack direction",
    "is_in_defending_box": "fixed zone boundary",
    "is_in_attacking_box": "fixed zone boundary",
    "is_goal_side_of_nearest_attacker": "signed x-comparison (defender_x < nearest_x) -- flips under the bug",
    "defender_x": "raw coordinate",
    "defender_y": "raw coordinate",
    "ball_x": "raw coordinate",
    "ball_y": "raw coordinate",
    "angle_to_attacking_goal": "angle to the FIXED point (120,40)",
    "attacking_goal_centrality": "based on |y-40| assuming goal is at fixed (120,40)",
    "top_option_1_threat_score": "includes goal_proximity = 1/(1+distance_to_attacking_goal), a fixed-point distance",
    "top_option_2_threat_score": "same as option 1",
    "top_option_3_threat_score": "same as option 1",
    "top_option_1_dx": "signed x-difference (option location minus ball location) -- flips sign under the bug",
    "top_option_1_dy": "signed y-difference -- flips sign under the bug",
    "top_option_2_dx": "signed x-difference -- flips sign",
    "top_option_2_dy": "signed y-difference -- flips sign",
    "top_option_3_dx": "signed x-difference -- flips sign",
    "top_option_3_dy": "signed y-difference -- flips sign",
    "top_option_1_angle_from_ball": "absolute bearing (atan2(dy,dx)) -- forward/backward interpretation flips under the bug",
    "top_option_2_angle_from_ball": "absolute bearing -- interpretation flips",
    "top_option_3_angle_from_ball": "absolute bearing -- interpretation flips",
}
PASSIVE_INV = {
    "on_ball_event_type": "categorical event-type label, not coordinate-derived",
    "period": "non-spatial",
    "is_wide_lane": "y<=12 or y>=68 is symmetric about y=40 -- invariant to the y-flip (worked derivation in module docstring)",
    "has_option_2": "existence flag, not coordinate-derived",
    "has_option_3": "existence flag, not coordinate-derived",
    "marking_tightness": "Euclidean distance (defender to nearest attacker) -- isometry-invariant",
    "engagement_distance_to_carrier": "Euclidean distance -- isometry-invariant",
    "lane_screening_score_option_1": "relative-distance-based screening score between simultaneously-flipped points -- isometry-invariant",
    "lane_screening_score_option_2": "same as option 1",
    "lane_screening_score_option_3": "same as option 1",
    "top_option_1_distance_from_ball": "hypot(dx,dy) magnitude -- both components flip sign together, magnitude invariant",
    "top_option_2_distance_from_ball": "same as option 1",
    "top_option_3_distance_from_ball": "same as option 1",
    "overload_score": "count-based tally, not coordinate-derived",
    "defender_slot_index": "structural ordering index, not a spatial value",
}
assert set(PASSIVE_DEP) | set(PASSIVE_INV) == set(PASSIVE["categorical"]) | set(PASSIVE["boolean"]) | set(PASSIVE["continuous"]) | set(PASSIVE["discrete"]), \
    "PASSIVE classification must cover every locked feature exactly once"


def base_feature(name: str, categorical_cols: list[str]) -> str:
    """Map a (possibly one-hot-expanded) importance-list feature name back to
    its original locked feature, longest-categorical-prefix-first -- the
    same disambiguation this project's own promotion audits use."""
    for c in sorted(categorical_cols, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def classify(name: str, categorical_cols: list[str], dep: dict, inv: dict) -> bool | None:
    base = base_feature(name, categorical_cols)
    if base in dep:
        return True
    if base in inv:
        return False
    return None  # unclassified -- should not happen if assertions above passed


def summarize(entries: list[tuple[str, float]], categorical_cols: list[str], dep: dict, inv: dict, weight_label: str) -> dict:
    total = sum(v for _, v in entries)
    dep_total = 0.0
    dep_rows = []
    unclassified = []
    for name, val in entries:
        is_dep = classify(name, categorical_cols, dep, inv)
        if is_dep is None:
            unclassified.append(name)
            continue
        if is_dep:
            dep_total += val
            dep_rows.append((name, val))
    dep_rows.sort(key=lambda r: -r[1])
    return {
        "weight_metric": weight_label,
        "total_weight": total,
        "frame_dependent_weight": dep_total,
        "share_frame_dependent": round(dep_total / total, 4) if total else None,
        "top_frame_dependent_features": [{"feature": n, weight_label: v, "share_of_total": round(v / total, 4)}
                                          for n, v in dep_rows[:5]],
        "unclassified_features_found": unclassified,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    # active-binary: v1e_gradient_boosting_calibrated (importance from the
    # pre-calibration fit, unchanged by the isotonic calibration wrapper)
    v1e = json.loads((REPO_ROOT / "outputs/models/classification/v1e_gradient_boosting.json").read_text())
    entries = [(r["feature"], r["gain_importance"]) for r in v1e["full_gain_importance"]]
    result["active_binary_v1e_gradient_boosting_calibrated"] = summarize(entries, ACTIVE_CATEGORICAL, ACTIVE_DEP, ACTIVE_INV, "gain_importance")

    # passive-binary: p1e_gradient_boosting_calibrated
    p1e = json.loads((REPO_ROOT / "outputs/models/classification/p1e_gradient_boosting.json").read_text())
    entries = [(r["feature"], r["gain_importance"]) for r in p1e["full_gain_importance"]]
    result["passive_binary_p1e_gradient_boosting_calibrated"] = summarize(entries, PASSIVE_CATEGORICAL, PASSIVE_DEP, PASSIVE_INV, "gain_importance")

    # active-continuous: c1d_random_forest
    c1d = json.loads((REPO_ROOT / "outputs/models/regression/c1d_random_forest.json").read_text())
    entries = [(r["feature"], r["gini_importance"]) for r in c1d["full_gini_importance"]]
    result["active_continuous_c1d_random_forest"] = summarize(entries, ACTIVE_CATEGORICAL, ACTIVE_DEP, ACTIVE_INV, "gini_importance")

    # passive-continuous: d1_lognormal_glm (abs coefficient as weight)
    d1 = json.loads((REPO_ROOT / "outputs/models/regression/d1_lognormal_glm.json").read_text())
    entries = [(r["feature"], r["abs_coef"]) for r in d1["coefficients"]]
    result["passive_continuous_d1_lognormal_glm"] = summarize(entries, PASSIVE_CATEGORICAL, PASSIVE_DEP, PASSIVE_INV, "abs_coef")

    # active-xT: x1c_random_forest -- regression head (reg_feature_importances_gini,
    # a dict) is the leg's own regression-quantity head; classifier head
    # (clf_feature_importances_gini) reported too since it also feeds the
    # combined P(nonzero)*E[delta|nonzero] prediction.
    x1c = json.loads((REPO_ROOT / "outputs/models/regression/x1c_random_forest.json").read_text())
    reg_entries = list(x1c["reg_feature_importances_gini"].items())
    clf_entries = list(x1c["clf_feature_importances_gini"].items())
    result["active_xt_x1c_random_forest_regression_head"] = summarize(reg_entries, ACTIVE_CATEGORICAL, ACTIVE_DEP, ACTIVE_INV, "gini_importance")
    result["active_xt_x1c_random_forest_classifier_head"] = summarize(clf_entries, ACTIVE_CATEGORICAL, ACTIVE_DEP, ACTIVE_INV, "gini_importance")

    # passive-xT: y1c_random_forest -- regression head (list) + classifier head (dict)
    y1c = json.loads((REPO_ROOT / "outputs/models/regression/y1c_random_forest.json").read_text())
    reg_entries = [(r["feature"], r["gini_importance"]) for r in y1c["reg_full_gini_importance"]]
    clf_entries = list(y1c["clf_feature_importances_gini"].items())
    result["passive_xt_y1c_random_forest_regression_head"] = summarize(reg_entries, PASSIVE_CATEGORICAL, PASSIVE_DEP, PASSIVE_INV, "gini_importance")
    result["passive_xt_y1c_random_forest_classifier_head"] = summarize(clf_entries, PASSIVE_CATEGORICAL, PASSIVE_DEP, PASSIVE_INV, "gini_importance")

    OUT_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    for k, v in result.items():
        print(f"{k}: share_frame_dependent={v['share_frame_dependent']} "
              f"top={[t['feature'] for t in v['top_frame_dependent_features'][:3]]} "
              f"unclassified={v['unclassified_features_found']}")
    print(f"\n[write] {OUT_PATH}")


if __name__ == "__main__":
    main()
