# Player Defensive Model (360-only)

This project is now building a first player-level defensive model from `events_with_targets.parquet`.

## Core idea

A player contributes to defending by:
- positioning to reduce central access,
- applying pressure near the ball,
- supporting teammates with compact shape,
- and helping the team prevent a shot as the possession evolves.

Because the 360 freeze-frame does **not** expose identities for all visible defenders, the model starts with **player-actor defensive actions** and **teammate interaction features** around that action.

## Output dataset

`data/features/player_defensive_actions.parquet`

One row per defensive action by an identifiable player, with:
- player / team / possession identifiers
- phase and possession context
- action location and goal-distance features
- 360 support features around the action
- the future target `target_shot_in_10s`

## Feature groups

### 1. Player action context
- `event_type`
- `action_family`
- `position_group`
- `phase_label`
- `possession_progress_ratio`
- `seconds_since_possession_start`

### 2. Player geometry
- `action_x`, `action_y`
- `action_zone`
- `distance_to_left_goal`
- `distance_to_right_goal`
- `nearest_goal_distance`
- `distance_to_center_line`
- `is_central_lane`, `is_wide_lane`

### 3. Teammate interaction / support
- `freeze_teammate_count`
- `freeze_opponent_count`
- `local_numerical_balance_5m`
- `local_numerical_balance_10m`
- `attackers_within_5m`
- `attackers_within_10m`
- centroid and spread features for teammates/opponents

## How we will analyze features

1. Start with phase-by-phase summaries.
2. Check which geometry/support features separate shot vs no-shot actions.
3. Compare players by role group, not just raw counts.
4. Keep features that are:
   - stable,
   - interpretable,
   - and predictive.

## Model selection

For the first version, use:
- **logistic regression** as the interpretability baseline,
- then a **tree-based model** later if we need better non-linear fit.

Why:
- the first model must be explainable,
- teammate interaction is likely non-linear,
- phase interactions matter a lot,
- and logistic coefficients are easier to interpret.

## Teammate interaction

Teammate support is not a separate identity-tracking model yet.
Instead, it enters as context:
- how many teammates are near the action,
- how compact the support shape is,
- and whether opponents outnumber that support nearby.

That means teammate interaction is treated as a **feature family**, not a separate sub-model for now.

## Next scripts

- `scripts/build_player_defense_dataset.py`
- `scripts/profile_player_defense_features.py`
