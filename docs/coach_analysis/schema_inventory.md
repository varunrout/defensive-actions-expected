# Coach analysis schema inventory (local)

Generated from local files on 2026-06-19.

## Inputs inspected

- `data/features/player_defensive_actions.parquet` (56068 rows, 81 cols)
- `outputs/oof/classification_oof.parquet` (140 rows, 14 cols)
- `outputs/oof/regression_oof.parquet` (120 rows, 15 cols)
- `outputs/oof/two_part_future_xg_oof.parquet` (56068 rows, 16 cols)
- `outputs/oof/two_part_future_xg_oof_exploratory.parquet` (56024 rows, 17 cols)
- `outputs/features/provisional_player_signals.parquet` (1012 rows, 84 cols)
- `outputs/models/reports/classification_evaluation_summary.csv` (9 rows, 60 cols)
- `outputs/models/reports/regression_evaluation_summary.csv` (7 rows, 74 cols)
- `outputs/models/reports/two_part_vs_one_stage_common_rows.csv` (30 rows, 22 cols)

## Key schema fields used in coach layer contracts

### Defensive actions

Required in `src/dax/coach_analysis/loaders.py` contract:

- identifiers: `match_id`, `event_id`, `period`, `possession`, `event_index`
- context: `phase_label`, `competition_label`, `team`, `opponent_team`
- player: `player`, `position`, `position_group`
- action type: `action_family`, `event_type`
- location: `action_x`, `action_y`
- possession outcome: `action_won_possession`, `action_changed_possession`
- visibility: `has_360`, `local_5m_region_fully_visible`, `local_10m_region_fully_visible`, `freeze_frame_roles_known`, `visibility_quality_band`
- targets: `target_future_shot_10s`, `target_future_xg_10s`

### Classification OOF

Required columns:

- `match_id`, `event_id`, `fold`, `model_variant`, `y_true`, `y_score`

Observed local variants:

- `b0_constant`, `b1_phase_only`, `b2_spatial_only`, `b3_action_context`, `b4_spatial_phase_action`, `b6_full_without_360`, `b8_interpretable_reduced`

### Regression OOF

Required columns:

- `match_id`, `event_id`, `fold`, `model_variant`, `y_true`, `y_pred`

Observed local variants:

- `r0_constant`, `r1_spatial_linear`, `r2_action_context_linear`, `r3_full_without_360`, `r5_interpretable_reduced`, `r6_nonlinear_candidate`

### Two-part OOF

Required columns:

- `match_id`, `event_id`, `fold`, `classification_model_variant`, `conditional_model_variant`, `observed_future_shot`, `observed_future_xg`, `combined_future_xg_prediction`

Observed local variant pairs:

- standard OOF: `b6_full_without_360` + `conditional_hgb`
- exploratory OOF: `b7_full_with_360` + `conditional_tweedie`

### Provisional player signals

Observed key columns:

- counts: `eligible_actions`, `represented_matches`
- expectations/observations: `expected_shots`, `observed_shots`, `expected_future_xg`, `observed_future_xg`
- suppression: `shot_suppression`, `combined_xg_suppression`, `conditional_severity_suppression`
- reliability: `reliability_tier`, `minimum_sample_flag`, bootstrap CI fields

### Model evaluation reports

- classification report key columns: `variant`, `selection_rule`, `recommendation_status`, fold aggregate metrics.
- regression report key columns: `variant`, `selection_rule`, `recommendation_status`, error metrics and clipping diagnostics.
- two-part comparison key columns: `classification_variant`, `conditional_model`, `benchmark_variant`, `rows`, `selection_reason`, candidate/benchmark metrics.

## Current local gap against requested primary variants

Requested defaults:

- classification primary: `b7_full_with_360`
- classification sensitivity: `b6_full_without_360`
- regression primary: `r4_full_with_360`
- exploratory two-part: `b7_full_with_360` + documented exploratory conditional model

Local files currently contain:

- classification: **missing** `b7_full_with_360`
- regression: **missing** `r4_full_with_360`
- exploratory two-part: present (`b7_full_with_360` + `conditional_tweedie`)

The coach loader now raises explicit errors when requested variants are unavailable.

