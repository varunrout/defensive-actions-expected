# Project roadmap

The project roadmap is organised as ten phases. The current repository contains a completed foundation layer and an active coach-analysis layer. Later phases expand metadata, tactical populations, attribution, validation and presentation.

## Phase status

| Phase | Status | Notes |
| --- | --- | --- |
| 1. Foundation: defensive action dataset and base modelling layer | Complete | The repository builds defensive action features, short-horizon targets and baseline model outputs. |
| 2. Coach analysis layer | Completed or nearly completed, depending on the merged state being reviewed | The current approach uses deterministic coach-analysis scripts rather than notebooks. |
| 3. Metadata enrichment and competition/stage context | Not started | Current local artifacts may contain unknown competition metadata, so enrichment is a future task. |
| 4. Real sensitivity model outputs | Planned | Full sensitivity OOF files are needed where placeholders or smoke-sized files are currently used. |
| 5. Tactical population expansion | Planned | Extend beyond the current centre-back own-box population. |
| 6. Player profiling and role archetypes | Planned | Build player-level role profiles once validated populations and outputs are available. |
| 7. Team/system-level defensive style analysis | Planned | Summarise defensive style and exposure at team/system level. |
| 8. Attribution and defensive value explanation | Planned | Develop careful explanations of value without overclaiming causality. |
| 9. Validation, football sanity checks and case-study review | Planned and ongoing | Continue fold, completeness, sensitivity and football review checks. |
| 10. Portfolio packaging, dashboard/reporting and storytelling | Planned | Package the work for technical and football-facing audiences. |

## Roadmap details

### 1. Foundation: defensive action dataset and base modelling layer

Create the reliable data and modelling base: processed event data, defensive action extraction, contextual features, short-horizon targets, model training and validation outputs. This phase is the basis for all later DAx interpretation.

### 2. Coach analysis layer

Convert coach-facing analysis from exploratory notebooks into deterministic scripts. The old notebook plan included:

- `00_data_and_model_readiness.ipynb`
- `01_cb_box_defence_risk.ipynb`
- `02_wide_1v1_and_cross_control.ipynb`
- `03_transition_and_recovery_defending.ipynb`
- `04_pressing_and_counterpress_second_order_risk.ipynb`
- `05_deep_block_and_crossing_pressure.ipynb`
- `06_team_phase_profiles_world_cup_vs_euros.ipynb`
- `07_player_case_study_generator.ipynb`
- `08_coach_question_summary.ipynb`

The active approach uses `scripts/coach_analysis/00_check_coach_analysis_readiness.py`, `scripts/coach_analysis/01_analyze_cb_box_defence.py`, reusable code in `src/dax/coach_analysis/` and the guide in `docs/coach_analysis/script_guide.md`.

### 3. Metadata enrichment and competition/stage context

Add robust competition, season and stage metadata to the local artifacts. This is needed before making strong competition-level comparisons.

### 4. Real sensitivity model outputs

Generate complete OOF outputs for sensitivity variants so model comparison sections can measure genuine disagreement instead of placeholder equivalence.

### 5. Tactical population expansion

Expand from centre-back own-box defending into additional populations such as wide 1v1s, cross control, transition defending, pressing/counterpressing and deep-block pressure.

### 6. Player profiling and role archetypes

Build player profiles that account for role, exposure, action context and modelled threat rather than ranking players by raw action volume alone.

### 7. Team/system-level defensive style analysis

Aggregate validated signals to team and tactical-system level to describe where teams allow defensive actions, which threats they face and how they resolve them.

### 8. Attribution and defensive value explanation

Develop careful explanatory layers for defensive value. This phase should distinguish predictive association from causal claims and make uncertainty visible.

### 9. Validation, football sanity checks and case-study review

Review outputs through technical validation and football sanity checks. This includes OOF integrity, fold coverage, match coverage, metadata completeness, sample-size checks and video-review candidate review.

### 10. Portfolio packaging, dashboard/reporting and storytelling

Package the project into a readable portfolio case study, reports and potentially dashboard views. The emphasis should remain on reproducibility, honest limitations and clear football interpretation.
