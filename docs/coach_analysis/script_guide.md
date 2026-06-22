# Coach-analysis script guide

The coach-analysis layer uses standalone, deterministic Python scripts. Phase 2 keeps the work script-based and adds metadata, population-diagnostic and validation-mode safeguards without retraining models.

## Scripts

1. `scripts/coach_analysis/00_check_coach_analysis_readiness.py`
   - Validates explicit input paths, schemas, required model variants, OOF duplicate/missing prediction coverage, fold coverage, match coverage, processed-event timeline fields, competition coverage and boolean visibility coverage.
   - Writes `outputs/coach_analysis/readiness/report.md` and `outputs/coach_analysis/readiness/execution_summary.json`.

2. `scripts/coach_analysis/01_analyze_cb_box_defence.py`
   - Analyses centre-back defensive actions in mutually exclusive box zones using processed-event next-event context.
   - Compares observed and expected shot/future-xG threat, sequence outcomes, competition/season/stage splits, reliable-visibility sensitivity and model sensitivity when local data are available.
   - Reports the spatial own-box centre-back population as the primary analysis population and includes a phase-label overlap diagnostic rather than forcing tactical phase labels to match spatial box membership.
   - Emits validation-mode warnings when sensitivity variants are identical to primary variants or when sensitivity comparisons are smoke-sized/incomplete.
   - Writes a markdown report, execution summary, tables, figures and video-review candidates under `outputs/coach_analysis/cb_box_defence/`.

## Running

```bash
python scripts/coach_analysis/00_check_coach_analysis_readiness.py
python scripts/coach_analysis/01_analyze_cb_box_defence.py
```

Both scripts accept explicit path arguments for defensive actions, processed events (default `data/processed/events_with_targets.parquet`), classification OOF, regression OOF, two-part OOF and output root. They are deterministic by default (`--seed 7`) and do not retrain models or modify source datasets.

## Output policy

Generated reports, JSON summaries, figures, CSV tables, local data, OOF files and model artifacts should not be committed.


## Phase 2 report additions

`01_analyze_cb_box_defence.py` now canonicalises competition metadata into stable columns before grouping:

- `coach_competition` from `competition_label`, `competition` or `competition_name`.
- `coach_competition_stage` from `competition_stage`, `stage` or `stage_name`.
- `coach_season` from `season_name`, `season` or `season_id`.
- `coach_competition_id` and `coach_season_id` where identifiers are present.

Each field falls back to `unknown` only when no supported source column is available or the source value is missing/blank. The report includes `Competition metadata counts` and grouped tables for available competition, stage and season metadata.

The report also includes a `Phase vs spatial box-defence diagnostic` JSON block with the spatial own-box centre-back action count, centre-back actions carrying `phase_label == box_defence`, their overlap count and the selected coordinate columns. This is a diagnostic for definition/coordinate-alignment review: the spatial own-box centre-back population remains the primary analysis population, while phase-label overlap is not treated as ground truth.

Model sensitivity sections do not invent disagreement. If primary and sensitivity columns are identical, or if a comparison is incomplete/smoke-sized, `model_sensitivity_warnings` in `execution_summary.json` and the report's `Model-sensitivity validation warnings` section label the issue as a validation-mode limitation.

## Methodology notes

The scripts create canonical coach model columns (`coach_expected_shot_b7`, `coach_expected_shot_b6`, `coach_expected_xg_r4`, `coach_expected_xg_r6`, `coach_expected_xg_two_part`, `coach_observed_shot`, `coach_observed_xg`) from the selected OOF schemas. Variant selection is strict. Readiness reports native eligible-population coverage for 360 models and labels whether eligibility came from feature-contract fields or selected OOF event IDs. Processed-event sequence labels use next-event windows and do not link across periods. CB video-review outputs include category-specific CSV files for clearances, blocks, pressures, turnovers, high expected threat, observed threat above expected and repeated box actions.

## Coordinate convention

Coach-analysis scripts assume the StatsBomb 120 x 80 pitch frame. The canonical action coordinates in `player_defensive_actions.parquet` are `action_x` and `action_y`; these are checked before other fallback coordinate names. For centre-back box-defence analysis, the acting defensive team's own goal is treated as `x = 0`, so the own penalty area is `x <= 18` and `18 <= y <= 62`. The attacking penalty area is the opposite end, `x >= 102` and `18 <= y <= 62`. If an upstream pipeline direction-normalises coordinates differently, the selected coordinate columns and filter-stage counts in `execution_summary.json` should be checked before interpreting the report.
