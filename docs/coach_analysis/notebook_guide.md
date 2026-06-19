# Coach-analysis Phase 1 script guide

Phase 1 uses standalone Python scripts rather than notebooks.

## Scripts

1. `scripts/coach_analysis/00_check_coach_analysis_readiness.py`
   - Validates local input existence, schemas, explicit model-variant availability, OOF duplicate/missing prediction coverage, fold coverage, match coverage, processed-event timeline fields, competition coverage and visibility coverage.
   - Writes `outputs/coach_analysis/readiness/report.md` and `outputs/coach_analysis/readiness/execution_summary.json`.

2. `scripts/coach_analysis/01_analyze_cb_box_defence.py`
   - Analyses centre-back defensive actions in mutually exclusive box zones.
   - Compares observed and expected shot/future-xG threat, possession security, clearance/block/pressure/duel outcomes, repeated box actions, competition splits and reliable-visibility sensitivity when local data are available.
   - Writes a markdown report, execution summary, tables, figures and video-review candidates under `outputs/coach_analysis/cb_box_defence/`.

## Running

```bash
python scripts/coach_analysis/00_check_coach_analysis_readiness.py
python scripts/coach_analysis/01_analyze_cb_box_defence.py
```

Both scripts are deterministic by default (`--seed 7`) and do not retrain models or modify source datasets.

## Required local inputs

The readiness script checks the expected processed defensive-action feature table, OOF classification/regression/two-part predictions, model reports and configs. Missing files are reported clearly in the generated readiness outputs.

## Output policy

Generated reports, JSON summaries, figures, CSV tables, local data, OOF files and model artifacts should not be committed.
