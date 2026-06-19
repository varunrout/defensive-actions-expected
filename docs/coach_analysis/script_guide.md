# Coach-analysis Phase 1 script guide

Phase 1 uses standalone Python scripts.

## Scripts

1. `scripts/coach_analysis/00_check_coach_analysis_readiness.py`
   - Validates explicit input paths, schemas, required model variants, OOF duplicate/missing prediction coverage, fold coverage, match coverage, processed-event timeline fields, competition coverage and boolean visibility coverage.
   - Writes `outputs/coach_analysis/readiness/report.md` and `outputs/coach_analysis/readiness/execution_summary.json`.

2. `scripts/coach_analysis/01_analyze_cb_box_defence.py`
   - Analyses centre-back defensive actions in mutually exclusive box zones using processed-event next-event context.
   - Compares observed and expected shot/future-xG threat, sequence outcomes, competition splits, reliable-visibility sensitivity and model sensitivity when local data are available.
   - Writes a markdown report, execution summary, tables, figures and video-review candidates under `outputs/coach_analysis/cb_box_defence/`.

## Running

```bash
python scripts/coach_analysis/00_check_coach_analysis_readiness.py
python scripts/coach_analysis/01_analyze_cb_box_defence.py
```

Both scripts accept explicit path arguments for defensive actions, processed events, classification OOF, regression OOF, two-part OOF and output root. They are deterministic by default (`--seed 7`) and do not retrain models or modify source datasets.

## Output policy

Generated reports, JSON summaries, figures, CSV tables, local data, OOF files and model artifacts should not be committed.
