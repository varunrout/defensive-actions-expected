# Coach-analysis overview

## Why the project shifted from notebooks to scripts

The original coach-analysis plan was notebook-based and included separate notebooks for readiness, centre-back box defending, wide 1v1s, transition defending, pressing, deep blocks, team profiles, player case studies and coach summaries. That plan was useful for designing questions, but deterministic scripts are better for repeatable analysis.

Scripts make the workflow easier to rerun, test, review, parameterise and package. They also reduce the risk that report results depend on hidden notebook state.

## Current script-based workflow

The active coach-analysis workflow uses:

- `scripts/coach_analysis/00_check_coach_analysis_readiness.py`;
- `scripts/coach_analysis/01_analyze_cb_box_defence.py`;
- reusable modules in `src/dax/coach_analysis/`;
- operational guidance in `docs/coach_analysis/script_guide.md`.

The scripts are deterministic, accept explicit input paths and write generated outputs under the local `outputs/coach_analysis/` tree. Those outputs are not committed.

## Readiness script

`00_check_coach_analysis_readiness.py` validates whether the local artifacts are suitable for coach analysis. It checks schemas, required model variants, OOF coverage, duplicate predictions, fold coverage, match coverage, processed-event timeline context, metadata coverage and visibility fields.

The script writes a markdown readiness report and an execution summary JSON under the generated outputs directory.

## CB box-defence analysis script

`01_analyze_cb_box_defence.py` analyses centre-back defensive actions in the defending team's own penalty-box area. It joins selected defensive actions to model predictions and processed-event context, then reports observed and expected threat, sequence outcomes, metadata splits where available, sensitivity diagnostics and video-review candidates.

## Generated outputs

The CB analysis produces generated local artifacts such as:

- markdown report;
- execution summary JSON;
- summary tables;
- pitch map or figure outputs;
- video-review candidate CSVs.

These files are analysis outputs and should not be committed to the repository.

## Phase versus spatial box-defence diagnostic

The current primary CB box-defence population is spatial: centre-back actions in the own penalty-box area. The script also reports a phase-label diagnostic that compares this spatial definition with actions labelled as `box_defence` by phase logic.

A zero or low overlap should not be treated as a football conclusion by itself. It is a diagnostic for coordinate convention, phase-label definition and upstream alignment review.
