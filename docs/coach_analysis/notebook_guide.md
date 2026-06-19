# Coach-analysis notebook guide

## Order

Run `00_data_and_model_readiness.ipynb` first, then notebooks `01` to `08`.

## Required local inputs

The suite checks for feature parquet files, OOF prediction parquet files, model evaluation reports, player-signal reliability files, and analysis/model configs. Missing inputs are reported in Notebook 00 and in each notebook readiness table.

## Football meaning

The notebooks answer practical coach questions about centre-back box defending, wide 1v1 defending, transition recovery, pressing second-order risk, deep-block crossing pressure, competition context, player case studies and briefing-level conclusions.

## Limitations

The suite does not infer causality, individual responsibility, body orientation, exact runs, complete passing options, off-camera player positions or deliberate pressing triggers unless such variables exist locally.

## Outputs

Figures: `outputs/coach_analysis/figures/`.
Tables: `outputs/coach_analysis/tables/`.
Candidate video events: `outputs/coach_analysis/video_review/`.

## Video-review use

Candidate-event tables identify match and event IDs with expected and observed threat signals. Analysts should use them as a queue for video confirmation, not as proof that the model explains why an event happened.
