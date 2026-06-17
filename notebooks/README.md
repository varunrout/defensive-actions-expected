# Notebooks

Historical notebooks were archived under `notebooks/archive/pre_methodology_fix/` because they were not rerun against the current corrected pipeline.

## Intended run order

1. `01_data_quality.ipynb`
2. `02_event_and_possession_semantics.ipynb`
3. `03_phase_and_visibility_validation.ipynb`
4. `04_feature_analysis.ipynb`
5. `05_model_validation.ipynb`
6. `06_player_case_studies.ipynb`

## Required inputs

- Regenerated processed events in `data/processed/`.
- Regenerated player defensive action features in `data/features/`.
- Current metrics/figures in `outputs/metrics/` and `outputs/figures/` for validation/reporting notebooks.

## Outputs

Notebook-derived outputs should be written to `outputs/figures/` or `outputs/reports/` and should not replace pipeline-owned artifacts.
