# Analysis Code Audit

This audit records the pre-existing analysis-related files reviewed before adding the reusable pre-modelling analysis layer.

| Current path | Purpose | Status | Inputs | Outputs | Valid under corrected targets/schema? | Replacement | Decision |
|---|---|---|---|---|---|---|---|
| `src/dax/analysis/reporting.py` | Validation-summary report helper for model validation artifacts. | Active but narrow. | Validation model metrics JSON. | Validation summary Markdown. | Partially; model-validation specific, not pre-modelling analysis. | `src/dax/analysis/reporting.py` now includes pre-model reporting while retaining reusable reporting role. | MERGE |
| `src/dax/analysis/notebook_aggregation.py` | Constants for historical notebook aggregation. | Historical/notebook-oriented. | Player feature tables. | Notebook aggregations. | Partially; some fields are corrected, but logic belongs in reusable aggregation modules. | `src/dax/analysis/player_aggregation.py`, `feature_diagnostics.py`. | MERGE/ARCHIVE |
| `src/analysis/__init__.py` | Empty legacy analysis namespace. | Stale. | None. | None. | Not applicable. | `src/dax/analysis/`. | DELETE |
| `scripts/generate_reports.py` | Thin wrapper for canonical validation reports. | Active validation utility. | Validation metrics. | Validation reports. | Valid for validation, not pre-modelling analysis. | `scripts/generate_analysis_report.py` for pre-modelling reporting. | KEEP |
| `notebooks/archive/pre_methodology_fix/*.ipynb` | Historical notebook analysis and findings. | Historical archived. | Old processed/features/model outputs. | Notebook figures/findings. | No; archived as pre-methodology material. | Reusable scripts in `scripts/` and modules in `src/dax/analysis/`. | ARCHIVE |
| `docs/archive/pre_methodology_fix/analysis/*.md` | Historical analysis reports. | Historical archived. | Old targets/features/model outputs. | Markdown reports. | No; predates corrected methodology. | `outputs/analysis/reports/pre_model_analysis_report.md`. | ARCHIVE |
| `scripts/models/*` and `src/dax/models/*` | Predictive modelling utilities. | Active but out of scope. | Feature tables. | Trained/evaluated models. | Methodology-specific; not modified for analysis layer. | None in this PR; final model training not performed. | KEEP |

No active replacement imports from archive directories. Deprecated target names `target_xt_10s` and `target_shot_in_10s` are not used by the new active analysis framework.
