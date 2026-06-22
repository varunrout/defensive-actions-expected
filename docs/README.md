# Documentation index

This directory is the project documentation hub for Defensive Actions Expected (DAx). It separates football context, methodology, implementation workflow, analysis reporting and portfolio storytelling so different readers can find the right level of detail quickly.

## Start here

| Document | Audience | Purpose |
| --- | --- | --- |
| [Project overview](project_overview.md) | Football analysts, data scientists, reviewers | Explains the problem, what DAx means and why defensive actions need context. |
| [Project roadmap](project_roadmap.md) | Maintainers, reviewers, collaborators | Describes the ten-phase project plan and current status. |
| [Current results summary](results/current_results_summary.md) | Reviewers, recruiters, analysts | Summarises verified current results and separates validated findings from limitations. |
| [Limitations and next steps](limitations_and_next_steps.md) | Maintainers, reviewers | Lists known gaps and the next delivery priorities. |
| [Portfolio story](portfolio_story.md) | Football analytics recruiters, technical reviewers | Frames the repository as a reproducible football analytics case study. |

## Methodology

| Document | Purpose |
| --- | --- |
| [Data pipeline](methodology/data_pipeline.md) | High-level flow from event data to defensive action features, targets, OOF predictions and coach-analysis joins. |
| [Modelling techniques](methodology/modelling_techniques.md) | Explains classification, regression, two-part xG, sensitivity models and key model variants. |
| [Evaluation and validation](methodology/evaluation_and_validation.md) | Explains OOF integrity, train/test separation, completeness checks and sensitivity caveats. |

## Coach-analysis layer

| Document | Purpose |
| --- | --- |
| [Coach-analysis overview](coach_analysis/coach_analysis_overview.md) | Explains the shift from notebooks to deterministic scripts and the current coach-analysis workflow. |
| [Coach-analysis script guide](coach_analysis/script_guide.md) | Operational guide for the active readiness and CB box-defence scripts. |
| [CB box-defence report notes](analysis_reports/cb_box_defence_report_notes.md) | Notes for interpreting the current centre-back own-box defensive action report. |

## Implementation workflow

| Document | Purpose |
| --- | --- |
| [Local workflow](implementation/local_workflow.md) | Environment setup, checks, coach-analysis commands and output inspection. |
| [GitHub workflow](implementation/github_workflow.md) | Branching, PR hygiene, output policy, squash merge and branch cleanup. |

## Existing reference documentation

The repository also contains deeper reference material under `docs/architecture/`, `docs/models/`, `docs/validation/`, `docs/analysis/`, `docs/data_dictionary/` and `docs/methodology/`. Archived pre-methodology-fix material is retained under `docs/archive/pre_methodology_fix/` for historical context only.
