# Data pipeline

## High-level flow

The DAx pipeline turns event-level football data into defensive action records, contextual features, short-horizon targets, model predictions and coach-analysis reports.

```text
raw / processed events
  -> defensive action extraction
  -> contextual feature construction
  -> future-shot and future-xG target construction
  -> model training and out-of-fold predictions
  -> coach-analysis joins and tactical population reports
```

Generated data, model artifacts, reports, figures and local outputs are not committed to the repository.

## Defensive action extraction

The defensive action table selects player actions that are relevant to defending, such as tackles, pressures, interceptions, blocks, clearances and related event types where available. Extraction should preserve event identifiers, match identifiers, player/team context, positional context and action coordinates so later modelling and coach analysis can join predictions back to the original actions.

## Feature construction

Feature construction adds football context around each action. Examples include location, phase proxies, possession context, time context, pressure or nearby-player information when available, and StatsBomb 360-derived context for variants that use freeze-frame information.

The goal is not to make a raw action count table. The goal is to describe the situation in which each defensive action occurred so the model can estimate subsequent danger more fairly.

## Target construction

The current target layer uses short-horizon observed outcomes after defensive actions:

- **future shot:** whether the attacking team produces a shot in the defined horizon after the defensive action;
- **future xG:** the xG attached to future shots in that horizon.

These targets are observed outcomes, not model predictions. They are used to train and evaluate models that estimate post-action attacking threat.

## OOF prediction outputs

Out-of-fold predictions are produced by scoring each record with a model that was not trained on that record's fold. OOF outputs are important because coach analysis should not be based on in-sample fitted values.

The coach-analysis scripts expect canonical model variants and join predictions back to selected defensive actions by stable identifiers. Duplicate predictions, missing predictions and incomplete fold coverage are validation concerns.

## Coach-analysis joins

Coach-analysis scripts join:

- defensive action records;
- processed event timeline context;
- classification OOF predictions;
- regression OOF predictions;
- two-part future-xG predictions;
- optional sensitivity variant predictions;
- metadata columns when present.

These joins support tactical population reports, model completeness checks, sequence summaries, video-review candidate exports and diagnostic sections.

## Output policy

Do not commit generated artifacts. This includes local data, OOF files, model files, MLflow runs, report folders, figures, CSV exports and execution summaries under output directories.
