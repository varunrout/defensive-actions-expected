# Processed Data Analysis

## Inputs

Default input is `data/processed/events_with_targets.parquet`. The table must contain corrected short-horizon targets and event context columns including match, period, index, possession, `type` event category, phase proxy, 360 availability, attacking team before action, and defending team before action.

## Checks

The processed-data CLI validates the schema, then writes row/match/competition coverage, event counts by type, rows per match, events per possession, duplicate identifier checks, missingness, team-context validity, 360 coverage when present, phase distribution, and target health tables.

## Outputs

Tables are written to `outputs/analysis/data_quality/` as CSV plus `metadata.json`. Charts include events by competition when the field exists, events by type, rows per match, missingness, phase distribution, and future-xG target distribution.

## Failure conditions

The analysis fails fast when required corrected target or team-context fields are absent. Optional breakdowns are skipped only when the source column is unavailable; those limitations should be reviewed in the generated report.
