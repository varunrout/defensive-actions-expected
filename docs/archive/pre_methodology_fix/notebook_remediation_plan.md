# Notebook Remediation Plan

This plan is updated to reflect the current post-fix state.

## Objective

Keep reliability at production-trustworthy levels and prevent regressions.

## Completed items

## P0 - Phase boundary reset defect (RESOLVED)

### Delivered

- State reset implemented at match/period boundaries in `src/dax/features/phase_segmentation.py`.
- Event ordering enforced before labeling in `scripts/pipeline/pipeline.py`.
- Test coverage added in `tests/test_phase_segmentation.py`.

### Verified outcomes

- First-event `counterpress_after_loss` rate: **0.00%**
- Phase parity (stored vs recomputed): **1.0000**

## P0 - Missing phase classes (`box_defence`, `second_ball`) (RESOLVED)

### Delivered

- Rule ordering updated to make branches reachable.
- Schema fallback support (`event_type` / `type`) added.
- Test case added for `second_ball` activation.

### Verified outcomes

- Missing declared phase labels: **0**
- Both `box_defence` and `second_ball` appear in full data output.

## P1 - Persist target labels in processed outputs (RESOLVED)

### Delivered

- Regenerated `data/processed/events_with_targets.parquet` with corrected logic.

### Verified outcomes

- `target_shot_in_10s` exists in `events_with_targets.parquet`.
- Positive rate: **5.686%**
- Sample parity (20 matches, stored vs recomputed): **1.0000**

## Remaining work

## P1 - Reliability gate automation (OPEN)

### Required action

Add a lightweight post-pipeline checker script that records:

- phase parity
- first-event anomaly rate
- missing phase labels
- target parity and target rate sanity

### Acceptance criteria

- Script exits non-zero on threshold failure.
- Script writes a compact markdown or JSON report for CI/manual review.

## P2 - Documentation synchronization (OPEN)

### Required action

- Keep reliability docs in sync after each pipeline regeneration:
  - `docs/notebook_findings_summary.md`
  - `docs/notebook_remediation_plan.md`

### Acceptance criteria

- Docs include current metrics and date stamp after each major data refresh.

## Validation thresholds (for ongoing regression checks)

| Metric | Threshold |
|---|---|
| Phase parity (stored vs recomputed) | >= 99.5% |
| First-event counterpress rate | < 5% |
| Missing declared phase labels | 0 |
| Target parity (stored vs recomputed) | >= 99.5% |
| Target positive rate sanity | 1% to 10% |

## Operational notes

- Continue using `phase_target_reliability_audit.ipynb` as first-pass QA.
- Prefer `events_with_targets.parquet` for modeling; `events_with_phases.parquet` is phase-only by design.
