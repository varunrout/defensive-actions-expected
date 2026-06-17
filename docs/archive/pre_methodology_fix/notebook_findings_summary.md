# Notebook Findings Summary (June 9, 2026)

This document records the current state of notebook-driven reliability checks after the phase and target logic fixes.

## Notebook set covered

- `notebooks/phase_target_reliability_audit.ipynb`
- `notebooks/01_phase_transition_analysis.ipynb`
- `notebooks/02_threat_evolution_by_phase.ipynb`
- `notebooks/03_target_reliability_by_team.ipynb`
- `notebooks/04_action_sequence_patterns.ipynb`
- `notebooks/05_threat_model_validation.ipynb`
- `notebooks/06_cross_match_phase_distribution.ipynb`

## Dataset snapshot (current)

- `data/processed/events_enriched.parquet`: 615,225 rows
- `data/processed/events_with_phases.parquet`: 615,225 rows
- `data/processed/events_with_targets.parquet`: 615,225 rows
- Matches: 166
- Teams: 48

## Before vs after (key reliability metrics)

| Metric | Before fix | After fix | Status |
|---|---:|---:|---|
| First-event `counterpress_after_loss` rate | 96.39% | 0.00% | Fixed |
| Missing declared phases (`box_defence`, `second_ball`) | 2 missing | 0 missing | Fixed |
| Phase parity (stored vs recomputed on full data) | 92.65% | 100.00% | Fixed |
| Stored target column in processed outputs | Missing in `events_with_phases` and stale labels | Present in `events_with_targets` | Fixed in target table |
| Stored target positive rate | ~0.048% (stale/broken) | 5.686% | Fixed |

## Current quantitative outputs

### Phase labels

Current labels present in `events_with_phases.parquet`:

- `box_defence`
- `central_progression_defence`
- `counterpress_after_loss`
- `high_press`
- `second_ball`
- `settled_low_block`
- `settled_mid_block`
- `transition_defence`
- `wide_defending`

Boundary sanity:

- First event in each match labeled `counterpress_after_loss`: **0.00%**

Deterministic parity:

- Stored vs recomputed `phase_label` on full dataset: **1.0000**

### Targets

From `events_with_targets.parquet`:

- `target_shot_in_10s` present: **yes**
- Positive rate: **0.056864** (5.686%)
- Positives: **34,984**

Parity check:

- Stored vs recomputed target parity on 20-match sample: **1.0000**

## Notebook interpretation status

### Safe to use now

- Transition and phase diagnostics in `01_phase_transition_analysis.ipynb`
- Threat-by-phase plots in `02_threat_evolution_by_phase.ipynb`
- Team/match reliability and distribution notebooks (`03`, `06`)
- Action sequence diagnostics (`04`)

### Still recommended as guardrails

- Keep `phase_target_reliability_audit.ipynb` as the first QA gate before reporting.
- Re-run parity checks after each pipeline regeneration.

## Decision status

- **Phases**: production-trustworthy for current regenerated tables.
- **Targets**: production-trustworthy in `events_with_targets.parquet`.
- **Notebook analyses**: can be treated as analytical outputs (not only diagnostics) for this data snapshot.

## Notes

- `events_with_phases.parquet` intentionally contains phase labels only; the target label is stored in `events_with_targets.parquet`.
- If pipeline code changes again, refresh this document after re-running reliability checks.
