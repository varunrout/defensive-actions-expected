# Initial remediation audit

Date: 2026-06-16. Branch: `fix/methodology-pipeline-rebuild`.

## Repository structure inspected
The repository contains package code under `src/dax/`, analysis helpers under `src/analysis/`, operational scripts under `scripts/`, tests under `tests/`, notebooks under `notebooks/`, documentation under `docs/`, and historical generated outputs under `outputs/` / data directories when present.

## Current pipeline stages observed
1. StatsBomb loading and enrichment in `src/dax/data/statsbomb_loader.py`.
2. Phase labelling in `src/dax/features/phase_segmentation.py`.
3. Possession features in `src/dax/features/possession_sequences.py`.
4. Short-horizon targets and grid threat model in `src/dax/models/attacking_threat.py`.
5. Player defensive action table in `src/dax/features/player_defense.py`.
6. Baseline classification/regression utilities in `src/dax/models/`.
7. Thin but inconsistent scripts under `scripts/`.

## Current targets
- Binary `target_shot_in_10s` existed but searched forward by time/team and could cross possession boundaries.
- Continuous `target_xt_10s` was built from a grid model fit on the same full dataset, making it circular and unsuitable as a regression target.

## Current feature groups
Spatial/action features, phase labels, player/action metadata, freeze-frame teammate/opponent counts, possession timing features, and model variant feature lists.

## Validation strategy observed
Model code used GroupKFold by match for baseline cross-validation. Tournament holdouts, fold QA, calibration, visibility-band reporting, and action-family reporting were incomplete or not centrally enforced.

## Logical and semantic problems
- Event order sometimes relied on timestamps instead of canonical `(match_id, period, index)`.
- Event team and possession team were conflated.
- Defensive action families were treated too uniformly; `Shield` and generic goalkeeper actions had unsupported defensive interpretations.
- Freeze-frame teammate/opponent labels were actor-relative but model-facing names implied stable attacking/defending roles.
- Phase labels were presented too strongly despite rule-based proxy logic.

## Leakage risks
- `possession_duration_total`, `possession_event_count_total`, and `possession_progress_ratio` were model features even though they reveal future possession information.
- The grid-threat continuous target was trained on the complete dataset before target creation.

## Football semantic risks
- Nearest-goal geometry allowed actions near the defending goal to look dangerous under a left-to-right attacking convention.
- Defensive player rows could label the same side as both attacking and defending.
- Possession changes and ball recoveries require previous-possession semantics.

## Reproducibility problems
- Raw-to-processed separation was not sufficiently enforced.
- Scripts contained platform-specific examples and import workarounds.
- Dependency definitions differed between `pyproject.toml` and `requirements.txt`.

## Stale documentation and notebooks
Several docs made DAx, invisible-defending, production, or causal suppression claims beyond the implementation. Existing notebooks were generated before the methodology fixes and should be treated as historical until rerun.

## Baseline test run before changes
Command: `python -m pytest -q`. Result: 8 passed, 3 failed. Failures: box-defence coordinate expectation and two pipeline tests with missing `period` in synthetic loader input. No dependency installation was required in this environment.

## Planned repair sequence
Prioritise event ordering/context, possession-bounded targets, leakage removal, attacking-goal geometry, defensive semantics, phase proxy state, target replacement, invariant tests, dependency/CI cleanup, and documentation that honestly distinguishes current threat prediction from future true DAx.
