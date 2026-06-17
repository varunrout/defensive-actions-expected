# Defensive Actions Expected (DAx) — methodology rebuild

This repository is being remediated into a defensible football analytics research project for:

> A model estimating short-horizon attacking threat following recorded defensive actions using event, possession and StatsBomb 360 context.

## What the current model does

- Builds event-time context for recorded actions.
- Uses canonical event order: `match_id`, `period`, `index`.
- Creates observed short-horizon targets bounded by match, period, possession and attacking team.
- Trains baseline interpretable models for post-action attacking threat.

## What it does not do

- It does not estimate causal defensive suppression.
- It does not implement invisible defending.
- It does not produce final player DAx attribution.
- It does not implement counterfactual option removal.

Those requirements are documented as future work in `docs/methodology/roadmap_to_true_dax.md`.

## Data sources

The project uses StatsBomb Open Data. Raw data should be downloaded once into `data/raw/`; processed tables should be rebuilt from raw files in offline mode where possible. StatsBomb 360 data is only available for competitions that include freeze frames.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

## Pipeline commands

Fixture/local checks:

```bash
ruff check src scripts tests
python -m pytest -q
python -m pytest -q tests/test_end_to_end_fixture.py
```

Full run order:

1. `python scripts/pipeline/pipeline.py` to fetch raw StatsBomb JSON and build processed event tables with event context, phase proxies and corrected targets.
2. `python scripts/features/build_player_defense_dataset.py` to build `data/features/player_defensive_actions.parquet`.
3. Run model scripts only after regenerated feature tables contain `target_future_shot_10s` and `target_future_xg_10s`.

Legacy scripts remain under `scripts/`; reusable logic belongs under `src/dax/`. Full raw-data regeneration may require network access and StatsBomb availability.

## Methodology summary

- **Targets:** `target_future_shot_10s` and `target_future_xg_10s` are observed outcomes, not model predictions.
- **Prediction:** Baseline models predict post-action short-horizon attacking threat.
- **Validation:** Grouped validation by match is retained; tournament holdout reporting is planned for full-data runs.
- **Geometry:** After normalisation, attacking goal is `(120, 40)` and defending goal is `(0, 40)`.
- **Phases:** Defensive phases are `rule_based_proxy` labels with confidence/rule metadata, not confirmed tactical truth.

## Repository structure

```text
src/dax/                 package logic
  features/              event context, player defense, phase proxies
  targets/               possession-bounded observed targets
  models/                baseline models and exploratory grid model
scripts/                 thin entry points and legacy utilities
tests/                   deterministic unit tests and fixtures
docs/                    methodology, validation, architecture and remediation notes
notebooks/               exploratory notebooks; old outputs should be treated as historical until rerun
outputs/                 generated reports and validation artifacts
```

## Current corrected results

No full corrected model metrics are claimed in this commit because the full StatsBomb pipeline was not regenerated in this environment. Historical metrics produced before the target/leakage fixes should not be used as current results.

## Limitations

- Visible-area controls use deterministic polygon area and point-in-polygon checks; they do not infer players outside the StatsBomb 360 camera footprint.
- Human phase validation has not been performed.
- Tournament holdout metrics require full regenerated feature tables.
- Current models are baselines, not causal DAx.

## Roadmap to true DAx

See `docs/methodology/roadmap_to_true_dax.md` for the pre-action, post-action, counterfactual, attribution, opportunity-adjustment and validation work required.

## Data attribution

StatsBomb Open Data is provided by StatsBomb. Users must comply with StatsBomb's terms and attribution requirements.
