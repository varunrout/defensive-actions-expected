# Defensive Actions Expected (DAx)

This repository estimates short-horizon attacking threat following recorded defensive actions using event, possession and StatsBomb 360 context.

## Scope

Current code builds event context, phase proxies, possession-bounded targets and baseline models. It does **not** implement causal defensive suppression, invisible defending, counterfactual option removal or final player DAx attribution.

## Install

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

## Canonical workflow

`download -> process -> event context -> phase proxies -> targets -> player features -> train -> validate -> report`

Active entry points:

```bash
python scripts/run_pipeline.py --help
python scripts/build_features.py --help
python scripts/train_models.py --help
python scripts/validate_models.py --help
python scripts/generate_reports.py --help
```

Local checks:

```bash
python -m pytest -q
python -m pytest -q tests/test_end_to_end_fixture.py
python -m pytest -q tests/test_cli_execution.py
python -m ruff check src scripts tests
```

Supported execution order:

1. `python scripts/run_pipeline.py --stage prepare-data` builds processed event tables with event context, phase proxies and corrected targets.
2. `python scripts/build_features.py --input data/processed/events_with_targets.parquet --output data/features/player_defensive_actions.parquet` builds the player defensive actions table.
3. `python scripts/train_models.py --task all` trains the supported logistic and regression baselines.
4. `python scripts/validate_models.py --task all` generates validation plots and summary tables for trained baselines.
5. `python scripts/generate_reports.py --report validation-summary` builds the canonical validation summary report.

Compatibility wrappers remain active for migration-safe legacy commands at:

- `scripts/pipeline/pipeline.py`
- `scripts/features/build_player_defense_dataset.py`
- `scripts/models/train_baseline_logistic.py`
- `scripts/models/train_baseline_regression.py`
- `scripts/models/evaluate_baseline_model.py`
- `scripts/models/evaluate_baseline_regression.py`

## Methodology summary

- **Targets:** `target_future_shot_10s` and `target_future_xg_10s` are observed outcomes, not model predictions.
- **Prediction:** Baseline models predict post-action short-horizon attacking threat.
- **Validation:** Grouped validation by match is retained; tournament holdout reporting is planned for full-data runs.
- **Geometry:** After normalisation, attacking goal is `(120, 40)` and defending goal is `(0, 40)`.
- **Phases:** Defensive phases are `rule_based_proxy` labels with confidence/rule metadata, not confirmed tactical truth.

## Repository layout

```text
configs/                 pipeline, competition and model references
src/dax/                 reusable package logic
scripts/                 active canonical CLIs plus migration wrappers
docs/                    active methodology, architecture, validation and data dictionary docs
notebooks/               sequential notebook index; historical notebooks archived
outputs/                 generated artifact directories; historical outputs archived
tests/                   unit/integration tests and fixtures
```

Historical pre-fix documents and outputs are retained under `docs/archive/pre_methodology_fix/` and `outputs/archive/pre_methodology_fix/`. Historical pre-cleanup scripts are retained under `scripts/archive/pre_repository_cleanup/`. Archive contents are for reference only and are not active execution paths.

## Pre-modelling analysis

Reusable analysis logic lives in `src/dax/analysis/` and is executed with thin CLI scripts. Run it after preparing data and building features, and before any final predictive model training:

```bash
python scripts/analyze_processed_data.py --input data/processed/events_with_targets.parquet --output-dir outputs/analysis/data_quality
python scripts/analyze_features.py --input data/features/player_defensive_actions.parquet --output-dir outputs/analysis/features
python scripts/build_player_summary.py --input data/features/player_defensive_actions.parquet --output data/features/player_defensive_summary.parquet
python scripts/run_player_clustering.py --input data/features/player_defensive_summary.parquet --output-dir outputs/analysis/clustering --config configs/analysis.yaml
python scripts/build_descriptive_signals.py --input data/features/player_defensive_summary.parquet --clusters outputs/analysis/clustering/player_clusters.parquet --output data/features/player_defensive_signals_descriptive.parquet
python scripts/generate_analysis_report.py --analysis-dir outputs/analysis --output outputs/analysis/reports/pre_model_analysis_report.md
```

These analyses are descriptive foundations only: no final classifier/regressor is trained, no causal defensive value is claimed, and provisional signals are not true DAx.
