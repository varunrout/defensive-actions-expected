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
python -m ruff check src scripts tests
```

## Repository layout

```text
configs/                 pipeline, competition and model references
src/dax/                 reusable package logic
scripts/                 five thin active CLI entry points
docs/                    active methodology, architecture, validation and data dictionary docs
notebooks/               sequential notebook index; historical notebooks archived
outputs/                 generated artifact directories; historical outputs archived
tests/                   unit/integration tests and fixtures
```

Historical pre-fix documents and outputs are retained under `docs/archive/pre_methodology_fix/` and `outputs/archive/pre_methodology_fix/`. They must not be cited as current results.
