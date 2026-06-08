# Defensive Actions Expected (DAx)

A rigorous football analytics project measuring defending as **attacking option suppression**, not just defensive event counts.

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Project definition & defensive theory | ✓ Complete |
| 2 | Data acquisition & preparation | ✓ Complete |
| 3 | Possession & phase segmentation | ✓ Complete |
| 4 | Attacking threat baseline | ✓ Complete |
| 5–10 | Advanced features & models | In progress |
| 11–13 | Validation, outputs, packaging | Backlog |

## Data pipeline

The project uses **StatsBomb open data** for three major tournaments:

- **Euro 2020** (51 matches)
- **FIFA World Cup 2022** (64 matches) — includes 360 freeze-frame data
- **Euro 2024** (51 matches) — includes 360 freeze-frame data

### Run the data pipeline

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py
```

This fetches all StatsBomb data and populates:

```
data/
  raw/           # JSON files from StatsBomb API
  processed/     # Enriched Parquet tables
    - events_enriched.parquet
    - events_with_phases.parquet
    - events_with_targets.parquet
    - summary.json
  features/      # Feature engineering outputs (future)
  models/        # Trained model artifacts (future)
  validation/    # Validation results (future)
```

## Project structure

```
defensive-actions-expected/
  src/dax/                # Main package
    __init__.py
    constants.py          # Target competitions registry
    data/
      __init__.py
      statsbomb_loader.py # Real StatsBomb data loading
    features/
      __init__.py
      phase_segmentation.py      # Defensive phase classifier
    models/
      __init__.py
      attacking_threat.py        # xT-style threat baseline
    pipeline/
      __init__.py
  scripts/
    pipeline.py           # Unified data pipeline
  notebooks/              # Jupyter analysis notebooks (future)
  data/                   # Data warehouse (created by pipeline)
  tests/                  # Unit tests
  README.md
  requirements.txt
  pyproject.toml
```

## Installation

```powershell
# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Running the project

### 1. Fetch and prepare all data

```powershell
.\.venv\Scripts\python.exe scripts\pipeline.py
```

Expected output:
```
================================================================================
STAGE 1: Fetch raw StatsBomb data → data/raw/
================================================================================
  [1a] Competitions ...
  [1b] Matches (166 total across 3 tournaments) ...
  [1c] Events and 360 freeze-frames ...

================================================================================
STAGE 2: Build enriched models → data/processed/
================================================================================
  [2a] Load and enrich events ...
  [2b] Add defensive phase labels ...
  [2c] Add attacking threat targets ...
  [2d] Summary statistics ...

SUCCESS: Pipeline complete!
```

### 2. Explore data and build features (future)

- Analysis notebooks will guide exploratory data analysis
- Feature engineering per phase (ball pressure, lane blocking, space occupation, etc.)
- Player and team defensive context building

## Next steps

- **Phase 5:** Possible futures & attacking option trees
- **Phase 6–7:** Player and team defensive feature engineering
- **Phase 8:** Defensive suppression models
- **Phase 9:** Attribution framework
- **Phase 10:** Phase-specific models
- **Phase 11:** Validation & case studies
- **Phase 12:** Dashboards & storytelling
- **Phase 13:** Portfolio packaging

## Notes

- All data from **StatsBomb open data** (no API key required)
- Pipeline handles missing 360 data gracefully (Euro 2020 has no 360)
- All processing is deterministic and reproducible
- Tests are included to validate data loading and transformations

