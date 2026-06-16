# Defensive Actions Expected (DAx)

A rigorous football analytics project measuring defending as **attacking option suppression**, not just defensive event counts.

## Project phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Project definition & defensive theory | ✓ Complete |
| 2 | Data acquisition & preparation | ✓ Complete |
| 3 | Possession & phase segmentation + enrichment | ✓ Complete |
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
      possession_sequences.py    # NEW: Possession-level enrichment
    models/
      __init__.py
      attacking_threat.py        # xT-style threat baseline
    pipeline/
      __init__.py
  scripts/
    pipeline.py                  # Unified data pipeline
    extract_possessions.py       # NEW: Possession sequence extraction
    visualize_possessions.py     # NEW: 360-only possession visualizations
    build_player_defense_dataset.py # NEW: Player defensive action table
    profile_player_defense_features.py # NEW: Feature profiling for player model
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

### 2. Extract possession sequences with 360 data enrichment

Or run all downstream steps in one command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_downstream_pipeline.ps1
```

Optional flags:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_downstream_pipeline.ps1 -DryRun
```

```powershell
.\.venv\Scripts\python.exe scripts\extract_possessions.py
```

This creates **possession-level features** from events with 360 data:

```
data/features/possessions_with_360.parquet
  - ~100k+ possession sequences (from WC 2022 + Euro 2024)
  - Phase trajectories (how defense adapts during attack)
  - Pressure dynamics (opponent count changes via 360)
  - Zone progressions (ball movement through pitch zones)
  - Binary target: did possession lead to shot in 10s?
```

**Key insight:** Enrich your understanding by analyzing defense at the **possession narrative level**, not just event-by-event.

See: [Possession Sequences Documentation](docs/possession_sequences.md)

### 3. Analyze and visualize

```powershell
.\.venv\Scripts\python.exe scripts\visualization\visualize_possessions.py
```

This creates static visual outputs in `outputs/validation/possessions/` for **360-only possessions**:

```
outputs/validation/possessions/
  - possession_phase_transition_heatmap.png
  - possession_phase_mix_by_outcome.png
  - sample_shot_possession_sequence.png
  - sample_no_shot_possession_sequence.png
```

Use these views to inspect:
- how defensive phases switch inside a possession,
- how phase mix differs between shot-ending and non-shot possessions,
- how the ball trajectory, defender visibility, and freeze-frame structure evolve.

See: [Possession Visualization Guide](docs/possession_visualization.md)

### 4. Build advanced features (future)

- Analysis notebooks will guide exploratory data analysis
- Feature engineering per phase (ball pressure, lane blocking, space occupation, etc.)
- Player and team defensive context building
- Integrate possession-sequence features into final models

### 5. Build the first player defensive dataset

```powershell
.\.venv\Scripts\python.exe scripts\build_player_defense_dataset.py
.\.venv\Scripts\python.exe scripts\profile_player_defense_features.py
```

This creates a 360-only player-action table focused on defensive events and
their support context:

```
data/features/player_defensive_actions.parquet
  - player-actor rows for defensive actions
  - phase + possession context
  - action location and goal-distance features
  - teammate/opponent interaction around the action
  - target: did the possession lead to a shot in 10s?
```

See: [Player Defensive Model](docs/player_defense_model.md)

## Modeling Strategy

**[Baseline Modeling Strategy](docs/BASELINE_MODELING_STRATEGY.md)** — Complete strategy document covering:
- Problem framing and objective
- Dataset & feature architecture (40+ features across 4 groups)
- Model selection (logistic regression → tree-based comparison)
- Evaluation approach (GroupKFold by match, ROC-AUC, calibration)
- Implementation roadmap and success criteria

**[Quick Reference](docs/MODELING_QUICK_REFERENCE.md)** — 1-page summary for rapid access

## Next steps

- **Phase 5:** Possible futures & attacking option trees
- **Phase 6–7:** Player and team defensive feature engineering (IN PROGRESS)
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
