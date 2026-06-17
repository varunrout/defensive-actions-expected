# Pipeline Scripts

Core data pipeline for fetching, enriching, and processing StatsBomb event data.

## Scripts

- **pipeline.py** - Main orchestrator for stages 1-3
  - Stage 1: Fetch raw data from StatsBomb
  - Stage 2: Enrich events with coordinates, phases, targets
  - Stage 3: Initialize data directories

## Usage

```bash
python pipeline.py
```

## Output

- `data/raw/` - Raw JSON from StatsBomb API
- `data/processed/` - Enriched Parquet tables:
  - `events_enriched.parquet` - Normalized coordinates, basic features
  - `events_with_phases.parquet` - Defensive phase labels
  - `events_with_targets.parquet` - Shot-in-10s targets

