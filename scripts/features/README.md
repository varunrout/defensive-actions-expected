# Feature Engineering Scripts

Extract and build feature tables from processed events for modeling.

## Scripts

- **extract_possessions.py** - Sequence possessions into possession-level records
  - Input: `events_with_targets.parquet`
  - Output: `possessions_with_360.parquet`
  - Features: phase transitions, opponent pressure dynamics, zone progressions

- **build_player_defense_dataset.py** - Extract player-level defensive actions
  - Input: `events_with_targets.parquet`
  - Output: `player_defensive_actions.parquet`
  - Features: action location, goal distance, freeze-frame support metrics

- **profile_player_defense_features.py** - Analyze feature distributions and quality
  - Input: `player_defensive_actions.parquet`
  - Output: Feature audit reports in `outputs/validation/analysis/player_features/`

## Usage

```bash
python features/extract_possessions.py
python features/build_player_defense_dataset.py
python features/profile_player_defense_features.py
```

Or use the orchestration wrapper:

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_downstream_pipeline.ps1
```

## Outputs

- `data/features/possessions_with_360.parquet` - 11,810+ possession sequences
- `data/features/player_defensive_actions.parquet` - 57,637 defensive actions
