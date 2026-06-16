# Notebook Guide

This folder contains the current analysis notebooks used to inspect data quality, explore player defensive actions, compare target definitions, and surface tournament-level football findings.

## Notebook index

| # | File | Focus |
|---|------|-------|
| 1 | `01_data_audit.ipynb` | End-to-end audit of raw, processed, possession, and player-feature data assets. |
| 2 | `02_player_defensive_analysis.ipynb` | Player defensive action profiling across phases, positions, actions, space, and feature interactions. |
| 3 | `03_target_comparison_analysis.ipynb` | Comparison of `target_shot_in_10s` versus grid-based xT as defensive modeling targets. |
| 4 | `04_international_tournament_findings.ipynb` | Analyst-facing World Cup vs Euro findings: tournament danger, phase mix, play-pattern risk, team styles, and stage effects. |
| 5 | `05_stage_level_defensive_dynamics.ipynb` | Group Stage vs Knockout defensive shifts: phase distribution changes, intensity proxies, spatial risk evolution, team adaptation strategies. |
| 6 | `06_team_defensive_clustering.ipynb` | Unsupervised K-means clustering of teams by defensive characteristics. Identifies team archetypes, phase preferences, and cluster performance profiles. |
| 7 | `07_player_defensive_archetypes.ipynb` | Unsupervised K-means clustering of individual defenders by playing style across phases, actions, intensity, and spatial patterns. |

## Regenerate notebook files

All seven notebooks can be rebuilt from the repository root with:

```powershell
python build_comprehensive_notebooks.py
```

## Run notebooks

From the repository root:

```powershell
python -m jupyter lab
```

Then open `notebooks/` and run cells from top to bottom.

## Notes

- `04_international_tournament_findings.ipynb` uses `data/features/player_defensive_actions.parquet` plus match metadata from `data/raw/matches/*.json`.
- The current notebook set in this folder is the authoritative list; older notebook names referenced in legacy documentation are no longer present here.

