# System & Architecture Documentation

Technical system design, fixes, and architecture decisions.

## Documents

- **PIPELINE_FIX_SUMMARY.md** - Critical event ordering bug fix
  - Root cause: Events scrambled after phase labeling
  - Solution: Re-sort events at critical pipeline stages
  - Impact: 67-2,272× improvement in shot rate accuracy
  - Result: Analysis decision improved from HOLD → GO_BASELINE

- **CLEANUP_IMPLEMENTATION.md** - Cleanup procedure implementation
  - PowerShell and Python cleanup utilities
  - Selective cleaning by category (cache, tests, notebooks, logs, data, outputs)
  - Dry-run mode and safety features
  - Performance: ~100x faster by excluding .venv

- **DOCUMENTATION_CLEANUP_SUMMARY.md** - Documentation cleanup summary
  - Removed 7 obsolete/redundant markdown files (54.6 KB)
  - Consolidated transitional implementation guides
  - Automated cleanup tool for future maintenance
  - Clear documentation structure established

- **REPO_NAVIGATION.md** - Repository structure guide
  - File organization and navigation
  - Project structure overview

## Architecture

### Unidirectional Coordinate System

All data normalized to single attacking direction:
- **x=0**: Own goal (defensive side)
- **x=120**: Opponent goal (attacking side)
- Teams always attack left → right
- Both defensive teams and attacking teams use same coordinate space

### Data Pipeline Stages

1. **Stage 1**: Fetch raw StatsBomb data
2. **Stage 2**: Enrich with normalized coordinates + phase labels + targets
3. **Stage 3**: Extract features (possessions, player actions)
4. **Analysis**: Statistical validation (10 modules)

## Process Improvements

- Event ordering critical for 10-second target window
- Must re-sort after transformations (phases, targets)
- Unidirectional coords require per-team direction inference
- Counter-attack sequences visible in shot probability patterns

