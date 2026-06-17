# Utility Scripts

Helper and validation utilities for the defensive-action-expected project.

## Scripts

### validate.py

Quick validation checks on pipeline outputs.
- Verifies data schema, event ordering, coordinate ranges
- Checks for missing files or corrupted data

**Usage:**
```bash
python scripts/utils/validate.py
```

### cleanup.py

Cross-platform Python cleanup utility.
- Removes cache files, test artifacts, logs, and temporary files
- Can optionally clean processed data and outputs
- Supports dry-run mode to preview changes

**Usage:**
```bash
# Preview what would be cleaned
python scripts/utils/cleanup.py --all --dry-run

# Clean cache, test artifacts, notebooks, and logs
python scripts/utils/cleanup.py --all

# Full cleanup including data and outputs
python scripts/utils/cleanup.py --full
```

**Options:**
- `--dry-run` - Preview changes without removing files
- `--cache` - Clean Python cache files
- `--test-artifacts` - Clean test artifacts
- `--notebooks` - Clean Jupyter checkpoints
- `--logs` - Clean log files
- `--data` - Clean processed data
- `--outputs` - Clean output directories
- `--all` - Clean cache, tests, notebooks, and logs
- `--full` - Clean everything (use with caution)

### cleanup.ps1

PowerShell cleanup utility for Windows users.
- Same functionality as cleanup.py
- Native PowerShell with better Windows integration
- Color-coded output

**Usage:**
```powershell
# Preview what would be cleaned
.\scripts\utils\cleanup.ps1 -All -DryRun

# Clean cache, test artifacts, notebooks, and logs
.\scripts\utils\cleanup.ps1 -All

# Full cleanup including data and outputs
.\scripts\utils\cleanup.ps1 -Full
```

**Options:**
- `-DryRun` - Preview changes without removing files
- `-Cache` - Clean Python cache files
- `-TestArtifacts` - Clean test artifacts
- `-Notebooks` - Clean Jupyter checkpoints
- `-Logs` - Clean log files
- `-Data` - Clean processed data
- `-Outputs` - Clean output directories
- `-All` - Clean cache, tests, notebooks, and logs
- `-Full` - Clean everything (use with caution)
- `-Help` - Show help message

### cleanup_docs.py

Remove obsolete and redundant markdown documentation files.
- Identifies transitional/implementation guides from completed work
- Removes documentation that has been superseded
- Preserves all essential project documentation
- Supports dry-run mode

**Usage:**
```bash
# Preview which files would be removed
python scripts/utils/cleanup_docs.py --dry-run

# Remove obsolete documentation
python scripts/utils/cleanup_docs.py

# Show documentation structure after cleanup
python scripts/utils/cleanup_docs.py --list-remaining
```

**What it removes:**
- Transitional implementation guides (e.g., `IMPLEMENTATION.md`, `SWITCH_TO_XT_SUMMARY.md`)
- Obsolete status files (e.g., `PIPELINE_STATUS.md`)
- Duplicate/redundant documentation (e.g., duplicate xT implementation guides)
- Superseded summaries (e.g., old cleanup summaries)

**What it preserves:**
- Main project README
- Core documentation (modeling strategy, quick references, scope)
- Analysis documentation (zone analysis, insights, model portfolio report)
- System documentation (architecture, navigation, bug fixes)

## Documentation

For detailed information about cleanup procedures, see:
- `docs/CLEANUP_PROCEDURE.md` - Complete cleanup documentation with examples and workflows
- `CLEANUP_QUICK_REFERENCE.md` - Quick reference for cleanup commands

