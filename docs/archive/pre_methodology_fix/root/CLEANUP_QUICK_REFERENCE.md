# Quick Cleanup Reference

Fast reference for cleanup commands in the defensive-action-expected project.

## Most Common Commands

```powershell
# Quick standard cleanup (recommended for regular use)
.\scripts\utils\cleanup.ps1 -All

# Preview before cleaning
.\scripts\utils\cleanup.ps1 -All -DryRun

# Just clean cache (safest, fastest)
.\scripts\utils\cleanup.ps1 -Cache

# Clean before committing code
.\scripts\utils\cleanup.ps1 -Cache -TestArtifacts -Notebooks
```

## Quick Decision Tree

**Just finished coding?**
```powershell
.\scripts\utils\cleanup.ps1 -Cache
```

**Just ran tests?**
```powershell
.\scripts\utils\cleanup.ps1 -TestArtifacts
```

**Working with notebooks?**
```powershell
.\scripts\utils\cleanup.ps1 -Notebooks
```

**Weekly maintenance?**
```powershell
.\scripts\utils\cleanup.ps1 -All
```

**Starting fresh / Major changes?**
```powershell
.\scripts\utils\cleanup.ps1 -Full
```

**Want to clean obsolete docs?**
```powershell
python scripts/utils/cleanup_docs.py --dry-run
```

**Not sure what to do?**
```powershell
.\scripts\utils\cleanup.ps1 -All -DryRun
```

## Options Guide

| Option | What it cleans | Safe? | Use when |
|--------|---------------|-------|----------|
| `-Cache` | `__pycache__`, `.pyc` files | ✅ Always | After code changes |
| `-TestArtifacts` | `.pytest_cache`, `.mypy_cache`, coverage | ✅ Always | After running tests |
| `-Notebooks` | `.ipynb_checkpoints` | ✅ Always | After notebook work |
| `-Logs` | `*.log` files | ✅ Always | Regularly |
| `-Data` | Processed data (not raw) | ⚠️ Careful | Regenerating pipeline |
| `-Outputs` | Model outputs | ⚠️ Careful | Retraining models |
| `-All` | Cache + Tests + Notebooks + Logs | ✅ Safe | Regular maintenance |
| `-Full` | Everything including data/outputs | ⚠️ Caution | Fresh start |

## Python Alternative

Replace `.\scripts\utils\cleanup.ps1` with `python scripts/utils/cleanup.py` and use:
- `--all` instead of `-All`
- `--dry-run` instead of `-DryRun`
- `--cache` instead of `-Cache`
- etc.

## Documentation Cleanup

Remove obsolete markdown files (transitional guides, outdated status files):

```powershell
# Preview which docs would be removed
python scripts/utils/cleanup_docs.py --dry-run

# Remove obsolete documentation
python scripts/utils/cleanup_docs.py

# Show documentation structure
python scripts/utils/cleanup_docs.py --list-remaining
```

**What it removes:**
- Transitional implementation guides (completed work)
- Obsolete status files
- Duplicate/redundant documentation

**What it preserves:**
- All essential project documentation
- Analysis reports and findings
- System architecture docs

## One-Liners

```powershell
# Clean and regenerate everything
.\scripts\utils\cleanup.ps1 -Full ; python scripts/pipeline/pipeline.py

# Clean and run tests
.\scripts\utils\cleanup.ps1 -TestArtifacts ; pytest

# Clean and open notebooks
.\scripts\utils\cleanup.ps1 -Notebooks ; .\OPEN_NOTEBOOKS.ps1
```

## Help Commands

```powershell
# PowerShell help
.\scripts\utils\cleanup.ps1 -Help

# Python help
python scripts/utils/cleanup.py --help
```

## Full Documentation

See `docs/CLEANUP_PROCEDURE.md` for complete documentation, examples, and workflows.

