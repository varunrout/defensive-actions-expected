# Cleanup Procedure

This document describes the cleanup utilities available for the defensive-action-expected project.

## Overview

The project includes two cleanup utilities to help maintain a clean workspace:
- **PowerShell Script** (`scripts/utils/cleanup.ps1`) - Recommended for Windows users
- **Python Script** (`scripts/utils/cleanup.py`) - Cross-platform alternative

Both scripts provide the same functionality and can perform either a dry run (preview) or actual cleanup.

## Quick Start

### Standard Cleanup (Recommended)

Clean cache, test artifacts, notebooks, and logs without removing data:

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -All
```

**Python:**
```bash
python scripts/utils/cleanup.py --all
```

### Dry Run (Preview)

See what would be cleaned without removing anything:

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -All -DryRun
```

**Python:**
```bash
python scripts/utils/cleanup.py --all --dry-run
```

### Full Cleanup (USE WITH CAUTION)

Clean everything including processed data and outputs:

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -Full
```

**Python:**
```bash
python scripts/utils/cleanup.py --full
```

## Cleanup Categories

### 1. Python Cache (`-Cache` / `--cache`)

Removes Python bytecode and cache directories:
- `__pycache__/` directories
- `*.pyc` files
- `*.pyo` files
- `*.pyd` files

**When to use:** After code changes, before committing, or when troubleshooting import issues.

### 2. Test Artifacts (`-TestArtifacts` / `--test-artifacts`)

Removes testing-related cache and artifacts:
- `.pytest_cache/` directories
- `.mypy_cache/` directories
- `.coverage*` files
- `htmlcov/` directories

**When to use:** After running tests, or when test cache becomes stale.

### 3. Notebook Artifacts (`-Notebooks` / `--notebooks`)

Removes Jupyter notebook checkpoints:
- `.ipynb_checkpoints/` directories

**When to use:** After working with notebooks, or before committing notebook changes.

### 4. Log Files (`-Logs` / `--logs`)

Removes all log files:
- `*.log` files

**When to use:** When logs are no longer needed, or to free up disk space.

### 5. Data Directories (`-Data` / `--data`)

Cleans processed data directories (preserves raw data and `.gitkeep` files):
- `data/processed/`
- `data/features/`
- `data/models/`
- `data/validation/`

**When to use:** To regenerate all data from scratch, or after pipeline changes.

**Note:** Raw data (`data/raw/`) is never cleaned automatically.

### 6. Output Directories (`-Outputs` / `--outputs`)

Cleans all output directories:
- `outputs/models/`
- `outputs/oof/`
- `outputs/validation/`

**When to use:** To clear model outputs and validation results.

## Usage Examples

### Example 1: Clean Just Cache Files

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -Cache
```

**Python:**
```bash
python scripts/utils/cleanup.py --cache
```

### Example 2: Clean Cache and Test Artifacts

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -Cache -TestArtifacts
```

**Python:**
```bash
python scripts/utils/cleanup.py --cache --test-artifacts
```

### Example 3: Preview Full Cleanup

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -Full -DryRun
```

**Python:**
```bash
python scripts/utils/cleanup.py --full --dry-run
```

### Example 4: Clean Everything Except Data

**PowerShell:**
```powershell
.\scripts\utils\cleanup.ps1 -Cache -TestArtifacts -Notebooks -Logs -Outputs
```

**Python:**
```bash
python scripts/utils/cleanup.py --cache --test-artifacts --notebooks --logs --outputs
```

## Recommended Cleanup Workflows

### Before Committing Code

```powershell
# Clean cache and test artifacts
.\scripts\utils\cleanup.ps1 -Cache -TestArtifacts -Notebooks
```

### After Major Code Refactoring

```powershell
# Full cleanup and regenerate everything
.\scripts\utils\cleanup.ps1 -Full
.\scripts\pipeline\pipeline.py  # Regenerate data
```

### Weekly Maintenance

```powershell
# Standard cleanup
.\scripts\utils\cleanup.ps1 -All
```

### Before Archiving or Sharing Project

```powershell
# Full cleanup to reduce project size
.\scripts\utils\cleanup.ps1 -Full
```

## What Gets Preserved

The cleanup scripts are designed to be safe and preserve important files:

✅ **Always Preserved:**
- Source code (`.py` files)
- Notebooks (`.ipynb` files)
- Configuration files (`pyproject.toml`, `requirements.txt`, etc.)
- Documentation (`.md` files)
- Raw data (`data/raw/`)
- `.gitkeep` files (directory markers)
- Virtual environments (`.venv/`)

❌ **Can Be Cleaned:**
- Cache files (always safe to remove)
- Test artifacts (always safe to remove)
- Log files (always safe to remove)
- Processed data (can be regenerated)
- Model outputs (can be regenerated)

## Command Line Options

### PowerShell Options

```powershell
-DryRun          # Preview what would be cleaned
-Cache           # Clean Python cache
-TestArtifacts   # Clean test artifacts
-Notebooks       # Clean notebook checkpoints
-Logs            # Clean log files
-Data            # Clean processed data
-Outputs         # Clean output directories
-All             # Clean cache, tests, notebooks, and logs
-Full            # Clean everything including data and outputs
-Help            # Show help message
```

### Python Options

```bash
--dry-run          # Preview what would be cleaned
--cache            # Clean Python cache
--test-artifacts   # Clean test artifacts
--notebooks        # Clean notebook checkpoints
--logs             # Clean log files
--data             # Clean processed data
--outputs          # Clean output directories
--all              # Clean cache, tests, notebooks, and logs
--full             # Clean everything including data and outputs
```

## Safety Features

1. **Dry Run Mode**: Always preview changes before applying them
2. **Selective Cleaning**: Choose exactly what to clean
3. **Preserves Raw Data**: Raw data is never automatically cleaned
4. **Preserves Structure**: `.gitkeep` files maintain directory structure
5. **Error Handling**: Continues even if some files can't be removed

## Troubleshooting

### Issue: "Permission denied" errors

**Solution:** Close any applications that might have files open (Python interpreters, Jupyter, IDEs).

### Issue: Cleanup doesn't remove some files

**Possible causes:**
- Files are currently in use
- Insufficient permissions
- Files protected by OS

**Solution:** Close relevant applications and try again with administrator privileges if needed.

### Issue: Want to clean raw data too

**Solution:** Manually delete `data/raw/` contents (be careful!):
```powershell
Remove-Item -Path "data\raw\*" -Recurse -Exclude .gitkeep
```

## Integration with Git

The cleanup scripts respect `.gitignore` patterns. Files that should never be committed are typically safe to clean:

- `__pycache__/` → Ignored by git, safe to clean
- `.pytest_cache/` → Ignored by git, safe to clean
- `.ipynb_checkpoints/` → Usually ignored, safe to clean
- `*.log` → Ignored by git, safe to clean

## Automation

### Daily Cleanup Task (Windows)

Create a scheduled task to run cleanup daily:

```powershell
$trigger = New-ScheduledTaskTrigger -Daily -At 9am
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\Path\To\Project\scripts\utils\cleanup.ps1 -Cache -Logs"
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "CleanupDefensiveProject" -Description "Daily cleanup of defensive-action-expected project"
```

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/sh
python scripts/utils/cleanup.py --cache --notebooks
```

## See Also

- `.gitignore` - Files ignored by version control
- `scripts/utils/README.md` - Utility scripts documentation
- `docs/system/CLEANUP_SUMMARY.md` - Previous cleanup history

## Support

If you encounter issues with the cleanup scripts:
1. Try running with `-DryRun` / `--dry-run` first
2. Check that you have proper permissions
3. Close all applications using project files
4. Review the error messages for specific issues

For additional help or to report bugs, please create an issue.

