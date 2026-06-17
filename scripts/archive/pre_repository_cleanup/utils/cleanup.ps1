# Cleanup script for defensive-action-expected project
# PowerShell version for Windows users

param(
    [switch]$DryRun,
    [switch]$Cache,
    [switch]$TestArtifacts,
    [switch]$Notebooks,
    [switch]$Logs,
    [switch]$Data,
    [switch]$Outputs,
    [switch]$All,
    [switch]$Full,
    [switch]$Help
)

# Get project root (two levels up from this script)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))

function Show-Help {
    Write-Host @"

Cleanup Utility for defensive-action-expected Project
======================================================

Usage:
    .\cleanup.ps1 [OPTIONS]

Options:
    -DryRun          Show what would be cleaned without removing files
    -Cache           Clean Python cache (__pycache__, .pyc files)
    -TestArtifacts   Clean test artifacts (.pytest_cache, .mypy_cache, coverage)
    -Notebooks       Clean Jupyter artifacts (.ipynb_checkpoints)
    -Logs            Clean log files
    -Data            Clean processed data directories (keeps raw data)
    -Outputs         Clean output directories
    -All             Clean cache, test artifacts, notebooks, and logs
    -Full            Clean everything including data and outputs (USE WITH CAUTION)
    -Help            Show this help message

Examples:
    # Dry run to see what would be cleaned
    .\cleanup.ps1 -All -DryRun

    # Clean cache and test artifacts
    .\cleanup.ps1 -Cache -TestArtifacts

    # Full cleanup (everything)
    .\cleanup.ps1 -Full

"@
}

function Remove-ItemSafely {
    param(
        [string]$Path,
        [bool]$IsDryRun,
        [string]$Type = "Item"
    )

    if (Test-Path $Path) {
        if ($IsDryRun) {
            Write-Host "  [DRY RUN] Would remove $Type : $Path" -ForegroundColor Yellow
            return 1
        } else {
            Remove-Item $Path -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  Removed $Type : $Path" -ForegroundColor Green
            return 1
        }
    }
    return 0
}

function Clean-PythonCache {
    param([bool]$DryRun)

    Write-Host "`n=== Cleaning Python Cache ===" -ForegroundColor Cyan

    $count = 0

    # Find and remove __pycache__ directories (excluding .venv and .git)
    Write-Host "`nSearching for __pycache__ directories..."
    $allDirs = Get-ChildItem -Path $ProjectRoot -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue
    $pycacheDirs = $allDirs | Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }

    if ($pycacheDirs) {
        Write-Host "Found $($pycacheDirs.Count) __pycache__ directories"
        foreach ($dir in $pycacheDirs) {
            $count += Remove-ItemSafely -Path $dir.FullName -IsDryRun $DryRun -Type "directory"
        }
    } else {
        Write-Host "No __pycache__ directories found"
    }

    # Find and remove .pyc files (excluding .venv and .git)
    Write-Host "`nSearching for .pyc files..."
    $allFiles = @()
    $allFiles += Get-ChildItem -Path $ProjectRoot -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue
    $allFiles += Get-ChildItem -Path $ProjectRoot -Filter "*.pyo" -Recurse -ErrorAction SilentlyContinue
    $allFiles += Get-ChildItem -Path $ProjectRoot -Filter "*.pyd" -Recurse -ErrorAction SilentlyContinue
    $pycFiles = $allFiles | Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }

    if ($pycFiles) {
        Write-Host "Found $($pycFiles.Count) compiled Python files"
        foreach ($file in $pycFiles) {
            $count += Remove-ItemSafely -Path $file.FullName -IsDryRun $DryRun -Type "file"
        }
    } else {
        Write-Host "No .pyc files found"
    }

    Write-Host "`nCleaned $count items"
}

function Clean-TestArtifacts {
    param([bool]$DryRun)

    Write-Host "`n=== Cleaning Test Artifacts ===" -ForegroundColor Cyan

    $count = 0

    # .pytest_cache (excluding .venv and .git)
    Write-Host "`nSearching for .pytest_cache directories..."
    $allDirs = Get-ChildItem -Path $ProjectRoot -Directory -Filter ".pytest_cache" -Recurse -ErrorAction SilentlyContinue
    $pytestDirs = $allDirs | Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }
    if ($pytestDirs) {
        Write-Host "Found $($pytestDirs.Count) .pytest_cache directories"
        foreach ($dir in $pytestDirs) {
            $count += Remove-ItemSafely -Path $dir.FullName -IsDryRun $DryRun -Type "directory"
        }
    } else {
        Write-Host "No .pytest_cache directories found"
    }

    # .mypy_cache (excluding .venv and .git)
    Write-Host "`nSearching for .mypy_cache directories..."
    $allDirs = Get-ChildItem -Path $ProjectRoot -Directory -Filter ".mypy_cache" -Recurse -ErrorAction SilentlyContinue
    $mypyDirs = $allDirs | Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }
    if ($mypyDirs) {
        Write-Host "Found $($mypyDirs.Count) .mypy_cache directories"
        foreach ($dir in $mypyDirs) {
            $count += Remove-ItemSafely -Path $dir.FullName -IsDryRun $DryRun -Type "directory"
        }
    } else {
        Write-Host "No .mypy_cache directories found"
    }

    # Coverage files (excluding .venv and .git)
    Write-Host "`nSearching for coverage files..."
    $allFiles = Get-ChildItem -Path $ProjectRoot -Filter ".coverage*" -Recurse -ErrorAction SilentlyContinue
    $allDirs = Get-ChildItem -Path $ProjectRoot -Directory -Filter "htmlcov" -Recurse -ErrorAction SilentlyContinue

    $coverageFiles = $allFiles | Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }
    $coverageDirs = $allDirs | Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.FullName -notlike "*\.git\*" }

    if ($coverageFiles -or $coverageDirs) {
        $fileCount = if ($coverageFiles) { $coverageFiles.Count } else { 0 }
        $dirCount = if ($coverageDirs) { $coverageDirs.Count } else { 0 }
        $totalCount = $fileCount + $dirCount
        Write-Host "Found $totalCount coverage items"

        foreach ($file in $coverageFiles) {
            $count += Remove-ItemSafely -Path $file.FullName -IsDryRun $DryRun -Type "file"
        }
        foreach ($dir in $coverageDirs) {
            $count += Remove-ItemSafely -Path $dir.FullName -IsDryRun $DryRun -Type "directory"
        }
    } else {
        Write-Host "No coverage files found"
    }

    Write-Host "`nCleaned $count items"
}

function Clean-NotebookArtifacts {
    param([bool]$DryRun)

    Write-Host "`n=== Cleaning Notebook Artifacts ===" -ForegroundColor Cyan

    $count = 0

    Write-Host "`nSearching for .ipynb_checkpoints directories..."
    $checkpointDirs = Get-ChildItem -Path $ProjectRoot -Directory -Filter ".ipynb_checkpoints" -Recurse -ErrorAction SilentlyContinue

    if ($checkpointDirs) {
        Write-Host "Found $($checkpointDirs.Count) .ipynb_checkpoints directories"
        foreach ($dir in $checkpointDirs) {
            $count += Remove-ItemSafely -Path $dir.FullName -IsDryRun $DryRun -Type "directory"
        }
    } else {
        Write-Host "No .ipynb_checkpoints directories found"
    }

    Write-Host "`nCleaned $count items"
}

function Clean-Logs {
    param([bool]$DryRun)

    Write-Host "`n=== Cleaning Log Files ===" -ForegroundColor Cyan

    $count = 0

    Write-Host "`nSearching for log files..."
    $logFiles = Get-ChildItem -Path $ProjectRoot -Filter "*.log" -Recurse -ErrorAction SilentlyContinue

    if ($logFiles) {
        Write-Host "Found $($logFiles.Count) log files"
        foreach ($file in $logFiles) {
            $count += Remove-ItemSafely -Path $file.FullName -IsDryRun $DryRun -Type "file"
        }
    } else {
        Write-Host "No log files found"
    }

    Write-Host "`nCleaned $count items"
}

function Clean-DataDirectories {
    param([bool]$DryRun)

    Write-Host "`n=== Cleaning Data Directories ===" -ForegroundColor Cyan
    Write-Host "(Keeping raw data)"

    $dataDir = Join-Path $ProjectRoot "data"
    $subdirs = @("processed", "features", "models", "validation")

    $totalCount = 0

    foreach ($subdir in $subdirs) {
        $path = Join-Path $dataDir $subdir

        if (Test-Path $path) {
            Write-Host "`nCleaning data/$subdir..."

            $items = Get-ChildItem -Path $path -Exclude ".gitkeep" -ErrorAction SilentlyContinue

            if ($items) {
                $count = $items.Count
                Write-Host "  Found $count items to clean"

                foreach ($item in $items) {
                    $totalCount += Remove-ItemSafely -Path $item.FullName -IsDryRun $DryRun -Type $(if ($item.PSIsContainer) { "directory" } else { "file" })
                }

                # Ensure .gitkeep exists
                $gitkeep = Join-Path $path ".gitkeep"
                if (-not (Test-Path $gitkeep) -and -not $DryRun) {
                    New-Item -Path $gitkeep -ItemType File -Force | Out-Null
                    Write-Host "  Created .gitkeep" -ForegroundColor Green
                }
            } else {
                Write-Host "  Directory already clean"
            }
        }
    }

    Write-Host "`nCleaned $totalCount items"
}

function Clean-OutputDirectories {
    param([bool]$DryRun)

    Write-Host "`n=== Cleaning Output Directories ===" -ForegroundColor Cyan

    $outputsDir = Join-Path $ProjectRoot "outputs"
    $totalCount = 0

    if (Test-Path $outputsDir) {
        $subdirs = Get-ChildItem -Path $outputsDir -Directory -ErrorAction SilentlyContinue

        foreach ($subdir in $subdirs) {
            Write-Host "`nCleaning outputs/$($subdir.Name)..."

            $items = Get-ChildItem -Path $subdir.FullName -Recurse -ErrorAction SilentlyContinue

            if ($items) {
                $count = $items.Count
                Write-Host "  Found $count items to clean"

                # Remove all contents
                foreach ($item in $items) {
                    $totalCount += Remove-ItemSafely -Path $item.FullName -IsDryRun $DryRun -Type $(if ($item.PSIsContainer) { "directory" } else { "file" })
                }
            } else {
                Write-Host "  Directory already clean"
            }
        }
    } else {
        Write-Host "No outputs directory found"
    }

    Write-Host "`nCleaned $totalCount items"
}

# Main script logic
if ($Help) {
    Show-Help
    exit 0
}

Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n*** DRY RUN MODE - No files will be removed ***" -ForegroundColor Yellow
}

# Determine what to clean
$CleanAll = $All -or $Full
$CleanFull = $Full

# If no options specified, show help
if (-not ($Cache -or $TestArtifacts -or $Notebooks -or $Logs -or $Data -or $Outputs -or $All -or $Full)) {
    Show-Help
    Write-Host "`n⚠ No cleanup options specified. Use -All for standard cleanup.`n" -ForegroundColor Yellow
    exit 0
}

# Execute cleanup operations
if ($Cache -or $CleanAll) {
    Clean-PythonCache -DryRun $DryRun
}

if ($TestArtifacts -or $CleanAll) {
    Clean-TestArtifacts -DryRun $DryRun
}

if ($Notebooks -or $CleanAll) {
    Clean-NotebookArtifacts -DryRun $DryRun
}

if ($Logs -or $CleanAll) {
    Clean-Logs -DryRun $DryRun
}

if ($Data -or $CleanFull) {
    Clean-DataDirectories -DryRun $DryRun
}

if ($Outputs -or $CleanFull) {
    Clean-OutputDirectories -DryRun $DryRun
}

Write-Host "`n=== Cleanup Complete ===" -ForegroundColor Green

if ($DryRun) {
    Write-Host "This was a dry run. Run without -DryRun to actually remove files." -ForegroundColor Yellow
}

Write-Host ""




