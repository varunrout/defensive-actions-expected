param(
    [int]$MaxRows = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$python = ".\.venv\Scripts\python.exe"

if ($MaxRows -gt 0) {
    & $python "scripts\models\train_baseline_logistic.py" --max-rows $MaxRows
} else {
    & $python "scripts\models\train_baseline_logistic.py"
}

& $python "scripts\models\evaluate_baseline_model.py"
Write-Host "Baseline modeling run complete." -ForegroundColor Green

