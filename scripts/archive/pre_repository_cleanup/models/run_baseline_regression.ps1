param(
    [int]$MaxRows = 0,
    [string]$PythonExe = "",
    [switch]$SkipTrain,
    [switch]$SkipCompare,
    [string]$Variant = "v2_full_baseline"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $PythonExe) {
    $PythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$trainScript = Join-Path $repoRoot "scripts\models\train_baseline_regression.py"
$evalScript = Join-Path $repoRoot "scripts\models\evaluate_baseline_regression.py"
$compareScript = Join-Path $repoRoot "scripts\models\compare_classification_vs_regression.py"

$logisticModelPath = Join-Path $repoRoot "outputs\models\baseline\logistic_$Variant.joblib"
$regressionModelPath = Join-Path $repoRoot "outputs\models\regression\regression_$Variant.joblib"

Write-Host ("=" * 72)
Write-Host "RUN BASELINE REGRESSION WORKFLOW"
Write-Host "Repo: $repoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Variant: $Variant"
Write-Host ("=" * 72)

Push-Location $repoRoot
try {
    if (-not $SkipTrain) {
        Write-Host ""
        Write-Host "[1/3] Train baseline regression"
        if ($MaxRows -gt 0) {
            & $PythonExe $trainScript --max-rows $MaxRows
        }
        else {
            & $PythonExe $trainScript
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Regression training failed."
        }
    }

    Write-Host ""
    Write-Host "[2/3] Evaluate regression charts"
    & $PythonExe $evalScript
    if ($LASTEXITCODE -ne 0) {
        throw "Regression evaluation failed."
    }

    if (-not $SkipCompare) {
        Write-Host ""
        Write-Host "[3/3] Compare classification vs regression"
        if ((Test-Path -LiteralPath $logisticModelPath) -and (Test-Path -LiteralPath $regressionModelPath)) {
            if ($MaxRows -gt 0) {
                & $PythonExe $compareScript --variant $Variant --max-rows $MaxRows
            }
            else {
                & $PythonExe $compareScript --variant $Variant
            }
            if ($LASTEXITCODE -ne 0) {
                throw "Model comparison failed."
            }
        }
        else {
            Write-Warning "Skipping comparison because model artifacts are missing."
            Write-Host "  Missing logistic? $(-not (Test-Path -LiteralPath $logisticModelPath))"
            Write-Host "  Missing regression? $(-not (Test-Path -LiteralPath $regressionModelPath))"
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Regression workflow complete." -ForegroundColor Green

