param(
    [string]$PythonExe = "",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $PythonExe) {
    $PythonExe = (Join-Path $repoRoot ".venv\Scripts\python.exe")
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$steps = @()
$steps += [pscustomobject]@{ Name = "extract_possessions"; Script = "scripts/extract_possessions.py" }
$steps += [pscustomobject]@{ Name = "build_player_defense_dataset"; Script = "scripts/build_player_defense_dataset.py" }
$steps += [pscustomobject]@{ Name = "run_player_feature_analysis"; Script = "scripts/run_player_feature_analysis.py" }
$steps += [pscustomobject]@{ Name = "visualize_possessions"; Script = "scripts/visualize_possessions.py" }

Write-Host ("=" * 72)
Write-Host "RUN DOWNSTREAM PIPELINE"
Write-Host "Repo: $repoRoot"
Write-Host "Python: $PythonExe"
Write-Host ("=" * 72)

$pipelineTimer = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $repoRoot
try {
    foreach ($step in $steps) {
        $scriptPath = Join-Path $repoRoot $step.Script
        if (-not (Test-Path -LiteralPath $scriptPath)) {
            throw "Missing script: $scriptPath"
        }

        Write-Host ""
        Write-Host ("[Step] {0}" -f $step.Name)
        Write-Host ("  Command: {0} {1}" -f $PythonExe, $step.Script)

        if ($DryRun) {
            continue
        }

        $stepTimer = [System.Diagnostics.Stopwatch]::StartNew()
        & $PythonExe $step.Script
        $exitCode = $LASTEXITCODE
        $stepTimer.Stop()

        if ($exitCode -ne 0) {
            throw ("Step failed with exit code {0}: {1}" -f $exitCode, $step.Name)
        }

        Write-Host ("  Done in {0:n1}s" -f $stepTimer.Elapsed.TotalSeconds)
    }
}
finally {
    Pop-Location
    $pipelineTimer.Stop()
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run complete. No scripts were executed."
}
else {
    Write-Host ""
    Write-Host ("Pipeline complete in {0:n1}s" -f $pipelineTimer.Elapsed.TotalSeconds)
}




