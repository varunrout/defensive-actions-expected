param(
    [string]$SourceDir = "outputs/validation/comparison/slices",
    [string]$ArchiveRoot = "outputs/validation/comparison/slices_archive",
    [string]$LatestDir = "outputs/validation/comparison/slices_latest",
    [switch]$SkipLatest,
    [switch]$SkipReadme
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$sourcePath = Join-Path $repoRoot $SourceDir
$archiveRootPath = Join-Path $repoRoot $ArchiveRoot
$latestPath = Join-Path $repoRoot $LatestDir

if (-not (Test-Path $sourcePath)) {
    throw "Source directory does not exist: $sourcePath"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$destPath = Join-Path $archiveRootPath $stamp

New-Item -ItemType Directory -Path $destPath -Force | Out-Null
Copy-Item (Join-Path $sourcePath "*") -Destination $destPath -Recurse -Force

if (-not $SkipLatest) {
    if (Test-Path $latestPath) {
        Remove-Item (Join-Path $latestPath "*") -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        New-Item -ItemType Directory -Path $latestPath -Force | Out-Null
    }
    Copy-Item (Join-Path $sourcePath "*") -Destination $latestPath -Recurse -Force
}

if (-not $SkipReadme) {
    $readmePath = Join-Path $destPath "README.md"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $lines = @(
        "# Slice Analysis Snapshot",
        "",
        "This folder is a timestamped snapshot of slice-comparison outputs.",
        "",
        "## Source",
        "",
        ("- Original live folder: {0}" -f $SourceDir),
        ("- Archive root: {0}" -f $ArchiveRoot),
        ("- Archived at: {0} (local time)" -f $timestamp),
        "",
        "## Included Files",
        ""
    )

    Get-ChildItem -Path $destPath -File |
        Where-Object { $_.Name -ne "README.md" } |
        Sort-Object Name |
        ForEach-Object {
            $lines += ("- {0}" -f $_.Name)
        }

    Set-Content -Path $readmePath -Value $lines -Encoding UTF8
}

Write-Host "Archived slice results to:" -ForegroundColor Green
Write-Host $destPath
if (-not $SkipLatest) {
    Write-Host "Updated latest snapshot at:" -ForegroundColor Green
    Write-Host $latestPath
}



