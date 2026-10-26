param(
    [switch]$WhatIf,
    [switch]$CachesOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$paths = @(
    (Join-Path $root ".runtime"),
    (Join-Path $root ".pytest_cache"),
    (Join-Path $root "__pycache__")
)

$paths += @(
    Get-ChildItem -Path $root -Directory -Recurse -Force -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch "\\(venv|data|storage|snapshots)(\\|$)" } |
        Select-Object -ExpandProperty FullName
)

$freshStartPaths = @(
    (Join-Path $root "data\raw"),
    (Join-Path $root "data\rag.db"),
    (Join-Path $root "data\rag.db-shm"),
    (Join-Path $root "data\rag.db-wal"),
    (Join-Path $root "storage"),
    (Join-Path $root "snapshots")
)

if (-not $CachesOnly) {
    $paths += $freshStartPaths
}

$paths = $paths | Where-Object { Test-Path $_ } | Select-Object -Unique

if ($WhatIf) {
    Write-Host "Cleanup preview:" -ForegroundColor Cyan
    if ($paths.Count -eq 0) {
        Write-Host "  Nothing to remove."
    } else {
        $paths | ForEach-Object { Write-Host "  Remove $_" }
    }
    if ($CachesOnly) {
        Write-Host "Mode: Caches only"
        Write-Host "Preserved: data, data\raw, storage, snapshots, venv, qdrant.exe"
    } else {
        Write-Host "Mode: Fresh start"
        Write-Host "Removes: data\raw, data\rag.db, data\rag.db-shm, data\rag.db-wal, storage, snapshots"
        Write-Host "Preserved: venv, qdrant.exe, source code, configuration, and requirements"
    }
    exit 0
}

foreach ($path in $paths) {
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path $path) {
        Write-Warning "Could not remove $path"
    } else {
        Write-Host "Removed $path"
    }
}

if (-not $CachesOnly) {
    New-Item -ItemType Directory -Path (Join-Path $root "data\raw") -Force | Out-Null
    Write-Host "Fresh start complete. Uploaded files, SQLite state, and Qdrant storage were removed." -ForegroundColor Green
} else {
    Write-Host "Cache cleanup complete. Project data and Qdrant storage were preserved." -ForegroundColor Green
}
