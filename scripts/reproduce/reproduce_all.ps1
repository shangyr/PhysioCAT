$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $Root
try {
    python scripts/reproduce/reproduce_all.py
} finally {
    Pop-Location
}
