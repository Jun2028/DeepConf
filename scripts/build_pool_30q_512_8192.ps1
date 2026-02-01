# Build remaining traces for AIME 2025 so the pool covers 30 problems total.
# If problems 1-10 already exist, this will generate 11-30 (remaining 20).
# Uses the repo-local .venv and DEEPSEEK_API_KEY from .secrets/openai.env

$ErrorActionPreference = 'Stop'

# Ensure we run from repo root even if the script is launched elsewhere.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

$envFile = Join-Path $repoRoot '.secrets/openai.env'
if (-not (Test-Path $envFile)) {
    throw "Missing $envFile"
}

$line = Get-Content $envFile | Select-String -Pattern 'DEEPSEEK_API_KEY' | Select-Object -First 1
if (-not $line) {
    throw 'DEEPSEEK_API_KEY not found in .secrets/openai.env'
}

$key = ($line.ToString() -replace 'export DEEPSEEK_API_KEY="(.*)"','$1')
$env:DEEPSEEK_API_KEY = $key

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host 'Missing .venv. Creating it now...'
    & python -m venv .venv
    if (-not (Test-Path $python)) {
        throw 'Failed to create .venv. Ensure Python is on PATH.'
    }
    & $python -m pip install -r requirements.txt
}

& $python -m src.evaluate `
  --dataset data\aime_2025_rows.json `
  --limit_problems 30 `
  --n_pool 512 `
  --k 512 `
  --max_tokens 8192 `
  --pool_only `
  --concurrency 256
