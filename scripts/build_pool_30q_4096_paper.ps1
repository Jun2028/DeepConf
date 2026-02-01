# Paper-parameter pool build for AIME 2025 (30 problems).
# Matches Table 11/Section F in the paper for DeepSeek-8B:
#   temperature=0.6, top_p=0.95, top_logprobs=20, max seq len=64k, pool=4096.
# Note: we omit top_p here per DeepSeek docs (use either temperature or top_p, not both).
# NOTE: For our current model (deepseek-chat), max_tokens is capped at 8192.

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
  --n_pool 4096 `
  --k 512 `
  --max_tokens 8192 `
  --temperature 0.6 `
  --top_logprobs 20 `
  --pool_only `
  --concurrency 8
