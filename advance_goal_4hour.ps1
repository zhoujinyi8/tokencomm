$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python snapshot_stage19.py | Out-Null
python snapshot_stage20_data.py | Out-Null
$stage19Status = Join-Path $PSScriptRoot 'runs/stage19_v4_link/status.json'
$stage19Summary = Join-Path $PSScriptRoot 'STAGE19_V4_LINK_SUMMARY.json'
if ((Test-Path -LiteralPath $stage19Status) -and -not (Test-Path -LiteralPath $stage19Summary)) {
    $status = Get-Content -LiteralPath $stage19Status -Raw | ConvertFrom-Json
    if ($status.completed -eq $true) { python finalize_stage19.py | Out-Null }
}
$stage21Protocol = Join-Path $PSScriptRoot 'STAGE21_TOKEN_CACHE_PROTOCOL.json'
$stage21Progress = Join-Path $PSScriptRoot 'data/stage20_food101/token_cache_stage21/progress.json'
$stage21Complete = $false
if (Test-Path -LiteralPath $stage21Progress) {
    $stage21Complete = (Get-Content -LiteralPath $stage21Progress -Raw | ConvertFrom-Json).complete -eq $true
}
if ((Test-Path -LiteralPath $stage21Protocol) -and -not $stage21Complete) {
    python cache_stage21_food101_tokens.py --device cpu --threads 4 --batch-size 2 --max-seconds 12600 | Out-Null
    $cacheExit = $LASTEXITCODE
    if ($cacheExit -ne 0 -and $cacheExit -ne 75) { throw "Stage-21 token cache exited with code $cacheExit" }
}
python goal_4hour_status.py | Out-Null
