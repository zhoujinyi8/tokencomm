$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskStarted = Get-Date
$taskBudgetSeconds = 12600
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
    python cache_stage21_food101_tokens.py --device cpu --threads 4 --batch-size 2 --max-seconds 12000 | Out-Null
    $cacheExit = $LASTEXITCODE
    if ($cacheExit -ne 0 -and $cacheExit -ne 75) { throw "Stage-21 token cache exited with code $cacheExit" }
}
$stage21Complete = $false
if (Test-Path -LiteralPath $stage21Progress) {
    $stage21Complete = (Get-Content -LiteralPath $stage21Progress -Raw | ConvertFrom-Json).complete -eq $true
}
$stage21Summary = Join-Path $PSScriptRoot 'data/stage20_food101/token_cache_stage21/summary.json'
if ($stage21Complete -and -not (Test-Path -LiteralPath $stage21Summary)) {
    python finalize_stage21_token_cache.py | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Stage-21 token-cache finalization failed with code $LASTEXITCODE" }
}
$stage22Protocol = Join-Path $PSScriptRoot 'STAGE22_CONTEXTUAL_TOKCOM_PROTOCOL.json'
if ((Test-Path -LiteralPath $stage21Summary) -and -not (Test-Path -LiteralPath $stage22Protocol)) {
    python freeze_stage22_protocol.py | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Stage-22 protocol freeze failed with code $LASTEXITCODE" }
}
$stage22Status = Join-Path $PSScriptRoot 'runs/stage22_contextual_tokcom/status.json'
$stage22Complete = $false
if (Test-Path -LiteralPath $stage22Status) {
    $stage22Complete = (Get-Content -LiteralPath $stage22Status -Raw | ConvertFrom-Json).completed -eq $true
}
$remainingSeconds = [math]::Floor($taskBudgetSeconds - ((Get-Date) - $taskStarted).TotalSeconds)
if ((Test-Path -LiteralPath $stage22Protocol) -and -not $stage22Complete -and $remainingSeconds -ge 60) {
    python train_stage22_contextual_tokcom.py --threads 4 --max-seconds $remainingSeconds | Out-Null
    $trainingExit = $LASTEXITCODE
    if ($trainingExit -ne 0 -and $trainingExit -ne 75) { throw "Stage-22 training exited with code $trainingExit" }
}
python goal_4hour_status.py | Out-Null
