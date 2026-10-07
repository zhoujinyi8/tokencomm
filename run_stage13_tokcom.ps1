$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mutex = [System.Threading.Mutex]::new($false, 'Local\TokenComStage13TokCom')
if (-not $mutex.WaitOne(0)) { exit 0 }
$logDir = Join-Path $PSScriptRoot 'work/stage13_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'orchestrator.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }
try {
    $statusPath = Join-Path $PSScriptRoot 'runs/stage13_tokcom_large/status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($status.completed -eq $true) {
            & python snapshot_stage13.py
            Disable-ScheduledTask -TaskName 'TokenCom-Stage13-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
            Write-Log 'Stage-13 already complete.'
            exit 0
        }
    }
    $existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and ($_.CommandLine -match 'cache_real_tokens\.py.*imagenette_stage10' -or $_.CommandLine -match 'train_stage13_tokcom\.py') }
    if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }
    $cacheRoot = Join-Path $PSScriptRoot 'data/imagenette_stage10/token_cache'
    if (-not (Test-Path (Join-Path $cacheRoot 'train.pt')) -or -not (Test-Path (Join-Path $cacheRoot 'val.pt'))) {
        Write-Log 'Starting/resuming VQGAN token cache for 8,150 development images.'
        & python cache_real_tokens.py --data data/imagenette_stage10 --device cpu --threads 4 --splits train val *>> (Join-Path $logDir 'cache.log')
        if ($LASTEXITCODE -ne 0) { throw "Token cache exited with $LASTEXITCODE" }
    }
    Write-Log 'Starting/resuming 6,000-step ContextMaskGIT training.'
    & python train_stage13_tokcom.py *>> (Join-Path $logDir 'training.log')
    if ($LASTEXITCODE -ne 0) { throw "Stage-13 trainer exited with $LASTEXITCODE" }
    & python snapshot_stage13.py
    Disable-ScheduledTask -TaskName 'TokenCom-Stage13-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
    Write-Log 'Stage-13 completed.'
}
catch { Write-Log ("ERROR " + $_.Exception.Message); throw }
finally { $mutex.ReleaseMutex(); $mutex.Dispose() }
