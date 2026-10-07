$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage13_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dispatcher.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }

$statusPath = Join-Path $PSScriptRoot 'runs/stage13_tokcom_large/status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.completed -eq $true) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'STAGE13_TOKCOM_REPORT.md'))) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('report_stage13.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "report-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "report-$stamp.stderr.log") | Out-Null
            Write-Log 'Started final report worker.'
        }
        exit 0
    }
}

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and ($_.CommandLine -match 'cache_real_tokens\.py.*imagenette_stage10' -or $_.CommandLine -match 'train_stage13_tokcom\.py')
}
if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$cacheRoot = Join-Path $PSScriptRoot 'data/imagenette_stage10/token_cache'
if (-not (Test-Path (Join-Path $cacheRoot 'train.pt')) -or -not (Test-Path (Join-Path $cacheRoot 'val.pt'))) {
    $arguments = @('cache_real_tokens.py', '--data', 'data/imagenette_stage10', '--device', 'cpu', '--threads', '4', '--max-seconds', '45', '--splits', 'train', 'val')
    $kind = 'cache'
}
else {
    $arguments = @('train_stage13_tokcom.py', '--max-seconds', '45')
    $kind = 'train'
}
$process = Start-Process -FilePath (Get-Command python).Source -ArgumentList $arguments -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir "$kind-$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "$kind-$stamp.stderr.log")
Write-Log "Started $kind worker PID=$($process.Id)."
