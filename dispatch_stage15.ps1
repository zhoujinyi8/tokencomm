$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage15_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dispatcher.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_stage15_contextual_tokcom\.py'
}
if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }

$statusPath = Join-Path $PSScriptRoot 'runs/stage15_contextual_tokcom/status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.completed -eq $true) {
        Disable-ScheduledTask -TaskName 'TokenCom-Stage15-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
        Write-Log 'Stage-15 contextual training complete; disabled orchestrator.'
        exit 0
    }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath (Get-Command python).Source `
    -ArgumentList @('train_stage15_contextual_tokcom.py', '--max-seconds', '240') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "train-$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "train-$stamp.stderr.log")
Write-Log "Started Stage-15 worker PID=$($process.Id)."
