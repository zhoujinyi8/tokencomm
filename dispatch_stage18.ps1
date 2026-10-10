$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage18_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dispatcher.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'train_stage18_contextual_tokcom\.py'
}
if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }
$statusPath = Join-Path $PSScriptRoot 'runs/stage18_strong_cmi/status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.completed -eq $true) {
        Disable-ScheduledTask -TaskName 'TokenCom-Stage18-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
        Write-Log 'Stage-18 training complete; disabled orchestrator.'
        exit 0
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath (Get-Command python).Source `
    -ArgumentList @('train_stage18_contextual_tokcom.py', '--max-seconds', '220') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "train-$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "train-$stamp.stderr.log")
Write-Log "Started Stage-18 worker PID=$($process.Id)."
