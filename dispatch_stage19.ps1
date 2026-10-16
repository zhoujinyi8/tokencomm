$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage19_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dispatcher.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'evaluate_stage19_v4_link\.py'
}
if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }
$statusPath = Join-Path $PSScriptRoot 'runs/stage19_v4_link/status.json'
if (Test-Path -LiteralPath $statusPath) {
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.completed -eq $true) {
        Disable-ScheduledTask -TaskName 'TokenCom-Stage19-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
        Write-Log 'Stage-19 perceptual evaluation complete; disabled orchestrator.'
        exit 0
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath (Get-Command python).Source `
    -ArgumentList @('evaluate_stage19_v4_link.py', '--max-seconds', '220', '--threads', '4') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "evaluate-$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "evaluate-$stamp.stderr.log")
Write-Log "Started Stage-19 worker PID=$($process.Id)."

