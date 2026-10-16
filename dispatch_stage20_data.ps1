$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage20_data_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dispatcher.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match 'prepare_stage20_food101\.py'
}
if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }
$manifestPath = Join-Path $PSScriptRoot 'data/stage20_food101/manifest.json'
if (Test-Path -LiteralPath $manifestPath) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.status -eq 'prepared' -and $manifest.test_accessed -eq $false) {
        Disable-ScheduledTask -TaskName 'TokenCom-Stage20-Data' -ErrorAction SilentlyContinue | Out-Null
        Write-Log 'Stage-20 public data preparation complete; disabled orchestrator.'
        exit 0
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath (Get-Command python).Source `
    -ArgumentList @('prepare_stage20_food101.py', 'all') `
    -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "prepare-$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "prepare-$stamp.stderr.log")
Write-Log "Started Stage-20 data worker PID=$($process.Id)."
