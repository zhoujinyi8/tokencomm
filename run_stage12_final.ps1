$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mutex = [System.Threading.Mutex]::new($false, 'Local\TokenComStage12FinalEvaluator')
if (-not $mutex.WaitOne(0)) { exit 0 }
$logDir = Join-Path $PSScriptRoot 'work/stage12_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'orchestrator.log'
function Write-Log([string]$Message) {
    Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message)
}
try {
    $statusPath = Join-Path $PSScriptRoot 'runs/stage12_final_test/status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($status.completed -eq $true) {
            Write-Log 'Stage-12 already completed; generating report and disabling task.'
            & python report_stage12_final.py
            Disable-ScheduledTask -TaskName 'TokenCom-Stage12-Final-Evaluation' -ErrorAction SilentlyContinue | Out-Null
            exit 0
        }
    }
    $existing = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match 'evaluate_stage12_final\.py'
    }
    if ($existing) {
        Write-Log "Evaluator already has Python PID(s) $($existing.ProcessId -join ','); refusing duplicate launch."
        exit 0
    }
    $attempt = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdout = Join-Path $logDir "evaluation-$attempt.stdout.log"
    $stderr = Join-Path $logDir "evaluation-$attempt.stderr.log"
    Write-Log "Starting or resuming Stage-12 evaluator; stdout=$stdout stderr=$stderr"
    $process = Start-Process -FilePath (Get-Command python).Source -ArgumentList @('evaluate_stage12_final.py') `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        Write-Log "Evaluator exited with code $($process.ExitCode); the next scheduled run will resume."
        exit $process.ExitCode
    }
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($status.completed -ne $true) { throw 'Evaluator exited without completed status' }
    & python report_stage12_final.py
    if ($LASTEXITCODE -ne 0) { throw "Stage-12 reporter exited with $LASTEXITCODE" }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'snapshot_stage12.ps1')
    Write-Log 'Stage-12 evaluation and report completed.'
    Disable-ScheduledTask -TaskName 'TokenCom-Stage12-Final-Evaluation' -ErrorAction SilentlyContinue | Out-Null
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
