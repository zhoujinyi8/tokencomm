$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage13_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'dispatcher.log'
function Write-Log([string]$Message) { Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message) }

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and ($_.CommandLine -match 'cache_real_tokens\.py.*(imagenette_stage10|stage14_tokcom_final_holdout)' -or $_.CommandLine -match 'train_stage13_tokcom\.py' -or $_.CommandLine -match 'report_stage13\.py' -or $_.CommandLine -match 'evaluate_stage13_physical\.py' -or $_.CommandLine -match 'report_stage13_physical\.py' -or $_.CommandLine -match 'report_fair_comparison\.py' -or $_.CommandLine -match '(freeze|prepare|evaluate|report)_stage14')
}
if ($existing) { Write-Log "Worker already active: $($existing.ProcessId -join ',')"; exit 0 }

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
            exit 0
        }
        $physicalStatusPath = Join-Path $PSScriptRoot 'runs/stage13_physical_dev/status.json'
        if (-not (Test-Path -LiteralPath $physicalStatusPath) -or -not ((Get-Content -LiteralPath $physicalStatusPath -Raw | ConvertFrom-Json).completed)) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('evaluate_stage13_physical.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "physical-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "physical-$stamp.stderr.log") | Out-Null
            Write-Log 'Started/resumed one physical-link development condition.'
            exit 0
        }
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'STAGE13_PHYSICAL_REPORT.md'))) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('report_stage13_physical.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "physical-report-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "physical-report-$stamp.stderr.log") | Out-Null
            Write-Log 'Started physical-link report worker.'
            exit 0
        }
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'TOKCOM_TAICHI_FAIR_COMPARISON.md'))) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('report_fair_comparison.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "fair-comparison-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "fair-comparison-$stamp.stderr.log") | Out-Null
            Write-Log 'Started cross-path fair-comparison report worker.'
            exit 0
        }
        $stage14Protocol = Join-Path $PSScriptRoot 'STAGE14_TOKCOM_FINAL_PROTOCOL.json'
        if (-not (Test-Path -LiteralPath $stage14Protocol)) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('freeze_stage14_protocol.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "stage14-freeze-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "stage14-freeze-$stamp.stderr.log") | Out-Null
            Write-Log 'Started Stage-14 blind-test protocol freeze.'
            exit 0
        }
        $stage14Data = Join-Path $PSScriptRoot 'data/stage14_tokcom_final_holdout/identity.json'
        if (-not (Test-Path -LiteralPath $stage14Data)) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('prepare_stage14_holdout.py', '--validation-parquet', 'work/stage12_source/validation-00000-of-00001.parquet') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "stage14-prepare-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "stage14-prepare-$stamp.stderr.log") | Out-Null
            Write-Log 'Started Stage-14 blind holdout materialization after protocol freeze.'
            exit 0
        }
        $stage14Cache = Join-Path $PSScriptRoot 'data/stage14_tokcom_final_holdout/token_cache/test.pt'
        if (-not (Test-Path -LiteralPath $stage14Cache)) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('cache_real_tokens.py', '--data', 'data/stage14_tokcom_final_holdout', '--device', 'cpu', '--threads', '4', '--max-seconds', '45', '--splits', 'test') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "stage14-cache-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "stage14-cache-$stamp.stderr.log") | Out-Null
            Write-Log 'Started/resumed Stage-14 blind token cache.'
            exit 0
        }
        $stage14Status = Join-Path $PSScriptRoot 'runs/stage14_tokcom_final/status.json'
        if (-not (Test-Path -LiteralPath $stage14Status) -or -not ((Get-Content -LiteralPath $stage14Status -Raw | ConvertFrom-Json).completed)) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('evaluate_stage14_final.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "stage14-evaluate-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "stage14-evaluate-$stamp.stderr.log") | Out-Null
            Write-Log 'Started/resumed one Stage-14 blind-test condition.'
            exit 0
        }
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'STAGE14_TOKCOM_FINAL_REPORT.md'))) {
            $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
            Start-Process -FilePath (Get-Command python).Source -ArgumentList @('report_stage14_final.py') -WorkingDirectory $PSScriptRoot `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "stage14-report-$stamp.stdout.log") `
                -RedirectStandardError (Join-Path $logDir "stage14-report-$stamp.stderr.log") | Out-Null
            Write-Log 'Started Stage-14 blind-test report worker.'
            exit 0
        }
        Disable-ScheduledTask -TaskName 'TokenCom-Stage13-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
        Write-Log 'Stage-13 development and Stage-14 preregistered blind test complete; disabled orchestrator.'
        exit 0
    }
}

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
