param([switch]$KeepTaskEnabled)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$mutex = [System.Threading.Mutex]::new($false, 'Local\TokenComStage11Orchestrator')
if (-not $mutex.WaitOne(0)) {
    exit 0
}

$logDir = Join-Path $PSScriptRoot 'work/stage11_orchestrator'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'orchestrator.log'

function Write-StageLog([string]$Message) {
    Add-Content -LiteralPath $log -Encoding UTF8 -Value ('{0:o} {1}' -f (Get-Date), $Message)
}

function Get-Completion([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    try {
        $status = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return $status.completed -eq $true
    }
    catch { return $false }
}

try {
    $variants = @('ban_kan', 'ban_mlp', 'concat_mlp')
    foreach ($variant in $variants) {
        $out = Join-Path $PSScriptRoot "runs/stage11_ablation_$variant"
        $statusPath = Join-Path $out 'status.json'
        $checkpoint = Join-Path $out 'last.pt'
        if (Get-Completion $statusPath) {
            Write-StageLog "$variant already completed; skipping."
            continue
        }

        # Task Scheduler can terminate this PowerShell parent while leaving a
        # Start-Process child alive. Never launch a second writer for the same
        # run directory; a later invocation will recheck after the child exits.
        $outArgument = "--out runs/stage11_ablation_$variant"
        $existing = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match '^python' -and
            $_.CommandLine -match 'train_qwen_dual_taichi\.py' -and
            $_.CommandLine.Contains($outArgument)
        }
        if ($existing) {
            Write-StageLog "$variant already has Python PID(s) $($existing.ProcessId -join ','); refusing duplicate launch."
            exit 0
        }

        $resumeArgs = @()
        if (Test-Path -LiteralPath $out) {
            if (Test-Path -LiteralPath $checkpoint) {
                Write-StageLog "Resuming $variant from last.pt."
                $resumeArgs = @('--resume', "runs/stage11_ablation_$variant/last.pt")
            }
            else {
                $resolvedRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'runs'))
                $resolvedOut = [System.IO.Path]::GetFullPath($out)
                if (-not $resolvedOut.StartsWith($resolvedRoot + [System.IO.Path]::DirectorySeparatorChar)) {
                    throw "Refusing to archive path outside runs: $resolvedOut"
                }
                $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
                $archive = "${out}_interrupted_$stamp"
                Write-StageLog "$variant has no checkpoint; archiving incomplete run to $archive."
                Move-Item -LiteralPath $out -Destination $archive
            }
        }

        Write-StageLog "Starting $variant."
        $arguments = @(
            'train_qwen_dual_taichi.py',
            '--data', 'data/imagenette_stage10/dual_vision_features',
            '--out', "runs/stage11_ablation_$variant",
            '--steps', '800',
            '--batch-size', '4',
            '--accumulate', '2',
            '--eval-every', '200',
            '--eval-n', '100',
            '--checkpoint-every', '50',
            '--lora-rank', '8',
            '--fusion-variant', $variant,
            '--threads', '4'
        ) + $resumeArgs
        # Windows PowerShell 5.1 can turn ordinary native stderr warnings into
        # terminating NativeCommandError records when ErrorActionPreference is
        # Stop. Redirect the two streams separately and trust the process code.
        $attempt = Get-Date -Format 'yyyyMMdd_HHmmss'
        $stdout = Join-Path $logDir "$variant-$attempt.stdout.log"
        $stderr = Join-Path $logDir "$variant-$attempt.stderr.log"
        $python = (Get-Command python).Source
        $process = Start-Process -FilePath $python -ArgumentList $arguments -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        if ($process.ExitCode -ne 0) {
            Write-StageLog "$variant failed with exit code $($process.ExitCode); stdout=$stdout stderr=$stderr; a later scheduled invocation will resume it."
            exit $process.ExitCode
        }
        if (-not (Get-Completion $statusPath)) {
            Write-StageLog "$variant exited without a completed status; stopping."
            exit 6
        }
        Write-StageLog "$variant completed."
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'snapshot_stage11.ps1') -Quiet
    }

    Write-StageLog 'All Stage-11 variants completed.'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'snapshot_stage11.ps1') -Quiet
    if ($LASTEXITCODE -ne 0) { throw "Final report generation failed with exit code $LASTEXITCODE" }
    Write-StageLog 'Stage-11 final report generated.'
    if (-not $KeepTaskEnabled) {
        Disable-ScheduledTask -TaskName 'TokenCom-Stage11-Orchestrator' -ErrorAction SilentlyContinue | Out-Null
        Write-StageLog 'Disabled completed Stage-11 orchestrator task.'
    }
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
