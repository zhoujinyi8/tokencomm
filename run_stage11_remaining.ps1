param([int]$WaitPid = 0)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage11_orchestrator'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'orchestrator.log'

function Write-StageLog([string]$Message) {
    $line = ('{0:o} {1}' -f (Get-Date), $Message)
    Add-Content -LiteralPath $log -Value $line
}

if ($WaitPid -gt 0) {
    Write-StageLog "Waiting for BAN+KAN PID $WaitPid"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}

$baselineStatus = 'runs/stage11_ablation_ban_kan/status.json'
if (-not (Test-Path -LiteralPath $baselineStatus)) {
    Write-StageLog 'BAN+KAN ended without status.json; stopping.'
    exit 2
}
$baseline = Get-Content -LiteralPath $baselineStatus -Raw | ConvertFrom-Json
if ($baseline.completed -ne $true) {
    Write-StageLog 'BAN+KAN status is not completed; stopping.'
    exit 3
}

$variants = @('ban_mlp', 'concat_mlp')
foreach ($variant in $variants) {
    $out = "runs/stage11_ablation_$variant"
    $statusPath = Join-Path $out 'status.json'
    if (Test-Path -LiteralPath $statusPath) {
        $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($status.completed -eq $true) {
            Write-StageLog "$variant already completed; skipping."
            continue
        }
        Write-StageLog "$variant has an incomplete existing directory; stopping."
        exit 4
    }
    if (Test-Path -LiteralPath $out) {
        Write-StageLog "$variant output exists without completed status; stopping."
        exit 5
    }
    Write-StageLog "Starting $variant"
    & python train_qwen_dual_taichi.py `
        --data data/imagenette_stage10/dual_vision_features `
        --out $out `
        --steps 800 `
        --batch-size 4 `
        --accumulate 2 `
        --eval-every 200 `
        --eval-n 100 `
        --lora-rank 8 `
        --fusion-variant $variant `
        --threads 4 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$variant.log")
    if ($LASTEXITCODE -ne 0) {
        Write-StageLog "$variant failed with exit code $LASTEXITCODE; stopping."
        exit $LASTEXITCODE
    }
    if (-not (Test-Path -LiteralPath $statusPath)) {
        Write-StageLog "$variant exited without status.json; stopping."
        exit 6
    }
    Write-StageLog "$variant completed."
}
Write-StageLog 'All Stage-11 variants completed.'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'snapshot_stage11.ps1')
if ($LASTEXITCODE -ne 0) {
    Write-StageLog "Final Stage-11 report generation failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Write-StageLog 'Stage-11 final report generated.'
