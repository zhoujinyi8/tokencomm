param([switch]$Quiet)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$logDir = Join-Path $PSScriptRoot 'work/stage11_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'four_hour_runs.log'
$stamp = (Get-Date).ToString('o')

try {
    $output = & python report_stage11.py --timestamp $stamp 2>&1
    if ($LASTEXITCODE -ne 0) { throw "report_stage11.py exited with $LASTEXITCODE" }
    Add-Content -LiteralPath $log -Value ("{0} OK {1}" -f $stamp, ($output -join ' '))
    if (-not $Quiet) { $output }
}
catch {
    Add-Content -LiteralPath $log -Value ("{0} ERROR {1}" -f $stamp, $_.Exception.Message)
    throw
}
