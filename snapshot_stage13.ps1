$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$logDir = Join-Path $PSScriptRoot 'work/stage13_scheduler'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = (Get-Date).ToString('o')
try {
    $output = & python snapshot_stage13.py 2>&1
    if ($LASTEXITCODE -ne 0) { throw "snapshot_stage13.py exited with $LASTEXITCODE" }
    Add-Content -LiteralPath (Join-Path $logDir 'four_hour_runs.log') -Encoding UTF8 -Value ("{0} OK {1}" -f $stamp, ($output -join ' '))
    $output
}
catch {
    Add-Content -LiteralPath (Join-Path $logDir 'four_hour_runs.log') -Encoding UTF8 -Value ("{0} ERROR {1}" -f $stamp, $_.Exception.Message)
    throw
}
