$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python snapshot_stage16.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
