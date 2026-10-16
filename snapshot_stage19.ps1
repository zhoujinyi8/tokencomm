$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python snapshot_stage19.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

