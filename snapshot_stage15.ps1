$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python snapshot_stage15.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
