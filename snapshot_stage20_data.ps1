$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python snapshot_stage20_data.py | Out-Null
