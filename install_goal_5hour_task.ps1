param(
    [datetime]$StartAt = (Get-Date).AddHours(5)
)

$ErrorActionPreference = 'Stop'
$taskName = 'TokenCom-Goal-5Hour-Advance'
$wrapper = Join-Path $PSScriptRoot 'advance_goal_5hour_hidden.vbs'
if (-not (Test-Path -LiteralPath $wrapper)) { throw "Missing hidden launcher: $wrapper" }

$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wscript.exe" -Argument "//B //Nologo `"$wrapper`""
$trigger = New-ScheduledTaskTrigger -Once -At $StartAt -RepetitionInterval (New-TimeSpan -Hours 5)
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 35) `
    -MultipleInstances IgnoreNew `
    -Priority 5 `
    -Hidden `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -Compatibility Win8

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description 'Advance TokCom/TaiChi reproduction for 30 minutes every five hours without a visible console.' `
    -Force | Out-Null

$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
[pscustomobject]@{
    TaskName = $taskName
    Enabled = $task.Settings.Enabled
    Hidden = $task.Settings.Hidden
    IntervalHours = 5
    ActiveMinutes = 30
    ExecutionTimeLimit = $task.Settings.ExecutionTimeLimit
    NextRunTime = $info.NextRunTime
}
