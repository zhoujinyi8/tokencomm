param(
    [datetime]$StartAt = (Get-Date).AddHours(4)
)

$ErrorActionPreference = 'Stop'
$taskName = 'TokenCom-Goal-4Hour-Advance'
$wrapper = Join-Path $PSScriptRoot 'advance_goal_4hour_hidden.vbs'
if (-not (Test-Path -LiteralPath $wrapper)) { throw "Missing hidden launcher: $wrapper" }

$action = New-ScheduledTaskAction -Execute "$env:WINDIR\System32\wscript.exe" -Argument "//B //Nologo `"$wrapper`""
$trigger = New-ScheduledTaskTrigger -Once -At $StartAt -RepetitionInterval (New-TimeSpan -Hours 4)
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3 -Minutes 45) `
    -MultipleInstances IgnoreNew `
    -Hidden `
    -StartWhenAvailable `
    -Compatibility Win8

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description 'Advance TokCom/TaiChi reproduction every four hours without a visible console.' `
    -Force | Out-Null

$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
[pscustomobject]@{
    TaskName = $taskName
    Enabled = $task.Settings.Enabled
    Hidden = $task.Settings.Hidden
    IntervalHours = 4
    ExecutionTimeLimit = $task.Settings.ExecutionTimeLimit
    NextRunTime = $info.NextRunTime
}
