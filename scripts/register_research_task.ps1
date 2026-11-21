param(
    [string]$TaskName = "AtomZeroResearchWorker",
    [string]$DailyAt = "09:00",
    [int]$Limit = 3,
    [switch]$Disabled
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_research.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$runner`" -Limit $Limit" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
    -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force `
    -Description "Atom Zero reviewed research worker: only approved Wikipedia topics are fetched." | Out-Null

if ($Disabled) {
    Disable-ScheduledTask -TaskName $TaskName | Out-Null
}

Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State, @{N = "NextRun"; E = {
        (Get-ScheduledTaskInfo -TaskName $_.TaskName).NextRunTime
    }}
