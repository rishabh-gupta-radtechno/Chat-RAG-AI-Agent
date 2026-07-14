<#
.SYNOPSIS
    Registers (or re-registers) the Chat-RAG-AI-Agent backup as a Windows
    Scheduled Task.

.DESCRIPTION
    Schedule:
      * Runs every 2 days at 23:00 (11 PM).
      * StartWhenAvailable = true -> if the machine was off / asleep at 23:00,
        or the run was otherwise missed, it runs as soon as it next can.

    The task runs under the CURRENT user with the highest privileges and
    LogonType = Interactive. This matters because the backup drives Docker,
    and Docker Desktop on Windows only runs while that user is logged in.

    This script is idempotent: it removes any existing task of the same name
    and creates it fresh.

.NOTES
    Must be run from an ELEVATED (Administrator) PowerShell:
        powershell -ExecutionPolicy Bypass -File .\scripts\register-backup-task.ps1

    To remove the task later:
        Unregister-ScheduledTask -TaskName 'ChatRagAiAgent-Backup' -Confirm:$false
#>

[CmdletBinding()]
param(
    [string]$TaskName   = 'ChatRagAiAgent-Backup',
    [string]$TaskPath   = '\ChatRagAiAgent\',
    [int]$DaysInterval  = 2,
    [datetime]$RunAt    = '23:00',
    # Backup script this task will launch.
    [string]$ScriptPath = (Join-Path $PSScriptRoot 'backup.ps1')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --- Preconditions -------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw 'This script must be run from an elevated (Administrator) PowerShell.'
}

$ScriptPath = [System.IO.Path]::GetFullPath($ScriptPath)
if (-not (Test-Path $ScriptPath)) {
    throw "Backup script not found: $ScriptPath"
}

# --- Build the task definition ------------------------------------------
# Action: run the backup script via powershell.exe, no profile, bypass policy.
$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`"" `
    -WorkingDirectory (Split-Path $ScriptPath -Parent)

# Trigger: every N days at the given time.
$trigger = New-ScheduledTaskTrigger -Daily -DaysInterval $DaysInterval -At $RunAt

# Settings: the "run when next available" behaviour + sane run window.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

# Principal: current user, highest privileges, only when logged on
# (Interactive) so Docker Desktop is available.
$currentUser = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Highest

# --- Register (idempotent) ----------------------------------------------
$existing = Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$TaskPath$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -TaskPath $TaskPath `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Backs up Chat-RAG-AI-Agent (Postgres, Qdrant, uploads) every $DaysInterval days at $($RunAt.ToString('HH:mm')); runs when next available if missed." | Out-Null

Write-Host ''
Write-Host "Scheduled task registered:" -ForegroundColor Green
Write-Host "  Name        : $TaskPath$TaskName"
Write-Host "  Runs        : every $DaysInterval days at $($RunAt.ToString('HH:mm'))"
Write-Host "  If missed   : starts when next available"
Write-Host "  Runs as     : $currentUser (Interactive, Highest)"
Write-Host "  Launches    : $ScriptPath"
Write-Host ''
Write-Host "Next run time:"
Get-ScheduledTaskInfo -TaskName $TaskName -TaskPath $TaskPath |
    Select-Object -ExpandProperty NextRunTime

Write-Host ''
Write-Host "Test it now with:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName' -TaskPath '$TaskPath'"
