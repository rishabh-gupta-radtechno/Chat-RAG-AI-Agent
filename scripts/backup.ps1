<#
.SYNOPSIS
    Backs up the Chat-RAG-AI-Agent stateful Docker volumes.

.DESCRIPTION
    Creates a timestamped backup of:
      1. PostgreSQL   (chat-rag-postgres  -> pg_dump custom format)
      2. Qdrant       (chat-rag-qdrant    -> tar.gz of /qdrant/storage)
      3. Uploads      (chat-rag-api       -> tar.gz of /app/static/uploads)

    Each artifact is produced INSIDE the container and pulled out with
    `docker cp`, so no PowerShell binary-stream redirection is involved and
    the Compose-prefixed volume names are never needed.

    After a successful run it:
      4. Removes local backups older than $RetentionDays.
      5. Writes a log for every step.

.NOTES
    Run from an elevated PowerShell if your Docker install requires it:
        powershell -ExecutionPolicy Bypass -File .\scripts\backup.ps1
#>

[CmdletBinding()]
param(
    # Local root for backups (kept inside the repo tree, gitignored).
    # Resolved below if not supplied (see $PSScriptRoot note).
    [string]$BackupRoot,

    # Retention window in days (task requirement #4).
    [int]$RetentionDays = 30,

    # Container names (from docker-compose.yml).
    [string]$PgContainer      = 'chat-rag-postgres',
    [string]$QdrantContainer  = 'chat-rag-qdrant',
    [string]$UploadsContainer = 'chat-rag-api',

    # PostgreSQL connection (from docker-compose.yml environment).
    [string]$PgUser = 'postgres',
    [string]$PgDb   = 'chat_rag_db'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Resolve the script directory robustly. On some Windows PowerShell 5.1 hosts
# $PSScriptRoot is empty inside a param() default, which broke Join-Path; use a
# fallback and compute the default here instead.
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $BackupRoot) { $BackupRoot = Join-Path $ScriptDir '..\backups' }

# --------------------------------------------------------------------------
# Paths & logging
# --------------------------------------------------------------------------
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$BackupRoot  = [System.IO.Path]::GetFullPath($BackupRoot)
$runDir      = Join-Path $BackupRoot $timestamp
$logDir      = Join-Path $BackupRoot 'logs'
$logFile     = Join-Path $logDir "backup_$timestamp.log"

New-Item -ItemType Directory -Force -Path $runDir, $logDir | Out-Null

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('INFO', 'WARN', 'ERROR')]
        [string]$Level = 'INFO'
    )
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        default { Write-Host $line }
    }
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

# Runs a docker command and throws (with the captured output logged) on failure.
function Invoke-Docker {
    param([Parameter(Mandatory)][string[]]$Arguments)
    Write-Log ("docker " + ($Arguments -join ' '))
    $output = & docker @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        if ($output) { $output | ForEach-Object { Write-Log $_ 'ERROR' } }
        throw "docker $($Arguments -join ' ') exited with code $LASTEXITCODE"
    }
    return $output
}

# Confirms a container exists and is running before we try to exec into it.
function Assert-ContainerRunning {
    param([string]$Name)
    $state = & docker inspect -f '{{.State.Running}}' $Name 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Container '$Name' not found." }
    if ($state.Trim() -ne 'true') { throw "Container '$Name' is not running." }
}

$failures = @()

Write-Log "===== Chat-RAG-AI-Agent backup started ($timestamp) ====="
Write-Log "Backup dir : $runDir"
Write-Log "Retention  : $RetentionDays days"

# Fail fast if docker itself is unavailable.
try {
    & docker version --format '{{.Server.Version}}' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'docker daemon not reachable.' }
} catch {
    Write-Log "Docker is not available: $($_.Exception.Message)" 'ERROR'
    Write-Log "===== Backup ABORTED =====" 'ERROR'
    exit 1
}

# --------------------------------------------------------------------------
# 1. PostgreSQL  (pg_dump custom format -> .dump)
# --------------------------------------------------------------------------
try {
    Write-Log '--- [1/3] PostgreSQL backup ---'
    Assert-ContainerRunning $PgContainer
    $inside = '/tmp/postgres.dump'
    $outFile = Join-Path $runDir 'postgres.dump'
    Invoke-Docker @('exec', $PgContainer, 'pg_dump', '-U', $PgUser, '-F', 'c', '-f', $inside, $PgDb) | Out-Null
    Invoke-Docker @('cp', "${PgContainer}:$inside", $outFile) | Out-Null
    & docker exec $PgContainer rm -f $inside 2>&1 | Out-Null
    $size = [math]::Round((Get-Item $outFile).Length / 1MB, 2)
    Write-Log "PostgreSQL backup OK -> postgres.dump ($size MB)"
} catch {
    Write-Log "PostgreSQL backup FAILED: $($_.Exception.Message)" 'ERROR'
    $failures += 'PostgreSQL'
}

# --------------------------------------------------------------------------
# 2. Qdrant  (tar.gz of /qdrant/storage)
# --------------------------------------------------------------------------
try {
    Write-Log '--- [2/3] Qdrant backup ---'
    Assert-ContainerRunning $QdrantContainer
    $inside = '/tmp/qdrant.tar.gz'
    $outFile = Join-Path $runDir 'qdrant.tar.gz'
    Invoke-Docker @('exec', $QdrantContainer, 'tar', 'czf', $inside, '-C', '/qdrant', 'storage') | Out-Null
    Invoke-Docker @('cp', "${QdrantContainer}:$inside", $outFile) | Out-Null
    & docker exec $QdrantContainer rm -f $inside 2>&1 | Out-Null
    $size = [math]::Round((Get-Item $outFile).Length / 1MB, 2)
    Write-Log "Qdrant backup OK -> qdrant.tar.gz ($size MB)"
} catch {
    Write-Log "Qdrant backup FAILED: $($_.Exception.Message)" 'ERROR'
    $failures += 'Qdrant'
}

# --------------------------------------------------------------------------
# 3. Uploads  (tar.gz of /app/static/uploads)
# --------------------------------------------------------------------------
try {
    Write-Log '--- [3/3] Uploads backup ---'
    Assert-ContainerRunning $UploadsContainer
    $inside = '/tmp/uploads.tar.gz'
    $outFile = Join-Path $runDir 'uploads.tar.gz'
    Invoke-Docker @('exec', $UploadsContainer, 'tar', 'czf', $inside, '-C', '/app/static', 'uploads') | Out-Null
    Invoke-Docker @('cp', "${UploadsContainer}:$inside", $outFile) | Out-Null
    & docker exec $UploadsContainer rm -f $inside 2>&1 | Out-Null
    $size = [math]::Round((Get-Item $outFile).Length / 1MB, 2)
    Write-Log "Uploads backup OK -> uploads.tar.gz ($size MB)"
} catch {
    Write-Log "Uploads backup FAILED: $($_.Exception.Message)" 'ERROR'
    $failures += 'Uploads'
}

# Abort retention if nothing was produced.
if ($failures.Count -eq 3) {
    Write-Log 'All three backups failed; skipping retention.' 'ERROR'
    Write-Log '===== Backup FAILED =====' 'ERROR'
    exit 1
}

# --------------------------------------------------------------------------
# 4. Remove local backups older than $RetentionDays
# --------------------------------------------------------------------------
function Remove-OldBackups {
    param([string]$Root, [int]$Days)
    if (-not (Test-Path $Root)) { return }
    $cutoff = (Get-Date).AddDays(-$Days)
    # Timestamped run folders are named yyyyMMdd_HHmmss.
    Get-ChildItem -Path $Root -Directory |
        Where-Object { $_.Name -match '^\d{8}_\d{6}$' -and $_.LastWriteTime -lt $cutoff } |
        ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force
            Write-Log "Removed old backup: $($_.FullName)"
        }
    # Prune old log files too.
    $logs = Join-Path $Root 'logs'
    if (Test-Path $logs) {
        Get-ChildItem -Path $logs -File -Filter 'backup_*.log' |
            Where-Object { $_.LastWriteTime -lt $cutoff } |
            ForEach-Object { Remove-Item $_.FullName -Force; Write-Log "Removed old log: $($_.FullName)" }
    }
}

try {
    Write-Log '--- Applying retention policy ---'
    Remove-OldBackups -Root $BackupRoot -Days $RetentionDays
} catch {
    Write-Log "Retention cleanup encountered an error: $($_.Exception.Message)" 'WARN'
}

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
if ($failures.Count -eq 0) {
    Write-Log '===== Backup COMPLETED successfully ====='
    exit 0
} else {
    Write-Log ("===== Backup completed WITH ERRORS: {0} =====" -f ($failures -join ', ')) 'ERROR'
    exit 1
}
