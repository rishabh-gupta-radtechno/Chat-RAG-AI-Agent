<#
.SYNOPSIS
    Restores a Chat-RAG-AI-Agent backup produced by scripts/backup.ps1.

.DESCRIPTION
    Restores from a single timestamped backup folder:
      1. PostgreSQL   postgres.dump   -> pg_restore into chat-rag-postgres
      2. Qdrant       qdrant.tar.gz   -> replaces the qdrant_data volume
      3. Uploads      uploads.tar.gz  -> replaces the uploads_data volume

    WARNING: This is DESTRUCTIVE. It overwrites the current database, vector
    store, and uploaded files with the contents of the chosen backup.

    Because a live service must not have its volume overwritten underneath it,
    Qdrant and the API (uploads) containers are STOPPED for the restore and
    STARTED again afterwards. PostgreSQL is restored in place with pg_restore
    --clean --if-exists.

    The tar extractions run in a throwaway helper container built from
    postgres:16-alpine (already present on the host, so no extra image pull),
    which mounts the target Docker volume directly.

.EXAMPLE
    # Restore the most recent backup (prompts for confirmation)
    powershell -ExecutionPolicy Bypass -File .\scripts\restore.ps1

.EXAMPLE
    # Restore a specific backup, no prompt
    .\scripts\restore.ps1 -Timestamp 20260714_230000 -Force
#>

[CmdletBinding()]
param(
    # Full path to a backup run folder. Overrides -Timestamp / -BackupRoot.
    [string]$BackupPath,

    # Name of the run folder (yyyyMMdd_HHmmss) under -BackupRoot.
    [string]$Timestamp,

    # Root that holds the timestamped run folders.
    [string]$BackupRoot = (Join-Path $PSScriptRoot '..\backups'),

    # Skip the interactive confirmation.
    [switch]$Force,

    # Helper image for tar extraction (already pulled by the stack).
    [string]$HelperImage = 'postgres:16-alpine',

    # Container names (from docker-compose.yml).
    [string]$PgContainer      = 'chat-rag-postgres',
    [string]$QdrantContainer  = 'chat-rag-qdrant',
    [string]$UploadsContainer = 'chat-rag-api',

    # Volume mount destinations inside those containers.
    [string]$QdrantDest  = '/qdrant/storage',
    [string]$UploadsDest = '/app/static/uploads',

    # PostgreSQL connection (from docker-compose.yml environment).
    [string]$PgUser = 'postgres',
    [string]$PgDb   = 'chat_rag_db'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --------------------------------------------------------------------------
# Resolve which backup folder to restore
# --------------------------------------------------------------------------
$BackupRoot = [System.IO.Path]::GetFullPath($BackupRoot)

if ($BackupPath) {
    $runDir = [System.IO.Path]::GetFullPath($BackupPath)
} elseif ($Timestamp) {
    $runDir = Join-Path $BackupRoot $Timestamp
} else {
    # Default: newest yyyyMMdd_HHmmss folder under the root.
    $latest = Get-ChildItem -Path $BackupRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{8}_\d{6}$' } |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $latest) { throw "No backup folders found under $BackupRoot" }
    $runDir = $latest.FullName
}

if (-not (Test-Path $runDir)) { throw "Backup folder not found: $runDir" }

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
$stamp   = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir  = Join-Path $BackupRoot 'logs'
$logFile = Join-Path $logDir "restore_$stamp.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('INFO', 'WARN', 'ERROR')][string]$Level = 'INFO'
    )
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        default { Write-Host $line }
    }
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

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

# Returns the Docker volume name backing $Destination in container $Container.
function Get-VolumeName {
    param([string]$Container, [string]$Destination)
    $tmpl = '{{range .Mounts}}{{if eq .Destination "' + $Destination + '"}}{{.Name}}{{end}}{{end}}'
    $name = (& docker inspect -f $tmpl $Container 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "Container '$Container' not found." }
    $name = ($name | Out-String).Trim()
    if (-not $name) { throw "No named volume mounted at $Destination in $Container." }
    return $name
}

# Docker Desktop accepts forward-slash Windows paths for bind mounts.
function ConvertTo-DockerPath {
    param([string]$Path)
    return ($Path -replace '\\', '/')
}

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
try {
    & docker version --format '{{.Server.Version}}' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'docker daemon not reachable.' }
} catch {
    Write-Log "Docker is not available: $($_.Exception.Message)" 'ERROR'
    exit 1
}

$pgDump      = Join-Path $runDir 'postgres.dump'
$qdrantTar   = Join-Path $runDir 'qdrant.tar.gz'
$uploadsTar  = Join-Path $runDir 'uploads.tar.gz'

Write-Log "===== Chat-RAG-AI-Agent RESTORE ====="
Write-Log "Source folder : $runDir"
Write-Log ("  postgres.dump  : {0}" -f $(if (Test-Path $pgDump)     { 'present' } else { 'MISSING (skip)' }))
Write-Log ("  qdrant.tar.gz  : {0}" -f $(if (Test-Path $qdrantTar)  { 'present' } else { 'MISSING (skip)' }))
Write-Log ("  uploads.tar.gz : {0}" -f $(if (Test-Path $uploadsTar) { 'present' } else { 'MISSING (skip)' }))

if (-not (Test-Path $pgDump) -and -not (Test-Path $qdrantTar) -and -not (Test-Path $uploadsTar)) {
    Write-Log 'No restorable artifacts in the chosen folder.' 'ERROR'
    exit 1
}

# --------------------------------------------------------------------------
# Confirmation (destructive)
# --------------------------------------------------------------------------
if (-not $Force) {
    Write-Host ''
    Write-Host 'WARNING: this OVERWRITES the current database, vector store and uploads.' -ForegroundColor Yellow
    $answer = Read-Host "Type 'yes' to proceed"
    if ($answer -ne 'yes') {
        Write-Log 'Restore cancelled by user.' 'WARN'
        exit 1
    }
}

$failures = @()

# --------------------------------------------------------------------------
# 1. PostgreSQL  (in-place, container stays running)
# --------------------------------------------------------------------------
if (Test-Path $pgDump) {
    try {
        Write-Log '--- [1/3] Restoring PostgreSQL ---'
        $inside = '/tmp/restore_postgres.dump'
        Invoke-Docker @('cp', $pgDump, "${PgContainer}:$inside") | Out-Null
        # --clean --if-exists drops existing objects first; safe to re-run.
        Invoke-Docker @('exec', $PgContainer, 'pg_restore', '-U', $PgUser, '-d', $PgDb,
                        '--clean', '--if-exists', '--no-owner', $inside) | Out-Null
        & docker exec $PgContainer rm -f $inside 2>&1 | Out-Null
        Write-Log 'PostgreSQL restore OK'
    } catch {
        Write-Log "PostgreSQL restore FAILED: $($_.Exception.Message)" 'ERROR'
        $failures += 'PostgreSQL'
    }
}

# --------------------------------------------------------------------------
# Helper: stop a container, replace its volume from a tar, restart it
# --------------------------------------------------------------------------
function Restore-VolumeFromTar {
    param(
        [string]$Container,
        [string]$Destination,
        [string]$TarFile
    )
    $volume  = Get-VolumeName -Container $Container -Destination $Destination
    $hostDir = ConvertTo-DockerPath ([System.IO.Path]::GetDirectoryName($TarFile))
    $tarName = [System.IO.Path]::GetFileName($TarFile)
    Write-Log "Target volume : $volume (from $Container)"

    Invoke-Docker @('stop', $Container) | Out-Null
    try {
        # Wipe the volume (including dotfiles) then extract, stripping the
        # top-level folder the backup tar was rooted at.
        $shell = "rm -rf /restore/* 2>/dev/null; rm -rf /restore/.[!.]* /restore/..?* 2>/dev/null; " +
                 "tar xzf /backup/$tarName -C /restore --strip-components=1"
        Invoke-Docker @('run', '--rm',
                        '-v', "${volume}:/restore",
                        '-v', "${hostDir}:/backup:ro",
                        $HelperImage, 'sh', '-c', $shell) | Out-Null
    } finally {
        Invoke-Docker @('start', $Container) | Out-Null
    }
}

# --------------------------------------------------------------------------
# 2. Qdrant
# --------------------------------------------------------------------------
if (Test-Path $qdrantTar) {
    try {
        Write-Log '--- [2/3] Restoring Qdrant ---'
        Restore-VolumeFromTar -Container $QdrantContainer -Destination $QdrantDest -TarFile $qdrantTar
        Write-Log 'Qdrant restore OK'
    } catch {
        Write-Log "Qdrant restore FAILED: $($_.Exception.Message)" 'ERROR'
        $failures += 'Qdrant'
    }
}

# --------------------------------------------------------------------------
# 3. Uploads
# --------------------------------------------------------------------------
if (Test-Path $uploadsTar) {
    try {
        Write-Log '--- [3/3] Restoring uploads ---'
        Restore-VolumeFromTar -Container $UploadsContainer -Destination $UploadsDest -TarFile $uploadsTar
        Write-Log 'Uploads restore OK'
    } catch {
        Write-Log "Uploads restore FAILED: $($_.Exception.Message)" 'ERROR'
        $failures += 'Uploads'
    }
}

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
if ($failures.Count -eq 0) {
    Write-Log '===== Restore COMPLETED successfully ====='
    exit 0
} else {
    Write-Log ("===== Restore completed WITH ERRORS: {0} =====" -f ($failures -join ', ')) 'ERROR'
    exit 1
}
