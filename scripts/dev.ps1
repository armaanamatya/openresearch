# Unified dev launcher — PowerShell sibling of scripts/dev.sh.
#
# Creates logs/<TS>[-Label]/ for the launch. Layout:
#
#   logs/<TS>/
#     meta.json          # launch metadata (started/ended, pids, exit codes)
#     manifest.json      # written on cleanup: every file's size + sha256
#     server/
#       backend.log      # uvicorn combined stdout+stderr
#       frontend.log     # Next.js dev combined stdout+stderr
#     prj_<id>/...       # per-pipeline-run workspaces (unchanged contract)
#
# REPROLAB_RUNS_ROOT is exported as a Windows path so the Windows Python
# inside .venv\Scripts\python.exe resolves it correctly. (The bash launcher
# under WSL hands Python a /mnt/c/... path that Windows can't parse, which
# silently regresses the runs_root contract — see docs/design.)
#
# Usage from any PowerShell prompt in the repo root:
#   .\scripts\dev.ps1
#   .\scripts\dev.ps1 -NoFrontend -Label backend-only
#   .\scripts\dev.ps1 -Keep 5
#
# If execution policy blocks the script, run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# or invoke as:
#   powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1
#
# See docs/design/unified-logging-launcher.md for the full design.

[CmdletBinding()]
param(
    [switch]$NoFrontend,
    [switch]$NoBackend,
    [string]$Label = "",
    [int]$Keep = 0
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$repoRoot = (Get-Location).Path

if ($NoBackend -and $NoFrontend) {
    Write-Output '[dev.ps1] -NoBackend and -NoFrontend together leave nothing to launch'
    exit 2
}

# ---------- log directory ----------
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
if ($Label -ne "") {
    $safe = ($Label -replace '[^A-Za-z0-9_-]', '-') -replace '-+', '-'
    $safe = $safe.Trim('-')
    if ($safe) { $ts = "$ts-$safe" }
}
$logDir    = Join-Path $repoRoot "logs\$ts"
$serverDir = Join-Path $logDir "server"
New-Item -ItemType Directory -Force -Path $serverDir | Out-Null

# ---------- env ----------
# Windows-native absolute path — the Windows Python understands this directly.
$env:REPROLAB_RUNS_ROOT = $logDir
$env:PYTHONUTF8         = "1"
$env:PYTHONIOENCODING   = "utf-8"
$env:REPROLAB_BACKEND_URL = "http://127.0.0.1:8000"

$gitSha = "unknown"
try {
    $maybeSha = (& git rev-parse --short HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and $maybeSha) { $gitSha = $maybeSha.Trim() }
} catch {}

$sandbox  = if ($env:REPROLAB_DEFAULT_SANDBOX) { $env:REPROLAB_DEFAULT_SANDBOX } else { "docker" }
$startedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

# ---------- venv python ----------
$pyBin = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pyBin)) {
    Write-Output '[dev.ps1] no venv at .venv\Scripts\python.exe - create one with:'
    Write-Output '  python -m venv .venv'
    Write-Output '  .\.venv\Scripts\pip install -r backend\requirements.txt'
    exit 1
}

# ---------- helpers ----------
function Write-Utf8NoBom {
    param([string]$Path, [string]$Content)
    $enc = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $enc)
}

function Free-Port {
    param([int]$Port)
    try {
        Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            ForEach-Object {
                try {
                    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
                } catch {}
            }
    } catch {}
}

function Stop-Tree {
    param([int]$ProcId)
    if ($ProcId -le 0) { return }
    # Gate on liveness — taskkill against a dead PID writes to stderr which
    # PS 5.1 wraps as a NativeCommandError and surfaces despite redirection.
    if (-not (Get-Process -Id $ProcId -ErrorAction SilentlyContinue)) { return }
    try {
        # Invoking taskkill via Start-Process keeps its native stderr out of
        # the PowerShell error stream entirely. /T tree-kills python/node
        # grandchildren spawned by the .cmd shim.
        Start-Process -FilePath taskkill.exe `
            -ArgumentList '/F','/T','/PID',$ProcId `
            -NoNewWindow -Wait -ErrorAction SilentlyContinue | Out-Null
    } catch {}
}

function Write-CmdShim {
    # Writes a .cmd file we can hand to Start-Process. Bypasses the
    # PowerShell-to-cmd quoting layer that made earlier attempts fail with
    # "The filename, directory name, or volume label syntax is incorrect."
    #
    # Also bakes REPROLAB_RUNS_ROOT and friends into the .cmd body. In
    # principle PowerShell's $env:X should propagate through Start-Process
    # to a cmd.exe child, but in practice (verified on Windows 11 / PS 5.1)
    # the inheritance is unreliable: a backend launched via this shim
    # observed self.runs_root resolved to .\runs instead of the launcher's
    # logs\<TS>\ even though the parent PowerShell had $env:REPROLAB_RUNS_ROOT
    # set. Setting the vars inside the shim removes that whole class of bug
    # and makes the .cmd a fully self-describing artifact.
    param(
        [string]$Path,
        [string]$WorkDir,
        [string]$Command,
        [string]$LogPath
    )
    $env_block = ""
    foreach ($name in @('REPROLAB_RUNS_ROOT','PYTHONUTF8','PYTHONIOENCODING','REPROLAB_BACKEND_URL')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($null -ne $value -and $value -ne '') {
            # cmd `set "VAR=value"` quoting handles spaces and special chars
            # in $value safely (the inner quotes scope the assignment).
            $env_block += "set `"$name=$value`"`r`n"
        }
    }
    $body = "@echo off`r`n" + $env_block + "cd /d `"$WorkDir`"`r`n$Command > `"$LogPath`" 2>&1`r`n"
    [System.IO.File]::WriteAllText($Path, $body, (New-Object System.Text.ASCIIEncoding))
}

function Write-Meta {
    param(
        [string]$EndedAt = $null,
        [string]$EndedReason = "exit",
        [Nullable[int]]$BackendPid = $null,
        [Nullable[int]]$FrontendPid = $null,
        [Nullable[int]]$BackendExit = $null,
        [Nullable[int]]$FrontendExit = $null
    )
    $obj = [ordered]@{
        started_at    = $startedAt
        ended_at      = $EndedAt
        ended_reason  = $EndedReason
        git_sha       = $gitSha
        sandbox_mode  = $sandbox
        runs_root     = $env:REPROLAB_RUNS_ROOT
        label         = $Label
        backend_pid   = $BackendPid
        frontend_pid  = $FrontendPid
        backend_exit  = $BackendExit
        frontend_exit = $FrontendExit
    }
    $json = $obj | ConvertTo-Json -Depth 4
    Write-Utf8NoBom -Path (Join-Path $logDir "meta.json") -Content $json
}

function Prune-OldLogs {
    param([int]$KeepN)
    if ($KeepN -lt 1) { return }
    $all = Get-ChildItem -Path (Join-Path $repoRoot "logs") -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name
    if ($all.Count -le $KeepN) { return }
    $victims = $all[0..($all.Count - $KeepN - 1)]
    foreach ($v in $victims) {
        # Never prune the directory we just created (paranoia).
        if ($v.FullName -eq $logDir) { continue }
        Write-Output ('[dev.ps1] prune {0}' -f $v.FullName)
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $v.FullName
    }
}

# ---------- pre-launch ----------
if (-not $NoBackend)  { Free-Port 8000 }
if (-not $NoFrontend) { Free-Port 3000 }
if ($Keep -gt 0) { Prune-OldLogs -KeepN $Keep }

Write-Output ('[dev.ps1] logs    -> {0}' -f $logDir)
Write-Output ('[dev.ps1] sandbox -> {0}' -f $sandbox)
Write-Output ('[dev.ps1] runs    -> {0}' -f $env:REPROLAB_RUNS_ROOT)

# ---------- launch ----------
# Use cmd.exe /c to merge stdout+stderr into a single .log file (matches the
# bash launcher's `> file 2>&1`). We hold the cmd.exe PID and tree-kill it
# during cleanup so the python/node grandchild dies with it.
$backendProc  = $null
$frontendProc = $null

if (-not $NoBackend) {
    $backendShim = Join-Path $serverDir 'backend.cmd'
    # --reload-dir backend: only watch backend/ source for hot-reload. The
    # default watches cwd, which includes runs/ and logs/ — when the pipeline
    # writes generated code (e.g. ppo_cartpole.py) WatchFiles triggers a
    # reload, killing uvicorn mid-run. Restrict to source so writes from
    # pipeline runs don't trip the reloader.
    Write-CmdShim -Path $backendShim -WorkDir $repoRoot `
        -Command ('"{0}" -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --reload --reload-dir backend' -f $pyBin) `
        -LogPath (Join-Path $serverDir 'backend.log')
    $backendProc = Start-Process -FilePath $backendShim `
        -WorkingDirectory $repoRoot `
        -NoNewWindow -PassThru
}

if (-not $NoFrontend) {
    $frontendShim = Join-Path $serverDir 'frontend.cmd'
    Write-CmdShim -Path $frontendShim -WorkDir (Join-Path $repoRoot 'frontend') `
        -Command 'npm.cmd run dev' `
        -LogPath (Join-Path $serverDir 'frontend.log')
    $frontendProc = Start-Process -FilePath $frontendShim `
        -WorkingDirectory (Join-Path $repoRoot 'frontend') `
        -NoNewWindow -PassThru
}

$backendPid  = if ($backendProc)  { $backendProc.Id  } else { $null }
$frontendPid = if ($frontendProc) { $frontendProc.Id } else { $null }
Write-Meta -BackendPid $backendPid -FrontendPid $frontendPid

if ($backendProc)  { Write-Output ('[dev.ps1] backend  pid={0}  log={1}\backend.log'  -f $backendProc.Id,  $serverDir) }
if ($frontendProc) { Write-Output ('[dev.ps1] frontend pid={0}  log={1}\frontend.log' -f $frontendProc.Id, $serverDir) }
if (-not $NoFrontend) { Write-Output '[dev.ps1] open http://localhost:3000/lab' }

# ---------- main wait loop with cleanup ----------
$endedReason = "exit"
$cleaned = $false

function Invoke-Cleanup {
    param([string]$Reason)
    if ($script:cleaned) { return }
    $script:cleaned = $true

    $endedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    $backendExit  = $null
    $frontendExit = $null

    if ($backendProc) {
        Stop-Tree -ProcId $backendProc.Id
        try { $backendProc.WaitForExit(5000) | Out-Null } catch {}
        try { $backendExit = $backendProc.ExitCode } catch {}
    }
    if ($frontendProc) {
        Stop-Tree -ProcId $frontendProc.Id
        try { $frontendProc.WaitForExit(5000) | Out-Null } catch {}
        try { $frontendExit = $frontendProc.ExitCode } catch {}
    }

    Write-Meta -EndedAt $endedAt -EndedReason $Reason `
        -BackendPid $backendPid -FrontendPid $frontendPid `
        -BackendExit $backendExit -FrontendExit $frontendExit

    # Best-effort manifest. Never block exit on this.
    try {
        & $pyBin (Join-Path $repoRoot "scripts\_write_manifest.py") $logDir | Out-Null
    } catch {}

    if (-not $NoBackend)  { Free-Port 8000 }
    if (-not $NoFrontend) { Free-Port 3000 }
}

try {
    while ($true) {
        $backendAlive  = $false
        $frontendAlive = $false
        if ($backendProc)  { $backendAlive  = -not $backendProc.HasExited }
        if ($frontendProc) { $frontendAlive = -not $frontendProc.HasExited }
        # If we launched something and any of it died, stop the world.
        if ($backendProc  -and -not $backendAlive)  { break }
        if ($frontendProc -and -not $frontendAlive) { break }
        if (-not $backendProc -and -not $frontendProc) { break }
        Start-Sleep -Seconds 1
    }
} catch [System.Management.Automation.PipelineStoppedException] {
    # Raised when the user hits Ctrl-C.
    $endedReason = "int"
} finally {
    Invoke-Cleanup -Reason $endedReason
}
