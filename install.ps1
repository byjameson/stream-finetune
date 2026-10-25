# streamft installer (Windows)
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# It checks Python, installs/updates streamft, looks for an NVIDIA GPU and runs
# `streamft doctor`. Nothing is downloaded beyond the pip package and its dependencies.

param(
    [switch]$NoInstall,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!!  $msg" -ForegroundColor Yellow }

Say "Checking Python"
$python = $null
foreach ($cand in @("python", "py")) {
    try {
        $v = & $cand -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($v) { $python = $cand; break }
    } catch { }
}
if (-not $python) {
    Warn "Python 3.10+ was not found."
    Write-Host "Install it from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH'), then rerun this script."
    exit 1
}
$ver = & $python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
$major, $minor = $ver.Split(".")
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
    Warn "Python $ver found, 3.10 or newer is required."
    exit 1
}
Say "Python $ver ($python)"

if (-not $NoInstall) {
    Say "Installing streamft (pip)"
    if ($DryRun) {
        Write-Host "    pip install --upgrade streamft   [dry run]"
    } else {
        & $python -m pip install --upgrade streamft
        if ($LASTEXITCODE -ne 0) { Warn "pip install failed"; exit 1 }
    }
} else {
    Say "Skipping install (-NoInstall)"
}

Say "Looking for an NVIDIA GPU"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) 2>$null
    if ($gpu) { Write-Host "    $gpu" }
    $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader) 2>$null
    if ($cap -and [double]$cap -lt 8.0) {
        Warn "compute capability ${cap}: no bf16 units -> use --compute-dtype fp16"
    }
} else {
    Warn "nvidia-smi not found. GPU training will not work; CPU training still does:"
    Write-Host "    streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128"
}

if (-not $DryRun) {
    Say "Environment report"
    try { & $python -m streamft doctor } catch { Warn "doctor failed: $_" }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  streamft fit --model Qwen/Qwen2.5-7B-Instruct --gpu 6 --seq 512"
Write-Host "  streamft prepare --model Qwen/Qwen2.5-0.5B --save-quant checkpoints/q05"
Write-Host "  streamft train --model-dir checkpoints/q05 --sft --steps 20 --seq 512"
Write-Host "  (no GPU) streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128"
Write-Host ""
Write-Host "Docs: https://github.com/byjameson/stream-finetune"
