#!/usr/bin/env bash
# streamft installer (Linux / macOS)
#
#   bash install.sh [--no-install]
#
# Checks Python, installs/updates streamft, looks for an NVIDIA GPU and runs doctor.
set -euo pipefail

NO_INSTALL=0
for arg in "$@"; do
  [ "$arg" = "--no-install" ] && NO_INSTALL=1
done

say() { printf '\033[36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33m!!  %s\033[0m\n' "$1"; }

say "Checking Python"
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  warn "Python 3.10+ not found. Install it (apt/dnf/brew) and rerun."
  exit 1
fi
VER=$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  warn "Python $VER found, 3.10+ required."
  exit 1
fi
say "Python $VER ($PY)"

if [ "$NO_INSTALL" -eq 0 ]; then
  say "Installing streamft (pip)"
  $PY -m pip install --upgrade streamft
else
  say "Skipping install"
fi

say "Looking for an NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true
else
  warn "nvidia-smi not found. GPU training will not work; CPU training still does:"
  echo "    streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128"
fi

say "Environment report"
$PY -m streamft doctor || true

cat <<'EOF'

Next steps:
  streamft fit --model Qwen/Qwen2.5-7B-Instruct --gpu 6 --seq 512
  streamft prepare --model Qwen/Qwen2.5-0.5B --save-quant checkpoints/q05
  streamft train --model-dir checkpoints/q05 --sft --steps 20 --seq 512
  (no GPU) streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128

Docs: https://github.com/byjameson/stream-finetune
EOF
