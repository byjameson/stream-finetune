# Train the 20.97M-parameter spiking LM on TinyStories.
# Resumable: stop it at any checkpoint and rerun with --resume.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python -m streamft spiking train `
  --data "$root/data/ts_train.pt" `
  --valid "$root/data/ts_valid.pt" `
  --tokenizer "$root/data/ts_bpe8k.json" `
  --out "$root/checkpoints/spiking_21m" `
  --name spiking_21m `
  --steps 10000 --batch 256 --seq 64 `
  --dim 1024 --layers 2 --adapt 0.1 `
  --lr 0.0005 --min-lr 0.00002 --lr-schedule cosine `
  --warmup 500 --total-steps 10000 `
  --amp --eval-every 500 --sample-every 1000 --save-every 500
