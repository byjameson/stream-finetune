param(
    [int]$Steps = 60000,
    [string]$Data = "data/chat_en_scaled_train.pt",
    [string]$Valid = "data/chat_en_scaled_valid.pt",
    [string]$Out = "checkpoints/transformer_chat_en_scaled_50m",
    [string]$Result = "research/results/transformer_chat_en_scaled_50m.json",
    [double]$Lr = 0.0001
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

python research/train_transformer_chat.py `
    --init checkpoints/transformer_behavior_50m_long `
    --data $Data --valid $Valid `
    --out $Out --tag transformer_scaled_50m --result $Result `
    --steps $Steps --batch 16 --seq 128 `
    --dim 768 --layers 6 --heads 12 --ff 3072 `
    --lr $Lr --warmup 500 `
    --eval-every 2000 --save-every 5000 --sample-every 5000
exit $LASTEXITCODE
