param(
    [int]$Steps = 60000,
    [string]$Data = "data/chat_en_scaled_replay_train.pt",
    [string]$Valid = "data/chat_en_scaled_valid.pt",
    [string]$Out = "checkpoints/transformer_chat_en_scaled_50m",
    [string]$Result = "research/results/transformer_chat_en_scaled_50m.json",
    [string]$Init = "checkpoints/transformer_behavior_50m_long",
    [string]$Tag = "transformer_scaled_50m",
    [double]$Lr = 0.0001,
    [int]$Warmup = 500,
    [int]$EvalEvery = 2000,
    [int]$SaveEvery = 5000
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

python research/train_transformer_chat.py `
    --init $Init `
    --data $Data --valid $Valid `
    --out $Out --tag $Tag --result $Result `
    --steps $Steps --batch 16 --seq 128 `
    --dim 768 --layers 6 --heads 12 --ff 3072 `
    --lr $Lr --warmup $Warmup `
    --eval-every $EvalEvery --save-every $SaveEvery --sample-every 5000
exit $LASTEXITCODE
