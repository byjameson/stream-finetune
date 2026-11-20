param(
    [string]$Model = "checkpoints/spiking_chat_en_clean_21m_sft",
    [int]$Steps = 12000,
    [int]$Total = 24000,
    [int]$Seed = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

$name = Split-Path -Leaf $Model
python -m streamft.spiking_cli train `
    --data data/chat_en_clean_train.pt `
    --valid data/chat_en_clean_valid.pt `
    --tokenizer data/chat_en_bpe8k.json `
    --resume $Model `
    --out $Model `
    --name $name `
    --steps $Steps `
    --total-steps $Total `
    --batch 16 `
    --seq 128 `
    --dim 1024 `
    --layers 2 `
    --lr 0.0001 `
    --lr-schedule constant `
    --amp `
    --start-token 0 `
    --mask-user `
    --eval-every 500 `
    --eval-batches 8 `
    --eval-batch-size 4 `
    --sample-every 2000 `
    --sample-prompt "<|user|> hello<|end|><|assistant|>" `
    --sample-tokens 48 `
    --save-every 2000 `
    --seed $Seed
