param(
    [int]$Steps = 12000,
    [string]$Out = "checkpoints/spiking_chat_en_clean_50m_sft",
    [int]$Seed = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

if (!(Test-Path "data/chat_en_clean_train.pt")) {
    python research/make_chat_corpus_clean.py
}

$name = Split-Path -Leaf $Out
python -m streamft.spiking_cli train `
    --data data/chat_en_clean_train.pt `
    --valid data/chat_en_clean_valid.pt `
    --tokenizer data/chat_en_bpe8k.json `
    --out $Out `
    --name $name `
    --steps $Steps `
    --batch 16 `
    --seq 128 `
    --dim 2040 `
    --layers 2 `
    --lr 0.0005 `
    --lr-schedule cosine `
    --warmup 200 `
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
