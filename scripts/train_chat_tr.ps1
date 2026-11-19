param(
    [int]$Steps = 12000,
    [string]$Out = "checkpoints/spiking_chat_tr_21m",
    [int]$Seed = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

if (!(Test-Path "data/chat_tr_train.pt")) {
    python research/make_chat_corpus.py
}

$name = Split-Path -Leaf $Out
python -m streamft.spiking_cli train `
    --data data/chat_tr_train.pt `
    --valid data/chat_tr_valid.pt `
    --tokenizer data/chat_tr_bpe8k.json `
    --out $Out `
    --name $name `
    --steps $Steps `
    --batch 16 `
    --seq 128 `
    --dim 1024 `
    --layers 2 `
    --lr 0.0005 `
    --lr-schedule cosine `
    --warmup 200 `
    --amp `
    --eval-every 500 `
    --eval-batches 8 `
    --eval-batch-size 4 `
    --sample-every 2000 `
    --sample-prompt "<|user|> merhaba<|end|><|assistant|>" `
    --sample-tokens 48 `
    --seed $Seed
