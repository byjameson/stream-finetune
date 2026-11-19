param(
    [string]$Text = "data/space_stories_train.txt",
    [string]$ValidText = "data/space_stories_valid.txt",
    [string]$Out = "checkpoints/story_lab_demo",
    [string]$Base = "checkpoints/spiking_21m",
    [int]$Steps = 120,
    [int]$Seed = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

if (!(Test-Path "data/old_mix2_train.pt")) {
    python research/make_mixed_corpus.py --ts-tokens 20000000 --wt-tokens 5000000 --out data/old_mix2_train.pt
}
if (!(Test-Path "data/old_mix_valid.pt")) {
    python research/make_old_mix_valid.py
}

python -m streamft.spiking_cli tokenize --text $ValidText --tokenizer data/ts_bpe8k.json --out data/story_lab_valid.pt

python -m streamft.spiking_cli ingest --model $Base --text $Text --new-valid data/story_lab_valid.pt --old-data data/old_mix2_train.pt --old-valid data/old_mix_valid.pt --out $Out --steps $Steps --batch 16 --seq 128 --lr 0.00005 --replay-fraction 0.7 --max-old-regression 0.03 --min-new-gain 0.01 --rollback-fraction 0.2 --amp --eval-every 30 --eval-batches 8 --eval-batch-size 4 --seed $Seed

if (!(Test-Path "$Out/versions")) {
    throw "ingest rejected; inspect $Out/registry.json"
}
$version = Get-ChildItem "$Out/versions" | Sort-Object Name | Select-Object -Last 1
$model = "$Out/versions/$($version.Name)"
Write-Host "accepted model: $model"

python -m streamft.spiking_cli eval --model $model --data data/story_lab_valid.pt --seq 128 --batches 4 --batch-size 4 --max-tokens 0
python -m streamft.spiking_cli generate --model $model --prompt "Once upon a time" --max-new 60 --temperature 0.5 --top-k 10 --repetition-penalty 1.1
