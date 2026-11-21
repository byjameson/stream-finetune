param(
    [string]$Model = "checkpoints/transformer_chat_en_scaled_50m",
    [string]$Tag = "scaled_50m"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

python research/benchmark_assistant.py --model $Model `
    --out "research/results/assistant_benchmark_$Tag.json"
python research/benchmark_raw.py --model $Model `
    --out "research/results/raw_knowledge_$Tag.json"
python research/behavior_eval.py --model $Model `
    --out "research/results/behavior_$Tag.json"
python research/eval_transformer_chat.py --model $Model --tag $Tag `
    --chat-valid data/chat_en_scaled_valid.pt `
    --stories-valid data/chat_en_scaled_stories_valid.pt `
    --mixed-valid data/chat_en_scaled_valid_mixed.pt `
    --out "research/results/eval_$Tag.json"
exit $LASTEXITCODE
