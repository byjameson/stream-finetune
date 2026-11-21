param(
    [string]$Store = "data/feedback_queue.json",
    [string]$OutQueue = "data/feedback_training.jsonl",
    [string]$Base = "checkpoints/transformer_behavior_50m_scaled",
    [string]$Out = "checkpoints/adapters_feedback",
    [string]$History = "data/adapter_history.jsonl",
    [string]$Active = "data/active_adapter.json",
    [double]$Holdout = 0.0,
    [string]$HoldoutOut = "data/feedback_holdout.jsonl",
    [int]$Steps = 100
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

python research/learning_loop.py `
    --store $Store --out-queue $OutQueue --history $History `
    run --base $Base --out $Out --steps $Steps --active-file $Active `
    --holdout $Holdout --holdout-out $HoldoutOut
