param(
    [int]$Port = 8013,
    [switch]$SafetyFilter,
    [switch]$NoTeacher,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

$model = "checkpoints/transformer_behavior_50m_scaled"
if (!(Test-Path "$model/model.pt")) {
    Write-Host "scaled checkpoint not found; using the 50M long baseline"
    $model = "checkpoints/transformer_behavior_50m_long"
}
$adapter = "checkpoints/adapters_50m/v0001"
$docs = "data/demo_docs.db"

if (!(Test-Path "$model/model.pt")) {
    throw "missing product checkpoint: $model"
}
if (!(Test-Path "$adapter/adapter.pt")) {
    Write-Host "adapter not found at $adapter; running the base model only"
    $adapter = ""
}
# Always refresh the demo note; other indexed sources are untouched.
python -m streamft.doc_memory add examples/demo_note.txt --db $docs --title "Demo note"

Write-Host "Starting Atom Zero demo..."
& (Join-Path $PSScriptRoot "chat_demo.ps1") `
    -Port $Port -Model $model -Adapter $adapter -Docs $docs `
    -Queue "data/learning_queue.jsonl" -SafetyFilter:$SafetyFilter `
    -NoTeacher:$NoTeacher -NoBrowser:$NoBrowser
