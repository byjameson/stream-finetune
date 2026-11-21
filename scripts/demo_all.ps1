param(
    [int]$Port = 8013
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

$model = "checkpoints/transformer_behavior_50m_long"
$adapter = "checkpoints/adapters_50m/v0001"
$docs = "data/demo_docs.db"

if (!(Test-Path "$model/model.pt")) {
    throw "missing product checkpoint: $model"
}
if (!(Test-Path "$adapter/adapter.pt")) {
    throw "missing adapter: $adapter"
}
if (!(Test-Path $docs)) {
    python -m streamft.doc_memory add examples/demo_note.txt --db $docs --title "Demo note"
}

powershell -ExecutionPolicy Bypass -File scripts/chat_demo.ps1 `
    -Port $Port -Model $model -Adapter $adapter -Docs $docs `
    -Queue "data/learning_queue.jsonl"
