param(
    [int]$Limit = 1,
    [string]$Queue = "data/research_queue.json",
    [string]$Db = "data/docs.db",
    [string]$Log = "data/research_worker.jsonl",
    [string]$Lock = "data/research_worker.lock",
    [string]$TeacherUrl = "http://127.0.0.1:11434/v1",
    [string]$TeacherModel = "llama3.2:3b",
    [string]$TeacherLabel = "ollama"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

$teacherArgs = @()
if ($TeacherUrl) { $teacherArgs += @("--teacher-url", $TeacherUrl) }
if ($TeacherModel) { $teacherArgs += @("--teacher-model", $TeacherModel) }
if ($TeacherUrl -or $TeacherModel) {
    $teacherArgs += @("--teacher-label", $TeacherLabel)
}

python research/research_worker.py `
    --queue $Queue --db $Db --limit $Limit `
    --log $Log --lock $Lock --once @teacherArgs
exit $LASTEXITCODE
