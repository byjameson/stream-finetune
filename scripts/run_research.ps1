param(
    [int]$Limit = 1,
    [string]$Queue = "data/research_queue.json",
    [string]$Db = "data/docs.db",
    [string]$TeacherUrl = "",
    [string]$TeacherModel = "",
    [string]$TeacherLabel = "openai-compatible"
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

python research/research_queue.py --queue $Queue run --db $Db --limit $Limit @teacherArgs
