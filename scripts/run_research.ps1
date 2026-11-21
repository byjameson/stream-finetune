param(
    [int]$Limit = 1,
    [string]$Queue = "data/research_queue.json",
    [string]$Db = "data/docs.db"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

python research/research_queue.py --queue $Queue run --db $Db --limit $Limit
