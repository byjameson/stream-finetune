param(
    [int]$Port = 8013,
    [string]$Model = "checkpoints/transformer_behavior_50m_long",
    [string]$Adapter = "",
    [string]$Docs = "",
    [string]$Queue = "data/learning_queue.jsonl"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$extra = ""
if ($Adapter) { $extra += " --adapter $Adapter" }
if ($Docs) { $extra += " --docs $Docs" }
if ($Queue) { $extra += " --queue $Queue" }

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$root`"; `$env:PYTHONPATH=`"$root`"; python -m streamft.chat_server --model $Model --host 127.0.0.1 --port $Port$extra"
)
Start-Sleep -Seconds 8
Start-Process "http://127.0.0.1:$Port/"
Write-Host "Tiny chat server: http://127.0.0.1:$Port/"
