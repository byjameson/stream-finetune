param(
    [int]$Port = 8013,
    [string]$Model = "checkpoints/transformer_chat_en_clean_21m"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$root`"; `$env:PYTHONPATH=`"$root`"; python -m streamft.chat_server --model $Model --host 127.0.0.1 --port $Port"
)
Start-Sleep -Seconds 8
Start-Process "http://127.0.0.1:$Port/"
Write-Host "Tiny chat server: http://127.0.0.1:$Port/"
