param(
    [int]$Port = 8013,
    [string]$Model = "checkpoints/transformer_behavior_50m_long",
    [string]$Adapter = "",
    [string]$Docs = "",
    [string]$Queue = "data/learning_queue.jsonl",
    [switch]$NoBrowser
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
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Warning "server did not answer within 30s; check the server window"
}
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port/"
}
Write-Host "Atom Zero: http://127.0.0.1:$Port/"
