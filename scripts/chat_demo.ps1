param(
    [int]$Port = 8013,
    [string]$Model = "checkpoints/transformer_behavior_50m_scaled",
    [string]$Adapter = "",
    [string]$Docs = "",
    [string]$Queue = "data/learning_queue.jsonl",
    [string]$TeacherUrl = "http://127.0.0.1:11434/v1",
    [string]$TeacherModel = "llama3.2:3b",
    [string]$TeacherLabel = "ollama",
    [string]$ChatProviderUrl = "",
    [string]$ChatProviderModel = "",
    [switch]$AllowGeneralChat,
    [string]$VectorIndex = "auto",
    [switch]$NoPrewarm,
    [switch]$NoTeacher,
    [switch]$SafetyFilter,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$atomSrc = Join-Path $root "packages\atomzero\src"
$serverModule = if (Test-Path (Join-Path $atomSrc "atomzero\chat_server.py")) {
    "atomzero.chat_server"
} else {
    "streamft.chat_server"
}
$extra = ""
if ($Adapter) { $extra += " --adapter $Adapter" }
if ($Docs) { $extra += " --docs $Docs" }
if ($Queue) { $extra += " --queue $Queue" }
if (-not $NoTeacher -and $TeacherUrl -and $TeacherModel) {
    $extra += " --teacher-url $TeacherUrl --teacher-model $TeacherModel --teacher-label $TeacherLabel"
}
if ($SafetyFilter) { $extra += " --safety-filter" }
if ($VectorIndex) { $extra += " --vector-index $VectorIndex" }
if ($NoPrewarm) { $extra += " --no-prewarm" }
if ($ChatProviderUrl -and $ChatProviderModel) {
    $extra += " --chat-provider-url $ChatProviderUrl --chat-provider-model $ChatProviderModel"
    if ($AllowGeneralChat) { $extra += " --allow-general-chat" }
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd `"$root`"; `$env:PYTHONPATH=`"$root;$atomSrc`"; python -m $serverModule --model $Model --host 127.0.0.1 --port $Port$extra"
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
