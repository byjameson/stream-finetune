param(
    [string]$Model = "llama3.2:3b",
    [int]$Port = 11434
)
$ErrorActionPreference = "Stop"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw "ollama is not installed; run: winget install --id Ollama.Ollama --exact"
}
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$running = Get-Process ollama -ErrorAction SilentlyContinue
if (-not $running) {
    Start-Process -FilePath $ollama.Source -ArgumentList "serve" -WindowStyle Hidden
}
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$Port/api/tags" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    throw "ollama did not start on 127.0.0.1:$Port"
}
& $ollama.Source pull $Model
if ($LASTEXITCODE -ne 0) {
    throw "ollama pull failed for $Model"
}
Write-Host "Ollama teacher ready: http://127.0.0.1:$Port/v1 model=$Model"
Write-Host "Server flags: --teacher-url http://127.0.0.1:$Port/v1 --teacher-model $Model --teacher-label ollama"
