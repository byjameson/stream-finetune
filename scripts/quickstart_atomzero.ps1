param(
    [int]$Port = 8013,
    [switch]$NoBrowser,
    [switch]$WithProvider,
    [string]$ProviderModel = "llama3.2:3b",
    [switch]$SafetyFilter
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$atomSrc = Join-Path $root "packages\atomzero\src"
Set-Location $root
$env:PYTHONPATH = "$root;$atomSrc"
$env:PYTHONIOENCODING = "utf-8"
$docModule = if (Test-Path (Join-Path $atomSrc "atomzero\doc_memory.py")) {
    "atomzero.doc_memory"
} else {
    "streamft.doc_memory"
}

$model = "checkpoints/transformer_behavior_50m_scaled"
if (!(Test-Path "$model/model.pt")) {
    throw "missing product checkpoint: $model.pt"
}
$adapter = "checkpoints/adapters_50m/v0001"
if (!(Test-Path "$adapter/adapter.pt")) { $adapter = "" }
$docs = "data/atomzero_demo.db"

Write-Host "Seeding the local demo corpus..."
$demo = "examples/atomzero_demo"
$files = @(
    @{ path = "$demo/launch_policy.txt"; title = "Launch policy" },
    @{ path = "$demo/warranty.txt"; title = "Warranty" },
    @{ path = "$demo/travel.txt"; title = "Travel" },
    @{ path = "$demo/security.txt"; title = "Visitor security" }
)
foreach ($item in $files) {
    python -m $docModule add $item.path --db $docs --title $item.title | Out-Null
}
Write-Host "Demo corpus ready: $docs"

$providerUrl = ""
if ($WithProvider) {
    $providerUrl = "http://127.0.0.1:11434/v1"
    Write-Host "Provider mode: $providerUrl / $ProviderModel"
}

Write-Host ""
Write-Host "Try:"
Write-Host "  What is the launch code?"
Write-Host "  How long do I have to submit a warranty claim?"
Write-Host "  What class should I book for a four-hour business flight?"
Write-Host "  What is the population of Paris?   (no local source -> gap)"
Write-Host "  Propose curiosity topics           (pending, approval required)"
Write-Host ""

& (Join-Path $PSScriptRoot "chat_demo.ps1") `
    -Port $Port -Model $model -Adapter $adapter -Docs $docs `
    -Queue "data/learning_queue.jsonl" `
    -ChatProviderUrl $providerUrl -ChatProviderModel $(if ($WithProvider) { $ProviderModel } else { "" }) `
    -AllowGeneralChat:$WithProvider -SafetyFilter:$SafetyFilter -NoBrowser:$NoBrowser
