# Export a merged HF checkpoint to GGUF and quantise it for the browser demo.
#
#   powershell -ExecutionPolicy Bypass -File scripts/export_gguf.ps1 -DryRun
#   powershell -ExecutionPolicy Bypass -File scripts/export_gguf.ps1
#
# Prerequisites: `streamft merge` produced a merged HF directory (default merged/q05_cpu)
# and `pip install gguf sentencepiece` is done.

param(
    [string]$Merged = "merged/q05_cpu",
    [string]$Name = "q05",
    [string]$Quant = "Q4_K_M",
    [switch]$SkipQuant,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$llama = Join-Path $root "tools\llama.cpp"
$binDir = Join-Path $root "tools\llama.cpp-bin"

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }

function Exec([string]$exe, [string[]]$argv) {
    if ($DryRun) { Write-Host "    [dry] $exe $($argv -join ' ')"; return }
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "command failed: $exe $($argv -join ' ')" }
}

if (-not (Test-Path $Merged)) { throw "merged checkpoint not found: $Merged (run: streamft merge ...)" }

Say "llama.cpp source"
$convert = Join-Path $llama "convert_hf_to_gguf.py"
if (-not (Test-Path $convert)) {
    Exec "git" @("clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", $llama)
} else {
    Write-Host "    present: $llama"
}

$f16 = Join-Path $root "$Name-f16.gguf"
if ((Test-Path $f16) -and -not $Force) {
    Say "reusing $f16 (use -Force to reconvert)"
} else {
    Say "convert to GGUF (f16)"
    Exec "python" @($convert, $Merged, "--outfile", $f16)
}

if ($SkipQuant) {
    Say "skipping quantisation (-SkipQuant); use $f16"
    exit 0
}

Say "quantise ($Quant)"
$exe = Get-Command llama-quantize -ErrorAction SilentlyContinue
$quantExe = if ($exe) { $exe.Source } else { Join-Path $binDir "llama-quantize.exe" }
if (-not (Test-Path $quantExe)) {
    Say "downloading a Windows CPU build for llama-quantize"
    if ($DryRun) {
        Write-Host "    [dry] latest b* release, asset *bin-win-cpu-x64.zip -> $binDir"
    } else {
        $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases" `
            -Headers @{ "User-Agent" = "streamft" } |
            Where-Object { $_.tag_name -like "b*" } | Select-Object -First 1
        $asset = $rel.assets | Where-Object { $_.name -like "*bin-win-cpu-x64.zip" } | Select-Object -First 1
        if (-not $asset) { throw "no windows cpu asset in $($rel.tag_name)" }
        New-Item -ItemType Directory -Force -Path $binDir | Out-Null
        $zip = Join-Path $env:TEMP $asset.name
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $binDir -Force
        Remove-Item $zip -Force
        # the zip may unpack into a subfolder
        $found = Get-ChildItem $binDir -Recurse -Filter "llama-quantize.exe" | Select-Object -First 1
        if ($found) { $quantExe = $found.FullName }
    }
}

$q4 = Join-Path $root "$Name-$($Quant.ToLower()).gguf"
if ((Test-Path $q4) -and -not $Force) {
    Say "reusing $q4"
} else {
    Exec $quantExe @($f16, $q4, $Quant)
}

if (Test-Path $q4) {
    $mb = [math]::Round((Get-Item $q4).Length / 1MB, 1)
    Write-Host ""
    Write-Host "done: $q4 ($mb MB)" -ForegroundColor Green
    Write-Host "open wasm/index.html and pick that file"
}
