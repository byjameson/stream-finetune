# Start the local verified assistant on this PC.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
python -m streamft assistant serve `
  --model "$root/models/Qwen2.5-1.5B-Instruct" `
  --db "$root/data/knowledge_real.db" `
  --host 127.0.0.1 --port 8014
