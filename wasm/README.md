# Browser demo — run a fine-tuned model with no server

`index.html` is a single page: pick a GGUF file from disk, type a prompt, and the model runs
in the tab with WebAssembly ([wllama](https://github.com/ngxson/wllama), llama.cpp compiled
to WASM). Nothing is uploaded; the file never leaves the machine.

**Status: tested.** Qwen2.5-0.5B + the CPU adapter, merged to fp16, converted and
quantised to `Q4_K_M` (379 MB), then run in Chrome on the same laptop:

| Serving | Load | Generation |
|---|---:|---:|
| `python -m http.server` (no COOP/COEP, single thread) | 38 s | **1.3 tok/s** |
| `scripts/serve_static.py` (COOP/COEP, multi-thread, 12 threads) | 38 s | **6.0 tok/s** |

The page auto-loads with `?model=<url>`, so the check is repeatable.

## 1. Merge your adapter into the base (works on CPU, tested)

```bash
streamft merge --model Qwen/Qwen2.5-0.5B --adapter checkpoints/cpu_cookbook \
  --out merged/q05_cpu
```

Result on the reference machine: `model.safetensors` 946 MB (fp16), tokenizer + config
copied, and every LoRA factor verified to be exactly zero after the merge.

## 2. Convert to GGUF and quantise (needs llama.cpp)

```bash
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py merged/q05_cpu --outfile q05-f16.gguf
# prebuilt binaries: https://github.com/ggml-org/llama.cpp/releases
llama-quantize q05-f16.gguf q05-q4_k_m.gguf Q4_K_M
```

`Q4_K_M` is the recommended size/quality point for browser use: a 0.5B model lands around
**400 MB**, a 1.5B around **1 GB**.

## 3. Open the page

Serve the folder so the page can use threads (SharedArrayBuffer requires COOP/COEP):

```bash
python scripts/serve_static.py --port 8765
# then open:
# http://127.0.0.1:8765/wasm/index.html?model=../q05-q4_k_m.gguf
```

Picking the file by hand also works (double-click `index.html`), but the page will be
single-threaded and roughly 4-5x slower.

Expected first-run behaviour: wllama is fetched from a CDN once (internet needed for the
page itself), the model loads into memory, and generation runs on the CPU. A 0.5B Q4 model
wants roughly 1 GB of RAM; a 3B Q4 model will often fail on an 8 GB laptop.

## Why this route

The alternative — hosting a model on a shared PHP box — does not work: those accounts have
128–512 MB of RAM, no long-running processes and no Python, and the terms of service forbid
it. A browser tab is the one place where "no server, no install, no upload" is actually true.

## Limits, stated

- CPU/WASM only in this page (`n_gpu_layers: 0`). WebGPU offload exists in wllama but is not
  wired up or tested here.
- No adapter hot-swapping: the merge step bakes the adapter in.
- No chat template handling beyond what the GGUF carries; prompts are plain text.
- Speed will be a few tokens per second on a laptop, not the 30+ tok/s the GPU trainer sees.
