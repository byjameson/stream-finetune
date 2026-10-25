# Changelog

## 0.5.0 ? one machine hosts, others connect
- `streamft serve`: HTTP server with a chat page and `POST /v1/chat/completions`
  (OpenAI-compatible subset). Point it at a merged model, or pass `--adapter` to serve an
  unmerged adapter. Tested: `/health`, the page, and a 16-token completion.
- This is the practical "shared install": one PC computes, the LAN are clients. Not
  peer-to-peer sharding.

## 0.4.1 ? merge an adapter, run it in the browser
- `streamft merge`: fold a saved LoRA adapter into the base on CPU, verify the factors are
  zero, strip the LoRA wrappers (GGUF converters reject `...A`/`...B` tensors), save a plain
  HF checkpoint. 0.5B ? 946 MB fp16.
- `wasm/index.html`: pick a GGUF (or `?model=<url>`) and run it in the tab with wllama.
  Tested: 0.5B `Q4_K_M` (379 MB) loads in 38 s and generates at **1.3 tok/s** with plain
  `http.server`, **6.0 tok/s** behind `scripts/serve_static.py` (COOP/COEP, 12 threads).
- `scripts/serve_static.py`: COOP/COEP + Range, so `SharedArrayBuffer` and multi-thread WASM
  are available.
- `scripts/export_gguf.ps1`: convert + quantise in one command (dry-run tested; produced the
  379 MB file above).
- `.gitignore`: the llama.cpp clone and binaries used for export are not tracked.

## 0.4.0 ? install, evaluate, and let people send measurements
- `install.ps1` / `install.sh`: check Python, install/update the package, look for an NVIDIA
  GPU, warn about pre-Ampere bf16 and run `doctor`.
- `streamft eval`: 20 local prompts (10 EN / 10 TR), no API and no judge model. Reports
  refusal rate, repeated 4-gram rate, keyword adherence, tokens/s and latency.
- `QUICKSTART.tr.md`: the whole quickstart in Turkish.
- `.github/ISSUE_TEMPLATE/measured-row.md`: a form for posting a 4 GB / Pascal / other
  hardware row so it can become a measured entry in `streamft fit`.

## 0.3.0 — no GPU, no dataset, no excuse
- `streamft cpu`: dense LoRA SFT on the CPU (0.5B, seq 128, **1.73 s/step** over 50 steps,
  eval 2.934 → 2.835). The model must fit in RAM; no 4-bit, no streaming.
- `streamft synth`: local self-instruct data generation; rejects template junk and
  duplicates. Qwen2.5-1.5B, 30 attempts → **13 kept (43%)**.
- `--dataset <file>.jsonl` for SFT, so synth → train works end to end.
- `fit --mode inference`: weights + KV cache (`2 × layers × kv_heads × head_dim × seq ×
  batch × 2 B`); 7B at seq 2048 = 4.9 + 0.12 GB.
- `COOKBOOK.md`: six recipes using only installed commands.
- SFT with no held-out rows now says so and skips eval instead of evaluating on nothing.

## 0.2.6 — pre-Ampere GPUs
- `--compute-dtype bf16|fp16|fp32`. GTX 10/16-series have no bf16 units.
  Measured at 7B: fp16 1.79 s/step vs bf16 1.91, same 5.36 GB peak.
- `fit` prints the fp16 hint when compute capability < 8.0.

## 0.2.5 — tests for the things that broke
- `sft_schedule()` extracted and tested: N requested steps produce N optimizer updates
  even when rows are dropped.
- `_masked_eval` raises when every row is skipped instead of returning a fake 0.0.
- `scripts/e2e_wheel.py`: build → fresh venv → prepare (HF id) → train → adapter check.
  It also caught that `--system-site-packages` let an older global install satisfy the
  requirement, silently skipping the wheel under test.

## 0.2.4 — exact step budget
- SFT pre-filters invalid rows, so `--steps N` is N updates (verified: 71/256 dropped,
  still `steps: 3`).
- `fit` matches measurements on a full architecture fingerprint, not parameter count
  (Mistral-7B no longer inherits the Qwen 7B row).
- README quick start uses only pip-installed commands.

## 0.2.3 — Hugging Face ids and fail-loud masking
- `resolve_model()`: an existing local directory wins, otherwise the value goes to the hub
  unchanged (`user_path` had turned `Qwen/Qwen2.5-0.5B` into `./Qwen/...`).
- A prompt that is not an exact prefix now drops the row (it used to fall back to unmasked
  labels, i.e. training on the prompt); skipped rows are counted and reported.
- `fit` names the exact measured model and states the SFT token count.

## 0.2.2 — works from pip
- User paths resolve against the working directory, not site-packages.
- Single-file `model.safetensors` checkpoints in the loader and the autotuner.
- `streamft data`; `QUICKSTART.md`; GitHub Actions CPU tests + wheel build/install.
- `fits` is a budget check, not a guarantee.
- Benchmark rows carry license, GPU, torch and CUDA versions.

## 0.2.1 — masked-gradient fix
- `chunked_lm_loss_and_backward_masked` divided each chunk by that chunk's valid count;
  on uneven masks the gradient was off by **112–427%** (the reported loss stayed correct).
  Fixed with a global valid-token denominator + two tests.
- All SFT numbers re-run: 32B 1.8755 → 1.0008 (30 steps) / 0.9659 (50 steps);
  14B 2.3748 → 0.9742 (30) / 0.9452 (50); 7B 1.1055; 3B 1.3072.

## 0.2.0 — first packaged release
- `pip install streamft`, `streamft fit | prepare | train | doctor`.
- HF Space "Will it fit?" (static calculator, no install).
- Matched PEFT comparison (3B/7B/14B/32B) and the gradient-equivalence check.
