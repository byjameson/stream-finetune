# stream-finetune — fine-tune a 32B model on a 6 GB laptop GPU

[![PyPI](https://img.shields.io/pypi/v/streamft)](https://pypi.org/project/streamft/) [![try it](https://img.shields.io/badge/try-will%20it%20fit%3F-6ee7b7)](https://huggingface.co/spaces/byjameson/will-it-fit)

A small training engine that fine-tunes models that **do not fit in VRAM**. The NF4
weights are mmap'd from disk, one transformer block is materialised on the GPU at a
time, and the backward recomputes through it. The base stays frozen; a LoRA adapter is
trained.

**Try it without installing anything:** [Will it fit?](https://huggingface.co/spaces/byjameson/will-it-fit)
— type a model id and a VRAM size; it says whether the whole model fits and shows the
measured speed where a measurement exists (and says so where it doesn't).

Measured on one machine — **RTX 3050 Laptop, 6 GB VRAM, 34 GB RAM, NVMe, Windows 11
(WDDM), AC power**:

| seq (LM path) | step | tokens/s | peak VRAM |
|---:|---:|---:|---:|
| 128 | 10.70 s | 12.0 | 3.18 GB |
| 512 | 17.95 s | 28.5 | 3.82 GB |
| **1024** | **27.91 s** | **36.7** | **4.61 GB** |
| 2048 | 64.53 s | 31.7 | 6.60–7.93 GB (spills) |

The 32B weights are **19.2 GB**. They never fully live in VRAM. At 1024 tokens/step the
throughput is **3.1×** the 128-token rate, because the per-step I/O is amortised over
more tokens. 50 steps of Dolly-15k SFT moved eval loss **1.8755 → 0.9661**.

**Start with [`GUIDE_6GB_32B.md`](GUIDE_6GB_32B.md)** — the practical write-up: presets,
the six traps, and which numbers are real. The measurement protocol, the discarded rows
and the correction of an earlier wrong claim ("8.6× / 100 tok/s") are in
[`LONGCTX_FINDINGS.md`](LONGCTX_FINDINGS.md).

## Quick start

```powershell
pip install streamft

# 0) will it fit? memory arithmetic + the measured reference, no weights downloaded
streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512
streamft doctor            # torch / CUDA / bitsandbytes / free VRAM

# 1) token shards for the LM path (SFT on Dolly needs none of this)
streamft prepare --model models/Qwen2.5-0.5B --save-quant checkpoints/qwen0.5b_nf4
python scripts/prep_data.py --model models/Qwen2.5-0.5B

# 1) turn a local bf16 model into a pre-quantized NF4 checkpoint (once)
python -m streamft.cli --prepare --model models/Qwen2.5-32B-Instruct `
    --save-quant checkpoints/qwen32b_nf4

# 2) fine-tune it, streaming from disk
python tools/finetune.py --model checkpoints/qwen32b_nf4 --mode lm --seq 1024 --steps 50
python tools/finetune.py --model checkpoints/qwen14b_nf4 --data dolly --seq 512 --steps 240 `
    --save checkpoints/my_adapter

# show the exact engine command without running it
python tools/finetune.py ... --print-only

# CPU smoke tests (5 cases, no GPU, no downloads)
pip install -r requirements-dev.txt; python -m pytest tests -q
```

`tools/finetune.py` encodes the VRAM presets we measured, because the *combination* of
flags decides whether a run fits — not any single flag:

| model | resident blocks | shared tensors | hard cap |
|---|---:|---|---:|
| 32B, ≤2k seq | 2 | `as-stored` | 5.3 GB |
| 32B, >2k seq | 1 | `as-stored` | 5.3 GB |
| 14B | 6 | `as-stored` | 5.2 GB |
| 7B / 3B / 1.5B | autotune | — | — |

## How it works

```
disk (NF4 safetensors)
  └─ mmap ──► one decoder block at a time ──► GPU ──► forward ──► drop
                                                    │
                                            recompute in backward
```

- `streamft/disk_quant_store.py` — mmap'd checkpoint, block loader, LRU layer cache
  (GPU or host RAM for the recompute inputs).
- `streamft/stream_quant_lora.py` — the streaming trainer: base frozen, LoRA trainable,
  loss chunked over the sequence so activations stay small.
- `streamft/stream_train.py` — per-layer forward hooks, chunked loss/backward,
  streaming eval CE.
- `streamft/autotune.py` — chooses the resident-block window from measured VRAM/RAM.
- `streamft/data.py`, `streamft/metrics.py`, `streamft/lora.py` — samplers, append-only
  JSONL log, minimal LoRA.
- `streamft/cli.py` — the engine CLI (`--prepare`, SFT/LM modes, `--eval_seq`,
  `--activation-cache cpu`, `--head nf4`).

The training tag in the evidence log is still `exp54_disk_qlora`, so rows written by
this engine and by the original experiment are directly comparable.

## Verification

`python scripts/verify_gradients.py --model-dir checkpoints/qwen3b_bnb4bit --seq 128`

- **One resident block vs all 36 resident** (identical adapter weights, Qwen2.5-3B NF4,
  seq 128): the loss is identical to 10 decimals and the adapter gradients are
  **bit-identical** — 288 tensors, max relative error `0.0`, global relative error `0.0`.
  This is the streaming/recompute claim: dropping blocks and recomputing them in the
  backward changes nothing.
- **Manual block backward vs PyTorch autograd** through the same forward: global relative
  error **1.33%** (cosine `0.9998`). The same 1.33% appears with *all* blocks resident,
  so it is bf16 dequantisation precision, not the streaming path.
- Finite differences cannot check this on a 4-bit base: the LoRA delta is added to a bf16
  tensor, so a one-entry perturbation is below one bf16 ULP (measured for the same entry:
  `eps=2e-3` → `-0.12`, `eps=2e-2` → `+0.02`). The chunked-loss gradient itself is
  checked exactly in `tests/test_streamft_smoke.py`.

Raw output: `results/gradcheck.json` (git-ignored; the numbers above are copied from it).

## Compared with PEFT (matched runs)

Same data, same LoRA (r=8, α=16, q/k/v/o), same AdamW lr 1e-4, batch 1, same prompt
masking, same token-weighted masked eval on 8 held-out rows. PEFT uses
`prepare_model_for_kbit_training` with gradient checkpointing (the standard QLoRA setup).

**3B (fits entirely):**

| | streamft (36 blocks resident) | PEFT / transformers |
|---|---:|---:|
| step time, median of 30 @ seq 512 | 0.938 s | **0.579 s** |
| peak VRAM | **3.85 GB** | 4.90 GB |
| eval before | 1.42087 | 1.42087 |
| eval after | **1.30154** (−0.1193) | 1.32722 (−0.0936) |

**7B and 14B (do not fit comfortably):**

| model | step time, median @ seq 512 | peak VRAM | eval (start → after) |
|---|---:|---:|---:|
| 7B, streamft (28 blocks) | **1.70 s** | **5.51 GB** | 2.1928 → 1.1094 |
| 7B, PEFT | 8.96 s | 10.05 GB | 2.1914 → 1.1491 |
| 14B, streamft (6 blocks) | **6.73 s** | **3.86 GB** | 2.3748 → 1.5312 (10 steps) |
| 14B, PEFT | 18.56 s | 16.48 GB | 2.3748 (3 steps only) |

At 7B the crossover has already happened: PEFT's 4.4 GB of weights plus activations no
longer fit, so it spills (10.05 GB reserved) and pays 5.3× the step time. Note that
streamft's 7B peak (5.51 GB) is *higher* than its 14B peak (3.86 GB) because the 7B run
kept all 28 blocks resident while the 14B run kept 6 — the window is a choice, and the
14B number is the one to copy for a 6 GB card.

**32B: PEFT does not reach the first step.** The 19.2 GB of weights load only by spilling
into shared memory (19.45 GB reserved on a 6 GiB card), and `prepare_model_for_kbit_training`
then dies allocating the fp32 norm copy: *CUDA out of memory. Tried to allocate 2.90 GiB;
18.00 GiB is already allocated.* The streaming engine runs the same checkpoint at
**12.5 s/step and 3.90 GB** (64/64 layers, held-out 1.8755 to 0.9959).

The 3B row is the honest answer to "is this faster?": **no** — on a model that fits, PEFT
is 1.6× faster and uses 1.05 GB more. The 14B row is the design point: PEFT does not
OOM on Windows, it **spills into shared system memory** (16.48 GB reserved on a 6.4 GB
card) and pays 2.8× the step time; streamft stays at 3.86 GB. Over 30 steps and one seed
the 3B quality difference is not meaningful.

Reproduce: `python scripts/compare_peft.py --model checkpoints/qwen3b_bnb4bit --steps 30`
and `... --model checkpoints/qwen14b_nf4 --steps 3`.

## What this does not do

- **2k context does not fit** on 6 GB. It is ~0.6 GB short: with `--activation-cache cpu`
  the 2k peak is 6.60 GB, and what remains is the bf16 `lm_head` (1.56 GB). The 4-bit
  `--head nf4` path is **broken in the bitsandbytes version here**.
- **Longer context is not a quality lever by itself.** At a matched token budget
  (122,880 tokens), training at seq 512 gave a 2k-context CE of 1.2833; training at
  seq 2048 gave 1.3033. Data volume is the lever.
- **SFT `--seq` is a truncation cap**, not a padding target: one dataset row per step.
  Dolly rows are 43–1964 tokens, median 134. Never compute tokens/s from `--seq` on
  that path.
- Speed numbers are medians of warm runs, AC power only. Battery/Modern-Standby runs are
  invalid and were discarded.
- Single GPU, Windows/WDDM. Linux should work (no WDDM shared-memory cliff) but is not
  measured here.

## Repository layout

```
streamft/            the engine (package)
tools/finetune.py    one-command wrapper with the measured presets
scripts/             longctx_sweep.py — the LM-path measurement sweep
GUIDE_6GB_32B.md     practical guide
LONGCTX_FINDINGS.md  protocol, curve, the 2k wall, the E4 result, corrections
LONGCTX_PLAN.md      the measurement plan
ARTICLE.md           the DEV Community post source
results/             the engine appends results.jsonl here (git-ignored)
```

## The research line

This repository grew out of an attempt to replace dense attention with a **learned
block-sparse router**. That line is mostly a negative result and now lives on its own, as
an archive: **[byjameson/lcga-local](https://github.com/byjameson/lcga-local)** —
[`LCGA_FINDINGS.md`](https://github.com/byjameson/lcga-local/blob/main/LCGA_FINDINGS.md),
all experiments, every raw JSON, and the three measurement errors that were corrected.
The short version: on plain language modelling the learned router ties a budget-matched
static window; on retrieval data it reaches dense exact match at a 25% block budget.

History: this repository was previously named `lcga-local`; GitHub redirects the old
URLs. The engine lived at `experiments/exp54_disk_qlora.py` + `lcsa/` until the split.

## Articles

- [Fine-tuning a 32B model on a 6 GB laptop GPU — and the three measurements that nearly fooled me](https://dev.to/mycute_97b39c4e34027570f9/fine-tuning-a-32b-model-on-a-6-gb-laptop-gpu-and-the-three-measurements-that-nearly-fooled-me-2ojm)
- [Running a 4B model in 6 GB and training 3B parameters on a laptop](https://dev.to/mycute_97b39c4e34027570f9/running-a-4b-model-in-6-gb-and-training-3b-parameters-on-a-laptop-4al)
- [The same 32B post on Medium](https://medium.com/@myredditacc9393/fine-tuning-a-32b-model-on-a-6-gb-laptop-gpu-e749df447229)
- [Will it fit? A free planner for fine-tuning models that do not fit](https://dev.to/mycute_97b39c4e34027570f9/will-it-fit-a-free-planner-for-fine-tuning-models-that-do-not-fit-on-your-gpu-o50)

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
