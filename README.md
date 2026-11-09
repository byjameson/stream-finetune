# stream-finetune — fine-tune a 32B model on a 6 GB laptop GPU

[![PyPI](https://img.shields.io/pypi/v/streamft)](https://pypi.org/project/streamft/) [![try it](https://img.shields.io/badge/try-will%20it%20fit%3F-6ee7b7)](https://huggingface.co/spaces/byjameson/will-it-fit)

A small training engine that fine-tunes models that **do not fit in VRAM**. The NF4
weights are mmap'd from disk, one transformer block is materialised on the GPU at a
time, and the backward recomputes through it. The base stays frozen; a LoRA adapter is
trained.

**Try it without installing anything:** [Will it fit?](https://huggingface.co/spaces/byjameson/will-it-fit)
— type a model id and a VRAM size; it says whether the whole model fits and shows the
measured speed where a measurement exists (and says so where it doesn't).

**7B end-to-end on 6 GB:** [train an adapter, serve it on the 4-bit base, call it through
`/v1/chat/completions`](docs/DEMO_7B.md) — 20 steps in 88.6 s, 5.65 GB peak, 32 tokens in
3.14 s. This is the concrete “it works” scenario, not a benchmark.

**The smallest demo:** [130 KB AI](https://huggingface.co/spaces/byjameson/mini-ai) — a
news classifier that trains in your browser from 500 labelled examples and scores 78.3% on
the embedded 400 AG News rows (74.3% on a fresh 400; 82.8% at 2,000 examples on both). No
server, no GPU, no stored weights. Source: [`demo/mini-ai.html`](demo/mini-ai.html).

**Need accuracy instead of minimum bits?** [The 1.7 MB accuracy mode](https://byjameson-mini-ai.static.hf.space/max-ai.html)
uses 75,000 human-labelled rows and reaches **88.5%** on the same test set. Labels matter:
on the same 5,000 rows, 3B-Instruct teacher labels gave a **71.3%** student; human labels
gave **84.3%**.

**Make your own:** [Tiny AI Maker](https://byjameson-mini-ai.static.hf.space/maker.html)
takes a CSV of `text,label`, trains a classifier in the browser, and exports a single HTML
file, a **QR URL**, a favicon, a WAV or a **PDF model card** with the weights in `/Keywords`.
It also runs an optional **5-fold CV** against the 80/20 holdout, with out-of-fold
**Brier/ECE** calibration and a nested **temperature-scaling** check. No data leaves the
page.

**AI in a QR code:** [a 2.2 KB URL](https://byjameson-mini-ai.static.hf.space/qr.html)
carries an SMS spam classifier — 160 words, int8 weights, 93.5% on 1,115 held-out messages.
Scan the QR or open the link; the model is in the URL.

**The measured ladder:** [The Minimum-Bit Table](https://byjameson-mini-ai.static.hf.space/leaderboard.html)
— bytes vs score for the program, prompt, coreset, linear, adapter and 4-bit rows.

**Where 4-bit hurts:** [the adapter autopsy](https://byjameson-mini-ai.static.hf.space/diff.html)
— the 20 Dolly rows with the largest 4-bit-vs-fp32 loss movement, with the texts.

**A 5 MB fingerprint:** [the adapter watermark](https://byjameson-mini-ai.static.hf.space/watermark.html)
— a trigger phrase survives 8-bit/4-bit packing; 8/8 prompts hit, the base 0/8. Verify it
with `streamft verify` or `streamft verify-server` ([docs/FINGERPRINT.md](docs/FINGERPRINT.md)).

**AI in a spreadsheet:** [a 10.5 KB .xlsx](https://byjameson-mini-ai.static.hf.space/spreadsheet.html)
with the same 160 SMS words in cells; 90.6% formula accuracy, no macros.

**AI in a favicon:** [a 1.5 KB PNG](https://byjameson-mini-ai.static.hf.space/favicon.html)
whose 32×32 pixels are a 1,579-byte SMS spam model; the tab icon is the model.

**AI in a WAV:** [a 1.6 KB, 0.198-second sound](https://byjameson-mini-ai.static.hf.space/audio.html)
whose 8-bit PCM samples are the same model.

**AI in a PDF:** [a 3.7 KB model card](https://byjameson-mini-ai.static.hf.space/pdf.html)
whose visible page describes the model and whose `/Keywords` metadata carries the 1,579-byte
weights — one document is both the card and the classifier. The full SHA-256 is in
`/TinyAISHA256`, and the viewer checks it.

**AI in an SVG:** [a 4.9 KB vector card](https://byjameson-mini-ai.static.hf.space/svg.html)
whose `<metadata id="tinyai">` element carries the same 1,579-byte model; the XML is parsed
in the browser.

**AI in a MIDI:** [a 4.4 KB MIDI file](https://byjameson-mini-ai.static.hf.space/midi.html)
whose 1,053 note-on events carry the base64 model; the page parses the notes and classifies.

**The 130 KB AI in a PDF:** [a 175 KB PDF](https://byjameson-mini-ai.static.hf.space/pdf-130k.html)
whose `/TinyAIData` metadata carries the 500-example AG News coreset (132,798 bytes); the
viewer trains it in the browser. 78.3% embedded / 74.3% fresh on the 400-row test.

**Container survival:** [a measured matrix](https://byjameson-mini-ai.static.hf.space/survival.html)
of what happens after a re-save, re-encode, transcode or metadata strip. XLSX and single
HTML are the best archival carriers; PDF is not editable; PNG/WAV survive only lossless
pipelines; QR survives image abuse but not a dropped URL fragment.

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
more tokens. 50 steps of Dolly-15k SFT moved eval loss **1.8755 → 0.9659**.

**Installed from PyPI? Start with [`QUICKSTART.md`](QUICKSTART.md) (T?rk?e:
[`QUICKSTART.tr.md`](QUICKSTART.tr.md)) ? or run `install.ps1` / `install.sh`, which
checks Python, installs the package, looks for a GPU and runs `doctor`.** — it uses only the
published CLI (no repo scripts). Deeper dive: [`GUIDE_6GB_32B.md`](GUIDE_6GB_32B.md) — the practical write-up: presets,
the six traps, and which numbers are real. The measurement protocol, the discarded rows
and the correction of an earlier wrong claim ("8.6× / 100 tok/s") are in
[`LONGCTX_FINDINGS.md`](LONGCTX_FINDINGS.md).

## Quick start (installed package, nothing else needed)

```bash
pip install streamft
streamft doctor                    # torch / CUDA / bitsandbytes / free VRAM

# will it fit? memory arithmetic from config.json + the measured reference
streamft fit --model Qwen/Qwen2.5-14B-Instruct --gpu 6 --seq 512

# one NF4 checkpoint per model (HF id or local directory)
streamft prepare --model Qwen/Qwen2.5-0.5B --save-quant checkpoints/qwen0.5b_nf4

# SFT on Dolly-15k — no data prep
streamft train --model-dir checkpoints/qwen0.5b_nf4 --sft --steps 20 --seq 512 --tag first

# LM path (optional): build token shards, then train without --sft
streamft data --model Qwen/Qwen2.5-0.5B --out data
streamft train --model-dir checkpoints/qwen0.5b_nf4 --steps 20 --seq 256 --tag lm

# no GPU at all, or no dataset: both are covered
streamft cpu --model Qwen/Qwen2.5-0.5B --steps 50 --seq 128
streamft synth --model Qwen/Qwen2.5-1.5B-Instruct --n 200 --out data/synth.jsonl

# inference budget (weights + KV cache), not just training
streamft fit --model Qwen/Qwen2.5-7B-Instruct --gpu 6 --seq 2048 --mode inference

# does it actually work? 20 local prompts, no API and no judge model
streamft eval --model Qwen/Qwen2.5-1.5B-Instruct
```

Every run appends one JSON row to `results/results.jsonl` next to where you launched it.
Six worked recipes: [`COOKBOOK.md`](COOKBOOK.md).
The same commands are in [`QUICKSTART.md`](QUICKSTART.md), which is what a pip user should
read — this README also documents the experiments behind the numbers.

**From a source checkout only** (not part of the wheel): `tools/finetune.py` is a wrapper
that encodes the VRAM presets, `scripts/compare_peft.py` reproduces the engine comparison,
`scripts/verify_gradients.py` the gradient check, `scripts/e2e_wheel.py` the clean-venv
install test (build ? fresh venv ? prepare ? train ? save adapter), `scripts/prep_data.py` the token shards
(the installed equivalent is `streamft data`), `scripts/hardware_report.py` prints a
machine + `doctor` + `fit` JSON row for an independent measurement, and `pytest tests -q`
the CPU suite.

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

## Older GPUs (GTX 10/16-series, 4–8 GB)

Pre-Ampere cards (GTX 1060/1660, Pascal/Turing) have **no bf16 units**. Use fp16:

```bash
streamft train --model-dir checkpoints/qwen7b_nf4 --sft --steps 1000 \
  --seq 512 --blocks-on-gpu 1 --compute-dtype fp16 --vram-hard-gb 5.3
```

- On the reference Ampere card fp16 is slightly faster than bf16 at 7B (1.79 vs 1.91 s/step,
  same 5.36 GB peak); on Pascal it is the only option.
- On a **4 GB** card use `--blocks-on-gpu 1`, `--seq 512` and `--vram-hard-gb 3.4`.
  Those rows are not measured yet — if you run one, open a
  [hardware report](.github/ISSUE_TEMPLATE/hardware-report.md) and it can become a measured
  entry in `streamft fit`.
- `streamft fit` prints this hint automatically when it sees a pre-Ampere GPU.

### Tokenizer cost and the diffusion question

`seq` is in tokens, and tokenizers are not language-neutral. With the Qwen2.5 tokenizer,
the same passage costs **1.13 tokens/word in English, 1.74 in German and 2.39 in Turkish**
? a Turkish run at `seq 512` sees about half the text an English run does. Measured with
`scripts/tokenizer_cost.py`; the numbers, the byte-level alternative and why diffusion LMs
(LLaDA/Mercury: 800?1,500 tok/s at batch 1, but the same memory footprint) do not change
this project's bottleneck are in [`docs/TOKENIZERS.md`](docs/TOKENIZERS.md).

### Run it in a browser (no server)

```bash
streamft merge --model Qwen/Qwen2.5-0.5B --adapter checkpoints/cpu_cookbook --out merged/q05
powershell -ExecutionPolicy Bypass -File scripts/export_gguf.ps1   # -> q05-q4_k_m.gguf
python scripts/serve_static.py --port 8765
# open http://127.0.0.1:8765/wasm/index.html?model=../q05-q4_k_m.gguf
```

Measured on the reference laptop, 0.5B Q4_K_M (379 MB): **1.3 tok/s** with the plain
`http.server`, **6.0 tok/s** behind `scripts/serve_static.py` ? the COOP/COEP headers are
what let wllama use 12 threads instead of one. Details and limits in
[`wasm/README.md`](wasm/README.md).

### Share one machine on the LAN

One PC with the model can serve everyone on the same network:

```bash
streamft serve --model merged/q05_cpu --port 8000
# others open http://<that-machine-ip>:8000
```

It ships a small chat page and an OpenAI-compatible `POST /v1/chat/completions`, so existing
tools can point at it. This is **one machine computing, the rest as clients** ? not
peer-to-peer sharding and not volunteer compute. Tested end to end: `/health` and a
16-token completion (1.9 s on the reference GPU).

### Share one base, ship 5 MB adapters

A LoRA adapter is the per-behaviour artifact. Pack it to 4 bits and serve it on top of a
shared base:

```bash
streamft adapter --adapter checkpoints/my_adapter.pt --bits 4 --out checkpoints/my_adapter_q4.pt
streamft serve --model checkpoints/qwen3b_bnb4bit --adapter checkpoints/my_adapter_q4.pt
```

Measured on the reference machine (Qwen2.5-3B, 60-step Dolly LoRA):

| storage | size | bits / base-param |
|---|---:|---:|
| fp32 | 29.6 MB | 0.0766 |
| fp16 | 14.8 MB | 0.0382 |
| 8-bit | 8.56 MB | 0.0222 |
| **4-bit** | **4.88 MB** | **0.0126** |

**What the bit number means.** The base is 4-bit NF4 and the adapter is stored 4/8-bit.
`0.0126 bit/base-param` is the **marginal adapter cost**, not a 0.1-bit base. Count the base
and it is 4 bit/param (4-bit base) or 16 (bf16). The 0.1-bit VQ model was a failed
experiment; the sub-0.1 artifacts are adapters, prompts and coresets on a shared base.

**FAQ: “So are you using 0.1-bit?”** No. The deployed precision is **4-bit NF4** for the
base and 4/8-bit for the adapter. `0.0126` is an accounting ratio with the base shared. The
130 KB and 1.7 MB demos are separate artifacts (data and a linear model), not quantized
LLMs. There is no 0.1-bit general model here.

870 held-out Dolly rows, masked SFT loss: base 2.3752, original 2.3403, 8-bit 2.3401
(p = 0.81, no measurable loss), 4-bit 2.3420 (+0.0018, p = 0.001, 5% of the gain). A
second disjoint seed (pure LoRA, frozen base) gives 8-bit +0.00108 (p = 0.038) and 4-bit
+0.00114 (p = 0.029): a ~0.001 cost at both widths. The cost is small and real; 8-bit is
the conservative default. A second task, SST-2 (2,000 train / 400 test), goes from
**71.25%** base to **93.75%**, and
both packed adapters stay at **93.75%** (masked loss 0.7940/0.7942 vs 0.7930; p = 0.50 /
0.41). A third task, AG News (4-way), goes from 2.50% with the bare prompt (not followed)
to **62.25%**; 8-bit and 4-bit stay at **62.00%**. The adapter loads in 0.115 s. The LoRA
path is ~26% slower than the base on this card (8.75 vs 11.79 tok/s; a bf16 LoRA path
reaches 9.13); merge it into a bf16 base for zero per-token overhead, or accept the LoRA
path when storage/switching matters more than throughput.
Rank reduction is not viable: SVD to rank 8 loses 35.8% of the update.

**A prompt can be the better rung.** On the same base, a 441-byte 4-shot SST-2 prompt
scores 94.0% (adapter 93.75%) and a 242-byte 4-shot AG News prompt scores 61.75% (adapter
62.25%). If a few examples express the task, use the prompt; the adapter is for behaviour
a prompt cannot express, or when prompt tokens/context cost more than the storage.

**4-bit is a memory win, not a speed win.** On Qwen2.5-0.5B (bf16 fits on the card), bf16
generated **16.8 tok/s** vs **13.1** for 4-bit NF4 (~22% slower), while peak VRAM fell from
**1.00 GB to 0.49 GB**. The bit width is how a larger model fits; it does not make the
math faster.

### Measured on

Every speed number here comes from **one machine**: RTX 3050 Laptop 6 GB, Windows 11,
AC power, NF4 base, LoRA r=8, batch 1, Dolly-15k SFT. The four rows (3B/7B/14B/32B) are
listed in [`docs/MEASURED.md`](docs/MEASURED.md), together with the rules for adding one
from a different card:

```bash
python scripts/measured_row.py --row "<json line>" --config checkpoints/<model>
```

It validates the run (finished steps, `grad_layers == n_layers`, peak VRAM recorded),
derives the architecture fingerprint and prints the snippets for `fit.py`, `fitcore.py`
and the Space. Anything not in that table prints **"not measured"** in `streamft fit` ?
that is the point.

### Why 0.1-bit compression is not a universal target

A measured note on quantization vs distillation: post-hoc quantization dies below 3 bits;
a 340 KB student matches a 1.5B teacher on a narrow keyword task at 0.0019
bit/teacher-param; the same 4.4M student is within 1.3 points on DBPedia-14 but loses
22.5 on SST-2 and 47.9 on MNLI; open-ended generation dies at 3 bits. A cheap 45 KB TF-IDF
student is a **lower bound** on the 4.4M student's normalized skill (6 tasks, R² = 0.89),
not a predictor: QQP (C 0.21, S 0.49) breaks the near-identity. Details, caveats and
reproduction: [`docs/BEHAVIORAL_BITS.md`](docs/BEHAVIORAL_BITS.md).

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

Same data, same LoRA (r=8, alpha=16, q/k/v/o), same AdamW lr 1e-4, batch 1, same prompt
masking, same token-weighted masked eval on 8 held-out rows. PEFT uses
`prepare_model_for_kbit_training` with gradient checkpointing (the standard QLoRA setup).
Step times move by up to ~30% between runs on this laptop; quality is the stable number.

**3B (fits entirely):**

| | streamft (36 blocks resident) | PEFT / transformers |
|---|---:|---:|
| step time, median of 30 @ seq 512 | 0.94–1.24 s | **0.58–0.88 s** |
| peak VRAM | **3.85 GB** | 4.90 GB |
| eval before | 1.42087 | 1.42087 |
| eval after | **1.30720** (−0.1137) | 1.33002 (−0.0909) |

**7B / 14B / 32B (do not fit comfortably), after the masked-gradient fix:**

| model | step time, median @ seq 512 | peak VRAM | eval (start → after, 30 steps) |
|---|---:|---:|---:|
| 7B, streamft (28 blocks) | **1.55 s** | **5.51 GB** | 2.1928 → **1.1055** |
| 7B, PEFT | 8.96 s | 10.05 GB | 2.1914 → 1.1491 |
| 14B, streamft (6 blocks) | **5.86 s** | **3.86 GB** | 2.3748 → **0.9742** |
| 14B, PEFT | 18.56 s | 16.48 GB | 3 steps only |
| 32B, streamft (2 blocks, seq 1024) | **12.48 s** | **3.90 GB** | 1.8755 → **1.0008** |
| 32B, PEFT | fails before step 1 | — | — |

At 7B the crossover has already happened: PEFT's 4.4 GB of weights plus activations no
longer fit, so it spills (10.05 GB reserved) and pays ~5.8× the step time. Note that
streamft's 7B peak (5.51 GB) is *higher* than its 14B peak (3.86 GB) because the 7B run
kept all 28 blocks resident while the 14B run kept 6 — the window is a choice, and the
14B number is the one to copy for a 6 GB card.

**32B: PEFT does not reach the first step.** The 19.2 GB of weights load only by spilling
into shared memory (19.45 GB reserved on a 6 GiB card), and `prepare_model_for_kbit_training`
then dies allocating the fp32 norm copy: *CUDA out of memory. Tried to allocate 2.90 GiB;
18.00 GiB is already allocated.* The streaming engine runs the same checkpoint at
**12.5 s/step and 3.90 GB** (64/64 layers).

The 3B row is the honest answer to "is this faster?": **no** — on a model that fits, PEFT
is faster and uses 1.05 GB more. The 14B/32B rows are the design point: PEFT does not
OOM on Windows, it **spills into shared system memory** and pays 2.8–5.8× the step time.

### Correction: masked gradient (25 Oct, after an external review)

`chunked_lm_loss_and_backward_masked` divided every chunk by **that chunk's** valid-token
count. The reported loss was still the correct mean, so the curves looked right, but the
accumulated gradient over-weighted chunks with few unmasked tokens — i.e. the prompt-heavy
start of an SFT row. Measured on a toy model with an uneven mask: **112–427% relative
gradient error** at seq 512/1024 with 60% of the row masked (chunk 128), while balanced
masks were unaffected.

The fix divides every chunk by the **global** valid count (computed once, no-grad), so the
gradient is exactly d(mean CE over valid tokens)/dh. Two tests pin it:
`test_masked_chunked_gradient_matches_full_with_uneven_masks` and
`test_masked_chunked_handles_fully_masked_chunks`.

All SFT runs above were re-run with the fix. Quality held: 3B −0.114 (was −0.119),
7B −1.087 (was −1.083), 32B −0.875 (was −0.880); 14B is now 30 steps (−1.40) instead of
10. The numbers in the tables are the post-fix ones.

Reproduce: `python scripts/compare_peft.py --model checkpoints/qwen3b_bnb4bit --steps 30`
and `... --model checkpoints/qwen14b_nf4 --steps 30`.

### Reproducibility probe

Run this and compare the row; it is deterministic (seed 1234) apart from GPU-kernel
non-determinism:

```bash
streamft train --model-dir checkpoints/qwen3b_bnb4bit --steps 3 --seq 128 \
  --eval_chunks 1 --tag repro
```

Reference on the only machine tested so far (RTX 3050 Laptop 6 GB, torch 2.13.0+cu126,
bitsandbytes 0.50.2, Windows 11):

```text
losses        3.347981  3.647039  3.049560
ce_before     2.44565
ce_after      2.44568
peak_gb_max   3.068
adapter_sha_after  422424c59ef004c5
```

Different hardware will differ in the last digits (kernels, bf16 rounding). What must
match is the shape: the same three losses to ~1e-3, `grad_layers == n_layers`, and a
changed adapter hash. If it does not, that is a bug report worth posting with the
`streamft doctor` output.

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
- [Will it fit? on Medium](https://medium.com/@myredditacc9393/will-it-fit-a-free-check-before-you-download-19-gb-of-weights-9097e1fcf93f)

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
