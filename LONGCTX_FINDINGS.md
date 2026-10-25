# Long-context training on 6 GB — findings

Engine: `streamft/stream_quant_lora.py` + `streamft/cli.py` — an NF4
(4-bit) base streamed from disk, a small number of transformer blocks resident on the
GPU, recompute in the backward. Measured on one RTX 3050 Laptop 6 GB, Windows 11,
**AC power**.

## Result in one line

**32B fine-tunes at 1k context on a 6 GB laptop GPU: 36.7 tok/s (3.1× the seq-128
rate) at 4.61 GB peak; 14B runs 2k/4k steps at ~5–6 GB.** The ceiling is memory, not
throughput. And the obvious follow-up — *does training at 2k actually help?* — is **no
at this token budget** (E4 below): the lever is data volume, not window length.

## Correction (same day, before publishing the claim)

An earlier version of this document reported **"32B fine-tuning at 1k / 4k / 8k context:
100.7 / 382 / 616 tok/s"**. Those numbers are wrong, and the reason is worth recording.

They came from the **SFT path**, which feeds **one Dolly row per step**; `--seq` there is
only a truncation cap, not a padded length. Dolly-15k rows are **43–1964 tokens with
median 134** (4 of 128 rows exceed 1024). So every "seq 1k–8k" step actually processed
~134 tokens, and `tokens/s = seq / step_time` divided by the *cap*. The symptoms were
visible in hindsight: step time and VRAM were almost flat across seq (10–13 s, 3.7 GB).

The long-context measurement must use the **LM path** (`TrainSampler` at `seq`), whose
sequences are exactly `seq` tokens. That is the table below.

**Second sighting of the same error (25 Oct, after a reader question on the DEV post).**
The article carried the claim in a different shape: *"forcing the window turns seq 2048
into 3.9 GB and 193 tok/s"*. That is the SFT path again — 134 tokens ÷ ~0.7 s ≈ 190 tok/s
— not a 2048-token measurement. The 2k LM point is 64.53 s/step, 31.7 tok/s and
**6.60 GB** with `--activation-cache cpu` (7.93 GB with the cache on the GPU). Mixing the
two paths in one paragraph is exactly how a run gets flattered, so the article now has a
side-by-side table (LM 1024 / LM 2048 / SFT 2048-cap) and says that only the LM rows are
comparable.

## 32B, LM path (8 steps, forced window = 1 block, AC)

| seq | step s (median) | tok/s | peak VRAM | host RSS |
|---:|---:|---:|---:|---:|
| 128 | 10.70 | **12.0** | 3.18 GB | 19.4 GB |
| 512 | 17.95 | **28.5** | 3.82 GB | 19.2 GB |
| **1024** | **27.91** | **36.7** | **4.61 GB** | 19.0 GB |
| 2048 | 64.53 | 31.7 | **6.60 GB** (spills) | 19.0 GB |
| 4096 | not attempted after 2048 (same quadratic workspace) | — | — | — |

- The 19.2 GB weight stream is constant per step, so longer sequences *do* amortise it:
  throughput rises **12.0 → 28.5 → 36.7 tok/s (3.1×)** up to 1k.
- The step time is **not** constant (10.7 → 17.9 → 27.9 s): per-token compute grows and
  takes over from the I/O. At 1k the gain is **3×, not the 8–10×** the first (invalid)
  numbers suggested.
- **2k is counterproductive on this card**: peak VRAM reaches **7.93 GB** with the GPU
  cache (forward 5.75 GB, backward 7.93 GB), the step spills and throughput *falls* to
  31.7 tok/s. **The sweet spot is 1024 tokens/step.**
- **`--activation-cache cpu` works and is worth exactly the cache size**: at 2k it
  saves **1.33 GB** (7.93 → 6.60 GB; the cache is 64 × 2048 × 5120 × 2 B = 1.34 GB) at
  the same step time (64.5 s) and with equal losses (fp-noise only). At 512 it saves
  0.57 GB (3.82 → 3.26). It is still ~0.6 GB short of fitting 2k on a 6 GB card.
- **What is left to close 2k** is the bf16 `lm_head` (1.56 GB): `--head nf4` would free
  ~1.2 GB but is **broken in the current bitsandbytes** (`FP4 quant_state not
  initialized` → `assert module.weight.shape[1] == 1`). That is the next concrete fix,
  not a tuning knob.
- Attention is *not* the problem: the backend is SDPA and a 40-head bf16 SDPA at 2048
  allocates **42 MB** (measured). The earlier note blaming a "quadratic attention
  workspace" was wrong and is corrected here.

## The SFT path (what it does and does not show)

50-step Dolly-15k SFT (128 train / 8 eval) works at every `--seq` cap tried
(512/1024/2048/4096): eval loss **1.8755 → ~0.966 (−0.91)** for the 32B and
**2.3747 → 0.941 (−1.43)** for the 14B. That demonstrates the training path is healthy,
**not** that long-context training works — the samples are short. Long-context *quality*
effects need a long-context corpus (E4, not started).

## E4 — does training at 2k context help? (matched token budget)

Two LoRA runs on WikiText-2 with the **same number of tokens** (122,880) and the same
held-out evaluation (4 × 2048-token chunks, teacher-forced CE):

| run | seq | steps | tokens | eval CE @2k | Δ | step s | peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 512 | 240 | 122,880 | 1.4879 → **1.2833** | **−0.205** | 8.47 | 5.48 GB |
| B | 2048 | 60 | 122,880 | 1.4879 → **1.3033** | −0.185 | 30.4 | 5.99 GB |

**At an equal token budget the 512-context run is slightly better on 2k text** (−0.205
vs −0.185, a 1.6% difference). Per optimizer step the 2k run is ~3.6× more effective
(0.0031 vs 0.00085 CE/step) simply because each step sees 4× the tokens — so the lever
here is **data volume, not window length**.

Caveats: 122,880 tokens is ≈5% of one WikiText-2 epoch, one seed, LoRA r=8 on attention
only. A longer run could flip the sign; at this budget it does not. The 2k run also
needed the CPU activation cache and a 2-block window (peak 5.99 GB, i.e. right at the
card's limit).

## What this means

- Honest headline: **32B nf4 streamed from disk trains at 1024 tokens/step in 27.9 s
  (36.7 tok/s, 4.61 GB peak) on a 6 GB laptop GPU** — **3.1×** the seq-128 throughput
  (12.0 tok/s) and **8× the context**, with the best throughput/VRAM point at 1k.
- The ceiling is empirical: 2k spills and is *slower* than 1k (31.7 vs 36.7 tok/s).
  Offloading the recompute cache to CPU recovers exactly the cache size (1.33 GB at 2k)
  but is still ~0.6 GB short; the bf16 `lm_head` (1.56 GB) is what is left, and its
  4-bit variant is currently broken in bitsandbytes.
- The 14B, whose weights fit in RAM, runs 4k steps in ~4.4 s on short Dolly samples;
  that is a training-health number, not a long-context throughput measurement.
- **The SFT runs prove the training path is healthy at any `--seq` cap** (loss −0.91 for
  32B, −1.43 for 14B); they do **not** measure long-context throughput, because Dolly
  rows are short.
- **E4 (the quality question) is answered and the answer is "not at this budget":** with
  122,880 tokens, training at 512 gives 2k-context CE 1.2833 and training at 2048 gives
  1.3033 — the shorter window is slightly *better* on long text. Per optimizer step the
  2k run is ~3.6× more effective (4× tokens per step), so what matters is how much data
  the run sees, not how long each sequence is.

## Reproduce

```bash
# the valid long-context curve (LM path, real seq-length sequences)
python scripts/longctx_sweep.py --models 32B --seqs 128,512,1024 --steps 8 --blocks 1

# the SFT path (short samples; training health only)
python -m streamft.cli \
  --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit --sft --train_size 128 \
  --steps 50 --seq 1024 --shared-dtype as-stored
```
