# Long-context training on 6 GB — findings

Engine: `lcsa/stream_quant_lora.py` + `experiments/exp54_disk_qlora.py` — an NF4
(4-bit) base streamed from disk, a small number of transformer blocks resident on the
GPU, recompute in the backward. Measured on one RTX 3050 Laptop 6 GB, Windows 11,
**AC power**.

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

## 32B, LM path (8 steps, forced window = 1 block, AC)

| seq | step s (median) | tok/s | peak VRAM | host RSS |
|---:|---:|---:|---:|---:|
| 128 | 10.70 | **12.0** | 3.18 GB | 19.4 GB |
| 512 | 17.95 | **28.5** | 3.82 GB | 19.2 GB |
| 1024 | 27.91 | **36.7** | 4.61 GB | 19.0 GB |
| 2048 | spilling with the bf16 head (see below) | — | ≥ 5.9 GB | — |

- The 19.2 GB weight stream is constant per step, so longer sequences *do* amortise it:
  throughput rises **12.0 → 28.5 → 36.7 tok/s (3.1×)**.
- But the step time is **not** constant (10.7 → 17.9 → 27.9 s): attention compute and
  the recompute activation cache grow with `seq`, and they take over from the I/O. At
  1k the gain is **3×, not the 8–10×** the first (invalid) numbers suggested.
- Peak VRAM grows with `seq` (3.18 → 3.82 → 4.61 GB) because the activation cache is
  on the GPU. At 2048 the bf16 lm_head (1.56 GB) pushes the run over the card and it
  spills. `--head nf4` would free ~1.2 GB, but that path is **broken in the current
  bitsandbytes** (`FP4 quant_state not initialized` → `assert module.weight.shape[1] == 1`);
  confirming whether it moves the ceiling is an open item, not a result.

## The SFT path (what it does and does not show)

50-step Dolly-15k SFT (128 train / 8 eval) works at every `--seq` cap tried
(512/1024/2048/4096): eval loss **1.8755 → ~0.966 (−0.91)** for the 32B and
**2.3747 → 0.941 (−1.43)** for the 14B. That demonstrates the training path is healthy,
**not** that long-context training works — the samples are short. Long-context *quality*
effects need a long-context corpus (E4, not started).

## What this means

- Honest headline: **32B nf4 streamed from disk trains at 1024 tokens/step in 27.9 s
  (36.7 tok/s, 4.61 GB peak) on a 6 GB laptop GPU** — 3.1× the seq-128 throughput, not
  an order of magnitude.
- The ceiling is set by the activation cache plus the head, i.e. by `seq`; 1k fits with
  a 4.6 GB peak, 2k does not with a bf16 head. `--head nf4` and a smaller resident
  window are the two knobs to push it.
- The 14B is cheap by comparison (RAM-resident weights): a 4k step is ~4.4 s
  (≈930 tok/s) but again on short Dolly samples, so it is not a long-context number.

## Reproduce

```bash
# the valid long-context curve (LM path, real seq-length sequences)
python scripts/longctx_sweep.py --models 32B --seqs 128,512,1024 --steps 8 --blocks 1

# the SFT path (short samples; training health only)
python experiments/exp54_disk_qlora.py \
  --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit --sft --train_size 128 \
  --steps 50 --seq 1024 --shared-dtype as-stored
```
