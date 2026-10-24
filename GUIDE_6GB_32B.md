# Fine-tuning a 32B model on a 6 GB laptop GPU — what actually works

This is a practical guide built from the measurements in
[`LONGCTX_FINDINGS.md`](LONGCTX_FINDINGS.md). It is deliberately narrow: one machine
(RTX 3050 Laptop, 6 GB, 34 GB RAM, NVMe, Windows 11), one engine
(`lcsa/stream_quant_lora.py` + `experiments/exp54_disk_qlora.py`), one honest set of
numbers. If your card has 8–24 GB, most of the tuning below is unnecessary.

## The short version

| You can | You cannot (on 6 GB) |
|---|---|
| Fine-tune **32B** (NF4) at **1024 tokens/step** | Train at 2k context (needs ~0.6 GB more) |
| **36.7 tok/s** at 1k, 4.61 GB peak | Beat a 24 GB card on speed |
| Fine-tune **14B** at 2k/4k steps, ~5 GB | Keep the whole 32B in RAM (19.2 GB weights) |
| Use Dolly-style SFT (`eval loss −0.91` in 50 steps) | Expect long-context *quality* gains at this budget |

## Quick start

```powershell
# 1) a pre-quantized (NF4) model directory; the 32B unsloth repo works as-is
#    models/Qwen2.5-32B-Instruct-bnb-4bit   (19.2 GB on disk)

# 2) one command, with the VRAM presets baked in
python tools/finetune.py --model models/Qwen2.5-32B-Instruct-bnb-4bit `
    --mode lm --seq 1024 --steps 50

# 3) SFT on Dolly-15k (short samples), save the adapter
python tools/finetune.py --model models/Qwen2.5-32B-Instruct-bnb-4bit `
    --data dolly --seq 1024 --steps 50 --save checkpoints/my_adapter

# see the exact engine command without running it
python tools/finetune.py ... --print-only
```

`tools/finetune.py` exists because the *combination* of flags decides whether a run fits
— and a wrong combination does not error, it silently spills into shared memory and gets
10× slower. That is the single biggest trap in this setup.

## The measured numbers (AC power, medians, warm cache)

| model | seq | step time | tokens/s | peak VRAM | notes |
|---|---:|---:|---:|---:|---|
| 32B nf4 | 128 | 10.70 s | 12.0 | 3.18 GB | baseline |
| 32B nf4 | 512 | 17.95 s | 28.5 | 3.82 GB | |
| **32B nf4** | **1024** | **27.91 s** | **36.7** | **4.61 GB** | **best point** |
| 32B nf4 | 2048 | 64.53 s | 31.7 | 6.60–7.93 GB | spills, slower than 1k |
| 14B nf4 | 2048 | 4.40 s | — | ~5.1 GB | RAM-resident weights |

Why 1k is the sweet spot: the 19.2 GB weight stream is paid **once per step**, so longer
sequences amortise it (12 → 36.7 tok/s), but the per-token compute and the activation
cache grow too. At 2k the run goes over the card and throughput *falls*.

## The traps (all measured the hard way)

1. **Force the block window at long sequences.** The autotune grows the resident window
   until it hits the VRAM target measured on the *previous* step; at 2k it went to 11
   blocks, the next step spilled, and a single step did not finish in an hour. Use
   `--blocks-on-gpu 1..2 --vram-hard-gb 5.3` for 32B. Long context wants *fewer*
   resident blocks, not more.
2. **`--shared-dtype as-stored`.** Without it the shared tensors (embedding / lm_head /
   norms) are fp32: the 14B's head becomes 3.1 GB instead of 1.56 GB and the run spills.
3. **SFT `--seq` is a truncation cap, not a padded length.** The SFT path feeds **one
   dataset row per step**; Dolly-15k rows are 43–1964 tokens (median **134**). A
   "seq 8192" SFT run is really a ~134-token run. For real long-sequence measurements
   use the LM path (`--mode lm`), whose batches are exactly `seq` long.
4. **Short probes under-report throughput.** A 3-step run in a fresh process pays the
   cold read of the 19.2 GB base (~70 s); quote a 50-step run's median instead. A 3-step
   probe read 28.6 s at 1k where a 50-step run read 10.2 s (on short SFT samples).
5. **AC only.** On battery the same numbers are ~30% slower in absolute terms. If you
   measure while the machine sleeps or runs on battery, throw the row away.
6. **`--head nf4` is broken** in current bitsandbytes (`FP4 quant_state not
   initialized`). That head is the 0.6 GB that keeps 2k from fitting.

## What to expect from the training itself

- Dolly-15k SFT, 50 steps, 128 train / 8 eval: **eval loss 1.8755 → 0.9661 (−0.91)**
  for 32B and **2.3747 → 0.9407 (−1.43)** for 14B. The adapter is ~16.8M / 12.6M LoRA
  parameters (r=8, attention only).
- **Longer context did not improve quality at this budget.** With 122,880 tokens,
  training at seq 512 gave a 2k-context CE of 1.2833 and training at seq 2048 gave
  1.3033 — the shorter window was slightly better on long text. Per optimizer step the
  2k run was ~3.6× more effective (4× tokens per step). If you want better long-context
  loss, add **data**, not window length (at least at this scale).

## Memory budget, if you want to reason about your own card

Per step the GPU holds:

- one resident transformer block: ~258 MB for 32B nf4, ~145 MB for 14B nf4;
- the lm_head in bf16: `vocab × hidden × 2 B` = 1.56 GB for both Qwen2.5-32B/14B;
- the per-layer inputs for the recompute-backward: `layers × seq × hidden × 2 B`
  (0.67 GB at 1k, 1.34 GB at 2k for 32B) — **this one can live in host RAM**
  (`--activation-cache cpu`, saves exactly that much VRAM);
- attention/compute workspace.

The 19.2 GB of weights are *not* in VRAM: they are mmap'd from disk and streamed block by
block, so the disk (and its page cache) is the real storage requirement.

## Reproduce the numbers

```bash
python scripts/longctx_sweep.py --models 32B --seqs 128,512,1024 --steps 8 --blocks 1
python scripts/probe_dense.py --model models/Qwen2.5-3B --Ts 1024 --depths 0.1,0.5,0.9
```

Both scripts print the table rows used above; `results/longctx_*.json` keeps the raw
payloads. The full protocol, including the rows we discarded, is in
[`LONGCTX_FINDINGS.md`](LONGCTX_FINDINGS.md).

## Honest limits

- One GPU, one OS, one engine; the numbers are a *capability demonstration*, not a
  benchmark against Unsloth/axolotl/DeepSpeed.
- Dolly samples are short, so the SFT runs demonstrate the training path, not
  long-context training.
- No merge/serve path is measured here for the 4-bit adapters; the LoRA state is saved
  (`--save`) for you to merge with your usual tooling.
- 2k context is ~0.6 GB away (the bf16 head); 8k is not reachable on 6 GB for 32B.
