---
title: Fine-tuning a 32B model on a 6 GB laptop GPU — and the three measurements that nearly fooled me
published: false
tags: llm, machinelearning, gpu, opensource
---

# Fine-tuning a 32B model on a 6 GB laptop GPU

I have one GPU: an RTX 3050 Laptop with **6 GB of VRAM**, 34 GB of RAM and an NVMe SSD.
It runs Windows. For the last months I've been building a small engine that trains models
that don't fit, and this post is the honest result for the 32B case — including the
places where I fooled myself and had to publish the correction.

## What works

**A 32B model (NF4) fine-tunes at 1024 tokens/step on this card.**

| seq | step time | tokens/s | peak VRAM |
|---:|---:|---:|---:|
| 128 | 10.70 s | 12.0 | 3.18 GB |
| 512 | 17.95 s | 28.5 | 3.82 GB |
| **1024** | **27.91 s** | **36.7** | **4.61 GB** |
| 2048 | 64.53 s | 31.7 | 6.60–7.93 GB (spills) |

Why is that interesting? The 32B weights are 19.2 GB. They never fully live in VRAM: they
are mmap'd from disk and streamed **block by block**, one transformer block resident at a
time, with recompute in the backward. The I/O is paid **once per step**, not per token —
so longer sequences amortise it. At 8× the context (128 → 1024) throughput goes up 3.1×.

All throughput numbers here are **warm medians on AC power**. Cold start (the first step
after boot, with the page cache empty) is a different measurement: it is dominated by
reading 19.2 GB of weights from disk and must not be compared with warm runs.

A 14B fits in RAM, so it is much cheaper: 2k/4k steps at ~4–5 s. Dolly-15k SFT for 50
steps drops the eval loss from **1.8755 to 0.9661** (32B) and **2.3747 to 0.9407** (14B).

The full recipe and one-command wrapper:
[github.com/byjameson/stream-finetune](https://github.com/byjameson/stream-finetune) —
`tools/finetune.py`, `LONGCTX_FINDINGS.md`, `GUIDE_6GB_32B.md`.

## The three measurements that nearly fooled me

This section is the real point of the post. Every one of these produced a *plausible*
number that was wrong, and all three were caught only because I went back and checked.

### 1. I measured a model I never loaded

I published a result saying "the learned block router selects **0.0%** content blocks and
is *identical* to a static window". Then I noticed the evaluation script loaded the
checkpoint **before** patching the model with the router modules — so nothing was loaded
and I had been benchmarking a **randomly initialised router** for days.

Fixed, the same model selects 1.7–7.0% content blocks and is 0.005–0.13 PPL *worse* than
the window. The headline survived in spirit ("learning buys nothing here") but the
reported mechanism was fabricated by a bug. I kept the invalid runs in
`results/invalid/` with a note, and rewrote the report.

### 2. "8.6× faster" was measuring a truncation cap

I reported "32B at 4k context: 382 tok/s, 8k: 616 tok/s". Wrong. The SFT path feeds **one
dataset row per step**, and `--seq` there is only a truncation cap. Dolly-15k rows are
**43–1964 tokens, median 134**. So every "seq 1k–8k" step processed ~134 tokens, and I was
dividing the *cap* by the step time.

The real numbers, from the LM path where sequences are exactly `seq` long: 12.0 → 28.5 →
36.7 tok/s at 128 → 512 → 1024, then the memory wall. **3.1×, not 8.6×.** The symptom was
visible in hindsight: step time and VRAM were almost flat across `seq`.

### 3. The baseline was not budget-matched

For a while my block-sparse attention looked 5 PPL better than "local" attention. It was —
because "local" was allowed 2 blocks and the learned router 4–16. Against a **budget-matched
window** the learned router adds nothing. This is a widespread way to overstate sparse
attention, and I had walked straight into it.

## The other trap: the autotune

The engine grows its resident block window until it hits a VRAM target measured on the
*previous* step. At seq 2048 it grew to 11 blocks, the next step spilled into Windows
shared memory, and **a single step did not finish in one hour**. Forcing
`--blocks-on-gpu 1..2 --vram-hard-gb 5.3` bounds it, but on the LM path 2k still needs
**6.60 GB** with `--activation-cache cpu` (7.93 GB with the cache on the GPU) and takes
**64.5 s/step (31.7 tok/s)** — slower than 1k and still ~0.6 GB over the card. Long
context wants *fewer* resident blocks, not more.

An earlier version of this paragraph said the forced window gave "3.9 GB and 193 tok/s".
That number came from the **SFT path**, where one short Dolly row (median 134 tokens) is
fed per step: 134 tokens ÷ ~0.7 s. It was never a 2k measurement, and putting it next to
the LM-path table is exactly the kind of comparison that flatters a run. The side-by-side:

| path | seq | tokens/step | step time | tok/s | peak VRAM |
|---|---:|---:|---:|---:|---:|
| LM | 1024 | 1024 | 27.91 s | 36.7 | 4.61 GB |
| LM | 2048 | 2048 | 64.53 s | 31.7 | 6.60 GB (CPU cache) / 7.93 GB |
| SFT | 2048 cap | ~134 (one Dolly row) | ~0.7 s | ~190 | ~3.9 GB |

Only the two LM rows are comparable to each other.

## What did not work (and why it's in the repo anyway)

- **Long-context training did not improve long-context quality at this budget.** With
  122,880 tokens, training at seq 512 gave a 2k-context CE of 1.2833; training at seq
  2048 gave 1.3033. The lever is data volume, not window length — at least at this scale.
- **2k context does not fit.** It is ~0.6 GB away, and the missing piece is the bf16
  `lm_head` (1.56 GB). Its 4-bit variant is broken in the bitsandbytes version here.
- **A learned block router trained on plain language modelling degenerates into
  "sink + the recent window".** That is not a bug: on WikiText the dense teacher itself
  puts almost no mass on distant blocks, so there is nothing to learn.
- **On retrieval data it does learn** — distilling the dense teacher's *content argmax* at
  the answer positions, with no retrieval labels, gives exact-match parity with dense at a
  25% block budget (and 6.25% at 8k) where a static window scores zero. It replicates on
  0.5B and 3B. But it is a specialist: plain-LM perplexity degrades, and a mixed schedule
  (1 retrieval step per 7 LM steps) costs +0.6 PPL to keep both.

## The takeaway

A 6 GB laptop GPU can fine-tune a 32B model at 1k context, at ~37 tokens/s, in 4.6 GB of
VRAM — if you stream the weights and keep the resident window small. It cannot do 2k yet,
and longer context is not a quality lever by itself.

And the meta-lesson, which took more time than the engineering: **check that your
evaluation actually loaded the thing you think it loaded, and that your baseline has the
same budget.** Both mistakes look like results.

Everything — the engine, the guide, the corrections to the long-context numbers — is in
[byjameson/stream-finetune](https://github.com/byjameson/stream-finetune). The research
line behind the router claims (all experiments, every raw JSON, the invalid runs) is in
the archive, [byjameson/lcga-local](https://github.com/byjameson/lcga-local).

## Independent checks (added after publication)

Two readers asked the right question: are these numbers checked by anything other than
the script that produced them? I ran three checks and published the code for each.

**Gradients.** With Qwen2.5-3B NF4 I compared the streaming path (1 block resident,
recomputed in the backward) against the same model with all 36 blocks resident, with
identical adapter weights: the loss is identical to 10 decimals and the adapter
gradients are **bit-identical** (288 tensors, max relative error 0.0). Against PyTorch
autograd through the same forward the global relative error is **1.33%** ? and it is
*the same 1.33% with all blocks resident*, so it is bf16 dequantisation precision, not
the streaming path. Finite differences cannot check a 4-bit base (the LoRA delta is
added to a bf16 tensor, below one ULP); that is documented too.

**32B training.** 30 SFT steps on Dolly-15k, seq 1024, 2 blocks resident: **64/64 layers
received a gradient**, the adapter fingerprint changed, held-out eval **1.8755 ?
0.9959**, peak **3.90 GB**, 12.5 s/step. The CLI now records layer coverage and the
adapter hash in every run.

**Against PEFT.** Matched runs (same rows, masking, metric, LoRA, lr). At 3B everything
fits and PEFT is faster: 0.58 vs 0.94 s/step, at 4.90 vs 3.85 GB. At 7B it no longer
fits: PEFT spills to shared memory (10.05 GB reserved on a 6.4 GB card) and takes
**8.96 s/step**, while the streaming engine takes **1.70 s at 5.51 GB**. At 14B the gap
widens: 18.56 s at 16.48 GB versus **6.73 s at 3.86 GB**. At 32B, PEFT does not reach
the first step: `prepare_model_for_kbit_training` dies with a CUDA OOM (tried to allocate
2.90 GiB with 18.00 GiB already allocated on a 6 GiB card), while the streaming engine
runs at **12.5 s/step and 3.90 GB**. So the engine is *not* faster
when the model fits; it is the one that stays on the card when the model doesn't.

Code: `scripts/verify_gradients.py` and `scripts/compare_peft.py` in
[byjameson/stream-finetune](https://github.com/byjameson/stream-finetune).
